"""The runner polls origin for xpu/* heads, runs one, and pushes only the results branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.xpu.compare import compare_bundles, render_diff
from tools.xpu.health import check_gpu, queue_stopped, raise_stop, stop_path
from tools.xpu.meminfo import below_ram_warn, host_ram_gib, parse_meminfo, swap_gib
from tools.xpu.paths import branch_slug, bundle_rel, sha7
from tools.xpu.pins import DLE_PREFIX, MODELS, PCI_ID, PEAK_GBPS, TORCH_SERIES_PREFIX
from tools.xpu.run_yml import known_suite, parse_run_yml, suite_timeout_min
from tools.xpu.schema_check import validate_document
from tools.xpu.suites import SuiteResult

RESULTS_REFSPEC = "refs/heads/results:refs/heads/results"
_HEX_SHA = re.compile(r"[0-9a-fA-F]{40}\Z|[0-9a-fA-F]{64}\Z")
_TOKEN = re.compile(r"\b(?:gh[pousr]_|hf_)[A-Za-z0-9_]+")
_FLAGS = {
    "-C", "-W", "-f", "-fd", "-b", "-c", "-e", "-m", "--version", "--prune", "--detach", "--orphan",
    "--empty", "--ff-only", "--force", "--no-deps", "--constraint", "--format", "--", "-r", "--short",
    "--rm", "--pull", "--name", "--network", "--device", "--user", "--workdir", "--interactive",
    "--entrypoint", "--group-add", "--mount", "--env",
    "--porcelain", "--verify", "--cached", "--check", "--stat", "--left-right", "--count",
}
_FAIL = {"fail", "failed", "failure", "error"}


@dataclass(frozen=True)
class Head:
    """A queued branch head has a full SHA and a committer timestamp."""

    branch: str
    sha: str
    committer_unix: int


def select_pending(
    heads: list[Head], bundled_shas: set[str], run_yml_text: Callable[[str], str | None]
) -> list[Head]:
    """Only valid, unbundled xpu heads with known requested suites are queued."""
    pending = []
    for head in heads:
        if not head.branch.startswith("xpu/") or not _HEX_SHA.fullmatch(head.sha):
            continue
        if head.sha in bundled_shas:
            continue
        text = run_yml_text(head.sha)
        if text is None:
            continue
        try:
            branch_slug(head.branch)
            spec = parse_run_yml(text)
            if not all(known_suite(name) for name in spec.suites):
                continue
        except (ValueError, TypeError, KeyError):
            continue
        pending.append(head)
    return sorted(pending, key=lambda head: (head.committer_unix, head.branch))


def bundled_shas_from_index(text: str) -> set[str]:
    """Valid index records contribute full SHAs; malformed lines are ignored."""
    found = set()
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("sha"), str):
            found.add(record["sha"])
    return found


def secret_key(name: str) -> bool:
    """Credential-looking environment and document keys are excluded."""
    upper = name.upper()
    if upper in {"TOOLCHAIN_HASH", "PCI", "PCI_ID"}:
        return False
    words = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE", "API_KEY")
    return any(word in upper for word in words) or upper.endswith("_KEY")


def scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Secret environment keys never enter subprocess environments or documents."""
    return {name: value for name, value in env.items() if not secret_key(name)}


def knob_env(env: dict[str, str]) -> dict[str, str]:
    """Only sorted Intel Triton and IGC knobs participate in cache fingerprints."""
    return {
        name: env[name]
        for name in sorted(env)
        if name.startswith(("TRITON_INTEL_", "IGC_")) and not secret_key(name)
    }


def _redact(value: object, *, drop_keys: bool = False) -> object:
    if isinstance(value, str):
        return _TOKEN.sub("REDACTED", value)
    if isinstance(value, dict):
        return {
            key: _redact(item, drop_keys=drop_keys)
            for key, item in value.items() if not drop_keys or not secret_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, drop_keys=drop_keys) for item in value]
    return value


def _log_text(value: str) -> str:
    """Redacted logs have no trailing whitespace that can prevent results commits."""
    return "\n".join(line.rstrip() for line in str(_redact(value)).splitlines()).rstrip() + "\n"


def _has_secret(value: object) -> bool:
    if isinstance(value, dict):
        return any(secret_key(str(key)) or _has_secret(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_secret(item) for item in value)
    return False


def append_index(path: Path, record: dict) -> None:
    """The result index appends one redacted compact JSON record."""
    if _has_secret(record):
        raise ValueError("index record contains a secret-looking key")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_redact(record), separators=(",", ":"), allow_nan=False) + "\n")


def fingerprint(toolchain_hash: str, knobs: dict) -> dict:
    """The fingerprint pins the toolchain and compiler knobs."""
    return {"toolchain_hash": toolchain_hash, "knobs": dict(sorted(knobs.items()))}


def fingerprint_changed(previous: dict | None, current: dict) -> bool:
    """The first fingerprint preserves a cold cache; later changes invalidate it."""
    return previous is not None and previous != current


def toolchain_hash(parts: dict[str, str]) -> str:
    """Canonical version strings produce a stable SHA256 toolchain identity."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_push_refspec(refspec: str) -> None:
    """Only the explicit results-to-results refspec is publishable."""
    if refspec != RESULTS_REFSPEC:
        raise ValueError("only refs/heads/results may be pushed")


def assert_inside(parent: Path, child: Path) -> None:
    """Child paths resolve strictly inside their designated parent."""
    base, target = parent.resolve(), child.resolve()
    if base == target or not target.is_relative_to(base):
        raise ValueError(f"path escapes parent: {child}")


def refuse_shell_meta(argv: list[str]) -> None:
    """Argument vectors contain no shell operators or unapproved option names."""
    if not argv:
        raise ValueError("empty command")
    for part in argv:
        if not isinstance(part, str) or "\0" in part:
            raise ValueError("invalid command argument")
        if part in {";", "|", "||", "&&", "&", ">", ">>", "<", "`"}:
            raise ValueError("shell operators are forbidden")
        if part.startswith("-") and part.split("=", 1)[0] not in _FLAGS:
            raise ValueError(f"unapproved flag: {part}")


def run_cmd(
    argv: list[str], *, cwd: Path, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Commands execute without a shell and without secret environment keys."""
    refuse_shell_meta(argv)
    if Path(argv[0]).name in {"git", "git.exe"} and "push" in argv[1:]:
        if len(argv) != 4 or argv[1] != "push":
            raise ValueError("invalid git push command")
        assert_push_refspec(argv[3])
    return subprocess.run(
        argv, cwd=cwd, timeout=timeout, text=True, capture_output=True, env=scrub_env(dict(os.environ)), check=False
    )


def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    """Git commands share the subprocess and results-only publication guard."""
    return run_cmd(["git", *args], cwd=cwd)


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    completed = run_git(args, cwd=cwd)
    if completed.returncode:
        raise RuntimeError(str(_redact(completed.stderr or completed.stdout or f"git {args[0]} failed")))
    return completed


def push_results(results_dir: Path, remote: str = "origin", refspec: str = RESULTS_REFSPEC) -> None:
    """The only published ref is the local results branch."""
    assert_push_refspec(refspec)
    _valid_remote(remote)
    _git(["push", remote, refspec], cwd=results_dir)


def merge_env(*, versions: dict, probes: dict, models: dict, fingerprint_hash: str) -> dict:
    """The environment document records nullable versions and optional hardware probes."""
    aliases = {
        "torch": "torch", "torch_version_xpu": "torch_xpu", "triton": "triton", "dle": "dle",
        "compute_runtime": "compute_runtime", "igc": "igc", "level_zero": "level_zero_loader",
        "kernel_release": "kernel", "xe": "xe", "ocloc": "ocloc", "icpx": "icpx",
        "python": "python", "image": "image",
    }
    selected = {
        key: versions.get(key, versions.get(alias)) or probes.get(key)
        for key, alias in aliases.items()
    }
    doc = {
        "kind": "env",
        **selected,
        "versions": versions,
        "pci_id": PCI_ID,
        "peak_gbps": PEAK_GBPS,
        "torch_series_prefix": TORCH_SERIES_PREFIX,
        "model_revisions": models or {},
        "toolchain_hash": fingerprint_hash,
        "dpas_flags": probes.get("dpas_flags") or {
            "has_subgroup_matrix_multiply_accumulate": None,
            "has_subgroup_2d_block_io": None,
        },
        "data_ptr": probes.get("data_ptr") or {},
        "triton_add": probes.get("triton_add") or {},
        "host_ram_shadow": probes.get("host_ram_shadow") or {},
        "container_probes": probes.get("container_probes") or {},
        "host_ram_gib": probes.get("host_ram_gib"),
        "swap_gib": probes.get("swap_gib"),
        "xpu_smi": versions.get("xpu_smi", probes.get("xpu_smi", probes.get("device"))),
        "model_targets": MODELS,
    }
    doc = _redact(doc, drop_keys=True)
    errors = validate_document(doc)
    if errors:
        raise RuntimeError(f"invalid env document: {errors}")
    return doc


def summary_doc(
    branch: str, sha: str, status: str, suites: dict[str, str], pytest_failures: list[str]
) -> dict:
    """The summary identifies the head and records suite and pytest outcomes."""
    return {
        "branch": branch, "sha": sha, "status": status, "suites": suites, "pytest_failures": pytest_failures,
    }


def render_summary_md(doc: dict, diff_text: str | None) -> str:
    """The bundle summary includes suite outcomes and optional baseline differences."""
    lines = [f"# {doc['branch']} {sha7(doc['sha'])}", "", f"Status: **{doc['status']}**", "", "## Suites"]
    lines.extend(f"- `{name}`: {status}" for name, status in doc["suites"].items())
    if doc.get("pytest_failures"):
        lines.extend(["", "## Failures", *[f"- `{name}`" for name in doc["pytest_failures"]]])
    if diff_text:
        lines.extend(["", diff_text])
    return "\n".join(lines) + "\n"


def _valid_remote(remote: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", remote):
        raise ValueError("remote must be a configured remote name")


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_redact(doc), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _ff_or_ahead(results_dir: Path, remote: str) -> None:
    """Fast-forward when behind. An ahead local results branch is left for the later push."""
    counts = _git(
        ["rev-list", "--left-right", "--count", f"HEAD...{remote}/results"], cwd=results_dir
    ).stdout.split()
    ahead, behind = int(counts[0]), int(counts[1])
    if ahead and behind:
        raise RuntimeError("results branch diverged from origin; reconcile before the next run")
    if behind and not ahead:
        _git(["merge", "--ff-only", f"{remote}/results"], cwd=results_dir)


def ensure_results_worktree(repo: Path, results_dir: Path, remote: str, has_results: bool) -> None:
    """The results branch is created or fast-forwarded only in its own worktree."""
    _valid_remote(remote)
    if results_dir.resolve() == repo.resolve() or repo.resolve().is_relative_to(results_dir.resolve()):
        raise ValueError("results worktree cannot contain the caller repository")
    if results_dir.exists():
        branch = _git(["symbolic-ref", "--short", "HEAD"], cwd=results_dir).stdout.strip()
        if branch != "results":
            raise RuntimeError("existing results worktree is not on results")
        if _git(["status", "--porcelain"], cwd=results_dir).stdout.strip():
            raise RuntimeError("results worktree has uncommitted changes")
        if has_results:
            _git(["fetch", remote, "+refs/heads/results:refs/remotes/" + remote + "/results"], cwd=results_dir)
            _ff_or_ahead(results_dir, remote)
        return
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    local = run_git(["show-ref", "--verify", "refs/heads/results"], cwd=repo)
    if local.returncode == 0:
        _git(["worktree", "add", str(results_dir), "results"], cwd=repo)
        if has_results:
            _ff_or_ahead(results_dir, remote)
    elif has_results:
        _git(["worktree", "add", "-b", "results", str(results_dir), f"{remote}/results"], cwd=repo)
    else:
        orphan = run_git(["worktree", "add", "--orphan", "-b", "results", str(results_dir)], cwd=repo)
        if orphan.returncode:
            if results_dir.exists():
                raise RuntimeError("orphan worktree creation partially failed; operator inspection required")
            head = _git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
            _git(["worktree", "add", "--detach", str(results_dir), head], cwd=repo)
            _git(["checkout", "--orphan", "results"], cwd=results_dir)
            _git(["read-tree", "--empty"], cwd=results_dir)
            _git(["clean", "-fd"], cwd=results_dir)
        (results_dir / "index.jsonl").write_text("", encoding="utf-8")


def _heads(repo: Path, remote: str) -> list[Head]:
    output = _git(
        ["for-each-ref", "--format=%(refname:short)%09%(objectname)%09%(committerdate:unix)",
         f"refs/remotes/{remote}/xpu"], cwd=repo,
    ).stdout
    heads = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].startswith(remote + "/xpu/"):
            continue
        try:
            heads.append(Head(fields[0][len(remote) + 1:], fields[1], int(fields[2])))
        except ValueError:
            continue
    return heads


def _fetch(repo: Path, remote: str) -> bool:
    xpu = f"+refs/heads/xpu/*:refs/remotes/{remote}/xpu/*"
    results = f"+refs/heads/results:refs/remotes/{remote}/results"
    completed = run_git(["fetch", "--prune", remote, xpu, results], cwd=repo)
    if completed.returncode == 0:
        return True
    detail = (completed.stderr or completed.stdout or "").lower()
    if "results" not in detail or not any(word in detail for word in ("couldn't find", "not found", "missing")):
        raise RuntimeError(str(_redact(completed.stderr or completed.stdout or "fetch failed")))
    _git(["fetch", "--prune", remote, xpu], cwd=repo)
    return False


def _index_text(repo: Path, results_dir: Path, remote: str, has_results: bool) -> str:
    index = results_dir / "index.jsonl"
    if index.is_file():
        return index.read_text(encoding="utf-8")
    if not has_results:
        return ""
    completed = run_git(["show", f"{remote}/results:index.jsonl"], cwd=repo)
    return completed.stdout if completed.returncode == 0 else ""


def _venv_python(venv: Path) -> Path:
    unix = venv / "bin" / "python"
    return unix if unix.exists() else venv / "Scripts" / "python.exe"


def _collect_versions(
    worktree: Path, python: Path, logs: list[str], *, container: dict | None = None,
) -> tuple[dict, dict]:
    versions: dict[str, str | None] = {}
    probes: dict = {}

    def capture(name: str, argv: list[str]) -> str | None:
        try:
            completed = run_cmd(argv, cwd=worktree, timeout=20)
            value = (completed.stdout or "").strip() if completed.returncode == 0 else None
            if completed.returncode:
                logs.append(f"{name}: {_redact(completed.stderr or 'unavailable')}")
            return str(_redact(value)) if value else None
        except (OSError, subprocess.TimeoutExpired) as exc:
            logs.append(f"{name}: {_redact(str(exc))}")
            return None

    versions["kernel"] = capture("kernel", ["uname", "-r"])
    packages = {
        "compute_runtime": "libze-intel-gpu1", "ocloc": "intel-ocloc", "igc": "intel-igc-core-2",
        "level_zero_loader": "libze1", "level_zero_gpu": "libze-intel-gpu1",
    }
    for name, package in ({} if container else packages).items():
        versions[name] = capture(name, ["dpkg-query", "-W", "-f=${Version}", package])
    discovery = capture("discovery", ["xpu-smi", "discovery"])
    probes["device"] = discovery
    prefix = os.environ.get("DLE_PREFIX", DLE_PREFIX)
    icpx = Path(prefix) / "compiler" / "latest" / "bin" / "icpx" if prefix else None
    versions["icpx"] = capture("icpx", [str(icpx), "--version"]) if icpx and icpx.is_file() else None
    versions["dle"] = versions["icpx"]
    if container:
        from tools.xpu.container import docker_argv

        try:
            completed = _spawn_suite(
                docker_argv(**container, mode="versions"), cwd=worktree, env=scrub_env(dict(os.environ)),
                timeout_s=60, stdin="",
            )
        except subprocess.TimeoutExpired:
            run_cmd(["docker", "kill", container["name"]], cwd=worktree, timeout=20)
            raise
        if completed.returncode:
            raise RuntimeError(completed.stderr or "container version collection failed")
        collected = json.loads(completed.stdout)
        if not isinstance(collected, dict) or collected.get("image") != container["image"]:
            raise RuntimeError("invalid container runtime metadata")
        versions.update(collected)
    elif python.is_file():
        code = (
            "import json, sys, torch, triton; "
            "print(json.dumps({'torch':str(torch.__version__),"
            "'torch_xpu':getattr(torch.version,'xpu',None),'triton':str(triton.__version__),'python':sys.version.split()[0]}))"
        )
        text = capture("python versions", [str(python), "-c", code])
        if text:
            try:
                collected = json.loads(text)
                if isinstance(collected, dict):
                    versions.update({key: collected.get(key) for key in ("torch", "torch_xpu", "triton", "python")})
            except ValueError:
                logs.append("python versions: invalid JSON")
    for key in ("torch", "torch_xpu", "triton", "xe", "python", "image"):
        versions.setdefault(key, None)
    try:
        info = parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
        probes["host_ram_gib"], probes["swap_gib"] = host_ram_gib(info), swap_gib(info)
        if below_ram_warn(info):
            logs.append("warning: host RAM below 64 GiB")
    except (OSError, ValueError):
        probes["host_ram_gib"], probes["swap_gib"] = None, None
    pci = str(PCI_ID).lower().removeprefix("0x")
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        try:
            vendor = (device / "vendor").read_text(encoding="utf-8").strip().removeprefix("0x")
            identifier = (device / "device").read_text(encoding="utf-8").strip().removeprefix("0x")
            if f"{vendor}:{identifier}" == pci:
                driver = device / "driver"
                versions["xe"] = driver.resolve().name if driver.is_symlink() else None
                break
        except OSError:
            continue
    return versions, probes


def _model_revisions(model_cache: str | None = None) -> dict:
    home = model_cache or os.environ.get("HF_HOME")
    if not home:
        return {}
    return _redact(_read_json(Path(home) / "tensorfold-xpu-revisions.json") or {}, drop_keys=True)


def _update_fingerprint(state_dir: Path, cache_dir: Path, current: dict, logs: list[str]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "toolchain.json"
    if fingerprint_changed(_read_json(path), current):
        resolved = cache_dir.resolve()
        home = Path.home().resolve()
        if resolved in {Path(resolved.anchor), home} or len(resolved.parts) < 3:
            raise ValueError("unsafe Triton cache path")
        if cache_dir.is_symlink():
            raise ValueError("Triton cache must not be a symlink")
        if cache_dir.is_dir():
            for item in cache_dir.iterdir():
                if item.is_symlink():
                    item.unlink()
                    continue
                assert_inside(cache_dir, item)
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logs.append("Triton cache contents cleared after fingerprint change")
    _write_json(path, current)


def _spawn_suite(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float, stdin: str
) -> subprocess.CompletedProcess:
    """Posix suite processes die with their group so a hung kernel cannot outlive the deadline."""
    if os.name != "posix":
        return subprocess.run(
            argv, input=stdin, text=True, capture_output=True, timeout=timeout_s, cwd=cwd, env=env, check=False
        )
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=cwd, env=env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(stdin, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        raise
    return subprocess.CompletedProcess(argv, proc.returncode if proc.returncode is not None else 1, stdout, stderr)


def invoke_suite(
    name: str, *, repo: Path, worktree: Path, out_dir: Path, timeout_s: float, model_cache: str | None,
    container: dict | None = None,
) -> SuiteResult:
    """Suites run from the harness checkout, never from the branch tree, and die at timeout_s."""
    code = (
        "import json,sys\n"
        "from tools.xpu.suites import run_suite\n"
        "req=json.loads(sys.stdin.read())\n"
        "result=run_suite(req['name'],worktree=req['worktree'],out_dir=req['out_dir'],"
        "timeout_s=max(1,int(float(req['timeout_s']))),model_cache=req['model_cache'])\n"
        "json.dump({'status':result.status,'detail':result.detail,'payload':result.payload,"
        "'log':result.log},sys.stdout)\n"
    )
    env = scrub_env(dict(os.environ))
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo) if not previous else str(repo) + os.pathsep + previous
    request = json.dumps(
        {"name": name, "worktree": str(worktree), "out_dir": str(out_dir),
         "timeout_s": timeout_s, "model_cache": model_cache}
    )
    argv = [sys.executable, "-c", code]
    if container:
        from tools.xpu.container import docker_argv

        argv = docker_argv(**container)
    try:
        completed = _spawn_suite(argv, cwd=repo, env=env, timeout_s=timeout_s, stdin=request)
    except subprocess.TimeoutExpired:
        if container:
            run_cmd(["docker", "kill", container["name"]], cwd=repo, timeout=20)
        return SuiteResult(name, "fail", f"timed out after {timeout_s:g}s", {}, "timeout\n")
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "suite process failed").strip()
        return SuiteResult(name, "error", detail, {}, detail + "\n")
    try:
        body = json.loads(completed.stdout)
    except ValueError:
        return SuiteResult(name, "error", "suite process returned invalid JSON", {}, completed.stdout or "")
    status = str(body.get("status", "error")).lower()
    if status not in {"pass", "todo", "fail", "error"}:
        status = "error"
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
    return SuiteResult(name, status, str(body.get("detail") or ""), payload, str(body.get("log") or ""))


def _result_field(result: object, key: str, default: object = None) -> object:
    return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)


def _overall(suites: dict[str, str], failures: list[str], gpu_stopped: bool) -> str:
    if gpu_stopped or failures or any(status in _FAIL for status in suites.values()):
        return "fail"
    return "todo" if any(status != "pass" for status in suites.values()) else "pass"


def _junit(bundle: Path, suites: dict[str, str], failures: list[str]) -> None:
    cases = list(suites.items()) + [(name, "fail") for name in failures if name not in suites]
    xml = ET.Element(
        "testsuite", name="b70", tests=str(len(cases)),
        failures=str(sum(status in _FAIL for _, status in cases)),
        skipped=str(sum(status not in _FAIL and status != "pass" for _, status in cases)),
    )
    lines = []
    for name, status in cases:
        case = ET.SubElement(xml, "testcase", name=name, classname="tensorfold.xpu")
        if status in _FAIL:
            ET.SubElement(case, "failure", message=f"{name}: {status}")
        elif status != "pass":
            ET.SubElement(case, "skipped", message=f"{name}: {status}")
        lines.append(f"{name}: {status}")
    ET.ElementTree(xml).write(bundle / "pytest.xml", encoding="utf-8", xml_declaration=True)
    (bundle / "pytest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_bundle(results_dir: Path, branch: str | None) -> Path | None:
    if not branch:
        return None
    try:
        lines = (results_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("branch") != branch:
            continue
        relative = record.get("bundle")
        if isinstance(relative, str):
            path = results_dir / relative
            assert_inside(results_dir, path)
            if path.is_dir():
                return path
        return None
    return None


def _copy_artifacts(out_dir: Path, bundle: Path) -> None:
    for category in ("kernels", "e2e"):
        for source in sorted((out_dir / category).glob("*.json")):
            assert_inside(out_dir, source)
            doc = _read_json(source)
            if doc is None:
                raise RuntimeError(f"invalid artifact: {source.name}")
            errors = validate_document(_redact(doc))
            if errors:
                raise RuntimeError(f"invalid artifact {source.name}: {errors}")
            _write_json(bundle / category / source.name, doc)


def _remove_code_worktree(repo: Path, work_dir: Path, worktree: Path, results_dir: Path) -> None:
    assert_inside(work_dir, worktree)
    if worktree.resolve() == results_dir.resolve() or results_dir.resolve().is_relative_to(worktree.resolve()):
        raise ValueError("code worktree cleanup overlaps results")
    _git(["worktree", "remove", "--force", str(worktree)], cwd=repo)


def _container_mode(image: str | None, mode: str | None, state_dir: Path, sha: str) -> bool:
    """A persistent venv override is consumed by one queued head, then container mode resumes."""
    path = state_dir / "venv-fallback.json"
    if not image:
        return False
    if mode != "venv":
        if path.exists():
            _write_json(path, {})
        return True
    previous = _read_json(path)
    if previous and previous.get("image") == image:
        print("one-cycle venv fallback consumed; using container")
        return True
    _write_json(path, {"image": image, "sha": sha})
    print("using host venv for one cycle; subsequent heads use container")
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path(os.environ.get("TF_XPU_WORK", "~/.local/share/tensorfold-xpu/work")).expanduser(),
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--state-dir", type=Path,
        default=Path(os.environ.get("TF_XPU_STATE_DIR", "~/.local/state/tensorfold-xpu")).expanduser(),
    )
    parser.add_argument(
        "--venv", type=Path,
        default=Path(os.environ.get("TF_XPU_VENV", "~/.local/share/tensorfold-xpu/venv")).expanduser(),
    )
    parser.add_argument("--triton-cache", type=Path, default=Path("~/.triton/cache").expanduser())
    parser.add_argument("--image", default=os.environ.get("TF_XPU_IMAGE"))
    parser.add_argument("--mode", choices=("container", "venv"), default=os.environ.get("TF_XPU_MODE"))
    parser.add_argument("--native-ext", type=Path, default=os.environ.get("TF_XPU_EXT_DIR"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """One invocation selects at most one head and publishes one result bundle."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.once:
        parser.print_help()
        return 2
    if not math.isfinite(args.timeout_scale) or args.timeout_scale <= 0:
        parser.error("--timeout-scale must be finite and positive")
    if args.mode not in {None, "container", "venv"}:
        parser.error("TF_XPU_MODE must be container or venv")
    if args.mode == "container" and not args.image:
        parser.error("container mode requires --image or TF_XPU_IMAGE")
    repo, work_dir, state_dir = args.repo.resolve(), args.work_dir.resolve(), args.state_dir.resolve()
    results_dir = (args.results_dir or work_dir / "results-wt").resolve()
    if queue_stopped(state_dir):
        print(stop_path(state_dir).read_text(encoding="utf-8").strip())
        return 3
    worktree = None
    gpu_stopped = False
    logs: list[str] = []
    try:
        _valid_remote(args.remote)
        has_results = _fetch(repo, args.remote)
        index_text = _index_text(repo, results_dir, args.remote, has_results)
        run_texts = {}

        def read_run(sha: str) -> str | None:
            completed = run_git(["show", f"{sha}:.b70/run.yml"], cwd=repo)
            text = completed.stdout if completed.returncode == 0 else None
            run_texts[sha] = text
            return text

        pending = select_pending(_heads(repo, args.remote), bundled_shas_from_index(index_text), read_run)
        if not pending:
            print("idle: no pending xpu heads")
            return 0
        head = pending[0]
        spec = parse_run_yml(run_texts[head.sha])
        if args.dry_run:
            print(f"{head.branch} {head.sha} suites={','.join(spec.suites)}")
            return 0
        use_container = _container_mode(args.image, args.mode, state_dir, head.sha)
        work_dir.mkdir(parents=True, exist_ok=True)
        code_path = work_dir / head.sha
        assert_inside(work_dir, code_path)
        if code_path.exists():
            raise RuntimeError("code worktree path already exists")
        if code_path == results_dir or results_dir.is_relative_to(code_path):
            raise ValueError("code worktree overlaps results")
        _git(["worktree", "add", "--detach", str(code_path), head.sha], cwd=repo)
        worktree = code_path
        out_dir = work_dir / "suite-output" / head.sha
        assert_inside(work_dir, out_dir)
        if out_dir.exists():
            raise RuntimeError("suite output path already exists")
        out_dir.mkdir(parents=True)
        python = _venv_python(args.venv.resolve())
        failures: list[str] = []
        statuses: dict[str, str] = {}
        try:
            spec_path = worktree / ".b70" / "run.yml"
            assert_inside(worktree, spec_path)
            spec = parse_run_yml(spec_path.read_text(encoding="utf-8"))
            if not all(known_suite(name) for name in spec.suites):
                raise ValueError("run.yml requests unknown suites")
        except (OSError, ValueError, TypeError) as exc:
            failures.append("run.yml")
            logs.append(f"run.yml: {_redact(str(exc))}")
        container = None
        if use_container and not failures:
            from tools.xpu.container import container_name, image_id, render_gids

            args.triton_cache.mkdir(parents=True, exist_ok=True)
            model_home = spec.model_cache or os.environ.get("HF_HOME")
            container = {
                "image": image_id(args.image, repo=repo), "name": container_name(head.sha, "versions"),
                "repo": repo, "worktree": worktree, "out_dir": out_dir, "cache": args.triton_cache,
                "model_cache": Path(model_home) if model_home else None, "gids": render_gids(),
                "knobs": knob_env(dict(os.environ)), "native_ext": args.native_ext,
            }
        if not use_container and not python.is_file():
            failures.append("venv is missing")
            logs.append("venv is missing")
        elif not use_container and not failures:
            try:
                completed = run_cmd(
                    [str(python), "-m", "pip", "install", "-e", str(worktree), "--no-deps", "--constraint",
                     str(repo / "tools" / "xpu" / "constraints.txt")],
                    cwd=worktree, timeout=float(getattr(spec, "timeout_min", 90)) * 60 * args.timeout_scale,
                )
                logs.extend(str(_redact(text)) for text in (completed.stdout, completed.stderr) if text)
                if completed.returncode:
                    failures.append("pip install")
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append("pip install")
                logs.append(str(_redact(str(exc))))
        if container:
            try:
                versions, probes = _collect_versions(worktree, python, logs, container=container)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                versions, probes = {"image": container["image"]}, {}
                failures.append("container environment")
                logs.append(str(_redact(str(exc))))
        else:
            versions, probes = _collect_versions(worktree, python, logs)
        digest = toolchain_hash(versions)
        _update_fingerprint(state_dir, args.triton_cache, fingerprint(digest, knob_env(dict(os.environ))), logs)
        for name in spec.suites:
            if failures:
                statuses[name] = "todo"
                continue
            health = check_gpu(lambda argv, timeout: run_cmd(argv, cwd=repo, timeout=timeout))
            if not health.ok:
                statuses[name] = "fail"
                logs.append(f"GPU before {name}: {health.detail}")
                if health.hung or health.wedged:
                    gpu_stopped = True
                    raise_stop(state_dir, str(_redact(f"GPU before {name}: {health.detail}")))
                break
            try:
                result = invoke_suite(
                    name, repo=repo, worktree=worktree, out_dir=out_dir,
                    timeout_s=suite_timeout_min(spec, name) * 60 * args.timeout_scale,
                    model_cache=spec.model_cache,
                    **({"container": {**container, "name": container_name(head.sha, name)}} if container else {}),
                )
                status = str(_result_field(result, "status", "error")).lower()
                statuses[name] = status if status in {"pass", "todo", "fail", "error"} else "error"
                detail = _result_field(result, "detail", "")
                if detail:
                    logs.append(f"{name}: {_redact(str(detail))}")
                suite_log = _result_field(result, "log", "")
                if suite_log:
                    logs.append(str(_redact(str(suite_log))))
                payload = _result_field(result, "payload", {})
                suite_probes = _result_field(result, "probes", None)
                if suite_probes is None and isinstance(payload, dict):
                    suite_probes = payload.get("probes", payload if name == "env" else {})
                if isinstance(suite_probes, dict):
                    probes.update(suite_probes)
                if isinstance(payload, dict) and payload.get("kind") in {"kernel", "e2e"}:
                    category = "kernels" if payload["kind"] == "kernel" else "e2e"
                    filename = name.replace(":", "--") + ".json"
                    path = out_dir / category / filename
                    assert_inside(out_dir, path)
                    _write_json(path, payload)
            except Exception as exc:  # noqa: BLE001
                statuses[name] = "error"
                logs.append(f"{name}: {_redact(str(exc))}")
            health = check_gpu(lambda argv, timeout: run_cmd(argv, cwd=repo, timeout=timeout))
            if not health.ok:
                statuses[name] = "fail"
                logs.append(f"GPU after {name}: {health.detail}")
                if health.hung or health.wedged:
                    gpu_stopped = True
                    raise_stop(state_dir, str(_redact(f"GPU after {name}: {health.detail}")))
                break
        for name in spec.suites:
            statuses.setdefault(name, "todo")
        try:
            env_doc = merge_env(
                versions=versions, probes=probes, models=_model_revisions(spec.model_cache), fingerprint_hash=digest,
            )
        except RuntimeError as exc:
            failures.append("environment document")
            logs.append(str(_redact(str(exc))))
            env_doc = merge_env(versions={}, probes={}, models={}, fingerprint_hash=digest)
        ensure_results_worktree(repo, results_dir, args.remote, has_results)
        utc = datetime.now(UTC)
        relative = Path(bundle_rel(head.branch, head.sha, utc.strftime("%Y%m%dT%H%M%SZ")))
        bundle = results_dir / relative
        assert_inside(results_dir, bundle)
        bundle.mkdir(parents=True, exist_ok=False)
        _write_json(bundle / "env.json", env_doc)
        try:
            _copy_artifacts(out_dir, bundle)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append("suite artifacts")
            logs.append(str(_redact(str(exc))))
        status = _overall(statuses, failures, gpu_stopped)
        failures.extend(name for name, result in statuses.items() if result in _FAIL and name not in failures)
        doc = summary_doc(head.branch, head.sha, status, statuses, failures)
        _write_json(bundle / "summary.json", doc)
        _junit(bundle, statuses, failures)
        baseline_name = getattr(spec, "baseline", None)
        baseline = _baseline_bundle(results_dir, baseline_name)
        diff = render_diff(compare_bundles(bundle, baseline)) if baseline else "No baseline bundle available."
        (bundle / "summary.md").write_text(str(_redact(render_summary_md(doc, diff))), encoding="utf-8")
        (bundle / "logs").mkdir()
        for filename in ("unit-host.xml", "unit-host.txt"):
            source = out_dir / filename
            if source.is_file():
                assert_inside(out_dir, source)
                (bundle / "logs" / filename).write_text(
                    _log_text(source.read_text(encoding="utf-8")), encoding="utf-8",
                )
        (bundle / "logs" / "runner.log").write_text(_log_text("\n".join(logs)), encoding="utf-8")
        append_index(
            results_dir / "index.jsonl",
            {"branch": head.branch, "sha": head.sha, "sha7": sha7(head.sha), "time": utc.isoformat(),
             "status": status, "bundle": relative.as_posix(), "suites": statuses},
        )
        _git(["add", "--", relative.as_posix(), "index.jsonl"], cwd=results_dir)
        _git(["diff", "--cached", "--check"], cwd=results_dir)
        _git(["diff", "--cached", "--stat"], cwd=results_dir)
        _git(["status", "--porcelain"], cwd=results_dir)
        message = f"results: {branch_slug(head.branch)} {sha7(head.sha)} {status}"
        committed = run_git(["commit", "-m", message], cwd=results_dir)
        if committed.returncode:
            error = str(_redact(committed.stderr or committed.stdout or "results commit failed"))
            print(error, file=sys.stderr)
            logs.append(error)
            (bundle / "logs" / "runner.log").write_text(_log_text("\n".join(logs)), encoding="utf-8")
            identity = any(
                marker in error.lower()
                for marker in ("identity unknown", "unable to auto-detect email", "user.email", "tell me who you are")
            )
            return 5 if gpu_stopped else 4 if identity else 1
        push_results(results_dir, args.remote)
        print(f"{head.branch} {head.sha}: {status} ({relative.as_posix()})")
        return 5 if gpu_stopped else 1 if status == "fail" else 0
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(str(_redact(str(exc))), file=sys.stderr)
        return 5 if gpu_stopped else 1
    finally:
        if worktree is not None:
            try:
                _remove_code_worktree(repo, work_dir, worktree, results_dir)
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"code worktree cleanup: {_redact(str(exc))}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
