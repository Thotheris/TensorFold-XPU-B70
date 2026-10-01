"""Host-side harness tests: no GPU, no network, no real git remotes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.xpu import b70_runner as runner
from tools.xpu.b70_runner import Head, append_index, fingerprint, fingerprint_changed, merge_env, select_pending
from tools.xpu.compare import compare_bundles
from tools.xpu.health import check_gpu, clear_stop, parse_discovery, queue_stopped, raise_stop
from tools.xpu.kbench import ManualTimer, bench, locates_dpas, percentiles_us, rates, triton_kernel_stats
from tools.xpu.meminfo import chunk_plan, run_host_ram_shadow
from tools.xpu.paths import branch_slug, bundle_rel
from tools.xpu.run_yml import known_suite, parse_run_yml, suite_timeout_min
from tools.xpu.schema_check import validate_document
from tools.xpu.suites import run_suite

SAMPLE = """
# a branch asks the box to run
suites: [env, unit-xpu, kernels:gdn, e2e:27b-smoke]
baseline: xpu/main
timeout_min: 90
model_cache: /models
"""
SHA_OLD = "b" * 40
SHA_NEW = "a" * 40


def test_run_yml_sample_and_rejections():
    spec = parse_run_yml(SAMPLE)
    assert spec.suites == ("env", "unit-xpu", "kernels:gdn", "e2e:27b-smoke")
    assert spec.baseline == "xpu/main" and spec.timeout_min == 90 and spec.model_cache == "/models"
    assert suite_timeout_min(spec, "env") == 10
    assert suite_timeout_min(spec, "e2e:27b-bench") == 90
    assert known_suite("e2e:nemotron-bench") and not known_suite("e2e:*-bench")
    with pytest.raises(ValueError):
        parse_run_yml("suites: [env]\ncommands: rm -rf /\n")
    with pytest.raises(ValueError):
        parse_run_yml("suites: [../etc]\n")
    capped = parse_run_yml("suites: [env]\ntimeout_min: 5\nsuite_timeouts_min: {env: 30}\n")
    assert suite_timeout_min(capped, "env") == 5


def test_branch_slug_bundle_and_index(tmp_path: Path):
    assert branch_slug("xpu/k1/gdn-triton") == "xpu--k1--gdn-triton"
    assert bundle_rel("xpu/k1/gdn", SHA_OLD, "20261001T120000Z") == (
        "runs/xpu--k1--gdn/" + SHA_OLD[:7] + "-20261001T120000Z"
    )
    with pytest.raises(ValueError):
        branch_slug("main")
    with pytest.raises(ValueError):
        branch_slug("xpu/../x")
    index = tmp_path / "index.jsonl"
    append_index(index, {"branch": "xpu/k1/gdn", "sha": SHA_OLD, "status": "pass"})
    append_index(index, {"branch": "xpu/k1/gdn", "sha": SHA_NEW, "status": "fail"})
    lines = index.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["sha"] for line in lines] == [SHA_OLD, SHA_NEW]
    with pytest.raises(ValueError):
        append_index(index, {"HF_TOKEN": "secret", "sha": SHA_OLD})


def test_select_pending_is_oldest_unbundled_xpu_head():
    heads = [
        Head("xpu/b/new", SHA_NEW, 200),
        Head("feature/nope", SHA_OLD, 1),
        Head("xpu/a/old", SHA_OLD, 100),
        Head("xpu/short", "abc", 1),
        Head("xpu/bad yml", "c" * 40, 50),
    ]
    texts = {
        SHA_NEW: "suites: [env]\n",
        SHA_OLD: "suites: [unit-host, kernels:gdn]\n",
        "c" * 40: "suites: [not-a-suite]\n",
    }
    pending = select_pending(heads, {SHA_NEW}, lambda sha: texts.get(sha))
    assert [head.branch for head in pending] == ["xpu/a/old"]


def test_stop_flag_and_discovery(tmp_path: Path):
    assert not queue_stopped(tmp_path)
    raise_stop(tmp_path, "GPU after env: timeout")
    assert queue_stopped(tmp_path)
    assert "timeout" in (tmp_path / "STOP").read_text(encoding="utf-8")
    clear_stop(tmp_path)
    assert not queue_stopped(tmp_path)
    hung = parse_discovery("", 0, True)
    assert hung.hung and not hung.ok and not hung.wedged
    wedged = parse_discovery("Device Name: Arc\ngpu hang\n", 0, False)
    assert wedged.wedged and not wedged.ok
    healthy = parse_discovery("Device ID 0\nDevice Name: Intel Arc Pro B70\n", 0, False)
    assert healthy.ok
    def explode(_argv):
        raise FileNotFoundError("xpu-smi")

    missing = check_gpu(explode)
    assert not missing.ok and not missing.hung and not missing.wedged


def test_compare_failures_bits_and_three_percent(tmp_path: Path):
    base, cur = tmp_path / "base", tmp_path / "cur"
    for root, suites, gbps, median, bitwise in (
        (base, {"env": "pass", "unit-xpu": "fail"}, 100.0, 100.0, True),
        (cur, {"env": "fail", "unit-xpu": "fail"}, 96.0, 104.0, False),
    ):
        (root / "kernels").mkdir(parents=True)
        (root / "summary.json").write_text(
            json.dumps({"suites": suites, "pytest_failures": []}), encoding="utf-8"
        )
        (root / "kernels" / "gdn.json").write_text(
            json.dumps({"gbps": gbps, "median_us": median, "bitwise_ok": bitwise}), encoding="utf-8"
        )
    report = compare_bundles(cur, base)
    assert report["new_failures"] == ["env"]
    assert report["bitwise_breaks"][0]["file"] == "kernels/gdn.json"
    metrics = {item["metric"] for item in report["perf_regressions"]}
    assert metrics == {"gbps", "median_us"}
    assert not report["ok"]
    (cur / "kernels" / "gdn.json").write_text(
        json.dumps({"gbps": 97.0, "median_us": 103.0, "bitwise_ok": True}), encoding="utf-8"
    )
    (cur / "summary.json").write_text(json.dumps({"suites": {"env": "pass"}}), encoding="utf-8")
    exact = compare_bundles(cur, base)
    assert exact["perf_regressions"] == [] and exact["new_failures"] == []
    assert exact["bitwise_breaks"] == []


def test_schema_accepts_env_kernel_and_e2e_and_rejects_mixtures():
    env = merge_env(versions={}, probes={}, models={"repo": "a" * 40}, fingerprint_hash="abc")
    assert validate_document(env) == []
    kernel = {
        "kind": "kernel", "name": "gdn", "median_us": 1.0, "p10_us": 1.0, "p90_us": 2.0,
        "gbps": 1.0, "tflops": 0.0, "pct_peak_gbps": 1.0, "pct_peak_tflops": 0.0,
        "bitwise_ok": True, "n_regs": None, "n_spills": None, "threads_per_warp": 16, "dpas": False,
    }
    e2e = {"kind": "e2e", "name": "e2e:27b-smoke", "status": "todo", "tok_s": None, "ttft_s": None,
           "token_sha_match": None}
    assert validate_document(kernel) == [] and validate_document(e2e) == []
    assert validate_document({"kind": "kernel"}) != []
    assert validate_document({"kind": "nope"}) != []


def test_shadow_stops_on_low_available_and_refuses_oom():
    assert chunk_plan(16) == [2 * 1024**3] * 8
    state = {"MemAvailable": 5 * 1024 * 1024, "Committed_AS": 1000, "MemTotal": 32 * 1024 * 1024}

    def read():
        return dict(state)

    def allocate(nbytes: int):
        state["MemAvailable"] -= nbytes // 1024
        state["Committed_AS"] += nbytes // 1024
        return nbytes

    released = []

    def release(tokens):
        released.append(list(tokens))
        tokens.clear()

    result = run_host_ram_shadow(read, allocate, release, sizes=(4, 8, 16))
    assert result["error"] is None and result["stopped_early"]
    assert [step["size_gib"] for step in result["steps"]] == [4]
    assert result["steps"][0]["delta_mem_available_kib"] == -4 * 1024 * 1024
    assert released == [[2 * 1024**3, 2 * 1024**3]]

    def boom(_nbytes: int):
        raise MemoryError("no")

    state["MemAvailable"] = 8 * 1024 * 1024
    stopped = run_host_ram_shadow(read, boom, release, sizes=(4,))
    assert stopped["error"] is None and stopped["stop_reason"].startswith("allocation refused")


def test_kbench_rates_and_dpas(tmp_path: Path):
    stats = percentiles_us([5, 1, 9, 3, 7])
    assert stats["median_us"] == 5 and stats["p10_us"] == 1 and stats["p90_us"] == 9
    got = rates(median_us=1_000_000, nbytes=608_000_000_000, flops=183e12)
    assert got["gbps"] == pytest.approx(608) and got["pct_peak_gbps"] == pytest.approx(100)
    assert got["tflops"] == pytest.approx(183) and got["pct_peak_tflops"] == pytest.approx(100)
    assert locates_dpas("op #triton_intel_gpu.dpas") and not locates_dpas("fma")

    kernel = type("Kernel", (), {})()
    kernel.n_regs = 32
    kernel.n_spills = 0
    kernel.metadata = {"threads_per_warp": 16}
    kernel.asm = {"ttgir": "#triton_intel_gpu.dpas"}
    assert triton_kernel_stats(kernel)["dpas"] is True
    payload = bench(
        fn=lambda: None, nbytes=608_000_000_000, flops=183e12, name="gdn/step", out_dir=tmp_path,
        warmup=0, repeats=1, timer=ManualTimer([1000.0]), triton_kernel=kernel, bitwise_ok=True,
    )
    assert payload["median_us"] == pytest.approx(1_000_000)
    assert validate_document(payload) == []
    assert (tmp_path / "gdn_step.json").is_file()


def test_env_suite_with_hooks_and_stubs(tmp_path: Path):
    class Tensor:
        def data_ptr(self):
            return 2**63

        def __add__(self, _other):
            return self

    class XPU:
        def is_available(self):
            return True

        def get_device_properties(self, _index):
            return {"has_subgroup_matrix_multiply_accumulate": False, "has_subgroup_2d_block_io": True}

        def synchronize(self):
            return None

    class Torch:
        __version__ = "2.14.1+xpu"
        version = type("V", (), {"xpu": "2026.1"})()
        xpu = XPU()

        def empty(self, *_args, **_kwargs):
            return Tensor()

    bad = run_suite("env", worktree=tmp_path, out_dir=tmp_path, timeout_s=5, hooks={"torch": Torch(), "triton": None})
    assert bad.status == "fail" and "ocloc" in bad.detail
    stub = run_suite("e2e:27b-bench", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert stub.status == "todo" and stub.detail.startswith("TODO")
    assert "bench_concurrent.py --serial" in stub.detail
    assert run_suite("rm -rf", worktree=tmp_path, out_dir=tmp_path, timeout_s=1).status == "error"


def test_fingerprint_does_not_purge_a_cold_cache(tmp_path: Path):
    assert fingerprint_changed(None, {"toolchain_hash": "a"}) is False
    assert fingerprint_changed({"toolchain_hash": "a"}, {"toolchain_hash": "b"}) is True
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "old.o").write_text("x", encoding="utf-8")
    state = tmp_path / "state"
    runner._update_fingerprint(state, cache, fingerprint("a", {}), [])
    assert (cache / "old.o").is_file()
    runner._update_fingerprint(state, cache, fingerprint("b", {"IGC_ShaderDumpEnable": "1"}), [])
    assert not (cache / "old.o").exists()


def test_runner_stop_dry_run_and_wedge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    state = tmp_path / "state"
    raise_stop(state, "already wedged")
    stopped = runner.main(
        ["--once", "--repo", str(tmp_path), "--state-dir", str(state), "--work-dir", str(tmp_path / "w")]
    )
    assert stopped == 3
    clear_stop(state)

    calls = {"git": [], "pip": [], "smi": 0, "suites": []}

    def fake_cmd(argv, *, cwd, timeout=None):
        exe = Path(argv[0]).name
        if exe in {"git", "git.exe"}:
            return fake_git(argv[1:], cwd)
        if "pip" in argv:
            calls["pip"].append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["xpu-smi", "discovery"]:
            calls["smi"] += 1
            return subprocess.CompletedProcess(argv, 1, "", "wedged")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fake_git(args, cwd):
        calls["git"].append(args)
        if args[:3] == ["worktree", "add", "--detach"]:
            wt = Path(args[3])
            (wt / ".b70").mkdir(parents=True)
            (wt / ".b70" / "run.yml").write_text("suites: [env]\n", encoding="utf-8")
        elif args[:2] == ["worktree", "add"] and "-b" in args:
            Path(args[args.index("-b") + 2]).mkdir(parents=True, exist_ok=True)
        if args[0] == "for-each-ref":
            text = f"origin/xpu/b/new\t{SHA_NEW}\t200\norigin/xpu/a/old\t{SHA_OLD}\t100\n"
            return subprocess.CompletedProcess(args, 0, text, "")
        if args[0] == "show" and str(args[1]).endswith("index.jsonl"):
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "show":
            return subprocess.CompletedProcess(args, 0, "suites: [env]\n", "")
        if args[0] == "show-ref":
            return subprocess.CompletedProcess(args, 1, "", "")
        if args[0] == "status":
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["rev-list", "--left-right"]:
            return subprocess.CompletedProcess(args, 0, "0\t0\n", "")
        if args[0] == "symbolic-ref":
            return subprocess.CompletedProcess(args, 0, "results\n", "")
        if args[:2] == ["push", "origin"]:
            assert args[2] == "refs/heads/results:refs/heads/results"
        return subprocess.CompletedProcess(args, 0, "", "")

    def fake_suite(name, **kwargs):
        calls["suites"].append(name)
        raise AssertionError("suite must not run after a wedged GPU")

    monkeypatch.setattr(runner, "run_cmd", fake_cmd)
    monkeypatch.setattr(runner, "invoke_suite", fake_suite)
    code = runner.main([
        "--once", "--dry-run", "--repo", str(tmp_path / "repo"), "--work-dir", str(tmp_path / "work"),
        "--results-dir", str(tmp_path / "results"), "--state-dir", str(state), "--venv", str(tmp_path / "venv"),
    ])
    assert code == 0
    assert SHA_OLD in capsys.readouterr().out
    assert calls["smi"] == 0 and calls["suites"] == []
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    code = runner.main([
        "--once", "--repo", str(tmp_path / "repo"), "--work-dir", str(tmp_path / "work"),
        "--results-dir", str(tmp_path / "results"), "--state-dir", str(state), "--venv", str(tmp_path / "venv"),
    ])
    assert code == 5
    assert queue_stopped(state)
    assert calls["suites"] == []
    assert calls["pip"]
    pushed = [args for args in calls["git"] if args[:1] == ["push"]]
    assert pushed and all(args[2] == "refs/heads/results:refs/heads/results" for args in pushed)


def test_invoke_suite_uses_harness_cwd_and_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def fake_spawn(argv, *, cwd, env, timeout_s, stdin):
        seen["cwd"] = cwd
        seen["timeout"] = timeout_s
        assert str(tmp_path / "branch") not in env["PYTHONPATH"].split(os.pathsep)[0]
        body = json.dumps({"status": "todo", "detail": "TODO: x", "payload": {}, "log": ""})
        return subprocess.CompletedProcess(argv, 0, body, "")

    monkeypatch.setattr(runner, "_spawn_suite", fake_spawn)
    result = runner.invoke_suite(
        "unit-host", repo=tmp_path / "harness", worktree=tmp_path / "branch", out_dir=tmp_path / "out",
        timeout_s=12, model_cache=None,
    )
    assert result.status == "todo" and seen["cwd"] == tmp_path / "harness" and seen["timeout"] == 12


def _bash() -> str | None:
    candidates = [os.environ.get("HERMES_GIT_BASH_PATH"), r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")]
    for item in candidates:
        if not item:
            continue
        path = Path(item)
        if path.is_dir() and (path / "bash.exe").is_file():
            return str(path / "bash.exe")
        if path.is_file():
            return str(path)
    return None


def _posix(path: Path) -> str:
    text = path.resolve().as_posix()
    if len(text) > 2 and text[1] == ":":
        return "/" + text[0].lower() + text[2:]
    return text


def test_bootstrap_dry_run_does_not_call_sudo(tmp_path: Path):
    bash = _bash()
    if bash is None:
        pytest.skip("bash is required for bootstrap.sh")
    root = Path(__file__).resolve().parents[2]
    boot = root / "tools" / "xpu" / "bootstrap.sh"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sentinel = tmp_path / "called.txt"
    for name in ("sudo", "wget", "dpkg", "pip", "curl"):
        (bindir / name).write_text("#!/bin/sh\necho \"$0\" >> \"$SENTINEL\"\nexit 99\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        HOME=_posix(tmp_path / "home"), SENTINEL=_posix(sentinel), BIN=_posix(bindir), BOOT=_posix(boot),
    )
    script = "chmod +x \"$BIN\"/*; export PATH=\"$BIN:$PATH\"; bash \"$BOOT\""
    done = subprocess.run([bash, "-c", script], env=env, text=True, capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "WOULD-RUN" in done.stdout
    assert "dry-run" in done.stdout
    assert not sentinel.exists()
    help_run = subprocess.run([bash, _posix(boot), "--help"], env=env, text=True, capture_output=True, timeout=30)
    assert help_run.returncode == 0 and "--apply" in help_run.stdout
    bad = subprocess.run([bash, _posix(boot), "--nope"], env=env, text=True, capture_output=True, timeout=30)
    assert bad.returncode == 2
    self_test = subprocess.run([bash, _posix(boot), "--self-test"], env=env, text=True, capture_output=True, timeout=30)
    assert self_test.returncode == 0, self_test.stderr
