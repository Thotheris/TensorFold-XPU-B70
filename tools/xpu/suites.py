"""Environment, Triton probes, GPU tests and kernel measurements execute in isolated suites."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .meminfo import below_ram_warn, host_ram_gib, parse_meminfo, run_host_ram_shadow, swap_gib
from .run_yml import known_suite


@dataclass
class SuiteResult:
    """A suite records its status, JSON-safe payload, and textual log."""

    name: str
    status: str
    detail: str
    payload: dict[str, Any]
    log: str


def _property(obj: object, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _read_meminfo() -> dict[str, int]:
    return parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))


def _triton_vector_add(torch: Any, triton: Any) -> dict[str, Any]:
    tl = getattr(triton, "language", None)
    if tl is None:
        tl = importlib.import_module("triton.language")

    @triton.jit
    def vector_add(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offsets, offsets < N, other=0)
        y = tl.load(Y + offsets, offsets < N, other=0)
        tl.store(OUT + offsets, x + y, offsets < N)

    n = 128
    x = torch.arange(n, dtype=torch.int32, device="xpu")
    y = torch.arange(n, dtype=torch.int32, device="xpu")
    out = torch.empty_like(x)
    vector_add[(1,)](x, y, out, n, BLOCK=128)
    torch.xpu.synchronize()
    equal = bool(torch.equal(out, x + y))
    return {"ok": equal, "n": n, "dtype": "int32", "error": None if equal else "Triton vector add mismatch"}


def _env_probes(hooks: dict[str, Any] | None = None) -> tuple[str, str, dict[str, Any]]:
    """Environment failures are data rather than uncaught import or device exceptions."""
    hooks = hooks or {}
    probes = {
        "torch": None,
        "torch_version_xpu": None,
        "triton": None,
        "host_ram_gib": None,
        "swap_gib": None,
        "dpas_flags": {
            "has_subgroup_matrix_multiply_accumulate": None,
            "has_subgroup_2d_block_io": None,
        },
        "data_ptr": {"raw": None, "high_bit": None},
        "triton_add": {"ok": False, "error": "not run"},
        "host_ram_shadow": {"baseline": {}, "steps": [], "stopped_early": False, "error": "not run"},
    }
    failures = []
    torch = triton = None
    try:
        torch = hooks["torch"] if "torch" in hooks else importlib.import_module("torch")
        if torch is None:
            raise ImportError("torch hook is missing")
        probes["torch"] = getattr(torch, "__version__", None)
        probes["torch_version_xpu"] = getattr(getattr(torch, "version", None), "xpu", None)
    except Exception as exc:  # noqa: BLE001 - backend import and device failures become probe data
        failures.append(f"torch is missing or failed to import: {exc}")
    try:
        triton = hooks["triton"] if "triton" in hooks else importlib.import_module("triton")
        if triton is None:
            raise ImportError("Triton hook is missing")
        probes["triton"] = getattr(triton, "__version__", None)
    except Exception as exc:  # noqa: BLE001 - backend import and device failures become probe data
        failures.append(f"Triton is missing or failed to import: {exc}")
        probes["triton_add"]["error"] = str(exc)
    device_ok = False
    if torch is not None:
        try:
            xpu = getattr(torch, "xpu", None)
            if xpu is None or not xpu.is_available():
                failures.append("torch.xpu is missing or no XPU device is available")
            else:
                device_ok = True
                properties = xpu.get_device_properties(0)
                for flag in probes["dpas_flags"]:
                    value = _property(properties, flag)
                    probes["dpas_flags"][flag] = value if isinstance(value, bool) else None
                    if value is not True:
                        failures.append(f"{flag} is not True; False suggests missing ocloc")
        except Exception as exc:  # noqa: BLE001 - backend import and device failures become probe data
            failures.append(f"XPU device query failed: {exc}")
    if device_ok:
        try:
            pointer = int(torch.empty(1, dtype=torch.uint8, device="xpu").data_ptr())
            probes["data_ptr"] = {"raw": str(pointer), "value": pointer, "high_bit": pointer >= 2**63}
        except Exception as exc:  # noqa: BLE001 - backend import and device failures become probe data
            probes["data_ptr"]["error"] = str(exc)
            failures.append(f"data_ptr probe failed: {exc}")
        if triton is not None:
            try:
                probes["triton_add"] = _triton_vector_add(torch, triton)
                if not probes["triton_add"]["ok"]:
                    failures.append("Triton vector add mismatch")
            except Exception as exc:  # noqa: BLE001 - backend import and device failures become probe data
                probes["triton_add"] = {"ok": False, "error": str(exc)}
                failures.append(f"Triton vector add failed: {exc}")
    read_meminfo = hooks.get("read_meminfo", _read_meminfo)

    def allocate(nbytes: int) -> Any:
        if not device_ok:
            raise RuntimeError("XPU is unavailable for the host RAM shadow probe")
        # mem_get_info().free is not a budget on this B70: it reported about 112 MiB free
        # while a 2 GiB allocation still succeeded, and the GPU may already be in use.
        token = torch.empty(nbytes, dtype=torch.uint8, device="xpu")
        torch.xpu.synchronize()
        return token

    def release(tokens: list[Any]) -> None:
        tokens.clear()
        if device_ok:
            torch.xpu.synchronize()
            empty_cache = getattr(torch.xpu, "empty_cache", None)
            if empty_cache is not None:
                empty_cache()

    if device_ok or "allocate" in hooks:
        shadow = run_host_ram_shadow(read_meminfo, hooks.get("allocate", allocate), hooks.get("release", release))
        probes["host_ram_shadow"] = shadow
        if shadow["error"]:
            failures.append(f"host RAM shadow failed: {shadow['error']}")
        info = shadow["baseline"]
    else:
        try:
            info = read_meminfo()
            probes["host_ram_shadow"]["baseline"] = info
        except Exception as exc:  # noqa: BLE001 - backend import and device failures become probe data
            info = {}
            probes["host_ram_shadow"]["error"] = str(exc)
            failures.append(f"host memory query failed: {exc}")
    probes["host_ram_gib"] = host_ram_gib(info)
    probes["swap_gib"] = swap_gib(info)
    probes["host_ram_warn"] = below_ram_warn(info)
    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else "XPU flags, pointer, Triton add, and host RAM shadow probes passed"
    return status, detail, {"probes": probes}


def def_env_probes(hooks: dict[str, Any] | None = None) -> tuple[str, str, dict[str, Any]]:
    """The environment probe entry point returns a mergeable probes mapping."""
    return _env_probes(hooks)


def _todo(name: str) -> SuiteResult:
    if name == "unit-host":
        command = "python -m pytest tests -q"
    elif name == "unit-xpu":
        command = "python -m pytest tests/cuda -q once the XPU device fixture exists"
    elif name.startswith("kernels:"):
        command = f"kbench for {name.split(':', 1)[1]} once its kernel card exists"
    elif name.endswith("-bench"):
        command = "tools/bench_openai.py; tools/bench_concurrent.py --serial; tools/prefill_cold.py"
    else:
        command = "serve check of drafted token_sha vs serial"
    detail = f"TODO: {command}"
    payload = {"command": command}
    if name.startswith("e2e:"):
        payload.update(kind="e2e", name=name, status="todo", tok_s=None, ttft_s=None, token_sha_match=None)
    return SuiteResult(name, "todo", detail, payload, detail + "\n")


def _xpu_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        return bool(torch.xpu.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def _pytest_counts(path: Path) -> dict[str, int]:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    counts = {"tests": len(cases), "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for case in cases:
        key = next((key for tag, key in (("failure", "failed"), ("error", "errors"), ("skipped", "skipped"))
                    if case.find(tag) is not None), "passed")
        counts[key] += 1
    return counts


def _pytest_suite(name: str, *, worktree: Path, out_dir: Path, timeout_s: int) -> SuiteResult:
    gpu = name != "unit-host"
    if gpu and not _xpu_available():
        return SuiteResult(name, "fail", "requested XPU is unavailable", {}, "requested XPU is unavailable\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name.replace(":", "--")
    report = out_dir / f"{stem}.xml"
    report.unlink(missing_ok=True)
    argv = [sys.executable, "-m", "pytest", "tests/cuda" if gpu else "tests", "-q", f"--junitxml={report}"]
    if not gpu:
        argv.append("--host-only")
    elif name.startswith("kernels:"):
        argv.append(f"--xpu-kernel={name.split(':', 1)[1]}")
    try:
        completed = subprocess.run(argv, cwd=worktree, text=True, capture_output=True, timeout=timeout_s, check=False,
                                   env={**os.environ, "TF_TEST_DEVICE": "xpu" if gpu else "cpu"})
        log = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        log = decoded(exc.stdout) + decoded(exc.stderr) + f"\ntimeout after {timeout_s}s\n"
        (out_dir / f"{stem}.txt").write_text(log, encoding="utf-8")
        return SuiteResult(name, "error", f"pytest timed out after {timeout_s}s", {}, log)
    (out_dir / f"{stem}.txt").write_text(log, encoding="utf-8")
    try:
        counts = _pytest_counts(report)
    except (OSError, ET.ParseError) as exc:
        return SuiteResult(name, "fail", f"missing or invalid pytest report: {exc}", {}, log)
    ok = completed.returncode == 0 and not counts["failed"] and not counts["errors"]
    if gpu:
        ok = ok and counts["passed"] > 0
    detail = f"pytest exited {completed.returncode}: {counts['passed']} passed, {counts['skipped']} skipped"
    if gpu and counts["passed"] == 0:
        detail += "; no XPU tests passed"
    return SuiteResult(name, "pass" if ok else "fail", detail, {"pytest": counts}, log)


def _kernel_benchmark(name: str, *, worktree: Path, out_dir: Path, timeout_s: int) -> dict:
    from .schema_check import validate_document

    path = out_dir / "kernels" / f"{name}.json"
    path.unlink(missing_ok=True)
    completed = subprocess.run([sys.executable, "-m", "tools.xpu.kernel_benchmarks", name, str(out_dir)],
                               cwd=worktree, text=True, capture_output=True, timeout=timeout_s, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout or "kernel benchmark failed")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_document(metrics)
    if errors or metrics.get("name") != name or metrics.get("kind") != "kernel":
        raise ValueError(f"invalid kernel measurements: {errors or 'wrong kernel name or kind'}")
    if metrics.get("bitwise_ok") is not True or metrics.get("status") != "pass":
        raise ValueError("kernel measurements did not pass bitwise checks")
    return metrics


def run_suite(
    name: str,
    *,
    worktree: str | Path,
    out_dir: str | Path,
    timeout_s: int,
    model_cache: str | None = None,
    hooks: dict[str, Any] | None = None,
) -> SuiteResult:
    """Implemented suites run real checks; unavailable kernels cannot produce a green bundle."""
    if not known_suite(name):
        detail = f"unknown suite: {name}"
        return SuiteResult(name, "error", detail, {}, detail + "\n")
    if name in {"unit-host", "unit-xpu"} or name.startswith("kernels:"):
        start = time.monotonic()
        result = _pytest_suite(name, worktree=Path(worktree), out_dir=Path(out_dir), timeout_s=timeout_s)
        if name.startswith("kernels:") and result.status == "pass":
            try:
                metrics = _kernel_benchmark(name.split(":", 1)[1], worktree=Path(worktree), out_dir=Path(out_dir),
                                            timeout_s=max(1, int(timeout_s - (time.monotonic() - start))))
                result.payload = {**metrics, **result.payload}
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                result.status = "fail"
                result.detail += f"; benchmark failed: {exc}"
        return result
    if name == "triton-smoke":
        from .triton_smoke import run_probes

        probes = run_probes(Path(out_dir))
        status = "pass" if probes["ok"] else "fail"
        log = json.dumps(probes, indent=2, allow_nan=False) + "\n"
        (Path(out_dir) / "triton-smoke.txt").write_text(log, encoding="utf-8")
        return SuiteResult(name, status, f"S0 probes: {status}", {"probes": {"triton_smoke": probes}}, log)
    if name != "env":
        return _todo(name)
    try:
        status, detail, payload = _env_probes(hooks)
        probes = payload["probes"]
        if os.environ.get("TF_XPU_IMAGE_ID") and hooks is None:
            from .container_probes import container_probes

            checks = container_probes()
            probes["container_probes"] = checks
            if not checks["ok"]:
                status = "fail"
                detail += "; container runtime probes failed"
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "env-probes.json").write_text(
            json.dumps(probes, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return SuiteResult(name, status, detail, probes, detail + "\n")
    except Exception as exc:  # noqa: BLE001 - backend import and device failures become probe data
        detail = f"environment probe failed: {type(exc).__name__}: {exc}"
        return SuiteResult(name, "error", detail, {}, detail + "\n")
