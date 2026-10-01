"""The environment suite runs probes and future GPU suites remain explicit TODOs."""

from __future__ import annotations

import importlib
import json
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
    except Exception as exc:
        failures.append(f"torch is missing or failed to import: {exc}")
    try:
        triton = hooks["triton"] if "triton" in hooks else importlib.import_module("triton")
        if triton is None:
            raise ImportError("Triton hook is missing")
        probes["triton"] = getattr(triton, "__version__", None)
    except Exception as exc:
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
        except Exception as exc:
            failures.append(f"XPU device query failed: {exc}")
    if device_ok:
        try:
            pointer = int(torch.empty(1, dtype=torch.uint8, device="xpu").data_ptr())
            probes["data_ptr"] = {"raw": str(pointer), "value": pointer, "high_bit": pointer >= 2**63}
        except Exception as exc:
            probes["data_ptr"]["error"] = str(exc)
            failures.append(f"data_ptr probe failed: {exc}")
        if triton is not None:
            try:
                probes["triton_add"] = _triton_vector_add(torch, triton)
                if not probes["triton_add"]["ok"]:
                    failures.append("Triton vector add mismatch")
            except Exception as exc:
                probes["triton_add"] = {"ok": False, "error": str(exc)}
                failures.append(f"Triton vector add failed: {exc}")
    read_meminfo = hooks.get("read_meminfo", _read_meminfo)

    def allocate(nbytes: int) -> Any:
        if not device_ok:
            raise RuntimeError("XPU is unavailable for the host RAM shadow probe")
        query = getattr(torch.xpu, "mem_get_info", None)
        if query is not None:
            free, _total = query()
            if int(free) < nbytes + 1024**3:
                raise MemoryError(f"XPU free {int(free)} bytes, refusing a {nbytes}-byte tensor")
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
        except Exception as exc:
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


def run_suite(
    name: str,
    *,
    worktree: str | Path,
    out_dir: str | Path,
    timeout_s: int,
    model_cache: str | None = None,
    hooks: dict[str, Any] | None = None,
) -> SuiteResult:
    """Only env executes code; all other recognized suites return TODO status."""
    if not known_suite(name):
        detail = f"unknown suite: {name}"
        return SuiteResult(name, "error", detail, {}, detail + "\n")
    if name != "env":
        return _todo(name)
    try:
        status, detail, payload = _env_probes(hooks)
        probes = payload["probes"]
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "env-probes.json").write_text(
            json.dumps(probes, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return SuiteResult(name, status, detail, probes, detail + "\n")
    except Exception as exc:
        detail = f"environment probe failed: {type(exc).__name__}: {exc}"
        return SuiteResult(name, "error", detail, {}, detail + "\n")
