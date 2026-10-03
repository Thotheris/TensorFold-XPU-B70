"""Fresh launcher probes record runtime requirements without changing the image."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = ["container_probes", "validate_manifest"]


def validate_manifest(path: Path, *, torch: str, sycl: str) -> dict:
    """Native libraries require the exact torch and SYCL runtime used to build them."""
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        ok = manifest["torch"] == torch and manifest["sycl"] == sycl and manifest["dle"] == "2026.1"
        return {"ok": ok, "manifest": manifest, "error": None if ok else "native runtime versions differ"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}


def _launcher_probe(env: dict[str, str]) -> dict:
    try:
        completed = subprocess.run([sys.executable, "-m", "tools.xpu.container_probes"],
                                   env=env, text=True, capture_output=True, timeout=120, check=False)
        if completed.returncode:
            return {"ok": False, "error": completed.stderr or completed.stdout}
        return json.loads(completed.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def container_probes() -> dict:
    """Level Zero and gcc variants compile fresh launchers; optional removals are evidence only."""
    import torch
    import triton

    checks = {"python": sys.version.split()[0], "image": os.environ["TF_XPU_IMAGE_ID"]}
    with tempfile.TemporaryDirectory(prefix="tf-container-probe-") as temp:
        root = Path(temp)
        no_icd = root / "no-icd"
        no_icd.mkdir()
        # Drop Python include flags only; retain gcc's normal system headers and Triton's other includes.
        wrapper = root / "gcc-no-python"
        wrapper.write_text(
            f"#!{sys.executable}\n"
            "import subprocess,sys\n"
            "args=[]; skip=False\n"
            "for i,arg in enumerate(sys.argv[1:]):\n"
            "    if skip: skip=False; continue\n"
            "    if arg=='-I' and i+2<len(sys.argv) and 'python' in sys.argv[i+2]: skip=True; continue\n"
            "    if arg.startswith('-I') and 'python' in arg: continue\n"
            "    args.append(arg)\n"
            "sys.exit(subprocess.call(['/usr/bin/gcc',*args]))\n", encoding="utf-8",
        )
        wrapper.chmod(0o755)
        variants = {
            "gcc_python_dev": {"CC": "/usr/bin/gcc"},
            "gcc_without_python_headers": {"CC": str(wrapper)},
            "level_zero_only": {"CC": "/usr/bin/gcc", "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                                "OCL_ICD_VENDORS": str(no_icd)},
        }
        for name, overrides in variants.items():
            checks[name] = _launcher_probe({**os.environ, **overrides, "TRITON_CACHE_DIR": str(root / name)})
    native = os.environ.get("TF_XPU_EXT_DIR")
    checks["native_manifest"] = (
        validate_manifest(Path(native), torch=str(torch.__version__), sycl=str(torch.version.xpu))
        if native else {"ok": True, "mounted": False}
    )
    # Requirement probes decide future image slimming, rather than gating on a deliberately absent header.
    checks["pinned_versions"] = {
        "ok": str(torch.__version__) == "2.14.1+xpu" and str(triton.__version__) == "3.8.0"
              and str(torch.version.xpu) == "20260100",
        "torch": str(torch.__version__), "triton": str(triton.__version__), "sycl": str(torch.version.xpu),
    }
    checks["ok"] = bool(checks["gcc_python_dev"].get("ok") and checks["native_manifest"]["ok"]
                        and checks["pinned_versions"]["ok"])
    checks["compute_runtime_compatible"] = checks["gcc_python_dev"].get("ok", False)
    return checks


def main() -> int:
    """A clean process queries DPAS and compiles the same Triton add as env."""
    import torch
    import triton

    from .suites import _property, _triton_vector_add

    flags = {name: _property(torch.xpu.get_device_properties(0), name)
             for name in ("has_subgroup_matrix_multiply_accumulate", "has_subgroup_2d_block_io")}
    add = _triton_vector_add(torch, triton)
    json.dump({"ok": bool(add["ok"] and all(value is True for value in flags.values())),
               "dpas_flags": flags, "triton_add": add}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
