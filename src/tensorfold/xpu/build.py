"""Native XPU extensions build ahead of time in the DLE build image and load by toolchain; run time never compiles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["AOT_TARGET", "EXTENSIONS", "build_aot", "ext_dir", "load", "manifest_errors", "source_hashes",
           "sycl_flags"]

HERE = Path(__file__).parent
AOT_TARGET = "intel_gpu_bmg_g31"
# icpx compiles device code fast-math by default; these keep every fp op as written (CUDA's --fmad=false)
PRECISE = ["-O3", "-fp-model=precise", "-ffp-contract=off"]

# name -> sources (relative to this package) and whether it is ESIMD
EXTENSIONS: dict[str, dict[str, Any]] = {
    "tensorfold_xpu_hello_v1": {"sources": ("kernels/hello/hello.cpp", "kernels/hello/hello.sycl"), "esimd": False},
}


def sycl_flags(mode: str, *, esimd: bool = False) -> list[str]:
    """Device flags: ``aot`` compiles for the B70 at build time, ``jit`` ships SPIR-V the driver finalises at load."""

    if mode not in ("aot", "jit"):
        raise ValueError("mode is aot or jit")
    flags = [*PRECISE, f"-fsycl-targets={AOT_TARGET if mode == 'aot' else 'spir64'}"]
    if esimd and mode == "aot":
        flags.append(f"-Xsycl-target-backend={AOT_TARGET} \"-options -vc-codegen\"")
    return flags


def source_hashes(name: str) -> dict[str, str]:
    """sha256 of each of ``name``'s sources as installed: a prebuilt library must come from exactly these."""

    return {s: hashlib.sha256((HERE / s).read_bytes()).hexdigest() for s in EXTENSIONS[name]["sources"]}


def _dpas_in_dump(directory: Path) -> bool | None:
    """Whether IGC's shader dump holds a dpas instruction (None: nothing was dumped)."""

    texts = [p for p in directory.rglob("*") if p.is_file() and p.suffix in (".asm", ".visaasm")]
    if not texts:
        return None
    return any("dpas" in p.read_text(errors="replace") for p in texts)


def _build_one(name: str, mode: str, work: Path) -> None:
    """One extension in one mode, in this process (``build_aot`` runs each attempt in a fresh one)."""

    from torch.utils import cpp_extension

    spec = EXTENSIONS[name]
    cpp_extension.load(name=name, sources=[str(HERE / s) for s in spec["sources"]], extra_cflags=["-O3"],
                       extra_sycl_cflags=sycl_flags(mode, esimd=spec["esimd"]), build_directory=str(work),
                       verbose=True, is_python_module=False)


def build_aot(output_dir: Path, names: list[str] | None = None, modes: tuple[str, ...] = ("aot", "jit")) -> dict:
    """Build each extension into ``output_dir/<name>.so``, AOT first and JIT SPIR-V if AOT fails; write build.json.

    Each attempt runs in its own process: torch renames a module it already built once in a process.
    """

    output_dir = Path(output_dir)
    report: dict[str, Any] = {}
    for name in names or list(EXTENSIONS):
        entry: dict[str, Any] = {"mode": None, "errors": {}, "igc_dpas": None, "sources": source_hashes(name)}
        for mode in modes:
            work = output_dir / "work" / f"{name}-{mode}"
            dump = output_dir / "igc" / f"{name}-{mode}"
            work.mkdir(parents=True, exist_ok=True)
            dump.mkdir(parents=True, exist_ok=True)
            # targets come from sycl_flags only, never torch's defaults; IGC dumps the AOT device code it builds
            env = {**os.environ, "TORCH_XPU_ARCH_LIST": "", "IGC_ShaderDumpEnable": "1",
                   "IGC_DumpToCustomDir": str(dump)}
            completed = subprocess.run([sys.executable, "-m", "tensorfold.xpu.build", name, mode, str(work)], env=env,
                                       capture_output=True, text=True, check=False)
            (work / "build.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode:
                entry["errors"][mode] = (completed.stdout + completed.stderr)[-4000:]
                continue
            shutil.copy2(work / f"{name}.so", output_dir / f"{name}.so")
            entry["mode"] = mode
            entry["igc_dpas"] = _dpas_in_dump(dump)
            break
        report[name] = entry
    (output_dir / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failed = [name for name, entry in report.items() if entry["mode"] is None]
    if failed:
        raise RuntimeError(f"native XPU extensions failed to build: {', '.join(failed)} (see build.json)")
    return report


def ext_dir() -> Path | None:
    """The prebuilt extension directory (TF_XPU_EXT_DIR), written by tools/xpu/build_ext.sh."""

    value = os.environ.get("TF_XPU_EXT_DIR")
    return Path(value) if value else None


def manifest_errors(manifest: dict, torch_version: str, sycl_version: str) -> list[str]:
    """Reasons a build directory does not match the running torch and SYCL runtime (empty: it matches)."""

    errors = []
    if manifest.get("torch") != torch_version:
        errors.append(f"built for torch {manifest.get('torch')}, running {torch_version}")
    if manifest.get("sycl") != sycl_version:
        errors.append(f"built for SYCL runtime {manifest.get('sycl')}, running {sycl_version}")
    return errors


@lru_cache(maxsize=None)
def load(name: str) -> Any:
    """Import the prebuilt ``name`` from TF_XPU_EXT_DIR after checking its manifest; never compiles."""

    import torch

    directory = ext_dir()
    if directory is None:
        raise RuntimeError(f"native XPU extension {name} needs TF_XPU_EXT_DIR (build it with tools/xpu/build_ext.sh)")
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{directory} has no readable manifest.json: {exc}") from None
    errors = manifest_errors(manifest, str(torch.__version__), str(torch.version.xpu))
    if errors:
        raise RuntimeError(f"{directory} does not match this toolchain: {'; '.join(errors)}")
    path = directory / f"{name}.so"
    if not path.is_file():
        raise RuntimeError(f"{path} is missing; rebuild the extensions for this source")
    try:
        built = json.loads((directory / "build.json").read_text(encoding="utf-8"))[name]["sources"]
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError(f"{directory}/build.json does not record {name}'s sources; rebuild") from None
    if built != source_hashes(name):
        raise RuntimeError(f"{path} was built from other sources than the installed {name}; rebuild the extensions")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    _build_one(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
