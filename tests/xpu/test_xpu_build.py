"""K0: native XPU extensions build ahead of time with precise fp flags and load only when the toolchain matches."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tensorfold.xpu import build


def test_flags_keep_fp_ops_as_written_and_pick_one_target():
    aot, jit = build.sycl_flags("aot"), build.sycl_flags("jit")
    for flags in (aot, jit):
        assert {"-O3", "-fp-model=precise", "-ffp-contract=off"} <= set(flags)
        assert sum(f.startswith("-fsycl-targets=") for f in flags) == 1
    assert "-fsycl-targets=intel_gpu_bmg_g31" in aot and "-fsycl-targets=spir64" in jit
    assert any("-vc-codegen" in f for f in build.sycl_flags("aot", esimd=True))
    with pytest.raises(ValueError):
        build.sycl_flags("native")


def test_manifest_must_match_torch_and_sycl():
    good = {"torch": "2.14.1+xpu", "sycl": "20260100", "dle": "2026.1"}
    assert build.manifest_errors(good, "2.14.1+xpu", "20260100") == []
    assert len(build.manifest_errors({**good, "torch": "2.14.0+xpu"}, "2.14.1+xpu", "20260100")) == 1
    assert len(build.manifest_errors({}, "2.14.1+xpu", "20260100")) == 2


def test_load_never_compiles(monkeypatch, tmp_path: Path):
    pytest.importorskip("torch")
    build.load.cache_clear()
    monkeypatch.delenv("TF_XPU_EXT_DIR", raising=False)
    with pytest.raises(RuntimeError, match="TF_XPU_EXT_DIR"):
        build.load("tensorfold_xpu_hello_v1")
    monkeypatch.setenv("TF_XPU_EXT_DIR", str(tmp_path))
    build.load.cache_clear()
    with pytest.raises(RuntimeError, match="manifest"):
        build.load("tensorfold_xpu_hello_v1")
    (tmp_path / "manifest.json").write_text(json.dumps({"torch": "0", "sycl": "0"}), encoding="utf-8")
    build.load.cache_clear()
    with pytest.raises(RuntimeError, match="does not match"):
        build.load("tensorfold_xpu_hello_v1")


def test_each_attempt_runs_alone_and_jit_follows_a_failed_aot(monkeypatch, tmp_path: Path):
    calls = []

    def run(argv, **kwargs):
        name, mode, work = argv[-3:]
        calls.append((mode, kwargs["env"]["TORCH_XPU_ARCH_LIST"]))
        if mode == "jit":
            Path(work, f"{name}.so").write_bytes(b"so")
            return subprocess.CompletedProcess(argv, 0, "built", "")
        return subprocess.CompletedProcess(argv, 1, "", "ocloc: link failed")

    monkeypatch.setattr(build.subprocess, "run", run)
    report = build.build_aot(tmp_path)
    entry = report["tensorfold_xpu_hello_v1"]
    assert calls == [("aot", ""), ("jit", "")]
    assert entry["mode"] == "jit" and "ocloc" in entry["errors"]["aot"]
    assert (tmp_path / "tensorfold_xpu_hello_v1.so").read_bytes() == b"so"
    assert json.loads((tmp_path / "build.json").read_text())["tensorfold_xpu_hello_v1"]["mode"] == "jit"


def test_every_native_source_is_package_data():
    for spec in build.EXTENSIONS.values():
        for source in spec["sources"]:
            assert (Path(build.__file__).parent / source).is_file(), source
