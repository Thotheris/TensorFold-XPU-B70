"""Tests of the lane kernels need Metal 4 tensor units (M5-generation GPUs); elsewhere they are skipped."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# fp32 matmuls in fp32 on M5-generation GPUs, as GLM-5.3-Flash serves (its MLX_ENV); MLX reads this once a process
os.environ.setdefault("MLX_ENABLE_TF32", "0")
# no release checks or first-run notes from tests; tests/test_update.py turns them on where it tests them
os.environ.setdefault("TENSORFOLD_NO_UPDATE_CHECK", "1")

TENSOR_UNIT_TESTS = {
    "test_lane_qmm.py", "test_lane_attention.py", "test_lane_tree.py", "test_lane_fuse.py", "test_lane_glue_norm.py",
    "test_dflash_draft_vocab.py",
}


def _tensor_units() -> bool:
    try:
        from tensorfold.families.qwen3_5 import tensor_units

        return tensor_units()
    except Exception:  # noqa: BLE001 - no MLX or no Metal: no tensor units
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "torch: needs PyTorch (the CUDA backend's code); skipped where it isn't installed",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--host-only"):
        mlx_defaults = {
            "test_context_override_cannot_exceed_model_window",
            "test_every_family_names_an_importable_kernel_version",
            "test_serve_finishes_a_config_only_cache_before_loading",
        }
        for item in items:
            if item.path.name == "test_hub_and_checks.py" and item.name in mlx_defaults:
                item.add_marker(pytest.mark.skip(reason="assumes MLX family kernels or the macOS default backend"))
    if importlib.util.find_spec("torch") is None:
        no_torch = pytest.mark.skip(reason="needs PyTorch (the CUDA backend's code)")
        for item in items:
            if item.get_closest_marker("torch") is not None:
                item.add_marker(no_torch)
    if _tensor_units():
        return
    skip = pytest.mark.skip(reason="needs Metal 4 tensor units (an M5-generation GPU)")
    for item in items:
        if item.path.name in TENSOR_UNIT_TESTS:
            item.add_marker(skip)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--host-only", action="store_true", help="Exclude MLX test modules and their test dependents")


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool | None:
    """The host suite retains MLX sources but does not import MLX test modules."""
    if config.getoption("--host-only"):
        if collection_path.name == "cuda" and collection_path.is_dir():
            return True
        if collection_path.suffix == ".py":
            from tools.xpu.host_tests import mlx_modules

            if collection_path.resolve() in mlx_modules(Path(__file__).parent.resolve()):
                return True
    return None
