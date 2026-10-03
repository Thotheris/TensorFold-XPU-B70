"""GPU tests retain CUDA coverage and collect migrated Triton tests on XPU."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.devices import DEV, _device_fixture, device_available

# GLM's engines keep the MTP head beside DFlash2 here (TF_GLM_MTP=1).
os.environ.setdefault("TF_GLM_MTP", "1")

XPU_MODULES = {"test_qwen27_glue.py", "test_prefill_attention.py", "test_qwen27_qmm.py", "test_qmm.py"}

if not device_available():
    collect_ignore_glob = ["test_*.py"]
elif DEV == "xpu":
    collect_ignore = [path.name for path in Path(__file__).parent.glob("test_*.py") if path.name not in XPU_MODULES]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--xpu-kernel", help="Collect the named xpu_kernel group from the GPU tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda_only: currently requires CUDA kernels or CUDA engine wiring")
    config.addinivalue_line("markers", "xpu_kernel(name): GPU kernel group selected by --xpu-kernel")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    kernel = config.getoption("--xpu-kernel")
    selected, deselected = [], []
    for item in items:
        if Path(__file__).parent not in item.path.parents:
            selected.append(item)
            continue
        if item.path.name not in XPU_MODULES:
            item.add_marker(pytest.mark.cuda_only)
        if DEV == "xpu" and item.get_closest_marker("cuda_only"):
            deselected.append(item)
            continue
        groups = [marker.args[0] for marker in item.iter_markers("xpu_kernel") if marker.args]
        if kernel is not None and kernel not in groups:
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


# Imported under its original name so pytest discovers the session-scoped DEV fixture.
__all__ = ["_device_fixture"]
