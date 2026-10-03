"""GPU tests select CUDA or XPU through one device and one fixture."""

from __future__ import annotations

import importlib.util
import os

import pytest

__all__ = ["DEV", "device_available", "select_device"]


def select_device(torch_module=None) -> str | None:
    """An explicit device wins; otherwise retain CUDA preference and fall back to XPU."""
    requested = os.environ.get("TF_TEST_DEVICE", "auto").lower()
    if requested == "cpu":
        return None
    if requested not in {"auto", "cuda", "xpu"}:
        raise ValueError("TF_TEST_DEVICE must be auto, cuda or xpu")
    if requested != "auto":
        return requested
    if torch_module is None:
        if importlib.util.find_spec("torch") is None:
            return None
        import torch as torch_module
    for name in ("cuda", "xpu"):
        backend = getattr(torch_module, name, None)
        if backend is not None and backend.is_available():
            return name
    return None


DEV = select_device()


def device_available(device: str | None = DEV) -> bool:
    """The selected PyTorch GPU backend must be installed and visible."""
    if device is None or importlib.util.find_spec("torch") is None:
        return False
    import torch

    backend = getattr(torch, device, None)
    return backend is not None and backend.is_available()


@pytest.fixture(name="DEV", scope="session")
def _device_fixture():
    if not device_available():
        pytest.skip(f"needs an available {DEV or 'CUDA or XPU'} device")
    import torch

    return torch.device(DEV)
