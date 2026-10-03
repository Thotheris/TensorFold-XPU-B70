"""The XPU 4-bit and bf16 decode matmuls (docs/xpu/kernels/qmm.md)."""

from __future__ import annotations

from .config import XPU_CONFIG, LaneConfig, lane_config, slices, split_k

__all__ = ["XPU_CONFIG", "LaneConfig", "bf16_matmul", "lane_config", "slices", "split_k", "sym_matmul"]


def __getattr__(name: str):
    """The Triton kernels import on first use, so the config can be read without torch or triton."""

    if name == "sym_matmul":
        from .lane import sym_matmul

        return sym_matmul
    if name == "bf16_matmul":
        from .bf16 import bf16_matmul

        return bf16_matmul
    raise AttributeError(name)
