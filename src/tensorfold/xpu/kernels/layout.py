"""Checks of the layouts the XPU kernels address: shapes, dtypes, strides and devices only, never tensor contents."""

from __future__ import annotations

import torch

__all__ = ["check_device", "check_out", "check_rows", "check_sym"]

SCALE_DTYPES = (torch.float16, torch.bfloat16)


def check_rows(op: str, x: torch.Tensor) -> None:
    """``x`` is 2-D bf16 with unit column stride; any row stride (the kernels take it as ``ldx``)."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(1) != 1:
        raise ValueError(f"{op}: x must be a 2-D bf16 tensor with unit column stride")


def check_sym(op: str, words: torch.Tensor, scales: torch.Tensor, k: int, gs: int) -> int:
    """Words (..., N, K/8) int32 and scales (..., N, K/gs) fp16 or bf16, both dense row-major; returns N."""

    if gs not in (64, 128):
        raise ValueError(f"{op}: the group size must be 64 or 128")
    if words.dtype != torch.int32 or words.dim() < 2 or words.shape[-1] * 8 != k or k % gs or words.shape[-2] < 1:
        raise ValueError(f"{op}: weight {tuple(words.shape)} {words.dtype} does not match K={k}, gs={gs}")
    if not words.is_contiguous():
        raise ValueError(f"{op}: weight {tuple(words.shape)} has strides {words.stride()}; the kernel reads row n at "
                         f"word n*{k // 8}, so it must be contiguous (a row slice of a contiguous weight is)")
    want = (*words.shape[:-1], k // gs)
    if scales.dtype not in SCALE_DTYPES or tuple(scales.shape) != want or not scales.is_contiguous():
        raise ValueError(f"{op}: scales must be a contiguous fp16 or bf16 {want} tensor, got {tuple(scales.shape)} "
                         f"{scales.dtype} with strides {scales.stride()}")
    return int(words.shape[-2])


def check_device(op: str, *tensors: torch.Tensor) -> None:
    """Every operand is on the first one's device."""

    device = tensors[0].device
    for t in tensors[1:]:
        if t.device != device:
            raise ValueError(f"{op}: operands on {device} and {t.device}; a launch takes one device")


def check_out(op: str, out: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    """``out`` is exactly the contiguous ``shape`` ``dtype`` buffer the kernel's stores address."""

    if tuple(out.shape) != shape or out.dtype != dtype or out.device != device or not out.is_contiguous():
        raise ValueError(f"{op}: out must be a contiguous {dtype} {shape} tensor on {device}, got {tuple(out.shape)} "
                         f"{out.dtype} with strides {out.stride()} on {out.device}")
