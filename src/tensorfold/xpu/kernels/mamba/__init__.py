"""The XPU Mamba-2 prompt scan: steps run in order, so any chunking gives the same bits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.xpu.kernels.gdn import _halving_sum

__all__ = ["ROWS", "scan_rows"]

ROWS = 4           # value rows a program steps; the step's bits never depend on it (B70 sweep: fastest)
DS = 128


@triton.jit(do_not_specialize=["W", "proj_w", "cd", "dt_off"])
def _scan_rows(PROJ, XC, STATE, A, DSK, DTB, Y, W, proj_w, cd, dt_off, lo, hi,
               XD: tl.constexpr, DH: tl.constexpr, PER_GROUP: tl.constexpr, GROUPS: tl.constexpr, R: tl.constexpr):
    """Program (head, row block): the chunk's steps in order; the state stays in registers and is stored once."""

    head = tl.program_id(0)
    rows = tl.program_id(1) * R + tl.arange(0, R)
    g = head // PER_GROUP
    i = tl.arange(0, 128)
    sp = STATE + ((head * DH + rows).to(tl.int64))[:, None] * 128 + i[None, :]
    s = tl.load(sp)
    ah = tl.load(A + head)
    d_skip = tl.load(DSK + head)
    bias = tl.load(DTB + head)
    b_col = XD + g * 128
    c_col = XD + GROUPS * 128 + g * 128
    for t in range(W):
        xrow = XC + t.to(tl.int64) * cd
        prow = PROJ + t.to(tl.int64) * proj_w
        b = tl.load(xrow + b_col + i).to(tl.float32)
        c = tl.load(xrow + c_col + i).to(tl.float32)
        x = tl.load(xrow + head * DH + rows).to(tl.float32)
        z = tl.load(prow + head * DH + rows).to(tl.float32)
        v = tl.load(prow + dt_off + head).to(tl.float32) + bias
        dt = tl.minimum(tl.maximum(tl.maximum(v, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(v))), lo), hi)
        da = tl.exp(ah * dt)
        xdt = x * dt
        s = tl.fma(tl.broadcast_to(xdt[:, None], (R, 128)), tl.broadcast_to(b[None, :], (R, 128)), s * da)
        out = _halving_sum(s * c[None, :], R)
        gz = (z / (1.0 + tl.exp(-z))).to(tl.bfloat16).to(tl.float32)
        y = gz * tl.fma(x, d_skip, out).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + t.to(tl.int64) * XD + head * DH + rows, y.to(tl.bfloat16))
    tl.store(sp, s)


def scan_rows(proj: torch.Tensor, xc: torch.Tensor, state: torch.Tensor, a: torch.Tensor, d_skip: torch.Tensor,
              dt_bias: torch.Tensor, rows: int, *, heads: int, head_dim: int, groups: int, state_dim: int, lo: float,
              hi: float) -> torch.Tensor:
    """A prompt chunk's y (rows, heads * head_dim) bf16; ``state`` ends holding the state after the chunk's last row."""

    if state_dim != DS or head_dim % ROWS or heads % groups:
        raise ValueError(f"the prompt scan takes {DS} states, head_dim a multiple of {ROWS} and whole groups")
    if state.dtype != torch.float32 or state.shape != (heads, head_dim, DS) or not state.is_contiguous():
        raise ValueError("state is contiguous fp32 (heads, head_dim, 128)")
    if proj.dtype != torch.bfloat16 or xc.dtype != torch.bfloat16 or proj.stride(1) != 1 or xc.stride(1) != 1:
        raise ValueError("proj and xc are bf16 rows with unit column stride")
    xd = heads * head_dim
    y = torch.empty((rows, xd), dtype=torch.bfloat16, device=proj.device)
    if rows:
        _scan_rows[(heads, head_dim // ROWS)](proj, xc, state, a, d_skip, dt_bias, y, rows, proj.stride(0),
                                              xc.stride(0), xd + xc.shape[1], float(lo), float(hi), XD=xd, DH=head_dim,
                                              PER_GROUP=heads // groups, GROUPS=groups, R=ROWS, num_warps=1,
                                              enable_fp_fusion=False)
    return y
