"""The XPU 4-bit lane matmul for symmetric INT4: w = s*q + b with b = -8*s formed in registers, never stored."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..launch import Launcher
from ..layout import check_device, check_rows, check_sym
from .config import LaneConfig, lane_config, slices, split_k

__all__ = ["sym_matmul"]


@triton.jit(do_not_specialize=["M", "N", "K", "ldx"])
def _qmm_sym(X, XS, W, S, OUT, PART, M, N, K, ldx,
             GS: tl.constexpr, SK: tl.constexpr, PER: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr,
             F32: tl.constexpr, KSPLIT: tl.constexpr):
    """Program (row tile, column tile, K slice): fma(xs, -8s, fma(P, s, acc)) over its groups in ascending order."""

    WPG: tl.constexpr = GS // 8
    CH: tl.constexpr = GS // KSPLIT              # columns of one chained sub-dot of a group
    CW: tl.constexpr = CH // 8
    KG = K // GS
    K8 = K // 8
    KX = K // 64
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, CH)
    rw = tl.arange(0, CW)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER):
        g = pid_s * PER + i
        for h in tl.static_range(KSPLIT):
            x = tl.load(X + rm[:, None] * ldx + (g * GS + h * CH + rk)[None, :], mask=m_ok[:, None], other=0.0)
            words = tl.load(W + rn[:, None] * K8 + (g * WPG + h * CW + rw)[None, :], mask=n_ok[:, None], other=0)
            q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
            q = tl.reshape(q, (BLOCK_N, CH)).to(tl.float32).to(tl.bfloat16)
            if h == 0:
                p = tl.dot(x, tl.trans(q))
            else:
                p = tl.dot(x, tl.trans(q), p)
        s = tl.load(S + rn * KG + g, mask=n_ok, other=0.0).to(tl.float32)
        b = s * -8.0
        if GS == 64:
            xs = tl.load(XS + rm * KX + g, mask=m_ok, other=0.0)
        else:
            xs = tl.load(XS + rm * KX + 2 * g, mask=m_ok, other=0.0) + tl.load(XS + rm * KX + 2 * g + 1, mask=m_ok,
                                                                                other=0.0)
        acc = tl.fma(p, tl.broadcast_to(s[None, :], (BM, BLOCK_N)), acc)
        acc = tl.fma(tl.broadcast_to(xs[:, None], (BM, BLOCK_N)), tl.broadcast_to(b[None, :], (BM, BLOCK_N)), acc)
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


@triton.jit(do_not_specialize=["total"])
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr):
    """OUT = part[0] + part[1] + ... in ascending slice order, rounded to bf16 unless F32."""

    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    if F32:
        tl.store(OUT + offs, acc, mask=ok)
    else:
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)


_LAUNCH_QMM = Launcher(lambda: _qmm_sym)
_LAUNCH_REDUCE = Launcher(lambda: _reduce)


def sym_matmul(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, xs: torch.Tensor, *, gs: int,
               sk: int | None = None, bm: int | None = None, f32: bool = False,
               config: LaneConfig | None = None) -> torch.Tensor:
    """x (M, K) bf16 times the symmetric INT4 ``weight`` (N, K/8) transposed -> (M, N) bf16; ``xs``: (M, K/64) fp32.

    ``weight`` and ``scales`` must be contiguous (row slices of a contiguous weight are); x rows may be strided.
    """

    check_rows("sym_matmul", x)
    m, k = x.shape
    if weight.dim() != 2:
        raise ValueError(f"sym_matmul: weight must be 2-D (N, K/8), got {tuple(weight.shape)}")
    n = check_sym("sym_matmul", weight, scales, k, gs)
    if xs.dtype != torch.float32 or xs.shape != (m, k // 64) or not xs.is_contiguous():
        raise ValueError(f"sym_matmul: xs must be a contiguous fp32 ({m}, {k // 64}) tensor")
    check_device("sym_matmul", x, weight, scales, xs)
    if m < 1:
        raise ValueError("sym_matmul takes at least one row")
    cfg = config or lane_config(n, k, gs)
    bm = cfg.bm if bm is None else int(bm)
    if bm not in (16, 32, 64, 128):
        raise ValueError("sym_matmul: row tile must be 16, 32, 64 or 128")
    sk = int(sk) if sk else (cfg.sk or split_k(n, k, gs, cfg.bn)) if config else slices(n, k, gs)
    if (k // gs) % sk:
        raise ValueError(f"sym_matmul: {sk} K slices do not divide {k // gs} groups")
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, cfg.bn), sk)
    _LAUNCH_QMM(grid, x, xs, weight, scales, out, part, m, n, k, x.stride(0), GS=gs, SK=sk, PER=(k // gs) // sk, BM=bm,
                   BLOCK_N=cfg.bn, F32=f32, KSPLIT=cfg.ksplit, num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                   grf_mode=cfg.grf_mode, enable_fp_fusion=False)
    if sk > 1:
        total = m * n
        block = 1024
        _LAUNCH_REDUCE((triton.cdiv(total, block),), part, out, total, SK=sk, BLOCK=block, F32=f32, num_warps=4,
                       enable_fp_fusion=False)
    return out
