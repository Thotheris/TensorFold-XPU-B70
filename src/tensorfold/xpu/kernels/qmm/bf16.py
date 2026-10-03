"""The XPU bf16-weight GEMV for the BF16 lm_head and in_proj_a/b: one fp32 chain per K slice, in a fixed order."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .config import BK, LaneConfig, lane_config, split_k
from .lane import _reduce

__all__ = ["bf16_matmul"]


@triton.jit(do_not_specialize=["M", "N", "K", "ldx"])
def _gemv(X, W, OUT, PART, M, N, K, ldx,
          SK: tl.constexpr, PER: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          F32: tl.constexpr):
    """Program (row tile, column tile, K slice): acc = dot(x chunk, w chunk) over its chunks in ascending order."""

    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER):
        k0 = (pid_s * PER + i) * BLOCK_K
        x = tl.load(X + rm[:, None] * ldx + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def bf16_matmul(x: torch.Tensor, weight: torch.Tensor, *, sk: int | None = None, bm: int | None = None,
                f32: bool = False, config: LaneConfig | None = None) -> torch.Tensor:
    """x (M, K) bf16 times the bf16 ``weight`` (N, K) transposed -> (M, N) bf16; a row's bits never depend on M."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(1) != 1:
        raise ValueError("bf16_matmul: x must be a 2-D bf16 tensor with unit column stride")
    m, k = x.shape
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or weight.shape[1] != k or not weight.is_contiguous():
        raise ValueError(f"bf16_matmul: weight {tuple(weight.shape)} {weight.dtype} does not match K={k}")
    if k % BK or m < 1:
        raise ValueError(f"bf16_matmul takes at least one row and K a multiple of {BK}")
    n = weight.shape[0]
    cfg = config or lane_config(n, k, 0)
    bm = cfg.bm if bm is None else int(bm)
    if bm not in (16, 32, 64, 128):
        raise ValueError("bf16_matmul: row tile must be 16, 32, 64 or 128")
    sk = int(sk) if sk else cfg.sk or split_k(n, k, BK, cfg.bn)
    if (k // BK) % sk:
        raise ValueError(f"bf16_matmul: {sk} K slices do not divide {k // BK} chunks")
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    part = out if sk == 1 else torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, cfg.bn), sk)
    _gemv[grid](x, weight, out, part, m, n, k, x.stride(0), SK=sk, PER=(k // BK) // sk, BM=bm, BLOCK_N=cfg.bn,
                BLOCK_K=BK, F32=f32, num_warps=cfg.num_warps, num_stages=cfg.num_stages, grf_mode=cfg.grf_mode,
                enable_fp_fusion=False)
    if sk > 1:
        total = m * n
        block = 1024
        _reduce[(triton.cdiv(total, block),)](part, out, total, SK=sk, BLOCK=block, F32=f32, num_warps=4,
                                              enable_fp_fusion=False)
    return out
