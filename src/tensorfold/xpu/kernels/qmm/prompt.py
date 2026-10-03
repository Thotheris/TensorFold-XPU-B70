"""The XPU 4-bit prompt GEMM: weights rounded once to bf16, then one fp32 DPAS chain over K (kernels/prompt_gemm.md)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

__all__ = ["PROMPT_CONFIG", "PromptConfig", "prompt_config", "prompt_matmul"]

BK = 64                   # K columns a step: one scale group or half of one


@dataclass(frozen=True)
class PromptConfig:
    """Launch constants of one weight shape; none depends on the prompt's row count."""

    bm: int = 64
    bn: int = 64
    num_warps: int = 8
    num_stages: int = 2
    grf_mode: str = "default"


# (n, k, gs) -> PromptConfig; T1 fills this per recipe shape
PROMPT_CONFIG: dict[tuple[int, int, int], PromptConfig] = {}


def prompt_config(n: int, k: int, gs: int) -> PromptConfig:
    """The launch constants for an (n, k) weight in groups of ``gs``."""

    return PROMPT_CONFIG.get((n, k, gs), PromptConfig())


@triton.jit(do_not_specialize=["M", "N", "K", "ldx"])
def _prompt(X, W, S, OUT, M, N, K, ldx, GS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BLOCK_K: tl.constexpr, F32: tl.constexpr):
    """Program (row tile, column tile): acc = dot(x, bf16(fma(q, s, -8s))^T, acc) over K in ascending steps."""

    KG = K // GS
    K8 = K // 8
    rm = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    rn = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K)
    rw = tl.arange(0, BLOCK_K // 8)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for t in range(K // BLOCK_K):
        x = tl.load(X + rm[:, None] * ldx + (t * BLOCK_K + rk)[None, :], mask=m_ok[:, None], other=0.0)
        words = tl.load(W + rn[:, None] * K8 + (t * (BLOCK_K // 8) + rw)[None, :], mask=n_ok[:, None], other=0)
        q = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (BN, BLOCK_K)).to(tl.float32)
        s = tl.load(S + rn * KG + (t * BLOCK_K) // GS, mask=n_ok, other=0.0).to(tl.float32)
        s2 = tl.broadcast_to(s[:, None], (BN, BLOCK_K))
        w = tl.fma(q, s2, s2 * -8.0).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc)
    out_mask = m_ok[:, None] & n_ok[None, :]
    if F32:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
    else:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)


def prompt_matmul(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, *, gs: int, f32: bool = False,
                  config: PromptConfig | None = None) -> torch.Tensor:
    """x (M, K) bf16 times the symmetric INT4 ``weight`` (N, K/8) transposed -> (M, N); any chunking, the same bits."""

    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(1) != 1:
        raise ValueError("prompt_matmul: x must be a 2-D bf16 tensor with unit column stride")
    if gs not in (64, 128):
        raise ValueError("prompt_matmul: the group size must be 64 or 128")
    m, k = x.shape
    n = weight.shape[0]
    if weight.dtype != torch.int32 or weight.shape[1] * 8 != k or k % gs or m < 1:
        raise ValueError(f"prompt_matmul: weight {tuple(weight.shape)} does not match K={k}, gs={gs}")
    if (scales.dtype not in (torch.float16, torch.bfloat16) or scales.shape != (n, k // gs)
            or not scales.is_contiguous()):
        raise ValueError(f"prompt_matmul: scales must be a contiguous fp16 or bf16 ({n}, {k // gs}) tensor")
    cfg = config or prompt_config(n, k, gs)
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    grid = (triton.cdiv(m, cfg.bm), triton.cdiv(n, cfg.bn))
    _prompt[grid](x, weight, scales, out, m, n, k, x.stride(0), GS=gs, BM=cfg.bm, BN=cfg.bn, BLOCK_K=BK, F32=f32,
                  num_warps=cfg.num_warps, num_stages=cfg.num_stages, grf_mode=cfg.grf_mode, enable_fp_fusion=False)
    return out
