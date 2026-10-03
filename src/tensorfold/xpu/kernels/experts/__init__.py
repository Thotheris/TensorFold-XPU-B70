"""XPU grouped MoE experts (docs/xpu/kernels/experts.md): a (row, slot) pair's bits never depend on the other pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ..launch import Launcher

__all__ = ["PREFILL_TILE", "TILE", "Plan", "decode", "max_items", "pack_xpu", "plan", "prompt"]

TILE = 16                # pairs an item holds in decode: the rows of one DPAS tile
PREFILL_TILE = 64        # pairs an item holds in a prompt
EPI_FP32, EPI_RELU2, EPI_BF16 = 0, 1, 3      # CUDA's epilogue numbering: down (fp32), up (relu^2), prompt down (bf16)
DECODE = {"bn": 32, "num_warps": 1, "num_stages": 3}
PROMPT = {"bn": 64, "num_warps": 8, "num_stages": 2}


def max_items(pairs: int, experts: int, tile: int) -> int:
    """Items a plan of ``pairs`` can hold: one per used expert, plus one per ``tile`` pairs past its first."""

    return min(pairs, experts) + pairs // tile


@dataclass
class Plan:
    """Pairs grouped by expert: ``members`` (pair order within an expert) and ``items`` (expert, first, count)."""

    members: torch.Tensor     # (pairs,) int32
    items: torch.Tensor       # (max_items, 3) int32; unused items have count 0
    tile: int
    slots: int


def plan(picks: torch.Tensor, experts: int, tile: int = TILE) -> Plan:
    """``picks`` (R, slots) int, each pair's expert; a stable sort keeps pair order within an expert (no host sync)."""

    rows, slots = picks.shape
    flat = picks.reshape(-1).to(torch.int64)
    pairs = flat.numel()
    members = torch.sort(flat, stable=True).indices.to(torch.int32)
    counts = torch.bincount(flat, minlength=experts)
    first = torch.cumsum(counts, 0) - counts
    tiles = (counts + tile - 1) // tile
    ends = torch.cumsum(tiles, 0)
    i = torch.arange(max_items(pairs, experts, tile), device=picks.device)
    e = torch.searchsorted(ends, i, right=True).clamp(max=experts - 1)
    j = i - (ends - tiles)[e]
    count = torch.clamp(counts[e] - tile * j, 0, tile) * (i < ends[-1])
    items = torch.stack([e, first[e] + tile * j, count], 1).to(torch.int32).contiguous()
    return Plan(members.contiguous(), items, tile, slots)


def pack_xpu(words: list[torch.Tensor], scales: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert stored SYM weights (N, K/8) int32 and scales (N, K/gs) -> stacks [E, N, K/8] and [E, N, K/gs]."""

    if len({tuple(w.shape) for w in words}) != 1 or len({(tuple(s.shape), s.dtype) for s in scales}) != 1:
        raise ValueError("experts must share one weight shape and scale format")
    if words[0].dtype != torch.int32 or scales[0].dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("SYM experts are int32 words with fp16 or bf16 scales")
    return torch.stack(words).contiguous(), torch.stack(scales).contiguous()


@triton.jit
def _relu2(acc):
    u = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
    return (u * u).to(tl.bfloat16)


@triton.jit(do_not_specialize=["x_stride", "slots", "N", "K"])
def _decode(X, x_stride, slots, W, S, ITEMS, MEMBERS, OUT, N, K,
            GS: tl.constexpr, BN: tl.constexpr, T: tl.constexpr, EPI: tl.constexpr):
    """Program (item, column tile): per group, fma(xs, -8s, fma(P, s, acc)) for each of the item's pairs."""

    it = tl.program_id(0)
    e = tl.load(ITEMS + 3 * it).to(tl.int64)
    first = tl.load(ITEMS + 3 * it + 1)
    count = tl.load(ITEMS + 3 * it + 2)
    if count > 0:
        KG = K // GS
        K8 = K // 8
        r = tl.arange(0, T)
        ok = r < count
        pair = tl.load(MEMBERS + first + r, mask=ok, other=0).to(tl.int64)
        row = tl.where(slots > 0, pair // tl.maximum(slots, 1), pair)
        rn = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
        n_ok = rn < N
        rk = tl.arange(0, GS)
        rw = tl.arange(0, GS // 8)
        shifts = tl.arange(0, 8) * 4
        wb = W + e * N * K8
        sb = S + e * N * KG
        acc = tl.zeros((T, BN), dtype=tl.float32)
        for g in range(KG):
            x = tl.load(X + row[:, None] * x_stride + g * GS + rk[None, :], mask=ok[:, None], other=0.0)
            xs = tl.sum(x.to(tl.float32), axis=1)
            words = tl.load(wb + rn[:, None] * K8 + g * (GS // 8) + rw[None, :], mask=n_ok[:, None], other=0)
            q = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (BN, GS)).to(tl.float32)
            p = tl.dot(x, tl.trans(q.to(tl.bfloat16)))
            s = tl.load(sb + rn * KG + g, mask=n_ok, other=0.0).to(tl.float32)
            acc = tl.fma(p, tl.broadcast_to(s[None, :], (T, BN)), acc)
            acc = tl.fma(tl.broadcast_to(xs[:, None], (T, BN)), tl.broadcast_to((s * -8.0)[None, :], (T, BN)), acc)
        mask = ok[:, None] & n_ok[None, :]
        dst = OUT + pair[:, None] * N + rn[None, :]
        if EPI == 1:
            tl.store(dst, _relu2(acc), mask=mask)
        elif EPI == 3:
            tl.store(dst, acc.to(tl.bfloat16), mask=mask)
        else:
            tl.store(dst, acc, mask=mask)


@triton.jit(do_not_specialize=["x_stride", "slots", "N", "K"])
def _prompt(X, x_stride, slots, W, S, ITEMS, MEMBERS, OUT, N, K,
            GS: tl.constexpr, BN: tl.constexpr, T: tl.constexpr, EPI: tl.constexpr):
    """Program (item, column tile): acc = dot(x, bf16(fma(q, s, -8s))^T, acc) over K in ascending 64-column steps."""

    it = tl.program_id(0)
    e = tl.load(ITEMS + 3 * it).to(tl.int64)
    first = tl.load(ITEMS + 3 * it + 1)
    count = tl.load(ITEMS + 3 * it + 2)
    if count > 0:
        KG = K // GS
        K8 = K // 8
        r = tl.arange(0, T)
        ok = r < count
        pair = tl.load(MEMBERS + first + r, mask=ok, other=0).to(tl.int64)
        row = tl.where(slots > 0, pair // tl.maximum(slots, 1), pair)
        rn = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
        n_ok = rn < N
        rk = tl.arange(0, 64)
        rw = tl.arange(0, 8)
        shifts = tl.arange(0, 8) * 4
        wb = W + e * N * K8
        sb = S + e * N * KG
        acc = tl.zeros((T, BN), dtype=tl.float32)
        for t in range(K // 64):
            x = tl.load(X + row[:, None] * x_stride + t * 64 + rk[None, :], mask=ok[:, None], other=0.0)
            words = tl.load(wb + rn[:, None] * K8 + t * 8 + rw[None, :], mask=n_ok[:, None], other=0)
            q = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (BN, 64)).to(tl.float32)
            s = tl.broadcast_to(tl.load(sb + rn * KG + (t * 64) // GS, mask=n_ok, other=0.0).to(tl.float32)[:, None],
                                (BN, 64))
            acc = tl.dot(x, tl.trans(tl.fma(q, s, s * -8.0).to(tl.bfloat16)), acc)
        mask = ok[:, None] & n_ok[None, :]
        dst = OUT + pair[:, None] * N + rn[None, :]
        if EPI == 1:
            tl.store(dst, _relu2(acc), mask=mask)
        elif EPI == 3:
            tl.store(dst, acc.to(tl.bfloat16), mask=mask)
        else:
            tl.store(dst, acc, mask=mask)


_LAUNCHERS = {"decode": Launcher(lambda: _decode), "prompt": Launcher(lambda: _prompt)}


def _run(kernel: str, launch: dict, x: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, p: Plan, *,
         from_tokens: bool, gs: int, epi: int, out: torch.Tensor | None) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(1) != 1:
        raise ValueError("experts: x must be 2-D bf16 rows with unit column stride")
    e, n, k8 = words.shape
    k = k8 * 8
    if x.shape[1] != k or k % gs or scales.shape != (e, n, k // gs) or gs not in (64, 128):
        raise ValueError(f"experts: x (., {x.shape[1]}) does not match weights {tuple(words.shape)} in groups of {gs}")
    pairs = p.members.numel()
    dtype = torch.float32 if epi == EPI_FP32 else torch.bfloat16
    out = torch.empty((pairs, n), dtype=dtype, device=x.device) if out is None else out
    grid = (p.items.shape[0], triton.cdiv(n, launch["bn"]))
    _LAUNCHERS[kernel](grid, x, x.stride(0), p.slots if from_tokens else 0, words, scales, p.items, p.members, out, n,
                       k, GS=gs, BN=launch["bn"], T=p.tile, EPI=epi, num_warps=launch["num_warps"],
                       num_stages=launch["num_stages"], enable_fp_fusion=False)
    return out


def decode(x: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, p: Plan, *, from_tokens: bool, gs: int,
           epi: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Each pair's projection by its expert: x holds tokens (``from_tokens``, up) or a row a pair (down)."""

    if p.tile != TILE:
        raise ValueError(f"decode takes a plan of {TILE}-pair items")
    return _run("decode", DECODE, x, words, scales, p, from_tokens=from_tokens, gs=gs, epi=epi, out=out)


def prompt(x: torch.Tensor, words: torch.Tensor, scales: torch.Tensor, p: Plan, *, from_tokens: bool, gs: int,
           epi: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """The prompt form: weights rounded once to bf16, one fp32 chain over K (not decode's bits)."""

    if p.tile != PREFILL_TILE:
        raise ValueError(f"prompt takes a plan of {PREFILL_TILE}-pair items")
    return _run("prompt", PROMPT, x, words, scales, p, from_tokens=from_tokens, gs=gs, epi=epi, out=out)
