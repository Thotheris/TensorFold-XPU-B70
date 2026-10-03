"""The XPU gated delta rule (docs/xpu/kernels/gdn.md): trees, chains and replays run one step, bit for bit."""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

__all__ = ["DK", "ROWS", "chain", "replay", "tree"]

DK = 128
ROWS = 8           # value rows a program steps; the step's bits never depend on it
WARPS = 1


@triton.jit
def _halving_sum(x, R: tl.constexpr):
    """Sum of 128 values per row: x[i] + x[i + 64], then + 32, ... + 1; each step adds exactly two values."""

    x = tl.sum(tl.reshape(x, (R, 2, 64)), axis=1)
    x = tl.sum(tl.reshape(x, (R, 2, 32)), axis=1)
    x = tl.sum(tl.reshape(x, (R, 2, 16)), axis=1)
    x = tl.sum(tl.reshape(x, (R, 2, 8)), axis=1)
    x = tl.sum(tl.reshape(x, (R, 2, 4)), axis=1)
    x = tl.sum(tl.reshape(x, (R, 2, 2)), axis=1)
    x = tl.sum(tl.reshape(x, (R, 2, 1)), axis=1)
    return tl.reshape(x, (R,))


@triton.jit
def _step(s, key, v, g, beta, R: tl.constexpr):
    """One delta-rule step for R state rows: decay, read, correct, write; each op rounded alone."""

    s = s * g
    mem = _halving_sum(s * key[None, :], R)
    delta = (v - mem) * beta
    return s + key[None, :] * delta[:, None]


@triton.jit(do_not_specialize=["prow_stride", "pcount_stride"])
def _tree(Q, K, V, G, BETA, STATES, SIDX, STARTS, PLAN, SLOTS, Y, FINAL, FIDX,
          PK, PV, PG, PB, PROWS, PCOUNTS, prow_stride, pcount_stride,
          HK: tl.constexpr, HV: tl.constexpr, DV: tl.constexpr, NSLOTS: tl.constexpr, R: tl.constexpr,
          CHAIN: tl.constexpr, HAS_PENDING: tl.constexpr, HAS_FINAL: tl.constexpr):
    """Program (row block, value head, stream): pending rows, then the stream's plan entries in order."""

    head = tl.program_id(1)
    stream = tl.program_id(2)
    key_head = head // (HV // HK)
    rows = tl.program_id(0) * R + tl.arange(0, R)
    row_ok = rows < DV
    mask = row_ok[:, None]
    dk = tl.arange(0, 128)
    state_rows = (tl.load(SIDX + stream).to(tl.int64) * HV + head) * DV + rows
    sp = STATES + state_rows[:, None] * 128 + dk[None, :]
    s0 = tl.load(sp, mask=mask, other=0.0)
    if HAS_PENDING:
        n = tl.load(PCOUNTS + stream * pcount_stride)
        for j in range(n):
            row = tl.load(PROWS + stream * prow_stride + j).to(tl.int64)
            key = tl.load(PK + (row * HK + key_head) * 128 + dk).to(tl.float32)
            v = tl.load(PV + (row * HV + head) * DV + rows, mask=row_ok, other=0.0).to(tl.float32)
            s0 = _step(s0, key, v, tl.load(PG + row * HV + head), tl.load(PB + row * HV + head), R)
        if n > 0:
            tl.store(sp, s0, mask=mask)
    begin = tl.load(STARTS + stream)
    end = tl.load(STARTS + stream + 1)
    cur = s0
    for i in range(begin, end):
        if CHAIN:
            node = i.to(tl.int64)
            st = cur
        else:
            node = tl.load(PLAN + i * 3).to(tl.int64)
            source = tl.load(PLAN + i * 3 + 1)
            dest = tl.load(PLAN + i * 3 + 2)
            slot_rows = ((stream * NSLOTS + tl.maximum(source, 0)).to(tl.int64) * HV + head) * DV + rows
            kept = tl.load(SLOTS + slot_rows[:, None] * 128 + dk[None, :], mask=mask & (source >= 0), other=0.0)
            st = tl.where(source == -1, s0, tl.where(source == -2, cur, kept))
        key = tl.load(K + (node * HK + key_head) * 128 + dk).to(tl.float32)
        q = tl.load(Q + (node * HK + key_head) * 128 + dk).to(tl.float32)
        v = tl.load(V + (node * HV + head) * DV + rows, mask=row_ok, other=0.0).to(tl.float32)
        st = _step(st, key, v, tl.load(G + node * HV + head), tl.load(BETA + node * HV + head), R)
        y = _halving_sum(st * q[None, :], R)
        tl.store(Y + (node * HV + head) * DV + rows, y.to(tl.bfloat16), mask=row_ok)
        if not CHAIN:
            slot_rows = ((stream * NSLOTS + tl.maximum(dest, 0)).to(tl.int64) * HV + head) * DV + rows
            tl.store(SLOTS + slot_rows[:, None] * 128 + dk[None, :], st, mask=mask & (dest >= 0))
        cur = st
    if HAS_FINAL:
        final_rows = (tl.load(FIDX + stream).to(tl.int64) * HV + head) * DV + rows
        tl.store(FINAL + final_rows[:, None] * 128 + dk[None, :], cur, mask=mask)


@triton.jit(do_not_specialize=["row_stride", "count_stride", "W", "L"])
def _replay(K, V, G, BETA, STATES, OUT, ROWS_, COUNTS, row_stride, count_stride, W, L,
            HK: tl.constexpr, HV: tl.constexpr, DV: tl.constexpr, R: tl.constexpr):
    """Program (row block, value head, stream x layer): the stream's accepted rows from its state, in order."""

    head = tl.program_id(1)
    stream = tl.program_id(2) // L
    layer = tl.program_id(2) % L
    key_head = head // (HV // HK)
    rows = tl.program_id(0) * R + tl.arange(0, R)
    row_ok = rows < DV
    mask = row_ok[:, None]
    dk = tl.arange(0, 128)
    state_rows = (((stream * L + layer).to(tl.int64)) * HV + head) * DV + rows
    s = tl.load(STATES + state_rows[:, None] * 128 + dk[None, :], mask=mask, other=0.0)
    n = tl.load(COUNTS + stream * count_stride)
    for j in range(n):
        lr = layer.to(tl.int64) * W + tl.load(ROWS_ + stream * row_stride + j)
        key = tl.load(K + (lr * HK + key_head) * 128 + dk).to(tl.float32)
        v = tl.load(V + (lr * HV + head) * DV + rows, mask=row_ok, other=0.0).to(tl.float32)
        s = _step(s, key, v, tl.load(G + lr * HV + head), tl.load(BETA + lr * HV + head), R)
    tl.store(OUT + state_rows[:, None] * 128 + dk[None, :], s, mask=mask)


def _check_window(q, k, v, g, beta) -> tuple[int, int, int, int]:
    if q.dim() != 3 or q.shape[2] != DK or k.shape != q.shape or q.dtype != k.dtype:
        raise ValueError("q and k are (W, Hk, 128) of one dtype")
    if q.dtype not in (torch.bfloat16, torch.float32) or v.dtype != torch.bfloat16 or v.dim() != 3:
        raise ValueError("q, k bf16 or fp32; v (W, Hv, Dv) bf16")
    w, hk, hv, dv = q.shape[0], q.shape[1], v.shape[1], v.shape[2]
    if v.shape[0] != w or g.shape != (w, hv) or beta.shape != (w, hv) or hv % hk:
        raise ValueError("v, g and beta must cover the window's rows and value heads")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("g and beta are fp32")
    if not all(t.is_contiguous() for t in (q, k, v, g, beta)):
        raise ValueError("window inputs must be contiguous")
    return w, hk, hv, dv


def _stack(state: torch.Tensor | None, states: torch.Tensor | None, index: torch.Tensor | None, streams: int, hv: int,
           dv: int) -> tuple[torch.Tensor, torch.Tensor]:
    if (state is None) == (states is None):
        raise ValueError("pass one stream's state or a stacked states tensor")
    stack = state[None] if state is not None else states
    if stack.dtype != torch.float32 or stack.dim() != 4 or stack.shape[1:] != (hv, dv, DK) or not stack.is_contiguous():
        raise ValueError(f"states are contiguous fp32 (S, {hv}, {dv}, {DK})")
    if index is None:
        if stack.shape[0] != streams:
            raise ValueError("without an index, one state a stream")
        index = torch.arange(streams, dtype=torch.int32, device=stack.device)
    return stack, index.to(torch.int32).contiguous()


def tree(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, plan,
         state: torch.Tensor | None = None, *, states: torch.Tensor | None = None,
         state_index: torch.Tensor | None = None, pending: Sequence[torch.Tensor] | None = None,
         final: torch.Tensor | None = None, final_index: torch.Tensor | None = None) -> torch.Tensor:
    """Outputs (W, Hv, Dv) bf16 for ``plan``'s streams (``gdn.plan``); ``pending`` rows step each state first, in place.

    A plan with no slots is a chain per stream (rows in order); ``final`` then receives each stream's last state.
    """

    w, hk, hv, dv = _check_window(q, k, v, g, beta)
    streams = int(plan.starts.numel()) - 1
    stack, index = _stack(state, states, state_index, streams, hv, dv)
    dev = q.device
    y = torch.empty((w, hv, dv), dtype=torch.bfloat16, device=dev)
    chain_mode = plan.slots == 0
    slots = torch.empty((streams, plan.slots, hv, dv, DK), dtype=torch.float32, device=dev) if plan.slots else y
    if pending is not None:
        pk, pv, pg, pb, prows, pcounts = pending
        _check_window(pk, pk, pv, pg, pb)
        if prows.dtype != torch.int32 or pcounts.dtype != torch.int32 or prows.dim() != 2 or prows.stride(1) != 1:
            raise ValueError("pending rows (streams, P) and counts (streams,) are int32")
        pargs = (pk, pv, pg, pb, prows, pcounts, prows.stride(0), pcounts.stride(0))
    else:
        pargs = (q, v, g, beta, plan.starts, plan.starts, 0, 0)
    if final is not None:
        fstack = final[None] if final.dim() == 3 else final
        if fstack.dtype != torch.float32 or fstack.shape[1:] != (hv, dv, DK) or not fstack.is_contiguous():
            raise ValueError(f"final is contiguous fp32 (S, {hv}, {dv}, {DK})")
        findex = (torch.arange(streams, dtype=torch.int32, device=dev) if final_index is None
                  else final_index.to(torch.int32).contiguous())
    else:
        fstack, findex = stack, index
    grid = (triton.cdiv(dv, ROWS), hv, streams)
    _tree[grid](q, k, v, g, beta, stack, index, plan.starts, plan.entries, slots, y, fstack, findex, *pargs,
                HK=hk, HV=hv, DV=dv, NSLOTS=max(plan.slots, 1), R=ROWS, CHAIN=chain_mode,
                HAS_PENDING=pending is not None, HAS_FINAL=final is not None, num_warps=WARPS,
                enable_fp_fusion=False)
    return y


class _ChainPlan:
    """One stream whose rows run in order: no entries to read."""

    slots = 0

    def __init__(self, rows: int, device) -> None:
        self.starts = torch.tensor([0, rows], dtype=torch.int32, device=device)
        self.entries = self.starts


def chain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
          state: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
    """Prompt chain from ``state`` (read only) to ``final``: the verify step, so any chunking gives the same bits."""

    return tree(q, k, v, g, beta, _ChainPlan(q.shape[0], q.device), states=state[None].contiguous(),
                state_index=torch.zeros(1, dtype=torch.int32, device=q.device), final=final)


def replay(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, states: torch.Tensor,
           rows: torch.Tensor, counts: torch.Tensor, *, in_place: bool = False) -> torch.Tensor | None:
    """Each stream's accepted rows replayed into its states for every layer.

    k (L, W, Hk, 128), v (L, W, Hv, Dv), g and beta (L, W, Hv); states (S, L, Hv, Dv, 128) fp32; rows (S, P) and
    counts (S,) int32. Returns the new states, or None after overwriting ``states``.
    """

    if k.dim() != 4 or v.dim() != 4 or k.shape[0] != v.shape[0]:
        raise ValueError("replay takes per-layer stacks: k (L, W, Hk, 128), v (L, W, Hv, Dv)")
    layers, w, hk = k.shape[0], k.shape[1], k.shape[2]
    hv, dv = v.shape[2], v.shape[3]
    if (states.dtype != torch.float32 or states.dim() != 5 or states.shape[1:] != (layers, hv, dv, DK)
            or not states.is_contiguous()):
        raise ValueError(f"states are contiguous fp32 (S, {layers}, {hv}, {dv}, {DK})")
    if g.shape != (layers, w, hv) or beta.shape != (layers, w, hv) or k.shape[3] != DK or hv % hk:
        raise ValueError("g and beta are (L, W, Hv) and match k and v")
    if rows.dtype != torch.int32 or counts.dtype != torch.int32 or rows.dim() != 2 or rows.stride(1) != 1:
        raise ValueError("rows (S, P) and counts (S,) are int32")
    if not all(t.is_contiguous() for t in (k, v, g, beta)):
        raise ValueError("replay inputs must be contiguous")
    streams = states.shape[0]
    out = states if in_place else torch.empty_like(states)
    grid = (triton.cdiv(dv, ROWS), hv, streams * layers)
    _replay[grid](k, v, g, beta, states, out, rows, counts, rows.stride(0), counts.stride(0), w, layers,
                  HK=hk, HV=hv, DV=dv, R=ROWS, num_warps=WARPS, enable_fp_fusion=False)
    return None if in_place else out
