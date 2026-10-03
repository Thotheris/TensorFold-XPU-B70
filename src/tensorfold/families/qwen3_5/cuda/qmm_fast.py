"""Keep the optimized four-bit path and dispatch other affine formats without converting their weights."""

from __future__ import annotations

import torch

from tensorfold.cuda.kernels import qmm as shared

from .qmm import group_sums as lane_group_sums
from .qmm import lane_matmul
from .weights import QLinear, Weights


def tile(q: QLinear) -> QLinear:
    if q.layout == "tiled" or not q.fast or q.sym:       # symmetric INT4 (XPU) stays in the stored N-major layout
        return q
    p = shared.pack(q.weight, q.scales, q.biases, 64)
    return QLinear(p.weight, p.scales, p.biases, layout="tiled", rows=q.n)


def untile(q: QLinear) -> QLinear:
    """The stored MLX layout again (for the fp32 reference, TP sharding or slicing rows)."""

    if q.layout != "tiled":
        return q
    return QLinear(*shared.unpack(shared.Q4(q.weight, q.scales, q.biases, q.n, q.k, 64)))


def _xpu_rows(q: QLinear, a: int, b: int) -> QLinear:
    """Rows [a, b) of a stored SYM or bf16 XPU weight as views that keep the parent weight's launch plan."""

    if q.layout == "dense":
        return QLinear(q.weight[a:b], None, None, layout="dense", bits=0, gs=0, parent=q.parent or (q.n, q.k, 0))
    return QLinear(q.weight[a:b], q.scales[a:b], None, gs=q.gs, sym=True, parent=q.parent or (q.n, q.k, q.gs))


def _xpu_stored(q: QLinear) -> bool:
    return q.sym or (q.layout == "dense" and q.weight.device.type == "xpu")


def rows(q: QLinear, a: int, b: int) -> QLinear:
    """Rows [a, b) of a tiled weight: a view when they are whole 128-row blocks from a tile edge, else a small copy.

    On XPU (stored SYM or bf16) always views, which keep the parent's plan: each column keeps its full-head bits.
    """

    if _xpu_stored(q):
        return _xpu_rows(q, a, b)
    if a % 64 == 0 and (b - a) % 128 == 0:
        return QLinear(q.weight[a // 64:b // 64], q.scales[:, a:b], q.biases[:, a:b], layout="tiled", rows=b - a)
    t0, t1 = a // 64, -(-b // 64)
    part = shared.Q4(q.weight[t0:t1], q.scales[:, t0 * 64:t1 * 64].contiguous(),
                     q.biases[:, t0 * 64:t1 * 64].contiguous(), (t1 - t0) * 64, q.k, q.gs)
    w, s, bias = shared.unpack(part)
    lo, hi = a - t0 * 64, b - t0 * 64
    return tile(QLinear(w[lo:hi].contiguous(), s[lo:hi].contiguous(), bias[lo:hi].contiguous()))


def matmul_rows(x: torch.Tensor, parts: list[QLinear]) -> torch.Tensor:
    """``x`` against row blocks of one weight, with the bits of the stacked weight's matmul (XPU: the parent's)."""

    if _xpu_stored(parts[0]):
        xs = None if parts[0].layout == "dense" else lane_group_sums(x.contiguous())
        return torch.cat([matmul(x, p, xs) for p in parts], dim=1)
    sk = shared.split_k(sum(p.n for p in parts), parts[0].k, parts[0].gs)
    xs = shared.group_sums(x, parts[0].gs)
    return torch.cat([shared.matmul(x, p, xs, sk=sk) for p in parts], dim=1)


def matmul(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """The lane matmul for either layout; both give the same bits."""

    if q.layout == "dense" and x.device.type == "xpu":
        from tensorfold.xpu.kernels.qmm import bf16_matmul, lane_config, slices

        if q.parent:
            return bf16_matmul(x, q.weight, config=lane_config(*q.parent), sk=slices(*q.parent))
        return bf16_matmul(x, q.weight)
    if q.sym and q.parent:
        from tensorfold.xpu.kernels.qmm import lane_config, slices, sym_matmul

        xs = lane_group_sums(x.contiguous()) if xs is None else xs
        return sym_matmul(x, q.weight, q.scales, xs, gs=q.gs, sk=slices(*q.parent), config=lane_config(*q.parent))
    if not q.fast:
        from tensorfold.cuda.kernels.affine import matmul as affine_matmul

        return affine_matmul(x, q)
    if q.layout == "tiled":
        return shared.matmul(x, q, xs)
    return lane_matmul(x, q.weight, q.scales, q.biases, xs=xs, gs=q.gs)


def matmul_partial(x: torch.Tensor, q: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 sums for a tiled weight, unrounded: a row-parallel rank's share of a projection."""

    if x.device.type == "xpu":
        raise ValueError("matmul_partial (row-parallel fp32 shares) is not supported on XPU")
    if not q.fast:
        from tensorfold.cuda.kernels.affine import matmul as affine_matmul

        return affine_matmul(x, q, f32=True)
    if q.layout != "tiled":
        raise ValueError("matmul_partial takes tiled weights")
    return shared.matmul(x, q, xs, f32=True)


def stack(parts: list[QLinear]) -> QLinear:
    """Several projections of the same input as one: stored-layout rows concatenated in order."""

    if any(q.layout != "mlx" for q in parts):
        raise ValueError("stack the stored layout, then tile")
    if len({_format(q) for q in parts}) != 1:
        raise ValueError("stacked projections must share an affine format and input width")
    biases = None if parts[0].sym else torch.cat([q.biases for q in parts]).contiguous()
    return QLinear(torch.cat([q.weight for q in parts]).contiguous(), torch.cat([q.scales for q in parts]).contiguous(),
                   biases, gs=parts[0].gs, bits=parts[0].bits, sym=parts[0].sym)


def _format(q: QLinear) -> tuple:
    return (q.bits, q.gs, q.k, q.scales.dtype, None if q.biases is None else q.biases.dtype, q.sym)


def _stackable(parts: list[QLinear]) -> bool:
    return all(q.layout == "mlx" for q in parts) and len({_format(q) for q in parts}) == 1


def stack_small(layer) -> None:
    """[z | b | a] and [k | v] as one matmul each: the gates and k/v are too narrow to fill the GPU alone."""

    if layer.gdn is not None and layer.gdn.zba is None and _stackable([layer.gdn.z, layer.gdn.b, layer.gdn.a]):
        layer.gdn.zba = stack([layer.gdn.z, layer.gdn.b, layer.gdn.a])
    if layer.attn is not None and layer.attn.kv is None and _stackable([layer.attn.k, layer.attn.v]):
        layer.attn.kv = stack([layer.attn.k, layer.attn.v])


def prepare(w: Weights, *, fuse: bool = False) -> None:
    """Pack every projection and the head in place; ``fuse`` changes K splits and bits, so all rounds must share it."""

    for layer in w.layers:
        if fuse:
            stack_small(layer)
        for owner, names in ((layer, ("gate", "up", "down")), (layer.gdn, ("qkv", "z", "b", "a", "out")),
                             (layer.attn, ("q", "k", "v", "o"))):
            if owner is None:
                continue
            for name in names:
                setattr(owner, name, tile(getattr(owner, name)))
        if layer.gdn is not None and layer.gdn.zba is not None:
            layer.gdn.zba = tile(layer.gdn.zba)
        if layer.attn is not None and layer.attn.kv is not None:
            layer.attn.kv = tile(layer.attn.kv)
    w.head = tile(w.head)
    torch.cuda.empty_cache()
