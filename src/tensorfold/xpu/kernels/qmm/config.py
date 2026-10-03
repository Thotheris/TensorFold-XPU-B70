"""Launch constants of the XPU qmm kernels: functions of the weight shape only, never of the row count or a tuner."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BK", "BN", "XPU_CONFIG", "LaneConfig", "lane_config", "slices", "split_k"]

BN = 64                   # default output columns per program
BK = 64                   # K columns per step of the bf16 GEMV
SPLIT_TARGET = 192        # programs the K slices aim to reach (T1 retunes it for the B70's 32 Xe cores)


@dataclass(frozen=True)
class LaneConfig:
    """Tile and compile options of one weight shape; every field is part of the arithmetic contract."""

    bm: int = 32
    bn: int = BN
    num_warps: int = 4
    num_stages: int = 3
    grf_mode: str = "default"
    ksplit: int = 1             # chained sub-dots per group (fewer live registers; the dot chain keeps its K order)
    sk: int = 0                 # 0: split_k(n, k, gs)


# (n, k, gs) -> LaneConfig; gs is 0 for bf16 weights. T1 sweeps on the B70 (docs/xpu/kernels/qmm.md, Measurements):
# small single-warp programs and chained sub-dots keep the 4-bit decode out of spills.
XPU_CONFIG: dict[tuple[int, int, int], LaneConfig] = {
    (5120, 17408, 128): LaneConfig(bm=16, bn=32, num_warps=1, num_stages=3, ksplit=4, sk=4),     # A down
    (17408, 5120, 128): LaneConfig(bm=16, bn=16, num_warps=1, num_stages=3, ksplit=4, sk=4),     # A gate / up
    (10240, 5120, 128): LaneConfig(bm=16, bn=16, num_warps=1, num_stages=3, ksplit=8, sk=2),     # A in_proj_qkv
    (10304, 2688, 64): LaneConfig(bm=16, bn=32, num_warps=1, num_stages=3, ksplit=2, sk=2),      # B in_proj
    (2688, 4096, 64): LaneConfig(bm=16, bn=32, num_warps=1, num_stages=3, ksplit=4, sk=4),       # B out_proj
}


def lane_config(n: int, k: int, gs: int = 0) -> LaneConfig:
    """The launch constants for an (n, k) weight: the table entry, else the default for its kind."""

    if (n, k, gs) in XPU_CONFIG:
        return XPU_CONFIG[(n, k, gs)]
    if gs == 0:
        return LaneConfig(bm=16, bn=32, num_warps=2, num_stages=2)          # bf16: 82-84% on both heads, M=1 and 16
    return LaneConfig(bm=16, bn=32, num_warps=1, num_stages=3, ksplit=gs // 32)


def split_k(n: int, k: int, gs: int, bn: int = BN) -> int:
    """K slices for an (n, k) weight in groups of ``gs`` columns: fixed by the shape, never by the row count."""

    tiles, groups, sk = -(-n // bn), k // gs, 1
    while sk < 8 and tiles * sk < SPLIT_TARGET and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def slices(n: int, k: int, gs: int = 0) -> int:
    """The K slices production launches use for an (n, k) weight: the table's, else ``split_k`` at its tile width."""

    cfg = lane_config(n, k, gs)
    return cfg.sk or split_k(n, k, gs or BK, cfg.bn)
