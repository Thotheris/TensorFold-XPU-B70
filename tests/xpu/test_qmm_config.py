"""The qmm launch constants are functions of the weight shape only, and the symmetric flag leaves CUDA weights alone."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tensorfold.xpu.kernels.qmm import XPU_CONFIG, LaneConfig, lane_config, slices, split_k
from tools.xpu.kbench import ManualTimer, bench

RECIPE = [(5120, 17408, 128), (17408, 5120, 128), (10240, 5120, 128), (10304, 2688, 64), (2688, 4096, 64)]


def test_launch_constants_take_no_row_count():
    for fn in (lane_config, split_k, slices):
        assert {"m", "rows", "x"}.isdisjoint(inspect.signature(fn).parameters), fn.__name__
    assert list(inspect.signature(lane_config).parameters) == ["n", "k", "gs"]
    assert isinstance(lane_config(5120, 17408, 128), LaneConfig)
    assert all(len(key) == 3 for key in XPU_CONFIG)


@pytest.mark.parametrize("n,k,gs", RECIPE)
def test_split_k_divides_the_groups_and_matches_the_cuda_rule(n, k, gs):
    sk = split_k(n, k, gs)
    assert (k // gs) % sk == 0 and 1 <= sk <= 8
    tiles, groups, want = -(-n // 64), k // gs, 1          # the shared CUDA rule, restated
    while want < 8 and tiles * want < 192 and groups % (want * 2) == 0 and groups // (want * 2) >= 8:
        want *= 2
    assert sk == want


def test_sym_flag_leaves_cuda_weights_unchanged():
    torch = pytest.importorskip("torch")
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    words = torch.zeros((64, 8), dtype=torch.int32)
    affine = QLinear(words, torch.ones((64, 1), dtype=torch.bfloat16), torch.zeros((64, 1), dtype=torch.bfloat16))
    assert affine.fast and not affine.sym
    sym = QLinear(words, torch.ones((64, 1), dtype=torch.float16), None, sym=True)
    assert sym.fast and sym.n == 64 and sym.k == 64
    assert not QLinear(words, torch.ones((64, 1), dtype=torch.float32), None, sym=True).fast
    assert not QLinear(words, torch.ones((64, 1), dtype=torch.float16), torch.zeros((64, 1)), gs=64).fast


def test_batched_timing_reports_the_per_launch_mean(tmp_path: Path):
    calls = []
    payload = bench(fn=lambda: calls.append(1), nbytes=1000, flops=0, name="batched", out_dir=tmp_path, warmup=2,
                    repeats=3, batch=4, timer=ManualTimer([8.0, 4.0, 12.0]), bitwise_ok=True)
    assert len(calls) == 2 + 3 * 4 and payload["batch"] == 4
    assert payload["median_us"] == pytest.approx(2000.0)      # samples of 4 launches: 2000, 1000, 3000 us each
    with pytest.raises(ValueError):
        bench(fn=lambda: None, nbytes=1, flops=0, name="bad", out_dir=tmp_path, batch=0, timer=ManualTimer([1.0]))


@pytest.mark.parametrize("n,k,gs", RECIPE + [(248320, 5120, 0), (48, 5120, 0), (1000, 3072, 128), (700, 1856, 64)])
def test_every_shape_gets_a_whole_split_and_whole_sub_dots(n, k, gs):
    cfg, sk = lane_config(n, k, gs), slices(n, k, gs)
    assert (k // (gs or 64)) % sk == 0 and cfg.bm in (16, 32, 64, 128) and cfg.bn in (16, 32, 64, 128)
    if gs:
        assert gs % cfg.ksplit == 0 and (gs // cfg.ksplit) % 16 == 0       # each sub-dot is whole DPAS K steps
