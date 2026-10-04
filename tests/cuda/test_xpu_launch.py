"""Cached direct launches run on the current stream: a side stream gets the launches and the default stream's bits."""

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

pytestmark = [pytest.mark.xpu_kernel("qmm"),
              pytest.mark.skipif(DEVICE != "xpu", reason="the direct-launch cache is the XPU wrappers' path")]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _sym_case(dev):
    from tensorfold.families.qwen3_5.cuda.qmm import group_sums
    from tensorfold.xpu.kernels.qmm import sym_matmul

    n, k, gs = 2688, 4096, 64                  # Nemotron out_proj: split-K, so the reduce launch is cached too
    g = torch.Generator(device=dev).manual_seed(5)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device=dev, dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((n, k // gs), generator=g, device=dev) * 0.02 + 0.001).half()
    return k, lambda x: sym_matmul(x, words, scales, group_sums(x), gs=gs)


def _bf16_case(dev):
    from tensorfold.xpu.kernels.qmm import bf16_matmul

    g = torch.Generator(device=dev).manual_seed(6)
    w = torch.randn((1000, 5120), generator=g, device=dev).bfloat16()
    return 5120, lambda x: bf16_matmul(x, w)


@pytest.mark.parametrize("case", [_sym_case, _bf16_case], ids=["sym", "bf16"])
@pytest.mark.parametrize("m", [1, 5])
def test_cached_launches_run_on_the_current_stream_with_the_default_stream_bits(DEV, case, m):
    from tensorfold.xpu.kernels.qmm import bf16, lane

    k, launch = case(DEV)
    src = torch.randn((m, k), generator=torch.Generator(device=DEV).manual_seed(m), device=DEV).bfloat16()
    fresh = launch(src).clone()                # the JIT dispatch, or a cached launch if another test came first
    cached = launch(src).clone()
    torch.xpu.synchronize()
    assert torch.equal(_bits(fresh), _bits(cached))

    launchers = (lane._LAUNCH_QMM, lane._LAUNCH_REDUCE, bf16._LAUNCH_GEMV)
    saved = [dict(launcher.cache) for launcher in launchers]
    streams = []                               # the stream argument each cached direct launch hands the driver
    for launcher in launchers:
        for key, (fn, compiled, run) in list(launcher.cache.items()):
            launcher.cache[key] = (fn, compiled, lambda *a, run=run: streams.append(a[3]) or run(*a))
    side = torch.xpu.Stream()
    assert side.sycl_queue != torch.xpu.current_stream().sycl_queue
    x = torch.zeros_like(src)
    big = torch.randn((4096, 4096), device=DEV)
    torch.xpu.synchronize()
    try:
        with torch.xpu.stream(side):
            for _ in range(4):                 # a launch on another stream reads x early if the queues run concurrently
                big = big @ big / 64
            x.copy_(src)
            out = launch(x)
        side.synchronize()
        sizes = [len(launcher.cache) for launcher in launchers]
    finally:
        for launcher, cache in zip(launchers, saved):
            launcher.cache.clear()
            launcher.cache.update(cache)
    assert sizes == [len(cache) for cache in saved]                        # every launch took the cached path
    assert streams and set(streams) == {side.sycl_queue}
    assert torch.equal(_bits(out), _bits(fresh))
    again = launch(src)
    torch.xpu.synchronize()
    assert torch.equal(_bits(again), _bits(fresh))
