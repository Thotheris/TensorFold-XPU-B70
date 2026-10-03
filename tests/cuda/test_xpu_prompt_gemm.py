"""K5 prompt GEMM on XPU: weights rounded once to bf16, one fp32 chain over K; a row never depends on its chunk."""

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available() or DEVICE != "xpu":
    pytest.skip("needs an XPU", allow_module_level=True)

from tensorfold.xpu.kernels.qmm import prompt as P  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("prompt")
SHAPES = [(500, 5120, 128), (320, 2688, 64), (256, 17408, 128)]


def _bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _weights(n, k, gs, seed, dtype=torch.float16):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device=DEVICE, dtype=torch.int64)
    return words.to(torch.int32), (torch.rand((n, k // gs), generator=g, device=DEVICE) * 0.02 + 0.001).to(dtype)


def _rounded(words, scales, gs):
    """bf16(s * (q - 8)): exact in fp32 before the one rounding to bf16."""

    n, k8 = words.shape
    w = words.to(torch.int64) & 0xFFFFFFFF
    q = ((w[:, :, None] >> (torch.arange(8, device=w.device) * 4)) & 0xF).reshape(n, k8 * 8).float()
    return ((q - 8) * scales.float().repeat_interleave(gs, 1)).bfloat16()


@pytest.mark.parametrize("f32", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("n,k,gs", SHAPES)
def test_rows_do_not_depend_on_the_chunk(n, k, gs, f32):
    words, scales = _weights(n, k, gs, n + k)
    x = torch.randn((1024, k), generator=torch.Generator(device=DEVICE).manual_seed(3), device=DEVICE).bfloat16()
    whole = P.prompt_matmul(x, words, scales, gs=gs, f32=f32)
    for size in (1, 17, 300):
        parts = [P.prompt_matmul(x[a:a + size], words, scales, gs=gs, f32=f32) for a in range(0, 1024, size)]
        assert torch.equal(_bits(torch.cat(parts)), _bits(whole)), size
    resumed = torch.cat([P.prompt_matmul(x[:511], words, scales, gs=gs, f32=f32),
                         P.prompt_matmul(x[511:], words, scales, gs=gs, f32=f32)])
    assert torch.equal(_bits(resumed), _bits(whole))
    for _ in range(20):
        assert torch.equal(_bits(P.prompt_matmul(x, words, scales, gs=gs, f32=f32)), _bits(whole))


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gs", [64, 128])
def test_one_hot_rows_read_back_the_rounded_weight(gs, scale_dtype):
    n, k = 128, 1024
    words, scales = _weights(n, k, gs, 29, scale_dtype)
    eye = torch.eye(k, device=DEVICE).bfloat16()
    want = _rounded(words, scales, gs).T.contiguous()
    assert torch.equal(_bits(P.prompt_matmul(eye, words, scales, gs=gs)), _bits(want))


@pytest.mark.parametrize("n,k,gs", SHAPES)
def test_matches_fp64_on_the_rounded_weight(n, k, gs):
    words, scales = _weights(n, k, gs, 2 * n + k)
    x = torch.randn((64, k), generator=torch.Generator(device=DEVICE).manual_seed(5), device=DEVICE).bfloat16()
    ref = x.double() @ _rounded(words, scales, gs).double().T
    err = (P.prompt_matmul(x, words, scales, gs=gs, f32=True).double() - ref).abs().max().item()
    assert err <= ref.abs().max().item() * 1e-5, err


def test_the_dot_lowers_to_dpas(monkeypatch):
    compiled = {}
    kernel = P._prompt

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                compiled["k"] = kernel[grid](*args, **kwargs)
                return compiled["k"]
            return launch

    monkeypatch.setattr(P, "_prompt", Capture())
    words, scales = _weights(128, 1024, 128, 1)
    P.prompt_matmul(torch.randn((64, 1024), device=DEVICE).bfloat16(), words, scales, gs=128)
    ttgir = compiled["k"].asm["ttgir"]
    assert "#ttig.dpas" in ttgir or "#triton_intel_gpu.dpas" in ttgir
