"""Prompt attention preserves its row bits at any depth, row offset and chunking."""

import pytest
import torch

from tests.devices import DEV, device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

from tensorfold.cuda.kernels.prefill_attention import attention, triton_attention

# 27B, Qwen3.6-35B-A3B, Nemotron, GLM-5.3 per-head keys (TF_GLM_LATENT=0), a small group
pytestmark = pytest.mark.xpu_kernel("prefill-attention")

SHAPES = [(24, 4, 256), (16, 2, 256), (32, 2, 128), (32, 32, 256), (12, 4, 128)]
CHUNKS = [(0, 1), (0, 100), (0, 333), (17, 7), (64, 64), (1000, 129), (4133, 300)]


def _caches(gen, keys, kv_heads, dim):
    k = (torch.randn(keys, kv_heads, dim, generator=gen, device=DEV) * 1.5).bfloat16()
    v = torch.randn(keys, kv_heads, dim, generator=gen, device=DEV).bfloat16()
    return k, v


@pytest.mark.cuda_only
@pytest.mark.parametrize("heads,kv_heads,dim", SHAPES)
def test_cuda_prompt_attention_has_the_triton_bits(heads, kv_heads, dim, DEV):
    gen = torch.Generator(device=DEV).manual_seed(heads * dim + kv_heads)
    scale = dim ** -0.5
    for p0, w in CHUNKS + ([(70001, 257)] if heads == 24 else []):
        k, v = _caches(gen, p0 + w + 50, kv_heads, dim)
        k[p0 + w:] = float("nan")                                   # past the chunk's keys: never read
        v[p0 + w:] = float("nan")
        q = (torch.randn(w, heads, dim, generator=gen, device=DEV) * 1.5).bfloat16()
        got = attention(q, k, v, p0, scale=scale)
        want = triton_attention(q, k, v, p0, scale=scale)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (p0, w)


@pytest.mark.parametrize("heads,kv_heads,dim", SHAPES[:3])
def test_cuda_prompt_attention_rows_do_not_depend_on_chunking(heads, kv_heads, dim, DEV):
    gen = torch.Generator(device=DEV).manual_seed(9)
    total, p0 = 900, 2000
    k, v = _caches(gen, p0 + total, kv_heads, dim)
    q = torch.randn(total, heads, dim, generator=gen, device=DEV).bfloat16()
    attend = triton_attention if DEV.type == "xpu" else attention
    whole = attend(q, k, v, p0, scale=dim ** -0.5)
    for size in (1, 15, 16, 17, 64, 100, 512):
        parts = [attend(q[a:a + size].contiguous(), k, v, p0 + a, scale=dim ** -0.5) for a in range(0, total, size)]
        assert torch.equal(whole.view(torch.int16), torch.cat(parts).view(torch.int16)), size
