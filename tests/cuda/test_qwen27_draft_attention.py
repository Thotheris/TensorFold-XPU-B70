"""DFlash2's block attention reads each stream's context in place and matches masked attention over the same keys."""

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import draft_attention  # noqa: E402
from tensorfold.families.qwen3_5.cuda.draft_attention import append, block_attention  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("draft-attention")

D = 128


def _reference(q, k, v, keys, values, length, window, scale, causal):
    """Each stream's block over [its context | its block] with DFlash2's window mask, in fp32."""

    group, outs = q.shape[0] // k.shape[0], []
    for j, (kc, vc) in enumerate(zip(keys, values)):
        rows, s = slice(j * length, (j + 1) * length), kc.shape[1]
        kk = torch.cat((kc, k[:, rows]), 1).float().repeat_interleave(group, 0)
        vv = torch.cat((vc, v[:, rows]), 1).float().repeat_interleave(group, 0)
        qi = torch.arange(length, device=q.device)[:, None]
        ki = torch.arange(s + length, device=q.device)[None, :]
        block = ki >= s
        if causal:
            block = block & (ki <= s + qi)
        allowed = ((ki < s) & (s + qi - ki < window + 1)) | block
        att = (q[:, rows].float() @ kk.transpose(1, 2)) * scale
        o = att.masked_fill(~allowed, float("-inf")).softmax(-1) @ vv
        outs.append(o.transpose(0, 1).reshape(length, -1))
    return torch.cat(outs)


@pytest.mark.parametrize("heads,kv_heads,length", [(32, 8, 16), (16, 4, 16), (8, 2, 8)])
@pytest.mark.parametrize("causal", [False, True])
def test_block_attention_matches_masked_attention(heads, kv_heads, length, causal):
    gen = torch.Generator(device=DEVICE).manual_seed(heads + length + int(causal))
    lens, window = [1, 40, 63, 63, 130], 63             # empty-ish, short, full and past-window contexts
    lens = [min(n, window) for n in lens]
    streams = len(lens)
    q = torch.randn(heads, streams * length, D, generator=gen, device=DEVICE).bfloat16()
    k = torch.randn(kv_heads, streams * length, D, generator=gen, device=DEVICE).bfloat16()
    v = torch.randn(kv_heads, streams * length, D, generator=gen, device=DEVICE).bfloat16()
    keys = [torch.randn(kv_heads, n, D, generator=gen, device=DEVICE).bfloat16() for n in lens]
    values = [torch.randn(kv_heads, n, D, generator=gen, device=DEVICE).bfloat16() for n in lens]
    scale = D ** -0.5
    out = block_attention(q, k, v, keys, values, length, window, scale, causal)
    ref = _reference(q, k, v, keys, values, length, window, scale, causal)
    assert out.shape == (streams * length, heads * D)
    assert (out.float() - ref).abs().max().item() < 2e-2


def test_a_stream_gets_the_same_bits_alone_and_among_others():
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    heads, kv_heads, length, window = 32, 8, 16, 2047
    lens = [2047, 500, 2047]
    q = torch.randn(heads, 3 * length, D, generator=gen, device=DEVICE).bfloat16()
    k = torch.randn(kv_heads, 3 * length, D, generator=gen, device=DEVICE).bfloat16()
    v = torch.randn(kv_heads, 3 * length, D, generator=gen, device=DEVICE).bfloat16()
    keys = [torch.randn(kv_heads, n, D, generator=gen, device=DEVICE).bfloat16() for n in lens]
    values = [torch.randn(kv_heads, n, D, generator=gen, device=DEVICE).bfloat16() for n in lens]
    together = block_attention(q, k, v, keys, values, length, window, D ** -0.5)
    rows = slice(length, 2 * length)
    alone = block_attention(q[:, rows].contiguous(), k[:, rows].contiguous(), v[:, rows].contiguous(), keys[1:2],
                            values[1:2], length, window, D ** -0.5)
    assert torch.equal(together[rows], alone)


def test_append_keeps_each_streams_last_window_rows():
    """Pointer tables carry every stream's context in place (B70 addresses pass through the signed conversion)."""

    gen = torch.Generator(device=DEVICE).manual_seed(4)
    heads, window = 4, 40
    olds = [None, torch.randn(heads, 30, D, generator=gen, device=DEVICE).bfloat16(),
            torch.randn(heads, 40, D, generator=gen, device=DEVICE).bfloat16()]
    sizes = [16, 16, 7]
    new = torch.randn(heads, sum(sizes), D, generator=gen, device=DEVICE).bfloat16()
    outs = append(new, olds, sizes, window)
    first = 0
    for old, add, out in zip(olds, sizes, outs):
        rows = new[:, first:first + add]
        want = rows if old is None else torch.cat((old, rows), 1)
        assert torch.equal(out, want[:, -window:].contiguous())
        first += add


@pytest.mark.skipif(DEVICE != "xpu", reason="the XPU DPAS lowering")
def test_xpu_block_attention_lowers_to_dpas(monkeypatch):
    compiled = {}
    kernel = draft_attention._block_attention

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                compiled["k"] = kernel[grid](*args, **kwargs)
                return compiled["k"]
            return launch

    monkeypatch.setattr(draft_attention, "_block_attention", Capture())
    gen = torch.Generator(device=DEVICE).manual_seed(2)
    q = torch.randn(32, 16, D, generator=gen, device=DEVICE).bfloat16()
    k = torch.randn(8, 16, D, generator=gen, device=DEVICE).bfloat16()
    keys = [torch.randn(8, 100, D, generator=gen, device=DEVICE).bfloat16()]
    block_attention(q, k, k.clone(), keys, [keys[0].clone()], 16, 2047, D ** -0.5)
    ttgir = compiled["k"].asm["ttgir"]
    assert "#ttig.dpas" in ttgir or "#triton_intel_gpu.dpas" in ttgir
