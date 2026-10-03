"""DFlash2's own kernels: the dynamic conv and the q/k prep give each row the bits it gets alone, and match torch."""

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import dflash2  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("dflash2")


def _bits(t):
    return t.view(torch.int16)


@pytest.mark.parametrize("residual", [False, True])
def test_dconv_rows_alone_and_against_torch(residual):
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    rows, d, gs = 16, 2048, 128
    x = torch.randn(rows, d, generator=gen, device=DEVICE).bfloat16()
    dyn = (torch.randn(rows, 2, 2, d // gs, generator=gen, device=DEVICE) * 0.1).bfloat16()
    base = (torch.randn(2, 2, d, generator=gen, device=DEVICE) * 0.5).bfloat16()
    res = torch.randn(rows, d, generator=gen, device=DEVICE).bfloat16() if residual else None
    for branch in (0, 1):
        out = dflash2._dconv(x, dyn, base, branch, gs, res, seg=rows)
        ref = dflash2._conv(x, dyn, base, branch, gs)
        if residual:
            ref = (res.float() + ref.float()).bfloat16()
        assert (out.float() - ref.float()).abs().max() <= ref.float().abs().max() * 2 ** -7
        # two 8-row blocks in one launch equal each block alone (a row reads only its block's previous row)
        halves = [dflash2._dconv(x[a:a + 8], dyn[a:a + 8], base, branch, gs, None if res is None else res[a:a + 8],
                                 seg=8) for a in (0, 8)]
        assert torch.equal(_bits(dflash2._dconv(x, dyn, base, branch, gs, res, seg=8)), _bits(torch.cat(halves)))


def test_prep_rows_alone_and_against_torch():
    gen = torch.Generator(device=DEVICE).manual_seed(9)
    rows, heads, kv, d, eps = 16, 4, 2, 128, 1e-6
    qkv = torch.randn(rows, (heads + 2 * kv) * d, generator=gen, device=DEVICE).bfloat16()
    qn = (torch.rand(d, generator=gen, device=DEVICE) + 0.5).bfloat16()
    kn = (torch.rand(d, generator=gen, device=DEVICE) + 0.5).bfloat16()
    inv = 1.0 / (10000 ** (torch.arange(d // 2, device=DEVICE).float() / (d // 2)))
    phase = torch.arange(100, 100 + rows, device=DEVICE).float()[:, None] * inv[None, :]
    cos, sin = phase.cos().contiguous(), phase.sin().contiguous()

    class Host:
        weights = {"layers.0.self_attn.q_norm.weight": qn, "layers.0.self_attn.k_norm.weight": kn}
        device, head_dim, kv_local = DEVICE, d, kv

    host = Host()
    host.eps = eps
    q, k, v = dflash2.DFlash2._prep(host, qkv, 0, cos, sin, heads)
    for r in range(rows):
        one = dflash2.DFlash2._prep(host, qkv[r:r + 1].contiguous(), 0, cos[r:r + 1].contiguous(),
                                    sin[r:r + 1].contiguous(), heads)
        assert all(torch.equal(_bits(a[:, r:r + 1]), _bits(b)) for a, b in zip((q, k, v), one)), f"row {r}"
    x = qkv[:, :heads * d].reshape(rows, heads, d).transpose(0, 1)
    want = dflash2._rope(dflash2._norm(x, qn, eps), torch.arange(100, 100 + rows, device=DEVICE), inv)
    assert (q.float() - want.float()).abs().max() < 0.05
    assert torch.equal(v, qkv[:, (heads + kv) * d:].reshape(rows, kv, d).transpose(0, 1).contiguous())
