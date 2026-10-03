"""K2 on XPU: the Mamba-2 prompt scan gives the same y and state for any chunking, and matches an fp64 loop."""

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available() or DEVICE != "xpu":
    pytest.skip("needs an XPU", allow_module_level=True)

from tensorfold.xpu.kernels import mamba as M  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("mamba")
HEADS, DH, GROUPS, DS = 64, 64, 8, 128           # Nemotron 3.5's Mamba-2 layer
XD, CD = HEADS * DH, HEADS * DH + 2 * GROUPS * DS
LO, HI = 0.001, 100.0


def _inputs(rows, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    proj = (torch.randn((rows, XD + CD + HEADS), generator=g, device=DEVICE) * 0.5).bfloat16()
    xc = (torch.randn((rows, CD), generator=g, device=DEVICE) * 0.5).bfloat16()
    state = torch.randn((HEADS, DH, DS), generator=g, device=DEVICE) * 0.1
    a = -torch.rand(HEADS, generator=g, device=DEVICE) * 2 - 0.1
    d = torch.randn(HEADS, generator=g, device=DEVICE)
    dtb = torch.randn(HEADS, generator=g, device=DEVICE) * 0.5
    return proj, xc, state, a, d, dtb


def _scan(proj, xc, state, a, d, dtb):
    s = state.clone()
    y = M.scan_rows(proj, xc, s, a, d, dtb, proj.shape[0], heads=HEADS, head_dim=DH, groups=GROUPS, state_dim=DS,
                    lo=LO, hi=HI)
    return y, s


def test_chunks_give_the_one_chunk_bits():
    proj, xc, state, a, d, dtb = _inputs(300, 1)
    whole, last = _scan(proj, xc, state, a, d, dtb)
    for size in (1, 7, 64, 299):
        cur, parts = state, []
        for s0 in range(0, 300, size):
            y, cur = _scan(proj[s0:s0 + size], xc[s0:s0 + size], cur, a, d, dtb)
            parts.append(y)
        assert torch.equal(torch.cat(parts).view(torch.int16), whole.view(torch.int16)) and torch.equal(cur, last), size
    for _ in range(20):
        assert torch.equal(_scan(proj, xc, state, a, d, dtb)[0], whole)


@pytest.mark.parametrize("rows_per_program", [1, 4, 16, 64])
def test_row_tile_changes_no_bits(monkeypatch, rows_per_program):
    proj, xc, state, a, d, dtb = _inputs(40, 2)
    want = _scan(proj, xc, state, a, d, dtb)
    monkeypatch.setattr(M, "ROWS", rows_per_program)
    got = _scan(proj, xc, state, a, d, dtb)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_matches_an_fp64_loop():
    rows = 24
    proj, xc, state, a, d, dtb = _inputs(rows, 3)
    y, last = _scan(proj, xc, state, a, d, dtb)
    p, x = proj.double(), xc.double()
    s = state.double()
    rep = HEADS // GROUPS
    ref = []
    for t in range(rows):
        z = p[t, :XD].view(HEADS, DH)
        dt = torch.nn.functional.softplus(p[t, XD + CD:] + dtb.double()).clamp(LO, HI)
        xs = x[t, :XD].view(HEADS, DH)
        b = x[t, XD:XD + GROUPS * DS].view(GROUPS, DS).repeat_interleave(rep, 0)
        c = x[t, XD + GROUPS * DS:].view(GROUPS, DS).repeat_interleave(rep, 0)
        s = s * torch.exp(a.double() * dt)[:, None, None] + (xs * dt[:, None])[:, :, None] * b[:, None, :]
        out = (s * c[:, None, :]).sum(-1) + d.double()[:, None] * xs
        ref.append((out * torch.nn.functional.silu(z)).reshape(-1))
    ref = torch.stack(ref)
    assert ((y.double() - ref).abs().max() <= ref.abs().max() * 2 ** -6).item()
    assert ((last.double() - s).abs().max() <= s.abs().max() * 1e-4).item()
