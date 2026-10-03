"""Nemotron-H CUDA kernels at the real shapes: a row alone equals that row in a window, and values match fp32 torch."""

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

from nemotron_fakes import random_q  # noqa: E402

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.families.nemotron_h.cuda import attention as A, glue as G, mamba as M  # noqa: E402
from tensorfold.families.nemotron_h.cuda import reference as ref  # noqa: E402
from tensorfold.families.nemotron_h.cuda.weights import MoE  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("nemotron")

D, E, TOPK, W = 2688, 128, 6, 1856
NS = TOPK + 2
H, DH, NG, DS = 64, 64, 8, 128
XD, CD = H * DH, H * DH + 2 * NG * DS
PROJ = XD + CD + H


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _moe(seed=0):
    g = _gen(seed)
    ex = grouped.make([random_q(W, D, g, scale=0.01, lead=(E + 2,))], random_q(D, W, g, scale=0.01, lead=(E + 2,)), 64)
    return MoE(router=(torch.randn(E, D, generator=g, device=DEVICE) * 0.05).bfloat16(),
               bias=torch.randn(E, generator=g, device=DEVICE) * 0.02, experts=ex)


def _moe_rows(moe, x):
    rows = x.shape[0]
    ids = torch.empty((rows, NS), dtype=torch.int32, device=DEVICE)
    wts = torch.empty((rows, NS), dtype=torch.float32, device=DEVICE)
    G.route(x, moe.router, moe.bias, ids, wts, top_k=TOPK, scaling=2.5, norm=True)
    plan = grouped.Plan(rows, NS, E + 2, DEVICE)
    grouped.route(ids, plan)
    act = torch.empty((rows * NS, W), dtype=torch.bfloat16, device=DEVICE)
    y = torch.empty((rows * NS, D), dtype=torch.float32, device=DEVICE)
    grouped.gate_up(x, moe.experts, plan, act, rows)
    grouped.down(act, moe.experts, plan, y, rows)
    return ids, wts, y


@pytest.mark.cuda_only        # CUDA's grouped experts; XPU's are in test_xpu_experts.py
def test_moe_rows_alone_equal_window_and_reference():
    moe = _moe()
    rows = 40
    x = (torch.randn(rows, D, device=DEVICE) * 0.5).bfloat16()
    ids, wts, y = _moe_rows(moe, x)
    for r in (0, 5, 17, 39):
        i1, w1, y1 = _moe_rows(moe, x[r:r + 1].contiguous())
        assert torch.equal(i1, ids[r:r + 1]) and torch.equal(w1, wts[r:r + 1])
        assert torch.equal(y1, y[NS * r:NS * r + NS]), f"row {r}"
    for m in (2, 3, 7, 16, 33):
        _, _, ym = _moe_rows(moe, x[:m].contiguous())
        assert torch.equal(ym, y[:NS * m]), m
    # the routing and the combined output against fp32 torch
    logits = x.float() @ moe.router.float().T
    ridx, rw = ref.route(logits, moe.bias, TOPK, 2.5, True)
    assert torch.equal(ridx.int(), ids[:, :TOPK])
    assert torch.allclose(rw, wts[:, :TOPK], rtol=1e-5, atol=1e-6)
    combined = (y.reshape(rows, NS, D) * wts[:, :, None]).sum(1)
    want = ref.moe(moe, x.float(), TOPK, 2.5, True)
    err = (combined - want).abs().max().item()
    assert err <= want.abs().max().item() * 2 ** -6, err


def test_route_ties_go_to_the_lower_expert():
    x = torch.zeros((3, 256), dtype=torch.bfloat16, device=DEVICE)
    router = torch.zeros((E, 256), dtype=torch.bfloat16, device=DEVICE)
    ids = torch.empty((3, NS), dtype=torch.int32, device=DEVICE)
    wts = torch.empty((3, NS), dtype=torch.float32, device=DEVICE)
    G.route(x, router, torch.zeros(E, device=DEVICE), ids, wts, top_k=TOPK, scaling=2.5, norm=True)
    assert ids[:, :TOPK].tolist() == [list(range(TOPK))] * 3
    assert ids[:, TOPK:].tolist() == [[E, E + 1]] * 3
    assert torch.allclose(wts[:, :TOPK], torch.full((3, TOPK), 2.5 / TOPK, device=DEVICE))
    assert wts[:, TOPK:].tolist() == [[1.0, 1.0]] * 3


def _mamba_inputs(seed):
    g = _gen(seed)
    conv_w = (torch.randn(4, CD, generator=g, device=DEVICE) * 0.4).bfloat16().float()
    conv_b = (torch.randn(CD, generator=g, device=DEVICE) * 0.1).bfloat16().float()
    a = -torch.exp(torch.rand(H, generator=g, device=DEVICE) * 2)
    d = (1 + torch.randn(H, generator=g, device=DEVICE) * 0.1).bfloat16().float()
    dtb = torch.randn(H, generator=g, device=DEVICE) * 0.5 - 1
    proj = (torch.randn(24, PROJ, generator=g, device=DEVICE)).bfloat16()
    return conv_w, conv_b, a, d, dtb, proj


class _MambaState:
    def __init__(self):
        self.base = torch.zeros((3, CD), dtype=torch.bfloat16, device=DEVICE)
        self.raw = torch.zeros((2, 16, CD), dtype=torch.bfloat16, device=DEVICE)
        self.xc = torch.zeros_like(self.raw)
        self.dt = torch.zeros((2, 16, H), device=DEVICE)
        self.ssm = torch.zeros((H, DH, DS), device=DEVICE)
        self.meta = torch.zeros(4, dtype=torch.int32, device=DEVICE)
        self.parity = 0
        self.prev_keep = 0

    def window(self, proj, consts):
        conv_w, conv_b, a, d, dtb = consts
        rows = proj.shape[0]
        self.meta[1], self.meta[2] = self.parity, self.prev_keep
        M.conv(proj, self.base, self.raw, self.xc, conv_w, conv_b, self.meta, rows, xd=XD)
        y = M.scan(proj, self.xc, self.dt, self.ssm, a, d, dtb, self.meta, rows, heads=H, head_dim=DH, groups=NG,
                   state_dim=DS, lo=0.0, hi=float("inf"))
        self.parity ^= 1
        return y

    def commit(self, keep):
        self.prev_keep = keep


def test_mamba_window_equals_serial_with_partial_keeps():
    conv_w, conv_b, a, d, dtb, proj = _mamba_inputs(1)
    consts = (conv_w, conv_b, a, d, dtb)
    serial = _MambaState()
    ys = []
    for t in range(12):
        ys.append(serial.window(proj[t:t + 1].contiguous(), consts))
        serial.commit(1)
    ys = torch.cat(ys)
    win = _MambaState()
    at = 0
    for rows, keep in ((5, 2), (4, 4), (6, 3), (3, 3)):
        y = win.window(proj[at:at + rows].contiguous(), consts)
        assert torch.equal(y[:keep], ys[at:at + keep]), (at, rows, keep)
        win.commit(keep)
        at += keep
    # flushing both with a one-row window leaves identical committed states
    serial.window(proj[20:21].contiguous(), consts)
    win.window(proj[20:21].contiguous(), consts)
    assert torch.equal(serial.ssm, win.ssm) and torch.equal(serial.base, win.base)


def test_mamba_matches_fp32_torch():
    conv_w, conv_b, a, d, dtb, proj = _mamba_inputs(2)
    y = _MambaState().window(proj[:10].contiguous(), (conv_w, conv_b, a, d, dtb))
    T = 10
    p = proj[:10].float()
    z, xbc, dtr = p[:, :XD], p[:, XD:XD + CD], p[:, XD + CD:]
    padded = torch.cat([torch.zeros((3, CD), device=DEVICE), xbc])
    conv = torch.nn.functional.silu(conv_b + sum(conv_w[k] * padded[k:k + T] for k in range(4)))
    xs = conv[:, :XD].reshape(T, H, DH)
    B = conv[:, XD:XD + NG * DS].reshape(T, NG, DS).repeat_interleave(H // NG, 1)
    C = conv[:, XD + NG * DS:].reshape(T, NG, DS).repeat_interleave(H // NG, 1)
    dt = torch.nn.functional.softplus(dtr + dtb)
    s = torch.zeros((H, DH, DS), device=DEVICE)
    out = []
    for t in range(T):
        s = s * torch.exp(a * dt[t])[:, None, None] + (xs[t] * dt[t][:, None])[:, :, None] * B[t][:, None, :]
        out.append((s * C[t][:, None, :]).sum(-1) + d[:, None] * xs[t])
    want = torch.stack(out).reshape(T, XD) * torch.nn.functional.silu(z)
    err = (y.float() - want).abs().max().item()
    assert err <= want.abs().max().item() * 2 ** -6, err


def test_attention_rows_alone_equal_window_and_torch():
    g = _gen(3)
    hq, hk, hd = 32, 2, 128
    past, rows = 700, 5
    kc = (torch.randn(2048, hk, hd, generator=g, device=DEVICE)).bfloat16()
    vc = (torch.randn(2048, hk, hd, generator=g, device=DEVICE)).bfloat16()
    qkv = torch.zeros(rows, (hq + 2 * hk) * hd, device=DEVICE).bfloat16()
    qkv[:, :hq * hd] = torch.randn(rows, hq * hd, generator=g, device=DEVICE).bfloat16()
    meta = torch.tensor([past, 0, 0, 0], dtype=torch.int32, device=DEVICE)
    out, xs = A.attention(qkv, kc, vc, meta, rows, heads=hq, kv_heads=hk, head_dim=hd, max_chunks=4)
    for r in range(rows):
        one_meta = torch.tensor([past + r, 0, 0, 0], dtype=torch.int32, device=DEVICE)
        o1, x1 = A.attention(qkv[r:r + 1].contiguous(), kc, vc, one_meta, 1, heads=hq, kv_heads=hk, head_dim=hd,
                             max_chunks=4)
        assert torch.equal(o1, out[r:r + 1]) and torch.equal(x1, xs[r:r + 1]), r
    q = qkv[:, :hq * hd].float().reshape(rows, hq, hd)
    kf = kc.float().repeat_interleave(hq // hk, 1)
    vf = vc.float().repeat_interleave(hq // hk, 1)
    for r in range(rows):
        L = past + r + 1
        s = torch.einsum("hd,lhd->hl", q[r], kf[:L]) * hd ** -0.5
        want = torch.einsum("hl,lhd->hd", torch.softmax(s, -1), vf[:L]).reshape(-1)
        assert (out[r].float() - want).abs().max() < 2e-2



@pytest.mark.skipif(DEVICE != "xpu", reason="the XPU DPAS lowering")
def test_xpu_router_dot_lowers_to_dpas(monkeypatch):
    compiled = {}
    kernel = G._router

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                compiled["k"] = kernel[grid](*args, **kwargs)
                return compiled["k"]
            return launch

    monkeypatch.setattr(G, "_router", Capture())
    x = torch.randn((5, D), device=DEVICE).bfloat16()
    ids = torch.empty((5, NS), dtype=torch.int32, device=DEVICE)
    wts = torch.empty((5, NS), dtype=torch.float32, device=DEVICE)
    G.route(x, (torch.randn(E, D, device=DEVICE) * 0.05).bfloat16(), torch.zeros(E, device=DEVICE), ids, wts,
            top_k=TOPK, scaling=2.5, norm=True)
    ttgir = compiled["k"].asm["ttgir"]
    assert "#ttig.dpas" in ttgir or "#triton_intel_gpu.dpas" in ttgir


@pytest.mark.parametrize("top_p", [1.0, 0.9])
def test_keyed_sampler_matches_exact_sampling_over_many_draws(top_p):
    """100,000 draws: each GPU token equals the host rule's (exact_sampling.choose_rows) on the same candidates."""

    import numpy as np

    from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
    from tensorfold.families.nemotron_h.cuda import sampler as S

    rows, vocab, top_k = 100_000, 96, 20
    g = _gen(13)
    logits = (torch.randn((rows, vocab), generator=g, device=DEVICE) * 3).bfloat16()
    logits[:, 7] = logits[:, 8]                          # ties: the lower id ranks first
    s = Sampling(temperature=0.8, top_p=top_p, top_k=top_k, min_p=0.0, seed=1234)
    params = S.Params(DEVICE)
    params.set(s)
    meta = torch.tensor([41], dtype=torch.int32, device=DEVICE)
    out = torch.empty(rows, dtype=torch.int32, device=DEVICE)
    S.keyed(logits, meta, params, out)
    vals, ids = torch.topk(logits.float(), top_k + MARGIN, dim=-1, sorted=False)
    want = choose_rows(vals.double().cpu().numpy(), ids.cpu().numpy(), np.arange(42, 42 + rows), s)
    got = out.cpu().numpy()
    assert int((got != np.asarray(want)).sum()) == 0


@pytest.mark.parametrize("rows", [1, 2, 3, 9])
def test_conv_commit_keeps_the_last_three_raw_rows(rows):
    """BASE becomes the last three raw inputs of [BASE; the chunk] (XPU: double-buffered, no barrier)."""

    g = _gen(rows)
    proj = torch.randn((rows, PROJ), generator=g, device=DEVICE).bfloat16()
    base = torch.randn((3, CD), generator=g, device=DEVICE).bfloat16()
    want = torch.cat([base, proj[:, XD:XD + CD]])[-3:].clone()
    M.commit_conv_rows(proj, base, rows, xd=XD)
    assert torch.equal(base, want)


def test_candidate_sampler_matches_exact_sampling_with_many_candidates():
    """``sample_candidates`` (the ranks' union) at top_k 50: every token equals the host rule's."""

    import numpy as np

    from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
    from tensorfold.families.nemotron_h.cuda import sampler as S

    rows, top_k = 20_000, 50
    g = _gen(17)
    vals = (torch.randn((rows, top_k + MARGIN), generator=g, device=DEVICE) * 3).float()
    ids = torch.stack([torch.randperm(4096, generator=torch.Generator().manual_seed(r))[:top_k + MARGIN]
                       for r in range(64)]).to(DEVICE).repeat(rows // 64 + 1, 1)[:rows].contiguous()
    s = Sampling(temperature=0.7, top_p=0.9, top_k=top_k, min_p=0.0, seed=99)
    params = S.Params(DEVICE)
    params.set(s)
    meta = torch.tensor([5], dtype=torch.int32, device=DEVICE)
    out = torch.empty(rows, dtype=torch.int32, device=DEVICE)
    S.sample_candidates(vals, ids, meta, params, out)
    want = choose_rows(vals.double().cpu().numpy(), ids.cpu().numpy(), np.arange(6, 6 + rows), s)
    assert int((out.cpu().numpy() != np.asarray(want)).sum()) == 0
