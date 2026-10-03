"""K3 on XPU: the expert plan keeps pair order, and a pair's bits do not depend on the other pairs, window or chunk."""

import random

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available() or DEVICE != "xpu":
    pytest.skip("needs an XPU", allow_module_level=True)

from tensorfold.xpu.kernels import experts as X  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("experts")
E, SLOTS = 10, 8


def _bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _weights(n, k, gs, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    words = [torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device=DEVICE, dtype=torch.int64)
             .to(torch.int32) for _ in range(E)]
    scales = [(torch.rand((n, k // gs), generator=g, device=DEVICE) * 0.01 + 0.0005).half() for _ in range(E)]
    return X.pack_xpu(words, scales)


def _dequant(words, scales, gs):
    n, k8 = words.shape
    w = words.to(torch.int64) & 0xFFFFFFFF
    q = ((w[:, :, None] >> (torch.arange(8, device=w.device) * 4)) & 0xF).reshape(n, k8 * 8).double()
    return (q - 8) * scales.double().repeat_interleave(gs, 1)


def _picks(rows, seed, experts=E):
    rng = random.Random(seed)
    used = list(range(0, experts, 2))                    # odd experts stay empty
    return torch.tensor([[rng.choice(used) for _ in range(SLOTS)] for _ in range(rows)], dtype=torch.int32,
                        device=DEVICE)


@pytest.mark.parametrize("tile", [X.TILE, X.PREFILL_TILE])
def test_plan_groups_pairs_by_expert_in_pair_order(tile):
    picks = _picks(37, 1)
    p = X.plan(picks, E, tile)
    flat = picks.flatten().tolist()
    want_members = [i for e in range(E) for i, x in enumerate(flat) if x == e]
    assert p.members.tolist() == want_members
    items, first = [], 0
    for e in range(E):
        c = flat.count(e)
        items += [[e, first + tile * j, min(tile, c - tile * j)] for j in range(-(-c // tile))]
        first += c
    got = p.items.tolist()
    assert got[:len(items)] == items and all(row[2] == 0 for row in got[len(items):])
    assert len(got) == X.max_items(len(flat), E, tile)


@pytest.mark.parametrize("up", [True, False], ids=["up-relu2", "down-fp32"])
def test_decode_pair_alone_equals_pair_in_window(up):
    n, k, gs = (256, 512, 64) if up else (512, 256, 64)
    words, scales = _weights(n, k, gs, 3 if up else 4)
    rows = 16
    picks = _picks(rows, 2)
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    x = torch.randn((rows if up else rows * SLOTS, k), generator=gen, device=DEVICE).bfloat16()
    epi = X.EPI_RELU2 if up else X.EPI_FP32
    window = X.decode(x, words, scales, X.plan(picks, E), from_tokens=up, gs=gs, epi=epi)
    for _ in range(20):
        assert torch.equal(_bits(X.decode(x, words, scales, X.plan(picks, E), from_tokens=up, gs=gs, epi=epi)),
                           _bits(window))
    for pair in range(0, rows * SLOTS, 7):
        row = pair // SLOTS if up else pair
        one = X.plan(picks.flatten()[pair:pair + 1].view(1, 1), E)
        alone = X.decode(x[row:row + 1], words, scales, one, from_tokens=up, gs=gs, epi=epi)
        assert torch.equal(_bits(alone[0]), _bits(window[pair])), f"pair {pair}"
        e = int(picks.flatten()[pair])
        ref = x[row].double() @ _dequant(words[e], scales[e], gs).T
        if up:
            ref = torch.relu(ref.float().bfloat16().float()).double() ** 2
        assert (window[pair].double() - ref).abs().max() <= ref.abs().max() * 2 ** -6 + 1e-6, f"pair {pair}"


@pytest.mark.parametrize("up", [True, False], ids=["up-relu2", "down-bf16"])
def test_prompt_pairs_do_not_depend_on_the_chunk(up):
    n, k, gs = (256, 512, 64) if up else (512, 256, 64)
    words, scales = _weights(n, k, gs, 6 if up else 7)
    rows = 300
    picks = _picks(rows, 8)
    gen = torch.Generator(device=DEVICE).manual_seed(9)
    x = torch.randn((rows if up else rows * SLOTS, k), generator=gen, device=DEVICE).bfloat16()
    epi = X.EPI_RELU2 if up else X.EPI_BF16
    whole = X.prompt(x, words, scales, X.plan(picks, E, X.PREFILL_TILE), from_tokens=up, gs=gs, epi=epi)
    for split in (1, 117):
        a = X.prompt(x[:split] if up else x[:split * SLOTS], words, scales, X.plan(picks[:split], E, X.PREFILL_TILE),
                     from_tokens=up, gs=gs, epi=epi)
        b = X.prompt(x[split:] if up else x[split * SLOTS:], words, scales, X.plan(picks[split:], E, X.PREFILL_TILE),
                     from_tokens=up, gs=gs, epi=epi)
        assert torch.equal(_bits(torch.cat([a, b])), _bits(whole)), split
    e = int(picks[0, 0])
    w = _dequant(words[e], scales[e], gs).float().bfloat16().double()
    ref = x[0].double() @ w.T
    if up:
        ref = torch.relu(ref.float().bfloat16().float()).double() ** 2
    assert (whole[0].double() - ref).abs().max() <= ref.abs().max() * 2 ** -6 + 1e-6


def test_both_kernels_lower_to_dpas(monkeypatch):
    compiled = {}
    for name in ("_decode", "_prompt"):
        kernel = getattr(X, name)

        class Capture:
            def __getitem__(self, grid, kernel=kernel, name=name):
                def launch(*args, **kwargs):
                    compiled[name] = kernel[grid](*args, **kwargs)
                    return compiled[name]
                return launch

        monkeypatch.setattr(X, name, Capture())
    words, scales = _weights(128, 256, 64, 1)
    picks = _picks(4, 1)
    x = torch.randn((4, 256), device=DEVICE).bfloat16()
    X.decode(x, words, scales, X.plan(picks, E), from_tokens=True, gs=64, epi=X.EPI_RELU2)
    X.prompt(x, words, scales, X.plan(picks, E, X.PREFILL_TILE), from_tokens=True, gs=64, epi=X.EPI_RELU2)
    for name in ("_decode", "_prompt"):
        ttgir = compiled[name].asm["ttgir"]
        assert "#ttig.dpas" in ttgir or "#triton_intel_gpu.dpas" in ttgir, name
