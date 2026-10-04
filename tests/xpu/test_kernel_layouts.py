"""The XPU kernel wrappers refuse the layouts, buffers and devices their kernels do not address, before any launch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.xpu.kernels import experts as X  # noqa: E402
from tensorfold.xpu.kernels.qmm import bf16 as B  # noqa: E402
from tensorfold.xpu.kernels.qmm import lane as L  # noqa: E402
from tensorfold.xpu.kernels.qmm import prompt as P  # noqa: E402


class Launched(Exception):
    """A launch was reached: the inputs passed every check."""


def _launch(*args, **kwargs):
    raise Launched


class _Kernel:
    def __getitem__(self, grid):
        return _launch


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    """Every kernel launch raises Launched, so a ValueError proves the wrapper refused before launching."""

    monkeypatch.setattr(L, "_LAUNCH_QMM", _launch)
    monkeypatch.setattr(L, "_LAUNCH_REDUCE", _launch)
    monkeypatch.setattr(B, "_LAUNCH_GEMV", _launch)
    monkeypatch.setattr(P, "_prompt", _Kernel())
    monkeypatch.setattr(X, "_LAUNCHERS", {"decode": _launch, "prompt": _launch})


def _refused(name, fn, *args, **kwargs):
    """``fn`` raises ValueError before any launch; returns the message."""

    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    except Launched:
        pytest.fail(f"{name}: launched")
    pytest.fail(f"{name}: accepted without a launch")


def _launches(name, fn, *args, **kwargs):
    with pytest.raises(Launched):
        fn(*args, **kwargs)
        pytest.fail(f"{name}: no launch")


def _sym(n, k, gs, scale_dtype=torch.float16, device="cpu"):
    return (torch.zeros((n, k // 8), dtype=torch.int32, device=device),
            torch.ones((n, k // gs), dtype=scale_dtype, device=device))


def _x(m, k, device="cpu"):
    return torch.zeros((m, k), dtype=torch.bfloat16, device=device)


def _xs(x):
    return torch.zeros((x.shape[0], x.shape[1] // 64), dtype=torch.float32, device=x.device)


def _qmm(op, x, words, scales, gs, xs=None):
    if op == "sym":
        return L.sym_matmul(x, words, scales, _xs(x) if xs is None else xs, gs=gs)
    return P.prompt_matmul(x, words, scales, gs=gs)


OPS = ["sym", "prompt"]
N, K = 96, 512


# ---- qmm decode (sym_matmul) and prompt (prompt_matmul) ----


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("gs", [64, 128])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16])
def test_supported_layouts_reach_the_launch(op, gs, scale_dtype):
    words, scales = _sym(N, K, gs, scale_dtype)
    wide = _x(5, K + 64)
    cases = {
        "contiguous": (_x(5, K), words, scales),
        "strided rows": (wide[:, 64:], words, scales),
        "every other row": (_x(10, K)[::2], words, scales),
        "head-row slice": (_x(3, K), words[63:65], scales[63:65]),
        "one weight row": (_x(1, K), words[95:], scales[95:]),
    }
    for name, (x, w, s) in cases.items():
        _launches(name, _qmm, op, x, w, s, gs)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("gs", [64, 128])
def test_unsupported_weights_are_refused(op, gs):
    words, scales = _sym(N, K, gs)
    k8 = K // 8
    bad = {
        "int64 words": words.to(torch.int64),
        "int16 words": words.to(torch.int16),
        "row stride 2 K/8": torch.zeros((N, 2 * k8), dtype=torch.int32)[:, :k8],        # the review's reproduction
        "column offset": torch.zeros((N, k8 + 1), dtype=torch.int32)[:, 1:],
        "transposed": torch.zeros((k8, N), dtype=torch.int32).T,
        "every other row": torch.zeros((2 * N, k8), dtype=torch.int32)[::2],
        "3-D": words[None],
        "wrong K": torch.zeros((N, k8 // 2), dtype=torch.int32),
        "no rows": words[:0],
    }
    for name, w in bad.items():
        _refused(name, _qmm, op, _x(4, K), w, scales[:w.shape[0]] if w.dim() == 2 else scales, gs)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("gs", [64, 128])
def test_unsupported_scales_are_refused(op, gs):
    words, scales = _sym(N, K, gs)
    kg = K // gs
    other = 128 if gs == 64 else 64
    bad = {
        "the other group size's scales": torch.ones((N, K // other), dtype=torch.float16),
        "fp32": scales.float(),
        "int8": torch.ones((N, kg), dtype=torch.int8),
        "every other column": torch.ones((N, 2 * kg), dtype=torch.float16)[:, ::2],
        "row stride 2 groups": torch.ones((N, 2 * kg), dtype=torch.float16)[:, :kg],
        "transposed": torch.ones((kg, N), dtype=torch.float16).T,
        "too few rows": scales[:-1],
        "3-D": scales[None],
    }
    for name, s in bad.items():
        assert "scales" in _refused(name, _qmm, op, _x(4, K), words, s, gs), name


@pytest.mark.parametrize("op", OPS)
def test_unsupported_rows_and_group_sizes_are_refused(op):
    words, scales = _sym(N, K, 64)
    for x in (_x(4, K).half(), _x(4, 2 * K)[:, ::2], _x(4, K)[None], _x(0, K), _x(4, K + 64)):
        with pytest.raises(ValueError):
            _qmm(op, x, words, scales, 64)
    for gs in (32, 256):
        with pytest.raises(ValueError):
            _qmm(op, _x(4, K), words, torch.ones((N, K // gs), dtype=torch.float16), gs)
    w, s = _sym(N, 576, 64)                     # K = 576 is whole g64 groups, not g128
    with pytest.raises(ValueError):
        _qmm(op, _x(4, 576), w, torch.ones((N, 4), dtype=torch.float16), 128)
    with pytest.raises(Launched):
        _qmm(op, _x(4, 576), w, s, 64)


def test_sym_matmul_refuses_unsupported_group_sums():
    words, scales = _sym(N, K, 128)
    x = _x(4, K)
    for xs in (_xs(x).double(), _xs(_x(5, K)), torch.zeros((4, 2 * K // 64))[:, ::2], torch.zeros((K // 64, 4)).T):
        with pytest.raises(ValueError):
            _qmm("sym", x, words, scales, 128, xs)


@pytest.mark.parametrize("op", OPS)
def test_operands_on_other_devices_are_refused(op):
    words, scales = _sym(N, K, 64)
    meta_w, meta_s = _sym(N, K, 64, device="meta")
    for w, s in ((meta_w, scales), (words, meta_s), (meta_w, meta_s)):
        with pytest.raises(ValueError, match="device"):
            _qmm(op, _x(4, K), w, s, 64)
    if op == "sym":
        with pytest.raises(ValueError, match="device"):
            _qmm(op, _x(4, K), words, scales, 64, _xs(_x(4, K, device="meta")))


def test_bf16_gemv_refuses_strided_weights_and_other_devices():
    w = torch.zeros((N, K), dtype=torch.bfloat16)
    with pytest.raises(Launched):
        B.bf16_matmul(_x(5, K + 64)[:, 64:], w[7:9])
    for bad in (torch.zeros((N, 2 * K), dtype=torch.bfloat16)[:, :K], torch.zeros((K, N), dtype=torch.bfloat16).T,
                w[:0]):
        with pytest.raises(ValueError):
            B.bf16_matmul(_x(5, K), bad)
    with pytest.raises(ValueError, match="device"):
        B.bf16_matmul(_x(5, K), w.to("meta"))


# ---- grouped experts (decode and prompt) ----

E, SLOTS, EN, EK = 4, 2, 32, 256


def _experts(gs, n=EN, k=EK, scale_dtype=torch.float16, device="cpu"):
    return (torch.zeros((E, n, k // 8), dtype=torch.int32, device=device),
            torch.ones((E, n, k // gs), dtype=scale_dtype, device=device))


def _plan(kernel, rows=3, experts=E):
    picks = torch.tensor([[(r + s) % E for s in range(SLOTS)] for r in range(rows)], dtype=torch.int32)
    return X.plan(picks, experts, X.TILE if kernel == "decode" else X.PREFILL_TILE)


def _run(kernel, x, words, scales, p, *, up, gs=64, epi=None, out=None):
    epi = (X.EPI_RELU2 if up else X.EPI_FP32 if kernel == "decode" else X.EPI_BF16) if epi is None else epi
    fn = X.decode if kernel == "decode" else X.prompt
    return fn(x, words, scales, p, from_tokens=up, gs=gs, epi=epi, out=out)


def _out_dtype(kernel, up):
    return torch.float32 if kernel == "decode" and not up else torch.bfloat16


KERNELS = ["decode", "prompt"]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("gs", [64, 128])
@pytest.mark.parametrize("up", [True, False], ids=["up", "down"])
def test_expert_supported_layouts_reach_the_launch(kernel, gs, up):
    words, scales = _experts(gs)
    p = _plan(kernel)
    rows = 3 if up else 3 * SLOTS
    for i, x in enumerate((_x(rows, EK), _x(rows, EK + 64)[:, 64:], _x(2 * rows, EK)[::2])):
        _launches(f"x {i}", _run, kernel, x, words, scales, p, up=up, gs=gs)
    big = torch.zeros((3 * SLOTS + 2, EN), dtype=_out_dtype(kernel, up))
    # an exact-size contiguous slice of a larger buffer
    _launches("out slice", _run, kernel, _x(rows, EK), words, scales, p, up=up, gs=gs, out=big[1:-1])


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("gs", [64, 128])
def test_expert_unsupported_weights_are_refused(kernel, gs):
    words, scales = _experts(gs)
    k8, kg = EK // 8, EK // gs
    other = 128 if gs == 64 else 64
    bad_words = {
        "int64": words.to(torch.int64),
        "row stride 2 K/8": torch.zeros((E, EN, 2 * k8), dtype=torch.int32)[..., :k8],
        "expert stride padded": torch.zeros((E, EN + 1, k8), dtype=torch.int32)[:, :EN],
        "N, K swapped": torch.zeros((E, k8, EN), dtype=torch.int32).transpose(1, 2),
        "one expert": words[0],
    }
    bad_scales = {
        "the other group size's scales": torch.ones((E, EN, EK // other), dtype=torch.float16),
        "fp32": scales.float(),
        "every other column": torch.ones((E, EN, 2 * kg), dtype=torch.float16)[..., ::2],
        "transposed": torch.ones((E, kg, EN), dtype=torch.float16).transpose(1, 2),
        "fewer experts": scales[:-1],
    }
    p = _plan(kernel)
    for name, w in bad_words.items():
        _refused(name, _run, kernel, _x(3, EK), w, scales, p, up=True, gs=gs)
    for name, s in bad_scales.items():
        assert "scales" in _refused(name, _run, kernel, _x(3, EK), words, s, p, up=True, gs=gs), name


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("up", [True, False], ids=["up", "down"])
def test_expert_unsupported_out_buffers_are_refused(kernel, up):
    words, scales = _experts(64)
    p = _plan(kernel)
    pairs = 3 * SLOTS
    dtype = _out_dtype(kernel, up)
    other = torch.bfloat16 if dtype == torch.float32 else torch.float32
    bad = {
        "one pair short": torch.empty((pairs - 1, EN), dtype=dtype),
        "one column short": torch.empty((pairs, EN - 1), dtype=dtype),
        "one pair long": torch.empty((pairs + 1, EN), dtype=dtype),
        "wrong dtype": torch.empty((pairs, EN), dtype=other),
        "fp16": torch.empty((pairs, EN), dtype=torch.float16),
        "row stride 2 N": torch.empty((pairs, 2 * EN), dtype=dtype)[:, :EN],
        "transposed": torch.empty((EN, pairs), dtype=dtype).T,
        "flat": torch.empty((pairs * EN,), dtype=dtype),
        "other device": torch.empty((pairs, EN), dtype=dtype, device="meta"),
    }
    for name, out in bad.items():
        assert "out" in _refused(name, _run, kernel, _x(3 if up else pairs, EK), words, scales, p, up=up, out=out), name


@pytest.mark.parametrize("kernel", KERNELS)
def test_expert_plans_and_rows_that_do_not_match_are_refused(kernel):
    words, scales = _experts(64)
    p = _plan(kernel)
    pairs = 3 * SLOTS
    with pytest.raises(ValueError):                  # pair rows handed to the up projection
        _run(kernel, _x(pairs, EK), words, scales, p, up=True)
    with pytest.raises(ValueError):                  # token rows handed to the down projection
        _run(kernel, _x(3, EK), words, scales, p, up=False)
    with pytest.raises(ValueError):                  # a plan over more experts than the stack holds
        _run(kernel, _x(3, EK), words, scales, _plan(kernel, experts=E + 1), up=True)
    with pytest.raises(ValueError):
        _run(kernel, _x(3, EK), words, scales, X.Plan(p.members.long(), p.items, p.tile, p.slots, p.experts), up=True)
    with pytest.raises(ValueError):
        items = torch.zeros((3, p.items.shape[0]), dtype=torch.int32).T
        _run(kernel, _x(3, EK), words, scales, X.Plan(p.members, items, p.tile, p.slots, p.experts), up=True)
    with pytest.raises(ValueError):
        _run(kernel, _x(3, EK), words, scales, p, up=True, epi=2)
    for x in (_x(3, EK).half(), _x(3, 2 * EK)[:, ::2], _x(3, EK + 64)):
        with pytest.raises(ValueError):
            _run(kernel, x, words, scales, p, up=True)


@pytest.mark.parametrize("kernel", KERNELS)
def test_expert_operands_on_other_devices_are_refused(kernel):
    words, scales = _experts(64)
    meta_w, meta_s = _experts(64, device="meta")
    p = _plan(kernel)
    for w, s in ((meta_w, scales), (words, meta_s)):
        with pytest.raises(ValueError, match="device"):
            _run(kernel, _x(3, EK), w, s, p, up=True)
    meta_plan = X.Plan(p.members.to("meta"), p.items.to("meta"), p.tile, p.slots, p.experts)
    with pytest.raises(ValueError, match="device"):
        _run(kernel, _x(3, EK), words, scales, meta_plan, up=True)
