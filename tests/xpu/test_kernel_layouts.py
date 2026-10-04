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
