"""XPU wrappers: supported views (strided rows, parent-head row slices, exact out buffers) give the contiguous bits."""

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available() or DEVICE != "xpu":
    pytest.skip("needs an XPU", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.qmm import group_sums  # noqa: E402
from tensorfold.xpu.kernels import experts as X  # noqa: E402
from tensorfold.xpu.kernels.qmm import bf16_matmul, lane_config, prompt_matmul, slices, sym_matmul  # noqa: E402
from tensorfold.xpu.kernels.qmm.prompt import prompt_config  # noqa: E402

qmm = pytest.mark.xpu_kernel("qmm")
prompt = pytest.mark.xpu_kernel("prompt")
experts = pytest.mark.xpu_kernel("experts")

SHAPES = [(1000, 5120, 128), (320, 2688, 64)]
SPANS = [(0, 1), (63, 65), (17, 300), (999, 1000)]                 # clipped to N below
ROWS = [1, 16, 17, 129]


def _bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _sym(n, k, gs, seed, scale_dtype=torch.float16):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device=DEVICE, dtype=torch.int64)
    return words.to(torch.int32), (torch.rand((n, k // gs), generator=g, device=DEVICE) * 0.02 + 0.001).to(scale_dtype)


def _views(m, k, seed):
    """The contiguous rows and two strided views of the same values: a column offset and every other row."""

    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn((m, k), generator=g, device=DEVICE).bfloat16()
    wide = torch.zeros((m, k + 64), device=DEVICE, dtype=torch.bfloat16)
    wide[:, 64:] = x
    tall = torch.zeros((2 * m, k), device=DEVICE, dtype=torch.bfloat16)
    tall[::2] = x
    return x, {"offset": wide[:, 64:], "every other row": tall[::2]}


def _spans(n):
    return [(a, min(b, n)) for a, b in SPANS if a < n]


@qmm
@pytest.mark.parametrize("f32", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("n,k,gs", SHAPES)
def test_sym_views_give_the_contiguous_bits(n, k, gs, f32):
    words, scales = _sym(n, k, gs, n + k)
    plan = {"sk": slices(n, k, gs), "config": lane_config(n, k, gs)}
    for m in ROWS:
        x, views = _views(m, k, m)
        xs = group_sums(x)
        full = sym_matmul(x, words, scales, xs, gs=gs, f32=f32, **plan)
        for name, v in views.items():
            assert v.stride(0) != k and _same(sym_matmul(v, words, scales, xs, gs=gs, f32=f32, **plan), full), name
        for a, b in _spans(n):                    # parent-head row slices keep the parent's plan
            got = sym_matmul(views["offset"], words[a:b], scales[a:b], xs, gs=gs, f32=f32, **plan)
            assert _same(got, full[:, a:b].contiguous()), (m, a, b)
        r = m // 2                                # one strided row alone
        assert _same(sym_matmul(views["every other row"][r:r + 1], words, scales, xs[r:r + 1], gs=gs, f32=f32,
                                **plan), full[r:r + 1])


@qmm
def test_bf16_gemv_views_give_the_contiguous_bits():
    n, k = 1000, 5120
    w = torch.randn((n, k), generator=torch.Generator(device=DEVICE).manual_seed(1), device=DEVICE).bfloat16()
    plan = {"sk": slices(n, k, 0), "config": lane_config(n, k, 0)}
    for m in ROWS:
        x, views = _views(m, k, m)
        full = bf16_matmul(x, w, **plan)
        for name, v in views.items():
            assert _same(bf16_matmul(v, w, **plan), full), name
        for a, b in _spans(n):
            assert _same(bf16_matmul(views["offset"], w[a:b], **plan), full[:, a:b].contiguous()), (m, a, b)


@prompt
@pytest.mark.parametrize("f32", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("n,k,gs", SHAPES)
def test_prompt_views_give_the_contiguous_bits(n, k, gs, f32):
    words, scales = _sym(n, k, gs, 3 * n + k, torch.bfloat16 if gs == 64 else torch.float16)
    cfg = prompt_config(n, k, gs)
    for m in (1, 63, 64, 65, 300):
        x, views = _views(m, k, m)
        full = prompt_matmul(x, words, scales, gs=gs, f32=f32, config=cfg)
        for name, v in views.items():
            assert _same(prompt_matmul(v, words, scales, gs=gs, f32=f32, config=cfg), full), (name, m)
        for a, b in _spans(n):
            got = prompt_matmul(views["every other row"], words[a:b], scales[a:b], gs=gs, f32=f32, config=cfg)
            assert _same(got, full[:, a:b].contiguous()), (m, a, b)


E, SLOTS = 6, 4


def _experts(n, k, gs, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (E, n, k // 8), generator=g, device=DEVICE, dtype=torch.int64)
    scales = (torch.rand((E, n, k // gs), generator=g, device=DEVICE) * 0.01 + 0.0005).half()
    return words.to(torch.int32), scales


def _picks(rows, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randint(0, E - 1, (rows, SLOTS), generator=g, device=DEVICE, dtype=torch.int32)   # expert E-1 empty


@experts
@pytest.mark.parametrize("kernel", ["decode", "prompt"])
@pytest.mark.parametrize("up", [True, False], ids=["up", "down"])
@pytest.mark.parametrize("n,k,gs", [(48, 256, 64), (96, 384, 128)])          # N off the column tile
def test_expert_views_and_exact_out_give_the_contiguous_bits(kernel, up, n, k, gs):
    fn, tile = (X.decode, X.TILE) if kernel == "decode" else (X.prompt, X.PREFILL_TILE)
    epi = X.EPI_RELU2 if up else X.EPI_FP32 if kernel == "decode" else X.EPI_BF16
    words, scales = _experts(n, k, gs, n + k)
    for rows in (1, 4, 16, 17):                   # 16 tokens of one expert fill whole decode items; 17 leaves a tail
        picks = _picks(rows, rows) if rows != 16 else torch.zeros((rows, SLOTS), dtype=torch.int32, device=DEVICE)
        p = X.plan(picks, E, tile)
        pairs = rows * SLOTS
        x, views = _views(rows if up else pairs, k, rows)
        want = fn(x, words, scales, p, from_tokens=up, gs=gs, epi=epi)
        for name, v in views.items():
            assert _same(fn(v, words, scales, p, from_tokens=up, gs=gs, epi=epi), want), (name, rows)
        # an exact-size out inside a larger buffer: the kernel writes it all and nothing on either side
        big = torch.full((pairs + 2, n), -1, dtype=torch.int32 if want.element_size() == 4 else torch.int16,
                         device=DEVICE).view(want.dtype)
        got = fn(views["offset"], words, scales, p, from_tokens=up, gs=gs, epi=epi, out=big[1:-1])
        assert got.data_ptr() == big[1].data_ptr() and _same(got, want), rows
        assert (_bits(big[0]) == -1).all() and (_bits(big[-1]) == -1).all(), rows
