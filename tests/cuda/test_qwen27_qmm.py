"""The Qwen lane matmul: rows never depend on the row count. Affine bf16 on CUDA; symmetric INT4 and bf16 on XPU."""

import inspect

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import qmm  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("qmm")
cuda_only = pytest.mark.cuda_only
xpu_only = pytest.mark.skipif(DEVICE != "xpu", reason="symmetric INT4 and the bf16 GEMV are the XPU path")

SHAPES = [(48, 5120), (1024, 5120), (5120, 6144), (10240, 5120), (5120, 17408)]
ROWS = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 384]
SYM_ROWS = [1, 2, 15, 16, 17, 33, 65, 129]
SYM_SHAPES = [(500, 5120, 128), (320, 2688, 64), (256, 17408, 128)]      # N off a tile edge; g64 and g128; long K
BF16_SHAPES = [(1000, 5120), (48, 5120), (320, 2688)]
RECIPE_SHAPES = [(5120, 17408, 128), (17408, 5120, 128), (10240, 5120, 128), (10304, 2688, 64), (2688, 4096, 64)]
BF16_HEADS = [(248320, 5120), (131072, 2688)]      # 2.5 GB and 0.7 GB: drawn in bf16 (no fp32 copy over 4 GB)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _weights(dev, n: int, k: int, seed: int):
    g = torch.Generator(device=dev).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device=dev, dtype=torch.int64)
    weight = words.to(torch.int32)
    scales = (torch.rand((n, k // 64), generator=g, device=dev) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=g, device=dev) * 0.05).to(torch.bfloat16)
    return weight, scales, biases


def _sym_weights(dev, n: int, k: int, gs: int, seed: int, scale_dtype=torch.float16):
    g = torch.Generator(device=dev).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device=dev, dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((n, k // gs), generator=g, device=dev) * 0.02 + 0.001).to(scale_dtype)
    return words, scales


def _nibbles(words: torch.Tensor) -> torch.Tensor:
    n, k8 = words.shape
    w = words.to(torch.int64) & 0xFFFFFFFF
    return ((w[:, :, None] >> (torch.arange(8, device=w.device) * 4)) & 0xF).reshape(n, k8 * 8)


def _sym_reference(words, scales, gs, dtype=torch.float64):
    """w = s * (q - 8), exact in fp32 and fp64."""

    return (_nibbles(words).to(dtype) - 8) * scales.to(dtype).repeat_interleave(gs, 1)


# ---- affine bf16 scales and biases (the CUDA lane matmul; the Triton _qmm is not read on XPU) ----


@cuda_only
@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(DEV, n, k):
    weight, scales, biases = _weights(DEV, n, k, n + k)
    g = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn((384, k), generator=g, device=DEV).to(torch.bfloat16)
    alone = torch.cat([qmm.lane_matmul(x[r:r + 1], weight, scales, biases) for r in range(384)])
    for m in ROWS:
        batch = qmm.lane_matmul(x[:m], weight, scales, biases)
        assert torch.equal(batch, alone[:m]), f"{n}x{k}: rows differ at M={m}"
    # a row placed anywhere in a window gives the same bits
    perm = torch.randperm(384, generator=torch.Generator().manual_seed(3)).to(DEV)
    shuffled = qmm.lane_matmul(x[perm], weight, scales, biases)
    assert torch.equal(shuffled, alone[perm])


@cuda_only
@pytest.mark.parametrize("n,k", SHAPES)
def test_accuracy_matches_fp32_reference(DEV, n, k):
    weight, scales, biases = _weights(DEV, n, k, 11 * n + k)
    x = torch.randn((16, k), device=DEV).to(torch.bfloat16)
    y = qmm.lane_matmul(x, weight, scales, biases).float()
    ref = x.float() @ qmm.dequantize(weight, scales, biases).T
    err = (y - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= scale * 2 ** -7, (err, scale)


def test_split_depends_only_on_shape():
    for n, k in SHAPES:
        assert qmm.split_k(n, k) == qmm.split_k(n, k)
        assert (k // 64) % qmm.split_k(n, k) == 0


@cuda_only
@pytest.mark.parametrize("n,k", SHAPES)
def test_tiled_layout_gives_the_same_bits(DEV, n, k):
    from tensorfold.families.qwen3_5.cuda import qmm_fast
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    weight, scales, biases = _weights(DEV, n, k, 5 * n + k)
    q = QLinear(weight, scales, biases)
    t = qmm_fast.tile(q)
    back = qmm_fast.untile(t)
    assert torch.equal(back.weight, weight) and torch.equal(back.scales, scales)
    x = torch.randn((384, k), device=DEV).to(torch.bfloat16)
    for m in (1, 7, 16, 17, 32, 33, 64, 100, 128, 129, 384):
        assert torch.equal(qmm_fast.matmul(x[:m], t), qmm.lane_matmul(x[:m], weight, scales, biases)), (n, k, m)


@cuda_only
@pytest.mark.parametrize("m", [1, 16, 37, 256])
def test_head_row_views_give_the_stacked_copy_bits(DEV, m):
    """The drafter's rows of the head as views (plus a small copy off a tile edge) equal a stacked copy's matmul."""

    from tensorfold.families.qwen3_5.cuda.qmm_fast import matmul, matmul_rows, rows, tile
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    n, k, spans = 248320, 256, ((0, 98304), (248032, 248320))
    weight, scales, biases = _weights(DEV, n, k, 5)
    head = tile(QLinear(weight, scales, biases))
    stacked = tile(QLinear(*(torch.cat([t[a:b] for a, b in spans]).contiguous() for t in (weight, scales, biases))))
    parts = [rows(head, a, b) for a, b in spans]
    assert parts[0].weight.data_ptr() == head.weight.data_ptr()
    x = torch.randn((m, k), generator=torch.Generator(device=DEV).manual_seed(m), device=DEV).to(torch.bfloat16)
    assert torch.equal(matmul_rows(x, parts), matmul(x, stacked))


# ---- symmetric INT4 (XPU): w = s * (q - 8) = s * q + b with b = -8 s formed in registers ----


@xpu_only
@pytest.mark.parametrize("f32", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("n,k,gs", SYM_SHAPES + RECIPE_SHAPES)
def test_sym_rows_do_not_depend_on_row_count_tile_or_position(DEV, n, k, gs, f32):
    """The unrounded fp32 sums are compared as well: bf16 rounding hides most reduction-order differences."""

    from tensorfold.xpu.kernels.qmm import slices, split_k, sym_matmul

    words, scales = _sym_weights(DEV, n, k, gs, n + k)
    x = torch.randn((max(SYM_ROWS), k), generator=torch.Generator(device=DEV).manual_seed(7), device=DEV).bfloat16()
    xs = qmm.group_sums(x)
    perm = torch.randperm(max(SYM_ROWS), generator=torch.Generator().manual_seed(3)).to(DEV)
    for sk in sorted({1, split_k(n, k, gs), slices(n, k, gs)}):
        def run(rows, xs_rows, bm=None, sk=sk):
            return sym_matmul(rows, words, scales, xs_rows, gs=gs, sk=sk, bm=bm, f32=f32)

        alone = torch.cat([run(x[r:r + 1], xs[r:r + 1], bm=16) for r in range(max(SYM_ROWS))])
        for bm in (16, 32, 64, 128):
            for m in SYM_ROWS:
                assert _same(run(x[:m], xs[:m], bm=bm), alone[:m]), f"{n}x{k} g{gs}: M={m} BM={bm} SK={sk}"
        assert _same(run(x[perm], xs[perm]), alone[perm]), f"{n}x{k} g{gs}: a row's place changed its bits"
        # xs computed inside the call from the window's rows equals xs computed for the whole window
        if not f32:
            assert _same(qmm.lane_matmul(x[:33], words, scales, None, sk=sk, gs=gs), alone[:33])


@xpu_only
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gs", [64, 128])
def test_sym_dequant_is_exact(DEV, gs, scale_dtype):
    """One-hot rows read back each weight: bf16(s * (q - 8)) bit for bit, through every K slice."""

    from tensorfold.xpu.kernels.qmm import split_k

    n, k = 128, 4096
    words, scales = _sym_weights(DEV, n, k, gs, 29, scale_dtype)
    want = _sym_reference(words, scales, gs, torch.float32).bfloat16().T.contiguous()       # (k, n)
    eye = torch.eye(k, device=DEV).bfloat16()
    xs = qmm.group_sums(eye)
    assert split_k(n, k, gs) > 1 and set(_nibbles(words).unique().tolist()) == set(range(16))
    for sk in (1, split_k(n, k, gs)):
        got = qmm.lane_matmul(eye, words, scales, None, xs=xs, sk=sk, gs=gs)
        assert _same(got, want), (gs, scale_dtype, sk)


@xpu_only
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,k,gs", SYM_SHAPES)
def test_sym_accuracy_matches_fp64_reference(DEV, n, k, gs, scale_dtype):
    words, scales = _sym_weights(DEV, n, k, gs, 11 * n + k, scale_dtype)
    x = torch.randn((16, k), generator=torch.Generator(device=DEV).manual_seed(5), device=DEV).bfloat16()
    y = qmm.lane_matmul(x, words, scales, None, gs=gs).double()
    ref = x.double() @ _sym_reference(words, scales, gs).T
    err = (y - ref).abs().max().item()
    assert err <= ref.abs().max().item() * 2 ** -7, (err, ref.abs().max().item())


def test_sym_config_depends_only_on_the_weight_shape():
    from tensorfold.xpu.kernels.qmm import lane_config, split_k

    for fn in (lane_config, split_k):
        assert {"m", "rows", "x"}.isdisjoint(inspect.signature(fn).parameters), fn.__name__
    assert list(inspect.signature(lane_config).parameters) == ["n", "k", "gs"]
    for n, k, gs in RECIPE_SHAPES:
        sk = split_k(n, k, gs)
        assert sk in (1, 2, 4, 8) and (k // gs) % sk == 0, (n, k, gs, sk)


@xpu_only
def test_xs_from_every_producer_equals_group_sums(DEV):
    """The matmul reads one xs definition: each fused producer's 64-wide sums are group_sums of what it stored."""

    from tensorfold.families.qwen3_5.cuda import glue

    g = torch.Generator(device=DEV).manual_seed(41)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).bfloat16()

    rows = 9
    _h, y, xs = glue.add_rmsnorm(randn(rows, 5120), randn(rows, 5120), (torch.rand(5120, generator=g, device=DEV)
                                                                          + 0.5).bfloat16(), 1e-6)
    assert _same(xs, qmm.group_sums(y)), "add_rmsnorm"
    a, xs = glue.swiglu(randn(rows, 17408), randn(rows, 17408))
    assert _same(xs, qmm.group_sums(a)), "swiglu"
    out, xs = glue.gated_norm(randn(rows, 48, 128), randn(rows, 48, 128), (torch.rand(128, generator=g, device=DEV)
                                                                            + 0.5).bfloat16(), 1e-6)
    assert _same(xs, qmm.group_sums(out)), "gated_norm"
    out, xs = glue.gate_mul(randn(rows, 24, 256), randn(rows, 24 * 512), heads=24, head_dim=256)
    assert _same(xs, qmm.group_sums(out)), "gate_mul"
    # and a row's group sums do not depend on the window
    x = randn(33, 5120)
    full = qmm.group_sums(x)
    assert all(_same(qmm.group_sums(x[r:r + 1]), full[r:r + 1]) for r in range(33))


@xpu_only
def test_sym_qlinear_takes_the_symmetric_path(DEV):
    from tensorfold.families.qwen3_5.cuda import qmm_fast
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    words, scales = _sym_weights(DEV, 256, 5120, 128, 3)
    q = QLinear(words, scales, None, gs=128, sym=True)
    assert q.fast and q.n == 256 and q.k == 5120 and qmm_fast.tile(q) is q
    x = torch.randn((5, 5120), device=DEV).bfloat16()
    assert _same(qmm_fast.matmul(x, q), qmm.lane_matmul(x, words, scales, None, gs=128))
    words2, scales2 = _sym_weights(DEV, 64, 5120, 128, 4)
    both = qmm_fast.stack([q, QLinear(words2, scales2, None, gs=128, sym=True)])
    assert both.sym and both.biases is None and both.n == 320 and both.k == 5120 and both.fast
    assert not QLinear(words, scales.float(), None, gs=128, sym=True).fast         # fp32 scales are refused
    assert not QLinear(words, scales, None, gs=128).fast                           # no sym flag, no biases: not ours


# ---- bf16 weights (XPU): the BF16 lm_head and linear_attn.in_proj_a/b ----


@xpu_only
@pytest.mark.parametrize("f32", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("n,k", BF16_SHAPES + BF16_HEADS)
def test_bf16_gemv_rows_do_not_depend_on_row_count_tile_or_position(DEV, n, k, f32):
    from tensorfold.xpu.kernels.qmm import bf16_matmul, slices, split_k

    g = torch.Generator(device=DEV).manual_seed(n + k)
    w = torch.randn((n, k), generator=g, device=DEV, dtype=torch.bfloat16)
    x = torch.randn((max(SYM_ROWS), k), generator=g, device=DEV).bfloat16()
    perm = torch.randperm(max(SYM_ROWS), generator=torch.Generator().manual_seed(3)).to(DEV)
    for sk in sorted({1, split_k(n, k, 64), slices(n, k, 0)}):
        alone = torch.cat([bf16_matmul(x[r:r + 1], w, sk=sk, bm=16, f32=f32) for r in range(max(SYM_ROWS))])
        for bm in (16, 32, 64, 128):
            for m in SYM_ROWS:
                assert _same(bf16_matmul(x[:m], w, sk=sk, bm=bm, f32=f32), alone[:m]), f"{n}x{k}: M={m} BM={bm} SK={sk}"
        assert _same(bf16_matmul(x[perm], w, sk=sk, f32=f32), alone[perm])


@xpu_only
@pytest.mark.parametrize("n,k", BF16_SHAPES)
def test_bf16_gemv_matches_fp64_reference(DEV, n, k):
    from tensorfold.xpu.kernels.qmm import bf16_matmul

    g = torch.Generator(device=DEV).manual_seed(2 * n + k)
    w = torch.randn((n, k), generator=g, device=DEV).bfloat16()
    x = torch.randn((16, k), generator=g, device=DEV).bfloat16()
    ref = x.double() @ w.double().T
    err = (bf16_matmul(x, w).double() - ref).abs().max().item()
    assert err <= ref.abs().max().item() * 2 ** -8, (err, ref.abs().max().item())


@xpu_only
def test_dense_projections_route_to_the_bf16_gemv(DEV):
    from tensorfold.families.qwen3_5.cuda import qmm_fast
    from tensorfold.families.qwen3_5.cuda.weights import QLinear
    from tensorfold.xpu.kernels.qmm import bf16_matmul

    w = torch.randn((48, 5120), device=DEV).bfloat16()
    x = torch.randn((3, 5120), device=DEV).bfloat16()
    q = QLinear(w, None, None, layout="dense", bits=0, gs=0)
    assert _same(qmm_fast.matmul(x, q), bf16_matmul(x, w))


SPANS = {
    "contiguous": ((0, 300), (300, 1000)),
    "disjoint": ((0, 64), (500, 700), (936, 1000)),
    "boundary": ((63, 65), (999, 1000), (0, 1)),
}


@xpu_only
@pytest.mark.parametrize("dense", [False, True], ids=["sym", "bf16"])
@pytest.mark.parametrize("spans", sorted(SPANS))
def test_xpu_head_rows_give_the_full_heads_columns(DEV, spans, dense):
    """DFlash2/MTP row selections of a stored head: each column has its bits in the full head's matmul."""

    from tensorfold.families.qwen3_5.cuda import qmm_fast
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    n, k = 1000, 5120
    if dense:
        head = QLinear(torch.randn((n, k), device=DEV).bfloat16(), None, None, layout="dense", bits=0, gs=0)
    else:
        words, scales = _sym_weights(DEV, n, k, 128, 17)
        head = QLinear(words, scales, None, gs=128, sym=True)
    parts = [qmm_fast.rows(head, a, b) for a, b in SPANS[spans]]
    assert all(p.weight.data_ptr() == head.weight[a:].data_ptr() for p, (a, _) in zip(parts, SPANS[spans]))
    for m in (1, 5, 16):
        x = torch.randn((m, k), generator=torch.Generator(device=DEV).manual_seed(m), device=DEV).bfloat16()
        full = qmm_fast.matmul(x, head)
        want = torch.cat([full[:, a:b] for a, b in SPANS[spans]], dim=1)
        assert _same(qmm_fast.matmul_rows(x, parts), want), (spans, m)
        for p, (a, b) in zip(parts, SPANS[spans]):
            assert _same(qmm_fast.matmul(x, p), full[:, a:b].contiguous()), (spans, m, a, b)


@xpu_only
def test_xpu_refuses_row_parallel_partials(DEV):
    from tensorfold.families.qwen3_5.cuda import qmm_fast
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    words, scales = _sym_weights(DEV, 64, 512, 128, 1)
    with pytest.raises(ValueError, match="not supported on XPU"):
        qmm_fast.matmul_partial(torch.randn((1, 512), device=DEV).bfloat16(), QLinear(words, scales, None, gs=128,
                                                                                         sym=True))


@xpu_only
def test_xpu_head_rows_take_strided_rows_and_derive_the_group(DEV):
    from tensorfold.families.qwen3_5.cuda import qmm_fast
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    words, scales = _sym_weights(DEV, 1000, 5120, 128, 31)
    head = QLinear(words, scales, None, gs=128, sym=True)
    parts = [qmm_fast.rows(head, a, b) for a, b in SPANS["disjoint"]]
    wide = torch.randn((5, 5120 + 64), device=DEV).bfloat16()
    x = wide[:, 64:]                                     # rows strided by K + 64
    assert _same(qmm_fast.matmul_rows(x, parts), qmm_fast.matmul_rows(x.contiguous(), parts))
    # the symmetric lane matmul takes its group size from the scales when none is given
    assert _same(qmm.lane_matmul(x.contiguous(), words, scales, None), qmm.lane_matmul(x.contiguous(), words, scales,
                                                                                         None, gs=128))
