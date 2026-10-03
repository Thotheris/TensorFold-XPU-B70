"""The shared 4-bit lane matmul: rows never depend on the row count, and the 27B's Triton kernel's bits."""

import pytest
import torch

from tests.devices import device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

from tensorfold.cuda.kernels import qmm  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("qmm")
cuda_only = pytest.mark.cuda_only          # the nvcc extension; XPU runs the Triton lane matmul (test_qwen27_qmm.py)

ROWS = [1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 256, 384, 512, 700, 1024, 1500, 2048]


def _weights(dev, n: int, k: int, gs: int, seed: int):
    g = torch.Generator(device=dev).manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8), generator=g, device=dev, dtype=torch.int64)
    scales = (torch.rand((n, k // gs), generator=g, device=dev) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // gs), generator=g, device=dev) * 0.05).to(torch.bfloat16)
    return words.to(torch.int32), scales, biases


def _dequant(words, scales, biases, gs):
    n, k8 = words.shape
    w = words.to(torch.int64) & 0xFFFFFFFF
    q = ((w[:, :, None] >> (torch.arange(8, device=w.device) * 4)) & 0xF).reshape(n, k8 * 8).float()
    return q * scales.float().repeat_interleave(gs, 1) + biases.float().repeat_interleave(gs, 1)


@pytest.mark.parametrize("n,k,gs", [(48, 5120, 64), (1024, 5120, 64), (5120, 6144, 64), (200, 1024, 32),
                                    (1536, 2048, 32)])
@cuda_only
def test_pack_round_trip(DEV, n, k, gs):
    w = _weights(DEV, n, k, gs, n + k)
    back = qmm.unpack(qmm.pack(*w, gs))
    assert all(torch.equal(a, b) for a, b in zip(back, w))


@pytest.mark.parametrize("n,k,gs", [(48, 5120, 64), (1024, 5120, 64), (5120, 17408, 64), (200, 1024, 32),
                                    (1536, 2048, 32)])
@cuda_only
def test_rows_do_not_depend_on_row_count(DEV, n, k, gs):
    q = qmm.pack(*_weights(DEV, n, k, gs, 3 * n + k), gs)
    x = torch.randn((max(ROWS), k), generator=torch.Generator(device=DEV).manual_seed(7), device=DEV).bfloat16()
    alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(max(ROWS))])
    for m in ROWS:
        assert torch.equal(qmm.matmul(x[:m], q), alone[:m]), f"{n}x{k}: rows differ at M={m}"
    perm = torch.randperm(max(ROWS), generator=torch.Generator().manual_seed(3)).to(DEV)
    assert torch.equal(qmm.matmul(x[perm], q), alone[perm])
    f32 = qmm.matmul(x[:40], q, f32=True)
    assert torch.equal(torch.cat([qmm.matmul(x[r:r + 1], q, f32=True) for r in range(40)]), f32)


@pytest.mark.parametrize("n,k", [(48, 5120), (1024, 5120), (5120, 6144), (10240, 5120), (5120, 17408)])
@cuda_only
def test_27b_triton_bits(DEV, n, k):
    """Groups of 64 give the 27B's stored-layout Triton kernel's bits (its serial reference) at every row count."""

    from tensorfold.families.qwen3_5.cuda import qmm as triton_qmm

    w = _weights(DEV, n, k, 64, 5 * n + k)
    q = qmm.pack(*w, 64)
    x = torch.randn((128, k), device=DEV).bfloat16()
    for m in (1, 7, 16, 17, 32, 64, 100, 128):
        assert torch.equal(qmm.matmul(x[:m], q), triton_qmm.lane_matmul(x[:m], *w)), (n, k, m)


@pytest.mark.parametrize("n,k,gs", [(1024, 5120, 64), (512, 2048, 32)])
@cuda_only
def test_accuracy_matches_fp32_reference(DEV, n, k, gs):
    w = _weights(DEV, n, k, gs, 11 * n + k)
    x = torch.randn((16, k), device=DEV).bfloat16()
    y = qmm.matmul(x, qmm.pack(*w, gs)).float()
    ref = x.float() @ _dequant(*w, gs).T
    assert (y - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -7


def test_split_depends_only_on_shape():
    for n, k in [(48, 5120), (1024, 5120), (5120, 6144), (17408, 5120), (5120, 17408), (248320, 5120)]:
        sk = qmm.split_k(n, k)
        assert sk == qmm.split_k(n, k) and (k // 64) % sk == 0


@cuda_only
def test_strided_rows_buffers_and_unreduced_slices(DEV):
    n, k, gs = 5120, 17408, 64
    q = qmm.pack(*_weights(DEV, n, k, gs, 17), gs)
    wide = torch.randn((24, k + 64), device=DEV).bfloat16()
    x = wide[:, :k]                                   # rows 16-byte aligned, stride k + 64
    want = qmm.matmul(x.contiguous(), q)
    assert torch.equal(qmm.matmul(x, q), want)
    out = torch.empty_like(want)
    assert qmm.matmul(x, q, out=out) is out and torch.equal(out, want)
    sk = qmm.split_k(n, k)
    assert sk > 1
    slices = qmm.matmul(x, q, reduce=False, f32=True)
    assert slices.shape == (sk, 24, n)
    total = slices[0]
    for s in range(1, sk):
        total = total + slices[s]
    assert torch.equal(total, qmm.matmul(x, q, f32=True))


@pytest.mark.parametrize("n,k,gs", [(128, 5120, 64), (200, 1024, 32)])
@cuda_only
def test_every_nibble_decodes_exactly(DEV, n, k, gs):
    """One-hot rows read back each stored nibble (scale 1, bias 0): pair() and every K split are exact on any GPU."""

    words = _weights(DEV, n, k, gs, 29)[0]
    ones = torch.ones((n, k // gs), device=DEV).bfloat16()
    zeros = torch.zeros_like(ones)
    q = qmm.pack(words, ones, zeros, gs)
    want = _dequant(words, ones, zeros, gs).T.contiguous()                                  # (k, n): every q, 0-15
    eye = torch.eye(k, device=DEV).bfloat16()
    assert qmm.split_k(n, k, gs) > 1 and set(want.unique().tolist()) == set(range(16))
    for f32 in (False, True):
        assert torch.equal(qmm.matmul(eye, q, f32=f32).float(), want), f32
        assert torch.equal(qmm.prefill_matmul(eye, q, f32=f32).float(), want), f32
