# experts  (K3, kernel engineer, status: T0)

Op: Nemotron 3.5's grouped MoE experts: route (row, slot) pairs to experts, then each pair's up projection (relu²)
and down projection. CUDA: `cuda/experts.{py,cu,cuh}`, `experts_prefill.cu`, `experts_pack.cu`. Reference:
`families/nemotron_h/cuda/reference.py::moe`. XPU: `src/tensorfold/xpu/kernels/experts/` (torch plan, Triton kernels).

## Shapes and dtypes

128 routed experts + the shared expert folded as 2 halves = 130 "experts"; top-6 + 2 shared = 8 slots a token.
up: `[E, 1856, 2688]`, down: `[E, 2688, 1856]`, symmetric INT4 g64, fp16 scales. x bf16 `[R, 2688]`.
Decode windows R <= 16 (128 pairs); prompt chunks up to 4096 rows.

## K3a plan (torch, integer-exact)

Pair `p = row * slots + slot`, expert `picks[p]`. `members` = pairs sorted by expert with a **stable** sort, so pairs
of one expert keep pair order. `count[e]` = bincount, `first[e]` = exclusive cumsum. Items = `(e, first[e] + T j,
min(T, count[e] - T j))` for `j < ceil(count[e] / T)`, experts ascending; `T` = 16 (decode) or 64 (prompt). The
item table has a fixed size (`max_items`) and unused items carry count 0, so building it needs no host sync.

## K3b layout (`pack_xpu`)

The stored N-major SYM layout, stacked: words `[E, N, K/8]` int32 (low nibble first), scales `[E, N, K/gs]` fp16 as
stored. No CUDA fragment layout and no biases (`b = -8 s` in registers). Each expert's tensors are views of one
stack; no single allocation reaches 4 GB (up: 130 x 1856 x 336 x 4 B = 324 MB).

## Layouts in/out (enforced before any launch)

`decode` and `prompt` check metadata only (no GPU sync, item and member contents are not read) and raise `ValueError`
before allocating or launching:

| Operand | Contract | Why |
|---|---|---|
| `epi` | `EPI_FP32`, `EPI_RELU2` or `EPI_BF16` | any other value would store fp32 into a bf16 buffer |
| `x` | 2-D bf16, `stride(1) == 1`, any row stride; exactly `pairs / slots` rows (up, `from_tokens`) or `pairs` rows (down) | the kernel reads row `pair // slots` or `pair` |
| `words` | int32 `[E, N, K/8]`, **contiguous** | expert e, row n at word `(e*N + n)*K/8` |
| `scales` | fp16 or bf16 `[E, N, K/gs]`, **contiguous**; gs in {64, 128} | `(e*N + n)*K/gs + g` |
| plan | `members` contiguous `(pairs,)` int32, `items` contiguous `(., 3)` int32, `tile` = 16 (decode) / 64 (prompt), `experts == E` | `plan()` clamps every item's expert below `experts`, so equal counts keep weight reads in the stack |
| `out` (optional) | exactly a contiguous `(pairs, N)` tensor, fp32 for `EPI_FP32`, else bf16 | stores are unmasked past the logical rows: pair p, column n at `p*N + n` |
| devices | `x`, words, scales, `members`, `items`, `out` on one device | one launch device |

Plans are built by `plan()` (members are a permutation of the pairs; item counts fit the tile); a hand-built plan's
contents are not checked. An exact-size `out` may be a contiguous slice of a larger buffer: the B70 test writes into
`big[1:-1]` and checks the guard rows on either side are untouched.

## K3c decode (Triton grouped GEMV), contract

Program (item, column tile). Rows of the dot are the item's pairs (padded to 16), so a pair's bits do not depend on the
other pairs, the item's fill or the window. Per group g of 64 inputs, ascending:

```
P   = dot(x_pair[64g : 64g+64], q[n, 64g : 64g+64])         # DPAS, fp32
xs  = in-kernel fp32 sum of the pair's 64 inputs (same tile shape and layout in every launch)
acc = fma(xs, -8 s, fma(P, s, acc))
```

Epilogues: up `out = bf16(relu(bf16(acc))^2)` (CUDA's `relu2`); down `out = acc` (fp32, per pair). Combining a token's
slots (routing weights, shared expert) stays with the caller, in slot order.

## K3d prompt (Triton grouped GEMM), contract

Program (item of 64 pairs, column tile): `w = bf16(fma(q, s, -8 s))`, `acc = dot(x, w^T, acc)` over K in 64-column
steps ascending (the K5 contract, `prompt_gemm.md`). Epilogues: up relu² as above, down bf16 (CUDA EPI 3).
Prompt bits are not decode bits; a pair's prompt bits do not depend on its chunk.

## Invariances required

pair alone == pair in any window (decode) and any chunk (prompt) | stable ties: pairs of one expert in pair order |
empty experts, single-pair experts, tails | 20 repeats | fp64 tolerance against `reference.moe`-style per-pair math.

## Tests

`tests/cuda/test_xpu_experts.py` (`xpu_kernel("experts")`). `tests/cuda/test_experts.py` stays CUDA-only (packed
fragment layout and the nvcc extension). Layouts: `tests/xpu/test_kernel_layouts.py` (host refusals before a patched
launch) and `tests/cuda/test_xpu_layouts.py` (`xpu_kernel("experts")`: offset and every-other-row x views and an
exact-size `out` inside guard rows give the contiguous bits, decode and prompt, up and down, g64 and g128, N off the
column tile, 1/4/16/17 tokens including whole 16-pair items and a tail).

## Measurements

T0 qualified on `3a28fb4`, bundle `runs/xpu--main/3a28fb4-20261003T101551Z` (toolchain hash 858a0a59). Per-launch means
of 5 launches queued back to back; every case `bitwise_ok` (20 repeats plus the in-bench chunk / alone check); 0 spill
bytes. % of 608 GB/s or 183 TFLOPS on the stated bytes / flops models. `kernels:experts` 7 tests passed. 16 lanes, DPAS
in both kernels, `n_regs` 256 (driver). 130 experts, 8 slots; bytes are the weights of the experts used (once) plus
inputs.

| Case | us | GB/s | % of 608 | TFLOPS |
|---|---|---|---|---|
| decode up, 16 tokens (relu^2) | 510.2 | 348.2 | 57.3 | 2.50 |
| decode down, 16 tokens (fp32) | 551.6 | 322.8 | 53.1 | 2.32 |
| prompt up, 1024 tokens | 5325.0 | 65.7 | 10.8 | 15.35 (8.4% of 183) |
| prompt down, 1024 tokens (bf16) | 6202.2 | 60.5 | 9.9 | 13.18 (7.2% of 183) |
