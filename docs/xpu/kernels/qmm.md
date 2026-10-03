# qmm  (K4, kernel engineer, status: T1)

Op: decode matmul `y = x @ W.T` for 4-bit symmetric weights (`lane_matmul`), plus a bf16-weight GEMV for the BF16 `lm_head`
and `linear_attn.in_proj_a/b`.
CUDA source: `cuda/kernels/qmm.cu::qmm_kernel` (not available on XPU). Triton reference: `families/qwen3_5/cuda/qmm.py::_qmm`
(affine, g64, bf16 scales and biases; **unchanged**).
Python wrapper: `families/qwen3_5/cuda/qmm.py::lane_matmul` (dispatches to the XPU kernels when `biases is None`);
kernels in `src/tensorfold/xpu/kernels/qmm/`.

## Shapes and dtypes

| Recipe | Weight (N x K) | Format | Role |
|---|---|---|---|
| A | 5120 x 17408 | g128 SYM, fp16 scales | `down_proj` |
| A | 17408 x 5120 | g128 SYM, fp16 scales | `gate_proj` / `up_proj` |
| A | 10240 x 5120 | g128 SYM, fp16 scales | GDN `in_proj_qkv` |
| A | 248320 x 5120 | bf16 | `lm_head` (BF16 GEMV) |
| A | 48 x 5120 | bf16 | `linear_attn.in_proj_a/b` (BF16 GEMV) |
| B | 10304 x 2688 | g64 SYM, fp16 scales | Mamba `in_proj` |
| B | 2688 x 4096 | g64 SYM, fp16 scales | attention / Mamba `out_proj` |
| B | 131072 x 2688 | bf16 | `lm_head` (BF16 GEMV) |

`x` is bf16 `(M, K)` (rows may be strided, `stride(1) == 1`), output bf16 `(M, N)`. M is the window size (1..~129).
Scales are fp16 or bf16 and stay as stored. Words are the N-major layout `(N, K/8)` int32, low nibble first, which is
what the WS3b loader writes (GPTQ `[K/8, N]` transposed). `q in 0..15`, `z = 8` always (from the `sym` flag).

## Arithmetic contract (SYM)

Per output element `(m, n)`, with `G = K / GS` groups, `SK` K slices of `PER = G / SK` consecutive groups:

```
P_g   = dot(x[m, g*GS : (g+1)*GS], q[n, g*GS : (g+1)*GS])        # fp32, tl.dot on bf16 operands (q exact in bf16)
s     = fp32(scale[n, g])                                         # exact (fp16 or bf16 -> fp32)
b     = -8 * s                                                    # fp32, exact, never stored
xs_g  = xs[m, g]                       (GS = 64)                  # fp32 group sum of the 64 inputs (canonical producer)
xs_g  = xs[m, 2g] + xs[m, 2g+1]        (GS = 128)                 # one fp32 add, fixed order
acc   = fma(P_g, s, acc)               then   acc = fma(xs_g, b, acc)      # acc starts at +0, groups ascending
part[slice] = acc ;  y = bf16_rne( ((part[0] + part[1]) + part[2]) + ... )   # ascending slice order (SK == 1: acc itself)
```

- This is `w = s*q + b` with `b = -8*s`, the same algebra and FMA points as the CUDA contract
  `fma(xs, b, fma(P, s, acc))`. XPU bits need not equal CUDA bits (`P_g` is a DPAS accumulation).
- `enable_fp_fusion=False` on the launch; the two FMAs are explicit `tl.fma`.
- The integer to bf16 cast of `q` goes through fp32 (int to bf16 crash on triton-xpu).
- `xs` has one canonical granularity: 64 inputs, fp32. `lane_matmul` computes it with `group_sums` when not given, and the
  fused producers (`_add_rmsnorm`, `_gated_norm`, `_swiglu`, `_gate_mul`) emit the same 64-wide sums. g128 weights fold
  adjacent pairs in the kernel, so no producer needs a second definition.
- Considered, not chosen: `dot(x, q - 8) * s` (signed nibbles are exact in bf16) avoids `xs` and the `P*s - 8*s*xs`
  cancellation. It is a different contract from the CUDA/native one; revisit as an A/B at N0.

### BF16 GEMV

```
acc_slice = 0 ; for c in slice chunks (BK = 64 columns, ascending): acc_slice = dot(x_chunk, w_chunk^T, acc_slice)   # fp32
y = bf16_rne( ((acc_0 + acc_1) + ...) )                      # ascending slices; SK == 1: acc itself
```

## Row, tile and split rules

- Every launch constant comes from `lane_config(n, k, gs)` and `slices(n, k, gs)` (`xpu/kernels/qmm/config.py`):
  the `XPU_CONFIG` entry for the recipe shapes, else the default for the weight's kind. BM, BN, `num_warps`,
  `num_stages`, `grf_mode`, `ksplit` and the K slices are functions of the weight shape only; none is tuned at run time.
- `ksplit` splits a group's dot into chained sub-dots, `p = dot(x_h, q_h, p)` in ascending K. Each sub-dot is whole
  DPAS K steps, and on the B70 the chain gives the same bits as one dot over the group (T1 sweep: every `ksplit` at
  the T0 split was bit-equal to T0 on fp32 outputs).
- Changing the K slices changes the bits by design (different partial sums); the table's slices are part of the
  contract, and the tests run every shape at 1, `split_k` and the production slices.
- Row invariance means a row's bits do not depend on M, its position in the window, the other rows, or the grid padding.
  BM is forced to {16, 32, 64, 128} in the tests to prove the dot does not change the bits per row.

## Invariances required

alone == in window (M in {1, 2, 15, 16, 17, 33, 65, 129}, any position) | BM-invariant | split-K slices fixed by shape |
20 repeats, warm and cold cache | xs from `group_sums` == xs from each fused producer.

## References and tests that pin bits

- fp64 reference: `x.double() @ (s * (q - 8)).double().T` (tolerance 2^-7 of the max).
- Exactness of the dequant: one-hot rows of a synthetic GPTQ-layout weight read back `bf16(s * (q - 8))` bit for bit.
- `tests/cuda/test_qwen27_qmm.py` (`xpu_kernel("qmm")`): row invariance across M x BM x SK, SYM dequant, tolerance,
  cross-producer xs, bf16 GEMV invariance.
- Suite `kernels:qmm` (`tools/xpu/bench_qmm.py`): kbench at the shapes above, M = 1.

## Roofline target on B70

Decode is bandwidth-bound: `bytes = N*K/2 + scales + x`. Peak 608 GB/s. T0 goal: correct. N1 goal (K4.N): >= 80% of
608 GB/s at M = 1 on 5120 x 17408 (`PORT_PLAN.md` K4). Recipe traffic must be computed from loaded
weights/scales, the 2.543 GB BF16 head, recurrent state, KV and workspace; checkpoint size alone is not
a tokens/s model.

## Measurements

Bundle `runs/xpu--main/da2a34f-20261003T054251Z` (torch 2.14.1+xpu, triton 3.8.0, toolchain hash 858a0a59). All rows
`bitwise_ok`, 20 repeats; 16 lanes per warp. `n_regs` from the driver unless marked (g = GRF budget implied by `grf_mode`).
`pct` is of 608 GB/s; DPAS is present in the 4-bit kernels' TTGIR.

| Shape (N x K) | Role | M=1 us | M=1 GB/s | pct | M=16 pct | n_regs | n_spills (B) |
|---|---|---|---|---|---|---|---|
| 5120 x 17408 | A down | 1302 | 35.3 | 5.8 | 5.8 | 256 | 9472 |
| 17408 x 5120 | A gate/up | 775 | 59.3 | 9.8 | 9.6 | 256 | 5120 |
| 10240 x 5120 | A in_proj_qkv | 764 | 35.4 | 5.8 | 6.0 | 256 | 9472 |
| 248320 x 5120 bf16 | A lm_head | 7140 | 356.2 | 58.6 | 57.6 | 128 (g) | 0 |
| 48 x 5120 bf16 | A in_proj_a/b | 117 | 4.3 | 0.7 | 0.9 | 128 (g) | 0 |
| 10304 x 2688 | B in_proj | 125 | 118.2 | 19.4 | 20.0 | 256 | 0 |
| 2688 x 4096 | B out_proj | 125 | 46.9 | 7.7 | 7.9 | 256 | 0 |
| 131072 x 2688 bf16 | B lm_head | 2023 | 348.4 | 57.3 | 56.6 | 128 (g) | 0 |

Correctness on the same bundle (`unit-xpu` 37 passed, `kernels:qmm` 31 passed, 0 skipped): rows equal across
M x BM x split-K for the SYM and bf16 kernels, SYM dequant exact, fp64 tolerance, xs from every producer equals
`_group_sums`.

Timings are per-launch means of 20 queued launches through the Python wrapper (sym: plus the split-K reduce launch).
Every small shape sits on a 117-125 us floor (in_proj_a/b moves 0.5 MB in 117 us; Nemotron in_proj and out_proj take
the same time for 14.7 and 5.5 MB). This suggests fixed wrapper/submission/launch costs, but does not identify
the cause. `triton-smoke`'s `launch_latency` probe splits host submit and device time; its current-head bundle
must be inspected before assigning that floor to GPU launch latency. The A shapes above the floor (down
1302 us, gate/up 775 us, qkv 764 us) are genuinely slow: 6-10% of peak, spilling 5-9 KB at 256 GRF. M=1 and M=16
cost the same because the dot pads M to BM = 32. T1 (`num_warps`, `grf_mode`, BN, split-K per shape) is the next step.

### T1 configuration (sweeps on the B70, dev runs; the bundle numbers replace these when the run lands)

Method: `kbench`-style back-to-back launches, weights cycled over copies larger than the 24 MB LLC, every config checked
bit-equal on fp32 outputs against the T0 config at the same slices. Small single-warp programs (1 warp of 16 lanes,
BM = BN = 16 or 32) remove the spills; the chained sub-dots cut the live `q` tile.

| Weight | T1 config | M=1 % of 608 GB/s (T0) |
|---|---|---|
| A down 5120 x 17408 g128 | bm16 bn32 1 warp ksplit 4, 4 slices | 62.5 (5.8) |
| A gate/up 17408 x 5120 g128 | bm16 bn16 1 warp ksplit 4, 4 slices (T0: 1) | 53.7 (9.8) |
| A qkv 10240 x 5120 g128 | bm16 bn16 1 warp ksplit 8, 2 slices | 36.8 (5.8) |
| B in_proj 10304 x 2688 g64 | bm16 bn32 1 warp ksplit 2, 2 slices | 19.5 (19.4), launch-bound |
| B out_proj 2688 x 4096 g64 | bm16 bn32 1 warp ksplit 4, 4 slices | 7.9 (7.7), launch-bound |
| bf16 heads (default for bf16) | bm16 bn32 2 warps 2 stages | 82-84 at M=1 and M=16 (57-59) |

The bf16 optimum at M=1 alone (4 warps, 86-90%) fell to 37% at M=16, so the bf16 default is scored on M=1 + M=16.

## Open issues

- Resolved on da2a34f: the BM sweep (16..128) is bitwise row-invariant on this shape, and xs from `_group_sums`,
  `_add_rmsnorm` and `_swiglu` is equal.
- 4-bit decode is far from the roofline (above); the A shapes spill 5-9 KB. Run one bounded, shape-only T1
  pass after current-head qualification, then return to missing correctness kernels. Test M=1/2/4/8/12/16
  and retain the full invariance sweep; do not runtime-dispatch FMA vs DPAS by M.
- `55d2095` adds recipe-shape coverage and `f569957` adds timing probes. Neither has a bundle in the index
  inspected for this documentation revision; the `da2a34f` pass does not qualify those descendant changes.
- The `da2a34f` summary's glue/prompt-attention "Bitwise break" entries are absent artifacts from omitted
  suites. They are missing coverage, not measured mismatches; combined current-head qualification is pending.
- A ~120 us per-launch floor would dominate decode (several launches per layer); source pending `launch_latency`.
- `qmm_fast.rows` and `matmul_rows` assume tiled CUDA weights; add stored-SYM/BF16 row-selection adapters
  before WS5, with contiguous/disjoint/boundary-span equality against full-head slices. Retain the parent
  arithmetic plan when row selection changes output width. Reject unsupported single-GPU `matmul_partial`
  explicitly; do not enter a CUDA extension.
- The SYM contract computes `P*s - 8*s*xs`, which cancels when `q` sits near 8; the fp64 tolerance (2^-7 of the max)
  passes, but the A/B against `dot(x, q - 8) * s` noted above is still open.
