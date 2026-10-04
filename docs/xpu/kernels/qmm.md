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

### T1 (qualified on `44a106d`, bundle `runs/xpu--main/44a106d-20261003T071529Z`, toolchain hash 858a0a59)

`kernels:qmm` 45 passed; `unit-xpu` 51 passed (both include the recipe-shape and head invariance sweeps); every case
`bitwise_ok`, 20 repeats, row alone == window on fp32 sums; 16 lanes, DPAS in every TTGIR, 0 spill bytes everywhere
(T0: 5120-9472). `n_regs`: 256 from the driver (automatic large-GRF build) for down and Nemotron in_proj, else 128
from `grf_mode` (a budget, not measured usage).

Columns: `call` = per-call mean of 20 calls queued back to back, weights cold (cycled over copies larger than the
LLC); `device` = the same 20 calls queued behind a long matmul, so it has no host gaps; `host` = host submit time per
call. % is of 608 GB/s on the bytes model (weights + scales + x + out).

| Weight (N x K) | T1 config | M | call us | device us | device % | call % | host us | launches |
|---|---|---|---|---|---|---|---|---|
| A down 5120 x 17408 g128 | bm16 bn32 1w ks4, 4 slices | 1 / 16 | 120.4 / 139.9 | 115.0 / 136.8 | 65.8 / 56.1 | 62.9 / 54.9 | 127 / 117 | 2 |
| A gate/up 17408 x 5120 g128 | bm16 bn16 1w ks4, 4 slices | 1 / 16 | 137.7 / 212.2 | 135.7 / 205.0 | 55.8 / 37.5 | 54.9 / 36.2 | 117 / 116 | 2 |
| A qkv 10240 x 5120 g128 | bm16 bn16 1w ks8, 2 slices | 1 / 16 | 120.9 / 120.4 | 82.5 / 115.3 | 53.9 / 39.3 | 36.8 / 37.6 | 114 / 116 | 2 |
| A head 248320 x 5120 bf16 | bm16 bn32 2w ns2 | 1 / 16 | 4959 / 5115 | 4977 / 5112 | 84.0 / 82.1 | 84.3 / 82.0 | 77 / 76 | 1 |
| A in_proj_a/b 48 x 5120 bf16 | bf16 default, 8 slices | 1 / 16 | 122.2 / 129.5 | 45.1 / 47.4 | 1.8 / 2.3 | 0.7 / 0.8 | 116 / 125 | 2 |
| B in_proj 10304 x 2688 g64 | bm16 bn32 1w ks2, 2 slices | 1 / 16 | 130.9 / 131.0 | 50.2 / 57.3 | 48.3 / 43.4 | 18.5 / 19.0 | 126 / 116 | 2 |
| B out_proj 2688 x 4096 g64 | bm16 bn32 1w ks4, 4 slices | 1 / 16 | 122.9 / 123.8 | 58.6 / 104.5 | 16.5 / 9.5 | 7.8 / 8.1 | 118 / 119 | 2 |
| B head 131072 x 2688 bf16 | bm16 bn32 2w ns2 | 1 / 16 | 1351 / 1386 | 1364 / 1391 | 85.0 / 83.9 | 85.8 / 84.1 | 77 / 79 | 1 |

All six windows (M = 1, 2, 4, 8, 12, 16) are in the bundle's `kernels/qmm-*.json`; time per verified row at M=16 is
the call time / 16 (e.g. A down 8.7 us, A head 320 us).

**The ~120 us floor is host submission, not the GPU** (same bundle, `host_breakdown` for B out_proj at M=1): the
`sym_matmul` wrapper takes 98.5 us of host time, of which Triton's JIT dispatch of the main kernel is 32.3 us (the
driver launch inside it, `CompiledKernel.run`, is 11.4 us), the split-K reduce is a second such launch, `group_sums`
alone is 41.9 us, and an allocation 2.1 us. Every split-K call is two launches, so calls whose device time is under
~115 us run at the host's pace (Nemotron in_proj/out_proj, in_proj_a/b, A qkv at small M). `triton-smoke`'s
`launch_latency` on the same bundle: a trivial Triton add submits in 38.5 us vs 18.1 us for `torch.add`.
Removing launches (no separate reduce, shared group sums, cached launches or graphs) is WS6 work, not done here.

## Open issues

- T1 pass done (bounded: one sweep per weight; stop here). Device bandwidth: A down 56-66%, gate/up 37-56%,
  qkv 39-54%, bf16 heads 82-86%; the 80% N1 goal stays with K4.N.
- Host submission (two Triton launches, about 115 us) bounds every split-K call; see the floor note above.
- gate/up loses bandwidth as M grows (56% at M=1, 37% at M=16): BM=16 tiles re-read x per column tile. Not tuned
  further (bounded pass).
- `qmm_fast.rows` and `matmul_rows` assume tiled CUDA weights; add stored-SYM/BF16 row-selection adapters before WS5
  (STEP 4), with contiguous/disjoint/boundary-span equality against full-head slices. Reject `matmul_partial` on XPU.
- Closed (2026-10-03, dev A/B on the B70, 1024 x 5120 g128, 16 rows): the cancellation in `P*s - 8*s*xs` is real but
  negligible. Worst fp32 relative error against fp64: 3.2e-6 for nibbles clustered at 7-9 with activation mean 1-4,
  against 2.3e-7 for `dot(x, q - 8) * s`; both are about 1000x below the bf16 output rounding (2^-9). The contract
  stays as is (CUDA's and the native plan's).
- Launch overhead, part fixed: `xpu/kernels/launch.py::Launcher` calls the compiled kernel's `run` after the first
  JIT dispatch (the key covers the function, dtypes, 16-byte alignment and int value / divisibility / i32-i64-u64
  width). An entry holds its function, so a replaced or temporary function never reuses another's compiled kernel;
  cached launches pass this call's grid and args on the device's current stream (`tests/xpu/test_launch.py`,
  `tests/cuda/test_xpu_launch.py`) and return the compiled kernel, which `kernels:qmm` reads lanes, spills and DPAS
  from on every case (`counter_gaps` names any counter the driver did not report). A split-K call's host
  time went from about 97 us to 71 us (dev). The rest is the driver's launch (~16 us each), two allocations and the
  wrapper's checks; fewer launches (fused reduce, graphs) is WS6.
