# Triton on Intel XPU (Arc Pro B70) Developer Guide

Audience: agents porting or writing TensorFold Triton kernels for the Intel XPU backend (`intel-xpu-backend-for-triton`,
shipped as `triton-xpu` with PyTorch XPU). Companion docs: [B70_NATIVE_KERNEL_GUIDE.md](B70_NATIVE_KERNEL_GUIDE.md) for
hardware and SYCL, and [KERNEL_MAP.md](KERNEL_MAP.md) for which kernels each recipe runs.

Compiled 2026-09-30. `[UNVERIFIED]` marks claims that could not be confirmed. Sources are keyed `[Sn]` (table in §7).

---

## 0. The rules

1. **Pin torch 2.14.1+xpu with triton-xpu 3.8.x** (runtime 2026.1). G31 (B70) support arrived in triton-xpu 3.7.1.
2. **Never have oneAPI `setvars.sh` or `icpx` on PATH when running Triton.** It SIGSEGVs from a SYCL runtime mismatch
   [S24]. Native extension builds happen in a separate shell.
3. **Warp size is chosen per kernel.** A kernel with a DPAS-lowerable `tl.dot` compiles at **16** threads per warp;
   other kernels use 32 [S9, S12]. `num_warps=4` is therefore 64 threads in one kernel and 128 in another. Reduction
   trees follow from this. Record the compiled `threads_per_warp` per kernel.
4. **Set `enable_fp_fusion=False` on every bit-sensitive kernel.** Otherwise `a + b*c` may become an FMA.
5. **Assert that DPAS was used** (`#triton_intel_gpu.dpas` in the TTGIR) for every `tl.dot` kernel. DPAS needs N ≥ 16.
   Smaller shapes **silently fall back to FMA** [S17], which is slower and has a different accumulation order.
6. **Check `n_spills`/`n_regs`** on the compiled kernel. For big accumulators use `grf_mode="256"`, or more warps.
7. **Purge `~/.triton/cache` when you change Intel env knobs or the toolchain.** Knobs are not part of the cache key
   [S40].
8. **XPU bits ≠ CUDA bits.** Accept that, and re-baseline goldens on XPU. The contract is self-consistency (row alone
   == in window, chunked == one-shot, drafted == serial), not CUDA parity.

---

## 1. Backend facts

### 1.1 Versions

| torch (XPU wheel) | Triton | Intel runtime |
|---|---|---|
| 2.8.0 | pytorch-triton-xpu 3.4.0 | 2025.1 |
| 2.9.1 | pytorch-triton-xpu 3.5.0 | 2025.2.1 |
| 2.10.0 | triton-xpu 3.6.0 | 2025.3 |
| 2.11.0 | triton-xpu 3.7.0 | 2025.3.2 |
| 2.13.0 | triton-xpu 3.7.2 | 2026.0 |
| **2.14.1** | **triton-xpu ~=3.8.0** | **2026.1**, intel-pti 1.0.1 |

Sources: wheel metadata [S4], index [S3, S5]. Release notes [S2]:
- **3.7.1:** G31 support, block-scale DPAS, FP8 conversions, descriptor prefetch.
- **3.7.2:** block pointers lowered to tensor descriptors in the frontend; register-pressure heuristic.
- **3.8.0:** CodeSinking, fast sin/cos (fast-math only), sigmoid SPIR-V builtin, **default predicated loads**; needs DLE
  2026.1.

Supported: Arc B-series and Arc Pro B-series. Not compatible with IPEX or the full oneAPI Base Toolkit in the same
environment [S1].

### 1.2 Options (`XPUOptions`) [S9]
Defaults: `num_warps=4`, `num_ctas=1` (**must stay 1** [S28]), `num_stages=2`, `warp_size=32`, `grf_mode='default'`,
`enable_fp_fusion=True`, `default_dot_input_precision="tf32"`, `allow_fp8e4nv=True`.

- **`grf_mode`:**
  - `'default'` compiles at 128 GRF and automatically rebuilds at large GRF once spill reaches 1 KB per thread (PRs
    #8147/#8217, gated on `num_warps ≤ 32`).
  - `'auto'` lets IGC decide. `'128'` and `'256'` are explicit. `'512'` is PVC only.
  - 256 GRF halves the maximum work-group size.
  - Pass it at launch: `kernel[grid](..., grf_mode="256")`. `maxnreg` must not be below an explicit `grf_mode`.
- **Work-group limit:** 1024 threads. That is `num_warps ≤ 32` at 32 lanes and `≤ 64` at 16 lanes [S13].
- **`num_stages`:** software-pipelining depth for dot loops (modulo scheduling plus prefetch). Loops without a dot gain
  little [inferred].
- **Register budget rule of thumb [derived]:** per-lane fp32 values = `M·D / (num_warps · threads_per_warp)`. Keep it
  ≤ about 100 in 128-GRF mode; otherwise raise `num_warps` or set `grf_mode="256"`.
- `TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32` exists but is experimental; don't use it.

### 1.3 `tl.dot` → DPAS
- The DPAS tile is 8×16×16 for bf16/fp16 → fp32 (`repeatCount=8, systolicDepth=8, executionSize=16, opsPerChan=2`).
- **Minimum N = 16.** M and K minimums were relaxed to 1 for non-int8. Rejected shapes fall back to FMA silently
  [S17].
- fp32 inputs default to **tf32**. Pass `input_precision="ieee"` or cast to bf16 first. TF32 DPAS on Xe2 is
  `[UNVERIFIED]`.
- Write `acc = tl.dot(a, b, acc)` (explicit accumulator) [S1].
- **FP8:**
  - There is no hardware FP8. Conversions are software sequences, and fp32→fp8 is RTNE only [S29]. Saturation vs NaN on
    overflow is `[UNVERIFIED]`.
  - `tl.float8e4nv` compiles on XPU.
  - TensorFold's FP8 prefill path (`--prefill-fp8`) is **refused on XPU**.
- **Operand loading:**
  - **Tensor descriptors** (`tl.make_tensor_descriptor`; 2D, last stride 1, built *inside* the kernel [S1, #8040])
    lower to 2D block loads and give **> 2× speedup** over tensors of pointers for dot operands.
  - `tl.make_block_ptr` is acceptable; it lowers to descriptors as of 3.7.2.
  - Don't mix the two APIs on one load.
  - Atomics need plain pointers.

### 1.4 Feature support

| Feature | Status | Note |
|---|---|---|
| fp64 arithmetic | `has_fp64=True`; **rate on B70 `[UNVERIFIED]`** | treat as slow until measured |
| uint64 / int64 | works in practice | verify the uint64 hash against numpy |
| `tl.histogram` | works via SLM atomics (slow for small inputs) | [S26]; GLM-only in TensorFold |
| `tl.cumsum` | works via a shuffle chain | [S27] |
| `tl.debug_barrier` as a global-memory fence | **`[UNVERIFIED]`** | don't rely on it; double-buffer instead |
| `tl.static_range` | works, but large unrolls can abort IGC | [S23]; use runtime `range` above about 64 |
| `while` loops (`scf.while`) | open RemoveLayoutConversions bug (#8189) | rewrite as a bounded `for` with predicates |
| `tl.reshape/trans/join/split` | works; some layout bugs fixed | 3D reshape + reduce causes heavy SLM conversions |
| int64 → `tl.pointer_type` load | **`[UNVERIFIED]`** | see §2.1; day-one probe |
| `enable_fp_fusion=False` | supported | |
| `tl.exp` | approximate `exp2` path | expect last-ulp differences from CUDA |
| `tl.sigmoid` | may use a SPIR-V builtin in 3.8 (accuracy `[UNVERIFIED]`) | test against torch |
| div / sqrt | accuracy depends on driver flags; `tl.math.div_rn`/`sqrt_rn` for exactness | [S29] |
| atomics | 32-bit fine; sub-word emulated via CAS | [S42] |
| i1 `tl.sum` | **wrong** (XOR vs OR) | cast to int32 first [S27] |

### 1.5 Known bugs that affect TensorFold

| Bug | Impact here | Mitigation |
|---|---|---|
| **Recurrence kernel with persistent `[BV,BK]` state → `DEVICE_LOST` on B70** [S22, #6658] | GDN tree/replay/chain (K1.T0), Nemotron `mamba._scan`/`_conv`, Mamba prompt scan (K2.T0) | Smaller state tiles, `grf_mode="256"`, hoist state stores out of the loop, sync after each launch in tests. **A native SYCL fallback is developed in parallel** |
| **`BLOCK_M=16` `tl.dot` miscompile** (zeroed output with `NPID_FACTOR>1`) [S20, #8121] | `qwen3_5/cuda/qmm.py::_qmm` BM=16, `affine_kernels.matmul` | Test every BM. If it reproduces, set a BM floor of 32 |
| constexpr-stride miscompile [S21] | any kernel with `tl.constexpr` strides | pass strides as runtime `tl.int64` |
| `scf.while` crash (#8189) | `cuda/kernels/attention.py::_paths` | bounded `for` rewrite (§3) |
| IGC abort on long `static_range` [S23] | `nemotron_h/cuda/sampler.py::_keyed` (K up to 256) | runtime `range` |
| int→bf16 and fp16→bf16 cast crash [S19] | `qmm.py:68`, `affine_kernels.py:27` | `.to(tl.float32).to(tl.bfloat16)` |
| 3.8 predicated-load regression (19–26% slower) [S25] | masked-load-heavy kernels | A/B `TRITON_INTEL_PREDICATED_LOAD=0` with a cache purge |
| host-side `TensorDescriptor` not lowered to block IO (#8040) | future descriptor rewrites | build descriptors in-kernel |
| knobs not in the cache key [S40] | all A/B runs | purge the cache |

### 1.6 Env vars, dumps and inspection
- Intel knobs [S11]: `TRITON_INTEL_ENABLE_IGC_SHADER_DUMP`, `TRITON_XPU_GEN_NATIVE_CODE`, `TRITON_INTEL_FAST_MATH`,
  `TRITON_INTEL_DISABLE_IGC_OPT`, `TRITON_INTEL_PREDICATED_LOAD`, `TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS`,
  `TRITON_INTEL_DEVICE_ARCH`, `TRITON_XPU_PROFILE=1`. `TRITON_INTEL_ADVANCED_PATH` is legacy or removed.
- Generic knobs: `TRITON_CACHE_DIR` (default `~/.triton/cache`), `TRITON_DUMP_DIR`, `TRITON_KERNEL_DUMP`,
  `TRITON_ALWAYS_COMPILE`, `MLIR_ENABLE_DUMP`, `TRITON_INTERPRET`.
- Compile stages: `ttir → ttgir → llir → spv (→ zebin)`.
  - **DPAS check:** grep the TTGIR for `#triton_intel_gpu.dpas` (also `#ttig.`). If it's absent, the dot fell back to
    FMA.
  - **Spills:** `k = kernel[grid](...)`, then `k.n_spills` and `k.n_regs`.
    - **`n_spills`** is Level Zero's `spillMemSize` in bytes, unchanged by the driver (triton-xpu 3.8.0 `driver.c`,
      verified on the B70). The PR #7976 unit change does not apply to the driver's `load_binary`.
    - **`n_regs` is not a measurement.** It is 128 or 256 only when a GRF flag was on the build line (`grf_mode="128"`,
      `"256"`, or the automatic large-GRF rebuild once spills pass 1000), and 0 for `default` and `auto`. Verified
      2026-10-03 on the B70 with `TRITON_XPU_GEN_NATIVE_CODE=1`: `default` and `auto` (for a trivial kernel) both
      build with `grf_count: 128` in the zebin's `.ze_info`.
- Harness helper: `tools/xpu/kbench.py` records `threads_per_warp`, `n_regs`, `n_spills` and DPAS-present for every
  Triton kernel it times. `n_regs` is the GRF budget per thread: the driver's value, else the zebin's `grf_count`, else
  128 for `default` and the explicit mode otherwise; `n_regs_source` says which, and `auto` without a zebin stays null.

### 1.7 Determinism
- Within a thread, reductions are a left fold (preserved by PR #8098). Across warps they go through SLM in a batched
  loop. The intra-subgroup order is IGC-defined [S28].
- **[inferred]** A kernel with no atomics, fixed constexprs, fixed `num_warps` and a pinned toolchain is
  bitwise-deterministic across launches and grid sizes.
- **Lane count changes the reduction tree.** A dot kernel (16 lanes) and a non-dot kernel (32 lanes) computing the same
  logical sum can differ in the last bit. TensorFold's `xs` group sums come from several producers: `_group_sums`,
  `_add_rmsnorm`, `_swiglu`, `_gated_norm`, `_gate_mul` and `_merge`. **Add cross-producer equality tests**, or make
  every consumer use a single producer.
- Torch eager is **not** deterministic by default (oneDNN split-K) [S34]. Set
  `torch.use_deterministic_algorithms(True)`.

### 1.8 Benchmarking
- Time with `torch.xpu.Event(enable_timing=True)` plus `torch.xpu.synchronize()`, after warm-up (the first call
  JIT-compiles).
- Intel's harness offers `ELAPSED_TIME`, `UPSTREAM_PYTORCH_PROFILER` (the default) and `PROTON_PROFILER`, and flushes a
  256 MB buffer between calls [S38].
- Don't enable both CUDA and XPU activities in `torch.profiler`; XPU profiling is ignored if you do.

---

## 2. TensorFold Triton audit (recipes A and B)

Risk levels:
- **H**: likely to break, or silently wrong.
- **M**: needs a test, or is slow.
- **L**: fine.

Paths are under `src/tensorfold/`.

### 2.1 Cross-cutting issues
1. **Pointer tables (H).**
   - `qwen3_5/cuda/draft_attention.py:30-31,107-108` loads int64 pointers from a table and casts them with
     `.to(tl.pointer_type(tl.bfloat16))`.
   - The host builds the tables with `torch.tensor([...data_ptr()...], dtype=int64)` (`:84,:134`). Level Zero addresses
     may be **≥ 2^63**, which raises `OverflowError` [S35]. Convert with
     `def _s64(p): return p - (1 << 64) if p >= (1 << 63) else p`.
   - Whether Triton's launcher sets Level Zero indirect access (residency of allocations not passed as arguments) is
     `[UNVERIFIED]` [S45].
   - Fallback: pass the K/V base pointers as normal arguments for a bounded number of streams, or keep one cache with
     offsets.
   - `cuda/kernels/attention.py` base-plus-offset addressing (`:68-69,:226-251`) needs the same residency probe.
2. **`num_warps` is part of the arithmetic** wherever an fp32 sum spans lanes: `_add_rmsnorm`, the `_gdn_pre` norm,
   `_scan` (8 warps, BD=32), the `_tile` max/sum, and `_group_rmsnorm`. Keep the CUDA constants, re-baseline the
   goldens, and record the compiled lanes.
3. **fp contraction:** `acc + p*s + xs*b` (`qwen3_5/cuda/qmm.py:73`) and the softmax scale expressions. Add
   `enable_fp_fusion=False`.
4. **Specialization:** Python ints specialize on divisibility by 16 and on the value 1. Add
   `do_not_specialize=[...]` for `p0`, `W`, `R`, `M` and similar runtime sizes.
5. **Transcendentals:** precompute rope cos/sin on the host (DFlash2's `_prep_kernel` already does). Use
   `tl.math.div_rn`/`sqrt_rn` where a reference needs exactness.
6. **Host code:**
   - `torch.cuda.current_device()` and `torch.device("cuda", i)` (`attention.py:235`)
   - `.is_cuda` (`attention.py:263`, `affine.py:46`)
   - `.pin_memory()`, which works with the XPU host allocator `[UNVERIFIED]`

   Route all of these through `tensorfold.accel`.

### 2.2 Per-file tables

**`families/qwen3_5/cuda/glue.py`**

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 11-39 | `_add_rmsnorm`: sums, `1/sqrt`, reshape + group sum, 8 warps | M | Cross-producer `xs` equality test; `div_rn`/`sqrt_rn` if matching a reference |
| 44-101 | `_gdn_pre`: small `static_range`, sigmoid, exp/log softplus, 2 warps | M | Fine; launch-bound |
| 106-127 | `_gated_norm` | L | |
| 131-150 | `_swiglu` BLOCK=1024, sigmoid | L | Test sigmoid against torch |
| 155-213 | `_attn_prep` in-kernel `tl.cos/sin(pos*inv)`, gathered loads | M | Precompute cos/sin tables on the host |
| 218-237 | `_gate_mul` | L | |
| 241-256 | `_embed` int64 token, nibble shifts | L | |

**`families/qwen3_5/cuda/qmm.py`** (the T0 4-bit matmul, `lane_matmul`)

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 47-124 | `_qmm` `tl.dot(x, tl.trans(q))`, BM ∈ {16,32,64,128}, BN=64, 4/8 warps, 3 stages | **H** | BM=16 miscompile risk: test row-alone vs in-window at every BM. The `q` operand is built in registers, so it takes an SLM round trip (perf). Add `grf_mode="256"` for BM=128 |
| 68 | int32→bf16 cast | M | route through fp32 |
| 73 | `acc + p*s + xs*b` | M | `enable_fp_fusion=False` |
| 82-88 | `_reduce` `static_range(1,SK≤8)` | L | fixed order |
| 35-44 | `_group_sums` | L | cross-producer test |

**`families/qwen3_5/cuda/dflash2.py`**: `_dconv_kernel` L; `_prep_kernel` L (`tl.rsqrt` accuracy `[UNVERIFIED]`).

**`families/qwen3_5/cuda/draft_attention.py`**

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 30-31, 107-108 | int64→pointer casts | **H** | §2.1(1) |
| 84, 134 | `data_ptr` tables as int64 | **H** | `_s64` |
| 13-92 | `_block_attention`: two dots, acc `(M,D)` fp32, 4 warps | **H** | M=64, D=256 means 256 fp32 per lane: use 8–16 warps and `grf_mode="256"`, and check `n_spills` |
| 97-138 | `_append` | L | |

**`families/qwen3_5/cuda/prefill_glue.py`**: FP8 prompt path only. **Refused on XPU.** Not ported.

**`cuda/kernels/qmm.py`**: `_ext()` builds the CUDA `qmm*.cu` (**H**: not available on XPU; XPU never calls it).
`_group_sums` L. `_quantize_rows` (FP8) is refused on XPU.

**`cuda/kernels/attention.py`** (27B tree attention)

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 22-37 | `_paths`: two data-dependent `while` loops, 1 warp | **H** | Rewrite (§3) |
| 41-51 | `_tile`: dot M=16, K=D, N=64, `tl.trans(k)` | M | `enable_fp_fusion=False`; check DPAS |
| 68-123 | base+offset int64 pointers, many masked gathers | M | Residency probe; `_tail` is gather-heavy (perf) |
| 159 | `_merge` | L | |
| 235, 263 | `torch.device("cuda")`, `is_cuda` | M | accel |

**`cuda/kernels/prefill_attention.py`**

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 31-82 | `_attend` BM=BN=64, D ≤ 256, 8 warps | M | `grf_mode="256"` for D=256 |
| 44-57 | two loop bodies (full and partial tiles) | M | XPU chunked-vs-one-shot test |
| 62-71 | routing: only `d==64` goes to Triton | **H** | On XPU route **all** head dims to `triton_attention` |
| 31 | `p0`, `W` as plain ints | M | `do_not_specialize` |

**`cuda/kernels/affine_kernels.py` / `affine.py`** (generic affine; not on the SYM fast path)

| Lines | Risk | Note |
|---|---|---|
| matmul BM=16 × BN=32, `input_precision="ieee"`, `enable_fp_fusion=False` | **H** | BM=16 bug. uint32→bf16 cast via fp32 |
| `codes` shift tricks | M | validate 2–8 bits |
| `embed` `tl.device_assert` | M | compiled out unless `TRITON_DEBUG` |
| `affine.py:46` `x.is_cuda` | M | accel |

**`families/nemotron_h/cuda/glue.py`**

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 33-98 | `_router` dot 16×64 by 64×16, 4 warps | M | Check that DPAS is used (N=16 is the minimum); FMA fallback is correct but slow |
| 51-100 | `_topk` small `static_range`, lowest-index ties | L | |
| 104-139 | `_add_moe_norm` 8 warps | L | Same warps as the fused producers |
| 13-22 | `dense`/`prefill_dense` → CUDA qmm | **H** | XPU routes to the T0/N kernels via `xpu/select.py` |

**`families/nemotron_h/cuda/mamba.py`**

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 12, 147-195 | `_scan`: dynamic loop carrying a `[32,128]` fp32 state, masked store in the loop, 8 warps | **H** | The DEVICE_LOST pattern [S22]. Test first; mitigations in §1.5; native fallback |
| 15-62 | `_conv` dynamic loop, many masked stores | M | |
| 78-101 | `_conv_rows` | L | |
| 104-120 | `_conv_commit` + `tl.debug_barrier()` reading and writing the same BASE | **H** | Double-buffer BASE; drop the barrier |
| 198-216 | `scan_rows` CUDA ext | **H** | K2.T0 Triton chain scan plus a K2.N0 SYCL spike |
| 220-239 | `_group_rmsnorm` | L | |

**`families/nemotron_h/cuda/attention.py`**: `_kv_write` L. `_tile/_chunk` M (dot M=16, many early-exit programs).
`_merge` M (3D reshape + sum → split into 2D reductions).

**`families/nemotron_h/cuda/sampler.py`**

| Lines | Construct | Risk | Action |
|---|---|---|---|
| 36-71 | fp64 values, `exp`, `log(-log(u))` | **H** | Check against `engine/exact_sampling.py` over 1e5 draws; measure fp64 cost |
| 66-70 | uint64 splitmix hash, `(x>>11).to(float64)` | M | Check against numpy |
| 38-40 | CP×CP compare | M | CP=256 → 65,536 elements, spills; `num_warps=4` for CP ≥ 128 |
| 53-61 | `static_range(1,K≤256)` | M | runtime `range` |
| 142-165 | torch fp64 `nucleus` | M | torch eager fp64 on XPU, unverified |

---

## 3. Rewrite recipes

```python
# 1. while -> bounded for (kernels/attention.py::_paths)
cur = node; depth = 0
for _ in range(MAXD):                         # MAXD constexpr
    live = cur >= 0
    depth += tl.where(live, 1, 0)
    cur = tl.where(live, tl.load(PARENTS + tl.maximum(cur, 0)), cur)

# 2. casts through fp32
w = q.to(tl.float32).to(tl.bfloat16)

# 3. pointer-table host side
def _s64(p): return p - (1 << 64) if p >= (1 << 63) else p

# 4. bit-sensitive launch
_kernel[grid](..., num_warps=NW, enable_fp_fusion=False)          # + do_not_specialize on runtime ints
```

Further rewrites:
- `_keyed`: `range(1, K)` instead of `static_range`; `num_warps=4` for CP ≥ 128.
- `_conv_commit`: double-buffer BASE.
- Performance, after correctness: tensor descriptors for dot operands (last stride 1, built in-kernel), and A/B
  `TRITON_INTEL_PREDICATED_LOAD=0`.

**Rule:** CUDA output must stay bit-identical. Put XPU differences behind `if dev.type == "xpu"` or per-device constant
tables (`XPU_CONFIG`). Do not edit the CUDA launch constants.

---

## 4. Smoke-test ladder (the `triton-smoke` suite; run in order, stop at the first failure)

**S0, probes:**
- `torch.xpu.is_available()`.
- Device properties: `has_fp64`, `sub_group_sizes`, `has_subgroup_matrix_multiply_accumulate`,
  `has_subgroup_2d_block_io` (**both must be True; otherwise install `ocloc`**), maximum work-group size.
- torch and triton versions.
- Vector add.
- `data_ptr() ≥ 2^63`?
- int64→pointer load round trip.
- uint64 `_mix` against numpy.
- fp64 exp/log values and timing.
- fp8 cast range (> 448, subnormals).
- fp32→bf16 rounding (inf, NaN, denormal).
- `debug_barrier` hazard with R=1,2.

**S1:** `qwen3_5/glue` kernels against torch, then self-consistency.

**S2:** `group_sums` + `lane_matmul` at M ∈ {1,2,15,16,17,33,65,129} × forced BM ∈ {16,32,64,128} × several `sk`. This
directly tests the BM=16 bug.

**S3:** affine kernels, 2/3/4/5/6/8 bits, against numpy.

**S4:** Nemotron route (`_router` + `_topk`): ties, SK, `top_k+2`.

**S5:** attentions: `triton_attention` chunked vs one-shot (splits at multiples of 64 and not); tree attention; Nemotron
`_chunk/_merge`; `draft_attention` (after the S0 pointer probe).

**S6:** Mamba `_conv`/`_scan`, last of the kernels. Call `torch.xpu.synchronize()` after every launch, with short
timeouts.

**S7:** sampler against `engine/exact_sampling.py` over 1e5 draws.

**Bitwise protocol, for every kernel:**
1. Row alone vs in window (sizes 2, 16, 17, 129): compare `int16`/`int32` views.
2. Chunked vs one-shot, with boundaries that are not multiples of 16.
3. Vary BM, `sk` and grid padding.
4. 20 repeats, warm and cold cache, with concurrent streams running.
5. Record DPAS-present, `n_spills == 0` and `threads_per_warp`.

Reuse the existing tests, after the device fixture from WS2 lands:
- `tests/cuda/test_qwen27_qmm.py`
- `test_qwen27_glue.py`
- `test_attention.py`
- `test_prefill_attention.py`
- `test_nemotron_kernels.py`
- `test_qwen27_draft_attention.py`
- `test_cuda_affine.py`
- `test_qmm.py`

---

## 5. Uncertainty register
- B70 fp64 rate and the fp64 libdevice implementations.
- TF32 DPAS on Xe2.
- Native 64-bit integer arithmetic on Xe2.
- Level Zero indirect access set by Triton's launcher.
- Device addresses ≥ 2^63 on B70.
- `tl.debug_barrier` fence semantics.
- fp8e4nv overflow behaviour.
- `tl.sigmoid` accuracy in 3.8.
- Triton version for torch 2.12 (3.7.1 inferred).
- The determinism claims in §1.7.

## 6. Top links
1. https://github.com/intel/intel-xpu-backend-for-triton
2. https://github.com/intel/intel-xpu-backend-for-triton/releases
3. https://raw.githubusercontent.com/intel/intel-xpu-backend-for-triton/main/third_party/intel/backend/compiler.py
4. https://raw.githubusercontent.com/intel/intel-xpu-backend-for-triton/main/python/triton/knobs.py
5. https://raw.githubusercontent.com/intel/intel-xpu-backend-for-triton/main/third_party/intel/lib/TritonAnnotateModule/TritonAnnotateModule.cpp
6. https://raw.githubusercontent.com/intel/intel-xpu-backend-for-triton/main/docs/BLOCK_LOADS_LAYOUT.md
7. https://download.pytorch.org/whl/xpu/torch/
8. https://github.com/intel/intel-xpu-backend-for-triton/pull/8147
9. https://github.com/intel/intel-xpu-backend-for-triton/issues/6658
10. https://github.com/intel/intel-xpu-backend-for-triton/issues/8121
11. https://github.com/intel/intel-xpu-backend-for-triton/issues/8200
12. https://github.com/intel/intel-graphics-compiler/issues/446
13. https://github.com/chriswagner-ai/intel-arc-b70-vllm-multi-gpu
14. https://github.com/intel/intel-xpu-backend-for-triton/pull/8098
15. https://huggingface.co/blog/danf/intel-xpu-kernels-skill

## 7. Source key
- S1 https://github.com/intel/intel-xpu-backend-for-triton
- S2 https://github.com/intel/intel-xpu-backend-for-triton/releases
- S3 https://download.pytorch.org/whl/xpu/torch/
- S4 wheel `.whl.metadata` files under https://download.pytorch.org/whl/xpu/
- S5 https://download.pytorch.org/whl/xpu/triton-xpu/
- S9 .../third_party/intel/backend/compiler.py
- S11 .../python/triton/knobs.py
- S12 .../TritonAnnotateModule.cpp
- S13 https://github.com/intel/intel-xpu-backend-for-triton/issues/8044
- S17 https://github.com/intel/intel-xpu-backend-for-triton/pull/7841
- S19 https://github.com/intel/intel-xpu-backend-for-triton/issues/8227
- S20 https://github.com/intel/intel-xpu-backend-for-triton/issues/8121
- S21 https://github.com/intel/intel-xpu-backend-for-triton/issues/5581
- S22 https://github.com/intel/intel-xpu-backend-for-triton/issues/6658
- S23 https://github.com/intel/intel-graphics-compiler/issues/446
- S24 https://github.com/intel/intel-xpu-backend-for-triton/issues/8200
- S25 https://github.com/intel/intel-xpu-backend-for-triton/issues/7755
- S26 https://github.com/intel/intel-xpu-backend-for-triton/issues/8191
- S27 https://github.com/intel/intel-xpu-backend-for-triton/issues/8225 , /issues/8212
- S28 https://github.com/intel/intel-xpu-backend-for-triton/pull/8098
- S29 .../third_party/intel/lib/TritonIntelGPUToLLVM/ElementwiseOpToLLVM.cpp
- S34 https://github.com/intel/torch-xpu-ops/issues/5524
- S35 https://github.com/vllm-project/vllm-xpu-kernels/pull/570
- S38 .../benchmarks/triton_kernels_benchmark/benchmark_testing.py
- S40 https://github.com/intel/intel-xpu-backend-for-triton/issues/6790
- S42 https://github.com/intel/intel-xpu-backend-for-triton/pull/8128
- S45 https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html
