# B70 (Battlemage / Xe2) Native XPU Kernel Developer Guide

Audience: agents porting TensorFold's hand-written CUDA kernels (`.cu` with `mma.sync`, `ldmatrix`, `cp.async`, warp
shuffles) to native SYCL / ESIMD kernels for the Intel Arc Pro B70. For Triton work, read
[TRITON_XPU_GUIDE.md](TRITON_XPU_GUIDE.md). For weight formats, read [QUANT_FORMATS.md](QUANT_FORMATS.md).

The guide was compiled on 2026-09-30 from public sources, and every fact carries a link. Tags:
- **[derived]**: computed by us.
- **[inferred]**: reasoned, not stated by a source.
- **[CONFLICT]**: sources disagree.
- **[UNVERIFIED]**: not confirmed.

Re-verify anything tagged before relying on it, and record what you measure on the box in `docs/xpu/STATUS.md`.

---

## 0. The ten rules

1. **The sub-group (warp) is 16 lanes for anything that touches XMX.** DPAS, `joint_matrix` and 2D block I/O all
   require sub-group size 16. Annotate with `[[sycl::reqd_sub_group_size(16)]]`. Redesign every piece of 32-lane logic:
   shuffle masks, `lane>>2`/`lane&3` fragment maps, and 5-step butterflies.
2. **The DPAS tile is M≤8 × N=16 × K=16 (bf16/fp16 → fp32), not m16n8k16.** Re-tile; never translate CUDA fragment
   layouts one to one.
3. **There is no FP8 or FP4 in Xe2 XMX.** Upconvert in registers. Int8 (K=32), int4 (K=64) and int2 DPAS exist.
4. **`ldmatrix` and `cp.async` become 2D block loads and prefetches from global memory straight into registers**, with
   built-in transpose and VNNI. SLM staging is usually unnecessary for GEMM operands. There is no async global→SLM copy
   engine.
5. **Decode is bandwidth-bound.** Peak is 608 GB/s; about 450–500 GB/s is achievable. Target ≥ 80% of peak for the
   GEMV.
6. **Always compile device code with `-fp-model=precise -ffp-contract=off`.** icpx defaults device code to fast-math.
   Use explicit `sycl::fma` where the arithmetic contract has an FMA.
7. **No float atomics in reductions.** Fix split-K, tile, sub-group size and GRF mode per shape. Reduce in a fixed
   order in a second pass. This is required for TensorFold's exactness contract.
8. **Never source oneAPI `setvars.sh` in the runtime environment.** It crashes Triton XPU with SIGSEGV. Build
   extensions AOT in a separate shell (`tools/xpu/build_ext.sh`); the server only loads prebuilt `.so` files.
9. **No single device allocation may exceed 4 GB.** Use 64-bit index math everywhere.
10. **Out-of-bounds bugs can wedge the GPU or the host.** Bounds-check in debug builds, and run new kernels with a
    `torch.xpu.synchronize()` after each launch and short timeouts.

---

## 1. Hardware

### 1.1 Board and chip

| Item | Value | Source |
|---|---|---|
| Die | BMG-G31 ("Big Battlemage"), full die, PCI ID 8086:E223 | [PMZFX hardware.md](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md), [zolotukhin ref](https://zolotukhin.ai/zinc/docs/intel-gpu-reference/) |
| Architecture | Xe2-HPG | [Intel datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf) |
| Xe-cores / render slices | 32 / 8 | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html) |
| Vector engines (XVE) | 256 (8 per Xe-core) | ARK |
| XMX engines | 256 (8 per Xe-core) | ARK |
| Clocks | 2280 MHz graphics, 2800 MHz max dynamic | ARK |
| FP32 vector | 22.94 TFLOPS | ARK |
| INT8 XMX dense | 367 TOPS | ARK |
| Memory | 32 GB GDDR6 ECC, 256-bit, 19 Gbps, **608 GB/s** | ARK, [VideoCardz](https://videocardz.com/newz/intel-launches-arc-pro-b70-at-949-with-32gb-gddr6-memory) |
| PCIe | Gen5 x16 per Intel. **[CONFLICT]** One rig reports x8, and Linux may misreport Gen1 x1 | ARK, PMZFX, [gist](https://gist.github.com/mploschiavo/9968c883c4a872a74e0f38edd7cda2ef) |
| Board power | 230 W (Intel card) | datasheet |

### 1.2 Peak rates [derived]
The Xe2 Xe-core does 2048 fp16 and 4096 int8 XMX ops per clock
([Xe2 Tech Tour deck](https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf)).

- INT8: 4096 × 32 × 2.8 GHz = 367 TOPS, which matches ARK.
- **FP16/BF16 XMX: about 183.5 TFLOPS.** TF32: about 92 TFLOPS **[UNVERIFIED]**.
- INT4/INT2: a paper reports that int2×int8 DPAS runs at the *same* throughput as int8 on Xe2
  ([arXiv 2508.06753](https://arxiv.org/html/2508.06753)). Assume there is no gain below 8 bits until you benchmark it.
- Measured on B70:
  - 4096² fp16 matmul: **128.3 TFLOPS**; a bandwidth test: **443 GB/s**
    ([IDFS diary](https://idfs.ai/blog/six-days-with-the-intel-arc-pro-b70)).
  - Int2/int8 GEMV: **about 500 GB/s** (arXiv 2508.06753).

### 1.3 Xe-core internals
- **Threads:** each XVE holds up to 8 hardware threads. SIMD16 and SIMD32 are supported; SIMD8 is gone
  ([chipsandcheese](https://chipsandcheese.com/p/intels-battlemage-architecture)).
- **GRF:** 64-byte registers, 64 KB per XVE.
  - 128 registers per thread with 8 threads per XVE, or **256 registers per thread with 4 threads per XVE**.
  - 512-register mode is PVC-only.
  - Whole device: 2048 resident sub-groups in 128-register mode, 1024 in 256-register mode [derived].
  - Sources: [Intel GRF guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/grf-mode-selection.html),
    zolotukhin.
- **L1/SLM:** 256 KB shared per Xe-core. **The maximum SLM per work-group is 128 KB**
  ([pytorch #179030](https://github.com/pytorch/pytorch/issues/179030)). SLM latency is about 15 ns.
- **L2:** **[CONFLICT]** 16 MB (spec databases) vs 24 MB (leak). Query it at runtime:
  `sycl::info::device::global_mem_cache_size`, or the PyTorch ≥ 2.13 device properties.
- **Atomics:** 32 atomic ALUs per Xe-core; 64-bit atomics are supported.
- **Co-issue:** XMX co-issues with vector FP and INT work.
- **Device properties** (B70/B580): `max_work_group_size=1024`, `sub_group_sizes=[16,32]`, `gpu_eu_count=256`,
  `has_fp64=1`, `has_atomic64=1`. FP64 *rate* is **[UNVERIFIED]**; benchmark it before using fp64 on hot paths.
- **Occupancy [inferred]:** a work-group lives on one Xe-core. At SG16 with 128 GRF, 1024 work-items = 64 sub-groups =
  every thread slot of the Xe-core. With 256 GRF there are only 32 slots, so **cap work-groups at 512 work-items
  (SG16) in large-GRF mode.**

### 1.4 XMX data types

| Type | Xe2 XMX | Notes |
|---|---|---|
| fp16/bf16 → fp32 | yes | K=16 ([Khronos MMA ext](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)) |
| fp16→fp16, bf16→bf16 accumulate | yes | |
| tf32 → fp32 | yes | M≤8, N=16, K=8 ([joint_matrix spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)) |
| s8/u8 → s32 | yes | K=32 |
| s4/u4 → s32 | yes | K=64 (`intel_sub_group_i4_i4_matrix_mad_k64`) |
| s2/u2, int2×int8 | yes | [ESIMD spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md), [TernSYCL](https://github.com/libxsmm/TernSYCL) |
| **FP8 e4m3/e5m2** | **no** | software upconvert ([llm-scaler #738](https://github.com/intel/llm-scaler/issues/738)) |
| **FP4 / MXFP4** | **no** | Xe3P (Crescent Island) only ([chipsandcheese Hot Chips 2026](https://chipsandcheese.com/p/hot-chips-2026-intels-crescent-island)) |

Consequence for TensorFold: the 4-bit weights are **dequantized to bf16 (or fp16) in registers and fed to bf16 DPAS**
(W4A16). Native int4 DPAS would need int8 or int4 activations (W4A8), which changes the model's arithmetic. It is out
of scope.

### 1.5 DPAS shape and layout
- Systolic depth 8, execution size **16** on Xe2 (8 on DG2), repeat count M ∈ {1,2,4,8}. The basic bf16 tile is
  **8×16×16**.
- K per dtype: fp16/bf16 16, int8 32, int4 64, tf32 8.
- SG16 per-lane layout for `float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)` [inferred,
  **verify with a unit test before relying on it**]:
  - Lane *n* holds column *n* of B (16 K-values, VNNI pairs).
  - Lane *n* holds column *n* of C (8 rows as `float8`).
  - A is spread across lanes.
- Compared with CUDA m16n8k16: both produce 128 accumulator elements per instruction, but the shape is transposed (8×16
  vs 16×8). **Rebuild the tiling from scratch.**
- `joint_matrix` on BMG-G21/G31: SG16 only. bf16/fp16 → fp32 support 16×16×16 among other shapes. Query
  `matrix_combinations` at runtime for the authoritative list.

**TensorFold mapping for decode GEMV [inferred, the core design idea for K4/K3]:**
- Put **weight output rows on DPAS M (8)** and **window tokens on DPAS N (16 lanes)**.
- TensorFold verify windows are ≤ 12–16 rows, so one N tile covers the whole window.
- For M=1 serial decode, bench both DPAS with padded N and a vector-FMA path with a sub-group reduction.

### 1.6 2D block load / store / prefetch (the replacement for `ldmatrix` + `cp.async`)
From [cl_intel_subgroup_2d_block_io](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html):
- Loads one or more 2D blocks from **global** memory into registers, optionally transposed or VNNI-transformed. Also
  writes 2D blocks, and prefetches to cache only.
- Requirements:
  - Full sub-group of 16.
  - Base address **64 B aligned**.
  - Surface width **64–224 B**; pitch ≥ width and a **multiple of 16 B**.
  - x-coordinate a multiple of 4 for 8-bit elements and of 2 for 16-bit.
- Block sizes: 8-bit widths 16/32; 16-bit width 16; 32-bit widths 8/16; heights 1–32. Transpose and transform support
  only subsets.
- ESIMD form ([ESIMD opt guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/optimizing-explicit-simd-kernels.html)):
  - `config_2d_mem_access<T,W,H,NBlk>` with `lsc_load_2d` / `lsc_store_2d` / `lsc_prefetch_2d`.
  - Width, height and pitch are passed as **value − 1**; width and pitch are in bytes; x is in elements.
  - W·sizeof(T) ≤ 64 B and H ≤ 32.
  - On Xe2 transposed loads work only on `uint32_t`: load fp16 as u32 and `bit_cast_view`. `store_2d` of half is
    limited to 8 rows
    ([LightX2V skill](https://skills.lc/ModelTC/LightX2V/modeltc-lightx2v-claude-skills-lightx2v-kernel-skills-intel-xpu-kernel-basic-skills-esimd-lsc-2d-gather-scatter-skill-md)).
- How CUDA patterns map [inferred]:
  - `ldmatrix(.trans)` → a 2D block load with transpose or transform, from **global memory**.
  - Multi-stage `cp.async` pipelines → `prefetch_2d` a few K-steps ahead plus direct 2D loads, with L1/L2 acting as the
    staging buffer.
- **Missing `ocloc` breaks this silently.** If `libocloc.so` is absent, `torch.xpu.get_device_properties()` reports
  `has_subgroup_2d_block_io` / `has_subgroup_matrix_multiply_accumulate` as False, and Intel Triton silently takes
  non-DPAS paths ([pytorch #196074](https://github.com/pytorch/pytorch/issues/196074)). The `env` suite asserts both are
  True.

**Weight layout implication for `pack_xpu`:** design packed 4-bit weight tiles so that one 2D block load (width ≤ 64 B,
pitch a multiple of 16 B, 64 B-aligned base) fetches exactly the nibbles one sub-group needs for one K=16 DPAS B operand
(or for M=8 rows of A). The CUDA "fragment order" layout in `cuda/kernels/qmm.py` is NVIDIA-specific; do not reuse it.

---

## 2. Software stack (pin these)

| Component | Version / requirement | Source |
|---|---|---|
| Kernel / driver | Linux **≥ 6.17** with `xe` (not i915); Ubuntu 24.04 HWE or 26.04; GuC/HuC `bmg_*` firmware; `xe.force_probe=e223` on older kernels | [IDFS](https://idfs.ai/blog/six-days-with-the-intel-arc-pro-b70), [kkornas guide](https://github.com/kkornas/intel-arc-pro-b70-ubuntu-guide/blob/main/docs/01-base-driver-setup.md) |
| compute-runtime / IGC / Level Zero loader | CR **26.31**, IGC **2.40.x**, L0 loader **≥ 1.32** (needed for B70 graph capture) | [exl3xpu #1](https://github.com/0xSero/exl3xpu/issues/1) |
| Avoid | CR 26.14 (broke multi-rank on G31, [CR #922](https://github.com/intel/compute-runtime/issues/922)) | |
| PyTorch | **2.14.x+xpu** (`pip install torch --index-url https://download.pytorch.org/whl/xpu`), bundled `triton-xpu~=3.8.0`, runtime 2026.1 | [2.14 blog](https://pytorch.org/blog/pytorch-2-14-release-blog/), [get-started](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html) |
| Compiler | **Intel Deep Learning Essentials 2026.1** (icpx / DPC++), matching torch's runtime. **No full oneAPI Base Toolkit and no IPEX in the same environment** | [Triton XPU README](https://github.com/intel/intel-xpu-backend-for-triton) |
| `ocloc` | installed (see §1.6) | |
| ReBAR | **enabled** | [Intel](https://www.intel.com/content/www/us/en/support/articles/000099073/graphics.html) |
| Host RAM | **The test box has 32 GB.** One report says each torch-xpu allocation commits matching host RAM on B70 (torch 2.14 / kernel 7.1.8). Unverified on our box: the `env` suite measures it, and a 32 GB swap file is recommended. See §5.4 | [torch-xpu-ops #5428](https://github.com/intel/torch-xpu-ops/issues/5428) |

Do **not** build on IPEX (end of life March 2026) or IPEX-LLM (archived 2026-01-28). XeTLA is archived; use SYCL*TLA.

### 2.1 torch.xpu API (mirrors torch.cuda)
- Streams and events: `Stream`, `Event`, `current_stream`, `stream`.
- Memory: `memory_allocated/reserved`, `empty_cache`, `mem_get_info`, `MemPool`.
- Graphs: **`XPUGraph`, `graph`, `graph_pool_handle`** (2.11+).
- Other: `synchronize`, `get_device_properties`.
- Telemetry (2.13+): memory used, utilization, power, clocks.
- Source: [docs](https://docs.pytorch.org/docs/main/xpu.html).
- **Graph caveats:**
  - vLLM produced gibberish with XPU graphs on 2× B70 ([vllm #48327](https://github.com/vllm-project/vllm/issues/48327)).
  - Graph-pool lifetime assertion bug ([pytorch #198794](https://github.com/pytorch/pytorch/issues/198794)).
  - Kernels must be capture-safe: no host sync and no allocation inside capture.
  - TensorFold gates graphs on a graphs == eager **bitwise** test.

### 2.2 Building a SYCL extension for torch
- JIT: `torch.utils.cpp_extension.load(name, sources, sycl_sources=[...], extra_sycl_cflags=[...], with_sycl=True)`.
- AOT: `SyclExtension` (Linux, torch ≥ 2.8). Needs ninja and `icpx` on PATH **in the build shell only**
  ([tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html),
  [cpp_extension 2.14](https://docs.pytorch.org/docs/2.14/cpp_extension.html)).
- Target architecture: `TORCH_XPU_ARCH_LIST=bmg`, but use **`intel_gpu_bmg_g31`** explicitly in `-fsycl-targets`.
  **[CONFLICT]** `-device bmg` may resolve to G21 (B580-class)
  ([exl3xpu #1](https://github.com/0xSero/exl3xpu/issues/1)).
- TensorFold flags (see `src/tensorfold/xpu/build.py` once written):
  ```
  -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -fp-model=precise -ffp-contract=off
  # ESIMD extensions additionally:
  -Xsycl-target-backend=intel_gpu_bmg_g31 "-options '-vc-codegen'"
  # large GRF (per extension, only when profiling says so):
  -Xs "-options -ze-opt-large-register-file"     # or the grf_size<256> kernel property
  ```
  TernSYCL found the `grf_size` property did not take effect without the `-ze-opt-large-register-file` option. **Verify
  the GRF mode in the IGC dump.**
- Binding pattern:
  ```cpp
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();   // ALWAYS launch on torch's queue
  TORCH_LIBRARY(tensorfold_xpu, m) { m.def("qmm(Tensor x, ...) -> Tensor"); }
  TORCH_LIBRARY_IMPL(tensorfold_xpu, XPU, m) { m.impl("qmm", &qmm_xpu); }
  ```
  A pybind `_ext()` shim exposing the same function names as the CUDA extension is also acceptable, and keeps the
  engines unchanged.

### 2.3 Distributed (future, second B70)
- `torch.distributed` backend **`"xccl"`** (native since 2.8; torch-ccl is dead)
  ([blog](https://pytorch.org/blog/pytorch-2-8-brings-native-xccl-support-to-intel-gpus-case-studies-from-argonne-national-laboratory/)).
- Multi-B70 reports include GPU faults and BCS engine resets with TP=2
  ([vllm #41663](https://github.com/vllm-project/vllm/issues/41663)). Prefer one process per card.

---

## 3. Programming models

| Model | Use for | Notes |
|---|---|---|
| Triton (Intel backend) | T0/T1 of every kernel | See TRITON_XPU_GUIDE.md |
| **SIMT SYCL + OpenCL DPAS builtins + `__builtin_IB_*` 2D IO** | hand-tuned W4A16 GEMV/GEMM, GDN, scans | The TernSYCL approach. SIMT reads naturally; IGC codegen is sensitive to small source changes |
| SYCL `joint_matrix` | portable SIMT GEMM | Fixed shapes, SG16; no published B70 perf numbers |
| **ESIMD** (`[[intel::sycl_explicit_simd]]`) | maximum control: `xmx::dpas<8,M,...>`, `lsc_load_2d`, `slm_init<N>()` | Needs `-vc-codegen` for AOT. exl3xpu reaches strong B70 numbers with it |
| **SYCL*TLA** (CUTLASS/CuTe for Intel) | GEMM, grouped GEMM (MoE), flash attention | Header-only; examples target `intel_gpu_bmg_g31` |

### 3.1 Code to study before writing a kernel

| Kernel | Study |
|---|---|
| K4 decode GEMV (W4A16) | [TernSYCL](https://github.com/libxsmm/TernSYCL): B70 SIMT DPAS + 2D IO GEMV/GEMM, SG16, grf 128 for GEMV and 256 for GEMM, inline vISA. [exl3xpu](https://github.com/0xSero/exl3xpu): B70 ESIMD GEMV/GEMM and the AOT recipe. [PrismML llama.cpp PR #294](https://github.com/PrismML-Eng/llama.cpp/pull/294): a transposed 2D load fetches 16 rows × 128 weights directly in the DPAS B layout, split-K across threads with an SLM reduce (61% of BW at batch 1–2). vllm-xpu-kernels `int4_gemm_w4a16` (oneDNN) as a **speed baseline only** |
| K5 prompt GEMM | sycl-tla `examples/02_bmg_gemm_mixed_dtype` (int4 dequant in the B path), `00_bmg_gemm` |
| K3 MoE experts | sycl-tla `10_bmg_grouped_gemm_mixed_dtype`, `12_xe20_moe_gemm_cute_interface`; vllm-xpu-kernels MoE grouped GEMM (W4A16 tiling that Intel AutoRound tunes against) |
| K1 GDN | GDN kernels in [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) and [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu). **Structure only**; their arithmetic differs from TensorFold's contract |
| K2 Mamba scan | kernels-community `mamba-ssm` (XPU builds) |
| K6 prompt attention | sycl-tla `06_bmg_flash_attention`; exl3xpu reports 80–90 TFLOPS at head_dim 256 |

`torch._weight_int4pack_mm` on XPU is **wrong at M=1** and slow; never use it as a reference or baseline
([torch-xpu-ops #4765](https://github.com/intel/torch-xpu-ops/issues/4765)).

---

## 4. CUDA → SYCL cheat-sheet

| CUDA | SYCL / Intel | Notes |
|---|---|---|
| warp (32) | sub-group (16 for XMX) | `[[sycl::reqd_sub_group_size(16)]]` |
| `__shfl_xor_sync(m,v,x)` | `sycl::permute_group_by_xor(sg, v, x)` | no mask; XOR offsets stop at 8 with SG16 |
| `__shfl_sync` | `sycl::select_from_group(sg, v, src)` | |
| `__shfl_up/down_sync` | `sycl::shift_group_right/left(sg, v, d)` | |
| butterfly sum | `sycl::reduce_over_group(sg, v, sycl::plus<>())` | **Order is implementation-defined.** For bit-exact contracts write explicit `permute_group_by_xor` butterflies |
| `__match_any_sync`, `__popc` | `sycl::ext::oneapi::group_ballot` + popcount, or restructure | the experts plan kernel uses these; prefer torch `argsort` first (K3a) |
| `__syncthreads()` | `sycl::group_barrier(it.get_group())` | |
| `__shared__` | `sycl::local_accessor<T,1>`; ESIMD: `slm_init<N>()` | at most 128 KB per work-group |
| `atomicAdd` | `sycl::atomic_ref<...>(x).fetch_add(v)` | never for float reductions (determinism) |
| `float4`/`int4` loads | `sycl::vec<T,N>`, sub-group block loads | |
| `mma.sync m16n8k16` | `joint_matrix_mad`; ESIMD `xmx::dpas<8,M>`; OpenCL `intel_sub_group_bf16_bf16_matrix_mad_k16` | 8×16×16 |
| `ldmatrix` | 2D block load (transpose/VNNI) | global memory only |
| `cp.async` + `wait_group` | `prefetch_2d` / `joint_matrix_prefetch` plus direct loads | |
| inline PTX `asm()` | ESIMD inline asm on `simd` types; inline vISA in SIMT SYCL under `__SYCL_DEVICE_ONLY__`; `__builtin_IB_*` | see TernSYCL |
| `__launch_bounds__`, maxrregcount | `grf_size<256>` property, `-ze-opt-large-register-file`, `-ze-intel-enable-auto-large-GRF-mode` | |
| `--fmad=false` | `-fp-model=precise -ffp-contract=off` + explicit `sycl::fma` | |
| `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` | not needed; just stay ≤ 128 KB SLM per work-group | |
| thread-block clusters + DSMEM | **none**; use a second-pass reduce kernel | TensorFold already has the `reduce_kernel` fallback |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | compute from GRF mode (§1.3) and `gpu_subslice_count` | |
| SYCLomatic `c2s` | scaffolding only | mma/ldmatrix come out as scalar emulation, not DPAS |

The bf16 nibble decode trick carries over directly: `bf16 bits (0x4300 | q) − 128.0` gives exactly `q` as bf16. In
SYCL, do the `−128` with an explicit bf16x2 subtract or in fp32; check that IGC does not fuse it.

---

## 5. Performance

### 5.1 Roofline [derived]
- Bandwidth peak is 608 GB/s; achievable is 443–500 GB/s.
- Ridge points: bf16 XMX about 302 FLOP/B, vector fp32 about 38 FLOP/B.
- W4A16 at M tokens has an intensity of about 4·M FLOP per weight byte, so it is **bandwidth-bound up to M ≈ 50–75**.
  Every TensorFold decode/verify window (≤ 16 rows) is bandwidth-bound.
- Targets:

  | Kernel | Target |
  |---|---|
  | K4 decode GEMV | ≥ 490 GB/s effective |
  | Prompt GEMM | > 120 TFLOPS |
  | 27B serial decode | ≈ 14–19 GB / 500 GB/s ≈ 26–35 tok/s roofline before drafting |

- Fill the GPU: 32 Xe-cores × 64 thread slots. Launch thousands of sub-groups, or size persistent kernels with
  `gpu_subslice_count`.
- GEMV tips:
  - 128-GRF mode for occupancy.
  - Contiguous lane access with block loads.
  - Reduce within the sub-group before going to SLM.
  - SG16 for dequant GEMV (llama.cpp lost 20–25% using SG32 on Xe2,
    [Hal9000AIML](https://github.com/Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes)).

### 5.2 Profiling
- **unitrace** ([README](https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md)):
  - `-d` (device timing), `--chrome-kernel-logging`
  - `-q -g ComputeBasic` (metrics), `--stall-sampling`
  - Wrap PyTorch in `torch.autograd.profiler.emit_itt()`.
- **VTune** GPU Compute/Media Hotspots shows XVE and XMX pipe utilization.
- **IGC dumps:** `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>`
  ([IGC docs](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/shader_dumps_instruction.md)).
  In `.asm`:
  - `grep dpas` (XMX in use);
  - `grep load_block2d|prefetch_block2d` (2D IO in use);
  - spill size in the header;
  - long `mach`/`macl` chains mean software address math, a red flag.

### 5.3 Environment variables
- `ONEAPI_DEVICE_SELECTOR=level_zero:0`: device selection.
- `SYCL_CACHE_PERSISTENT=0`: B70 reports JIT-cache corruption. **Prefer AOT.**
- `SYCL_PROGRAM_COMPILE_OPTIONS` / `SYCL_PROGRAM_APPEND_COMPILE_OPTIONS`: IGC options for JIT.
- `SYCL_UR_TRACE`, `SYCL_RT_WARNING_LEVEL=1`: debugging.
- `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`: **[CONFLICT]** some sources call it `UR_L0_USE_...`. Even with it,
  single allocations over 4 GB failed through torch 2.11 on B70.
- `ZES_ENABLE_SYSMAN=1`: free-memory queries.
- Source: [DPC++ env vars](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md).

### 5.4 Known B70 pitfalls
1. ReBAR off → slow, or a hang at probe on some platforms.
2. Allocations > 4 GB: raw Level Zero needs relaxed limits **and** kernels built with
   `-ze-opt-greater-than-4GB-buffer-required`
   ([CR guide](https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md)).
   **TensorFold keeps every tensor < 4 GB.**
3. Host-RAM shadowing of device allocations ([torch-xpu-ops #5428](https://github.com/intel/torch-xpu-ops/issues/5428)).
   The test box has **32 GB host RAM**. If shadowing applies, the worst case (about 22–28 GB of device allocations
   mirrored, plus 3–6 GB of OS, Python and load buffers) is about 25–34 GB.
   - The `env` suite's `host_ram_shadow` probe allocates 4/8/16 GB on XPU and records the change in `MemAvailable`
     (resident) vs `Committed_AS` (commit only).
   - If it is commit-only: swap or `vm.overcommit_memory` settings cover it.
   - If it is resident: cap the XPU memory budget via `TENSORFOLD_MEMORY_RESERVE_GIB` or `--context`, and prefer
     smaller KV.
   - `bootstrap.sh` offers a 32 GB swap file.
4. Driver wedges and reset storms under sustained load, which have even corrupted the host page cache
   ([CR #948](https://github.com/intel/compute-runtime/issues/948),
   [CR #966](https://github.com/intel/compute-runtime/issues/966)). The runner checks `xpu-smi` health and stops the
   queue when the GPU is wedged.
5. Silent wrong-output bugs in the ecosystem:
   - vLLM W4A16 `!` tokens ([vllm #53480](https://github.com/vllm-project/vllm/issues/53480));
   - llama.cpp B70 corruption ([#21893](https://github.com/ggml-org/llama.cpp/issues/21893));
   - empty boolean-mask indexing ([pytorch #199163](https://github.com/pytorch/pytorch/issues/199163));
   - NaNs in bf16 matmul ([pytorch #199179](https://github.com/pytorch/pytorch/issues/199179)).

   **Validate every kernel bitwise against its contract, and soak-test under concurrency.**
6. Device addresses may be ≥ 2^63. A `torch.tensor([p.data_ptr()], dtype=torch.int64)` then overflows; use a
   two's-complement conversion ([vllm-xpu-kernels PR #570](https://github.com/vllm-project/vllm-xpu-kernels/pull/570)).

---

## 6. Determinism (TensorFold's exactness contract on XPU)

TensorFold accepts a draft only if it bit-equals the serial token. Every kernel on a decode/verify path must give
**identical bits for a row regardless of which other rows share the launch.** Prompt kernels must be
**chunk-invariant**.

- **DPAS itself:** no source states it is deterministic. **[inferred]** A fixed instruction sequence on fixed inputs
  gives fixed bits. Nondeterminism comes from atomics, dynamic split-K and scheduling-dependent reductions; avoid all
  three.
- **Rules:**
  1. Never use float atomics. Use two-pass, fixed-order reductions (CUDA's `reduce_kernel` pattern).
  2. Split-K, tile sizes, sub-group size (`reqd_sub_group_size`) and GRF mode are pure functions of the **weight
     shape**, never of M (the row count) and never autotuned at runtime.
  3. Use `-fp-model=precise -ffp-contract=off` and explicit `sycl::fma` at exactly the points the kernel card specifies.
  4. Write explicit butterfly orders instead of `reduce_over_group` wherever two kernels must agree.
  5. Pin compiler, IGC and driver. A toolchain bump re-baselines the "bits" goldens; the runner records the toolchain
     hash.
  6. Torch eager ops on XPU are **not** deterministic by default: oneDNN split-K in `F.linear`
     ([torch-xpu-ops #5524](https://github.com/intel/torch-xpu-ops/issues/5524)). Set
     `torch.use_deterministic_algorithms(True)` and keep torch matmuls off exactness paths.
- **Tests every native kernel must pass:**
  - row alone == in window, at every window size;
  - chunked == one-shot;
  - 20 repeats give identical bits;
  - warm and cold extension cache give identical bits;
  - tolerance against the fp32/fp64 reference;
  - and, where the kernel card says so, bit-equality with the Triton T1 kernel.

---

## 7. Open questions to settle on the box (record answers in STATUS.md)
- [ ] L2 size, from `global_mem_cache_size`.
- [ ] FP64 throughput (affects the samplers' fp64 paths).
- [ ] int4 vs int8 DPAS throughput (only relevant if W4A8 is ever considered).
- [ ] Whether `reduce_over_group` order is stable across IGC versions.
- [ ] Achieved GB/s of a plain SG16 vectorized copy kernel (the practical roofline for K4).
- [ ] Whether `XPUGraph` capture of TensorFold's Triton kernels replays bit-identically.
- [ ] Whether `data_ptr() ≥ 2^63` on this box.

## 8. Most useful links
1. [Intel ARK B70](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html)
2. [chipsandcheese Battlemage](https://chipsandcheese.com/p/intels-battlemage-architecture)
3. [zolotukhin Intel GPU reference](https://zolotukhin.ai/zinc/docs/intel-gpu-reference/)
4. [Khronos subgroup MMA](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
5. [Khronos 2D block IO](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html)
6. [ESIMD spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md)
7. [joint_matrix spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)
8. [SYCL custom ops tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html)
9. [TernSYCL](https://github.com/libxsmm/TernSYCL)
10. [exl3xpu](https://github.com/0xSero/exl3xpu) and [its AOT recipe](https://github.com/0xSero/exl3xpu/issues/1)
11. [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)
12. [sycl-tla](https://github.com/intel/sycl-tla)
13. [unitrace](https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md)
14. [IGC shader dumps](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/shader_dumps_instruction.md)
15. [HF XPU kernel optimizations](https://raw.githubusercontent.com/huggingface/kernels/main/kernel-builder/skills/xpu-kernels/references/xpu_optimizations.yaml)
