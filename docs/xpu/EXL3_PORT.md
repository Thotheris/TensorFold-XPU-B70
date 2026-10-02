# EXL3 on XPU: how exl3xpu works, and the plan to port TensorFold's EXL3 path to the B70

Status: **planned, workstream WS10.** Starts after the W4A16 recipes reach P4. Read [PORT_PLAN.md](PORT_PLAN.md) §WS10
for scheduling.

Sources, read 2026-10-02 (code reading only; nothing was run):
- **The user's lab fork** `D:\projects\exl3xpu-b70-lab` (`Thotheris/exl3xpu-b70-lab`, branch `experiments/b70-qwen38`).
  It is pinned to upstream `0xSero/exl3xpu@15ded2f3`. MIT licence, © 2026 0xSero.
- **TensorFold's own EXL3 CUDA path:** `src/tensorfold/cuda/exl3/*`, `families/qwen3_5/cuda/exl3_load.py`,
  `families/qwen4_exp/cuda/exl3*.py`, `families/glm5_next/cuda/exl3*`, `docs/recipes/exl3.md`.

> **Never open PRs, issues or pushes against `0xSero/exl3xpu` or `ashhart/TensorFold`.** The lab clone's `upstream`
> push URL is disabled, as is this repo's. We learn from exl3xpu's code and may copy MIT-licensed code with
> attribution (keep the copyright notice, add an entry to `THIRD_PARTY_NOTICES.md`). Every change lands in
> `Thotheris/*` repos only.

---

## 1. Why EXL3, and why later

**Why:**
- Quality per bit is better than 4-bit GPTQ.
- At 3–4 bpw the 27B is about 11–14 GB, which leaves far more of the 32 GB device (and the 32 GB host) for KV and
  drafters.
- Upstream TensorFold already supports EXL3 on CUDA: loaders, format checks, row-invariant kernels and tests.
- exl3xpu proves the codebook decode is **bit-exact on B70** and fast. Its M=1 linear budget for Qwen3.8-27B 4.00bpw is
  27.2 ms, about 481 GB/s.

**Why later:**
- Decoding EXL3 is integer-ALU heavy (trellis windows plus codebook plus Hadamard). It is harder to keep near the
  bandwidth roofline and harder to make row-invariant.
- Intel's stack does not support it, so we carry it alone.
- It needs the device layer, harness, native build (K0) and B70 loop that the W4A16 work builds first.

---

## 2. How exl3xpu works

### 2.1 The format it consumes
- `trellis` int16 `[k/16, n/16, 16·K]`, the checkpoint's stored order. It is read as little-endian uint32 pairs forming
  an MSB-first bit stream.
- Value t is the 16-bit window ending at bit `(t+1)·K`, circular within the 16×16 tile.
- Tile position: `row = 8·((t>>1)&1) + 2·((t>>3)&3) + (t&1)`, `col = (t>>5) + 8·((t>>2)&1)`. This is identical to
  TensorFold's `tile_positions()`.
- `suh` fp16 `[k]`, `svh` fp16 `[n]`. A scalar `mul1`/`mcg` marker selects the codebook (ids: 0=3inst, 1=mcg, 2=mul1).
- Forward: `y = had(had(x·suh) @ W_inner) · svh`, with a blockwise-normalised Sylvester H128.
- **Limits:**
  - integer K only (no 1.5/2.5/3.5);
  - `static_assert(V >= 8)` makes K=8 impossible;
  - only mul1 K=4/6 by default (2/3/5 and mcg/3inst need `EXL3_ALL_CODEBOOKS` and were never benchmarked);
  - all dims must be multiples of 128;
  - no `su`/`sv` packed signs;
  - bias only in Python;
  - **no MoE**;
  - no TP.

### 2.2 Integration
- It is a vLLM general plugin (`exl3xpu.vllm_plugin:register`).
- `Exl3Config` claims `LinearBase` and `ParallelLMHead` only; fused modules stack per-part `suh` into `[S,K]`.
- The forward is one opaque C++ op, `torch.ops.exl3xpu_C.linear(...)`, with a fake impl so XPU graphs see a single
  node.
- `vllm_patches.py` fixes vLLM-XPU problems: a GDN mask-index host sync, fp8-KV prefill attention, and KV block size.

### 2.3 Build
- `scripts/build_ext.sh` calls `icpx` directly:
  `-fsycl -fsycl-targets=spir64 -O3 -ffast-math -fsycl-device-code-split=per_kernel`, linking torch/c10_xpu and
  optionally oneDNN. The result loads with `torch.ops.load_library`.
- **JIT `spir64` only. AOT failed** ("more than one module with an entry point" via ocloc). The first launch after boot
  pays the IGC finalisation cost.
- ESIMD kernels run one hardware thread per work-item with explicit `simd<>` widths, work-group size 8, default 128 GRF,
  and `grf_size<256>` for DPAS MB ≥ 40.
- Stack:
  - exl3xpu README: vLLM 0.26.1 XPU, oneAPI 2025.3, oneDNN 3.9, kernel 7.1.8 xe, CR 26.31, IGC 2.40, L0 1.32.
  - The lab host: kernel 7.0.0, BMG-G31 `8086:e223`.

### 2.4 Kernels (`csrc/exl3_esimd.h`, launchers in `csrc/exl3_ops.sycl`)

| Kernel | What it does | Notes |
|---|---|---|
| `decode_cb_h<CB,N>` | codebook decode. mul1: `x=st·0x83DCD12D`; `dp4a(0x6400,x,0x01010101)` → fp16 `1024+bytesum`; one fp16 mad `h·0.006767… − 10.3828` (≡ `__hfma(h,0x1eee,0xc931)`). mcg/3inst: multiply, mask, xor, two halves, `__hadd` | **bit-exact with exllamav3**. LUT (4× slower) and MUL16 variants rejected |
| `tile_states<K,…>` / `load_words` / `planarize` | extract 16-bit windows: strided selects + funnel shift; circular previous word built in registers; K=6 words de-interleaved | planarize took lm_head 235→555 GB/s |
| `fwht128` | unnormalised Sylvester butterflies, strides 1..64, on `simd<float,128>`, then ·1/√128 | |
| `HadInKernel` | `xh = fp16(H(fp16(x·suh))/√128)` | **rounds `x·suh` to fp16 before the FWHT** (like exllamav3 CUDA; **unlike TensorFold**) |
| `HadOutKernel` | sums split-K partials p=0..P−1 in order (fp32), FWHT, ·svh | fixed order |
| `GemvKernel<K,CB,MR,NT>` | vector "dp4a" GEMV, M ≤ 2 | inner products accumulate in **fp16** (`hacc`), then fp32 |
| `DpasKernel<K,CB,MB,NT>` | XMX GEMM, M 3..128 in row blocks MB ∈ {8,16,24,32,40,48,64}. One 16×16 EXL3 tile = one DPAS B operand (K16×N16 VNNI), decoded straight into VNNI. `xmx::dpas<8,RC,float,float,fp16,fp16>` with fp32 accumulators | `build_k4` fast path; `EXL3_K4_HALVES` gives wrong output (rejected); `EXL3_FOLD` is not bit-exact |
| `ReconstructKernel` | bit-exact fp16 `W_inner`, or int8 for the W8A8 prefill | |
| prefill (`exl3_linear`, M > 128) | had_in → reconstruct fp16 slices of 16384 cols → `at::matmul_out` (oneDNN) → **fp16 y** → had_out; or int8 W8A8 via oneDNN (NLL +0.17%) | **not row- or chunk-invariant** |
| attention | oneDNN Graph SDPA (80–90 TF at head_dim 256); an experimental ESIMD FA (`fa_esimd.h`, the only user of `lsc_load_2d`), slower than FA2 | not EXL3 |

The EXL3 kernels use 1-D `block_load` plus optional prefetch, not 2D block loads.

### 2.5 How exactness is proven there
- `exl3xpu/ref.py` is the pure-PyTorch spec. `tests/oracle_cuda.py` shows `ref.reconstruct_inner` is identical to
  exllamav3 on CUDA.
- Gate A1 (`tests/test_bitexact_xpu.py`):
  - ESIMD reconstruct == ref for all 401 tensors.
  - One-hot activation rows make every GEMM output an exact copy of a decoded weight, for the vector and DPAS paths.
  - **This pins weights, not accumulation order.**
- Gate A3: end-to-end logits against exllamav3 give 99.63% top-1 agreement, KL 9.8e-5.

### 2.6 Measured on one B70 (Qwen3.8-27B EXL3 4.00bpw, lm_head 6bpw)
- **Linear budget** (all 257 linears, graph-captured):

  | Rows (M) | Time | Notes |
  |---|---|---|
  | 1 | 27.2 ms | about 481 GB/s |
  | 4 | 31.3 ms | |
  | 16 | 34.6 ms | |
  | 64 | 67–69 ms | about 52 TFLOPS |

- **lm_head:** 555 GB/s. **DRAM ceiling with decode stripped:** 531 GB/s. Decode is near the integer-ALU roofline.
- **Decode with MTP k=3** (aggregate tok/s, prose): C1 91.2, C4 262.6, C16 365.2. Without MTP: C1 28.5.
- **Prefill** (int8 + oneDNN attention): 4K 2421 tok/s, 128K 1415 tok/s.
- **Baseline:** llama.cpp SYCL Q4_K_M gives C1 25.0 and prefill 4K 999.

### 2.7 Lessons from the lab, which apply to us
- **PCIe link drops**, not driver bugs, explained most "crashes": AER timeouts, then `pciehp Link Down`, then an xe
  re-probe and a renumbered render node. Both lab cards flapped.
  - The harness must detect a vanished or renumbered device and stop the queue.
  - Use `/dev/dri/by-path` rather than `renderD*` numbers.
- **The xe job timeout (5 s)** gives `UR DEVICE_LOST`. This happens during graph replay, and when another process starts
  on the same card. Serialise GPU jobs, and keep each kernel launch well under 5 s.
- **XPU graphs:** only full-decode capture worked; piecewise capture ran out of resources.
- **AOT link failure** for multi-kernel modules. Our K0 build plan assumes AOT (`intel_gpu_bmg_g31`). **Verify AOT on
  our box first**, and keep a JIT `spir64` fallback in `xpu/build.py`.
- **`-ffast-math`** is in exl3xpu's flags. **We must not copy it**: our contract needs `-fp-model=precise
  -ffp-contract=off`. Recheck that the codebook's fp16 mad still decodes bit-exactly without fast-math (it should,
  since it is an explicit `__hfma`-equivalent).
- Power: PL2 throttling to about 2600 MHz at 230 W. Use clock-stable medians in kbench.

---

## 3. TensorFold's EXL3 contract (what an XPU port must preserve)

**Formats:**
- Codebooks 3inst/mcg/mul1.
- Widths 1–8 including half-bit mul1 (1.5–3.5), **per tensor**, read from the trellis shape.
- Scales `suh`/`svh`, or packed sign words `su`/`sv`. Optional bias.
- Header-only `scan()` and `require_config()` refuse a checkpoint before download.

**Representation:** `Exl3Linear` with fields
- `words` int32, in layout `"strips"` `[N/128, K/16, 8, 8·bits]` (the default) or `"stored"` `[K/16, N/16, 8·bits]`
  (= exl3xpu's order);
- `suh`, `svh`, `bias`, `bits`, `codebook`;
- `split=(SK,WK)` from `plan(k,n)` (**shape only**), plus lazy `counters`.

**Decode / verify linear** (`linear.cu`):
1. `rot_in`: `xh = fp16(FWHT(fp32(x)·suh)·1/√128)`. **No fp16 rounding of `x·suh`.**
2. `linear_kernel`:
   - grid `(N/128, SK)`, WK warps, each warp owning a fixed k range;
   - rows in 16-row passes;
   - warp sums go through shared memory in warp order;
   - split partials go to `Z[SK,M,N]`, summed q=0..SK−1 in order by the last-arriving block (the counter only elects);
   - `finish`: FWHT, `·HAD_SCALE·svh`, `+bias`, output in fp16/bf16/fp32.
3. Result: a row's bits are independent of M (1..128) and of the window's composition. Above 128 rows, 128-row slices
   keep the bits.

**Prompt GEMM** (`cuda/exl3/prefill.py`):
- `rot_in`, then unpack W_q to fp16 once per chunk.
- Then Triton `_gemm` with fixed tiles (BM 128, BK 32, 8 warps, 4 stages), fp16×fp16→fp32.
- The epilogue rotates each 128-col block as `dot(rest_bf16,H) + dot(top_bf16,H)` (the hi/lo bf16 split of the fp32
  accumulator), then `·svh + bias`.
- Chunk-invariant. **Prefill bits ≠ decode bits** by design; engines re-prefill the reply.

**Experts** (Flash Next, GLM):
- `group_kernel` (device grouping);
- `grouped_kernel` with a per-expert trellis pointer table and a uniform `switch(K2)` (mixed widths in one launch);
- `gateup_epilogue` (SwiGLU modes), `down_epilogue`, `combine`, `down_combine`.
- No host sync; graph-capturable.

**Plain (unquantised) matrices:** `b16_linear`, a row-invariant one-warp-per-output fixed-order FMA.

**Engine-facing `_ext()` signatures** (the XPU kernels must expose these):
```
tensorfold_exl3_linear_v3:
  rot_in(x[M,K] f16|bf16|f32, suh f16[K], xh f16[M,K])
  linear(xh f16[M,K], T i32, stride_k, stride_nb, svh f16[N], bias|None, y[M,N] f16|bf16|f32,
         Z f32|None, counters i32[>=8N/128], K2, cb, SK, WK)        # M in 1..128
  unpack(T i32, W f16[K,N], stride_k, stride_nb, K2, cb)
tensorfold_exl3_experts_v1: grouped, dequant, group, rot_in, gateup_epilogue, down_epilogue, combine, down_combine
tensorfold_qwen_b16_v1: b16_linear(x, w[N,K], bias|empty)
```

**Tests that pin bits** (rerun on XPU through the device fixture):
- `tests/cuda/test_exl3_linear.py`: 25 codebook×bits combos, decode == `fmt.unpack`, row invariance at M ∈
  {1,2,3,16,17,64,128} under every plan, and strips == stored.
- `tests/cuda/test_exl3_checkpoint.py`: every width of a real checkpoint versus exllamav3.
- `tests/cuda/test_qwen27_exl3.py`: plans keep rows independent, window == serial, drafted == serial, prompt chunking
  and resume.
- Experts: `test_exl3_experts.py`, `test_qwen4_exp_exl3.py`, `test_glm_exl3.py`.
- CPU: `tests/test_exl3_format.py`.

---

## 4. Gap analysis: exl3xpu vs what TensorFold needs

| TensorFold piece | Closest exl3xpu piece | Gap / what changes |
|---|---|---|
| `decode2`/`decode_lane` (mma B fragments) | `decode_cb_h` + `tile_states` + `DpasKernel::build/build_k4` (DPAS VNNI B) | Same math and tile permutation. W_q is bit-identical. **Add**: half-bit streams, K=1/7/8 (lift `V>=8`), all codebooks by default |
| `unpack` | `ReconstructKernel` | Equivalent; exl3xpu needs `n` a multiple of 128 and the stored layout. Use TensorFold `layout="stored"` on XPU (or teach DPAS strips) |
| `rot_in` | `HadInKernel` | **Numerics differ** (fp16 pre-rounding of `x·suh`). Use TensorFold's contract (no pre-round) on XPU. Add fp32 input, bias, and fused-shard `suh[S,K]` only if needed |
| `linear_kernel` | `DpasKernel` + `HadOutKernel` (+ `GemvKernel`) | **exl3xpu is deterministic but NOT row-invariant**, for three reasons: (1) M ≤ 2 uses the fp16-accumulating vector GEMV and M ≥ 3 uses DPAS; (2) NT, P and rows-per-split depend on M (K=N=5120 gives P=26 at M=1, 13 at M=2, 18 at M=8/16, 7 at M=24, 13 at M=64); (3) M > 128 switches to oneDNN or int8. **Fix**: one kernel for all M ≤ 128 (DPAS with a fixed MB, looping row blocks), `(SK,WK)`-equivalent split from `plan(k,n)` only, fixed in-order sums, the bias epilogue, no env-driven knobs on exactness paths |
| `prefill.matmul` | `exl3_linear` M > 128 (oneDNN fp16, fp16 y; or int8) | Not chunk-invariant, and loses precision (fp16 y). **Replace** with a fixed-tile GEMM (Triton-XPU first, then an ESIMD DPAS) plus the hi/lo bf16 Hadamard epilogue. Never oneDNN or int8 on the exact path |
| experts (`group`, `grouped`, epilogues, combine) | **none** (exl3xpu has no MoE) | Write from scratch on top of the `DpasKernel` tile decode; per-expert pointer table plus a uniform K2 switch |
| `b16_linear` | none | Reuse the WS4 bf16 GEMV (K4 T0 bf16 path) |
| Flash Next Triton helpers (`_f16_mm`, `_ple_rows`, …) | none | WS3-style Triton port, only when Flash Next is targeted |

**Cross-device parity:**
- Only decoded weights (W_q) can be bit-identical between CUDA and XPU; both are already proven against exllamav3.
- GEMM outputs will differ (mma vs dpas internal summation). The XPU target is:
  - row-invariant on XPU;
  - decode == reconstruct == exllamav3;
  - tolerance against TensorFold's float64 reference (`test_matches_the_float64_reference`).

---

## 5. Porting plan (WS10)

**Owner:** an EXL3 kernel agent (K-EXL3), plus Engine-A for wiring. **Branches:** `xpu/exl3/<topic>` from `xpu/main`.
**Files:** `src/tensorfold/xpu/kernels/exl3/**`, `docs/xpu/kernels/exl3_*.md`, `tests/cuda/*exl3*` via the device
fixture.

**Phase E0: prerequisites** (none of this is EXL3-specific)
- WS1 device layer, the WS2 harness and the K0 native build have landed, with **AOT and JIT both validated** on the
  box.
- Lab lesson: AOT may fail for multi-kernel modules. Splitting per extension or a JIT fallback is decided in K0.

**Phase E1: format and loader on XPU**
- Enable `exl3` in `QUANT_METHODS["xpu"]` for `qwen3_5` only.
- Reuse `cuda/exl3/format.py` (`scan`, `require_config`) unchanged.
- `Exl3Linear` on XPU uses `layout="stored"` at first, matching the exl3xpu DPAS decode. Strips come later if
  measurements favour it.
- Support integer K 2–6 with mul1 (what the target checkpoints use) first. Then the other codebooks, half-bit streams,
  and K=7/8.
- Tests: `tests/test_exl3_format.py` unchanged. Host-side checks that XPU refuses unsupported widths and codebooks by
  name.

**Phase E2: decode and reconstruct (bit-exact weights)**
- Port `decode_cb_h`, `tile_states`, `load_words`, `planarize` and `ReconstructKernel` from exl3xpu, with MIT
  attribution, into `xpu/kernels/exl3/decode.hpp`. Compile with our flags (no `-ffast-math`).
- Expose `unpack(...)` with TensorFold's signature.
- **Gate:** `test_exl3_linear.py` "decode == `fmt.unpack`" for every supported combo, plus `test_exl3_checkpoint.py`
  (all widths of a real checkpoint == exllamav3 reconstruct).

**Phase E3: row-invariant decode / verify linear**
- `rot_in` with TensorFold numerics (no fp16 pre-round) as an ESIMD FWHT.
- `linear`:
  - ESIMD DPAS GEMM from `DpasKernel` with **one MB for all M ≤ 128** (MB=16 first: TensorFold windows are ≤ 16
    rows; larger M loops 16-row blocks);
  - split-K and threads from a shape-only `plan_xpu(k,n)` (per-shape table, like `exl3_load.PLANS`);
  - fp32 partials into `Z`, summed in fixed order;
  - finish: FWHT, ·svh, +bias, output dtype.
- Drop the fp16 vector GEMV. Bench an fp32-accumulating M=1 variant only if it is *also* used for every M (it must be
  the same arithmetic).
- **Gates:**
  - row invariance at M ∈ {1,2,3,16,17,64,128} under every plan;
  - 20 repeats, warm and cold;
  - tolerance against the float64 reference.
- **Perf target:** M=1 linear budget ≤ 32 ms for 27B 4.00bpw, which is about 85% of exl3xpu's 27.2 ms. Some loss is
  expected from fixed MB and fixed split.

**Phase E4: prompt GEMM**
- T0: port `cuda/exl3/prefill.py` `_gemm` to Triton-XPU (fixed tiles, the hi/lo bf16 Hadamard epilogue), with `unpack`
  from E2.
- N1: ESIMD DPAS GEMM with the fused epilogue, if profiles warrant it. No oneDNN, no int8 on the exact path.
- **Gates:** chunked == one-shot; resumed == fresh. `test_qwen27_exl3.py` prompt tests.

**Phase E5: recipe A-EXL3**
- `turboderp/Qwen3.8-27B-exl3` branches `3.00bpw` and `4.00bpw`, with the DFlash2 drafter (`b16_linear` → the bf16
  GEMV).
- **Gates:** `test_qwen27_exl3.py` (window == serial, drafted == serial), `e2e:27b-exl3-smoke`, and the quality gate
  against BF16.
- Compare with exl3xpu's published numbers, noting that we are under exactness constraints.

**Phase E6 (optional): EXL3 experts**
- Only if an EXL3 MoE target appears: Nemotron has a small community 4bpw EXL3; Flash Next and GLM are upstream
  targets.
- Grouped DPAS GEMV over a per-expert pointer table with a uniform K2 switch, epilogues and combine.
- Tests: `test_exl3_experts.py`.

**Risks:**
- The integer-ALU decode cost means fixed-MB DPAS may sit further from the roofline than exl3xpu's M-adaptive dispatch.
- AOT build failures.
- Graph capture limits.
- The B70 host's PCIe instability (lab) makes long benchmarks flaky; the runner must survive device renumbering.

---

## 6. Where to look in the lab code

| Topic | Path |
|---|---|
| kernels | `exl3xpu-b70-lab/csrc/exl3_esimd.h` (decode `:40-115`, states `:117-205`, FWHT/had `:210-338`, GEMV `:444-580`, DPAS `:592-862`, reconstruct `:869-936`) |
| launch, dispatch, split choice | `csrc/exl3_ops.sycl` (`:174-318`, `:644-767`) |
| spec / reference | `exl3xpu/ref.py`; `tests/oracle_cuda.py`; `tests/test_bitexact_xpu.py` |
| vLLM integration | `exl3xpu/vllm_plugin.py`, `vllm_patches.py`, `ops.py` |
| build | `scripts/build_ext.sh`, `tests/icpx_probe.sh`, `docker/Dockerfile` |
| perf and stability history | `docs/PROGRESS.md`, `docs/DESIGN.md`, `.superpowers/sdd/*/progress.md` |
