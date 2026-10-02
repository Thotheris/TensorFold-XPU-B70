# EXL3 on the XPU backend (survey, loader, kernels)

Status: **approved in scope by the repo owner; no code yet.** The owner reversed the "EXL3 refused on XPU" rule; AGENTS.md,
QUANT_FORMATS.md and PORT_PLAN.md now say EXL3 is opt-in (§1 records the remaining inputs). Every hardware fact here is
`[UNVERIFIED]` until a B70 bundle says otherwise. No code was run for this document; it is read from the CUDA sources.

Three parts, in the order they would be done: **§2 survey** (what exists), **§3 loader** (what to read), **§4 kernels**
(what to run), then **§5 plan** and **§6 risks**.

---

## 1. Decision gate

EXL3 conflicts with the current port decision, so the first step is not code:

| Question | Why it matters |
|---|---|
| Is EXL3 in scope for the B70 at all? | AGENTS.md §1/§6 and PORT_PLAN say "XPU reads only sym INT4"; the Qwen3.8-27B recipe A and Nemotron recipe B are defined on AutoRound checkpoints. |
| Which families? | CUDA EXL3 is wired into `qwen3_5` (recipe A), `qwen4_exp` and `glm5_next`. Recipe B (`nemotron_h`) has no EXL3 path today. |
| Priority vs K1–K5? | EXL3 reuses recipe A's GDN/attention kernels (K1, K6). It only replaces K4/K5 (the 4-bit matmuls), so it is additive, not a rewrite. |

Recommended: keep EXL3 **opt-in and behind its own flag** (`TF_XPU_EXL3=1` or a family `QUANT_METHODS["xpu"]` entry), do it
after M-A1 (recipe A running on INT4), and change AGENTS.md §1/§6 and QUANT_FORMATS.md in the same PR that adds the loader.
Why bother: EXL3 reaches better quality per bit than INT4 at 2 to 4 bits, which matters on a 32 GB card.

---

## 2. Survey: what the CUDA EXL3 stack does

Files (all under `src/tensorfold/`):

| File | Role | XPU reuse |
|---|---|---|
| `cuda/exl3/format.py` | header-only metadata, numpy reference decoder, `Exl3Tensor`, `require_config` | **as is**: numpy, device-free |
| `cuda/exl3/inspect.py` | `python -m tensorfold.cuda.exl3.inspect MODEL_DIR` | as is |
| `cuda/exl3/decode.cuh` | tile decode into `mma.m16n8k16` B fragments, `fwht128` | port (§4) |
| `cuda/exl3/linear.{py,cu,cpp}` | `Exl3Linear`, 1 to 128 rows, row-invariant, no cuBLAS | port |
| `cuda/exl3/prefill.py` | unpack W_q to fp16 once per call, then Triton GEMM with a Hadamard epilogue | port; Triton already |
| `cuda/exl3/experts*.{py,cu,cpp,cuh}` | grouped MoE experts, 3 codebook instantiations | out of scope (recipe B has no EXL3) |
| `families/qwen3_5/cuda/exl3_load.py` | recipe A loader | adapt (§3) |
| `families/qwen4_exp/cuda/exl3*.py`, `glm5_next` | other families' own readers | not needed for A/B |

### 2.1 The format

A layer `W [K, N]` (K and N multiples of 128) is a tensor group under one prefix:

- `trellis` int16 `[K/16, N/16, 16·bits]`, one 16×16 tile per `[k_tile, n_tile]`. A tile is 256 values in a **circular
  bitstream**; value `p` is a 16-bit window ending at `stream_ends(bits)[p]`. Widths: 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8
  (x.5 needs `mul1`).
- `suh` fp16 `[K]` and `svh` fp16 `[N]`, or packed sign words `su` / `sv` (older checkpoints), optional fp16 `bias`.
- A zero-size marker tensor `mcg` or `mul1` names the codebook; none means `3inst`.
- A codebook maps a 16-bit state to an fp16 value:
  - `3inst`: `x = s·89226354 + 64248484`, `mcg`: `x = s·0xCBAC1FED` (u32 wrap), then `x = (x & 0x8FFF8FFF) ^ 0x3B603B60`
    and the value is the fp16 sum of the low and high halves.
  - `mul1`: `x = s·0x83DCD12D`, `h = 1024 + (sum of x's four bytes)`, value `= fp16(h·fp16(0x1EEE) + fp16(0xC931))`.
- Value positions inside the tile follow the tensor-core permutation: lane `L` owns values `8L..8L+7`, at
  row `2(L%4) + (j&1) + 8((j>>1)&1)`, column `L/4 + 8(j>>2)` (`tile_positions()`).

The weight is `W = diag(suh) · H_K · W_q · H_N · diag(svh)` with `H` the 128-block Sylvester Hadamard divided by √128.

### 2.2 The CUDA arithmetic contract (what XPU has to mirror in structure)

`forward()` in `format.py` is the float64 reference. The CUDA kernels order it as:

1. **`rot_in`**: `xh = FWHT128(x · suh) / √128`, fp32 butterflies in a fixed order (`fwht128`: a radix-4 step per lane,
   then `__shfl_xor` for strides 1..16), result stored fp16.
2. **Matmul** `xh @ W_q`: fp16 inputs, **fp32 accumulate**, `mma.m16n8k16`; K split is `plan(k, n)` (shape only:
   `sk` K-splits, `wk` warps); `Exl3Linear.counters` orders the split reduction. [UNVERIFIED: I did not read the
   semaphore code in `linear.cu`; the kernel card must record exactly how split partials are combined.]
3. **Epilogue**: rotate the output by `H_N`, scale by `svh` and `1/√128`, add bias.

Prompt path (`prefill.py`): decode W_q once per call into an fp16 `[K, N]` workspace, then a Triton GEMM with fixed tiles
(`tiles()` returns `128, 32, 8, 4, 8` for every shape) whose epilogue applies `H` as two bf16 `tl.dot`s over a
`top + rest` split of the fp32 accumulator (so the rotation keeps near-fp32 precision on a bf16 MMA).

Row invariance there comes from "the shape's alone" tile constants, the same rule as AGENTS.md §5.5.

### 2.3 Gaps against the XPU rules

| CUDA code | XPU issue |
|---|---|
| `mma.sync m16n8k16` | DPAS is 8×16×16, SG16. B-fragment ownership differs, so the tile decode must be re-mapped. |
| `__dp4a`, `__byte_perm`, `__funnelshift_l` | no direct SYCL/Triton builtin; they become shifts, masks and integer adds. All exact. |
| `__shfl_xor_sync` over 32 lanes | SG16 on XPU: 8 values per lane for 128 elements, so the butterfly tree changes. |
| `__hfma2` / `__hadd2` fp16 | fp16 rounding must match `format.codebook()` (one rounding). Needs an exhaustive check. |
| bf16 `top/rest` Hadamard in prefill | Triton/XPU bf16 `tl.dot` fine, but `BLOCK_M=16` miscompile and int→bf16 cast crash apply (Triton guide §1.5). |
| Workspace of W_q fp16 `[K,N]` | single allocation must be < 4 GB; a 27B layer is far below, but cap it. |

---

## 3. Loader (WS3b, host-side testable)

Goal: `--backend xpu` accepts an EXL3 checkpoint for recipe A and puts each layer in an `Exl3Linear`-shaped object on `xpu`.
Most of this runs on any machine with torch; no GPU needed to test it.

Work items:

1. **Policy**. `families/__init__.py`: the EXL3 branch is `if backend == "cuda" and method == EXL3_QUANT ...`. Add the XPU
   backend to that check and `QUANT_METHODS["xpu"] = ("auto_round", "gptq", "compressed-tensors", "exl3")` for `qwen3_5` only.
   Infra owns `families/__init__.py`; Loader owns the `QUANT_METHODS["xpu"]` entries (AGENTS.md §7), so this is two
   branches or an issue labelled `ws:infra`.
2. **Reader**. New `src/tensorfold/xpu/quant/exl3.py`, importing `tensorfold.cuda.exl3.format` unchanged (do not edit it).
   Responsibilities: scan the safetensors headers (`format` already does), build one `Exl3Linear`-equivalent per prefix,
   keep trellis words as int32 views, **pre-pack into "strips" layout** `[N/128, K/16, 8, 8·bits]` (CUDA's default, a copy
   on host), and move to `xpu` in chunks.
3. **Refusals**. Anything outside the supported set errors with the layer prefix: K or N not a multiple of 128, bits
   outside `BITS`, x.5 bits without `mul1`, FP8 or int8 KV (already refused on XPU), a tensor group missing `trellis`/`svh`.
   `su`/`sv` sign words are expanded to fp16 on host once (`unpack_signs`) so kernels see only fp16 scales.
4. **Pointers and sizes**. Apply the `p - (1<<64) if p >= (1<<63)` fix wherever device pointers go into int64 tensors, and
   chunk uploads so no allocation reaches 4 GB.
5. **Capacity**. `cuda/capacity.py` already counts EXL3 bytes (`extra_files`, trellis bytes via `Exl3Tensor.trellis_bytes`);
   check it for CUDA-only assumptions rather than copying it.
6. **Quality gate**. The loader's gate compares a decoded layer against `format.dequantize` on a few layers per width and
   codebook. This is host-side numpy against the XPU Triton unpack (§4 E0) and is the first test that needs the B70.

Host-side tests (run here, in `tests/test_xpu_quant_exl3.py`): header scan on a tiny synthetic checkpoint written with
`format`'s own encoder if one exists (otherwise a fixture of random trellis words, which are valid for any bit pattern),
refusal messages, strips layout round trip, `su`→`svh` expansion, capacity accounting.

---

## 4. Kernels (WS4; new IDs E0–E3, owner paths `src/tensorfold/xpu/kernels/exl3/**`)

Follow AGENTS.md §6.3: **T0 correct Triton first**, then T1 tuned, native only if it beats T1 by ≥ 10% on a B70 bundle. Write
`docs/xpu/kernels/exl3_*.md` cards first (§11 template). Selection via `TF_XPU_KERNEL_EXL3=triton|native`.

### E0. Trellis unpack (`W_q` fp16 `[K, N]`)

The simplest kernel and the oracle for everything else. One program per 16×16 tile (or a few tiles), all integer math plus
one fp16 rounding.

- Arithmetic contract: the `states()` window extraction, the codebook formulas of §2.1, bitcast to fp16. Output must equal
  `format.unpack` **bit for bit** (`view(torch.int16).equal`), for every codebook and every width in `BITS`.
- Triton notes: u32 multiply wraps; `tl.cast(..., bitcast=True)` for fp16 reinterpret; avoid int/fp16→bf16 casts (go via
  fp32). `enable_fp_fusion=False`.
- Codebook rounding is the main risk: `mul1` is `fp16(h·scale + bias)` in one rounding; `3inst`/`mcg` are an fp16 add. Doing
  these in fp32 and rounding once can double-round. **Test exhaustively**: all 65536 states for each codebook against
  `format.codebook(name)` (a seconds-long test, gives certainty, no `[UNVERIFIED]` left).
- Roofline: reads `16·bits` bytes per 256 weights, writes 512 bytes. It is write-bound (about 0.5 B/weight out); fine, since
  prefill uses it once per call.

### E1. Input rotation (`rot_in`)

`xh = FWHT128(x · suh) / √128`, fp32, stored fp16.

- Contract: fixed butterfly tree. On CUDA each lane holds 4 values and shuffles strides 1..16 (`fwht128`). On XPU with SG16
  each lane holds 8, so the tree differs. That is allowed (§5.4 self-consistency), **but the XPU tree must not depend on M
  or on the program's row**: define it once, record it in the card, and pin it with a test against a fp64 reference to a stated
  tolerance plus a row-alone-vs-window bit test.
- Simplest T0: do the butterfly in registers inside one program per (row, 128-block) with `tl.reshape`/`tl.sum`-free
  ±add stages (7 explicit stages). Avoid `tl.dot` against a ±1 matrix here, since `tl.dot` order is not specified.
- `x`'s dtype may be bf16 or fp16; convert via fp32.

### E2. Decode matmul, 1 to 128 rows (the hot path)

CUDA fuses tile decode into the MMA loop (no W_q in memory). Port in two steps:

- **E2a (T0, correctness)**: Triton, one program per 128-column block × K split. Per K step: load the block's strip words
  (`[8, 8·bits]` words per k tile), decode 16×16 tiles in registers (E0 logic), `tl.dot` fp16 → fp32. Fixed `plan`-like
  `(SK, warps)` as a function of `(K, N)` only. Split partials reduced in fixed order via a scratch buffer, **never float
  atomics**.
- **E2b (T1/N0)**: tune; if Triton cannot keep the decode ALU hidden under the load, a native ESIMD kernel mirrors
  `decode.cuh`: each lane decodes the values it owns in the DPAS B layout and feeds `xmx::dpas<8, M, ...>`. B70 decode is
  bandwidth-bound at M=1 (about 0.5 B/weight at 4 bit); the decode ALU (about a dozen integer ops per 2 weights) should fit,
  [UNVERIFIED]. Measure achieved GB/s and compare with K4's INT4 GEMV on the same shapes.
- Arithmetic contract: accumulation order inside K step is whatever `tl.dot` compiles to, so record `threads_per_warp` and
  assert `#triton_intel_gpu.dpas` in the TTGIR. N must be ≥ 16 for DPAS (it is: 128-wide blocks).
- Epilogue: output rotation by `H_N` (use the same E1 butterfly or the `top/rest` split `tl.dot`; pick one and pin it),
  `· svh · 1/√128`, `+ bias`.
- Invariances: row alone == in any window of 1..128 rows == alongside other streams; 20 repeats, warm and cold cache.

### E3. Prompt matmul (any M)

Reuse `prefill.py`'s shape: E0 into a fixed fp16 workspace, then a fixed-tile Triton GEMM with the Hadamard epilogue
(`BM=128, BK=32, BN=128`, shape-independent). Chunk invariance: W_q is decoded identically every call, tile constants do not
depend on M, so chunked == one-shot should hold. It must be tested, not assumed.

- Watch the known Triton-XPU issues: `BLOCK_M=16` dot miscompile (not used; BM=128), long `static_range` abort (the K loop
  is a runtime `range`, but the guide says `while` loops crash; confirm `range` over a runtime bound is fine), and bf16 casts
  (the `top/rest` split casts fp32→bf16, which is a supported direction).
- Option: fold E0 into the GEMM to skip the W_q workspace if the 4 GB / memory budget matters. Not for T0.

### Not in scope (for now)

- EXL3 grouped MoE experts (`experts*.cu`): recipe B has no EXL3 checkpoint in the plan. Revisit with K3.
- 1 to 2.5-bit widths: supported by the format code but test only the widths the recipe A checkpoint uses first
  (`inspect` shows them); extend after.

### Test list (what pins which bits)

| Test | Where it runs | Pins |
|---|---|---|
| `states`/`unpack` vs fp64 `dequantize` | host (exists upstream: `tests/test_exl3_format.py`) | the format reference |
| exhaustive codebook, 65536 states × 3 | B70 | E0 fp16 rounding |
| E0 unpack == `format.unpack` for every `BITS` | B70 | E0 bits |
| E1 row-alone == window; vs fp64 within tolerance | B70 | rotation tree |
| E2 alone == window == with other streams, 20 repeats, warm and cold | B70 | row invariance |
| E3 chunked == one-shot; resumed == fresh | B70 | chunk invariance |
| E2 vs E3 on the same rows | B70 | **not** required equal (CUDA does not require it either); record the diff |
| `forward()` fp64 vs E2/E3 within tolerance | B70 | quality |
| drafted == serial `token_sha`, recipe A on an EXL3 checkpoint | B70 e2e | M-A2 equivalent |

---

## 5. Plan

| Step | Owner | Branch | Done when |
|---|---|---|---|
| 0 Decision + doc updates (AGENTS.md §1/§6, QUANT_FORMATS.md, this file) | repo owner, Docs | `xpu/docs/exl3` | owner says yes |
| 1 Kernel cards `exl3_unpack.md`, `exl3_rot.md`, `exl3_linear.md` | Kernel E | `xpu/e0/cards` | cards state contracts |
| 2 Loader + refusals + host tests | Loader | `xpu/loader/exl3` | host pytest and ruff green |
| 3 E0 Triton unpack + exhaustive codebook test | Kernel E | `xpu/e0/unpack` | green B70 bundle for the SHA |
| 4 E1 rot_in | Kernel E | `xpu/e1/rot` | green B70 bundle |
| 5 E2a decode matmul | Kernel E | `xpu/e2/linear` | invariance bundle, tolerance bundle |
| 6 E3 prompt matmul | Kernel E | `xpu/e3/prefill` | chunk-invariance bundle |
| 7 Engine-A wiring (`families/qwen3_5/**`, `exl3_load` XPU branch) | Engine-A | `xpu/engine-a/exl3` | e2e token_sha equality |
| 8 T1 tuning, N0/N1 only if ≥ 10% faster | Kernel E | per kernel | cards updated with GB/s |

Steps 2, 3 and 4 are independent and can run in parallel. Steps 5 to 7 are serial. CUDA behaviour stays unchanged
throughout: no edits to `cuda/exl3/*`, `.cu` files or CUDA launch constants; XPU code lives under `src/tensorfold/xpu/`.

## 6. Risks and open questions

- **Policy**: see §1. This is the largest risk; do not start code before it is answered.
- **Exactness of the fp16 codebook rounding** on Triton-XPU: resolved by the exhaustive test, but a failure forces native
  fp16 ops (a `tl` fp16 add) or an integer-only formulation.
- **FWHT tree** differs from CUDA, so XPU bits will not match CUDA bits (acceptable per §5.4), but quality must still pass.
- **Decode ALU on XPU**: if E2 is ALU-bound rather than bandwidth-bound, EXL3 will lose to INT4 on tok/s even though it wins on
  quality per bit. The card should record this honestly and the choice stays with the user.
- **Memory**: E3's fp16 W_q workspace is `K·N·2` bytes for the largest layer; check against the capacity gate.
- **Split-K combine** in `linear.cu` was not read in detail here; read it before writing the E2 card.
- **No fallback checkpoint**: a recipe A EXL3 checkpoint must be named (repo, revision, bits, codebook) before step 3 so the
  quality gate has real data. `python -m tensorfold.cuda.exl3.inspect` reports what a candidate holds.
