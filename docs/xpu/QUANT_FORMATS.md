# Weight Formats for the XPU Backend (W4A16 symmetric INT4)

**Decision:** on `--backend xpu`, TensorFold reads only symmetric INT4 weight-only checkpoints:
- AutoRound `auto_round:auto_gptq`
- GPTQ v1
- compressed-tensors `pack-quantized`

MLX, EXL3, NVFP4 and FP8 checkpoints are refused on XPU. EXL3 support is planned later as WS10; see
[EXL3_PORT.md](EXL3_PORT.md).

Compiled 2026-09-30 from public sources. Download counts and repo contents change; re-check before relying on them.

---

## 1. Why this format

- Intel's B-series inference stack is built around **symmetric INT4 W4A16 in GPTQ layout**:
  - vLLM XPU + `vllm-xpu-kernels` (`int4_gemm_w4a16` via oneDNN, and W4A16 MoE grouped GEMM).
  - llm-scaler: online `sym_int4`.
  - SGLang XPU.
  - Intel AutoRound: default scheme W4A16 g128 sym. The `auto_round` format is "recommended for CPU, Intel GPU, CUDA,
    HPU".
  - Sources: [vLLM #37979](https://github.com/vllm-project/vllm/issues/37979),
    [llm-scaler](https://github.com/intel/llm-scaler), [auto-round docs](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md),
    [int4_gemm_w4a16.h](https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/main/csrc/xpu/onednn/int4_gemm_w4a16.h).
- The B70's XMX has **no FP8 or FP4**, so NVFP4/FP8 checkpoints would run as software upconversion.
- Intel's own XPU constraints:
  - Dense W4A16: group size a multiple of 32; sym or asym; **no `g_idx`/act-order**.
  - MoE: **INT4 sym only**.
  - Source: [vllm xpu.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/kernels/linear/mixed_precision/xpu.py),
    [xpu_moe](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/experts/xpu_moe/).

### 1.1 The format fits TensorFold's existing kernel contract
TensorFold's 4-bit matmuls (`lane_matmul`, `qmm.cu`) compute, per input group `g`:
```
acc = fma(xs_g, b_g, fma(P_g, s_g, acc))      # P_g = Σ x·q over the group, xs_g = Σ x over the group
```
That is the affine dequant `w = s·q + b` folded into the dot product. Symmetric INT4 is `w = s·(q − 8) = s·q + b` with
**`b = −8·s`**:
- Multiplying by 8 is exact in fp16 and bf16, so the conversion is **bit-exact**. Every invariance property of the
  contract carries over.
- **XPU kernels never store `b`.** They take a `SYM` path and form `b = −8·s` in registers. This saves bandwidth and
  gives the same bits.
- **Keep fp16 scales as fp16.** fp16→bf16 conversion is lossy, at most 2⁻⁹ relative per group. Kernels are templated on
  `ScaleT ∈ {half, bf16}`.

| Source | Conversion to `s·q + b` | Exact? |
|---|---|---|
| sym GPTQ / AutoRound / compressed-tensors | `b = −8·s` | **yes** |
| asym with integer zero z | `b = −s·z` | rounds (≤ about 3% of a quant step, bf16). **Not supported on XPU yet** |
| MLX affine to the Intel zero-point form | `z = −b/s` | **no** (MLX biases are free floats) |

---

## 2. Tensor layouts

### 2.1 GPTQ / AutoRound (`auto_round:auto_gptq`, `gptq` v1)
Verified from safetensors headers of real checkpoints.

| Tensor | Shape, dtype | Meaning |
|---|---|---|
| `qweight` | `[K/8, N]` int32 | **Packed along K.** Row k sits in word `k//8` at bits `4·(k%8)`, low nibble first. q ∈ 0..15 |
| `scales` | `[K/gs, N]` fp16 (AutoRound, Qwen GPTQ) or bf16 (Nemotron GPTQ) | per group × output channel |
| `qzeros` | `[K/gs, N/8]` int32 | Packed along N; v1 stores **z−1**. Sym: z=8 stored as 7 → `0x77777777` |
| `g_idx` | `[K]` int32 (optional) | Must be the identity `k//gs`. XPU refuses anything else |

Dequant: `w[k,n] = s[k//gs, n] · (q[k,n] − z)`.

**Pitfalls:**
- `SergiioB/...-GPTQ-INT4-G64-sym` stores **qzeros = 0** while `sym: true`. **Derive z=8 from `sym`; never read qzeros
  for sym checkpoints.** A v1 reading would give z=1 and silently corrupt every weight. The loader asserts
  `qzeros ∈ {0x77777777, 0}` and records which.
- `checkpoint_format: gptq_v2` stores z directly. It is refused until tested.
- AutoRound's `packing_format` names include `auto_gptq`, `gptq`, `gptq_zp±1`. Check them per checkpoint.

### 2.2 compressed-tensors `pack-quantized` (llm-compressor, RedHatAI)

| Tensor | Shape, dtype | Meaning |
|---|---|---|
| `weight_packed` | `[N, K/8]` int32 | same nibble order, **N-major** (like MLX) |
| `weight_scale` | `[N, K/gs]` bf16 | |
| `weight_shape` | `[2]` int64 | `[N, K]` |
| `weight_zero_point` | absent for sym | |

Sym nibbles are offset-binary (unsigned = signed + 8); the histogram is centred at 8. Dequant: `w = s·(u − 8)`.

### 2.3 Layout moves
- GPTQ `[K/8, N]` is the transpose of compressed-tensors/MLX `[N, K/8]` with the same nibble order.
- T0 (Triton `lane_matmul`): the loader produces N-major `[N, K/8]` plus scales `[N, K/gs]` (or transposed as the
  kernel expects).
- N (native): the loader calls `pack_xpu` (defined by kernel cards K4/K3) to produce 2D-block-load-friendly tiles.
- Repacking streams one tensor at a time. No single staging allocation may reach 4 GB.

### 2.4 Unquantized modules
Honour these exactly:
- AutoRound `extra_config`
- GPTQ `modules_to_not_convert` (regex, e.g. `"-:.*mtp.*"`)
- compressed-tensors `ignore`

Tensors listed there are BF16 and use the bf16 dense path. Typical cases:
- Qwen: `embed_tokens`, `lm_head`, `visual.*`, `linear_attn.in_proj_a/b`, MTP `fc` and norms.
- Nemotron: `embeddings`, `lm_head`, router `mixer.gate*`, norms.

---

## 3. Target checkpoints

### 3.1 Recipe A: Qwen3.8-27B
Architecture: 64 layers (3 GDN : 1 full attention), hidden 5120, 4 KV heads × 256, FFN 17408, vocab 248320, untied
embeddings, 1 MTP layer, vision tower. Apache-2.0.

| Role | Repo | Format | Notes |
|---|---|---|---|
| **Primary** | [`devan-carlin/Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound) | `auto_round:auto_gptq`, **g128 sym**, fp16 scales, qzeros 0x77 | embed/lm_head/vision/`in_proj_a,b` BF16; MTP present (mlp/attn int4, `fc` BF16); 19.0 GB. Same recipe as [Intel/Qwen3.6-27B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound) |
| Secondary | [`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | compressed-tensors pack-quantized, g128 sym, bf16 scales | AWQ smoothing + GPTQ; MTP BF16 in a separate `model_mtp.safetensors`; 18.6 GB main file; 575k downloads |
| Other | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` | GPTQ g128 sym | BF16 MTP; 19.6 GB |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | BF16, 1.92B params, 3.85 GB | TensorFold re-quantizes draft weights at load; on XPU to **SYM** int4 |

Memory [derived]: weights about 19 GB, plus drafter about 1 GB (re-quantized), plus KV 64 KiB/token in bf16 (16
attention layers × 4 × 256 × 2 × 2 B). 64k context ≈ 4 GB, so it fits 32 GB.

**BF16 lm_head is 2.5 GB read per token** (about 18% extra decode traffic). An optional `--xpu-head-int8` re-quant is a
later, opt-in quality trade.

### 3.2 Recipe B: Nemotron 3.5 Lightning 30B-A3B
Architecture: 52 layers (23 Mamba-2, 23 MoE, 6 attention), hidden 2688, vocab 131072.
- MoE: 128 routed experts, top-6, plus 1 shared expert. Expert intermediate is 1856; shared is 3712 = 2 × 1856, which
  TensorFold folds as 2 extra experts.
- Experts are relu² up/down with no gate. Routed scaling is 2.5.
- 1 MTP layer (attention + MoE).
- Mamba: 64 heads × 64, state 128, 8 groups.
- License OpenMDW-1.1.

| Role | Repo | Format | Notes |
|---|---|---|---|
| **Primary** | [`letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound`](https://huggingface.co/letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound) | `auto_round:auto_gptq`, **g64 sym**, fp16 scales, qzeros 0x77 | backbone router/embed/lm_head/norms 16-bit; Mamba in/out_proj, experts, shared expert, attention int4. **MTP in `model_extra_tensors.safetensors`** (its router is int4). 18.8 GB. Single author, 210 downloads: validate quality |
| Secondary | [`SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym`](https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym) | GPTQ g64 sym, bf16 scales, **qzeros=0**, all-zero `g_idx` | **No MTP** (serve `--no-drafts`). Has published B70 numbers. 18.1 GB |
| Later drafter (WS9) | [`SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`](https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16) | BF16, about 1.67 GB | Reconstruction of NVIDIA's NVFP4 DFlash; 52% acceptance, 186.6 vs 87 tok/s on B70 under vLLM ([post](https://sergiiob.dev/posts/nemotron-35-lightning-dflash-arc-pro-b70/)) |

g64 is forced: 1856 = 29 × 64, which is not divisible by 128.

Memory: about 19 GB weights. KV is tiny (6 attention layers). Mamba state is about 50 MB per sequence.

Note: SergiioB reports native MTP acceptance of 0% on the vLLM XPU stack. Whether the cause is the stack or the
checkpoint is unknown. TensorFold's own MTP path plus the drafted == serial test will tell.

---

## 4. Loader requirements (WS3b, `src/tensorfold/xpu/quant/`)
1. **Header-only detection** before download: `quant_method`, `packing_format`/`checkpoint_format`, `bits=4`,
   `sym=true`, `group_size ∈ {64,128}`, `desc_act=false`. Refuse everything else by name: asym, act-order, gptq_v2,
   2/3/8-bit, mixed bits.
2. **Unquantized map** from `extra_config` / `modules_to_not_convert` / `ignore`.
3. z=8 from `sym`; qzeros only sanity-checked.
4. Stream and repack per tensor into the kernel layout (T0 N-major or `pack_xpu`), keeping fp16 scales.
5. Name mapping from HF to TensorFold `Weights`. Nemotron MTP comes from `model_extra_tensors.safetensors`. Stack the
   experts and fold the shared expert into 2 extra experts.
6. Tests:
   - synthetic round-trip, with dequant equal to `s·(q−8)` bit-exactly;
   - header fixtures from the real repos;
   - B70 load-only smoke with a memory report.
7. Quality gate: greedy agreement and perplexity on a fixed text set against BF16 (a small slice on the box, or
   published numbers).

## 5. Self-quantizing later (not first)
Use AutoRound `auto-round` W4A16 sym:
- **g128 for Qwen, g64 for Nemotron.**
- Export `auto_round:auto_gptq`.
- Keep 16-bit: norms, embeddings, router gates (including the MTP router), Qwen `in_proj_a/b`, vision, and optionally
  lm_head.
- No `desc_act`.
- Don't mix bit widths inside fused layers.

Reasons to do it: provenance, a calibration set that covers all 128 experts, and control over lm_head/MTP precision.

## 6. Links
1. https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/main/csrc/xpu/onednn/int4_gemm_w4a16.h
2. https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
3. https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/experts/xpu_moe/
4. https://github.com/vllm-project/vllm/issues/37979
5. https://github.com/intel/llm-scaler
6. https://github.com/intel/auto-round/blob/main/docs/step_by_step.md
7. https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound
8. https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4
9. https://huggingface.co/letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound
10. https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym
11. https://sergiiob.dev/posts/nemotron-35-lightning-dflash-arc-pro-b70/
12. https://github.com/steveseguin/b70-optimization-lab
13. https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound
14. https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
15. https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
