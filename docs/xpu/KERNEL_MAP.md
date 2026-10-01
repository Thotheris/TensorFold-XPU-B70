# Kernel Map: what recipes A and B run, and the XPU porting status

Recipes:
- **A:** Qwen3.8-27B W4A16 + DFlash2 (`families/qwen3_5`).
- **B:** Nemotron 3.5 Lightning 30B-A3B W4A16 + MTP (`families/nemotron_h`).

Paths are under `src/tensorfold/`. Port IDs (K0–K7) refer to [PORT_PLAN.md](PORT_PLAN.md) §4. The Status column is
kept up to date by the Analyst agent.

Kinds:
- **CUDA**: native nvcc extension, loaded via `cuda/build.load`. **Not available on XPU**; needs a T0 replacement.
- **Triton**: `@triton.jit`. Port and verify (see [TRITON_XPU_GUIDE.md](TRITON_XPU_GUIDE.md)).
- **torch**: plain torch ops.

---

## 1. Native CUDA extensions used by A/B

| Extension | Sources | A | B | XPU replacement | Port |
|---|---|---|---|---|---|
| `tensorfold_qmm_v3` | `cuda/kernels/qmm.{cpp,cu}`, `qmm_prefill.cu`, `qmm_prefill8.cu`, `qmm_frag.cuh` | yes | yes | decode: Triton `qwen3_5/cuda/qmm.py::lane_matmul` (bit-identical to CUDA on NVIDIA); prompt: new Triton dequant + `tl.dot`; FP8: refused | K4, K5 |
| `tensorfold_gdn_v2` | `cuda/kernels/gdn.{cpp,cu}`, `gdn_prefill.cu` | yes | – | **none, must write** (Triton T0 + SYCL spike) | K1 |
| `tensorfold_prefill_attention_v1` | `cuda/kernels/prefill_attention.{cpp,cu}` | yes (D=256) | yes (D=128) | Triton `prefill_attention.py::triton_attention` (it is the bit definition) | K6 |
| `tensorfold_experts_v7` | `cuda/experts.{cpp,cu,cuh}`, `experts_prefill.cu`, `experts_pack.cu` | – | yes | **none, must write**: torch plan, Triton grouped GEMV/GEMM | K3 |
| `tensorfold_nemotron_scan_rows` | `families/nemotron_h/cuda/scan_rows.{cpp,cu}` | – | yes (prompt) | **none, must write** (Triton chain scan + SYCL spike) | K2 |

Not needed for A/B: `cuda/nvfp4/*`, `cuda/exl3/*`, `qwen3_5/cuda/b16.*`, `glm5_next/cuda/*`, `qwen4_exp/cuda/*`,
`qwen3_5_moe/*`, `streaming/hostsync`.

### CUDA features those extensions use (what each port must replace)

| Feature | Where | XPU equivalent |
|---|---|---|
| `mma.sync.m16n8k16 bf16→f32` | `qmm_frag.cuh`, `experts.cuh`, `prefill_attention.cu` | DPAS 8×16×16 (`joint_matrix`, ESIMD `xmx::dpas`) or Triton `tl.dot` |
| `mma.sync.m16n8k32 e4m3` | `qmm_prefill8.cu` | none (FP8 refused) |
| `ldmatrix.x4(.trans)` | `qmm_frag.cuh`, `prefill_attention.cu` | 2D block load (transpose/VNNI) |
| `cp.async` (+zfill) | `qmm_frag.cuh`, `experts.cuh`, `prefill_attention.cu` | `prefetch_2d` + direct loads |
| `sub.rn.bf16x2`, `fma.rn.bf16x2` nibble decode `(0x4300\|q)−128` | `qmm_frag.cuh::pair` | the same bit trick in SYCL or Triton |
| `ex2.approx`, `div.full` (matching Triton numerics) | `prefill_attention.cu` | n/a (Triton is used directly) |
| `ld.global.nc.L1::no_allocate` | `experts.cu::ld_w` | LSC cache hints |
| clusters + DSMEM split-K reduce | `qmm.cu` (sm_90+) | second-pass `reduce` kernel (already exists) |
| `__shfl_xor_sync`, 32-lane maps (gdn: lane×4=128=dk) | gdn, scan_rows, experts, attention | SG16 `permute_group_by_xor`, re-mapped (16 lanes × 8 = 128) |
| `__match_any_sync`, `__popc`, `__shfl_up_sync` | experts plan | torch `argsort`/`bincount`/`cumsum` (K3a) |
| > 48 KB dynamic smem | qmm_prefill tile 0 (63 KB), prefill_attention (96 KB at D=256) | ≤ 128 KB SLM per work-group |
| `--fmad=false` | gdn, prefill_attention, scan_rows | `-fp-model=precise -ffp-contract=off`; Triton `enable_fp_fusion=False` |
| CUDA graphs | Nemotron `engine.py`, `mtp.py` | eager first; `torch.xpu.XPUGraph` later, gated on bitwise equality |

---

## 2. Recipe A: Qwen3.8-27B

Shapes: hidden 5120; 64 layers (48 GDN + 16 attention); attention 24 heads / 4 KV heads, D=256, rope 64; GDN 16 key
heads, 48 value heads, dk=dv=128, conv 4 (conv dim 10240); FFN 17408; vocab 248320. Draft (DFlash2): 5 layers, 32/8
heads, D=128. **No CUDA graphs.** KV is bf16 `(rows,4,256)` per attention layer. GDN state is fp32 `(48,128,128)` plus a
bf16 conv tail `(3,10240)`.

### Prefill (`qwen3_5/cuda/prefill.py::prefill_chunk`, chunks ≤ 4096)

| Op | Kernel | Kind | XPU | Port | Status |
|---|---|---|---|---|---|
| embed | `glue._embed` | Triton | port | WS3 | |
| add+RMSNorm+xs | `glue._add_rmsnorm` | Triton | port | WS3 | |
| all projections | `qmm_prefill.cu::prefill_kernel` (tile 9) | CUDA | Triton dequant GEMM → SYCL | K5 | |
| GDN conv/gates | `glue._gdn_pre` | Triton | port | WS3 | |
| GDN chain | `gdn_prefill.cu::chain_kernel` | CUDA | **new** | K1 | |
| gated norm | `glue._gated_norm` | Triton | port | WS3 | |
| q/k norm + rope | `glue._attn_prep` | Triton | port (host cos/sin) | WS3 | |
| KV write | slice assign | torch | – | – | |
| prompt attention | `prefill_attention.cu::pattn_kernel<256>` | CUDA | `triton_attention` | K6 | |
| gate mul / swiglu | `glue._gate_mul`, `_swiglu` | Triton | port | WS3 | |
| head (last row) | `qmm.cu::qmm_kernel` → BF16 lm_head on XPU | CUDA | bf16 GEMV (Triton → SYCL) | K4 | |
| DFlash2 taps | see the drafter table | mixed | | | |
| sampling | `cuda/sampling.py` | torch | port | WS1 | |

### Decode / verify window (`forward.tree_forward`, `decode.serial_decode`, `commit`)

| Op | Kernel | Kind | XPU | Port |
|---|---|---|---|---|
| projections + head | `qmm.cu::qmm_kernel` BM=16 (+`reduce_kernel`) | CUDA | `lane_matmul` (SYM, g128, fp16 scales) → SYCL GEMV | K4 |
| group sums | `kernels/qmm.py::_group_sums` | Triton | port | WS3 |
| glue | `_embed`, `_add_rmsnorm`, `_gdn_pre`, `_gated_norm`, `_attn_prep`, `_gate_mul`, `_swiglu` | Triton | port | WS3 |
| GDN tree step | `gdn.cu::tree_kernel` | CUDA | **new** | K1 |
| tree attention | `kernels/attention.py::_paths/_shared/_tail/_merge` | Triton | port (`_paths` while→for; base+offset probe) | WS3 / K7 |
| GDN commit replay (all 48 layers, one launch) | `gdn.cu::replay_kernel` | CUDA | **new** | K1 |
| conv-tail / KV commit | `index_select`, `cat`, `_foreach_copy_` | torch | – | – |

### DFlash2 drafter (`qwen3_5/cuda/dflash2.py`, `fast=True`)

| Op | Kernel | Kind | XPU |
|---|---|---|---|
| re-quantize draft weights at load | `quantize4` → on XPU: SYM int4 g64 | torch | WS3b/K4 |
| linears | `qmm_fast.matmul` / `F.linear` (small) | CUDA/torch | K4 T0 path |
| q/k norm + rope | `_prep_kernel` | Triton | port |
| dynamic conv | `_dconv_kernel` | Triton | port |
| block attention | `draft_attention._block_attention` (**int64 pointer table**) | Triton | port + pointer probe |
| context append | `draft_attention._append` | Triton | port |
| draft head | `qmm_fast.matmul_rows` over head row spans | CUDA | K4 (bf16 head) |
| top-k | `torch.topk(16)` → host `draft_tree.best_first` | torch | – |

### Sampling (`cuda/sampling.py`)
- Greedy is argmax.
- top-k uses `torch.topk`, then host `exact_sampling.choose_rows` (fp64 numpy).
- top_k=0 uses a **GPU fp64** nucleus. On XPU, measure fp64 first; fall back to CPU fp64 if it is slow (exact either
  way).

---

## 3. Recipe B: Nemotron 3.5 Lightning 30B-A3B

Shapes:
- Hidden 2688; vocab 131072.
- Mamba-2: 64 heads × 64, 8 groups, state 128, conv 4 (conv dim 6144, in_proj width 10304).
- Attention: 32 heads / 2 KV heads, D=128, **no RoPE**.
- MoE: 128 routed experts, top-6, intermediate 1856, relu²; the shared expert (3712) is folded as 2 extra experts, so 8
  slots per token. Sigmoid router with score-correction bias, ×2.5.
- MTP head: 1 attention block + 1 MoE block.
- **CUDA graphs** per (rows 1..16 × sampling mode). These are off on XPU at first.

### Prefill (`Engine.prefill_chunk`, chunks of 2048)

| Op | Kernel | Kind | XPU | Port |
|---|---|---|---|---|
| embed / norms | `qwen3_5 glue._embed`, `_add_rmsnorm`, `nemotron glue._add_moe_norm` | Triton | port | WS3 |
| dense projections | `qmm_prefill.cu` tile 0 via `glue.prefill_dense` | CUDA | K5 T0 | K5 |
| conv rows / commit | `mamba._conv_rows`, `_conv_commit` (**`debug_barrier`** → double-buffer) | Triton | port | WS3 |
| SSM prompt scan | `scan_rows.cu::scan_kernel` | CUDA | **new** | K2 |
| group RMSNorm | `mamba._group_rmsnorm` | Triton | port | WS3 |
| KV write | `attention._kv_write` | Triton | port | WS3 |
| prompt attention | `prefill_attention.cu::pattn_kernel<128>` | CUDA | `triton_attention` | K6 |
| router + top-k | `glue._router`, `_topk` | Triton | port (check DPAS at N=16) | WS3 |
| expert plan | `experts.cu::plan_rank/offsets/scatter` | CUDA | torch argsort plan | K3a |
| grouped expert GEMM (up relu², down) | `experts_prefill.cu::prefill_kernel` | CUDA | **new** Triton grouped GEMM | K3d |
| MTP absorb | `_embed`, `_concat_norms`, `prefill_dense`, `_add_rmsnorm`, `_kv_write` | mixed | as above | |

### Decode / verify (`Engine._forward(rows)`, rows ≤ 16)

| Op | Kernel | Kind | XPU | Port |
|---|---|---|---|---|
| dense projections + head | `qmm.cu::qmm_kernel` via `glue.dense` | CUDA | K4 T0 → SYCL; head is BF16 → bf16 GEMV | K4 |
| conv (replay kept rows + window) | `mamba._conv` | Triton | port (**DEVICE_LOST pattern risk**) | WS3 |
| SSM scan (replay + window) | `mamba._scan` (`SCAN_BD=32, SCAN_WARPS=8` are part of the arithmetic) | Triton | port (**DEVICE_LOST risk**) | WS3 / K2 |
| decode attention | `attention._kv_write/_chunk/_merge` | Triton | port (`_merge` 3D reshape) | WS3 |
| router + top-k | `glue._router`, `_topk` | Triton | port | WS3 |
| expert plan | `experts.cu::plan_kernel` | CUDA | torch plan | K3a |
| grouped experts | `experts.cu::expert_kernel` (register double-buffered `ld.global.nc`, in-kernel xs via quad shuffles) | CUDA | **new** Triton grouped GEMV | K3c |
| sampling | `sampler.sample` (`torch.topk` + Triton `_keyed` fp64/uint64; `nucleus` fp64 torch) | Triton/torch | port + fp64 check | WS3 |

### MTP drafting (`mtp.py::MTPHead._round`)
Same kernels as above. The draft head over the draft-id subset is built from `draft_ids.txt` by untile/index/tile, then
the head matmul. On XPU it is built from the BF16 head rows instead (no `tile()`). Commit is lazy: the next window's
`_conv`/`_scan` replays the kept rows.

---

## 4. Shared-memory and register budgets to re-check on Xe2 (max 128 KB SLM per work-group)

| Kernel | CUDA smem | Note |
|---|---|---|
| `qmm_kernel` | 17.7 / 26.1 / 43.0 KB (BM 16/32/64) | n/a on XPU (Triton/SYCL rewrite) |
| `qmm_prefill` | 42 KB (tile 9) / 63 KB (tile 0) | |
| `pattn` | 96 KB (D=256) / 64 KB (D=128) | Triton `_attend` instead: watch per-lane registers at D=256 → `grf_mode="256"` |
| `tree_kernel` | ≤ 32 KB + 12·rows | K1 SYCL: tree slots in SLM |
| `chain_kernel` | about 41 KB | |
| `scan_kernel` | about 21 KB | |
| experts `prefill_kernel` | 38 KB | |

---

## 5. Exactness tests that pin bits (run them on XPU via the device fixture)

| Area | Tests |
|---|---|
| 27B forward | `tests/cuda/test_qwen27_forward.py` (tree == serial, commit states, windows) |
| GDN | `tests/cuda/test_gdn.py` (tree == chain, replay, multi-stream, chunk invariance) |
| tree attention | `tests/cuda/test_attention.py` |
| 4-bit matmul | `tests/cuda/test_qmm.py`, `test_qwen27_qmm.py` (row-count invariance, tiled == untiled) |
| prompt | `tests/cuda/test_qwen27_prefill.py` (bf16 only on XPU), `test_prefill_attention.py`, `test_qwen27_prompt_end_cache.py` |
| multi-stream | `tests/cuda/test_qwen27_multi.py` |
| drafter | `tests/cuda/test_qwen27_dflash2_blocks.py`, `test_qwen27_draft_attention.py` |
| Nemotron | `tests/cuda/test_nemotron_forward.py` (windows == serial, drafted == serial, graphs == eager), `test_nemotron_kernels.py`, `test_experts.py`, `test_nemotron_prefix_end.py` |
| sampling | `tests/cuda/test_sampling.py`, `test_nemotron_forward.py::test_gpu_sampler_matches_host_rule` |

The pattern is `torch.equal` on bit views. Tolerance tests against `reference.py` fp32/fp64 are separate.
