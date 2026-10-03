# XPU port status

Kernel hand-off snapshot, 2026-10-03. Every number below is read from a B70 bundle on `origin/results`; refresh from
`origin/results:index.jsonl` before relying on it. No end-to-end recipe run exists yet. The
[execution prompt](prompts/kernel-engineer.txt), [plan](PORT_PLAN.md), [kernel map](KERNEL_MAP.md) and
[upstream atlas](reports/tensorfold-kernel-atlas-2026-10-02.html) describe the work; kernel cards are in
[kernels/](kernels/).

## How runs happen on this box

The B70 is the development machine. No runner service is installed: after each push the operator's agent runs
`python3 tools/xpu/b70_runner.py --once --image tensorfold-xpu:tc-349b7f516e06 --mode container --native-ext <dir>`
from the clean clone `~/.local/share/tensorfold-xpu/repo`, with `<dir>` from `tools/xpu/build_ext_image.sh`
(`TF_XPU_BUILD_IMAGE=tensorfold-xpu-build:dle-2026.1-81b58de60235`). Toolchain hash of every bundle below: 858a0a59.

## Verified evidence

| Commit | Bundle | Executed suites | Qualification |
|---|---|---|---|
| `c32b417` | `runs/xpu--main/c32b417-20261003T050422Z` | env, triton-smoke, unit-host, unit-xpu, kernels:glue, kernels:prefill-attention | pass; kbench GRF reporting |
| `da2a34f` | `runs/xpu--main/da2a34f-20261003T054251Z` | env, unit-host, unit-xpu, kernels:qmm | pass; K4.T0 |
| `44a106d` | `runs/xpu--main/44a106d-20261003T071529Z` | env, triton-smoke, unit-host, unit-xpu, kernels:glue, kernels:prefill-attention, kernels:qmm | pass; K4.T1, compare fix, launch-floor breakdown; qualifies `55d2095`, `f569957`, `d3008cb` |
| `15d4611` | `runs/xpu--main/15d4611-20261003T074218Z` | env, triton-smoke, unit-host, unit-xpu | **fail** on env only: the native smoke imported tensorfold before the branch install (fixed in `02b52bc`) |
| `02b52bc` | `runs/xpu--main/02b52bc-20261003T080109Z` | env, triton-smoke, unit-host, unit-xpu (native ext) | pass; K0: AOT hello extension, env smoke exact, IGC `dpas`; unit-xpu 54 |
| `95b3549` | `runs/xpu--main/95b3549-20261003T082711Z` | env, triton-smoke, unit-host, unit-xpu, kernels:gdn | pass; K1.T0; unit-xpu 90, gdn 36, no DEVICE_LOST |
| `b4120ac` | `runs/xpu--main/b4120ac-20261003T085557Z` | env, triton-smoke, unit-host, unit-xpu | pass; tree attention, DFlash2 block attention / append on XPU; unit-xpu 114 |
| `40c074b` | `runs/xpu--main/40c074b-20261003T095047Z` | env, triton-smoke, unit-host, unit-xpu, kernels:glue, kernels:prefill-attention, kernels:qmm | pass; prompt attention routing, `_attn_prep` host cos/sin, DFlash2 kernels, head-row views (`4dcf58b`); unit-xpu 126, qmm 52 |
| `3a28fb4` | `runs/xpu--main/3a28fb4-20261003T101551Z` | env, triton-smoke, unit-host, unit-xpu, kernels:prompt, kernels:experts, kernels:mamba | pass; K5 (`9240c88`), K3 (`529277e`), K2 (`4733cb5`); unit-xpu 153 |
| `c155552` | `runs/xpu--main/c155552-20261003T170920Z` | env, triton-smoke, unit-host, unit-xpu, kernels:prefill-attention, kernels:qmm, kernels:gdn, kernels:experts, kernels:mamba | pass; fix pass (`414da45` descriptor attention, `5f3bb56` barrier probe, `c155552` direct launches, 4-row GDN / scan); unit-xpu 169 |
| `56756cf` | `runs/xpu--main/56756cf-20261003T103858Z` | env, triton-smoke, unit-host, unit-xpu | pass; Nemotron router (DPAS), `_conv`/`_scan`, conv commit (double-buffered), attention merge, keyed sampler (100,000 draws equal `exact_sampling`); unit-xpu 164 |

unit-host is `pytest tests --host-only` in the runtime image (1485 passed, 30 skipped on the latest heads): it excludes
MLX modules and the GPU suite; it is not MLX or CUDA validation. CUDA paths are unchanged by construction (device
checks, new modules, CUDA launches untouched) but were not executed on CUDA hardware here.

## Per-kernel measurements (bundle numbers; cards hold every case)

| Kernel | Case | Result | % of peak | Spills | Lanes / DPAS |
|---|---|---|---|---|---|
| K4 SYM decode | Qwen down 5120 x 17408, M=1 (device-only) | 115 us | 66% of 608 GB/s | 0 | 16 / yes |
| K4 SYM decode | Qwen gate/up 17408 x 5120, M=1 / 16 | 136 / 205 us | 56 / 38% | 0 | 16 / yes |
| K4 bf16 GEMV | Qwen head 248320 x 5120, M=1 / 16 | 4977 / 5112 us | 84 / 82% | 0 | 16 / yes |
| K4 bf16 GEMV | Nemotron head 131072 x 2688, M=1 / 16 | 1364 / 1391 us | 85 / 84% | 0 | 16 / yes |
| K1 GDN | replay 48 layers x 4 rows (`c155552`) | 585 us | 85% of 608 GB/s | 0 | 32 / – |
| K1 GDN | 12-row tree, one layer (`c155552`) | 112 us | 33% | 0 | 32 / – |
| K1 GDN | 512-row prompt chain (`c155552`) | 950 us | latency-bound | 0 | 32 / – |
| K5 prompt GEMM | Qwen gate/up, 1024 rows | 8963 us, 20.4 TFLOPS | 11% of 183 TFLOPS | 0 | 16 / yes |
| K3 experts | decode up / down, 16 tokens x 8 slots | 510 / 552 us | 57 / 53% of 608 GB/s | 0 | 16 / yes |
| K3 experts | prompt up / down, 1024 tokens | 5325 / 6202 us | 8.4 / 7.2% of 183 TFLOPS | 0 | 16 / yes |
| K2 Mamba scan | 1024-row chunk, one layer (`c155552`) | 3507 us | latency-bound | 0 | 32 / – |
| K6 prompt attention | 129 rows, D=256, descriptor kernel (`c155552`) | 556 us | 0.5 TFLOPS | 62,016 B | 16 / yes |

**Host submission bounds small calls.** With direct launches (`c155552` `host_breakdown`) the split-K `sym_matmul`
wrapper takes 68.6 us of host time (98.5 us before); a Triton JIT launch is 31.6 us and a direct one 11.3 us;
`group_sums` alone is 41.8 us. Calls under ~90 us of device time still run at the host's pace (1- and 4-row GDN trees
at 101 us, Nemotron in/out_proj and in_proj_a/b at ~90 us).

## Open issues (after the 2026-10-03 fix pass, `414da45`..`c155552`, numbers from its bundle unless marked dev)

- **Launch overhead, part fixed.** `xpu/kernels/launch.py` calls compiled kernels directly after the first JIT
  dispatch (qmm, bf16 GEMV, reduce, GDN, experts): a split-K qmm call's host time 98.5 -> 68.6 us. What remains is
  the Intel driver's launch (~16 us each), allocations and wrapper checks; fewer launches (a fused split-K reduce,
  shared group sums, graphs after a graphs == eager test) is WS6.
- **K6 spills: structural in Triton XPU.** Every variant tried keeps 28-80 KB of spills (pointer loads, pre-transposed
  K, D split in halves, tensor descriptors; all bit-equal). The descriptor kernel is now the XPU route for D=128/256
  (dev: 11.5 vs 6.4 TFLOPS at D=128, 6.2 vs 5.8 at D=256). Removing the spills needs a native (N1) kernel.
- **GDN tree spills: fixed.** 4 rows a program, 1 warp: no spills; the prompt chain 1271 -> 950 us.
- **K1 / K2 prompt chains stay latency-bound** (sequential steps). 4 rows a program helped (scan 4128 -> 3507 us);
  `num_stages` did not. A chunked (WY / SSD) form changes the contract and needs its own card and tests.
- **`tl.debug_barrier`: verified.** It lowers to OpenCL `barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)`: a
  work-group barrier with a global fence, never cross-work-group; `triton-smoke` now checks the lowering.
- **SYM cancellation: closed.** Dev A/B: worst fp32 relative error 3.2e-6 (clustered nibbles, non-zero activation mean)
  against 2.3e-7 for `dot(x, q - 8) * s`, both ~1000x below bf16 output rounding; the contract stays.
- No DEVICE_LOST on any recurrence kernel so far.
- **Kernel code review** (`72edaec..c155552`, nine findings, all fixed in `f3b06d7` / `aa00945`): DFlash2 now takes
  SYM / bf16 head row views on XPU; Nemotron `_chunk`, `_conv`, `_conv_rows`, `_scan` launch with fusion off on XPU;
  `sample_candidates` uses runtime loops on XPU; strided rows in the XPU `matmul_rows`; `lane_matmul` derives the SYM
  group; no wasted `xs` in the Nemotron merge; one conv-commit kernel; `attn_prep` takes precomputed rope tables;
  `xpu/build.load` refuses a library built from other sources. Remaining WS5 item: engines should build the rope
  tables once per forward and pass them to `attn_prep`.

## WS3b / WS5 prerequisites (not done here)

- Loader: GPTQ / AutoRound SYM checkpoints to N-major `(N, K/8)` words + `(N, K/gs)` scales with `QLinear(sym=True)`;
  experts stacked with `experts.pack_xpu`; Nemotron MTP from `model_extra_tensors.safetensors`.
- DFlash2 on XPU: re-quantise the draft to SYM, and replace `dflash2._linear` (`F.linear`, not exact-path safe) with
  the bf16 GEMV.
- Engines: route projections to `qmm_fast.matmul` (decode) and `prompt_matmul` (prompt rows, never by chunk size);
  GDN through `xpu.select.module("gdn")` with stacked states; experts through `xpu.kernels.experts` plans; samplers as
  ported. `qmm_fast.matmul_partial` is refused on XPU.

## Recommended native (N1) targets (provisional: microbenchmarks, no decode profile yet)

1. Launch count (WS6, not N1): every small decode op is host-bound at ~32 us a Triton launch.
2. K4 SYM decode GEMV: 14 GB of weights a 27B token; T1 reaches 56-66% device bandwidth at M=1; target >= 80%.
3. K3 grouped decode: 53-57% of 608 GB/s; Nemotron's largest per-token weight stream.
4. K1 GDN tree step: 29% at a 12-row tree with spills; SG16 x 8 strided N0 mapping (gdn card) keeps the same bits.
5. K6 D=256 prompt attention and K5 prompt GEMM (11% of DPAS peak) for prefill throughput.

## Recovery rule

On DEVICE_LOST or a runner STOP flag, stop GPU submissions and record the failing SHA, logs and minimal repro.
Independent host work can continue. The operator must restore device health before new GPU experiments;
never clear STOP automatically or continue GPU work on another kernel while the device is wedged.
