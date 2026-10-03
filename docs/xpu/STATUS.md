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
| K1 GDN | replay 48 layers x 4 rows | 604 us | 82% of 608 GB/s | 0 | 32 / – |
| K1 GDN | 12-row tree, one layer | 125 us | 29% | 448 B | 32 / – |
| K1 GDN | 512-row prompt chain | 1271 us | latency-bound | 0 | 32 / – |
| K5 prompt GEMM | Qwen gate/up, 1024 rows | 8963 us, 20.4 TFLOPS | 11% of 183 TFLOPS | 0 | 16 / yes |
| K3 experts | decode up / down, 16 tokens x 8 slots | 510 / 552 us | 57 / 53% of 608 GB/s | 0 | 16 / yes |
| K3 experts | prompt up / down, 1024 tokens | 5325 / 6202 us | 8.4 / 7.2% of 183 TFLOPS | 0 | 16 / yes |
| K2 Mamba scan | 1024-row chunk, one layer | 4128 us | latency-bound | 0 | 32 / – |
| K6 prompt attention | 129 rows, D=256 (kernels:prefill-attention) | 595 us | 0.4 TFLOPS | 71,680 B | 16 / yes |

**Host submission bounds small calls.** One Triton launch costs about 32 us of host time (11 us of it the driver
launch); a split-K qmm call is about 115 us; `group_sums` alone is 42 us (`44a106d` `host_breakdown`). Calls whose
device time is below that run at the host's pace (1- and 4-row GDN trees at 121 us, Nemotron in/out_proj, in_proj_a/b).

## Open issues and minimal repros

- **Launch overhead** (above): the largest decode cost for small ops. Fixes are WS6 (fewer launches: fused reduce and
  group sums, cached launches, graphs after a graphs == eager test).
- **K6 D=256 spills ~71 KB** at every tile/warp/GRF choice (sweep in `40c074b`'s history, all bit-equal): structural
  (q, o and K/V tiles live together). Chaining the QK dot over D halves (as K4's `ksplit`) is the T1 candidate.
- **GDN tree spills 448-512 B** (R=8, 1 warp); R and warps change no bits, so T1 can retune freely.
- **K2 and K1 prompt chains are latency-bound** (2.5-4 us a step). A chunked form changes the contract; new card first.
- `tl.debug_barrier` as a global-memory fence on XPU stays `[UNVERIFIED]`; the one user (`_conv_commit`) is
  double-buffered on XPU instead.
- No DEVICE_LOST on any recurrence kernel so far (GDN tree/replay/chain, Mamba `_conv`/`_scan`, K2 scan).
- The SYM decode contract cancels (`P*s - 8*s*xs`) when q sits near 8; the `dot(x, q - 8) * s` A/B is open.

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
