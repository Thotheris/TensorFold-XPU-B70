# AGENTS.md: rules for agents working on the TensorFold XPU (Arc Pro B70) port

Read this first, then the doc for your workstream:

| Doc | What it holds |
|---|---|
| [docs/xpu/PORT_PLAN.md](docs/xpu/PORT_PLAN.md) | The plan: workstreams, agent roster, milestones, the B70 loop |
| [docs/xpu/KERNEL_MAP.md](docs/xpu/KERNEL_MAP.md) | Every kernel recipe A and B run, and its port ID |
| [docs/xpu/B70_NATIVE_KERNEL_GUIDE.md](docs/xpu/B70_NATIVE_KERNEL_GUIDE.md) | Hardware, SYCL/ESIMD, DPAS, 2D block IO, build, profiling |
| [docs/xpu/TRITON_XPU_GUIDE.md](docs/xpu/TRITON_XPU_GUIDE.md) | Intel Triton backend, bugs, and the audit of this repo's Triton |
| [docs/xpu/QUANT_FORMATS.md](docs/xpu/QUANT_FORMATS.md) | W4A16 sym formats and the target checkpoints |
| `docs/xpu/STATUS.md` | Current state, written by the Analyst |
| `docs/xpu/kernels/<k>.md` | Kernel cards |

## Goal
Add `--backend xpu` for one Intel Arc Pro B70, serving:
- **A:** `devan-carlin/Qwen3.8-27B-int4-AutoRound` with the DFlash2 drafter.
- **B:** `letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound` with MTP.

XPU reads **only symmetric INT4 W4A16 checkpoints**: AutoRound, GPTQ v1, or compressed-tensors `pack-quantized`.

## Non-negotiable rules
1. **Exactness.** A drafted token must bit-equal the serial token from the same engine. Every decode/verify kernel must
   give identical bits for a row regardless of the other rows in the launch. Prompt kernels must be chunk-invariant.
   XPU bits do **not** need to match CUDA bits.
2. **Don't change CUDA behaviour.** XPU differences go behind device checks or per-device constants. Host-side `tests/`
   must stay green.
3. **Arithmetic contracts are written down.** Before writing a kernel, fill in its kernel card: op order, FMA points,
   reduction order, and the invariances it must satisfy.
4. **No float atomics, no runtime autotuning, no M-dependent split-K** on exactness paths. Tiles, split-K, sub-group
   size and GRF mode are functions of the weight shape only.
5. **Native SYCL builds happen AOT in a separate shell** (`tools/xpu/build_ext.sh`). The runtime environment never has
   oneAPI `setvars.sh` or `icpx` on PATH, because it SIGSEGVs Triton.
6. **No single device allocation ≥ 4 GB.** Use 64-bit index math.
7. **Bit-sensitive Triton kernels:** `enable_fp_fusion=False`, explicit `num_warps`, `do_not_specialize` for runtime
   sizes, and a DPAS check for `tl.dot`.
8. **Stay inside the paths your role owns** (PORT_PLAN.md §1). If you need a change elsewhere, open an issue for the
   owner.

## Branches and the B70 loop
- Branch from `xpu/main` as `xpu/<ws>/<topic>`, for example `xpu/k1/gdn-triton`. Keep branches small.
- To get a hardware run, commit `.b70/run.yml` on your branch:
  ```yaml
  suites: [env, unit-xpu, kernels:gdn]
  baseline: xpu/main
  timeout_min: 90
  ```
- The B70 runner picks up new `xpu/*` heads, runs the suites, and commits a bundle to the `results` branch at
  `runs/<branch-slug>/<sha7>-<utc>/`. The bundle holds `summary.md`, `env.json`, `pytest.xml`, `kernels/*.json`,
  `e2e/*.json` and logs. One line per run is appended to `index.jsonl`.
- Read your results with:
  ```bash
  git fetch origin results && git show origin/results:index.jsonl | tail
  ```
- Only the Integrator merges to `xpu/main`, and only after a green bundle for the exact head SHA.

## Definition of done for a kernel
1. All invariance tests pass on the B70: alone == window, chunked == one-shot, 20 repeats, warm and cold cache.
2. Tolerance against the fp32/fp64 reference passes.
3. The kernel card records the measured `kbench` numbers (GB/s or TFLOP/s, % of B70 peak), the Triton `n_spills`,
   `threads_per_warp` and DPAS present, and the IGC dump facts for native kernels.
4. A native kernel replaces Triton only if it beats the tuned Triton version by ≥ 10% on the recipe's shapes.

## Commit messages
Imperative mood, scoped by workstream, e.g. `xpu/k1: add Triton GDN tree step`.
