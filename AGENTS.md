# AGENTS.md

This file is the operating manual for any coding agent (Claude Code, Codex, Cursor, others) working in this repository.
Read it in full before you change anything. If it conflicts with another doc, this file wins, then
[docs/xpu/PORT_PLAN.md](docs/xpu/PORT_PLAN.md).

## 1. What this repository is

`TensorFold-XPU-B70` is a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (v0.6.0). TensorFold
serves LLMs through an OpenAI-compatible API on Apple Silicon (MLX) and NVIDIA (CUDA), with **exact speculative
decoding**: a drafted token is accepted only if it is bit-identical to the token the same engine would produce
serially.

**This fork's goal** is to add a third backend, `--backend xpu`, for one **Intel Arc Pro B70** (Battlemage / Xe2,
32 GB, Linux, `xe` driver, PyTorch XPU). A second B70 is planned later. The first two target recipes:

| ID | Model | Checkpoint (W4A16 sym INT4) | Drafter | Family package |
|---|---|---|---|---|
| A | Qwen3.8-27B | `devan-carlin/Qwen3.8-27B-int4-AutoRound` (g128, fp16 scales) | `z-lab/Qwen3.8-27B-DFlash2` | `src/tensorfold/families/qwen3_5` |
| B | Nemotron 3.5 Lightning 30B-A3B | `letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound` (g64, fp16 scales) | built-in MTP | `src/tensorfold/families/nemotron_h` |

On XPU, TensorFold reads **only symmetric INT4 weight-only checkpoints**: AutoRound `auto_round:auto_gptq`, GPTQ v1,
or compressed-tensors `pack-quantized`. MLX, EXL3, NVFP4 and FP8 are refused. See
[docs/xpu/QUANT_FORMATS.md](docs/xpu/QUANT_FORMATS.md).

## 2. Read these before starting your task

| You are working on | Read |
|---|---|
| Anything | this file, then [docs/xpu/PORT_PLAN.md](docs/xpu/PORT_PLAN.md) (workstreams, roster, milestones) |
| Which kernels a recipe runs | [docs/xpu/KERNEL_MAP.md](docs/xpu/KERNEL_MAP.md) |
| Triton kernels | [docs/xpu/TRITON_XPU_GUIDE.md](docs/xpu/TRITON_XPU_GUIDE.md) |
| Native SYCL/ESIMD kernels, hardware, toolchain | [docs/xpu/B70_NATIVE_KERNEL_GUIDE.md](docs/xpu/B70_NATIVE_KERNEL_GUIDE.md) |
| Checkpoint loading, weight layouts | [docs/xpu/QUANT_FORMATS.md](docs/xpu/QUANT_FORMATS.md) |
| Current state, open problems | `docs/xpu/STATUS.md` (written by the Analyst; may not exist yet) |
| A specific kernel | its card in `docs/xpu/kernels/<kernel>.md` |
| How upstream adds a CUDA family | [docs/recipes/adding-a-cuda-family.md](docs/recipes/adding-a-cuda-family.md) |
| Upstream recipe details | [docs/recipes/qwen3.8-27b.md](docs/recipes/qwen3.8-27b.md), [docs/recipes/nemotron-3.5.md](docs/recipes/nemotron-3.5.md), [docs/recipes/cuda.md](docs/recipes/cuda.md) |

## 3. Repository map

```
src/tensorfold/
  cli.py, cli_args.py, serve_options.py   CLI; --backend selection (_backend, _serve_cuda) -> add xpu here
  families/__init__.py                    family registry: backends_of, require_readable, QUANT_METHODS
  families/qwen3_5/                       recipe A (Qwen3.8-27B); cuda/ holds its torch/Triton engine
  families/nemotron_h/                    recipe B (Nemotron); cuda/ holds engine, mamba, experts glue, MTP
  cuda/                                   shared torch backend: server/http (device-free), build.py (nvcc JIT),
                                          capacity.py, direct_read.py, comm.py (NCCL), sampling.py, experts.py
  cuda/kernels/                           shared kernels: qmm*.cu (4-bit matmul), gdn*.cu, prefill_attention.*,
                                          Triton attention.py, affine_kernels.py
  engine/, drafters/, server/             device-independent engine pieces, drafters, HTTP
  kernels/                                MLX/Metal kernels (arithmetic references only; not used on XPU)
  accel.py, xpu/                          NEW (planned): device abstraction; XPU build, quant loaders, kernels
tests/                                    host-side tests (run anywhere with torch)
tests/cuda/                               GPU tests; skipped unless a GPU is present (being generalized to XPU)
tools/                                    benchmarks: bench_openai.py, bench_concurrent.py, prefill_cold.py, ...
tools/xpu/                                NEW (planned): B70 bootstrap, runner, suites, kbench, compare
docs/xpu/                                 the port's docs
```

## 4. Environment and commands

Python ≥ 3.11. Run commands from the repo root.

```bash
python -m pip install -e '.[test]'      # dev install; on the B70 box use --no-deps (see below)
python -m pytest tests -q               # host-side suite: CLI, admission, server, geometry, fakes
python -m pytest tests/cuda -q          # GPU suite (CUDA today; XPU once the device fixture lands)
python -m pytest tests/test_cuda_cli.py -q -k backend   # a focused run
ruff check src tests                    # lint (config in pyproject.toml: line length 120)
```

**Rules for the B70 box (Linux):**
- Pinned toolchain:
  - torch **2.14.x+xpu** from `https://download.pytorch.org/whl/xpu`, with its bundled `triton-xpu~=3.8`;
  - Deep Learning Essentials **2026.1**;
  - compute-runtime 26.31, IGC 2.40, Level Zero loader ≥ 1.32;
  - kernel ≥ 6.17 with `xe`;
  - `ocloc` installed;
  - ReBAR on.
- **Never let pip replace torch or triton.** Install TensorFold with `pip install -e . --no-deps` into the pinned venv.
  The `grammar` extra pulls in torch; don't install it there.
- **Never source oneAPI `setvars.sh` or put `icpx` on PATH in a shell that runs Python/Triton.** It crashes Triton
  XPU with SIGSEGV. Native SYCL extensions are built ahead of time in a separate shell by `tools/xpu/build_ext.sh`
  (planned).
- After changing toolchain or Triton env knobs, clear `~/.triton/cache`; knobs are not part of its cache key.
- `TORCH_EXTENSIONS_DIR` controls the native build cache. `kill -USR1 <pid>` dumps Python stacks of a hung server.

You can run host-side tests and lint anywhere. You can only run GPU tests on the B70 box, through the loop in §8. Never
claim a GPU test passed unless a results bundle shows it.

## 5. The exactness contract (non-negotiable)

1. **Row invariance.** Every decode/verify kernel gives identical bits for a row whether it runs alone or inside any
   window or tree, and alongside any other streams.
2. **Chunk invariance.** Prompt kernels give identical bits whatever the chunk boundaries are, and for resumed vs
   fresh prompts.
3. **drafted == serial.** Compare a request against the same request with `"draft": false`. Equality is token-for-token
   (`token_sha`).
4. **Self-consistency, not CUDA parity.** XPU bits need not match CUDA bits. They must match XPU serial bits on a pinned
   toolchain.
5. **How to stay exact:**
   - Split-K, tile sizes, sub-group size, GRF mode and `num_warps` are functions of the **weight shape only**: never
     of the row count M, and never autotuned at runtime.
   - No float atomics.
   - Fixed-order reductions; rank-ordered fp32 sums.
   - Stable routing ties (lower expert id wins).
   - Explicit FMA points: `enable_fp_fusion=False` in Triton; `-fp-model=precise -ffp-contract=off` with explicit
     `sycl::fma` in SYCL.
6. **Never** use torch library matmuls (`F.linear`, oneDNN) on an exactness path. Their algorithm changes with shape,
   and XPU eager is non-deterministic by default. If eager torch is unavoidable, set
   `torch.use_deterministic_algorithms(True)`.
7. **Quality is a separate check:** tolerance against `reference.py` fp32/fp64 forwards, plus the loader quality gate.

## 6. XPU porting rules

1. **Don't change CUDA or MLX behaviour.** Put XPU differences behind `device.type == "xpu"` checks, per-device constant
   tables (`XPU_CONFIG`), or new modules under `src/tensorfold/xpu/`. Never edit CUDA launch constants or `.cu` files
   for XPU's sake.
2. **Device access goes through `tensorfold.accel`** (planned). Do not add new `torch.cuda.*` or `device="cuda"`
   literals in shared code.
3. **Port in stages:**
   - T0: correct Triton.
   - T1: tuned Triton.
   - N0: correct native SYCL/ESIMD behind the same Python `_ext()` signature.
   - N1: fast native.

   A native kernel replaces Triton only after it passes every invariance test and beats T1 by **≥ 10%** on the
   recipe's shapes in a B70 bundle. Selection lives in `src/tensorfold/xpu/select.py` (planned), with
   `TF_XPU_KERNEL_<OP>=triton|native` overrides.
4. **Write the kernel card first** (`docs/xpu/kernels/<kernel>.md`, template in §11).
5. **Hardware facts that change code** (details in the guides):
   - Sub-group **16** for anything using XMX.
   - DPAS tile **8×16×16** for bf16, not m16n8k16.
   - **No FP8/FP4** hardware.
   - `ldmatrix`/`cp.async` become **2D block loads and prefetch** from global memory.
   - At most **128 KB SLM** per work-group.
   - **No single allocation ≥ 4 GB**; use 64-bit index math.
   - Device pointers may be **≥ 2^63**: convert with `p - (1 << 64) if p >= (1 << 63) else p` before putting them in
     int64 tensors.
6. **Triton on XPU:**
   - A `tl.dot` kernel compiles at 16 lanes and others at 32, so reduction trees differ between kernels. Record the
     compiled `threads_per_warp`.
   - DPAS needs N ≥ 16 and silently falls back to FMA otherwise; assert `#triton_intel_gpu.dpas` in the TTGIR.
   - Known bugs: `BLOCK_M=16` dot miscompile, `while`-loop crash, long `static_range` IGC abort, int/fp16→bf16 cast
     crash (cast via fp32), and recurrence kernels with persistent state can hit `DEVICE_LOST`.
   - Workarounds are in the Triton guide §1.5.
7. **Weights:** symmetric INT4 means `w = s·(q−8) = s·q + b` with `b = −8·s`, which is exact. Kernels take a `SYM` path
   and never store `b`. Keep fp16 scales as fp16. Group sizes are 64 and 128. Refuse any non-identity `g_idx`. For sym
   checkpoints, z = 8 always comes from the `sym` flag; never from `qzeros`, which some repos store as 0.
8. **Graphs** are off on XPU until a graphs == eager bitwise test passes on the box.
9. **FP8 paths** (`--prefill-fp8`, int8/int4 KV) are refused on XPU.

## 7. Workstreams and file ownership

Each role owns paths. Change only files your role owns. If you need a change elsewhere, open a GitHub issue labelled
`ws:<owner>` describing it.

| Role | Owns |
|---|---|
| Integrator | merges to `xpu/main` and `main`; `.b70/`; upstream syncs; CHANGELOG |
| Infra (WS1) | `src/tensorfold/accel.py`, `cli*.py`, `families/__init__.py`, `serve_options.py`, `cuda/{capacity,memory_gate,direct_read,build}.py`, `src/tensorfold/xpu/__init__.py` |
| Harness (WS2) | `tools/xpu/**`, `tests/conftest.py`, `tests/cuda/conftest.py`, `tests/devices.py`, `tests/xpu/**`, test device-fixture rewrites |
| Triton-port (WS3) | portability fixes inside existing Triton modules on the A/B paths (see KERNEL_MAP) |
| Loader (WS3b) | `src/tensorfold/xpu/quant/**`, `QUANT_METHODS["xpu"]` entries, `tests/test_xpu_quant_*.py` |
| Kernel K0–K7 (WS4) | `src/tensorfold/xpu/kernels/<k>/**`, `src/tensorfold/xpu/build.py` (K0), `docs/xpu/kernels/<k>.md`, the kernel's tests and bench |
| Engine-A / Engine-B (WS5) | XPU wiring in `families/qwen3_5/**` and `families/nemotron_h/**` |
| Analyst | the `results` branch analysis, `docs/xpu/STATUS.md`, issues |
| Docs | `docs/xpu/*.md` guides |

## 8. Branches and the B70 test loop

- `main`: this fork's default branch. It holds docs, agent instructions and integrated work.
- `xpu/main`: the integration trunk for port code. Only the Integrator merges here, and only after a green B70 bundle
  for the exact head SHA.
- `xpu/<ws>/<topic>`: your work branch, cut from `xpu/main` (e.g. `xpu/k1/gdn-triton`, `xpu/infra/accel`). One topic
  per branch; keep it small.
- `results`: an orphan branch of result bundles. Never merge it anywhere.
- `upstream` remote: `ashhart/TensorFold`. **Fetch only. Never push, open PRs or file issues there** unless the repo
  owner explicitly says the work is ready. In every clone, disable it with
  `git remote set-url --push upstream DISABLED-do-not-push-to-upstream`. All pushes go to `origin`
  (`Thotheris/TensorFold-XPU-B70`). Upstream merges into this fork are the Integrator's job and go forward only.

**To get a hardware run**, commit `.b70/run.yml` on your branch and push:
```yaml
suites: [env, unit-xpu, kernels:gdn]   # names defined in tools/xpu/suites.py
baseline: xpu/main
timeout_min: 90
```

The B70 runner (`tools/xpu/b70_runner.py`, planned) checks out each new `xpu/*` head, installs it `--no-deps` into the
pinned venv and runs the suites. It then commits a bundle to `results` at `runs/<branch-slug>/<sha7>-<utc>/`:
- `summary.md`
- `env.json` (toolchain and checkpoint revisions)
- `pytest.xml`
- `kernels/*.json` (µs, GB/s, % peak, bitwise checks, `n_spills`, `threads_per_warp`, DPAS present)
- `e2e/*.json` (tok/s, TTFT, `token_sha` match)
- logs

It also appends one line to `index.jsonl`.

**To read results:**
```bash
git fetch origin results
git show origin/results:index.jsonl | tail -n 20
git show origin/results:runs/<branch-slug>/<sha7>-<utc>/summary.md
```

The runner executes branch code on the user's machine. Do not put anything in a branch that touches files outside the
worktree, downloads unvetted binaries, or needs secrets.

## 9. Code conventions (match upstream)

- Read the surrounding module before writing, and match its density and idiom. Upstream style:
  - one-line module, class and function docstrings stating *what is true* ("The shared GDN kernels: tree nodes, chains
    and commit replays run the serial step, bit for bit.");
  - terse one-line comments for non-obvious constraints only;
  - `from __future__ import annotations`; type hints on public functions; `__all__` in library modules.
- Line length 120 (ruff). No new runtime dependencies without the Integrator's sign-off. torch and triton are never
  dependencies.
- **Import torch inside backend code**, not at family-package import time, so family discovery works on MLX installs.
- Native sources and data files must be declared as package data in `pyproject.toml`. `tests/test_packaging.py`
  checks this, so add `*.sycl`/`*.hpp` patterns when you add them.
- Tests:
  - GPU tests go in `tests/cuda/` and use the device fixture (`DEV`), not `"cuda"` literals.
  - Bitwise checks compare integer views (`a.view(torch.int16).equal(b.view(torch.int16))`).
  - Tests that genuinely need CUDA are marked `cuda_only`.
  - Host-side tests use fakes, like `tests/test_cuda_capacity.py`.

## 10. Definition of done

**For any change:**
- Host-side `pytest tests` passes and `ruff check` is clean.
- CUDA behaviour is unchanged.
- Docs and the kernel card are updated.
- A B70 bundle for the head SHA is green for the suites your change affects.

**For a kernel:**
- Invariance passes on the B70: alone == window at every size, chunked == one-shot, 20 repeats, warm and cold cache.
- Tolerance against the reference passes.
- The card records measured performance against the B70 peak (608 GB/s, about 183 TFLOPS bf16), plus spills, lanes
  and DPAS use.

**For a recipe milestone:**
- `e2e:*-smoke` shows drafted `token_sha` == serial on fixed prompts and seeds.
- The loader quality gate passes.

## 11. Kernel card template (`docs/xpu/kernels/<kernel>.md`)

```markdown
# <kernel>  (port ID, owner, status: T0|T1|N0|N1)
Op / CUDA source / Python wrapper:
Shapes (recipe A, recipe B) and dtypes:
Arithmetic contract: exact fp32 op order, FMA points, reduction order and lanes
Invariances required: row-alone==window | chunk-invariant | replay==serial | ...
Layouts in/out (packing, strides, alignment for 2D block loads):
References: torch fp32/fp64 function, Triton baseline, tests that pin bits
Roofline target on B70 and why:
Measurements: (date, sha, toolchain hash, kbench numbers, n_spills, threads_per_warp, DPAS y/n)
Open issues:
```

## 12. Commits and pull requests

- Use upstream's Conventional Commits style with a scope: `feat(xpu): ...`, `fix(xpu/k1): ...`,
  `perf(xpu/k4): ...`, `test(xpu): ...`, `docs(xpu): ...`. Imperative, specific, and saying what is now true
  ("feat(xpu/k1): Triton GDN tree step runs the serial chain bit for bit").
- One topic per PR into `xpu/main`. In the description, link the B70 bundle (`results` path) for the head SHA and state
  which invariance and perf numbers changed.
- Never force-push shared branches (`main`, `xpu/main`, `results`). Never commit model weights, results bundles on code
  branches, `.so` builds, or secrets.

## 13. When you are stuck or unsure

- Hardware or toolchain behaviour you cannot verify: tag it `[UNVERIFIED]` in the relevant doc, and add a probe to the
  `env`/`triton-smoke` suites instead of guessing.
- A test fails on the B70 and you can't tell why: ask for a run with dumps (`TRITON_KERNEL_DUMP=1`,
  `IGC_ShaderDumpEnable=1`) via `.b70/run.yml`, and record findings in the kernel card.
- A `DEVICE_LOST` or GPU wedge: stop that branch's runs, reproduce with a minimal kernel, and report it in `STATUS.md`.
  Out-of-bounds bugs can wedge the box.
