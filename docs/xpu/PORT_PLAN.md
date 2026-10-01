# TensorFold → Intel XPU (Arc Pro B70 / Battlemage) — Multi-Agent Port Plan

## Context

TensorFold (upstream `ashhart/TensorFold` v0.6.0, fork `Thotheris/TensorFold-XPU-B70`) serves LLMs on MLX and CUDA with
**exact** speculative decoding: a drafted token is accepted only if it bit-equals what the same engine produces serially.
The goal is a third backend, `--backend xpu`, on one Intel Arc Pro B70 (Xe2 / Battlemage, 32 GB, Linux + `xe` driver,
PyTorch XPU), with a second B70 planned later. The first targets are two recipes:

- **A: Qwen3.8-27B W4A16 AutoRound + DFlash2** (`families/qwen3_5`): a GDN/attention hybrid with no CUDA graphs.
- **B: Nemotron 3.5 Lightning 30B-A3B W4A16 AutoRound + MTP** (`families/nemotron_h`): Mamba-2, attention and
  128-expert MoE, captured as CUDA graphs.

Development happens in cloud and local agent branches. The B70 box pulls a branch, builds and tests it, and pushes
results to a `results` branch. Analyst agents read the results and file the next work. Then the loop repeats.

**Three findings from the code maps shape this plan:**
1. The HTTP, OpenAI, scheduler and drafting layers are device-free. CUDA coupling sits in about 15 CLI/registry lines,
   5 engine constructors, `capacity/direct_read/memory_gate/comm/build`, and graphs.
2. Both recipes depend on five native nvcc extensions. Bit-exact Triton replacements already exist for the 4-bit decode
   matmul (`qwen3_5/cuda/qmm.py::lane_matmul`, asserted equal in `tests/cuda/test_qmm.py::test_27b_triton_bits`) and for
   prompt attention (`triton_attention`, the bit definition). **No portable version exists** of the GDN tree, replay and
   chain kernels, the grouped MoE experts with their plan and pack kernels, or the Mamba prompt scan.
3. Exactness on XPU means **self-consistency** (serial == drafted, alone == in-window, chunked == one-shot). It does not
   mean bit-equality with CUDA. Upstream already accepts per-backend serial bits.

### Target checkpoints (Intel-native W4A16, from the HF survey)

Intel's B-series stack (AutoRound, vllm-xpu-kernels `int4_gemm_w4a16`, the XPU W4A16 MoE) is built around
**symmetric INT4 weight-only, GPTQ layout**. **XPU targets only these formats**; MLX/EXL3/NVFP4 checkpoints are
refused on `--backend xpu` (decision: no MLX parity path).

| Recipe | Primary (perf target) | Secondary | Drafter |
|---|---|---|---|
| A Qwen3.8-27B | `devan-carlin/Qwen3.8-27B-int4-AutoRound` (`auto_round:auto_gptq`, **g128 sym**, fp16 scales; embed/lm_head/vision/`in_proj_a,b` BF16; 19 GB) | `RedHatAI/Qwen3.8-27B-INT4` (compressed-tensors `pack-quantized`, g128 sym, BF16 scales) | `z-lab/Qwen3.8-27B-DFlash2` (BF16 3.85 GB; TensorFold already re-quantizes it to 4-bit at load, about 1 GB; on XPU it is re-quantized to **SYM** g64) |
| B Nemotron 30B-A3B | `letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound` (**g64 sym**, fp16 scales, router/embed/lm_head 16-bit, **MTP in `model_extra_tensors.safetensors`**; 18.8 GB) | `SergiioB/...-GPTQ-INT4-G64-sym` (BF16 scales; **qzeros stored as 0, so z=8 must come from `sym`**; no MTP, so serve with `--no-drafts`; has published B70 numbers) | built-in MTP head now; **DFlash (BF16, `SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`) later** (WS9) |

**Why the existing kernel contract survives the format change.**
- Symmetric INT4 dequantizes as `w = s·(q−8) = s·q + b` with `b = −8·s`. Multiplying by 8 is exact in fp16 and bf16, so
  the lane-matmul arithmetic `fma(xs,b,fma(P,s,acc))` applies **unchanged and exactly**.
- Native kernels take a `SYM` flag and derive `b` in-register, so the bias is never read from memory. This saves
  bandwidth and gives identical bits.
- Keep **fp16 scales as fp16**: converting fp16 to bf16 would be lossy. Every XPU matmul kernel is templated on scale
  dtype (fp16 or bf16).

**Format-driven requirements.**
- Group size **128** (Qwen) as well as 64 (Nemotron, forced because 1856 = 29·64).
- GPTQ `[K/8,N]` is the transpose of MLX `[N,K/8]` with the same nibble order. It is packed at load into the XPU
  layout.
- No act-order/`g_idx`: refuse any checkpoint with a non-trivial `g_idx`.
- BF16 lm_head (248320×5120 = 2.5 GB read per token) needs a bf16 GEMV path; it adds about 18% to 27B decode traffic. An
  optional later `--xpu-head-int8` re-quantization flag is a quality trade-off and stays opt-in.
- Memory: 27B is about 19 GB + DFlash2 about 1 GB + KV (64 KiB/token bf16 → 64k context ≈ 4 GB) and fits 32 GB.
  Nemotron is about 19 GB with tiny KV.

**Strategy:** bring each recipe up **correct-first on Triton**, writing new Triton kernels for the three gaps. Only then
replace hot paths with **native SYCL/ESIMD kernels** that are measured against the Triton baseline. Each native kernel
sits behind the same Python `_ext()` signature, so engines don't change when it is swapped in.

---

## 0. Repository, branches and the B70 loop

### Branches (fork `Thotheris/TensorFold-XPU-B70`)
- `upstream-main`: mirrors `ashhart/TensorFold` main. It is refreshed by hand and only ever merged forward.
- `xpu/main`: the integration trunk. Only the Integrator agent merges here, and only after a green B70 run.
- `xpu/<ws>/<topic>`: one branch per work item, e.g. `xpu/k1/gdn-triton` or `xpu/infra/accel`. Each branches from
  `xpu/main` and stays small.
- `results`: an orphan branch that holds only result bundles. It never merges into code branches.

### How a branch asks the B70 to run it
A branch carries **`.b70/run.yml`**:
```yaml
suites: [env, unit-xpu, kernels:gdn, e2e:27b-smoke]   # names from tools/xpu/suites.py
baseline: xpu/main          # which run to diff against
timeout_min: 90
model_cache: /models        # where checkpoints live on the B70 box
```

### B70 runner (`tools/xpu/b70_runner.py`, run by a systemd user timer every 5 min)
1. `git fetch` the fork, list `xpu/*` heads, and skip any head SHA that already has a result in `results`.
2. For each new head, oldest first, one at a time: check out a clean worktree in `/work/<sha>`, then
   `pip install -e . --no-deps` into the pinned XPU venv. It never upgrades torch or triton.
3. Run the suites from `.b70/run.yml` through `tools/xpu/suites.py`. Every suite writes JSON plus a log.
4. Write the bundle `runs/<branch-slug>/<sha7>-<utc>/`:
   - `env.json`: torch, `torch.version.xpu`, triton, oneAPI/DPC++, compute-runtime, kernel and `xe` versions,
     `xpu-smi` device info, clocks, B70 firmware;
   - `pytest.xml` + `pytest.txt`;
   - `kernels/*.json` (microbench: shape, µs, GB/s, % of peak, bitwise checks);
   - `e2e/*.json` (`tools/bench_openai.py` and `bench_concurrent.py` output: tok/s, TTFT, token_sha match);
   - `build.log`, `server.log`, and IGC/Triton dumps when requested;
   - `summary.md`, generated with pass/fail counts and a perf diff against the baseline bundle.
5. Append one line to `index.jsonl` with branch, sha, time, status and key metrics. Commit to `results` and push.
6. Never push to code branches. Results are data only.

Security note: the runner executes code from branches in the fork. Restrict it to `xpu/*` refs on the fork, run it as an
unprivileged user, and keep secrets off the box (a push-only deploy key for `results`).

### Analysis step (Analyst agent)
- It reads `results/index.jsonl` and new bundles, and runs `tools/xpu/compare.py <bundle> <baseline>`. That tool reports
  regressions, new failures, bitwise-check breaks and perf deltas by kernel and end to end.
- It writes `runs/.../analysis.md` and updates `docs/xpu/STATUS.md` on its own `xpu/docs/status` branch. It then opens
  or updates GitHub issues labelled `ws:<n>` with concrete next tasks. Developer agents pick up those issues.

---

## 1. Agent roster and ownership

To avoid merge conflicts, each agent owns specific paths.

| Agent | Owns | Model |
|---|---|---|
| **Integrator** | merges to `xpu/main`, `.b70/`, release notes, upstream rebases | Opus |
| **Infra** (WS1) | `src/tensorfold/accel.py`, `cli*.py`, `families/__init__.py`, `serve_options.py`, `cuda/{capacity,memory_gate,direct_read,build}.py`, new `xpu/` package skeleton | Sonnet |
| **Harness** (WS2) | `tools/xpu/**`, `tests/conftest.py`, `tests/cuda/conftest.py`, the device fixture, `tests/xpu/**` | Sonnet |
| **Triton-port** (WS3) | existing Triton modules used by recipes A and B (portability fixes only) | Sonnet |
| **Kernel agents K1–K7** (WS4) | one kernel family each: new Triton/SYCL sources and their test and bench files | Opus |
| **Loader** (WS3b) | `src/tensorfold/xpu/quant/**`, the `QUANT_METHODS["xpu"]` entries, quant tests | Sonnet |
| **Engine-A / Engine-B** (WS5) | `families/qwen3_5/**`, `families/nemotron_h/**` engine wiring | Opus |
| **Analyst** | `results` bundles, `docs/xpu/STATUS.md`, issues | Sonnet |
| **Docs** | `docs/xpu/*.md` guides (kept current as findings come in) | Sonnet |

Every agent reads `AGENTS.md` (the fork root) first. It holds the rules: exactness contract, branch naming, `.b70/run.yml`,
never touch the CUDA path's behaviour, and how to read results.

---

## 2. Documentation deliverables (first PR: `xpu/docs/guides`)

- `docs/xpu/B70_NATIVE_KERNEL_GUIDE.md`: from the B70 research agent. Covers hardware, XMX/DPAS shapes and dtypes,
  subgroups, SLM/GRF, the 2D block loads that replace `ldmatrix`/`cp.async`, the SYCL `joint_matrix`, ESIMD and
  sycl-tla models, the CUDA→SYCL cheat-sheet, build via `torch.utils.cpp_extension`, profiling (unitrace/VTune/IGC
  dumps), determinism, pitfalls, and reference repos.
- `docs/xpu/TRITON_XPU_GUIDE.md`: from the Triton research agent. Covers the Intel Triton backend knobs (`num_warps`,
  `threads_per_warp`, `grf_mode`), `tl.dot`→DPAS, block pointers, fp64/fp8 status, dumping IR and asm, determinism, and a
  risk audit of this repo's Triton files.
- `docs/xpu/QUANT_FORMATS.md`: from the HF survey. Covers Intel's recommended formats, the checkpoint inventory for
  both models, exact GPTQ/AutoRound/compressed-tensors tensor layouts and zero-point conventions, the `b=-8s` mapping,
  and the self-quantization recipe (AutoRound W4A16 sym; g128 for Qwen, g64 for Nemotron) for later.
- **First action after approval:** I write these four research-derived docs (B70 native guide, Triton XPU guide, quant
  formats, kernel map) from the agent reports already gathered, so nothing is lost, and commit them on
  `xpu/docs/guides` in the fork.
- `docs/xpu/KERNEL_MAP.md`: the kernel inventory for recipes A and B (prefill, decode, verify, sampling): kind, CUDA
  features, fallback, owner and status. Sections 4–5 below are its seed.
- `docs/xpu/kernels/<kernel>.md`: one **kernel card** per port target (template in §4.0).
- `docs/xpu/HARNESS.md`: B70 box setup, runner, suites and result schema.
- `docs/xpu/STATUS.md`: kept by the Analyst.
- `AGENTS.md`: agent operating rules (above).

---

## 3. Layer plans (infrastructure)

### WS1 — Device abstraction and the `xpu` backend (Infra agent)
**New module `src/tensorfold/accel.py`:** `api(device)` returns `torch.cuda` or `torch.xpu`, plus helpers `synchronize`,
`empty_cache`, `Event`, `Stream`, `stream`, `current_stream`, `set_device`, `mem_get_info`, `memory_allocated`,
`memory_reserved`, `is_available`, `device_type()`, `name()`, and `is_discrete()`.

Steps:
1. **CLI and registry.**
   - `cli_args.py:113`: add `"xpu"` to the choices. `auto` picks XPU when `torch.xpu.is_available()` and CUDA isn't.
   - `cli.py:219-227` `_backend`: an `xpu` branch requiring `xpu_engine`.
   - `cli.py:230-299`: generalize `_serve_cuda` to `_serve_torch(..., device)`, and fix the "on CUDA" log lines.
   - `families/__init__.py:129` `backends_of`: add `("xpu","xpu_engine")`. Lines 144-167 `require_readable`: XPU reads
     `auto-round`/`gptq`/`compressed-tensors(pack-quantized)` W4A16 sym (WS3b) **only**. Refuse
     mlx/modelopt/NVFP4/EXL3/FP8 with a clear message.
   - `serve_options.py`: XPU refuses `--kv-dtype` (other than bf16), `--prefill-fp8`, `--tp 2` and `--decode-share`
     until supported.
2. **Families.** In `qwen3_5/__init__.py` and `nemotron_h/__init__.py`, export
   `xpu_engine = functools.partial(cuda_engine-equivalent, device="xpu")` and declare `QUANT_METHODS["xpu"]=("auto-round","gptq","compressed-tensors")`.
   Thread `device` through the engine constructors and loader defaults: `qwen3_5/cuda/{engine,weights}.py` and
   `nemotron_h/cuda/{app,engine,weights}.py`. Replace `torch.cuda.set_device(0)` with `accel.set_device`. Guard
   `_set_allocator_settings("expandable_segments")` to CUDA only. `gb10()` returns False on XPU, giving 12-row windows.
3. **Admission.**
   - `cuda/build.py`: `refuse_old_gpu` becomes a no-op on XPU. Add `xpu/build.py` (WS4-K0).
   - `cuda/capacity.py`: `available_bytes`/`total_bytes`/`unified` go through `accel`; the B70 is discrete.
     `gather_ints` takes a device.
   - `cuda/memory_gate.py`: `torch_live` goes through `accel`.
4. **I/O.** `cuda/direct_read.py:38,196`: treat `"xpu"` like `"cuda"` and use `accel.Stream/Event`. Guard
   `_host_emptyCache`. Verify that `pin_memory()` pins for XPU in the pinned torch version, and fall back to unpinned if
   it doesn't.
5. **Pinned staging in the hot loops.** Change sites that set `pin_memory=torch.cuda.is_available()` to
   `accel.pinnable()`. Covers `cuda/kernels/{attention.py:209,gdn.py:106}`, `qwen3_5/cuda/{forward,draft_attention}.py`
   and `nemotron_h/cuda/{engine,mtp}.py`.
6. **Graphs off on XPU.**
   - Nemotron: `app.py` always calls `capture`; make it conditional on `accel.graphs_supported()` (False at first).
     `Engine(graphs=False)` already exists.
   - The 27B uses no graphs.
7. **Sampling.** `cuda/sampling.py:21` `.is_cuda` guard → device-type check. The float64 nucleus path (top_k=0)
   depends on the FP64 findings in the B70 guide. If FP64 is emulated or slow on Xe2, add a CPU float64 fallback for
   top_k=0 only (exact either way).

Verification: `pytest tests/test_cuda_cli.py tests/test_cuda_capacity.py ...` (host-side, faked) plus new
`tests/test_xpu_cli.py`. On the B70: `tensorfold info` and `tensorfold serve --backend xpu` start up as far as kernel
load.

### WS2 — Test infrastructure and harness (Harness agent)
1. **Device fixture.** Add `tests/devices.py` with `DEV = os.environ.get("TF_TEST_DEVICE") or ("cuda" if
   torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else None)`. `tests/cuda/conftest.py` collects when
   `DEV` is set.
2. **Rewrite the tests.** Mechanically replace `device="cuda"` → `device=DEV` and `torch.cuda.synchronize` →
   `accel.synchronize` in `tests/cuda/*` (about 390 sites), and the 43 `pytest.skip("CUDA only")` guards with
   `requires_gpu`. Tests that genuinely need CUDA (nvfp4, exl3, cluster, graph capture) get
   `@pytest.mark.cuda_only`, which is skipped on XPU.
2b. **Synthetic SYM weights.** Upstream tiny models (`tests/cuda/nemotron_fakes.tiny_weights`, the qwen27 tiny builders
   in `test_qwen27_forward.py`, `test_experts.py`) emit MLX affine weights. Add a `fmt="sym4"` option that emits
   GPTQ/AutoRound-layout SYM tensors with fp16 or bf16 scales at g64 and g128, built with the WS3b loader. On XPU, the
   invariance tests run on those.
3. **Tiers** (`tools/xpu/suites.py`):
   - `env`: device query plus one tiny Triton and one tiny SYCL kernel.
   - `unit-host`: CPU tests.
   - `unit-xpu`: `tests/cuda` minus `cuda_only`.
   - `kernels:<name>`: microbench and bitwise invariance per kernel.
   - `e2e:27b-smoke` / `e2e:nemotron-smoke`: serve, `/health`, 3 fixed prompts, compare token_sha drafted vs
     `draft:false`.
   - `e2e:*-bench`: `bench_openai.py`, `prefill_cold.py`, `bench_concurrent.py --serial`.
4. **Kernel microbench harness** (`tools/xpu/kbench.py`): times with `torch.xpu.Event`, plus warm-up, median and p10/p90.
   Reports achieved GB/s and TFLOP/s against B70 peaks taken from the guide.
5. Write the runner, `compare.py`, the result schema (`tools/xpu/schema.json`), the systemd unit and `docs/xpu/HARNESS.md`.
6. **B70 box bootstrap script** (`tools/xpu/bootstrap.sh`): the exact pins from Appendix A.
   - Kernel ≥ 6.17 with `xe`, CR 26.31, IGC 2.40, L0 ≥ 1.32, `ocloc`, and Deep Learning Essentials 2026.1 (no full
     Base Toolkit, no IPEX).
   - A venv with `torch==2.14.*` from the XPU index plus its bundled triton. Never let `pip` replace them: install with
     `--no-deps` and keep a `constraints.txt`.
   - Checks: ReBAR enabled, host RAM ≥ 64 GB, `xpu-smi discovery`.
   - Model cache: `tensorfold pull` (or `hf download`) for `devan-carlin/Qwen3.8-27B-int4-AutoRound`,
     `RedHatAI/Qwen3.8-27B-INT4`, `letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound`,
     `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym` and `z-lab/Qwen3.8-27B-DFlash2`. Record each one's
     revision SHA in `env.json`.

### WS3 — Triton portability (Triton-port agent)
Scope: only the Triton modules on the two recipes' paths.
- Shared: `cuda/kernels/{qmm,attention,prefill_attention,affine_kernels}.py`.
- Recipe A: `qwen3_5/cuda/{glue,qmm,dflash2,draft_attention}.py`.
- Recipe B: `nemotron_h/cuda/{glue,mamba,attention,sampler}.py`.

Per module:
1. Run its existing tests on XPU.
2. Fix compile errors.
3. Re-check the invariance tests.
4. Record the changes in the Triton guide's audit table.

Known risks to resolve:
- **Pointer tricks:** offset-from-base addressing in `kernels/attention.py` (`base()`/`offsets()`) and int64→
  `tl.pointer_type` casts in `draft_attention.py`. If USM pointers misbehave, switch to per-cache index tables.
- **Device FP64/uint64:** `nemotron_h/cuda/sampler.py::_keyed`.
- **`tl.dot` defaults to TF32 on fp32 inputs:** set `input_precision="ieee"` where the reference expects it.
- **`num_warps` is "part of the arithmetic"** (`nemotron_h/cuda/mamba.py:12 SCAN_BD, SCAN_WARPS`): pin both
  `num_warps` and `threads_per_warp` explicitly on XPU and record them in the kernel card.
- **`tl.histogram`/`tl.cumsum`:** used only by GLM, out of scope.

Rule: changes must leave the CUDA output bit-identical. Use per-device constants (`XPU_CONFIG`) rather than editing CUDA
configs.

### WS3b — W4A16 checkpoint loaders (Loader agent; owns `src/tensorfold/xpu/quant/**`, the family `QUANT_METHODS["xpu"]` entries, `tests/test_xpu_quant_*.py`)
Purpose: read the primary and secondary checkpoints into one internal **`W4` weight object** that the matmul and expert
kernels consume. Precedent: upstream's `qwen3_5/cuda/{exl3_load,nvfp4_load}.py` and the `require_readable` /
`QUANT_METHODS` plumbing.

1. **Detect and validate** (header-only, before download; reuse `cuda/capacity.headers`):
   - Accept `quantization_config.quant_method ∈ {auto-round, gptq}` with `packing_format`/`checkpoint_format ∈
     {auto_gptq, gptq (v1)}`, and `compressed-tensors` with format `pack-quantized`.
   - Require `bits=4`, `sym=true`, `group_size ∈ {64,128}`, and `desc_act=false` (or an identity `g_idx`).
   - Refuse anything else by name: asym, act-order, gptq_v2 until tested, and 2/3/8-bit.
2. **Unquantized-module map.** Honour AutoRound `extra_config` / `modules_to_not_convert` / the compressed-tensors
   `ignore` list exactly. Expected BF16 tensors: Qwen `in_proj_a/b`, embed, lm_head, vision; Nemotron router gates,
   embed, lm_head, norms.
   - Loaders take a `bf16` path for those. Small ones use the existing `F.linear`/Triton dense path; lm_head uses the
     K4 bf16 GEMV.
3. **Zero-point handling.** Compute z from `sym` (=8). Never read qzeros for sym; the SergiioB repo stores 0. Assert
   that `qzeros ∈ {0x77777777, 0}` as a sanity check and record which.
4. **Repack** to the XPU layout (`pack_xpu`, defined by K4/K3) at load, streaming per tensor through `direct_read`. Avoid
   any single staging allocation ≥ 4 GB.
5. **Name mapping** from the HF `NemotronH`/`Qwen3_5` parameter names to TensorFold's `Weights` structures.
   - Nemotron MTP: `model_extra_tensors.safetensors` → `MTPHead`.
   - Experts: stack 128 routed experts plus the shared expert, matching the 2-extra-expert fold in `weights.fold_shared`
     (shared intermediate 3712 = 2×1856).
6. **Tests:**
   - Synthetic GPTQ/AutoRound/compressed-tensors tensors round-trip, and dequant equals `s·(q−8)` exactly.
   - Header fixtures from the real repos (in the style of `tests/cuda_27b_headers.py`).
   - On B70: load-only smoke for each target checkpoint with a memory report.
7. **Quality gate:** greedy agreement and perplexity on a fixed text set against the BF16 model run on the same B70 for a
   small slice (or published numbers). Results go into the bundle.

---

## 4. WS4 — Kernels (the critical path)

### 4.0 Method for every kernel
**Kernel card** `docs/xpu/kernels/<k>.md`, filled in before any code. It records:
- the op, shapes for A and B, and dtypes;
- the **arithmetic contract** (the exact fp32 operation order that defines its bits, and which invariances must hold:
  row-alone == in-window, chunk-invariant, slice order fixed by shape only);
- input and output layouts, the reference (torch fp32/fp64 plus the Triton baseline), and the tests that pin bits;
- the roofline target on the B70 and the owner.

Each kernel goes through five stages:
1. **T0 (correct):** a Triton version on XPU, or an existing Triton fallback. Must pass all invariance tests. This
   unblocks the engines.
2. **T1 (tuned Triton):** XPU configs (`num_warps`, `grf_mode`, block sizes, block pointers). Often good enough.
3. **N0 (native correct):** SYCL/ESIMD version behind the same `_ext()` signature. Bit-equal to T1 where the
   arithmetic contract allows; otherwise it only has to pass the invariance tests and the tolerance tests against the
   reference.
4. **N1 (native fast):** DPAS, 2D block loads, SLM pipelining, tuned by profile.
5. **Promotion rule:** a native kernel replaces Triton only after it passes the full test tier and beats T1 by ≥10% on
   the recipe's shapes in the B70 bundle.

**Weight-format matrix every matmul/expert kernel must cover:**

| | group 64 | group 128 |
|---|---|---|
| fp16 scales, SYM (AutoRound) | Nemotron primary | Qwen primary |
| bf16 scales, SYM (GPTQ / compressed-tensors) | Nemotron secondary, DFlash2 re-quant | Qwen secondary |

Template parameters: `GS ∈ {64,128}`, `ScaleT ∈ {half, bf16}`. SYM only: the bias `-8·s` is derived in-register and
never stored. The group-sum `xs` is computed
per group of `GS`.

Kernel selection lives in one place per op. Add `tensorfold/xpu/select.py` with `TF_XPU_KERNEL_<OP>=triton|native`
env overrides. The analyst can then A/B any op without code changes.

### K0 — Native build system for XPU (Kernel agent K0, first)
- `src/tensorfold/xpu/build.py` mirrors `cuda/build.py::load`. It calls `torch.utils.cpp_extension.load(name,
  sources, sycl_sources=[...], extra_sycl_cflags=[...])`.
  - Flags: `-fsycl-targets=intel_gpu_bmg_g31`, `-fp-model=precise -ffp-contract=off` (the equivalent of
    `--fmad=false`, required because icpx defaults device code to fast-math), and a per-extension GRF option.
  - ESIMD extensions add `-Xsycl-target-backend=intel_gpu_bmg_g31 "-options '-vc-codegen'"`.
  - Keep the same lock and announcement behaviour. `TORCH_EXTENSIONS_DIR` is per toolchain hash, so a driver or
    compiler bump forces a rebuild.
  - **Builds run AOT in a separate shell** (`tools/xpu/build_ext.sh`, which sources DLE in a subshell). The server and
    test process never has oneAPI on PATH (Triton SIGSEGV, Appendix B). `xpu/build.py::load` imports the prebuilt
    `.so` for the current toolchain hash, and invokes the build script only if it is missing.
- Package data: add `*.sycl`, `*.hpp` for `tensorfold.xpu.**` and update `tests/test_packaging.py`.
- Hello-world extension with an `add` kernel, a subgroup shuffle test, and a joint_matrix/DPAS bf16 16x16 smoke test
  checked against torch. This becomes the `env` suite.
- Common header `xpu/kernels/common.hpp`: subgroup helpers (`sg_xor_sum`), bf16 nibble decode (`(0x4300|q)-128`
  trick → exact bf16), 2D block load wrappers, `fma` without contraction.

### K1 — GDN: gated delta rule (recipe A; **no fallback today**; highest priority)
CUDA sources: `cuda/kernels/gdn.cu` (`tree_kernel`, `replay_kernel`), `gdn_prefill.cu` (`chain_kernel`), wrapper
`cuda/kernels/gdn.py`. Reference: `qwen3_5/cuda/reference.py::_gdn`. Tests: `tests/cuda/test_gdn.py`, and the GDN
parts of `test_qwen27_forward.py` and `test_qwen27_prefill.py`.

Shapes: 48 value heads, dk=dv=128, fp32 state (48,128,128) per layer, 48 layers. Decode is memory-bound on state:
48×64 KB = 3 MB per layer read and write per row.
- **K1.T0 Triton:**
  - `_gdn_step` per (head, value-row block). Each program holds a `[RB,128]` fp32 state tile and iterates the window's
    nodes in schedule order.
  - Branching trees use a slot scratch in global memory (an SLM equivalent is not available in Triton). The CUDA
    `schedule()` host table is reused as-is.
  - `replay` uses the same body over the accepted path. Its multi-layer pointer table becomes a stacked state tensor
    plus a layer index, avoiding pointer casts.
  - `chain` (prefill) runs a sequential scan over chunk rows with the state tile resident.
  - **Arithmetic contract:** copy gdn.cu's op order exactly (explicit fma points, the `warp_sum` reduction tree over
    dk=128 as `tl.sum` with a fixed reduction order). Do serial==tree via a single shared `@triton.jit` inline function
    used by both serial and tree paths.
- **K1.N0/N1 SYCL:** one sub-group of 16 lanes × 8 floats = 128 = dk (Xe2 SIMD16 natural fit). State rows go in
  registers (large GRF), tree slots in SLM, `permute_group_by_xor` butterflies 8→1. Replay across 48 layers stays one
  launch (USM pointer table is fine in SYCL).
- **Bench:** state bandwidth versus B70 peak, for windows of 1, 4 and 12 rows and trees of up to 12 nodes.

### K2 — Mamba-2 prompt scan (recipe B; no fallback)
CUDA: `nemotron_h/cuda/scan_rows.cu` (wrapper `nemotron_h/cuda/mamba.py` ~line 200). Decode `_conv`/`_scan` are
already Triton. Reference: `nemotron_h/cuda/reference.py::mamba`. Tests: `tests/cuda/test_nemotron_kernels.py`
(mamba), and prefill chunk tests in `test_nemotron_forward.py`.
- **K2.T0 Triton:** a chain scan over chunk rows. Grid (64 heads, head_dim/RB), state [RB,128] resident, and exact
  op order from scan_rows.cu (`dt=clamp(softplus)`, `da=exp(a·dt)`, `s=s·da+x·dt·B`, `y=bf16(gz·bf16(C·s+D·x))`). Prompt
  bits only need chunk invariance, not equality with decode `_scan`.
- **N0/N1:** subgroup-per-row-group SYCL. This is lower priority because prefill is a one-off cost.

### K3 — Grouped MoE experts (recipe B; no fallback; biggest native effort)
CUDA:
- `cuda/experts.cu` (`plan_kernel`, `plan_rank`, `plan_offsets`, `plan_scatter`, `expert_kernel`)
- `experts_prefill.cu` (`prefill_kernel`)
- `experts_pack.cu`, `experts.cuh`, wrapper `cuda/experts.py`

Reference: `nemotron_h/cuda/reference.py::moe`, `tests/cuda/test_experts.py::dequant`. Tests: `test_experts.py` and the
MoE parts of `test_nemotron_kernels.py`.

Shapes: 128 routed experts plus 2 shared halves (130 "experts"), top-6+2 = 8 slots per token. up is 1856×2688 with
relu², down is 2688×1856, 4-bit g64.
- **K3a plan:** torch first. A stable `argsort` on expert id plus `bincount`/`cumsum` keeps pair order, with the same
  items-of-16 or 64 tiling. This is exact (integer only). Port to SYCL later only if it shows up in profiles.
- **K3b pack:** write a torch `pack_xpu` that defines an **XPU expert layout** suited to 2D block loads (decided in the
  kernel card; it is not the CUDA mma fragment layout).
- **K3c decode `expert_kernel` T0:** a Triton grouped GEMV. Grid over (expert-item, N-tile). Load nibbles, then
  `fma(xs,b,fma(P,s,acc))` per group in fixed order, matching the lane-matmul contract. The in-kernel xs group sum must
  be computed in the same order for alone and in-window rows.
- **K3d prefill `prefill_kernel` T0:** a Triton grouped GEMM with dequant to bf16 (`fma(q,s,b)` rounded to bf16), then
  `tl.dot` with fp32 accumulation. Use the relu² epilogue for up and the bf16 epilogue for down.
- **N1:** an ESIMD or sycl-tla grouped GEMM with DPAS (bf16, M=8 rows × N=16 × K=16 per Xe2 DPAS per the guide). For
  decode, a GEMV specialisation with 2D block loads of packed weights and on-the-fly decode in registers.

### K4 — 4-bit decode matmul (`qmm.cu::qmm_kernel`) (both recipes)
- **T0 exists:** `qwen3_5/cuda/qmm.py::lane_matmul` on the N-major `[N,K/8]` layout, bit-identical to CUDA on NVIDIA.
  - On XPU the WS3b loader transposes GPTQ `[K/8,N]` into this layout. This is the T0 "XPU layout" until K4.N defines
    `pack_xpu`.
  - `qmm_fast.matmul` takes the Triton path because XPU never calls `tile()` (`weights.py` /
    `nemotron_h/cuda/weights.py`).
- **T0 extension for the format matrix:** generalize `lane_matmul` / `_group_sums` to `GS=128`, fp16 `ScaleT` and `SYM`
  (bias derived in-kernel as `-8*s`). Also add a bf16-weight GEMV for the BF16 lm_head and `in_proj_a/b`: a Triton
  row-invariant kernel with fixed K order.
- **T1:** XPU configs for M ∈ {1, 2–12, 16}. Use split-K by shape only (reuse `split_k`) and the reduce kernel; there
  are no clusters on XPU.
- **N0/N1:** a SYCL/ESIMD GEMV. This is the most important decode kernel: the 27B is about 14 GB of weights read per
  token, so the roofline is about 35 tok/s serial at 500 GB/s.
  - Pack weights into an XPU SoA layout (`pack_xpu`) for 2D block loads.
  - Decode nibbles with the exact bf16 trick `(0x4300|q)-128`.
  - Window rows (≤ 16) go on DPAS **N=16**; weight output rows go on DPAS **M=8**. P_g comes from one K=16 DPAS chain per
    64-group (4 steps), then `fma(xs,b,fma(P,s,acc))` on the vector engine.
  - M=1 variant: also bench a vector-FMA plus sub-group-reduce path.
  - Split-K is fixed by `split_k(n,k)` with an ordered second-pass reduce.
- Gates:
  - **Invariance first:** alone == in-window, at every M.
  - **Bit-equality with T0:** desired but not required. DPAS internal accumulation order may differ from Triton's
    lowering, so record which holds in the kernel card.
- Target: ≥ 80% of 608 GB/s (about 490 GB/s, shown achievable on B70 by TernSYCL/arXiv 2508.06753) at M=1 on
  5120×17408.
- Study first: TernSYCL GEMV, exl3xpu GEMV, PrismML llama.cpp PR #294 (transposed 2D load straight into the DPAS B
  layout, split-K across threads with an SLM reduce).

### K5 — 4-bit prompt matmul (`qmm_prefill.cu`) (both recipes)
- **T0:** `lane_matmul` (it is row- and chunk-invariant; its bits differ from CUDA's prefill, which is fine).
  Alternatively, a Triton dequant-to-bf16 + `tl.dot` kernel matching the CUDA prefill contract (`bf16(fma(q,s,b))` then
  one fp32 chain). Prefer the latter, since it is compute-bound and `tl.dot` uses DPAS.
- **N1:** sycl-tla/ESIMD GEMM with DPAS, dequant in the B-load path, and 128×128-class tiles in SLM. Compare against
  oneDNN int4 matmul as a sanity baseline.
- `qmm_prefill8.cu` (FP8): **deferred**. Refuse `--prefill-fp8` on XPU.

### K6 — Prompt attention (`prefill_attention.cu`) (both recipes)
- **T0 exists:** `prefill_attention.py::triton_attention` (the bit definition). Route XPU there:
  `prefill_attention.attention()` dispatches by device.
- Check: D=256 with BM=BN=64 and 8 warps fits the XPU register and SLM budget (the guide gives the SLM per Xe-core).
- **N1 (later):** an ESIMD flash-attention, only if profiling shows prompt attention is a bottleneck.

### K7 — Remaining Triton-only ops: verify and tune, no rewrite
- Tree attention `_paths/_shared/_tail/_merge` (A).
- Nemotron decode attention `_chunk/_merge`, `_conv`, `_scan`, router/top-k, `_add_moe_norm`, glue (B).
- DFlash2 `_prep/_dconv/_block_attention/_append` (A).
- Samplers.

These are handled by WS3. A kernel agent steps in only if a T1 profile shows any of them above 5% of decode time.

### Kernel priority / dependency order
```
K0 build ─┬─> K4.N (decode GEMV, both)      ─┐
          ├─> K1.N (GDN, A)                  ├─> perf phase
          └─> K3.N (experts, B)             ─┘
T0 path (no K0 needed): K4.T0, K6.T0 (exist) + K1.T0 + K2.T0 + K3.T0 → correctness of A and B
```

---

## 5. WS5 — Engine bring-up per recipe

### Recipe A — Qwen3.8-27B (Engine-A agent)
1. **M-A1 serial.** `--backend xpu --no-drafts`, one stream, Triton everywhere (K1.T0, K4.T0, K5.T0, K6.T0). Run
   `test_qwen27_forward.py` / `test_gdn.py` on XPU, then a real-checkpoint smoke with a coherence check and the WS3b
   quality gate (not bits).
2. **M-A2 DFlash2.** The drafter on XPU (its `fast=True` path uses `qmm_fast.matmul` → Triton when untiled). Gates:
   drafted == serial (`bench_concurrent.py --serial`, token_sha) and `test_qwen27_dflash2_blocks.py`.
3. **M-A3 prefill/cache.** `test_qwen27_prefill.py` (bf16 only) and `test_qwen27_prompt_end_cache.py`.
4. **M-A4 `--parallel N`.** `test_qwen27_multi.py`.
5. **M-A5 perf.** Swap in K4.N, then K1.N, then K5.N.

### Recipe B — Nemotron 30B-A3B (Engine-B agent)
1. **M-B1 serial eager.** Graphs off (WS1.6), K3.T0 experts, K2.T0 scan. Run `test_nemotron_kernels.py`, then
   `test_nemotron_forward.py` except the graph tests.
2. **M-B2 MTP drafting.** drafted == serial for greedy, top-k and min-p (`test_nemotron_forward.py`), plus the
   real-checkpoint test with `TF_NEMOTRON_MODEL`.
3. **M-B3 perf.** K3.N experts, K4.N, then XPU graphs if PyTorch's XPUGraph is available (see guide) and a graphs ==
   eager test passes.

Memory check: the 27B AutoRound at about 19 GB plus DFlash2 (about 1 GB re-quantized) plus KV fits 32 GB. Nemotron
needs about 19 GB. Admission
(`capacity.py`) must report the real B70 free memory.

---

## 6. WS6 — Performance, graphs and polish (after M-A2 and M-B2)
- Profile with unitrace or VTune per the guide, and produce per-op decode time breakdowns for each recipe.
- Graph capture (XPUGraph), once supported, for both recipes. Requires graphs == eager bitwise.
- Launch-overhead reduction: fuse glue kernels if per-token launches dominate. Xe launch latency is in the guide.
- Targets (bandwidth-bound roofline; real numbers come from the guide):
  - 27B serial decode at ≥ 70% of BW roofline.
  - Nemotron serial decode at ≥ 50% (MoE is gather-heavy).
  - Report drafted speedup separately.

## 7. WS7 — Second B70 (future, not scheduled)
- `cuda/comm.py` gets a `Comm` interface with an XCCL implementation (`torch.distributed` backend `"xccl"`).
- `qwen3_5/cuda/engine.py:67` gets a backend-chosen `init_process_group`. Generalize `distributed.py:271-282 ==
  "nccl"`.
- Nemotron `tp.py` and the TCPStore/`ready()` handshake are reused.
- Tests: `test_qwen27_distributed.py` and `test_nemotron_tp.py` on two cards.

## WS9 — Nemotron DFlash drafter (later, after M-B2)
- Source: `SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16` (about 1.67 GB, a BF16 reconstruction of NVIDIA's
  NVFP4 DFlash). It reports 52% acceptance and 186.6 vs 87 tok/s on B70 under vLLM.
- Work:
  - Generalize `qwen3_5/cuda/dflash2.py` (or `drafters/dflash_*`) to Nemotron hidden-state taps.
  - Re-quantize the drafter to SYM g64 at load, like DFlash2.
  - Add `--drafter` support to `nemotron_h`'s XPU engine.
- Gate: drafted == serial bitwise, and the acceptance and speed gain measured in the bundle.

---

## 8. Milestones and parallel schedule

| Phase | Parallel work items (separate branches/agents) | Exit gate (B70 bundle) |
|---|---|---|
| **P0 Setup** | Docs: guides plus AGENTS.md · Harness: bootstrap, runner, suites `env` · K0: build plus hello-SYCL/DPAS · Infra: accel and CLI | `env` suite green; SYCL and Triton smoke pass on the B70 |
| **P1 Triton bring-up** | Harness: device fixture, test rewrite · WS3 Triton port (A and B modules, smoke ladder S0–S7) · K1.T0 **and K1.N0 spike in parallel** (Triton recurrence DEVICE_LOST risk) · K2.T0 + K2.N0 spike · K3.T0 (plan, pack, decode, prefill) · K4.T0 format matrix (g128, fp16, SYM, bf16 GEMV) · WS3b loaders (detect, map, repack) · Infra: engines take a device | `unit-xpu` for A and B kernels green; invariance tests green |
| **P2 Recipes correct** | Engine-A M-A1..A3 · Engine-B M-B1..B2, on the **AutoRound primary** checkpoints (synthetic SYM tiny models before that) | `e2e:*-smoke` on the AutoRound checkpoints: drafted == serial token_sha for both recipes, plus the WS3b quality gate |
| **P3 Native kernels** | K4.N · K1.N · K3.N · K5.N, each an independent branch with kbench | each passes its promotion rule |
| **P4 Perf and polish** | WS6 graphs, fusion, `--parallel`, docs | roofline targets; full bench bundle published |
| **P5 Nemotron DFlash** | WS9 | drafted == serial; speedup reported |
| **P6 Two GPUs** | WS7 | when the second B70 arrives |

---

## 9. Verification (end to end)
- **On the B70, per branch** via `.b70/run.yml`:
  - `env` → `unit-xpu` → relevant `kernels:*` → `e2e:27b-smoke` and `e2e:nemotron-smoke`.
  - Smoke = start `tensorfold serve <ckpt> --backend xpu`, `curl /health`, `/v1/models`, and a chat completion.
  - Then `python tools/bench_concurrent.py ... --serial` asserts drafted token_sha == serial for fixed prompts and seeds.
- **Exactness tests (bitwise):** alone == in-window, chunked == one-shot, drafted == serial, multi-stream == solo, and
  native == Triton where the card says so.
- **Quality tests (tolerance):** vs `reference.py` fp32/fp64, and greedy argmax agreement ≥ 0.9 vs reference on the
  tiny models.
- **CUDA non-regression:** every PR keeps `tests/` host-side green. The CUDA path's behaviour must not change; the
  Integrator spot-checks on an NVIDIA box if one is available, or at least checks that `tests/cuda` collects unchanged.
- **Perf:** `kbench` and e2e numbers in each bundle; `compare.py` flags >3% regressions against `xpu/main`.

## Appendix A — B70 facts that drive this plan (full sourced report → `docs/xpu/B70_NATIVE_KERNEL_GUIDE.md`)

**Hardware.** BMG-G31 (Xe2-HPG):
- 32 Xe-cores, 256 XVE, 256 XMX, 2.8 GHz boost.
- **608 GB/s** GDDR6 (32 GB). Measured achievable is about 443–500 GB/s.
- FP16/BF16 XMX ≈ 183 TFLOPS [derived]; 128 TFLOPS measured on a dense 4096² matmul. INT8 is 367 TOPS.
- L2 is 16 or 24 MB (sources conflict). **Query it at runtime.**
- 256 KB L1/SLM per Xe-core, **128 KB maximum SLM per work-group**.
- GRF: 128 registers × 64 B per thread with 8 threads per XVE, or 256 registers with 4 threads.
- Work-group maximum is 1024. Sub-groups are 16 or 32. `has_fp64=1` (seen on B580); FP64 rate unverified, so benchmark it.

**XMX/DPAS.**
- **Sub-group 16 is required.** Tile is M≤8 × N=16 × K=16 for bf16/fp16 → fp32. K is 32 for int8, 64 for int4, 8 for tf32.
- **No FP8, no FP4** on Xe2. These must be upconverted in registers.

**Replacing CUDA memory instructions.**
- `ldmatrix` and `cp.async` become **2D block loads and prefetches, global memory to registers**, with transpose or VNNI.
- Requirements: 64 B-aligned base; surface width 64–224 B; pitch a multiple of 16 B.
- There is no async global-to-SLM copy.

**Decode GEMV layout insight.**
- Put the **weight rows on DPAS M (8)** and the **window tokens on N (16 lanes)**.
- TensorFold's verify windows are ≤12–16 rows, so one DPAS N tile covers a whole window.
- For M=1 serial, either use vector-FMA plus sub-group reduction, or DPAS with padded N. Bench both.

**Toolchain to pin (P0).**
- Ubuntu 24.04 HWE or 26.04. Kernel **≥ 6.17** (7.x reported working) with the `xe` driver. GuC/HuC `bmg_*` firmware. **ReBAR on**.
- compute-runtime **26.31** / IGC **2.40** / Level Zero loader **≥ 1.32**.
- **PyTorch 2.14.x+xpu** with its bundled `pytorch-triton-xpu`. **Deep Learning Essentials 2026.1**, which provides `icpx`.
  - Do **not** also install the full oneAPI Base Toolkit or IPEX in the same environment; this breaks Triton XPU. IPEX is end of life.
- **Install `ocloc`** (libocloc). Without it, `has_subgroup_2d_block_io` and `has_subgroup_matrix_multiply_accumulate` read False, and Triton silently skips DPAS.
- `env.json` must record all of the above, and the `env` suite must assert both DPAS flags are True.

**Native build.**
- `torch.utils.cpp_extension.load(..., sycl_sources=[...], extra_sycl_cflags=[...])`, or `SyclExtension` (Linux, PyTorch ≥ 2.8).
- Set `TORCH_XPU_ARCH_LIST=bmg`. AOT target **`intel_gpu_bmg_g31` explicitly** (`-device bmg` may resolve to G21).
- ESIMD needs `-Xsycl-target-backend=intel_gpu_bmg_g31 "-options '-vc-codegen ...'"`.
- Large GRF via `-ze-opt-large-register-file` or `grf_size<256>`. **Check in the IGC dump that it took effect.**
- Launch on `c10::xpu::getCurrentXPUStream().queue()`. Register ops with `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL(..., XPU, ...)`; an XPU pybind `_ext()` shim is also acceptable.

**Determinism.**
- icpx defaults to **fast fp-model for device code**. Always pass `-fp-model=precise -ffp-contract=off` (the `--fmad=false` equivalent) and use explicit `sycl::fma`.
- No float atomics in reductions. Split-K, tile, sub-group size and GRF are fixed per shape, with two-pass fixed-order reductions.
- Pin compiler, IGC and driver versions. A change to any of them invalidates the "bits" baselines, and the runner records them.

**Graphs.**
- `torch.xpu.XPUGraph` / `torch.xpu.graph` exist (2.11+, improved in 2.14), but capture has had correctness bugs (vLLM gibberish on B70).
- Gate on a graphs == eager bitwise test before enabling graphs for Nemotron.

**Pitfalls.**
- **Single allocations > 4 GB fail** even with `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`. Audit the loaders: no tensor ≥ 4 GB. The largest expected is the 27B head at about 0.65 GB packed, but check any concatenated or flattened staging buffers in `direct_read`/`weights`.
- Host RAM shadows XPU allocations (torch 2.14 / kernel 7.1.8): host RAM must be ≥ 64 GB.
- Driver resets on out-of-bounds bugs. The runner must detect a wedged GPU (`xpu-smi` health) and stop the queue.
- `torch._weight_int4pack_mm` is wrong at M=1 on XPU; never use it as a baseline. Keep `SYCL_CACHE_PERSISTENT=0` and prefer AOT.

**Repos to study, per kernel.**
- K4/K5: TernSYCL (B70 SIMT DPAS and 2D IO GEMV), exl3xpu (B70 ESIMD GEMV/GEMM, AOT recipe), vllm-xpu-kernels `int4_gemm_w4a16`, sycl-tla `02_bmg_gemm_mixed_dtype`.
- K3: sycl-tla `10_bmg_grouped_gemm_mixed_dtype` / `12_xe20_moe_gemm`; the vllm-xpu-kernels MoE grouped GEMM.
- K1: GDN kernels in vllm-xpu-kernels and sgl-kernel-xpu. Their arithmetic differs, so use them for structure only.
- K2: kernels-community mamba-ssm XPU build.
- K6: sycl-tla `06_bmg_flash_attention`.

**Profiling.**
- `unitrace -d --chrome-kernel-logging`, VTune GPU Hotspots (XMX utilization).
- `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=...`; grep for `dpas`, `load_block2d` and spill counts. Optionally collected per run by the runner.
## Appendix B — Triton-on-XPU facts that drive this plan (full report → `docs/xpu/TRITON_XPU_GUIDE.md`)

**Versions.** torch **2.14.1+xpu** pins `triton-xpu~=3.8.0` and Intel runtime 2026.1. G31 support arrived in triton-xpu
3.7.1. Pin both exactly.

**Runtime/build-environment conflict (important).** Having oneAPI `setvars.sh` or `icpx` on PATH at runtime makes
Triton XPU crash with SIGSEGV (SYCL runtime mismatch, issue #8200). Therefore:
- Native SYCL extensions are built **ahead of time in a separate build shell** by `tools/xpu/build_ext.sh`, which sources
  DLE 2026.1 in a subshell and writes `.so` files into `build/xpu-ext/<toolchain-hash>/`.
- The **runtime environment never sources oneAPI**. `xpu/build.py` loads the prebuilt `.so`, and in dev mode it shells
  out to the build script; it does not JIT inside the server process.
- The DLE version must match torch's bundled SYCL runtime (2026.1).

**Warp size is per kernel.** Kernels with a DPAS-lowerable `tl.dot` compile at **16** lanes; all others default to
**32**. So `num_warps` × lanes, and therefore the reduction trees, differ from CUDA, and can differ between two kernels
that compute the "same" sum. Example: `xs` group sums produced by `_add_rmsnorm`, `_swiglu` and `_merge` versus
`_group_sums`.
- Add **cross-producer equality tests**.
- Record the compiled `threads_per_warp`, `n_regs` and `n_spills` for every kernel in the bundle.

**Options.**
- `grf_mode ∈ {'default','128','256','auto'}`. The default escalates to large GRF at ≥ 1 KB of spill. Use `'256'` for
  big accumulators: per-lane fp32 load is `M·D/(num_warps·16)`; above about 100, raise `num_warps` or use 256.
- `enable_fp_fusion=False` is supported. **Set it on every bit-sensitive kernel**; today only `affine.py` does.
- `num_ctas` must be 1. The work-group maximum is 1024 threads.

**tl.dot.**
- DPAS needs N ≥ 16. Smaller N **silently falls back to FMA**, which is a performance cliff and changes accumulation
  order. Assert `#triton_intel_gpu.dpas` in the TTGIR of every dot kernel.
- Default precision is tf32 for fp32 inputs; pass `ieee` or cast to bf16.
- FP8 is emulated in software, and `.to(float8e4nv)` is RTNE only.
- Prefer tensor descriptors or block pointers for dot operands (>2× faster); TensorFold uses tensors of pointers
  throughout, which is a perf item and not a correctness one.

**Open bugs that hit this repo directly.**
1. **Recurrence-with-persistent-state kernels → `DEVICE_LOST` on B70** (#6658). This pattern is GDN (K1.T0), Nemotron
   `_scan`/`_conv` and the K2 scan. Mitigations: smaller tiles, `grf_mode='256'`, and moving state stores out of the
   loop. **K1 and K2 native SYCL spikes therefore start in P1 in parallel with T0, not after.**
2. **BLOCK_M=16 `tl.dot` miscompile** (#8121). This affects `lane_matmul` BM=16 and `affine_kernels.matmul`. Test every
   BM, and use a BM ≥ 32 floor if it reproduces.
3. Constexpr-stride miscompile (#5581): pass strides as `tl.int64`.
4. **`scf.while` crash** (#8189): rewrite `kernels/attention.py::_paths` `while` loops as a bounded `for` with
   predicates.
5. **IGC abort on large `tl.static_range`** (IGC #446): `sampler._keyed` `static_range(1,K≤256)` → runtime `range`.
6. int→bf16 and fp16→bf16 cast crashes: route through fp32.
7. 3.8 predicated-load slowdown: A/B `TRITON_INTEL_PREDICATED_LOAD=0`.
8. Intel knobs are **not part of the Triton cache key**: the runner purges `~/.triton/cache` whenever env knobs or the
   toolchain change.

**Pointer tricks.**
- `draft_attention.py` int64 → `tl.pointer_type` tables. Level Zero addresses may be **≥ 2^63**, which overflows
  `torch.tensor(..., int64)` (vllm-xpu-kernels PR #570); use `_s64(p)` two's-complement conversion. Indirect-access
  residency is unverified. Day-one probe; fallback is explicit base-pointer arguments.
- `kernels/attention.py` base+offset: same residency probe.

**Other risks.**
- `mamba._conv_commit` relies on `tl.debug_barrier` as a global-memory fence (unverified on XPU). Double-buffer BASE
  instead.
- fp64/uint64 sampler (`nemotron_h/cuda/sampler.py::_keyed`): `has_fp64` is True but the rate is unknown. Check against
  `engine/exact_sampling.py` over 1e5 draws.
- Precompute rope cos/sin on the host for `_attn_prep`, as DFlash2 already does, to avoid sin/cos range-reduction
  differences.

**Torch eager on XPU is not deterministic by default.** oneDNN split-K in `F.linear` (torch-xpu-ops #5524), so XPU
engines set `torch.use_deterministic_algorithms(True)`. There is also an open NaN report in bf16 matmul on B70
(pytorch #199179). Avoid torch matmuls on exactness paths.

**Smoke-test ladder** (becomes the `env` and `triton-smoke` suites, in order):
- **S0 probes:**
  - device properties, including `has_subgroup_matrix_multiply_accumulate` and `has_subgroup_2d_block_io`;
  - `data_ptr ≥ 2^63`, int64→pointer load, and the `debug_barrier` hazard;
  - uint64 `_mix` against numpy, fp64 exp/log, and fp8 cast range.
- **S1** glue elementwise/norm.
- **S2** `group_sums` + `lane_matmul` at M ∈ {1,2,15,16,17,33,65,129} × BM ∈ {16,32,64,128} × sk.
- **S3** affine 2–8 bit.
- **S4** nemotron route.
- **S5** attentions: prompt chunked vs one-shot, tree, nemotron, draft.
- **S6** mamba conv/scan, with a sync after each launch and short timeouts.
- **S7** sampler against the host rule.

Every bitwise check compares `int16`/`int32` views, warm and cold cache, over 20 repeats.
