# TensorFold → Intel XPU (Arc Pro B70 / Battlemage) — Port Plan

## Context

TensorFold (upstream `ashhart/TensorFold` v0.6.0, fork `Thotheris/TensorFold-XPU-B70`) serves LLMs on MLX and CUDA with
**exact** speculative decoding: a drafted token is accepted only if it bit-equals what the same engine produces serially.
The goal is a third backend, `--backend xpu`, on one Intel Arc Pro B70 (Xe2 / Battlemage, 32 GB, Linux + `xe` driver,
PyTorch XPU), with a second B70 planned later. The first targets are two recipes:

- **A: Qwen3.8-27B W4A16 AutoRound + DFlash2** (`families/qwen3_5`): a GDN/attention hybrid with no CUDA graphs.
- **B: Nemotron 3.5 Lightning 30B-A3B W4A16 AutoRound + MTP** (`families/nemotron_h`): Mamba-2, attention and
  128-expert MoE, captured as CUDA graphs.

Development happens on the `xpu/main` trunk, one topic per commit. The B70 box pulls each new head, builds and tests
it, and pushes results to a `results` branch. The results decide the next work. Then the loop repeats.

**Three findings from the code maps shape this plan:**
1. The HTTP, OpenAI, scheduler and drafting layers are device-free. CUDA coupling sits in about 15 CLI/registry lines,
   5 engine constructors, `capacity/direct_read/memory_gate/comm/build`, and graphs.
2. Both recipes depend on five native nvcc extensions. Bit-exact Triton replacements already exist for the 4-bit decode
   matmul (`qwen3_5/cuda/qmm.py::lane_matmul`, asserted equal in `tests/cuda/test_qmm.py::test_27b_triton_bits`) and for
   prompt attention (`triton_attention`, the bit definition). **No portable version exists** of the GDN tree, replay and
   chain kernels, the grouped MoE experts with their plan and pack kernels, or the Mamba prompt scan.
3. Exactness on XPU means **self-consistency** (serial == drafted, alone == in-window, chunked == one-shot). It does not
   mean bit-equality with CUDA. Upstream already accepts per-backend serial bits.

### Source audit and current execution order

The [kernel atlas](reports/tensorfold-kernel-atlas-2026-10-02.html) surveys original TensorFold v0.6.3
(`9356df5c424b0c36b7737e37873a6f968b08de79`) against this fork's v0.6.0 source baseline
(`c4646171139ee8a3c38103eaa1699dad226ec12b`). It confirms the three portable compute gaps above. Its broader
CUDA/MLX inventory is reference material; the scheduled scope remains recipes A/B and symmetric INT4. Do not merge
newer upstream code, enable additional families, or import alternative quantization paths without the owner's request.

Verified bundles and pending work are recorded in [STATUS.md](STATUS.md). At this revision:

- `c32b417` qualified the register-reporting fix plus glue/prompt-attention checks.
- `da2a34f` qualified K4.T0 SYM g64/g128, fp16/bf16 scales, BF16 GEMV and cross-producer activation sums.
- Later recipe-shape tests (`55d2095`) and launch-timing probes (`f569957`) have no bundle in the inspected result index.
  A pass on an ancestor does not qualify these later changes or the current head.

Resume in this order, using the [complete kernel-engineer prompt](prompts/kernel-engineer.txt):

1. Qualify the current head and distinguish missing suite coverage from numerical regression in the harness.
2. Complete one bounded K4.T1 tuning pass, measuring host submission, device timing, spills and verify windows.
   Preserve the correct T0 baseline; do not delay missing correctness kernels indefinitely to chase a roofline.
3. K0 isolated native build system, then K1 GDN T0 (with an N0 safety alternative when needed).
4. Remaining recipe A portability and K4 stored-INT4/BF16 head-row adapters; K5 prompt GEMM.
5. K3 grouped experts, then K2 prompt scan and remaining Nemotron portability.
6. Hand off qualified kernel contracts, adapters and measured priorities. WS5 engine wiring and N1 optimization
   require a separate request; this kernel task does not authorize them.

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
- BF16 lm_head (248320×5120×2 = 2.543 decimal GB for a full uncached read) needs a bf16 GEMV path.
  Its fraction of total decode traffic must be measured. An optional later `--xpu-head-int8` re-quantization flag
  is a quality trade-off and stays opt-in.
- Memory: 27B is about 19 GB + DFlash2 about 1 GB + KV (64 KiB/token bf16 → 64k context ≈ 4 GB) and fits 32 GB.
  Nemotron is about 19 GB with tiny KV.

**Strategy:** bring each recipe up **correct-first on Triton**, writing new Triton kernels for the three gaps. Only then
replace hot paths with **native SYCL/ESIMD kernels** that are measured against the Triton baseline. Each native kernel
sits behind the same Python `_ext()` signature, so engines don't change when it is swapped in.

---

## 0. Repository, branches and the B70 loop

### Branches (fork `Thotheris/TensorFold-XPU-B70`)
- `upstream-main`: mirrors `ashhart/TensorFold` main. It is refreshed by hand and only ever merged forward.
- `xpu/main`: the trunk. Work is committed here directly; every push gets a B70 run, and a red run is fixed forward.
- `xpu/<topic>`: optional and short-lived, for throwaway experiments only. Merged into `xpu/main` or deleted.
- `main`: a milestone snapshot, updated from `xpu/main` by the owner.
- `results`: an orphan branch that holds only result bundles. It never merges into code branches.

### How a commit asks the B70 to run it
`xpu/main` carries a standing **`.b70/run.yml`**, edited in the same commit when a change needs other suites:
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

### Analysis step
- It reads `results/index.jsonl` and new bundles, and runs `tools/xpu/compare.py <bundle> <baseline>`. That tool reports
  regressions, new failures, bitwise-check breaks and perf deltas by kernel and end to end.
- An absent current artifact that passed in the baseline means missing coverage, not a measured mismatch. Compare matching cases
  and arithmetic/toolchain contracts; retain missing required coverage as a qualification blocker. Until the
  comparator is corrected, inspect the suite list and JSON manually rather than accepting summary labels alone.
- It updates `docs/xpu/STATUS.md` on `xpu/main` with the regressions, failures and the next tasks.
- On `DEVICE_LOST` or a runner STOP flag, stop GPU submissions. Record the failure and prepare a minimal repro
  locally; independent host work may continue. Do not clear STOP or submit another GPU experiment until the
  operator restores device health.

---

## 1. Work areas

The fork is worked serially by one person and usually one agent. These areas label work, not owners; there is no
per-agent branch or file ownership (AGENTS.md §7). The model column is a suggestion for an agent working that area.

| Area | Paths | Model |
|---|---|---|
| **Infra** (WS1) | `src/tensorfold/accel.py`, `cli*.py`, `families/__init__.py`, `serve_options.py`, `cuda/{capacity,memory_gate,direct_read,build}.py`, new `xpu/` package skeleton | Sonnet |
| **Harness** (WS2) | `tools/xpu/**`, `tests/conftest.py`, `tests/cuda/conftest.py`, the device fixture, `tests/xpu/**` | Sonnet |
| **Triton-port** (WS3) | existing Triton modules used by recipes A and B (portability fixes only) | Sonnet |
| **Kernels K0–K7** (WS4) | one kernel family each: new Triton/SYCL sources and their test and bench files | Opus |
| **Loader** (WS3b) | `src/tensorfold/xpu/quant/**`, the `QUANT_METHODS["xpu"]` entries, quant tests | Sonnet |
| **Engine-A / Engine-B** (WS5) | `families/qwen3_5/**`, `families/nemotron_h/**` engine wiring | Opus |
| **Status** | `results` bundles, `docs/xpu/STATUS.md` | Sonnet |
| **Docs** | `docs/xpu/*.md` guides (kept current as findings come in) | Sonnet |

Every agent reads `AGENTS.md` (the fork root) first. It holds the rules: exactness contract, the trunk workflow, `.b70/run.yml`,
never touch the CUDA path's behaviour, and how to read results.

---

## 2. Documentation deliverables

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
- The research guides and kernel map are present. Keep them current through small documentation commits on
  `xpu/main`, with exact source revisions and result-bundle paths.
- `docs/xpu/KERNEL_MAP.md`: the kernel inventory for recipes A and B (prefill, decode, verify, sampling): kind, CUDA
  features, fallback, owner and status. Sections 4–5 below are its seed.
- `docs/xpu/kernels/<kernel>.md`: one **kernel card** per port target (template in §4.0).
- `docs/xpu/HARNESS.md`: B70 box setup, runner, suites and result schema.
- `docs/xpu/STATUS.md`: updated after each B70 run that changes the picture.
- `AGENTS.md`: agent operating rules (above).
- `reports/tensorfold-kernel-atlas-2026-10-02.html`: complete upstream CUDA/MLX survey and Intel translation.
- `prompts/kernel-engineer.txt`: copyable execution prompt for the bounded WS3/WS4 kernel task.

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
   - Checks: ReBAR enabled, `xpu-smi discovery`.
   - Host RAM: the box has **32 GB**. **Warn** below 64 GB rather than fail, and offer to create a 32 GB swap file
     (`--apply`).
   - `env` suite probe `host_ram_shadow`: allocate 4/8/16 GB on XPU; record the `MemAvailable` and `Committed_AS`
     deltas in `env.json`.
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

### WS3b — W4A16 checkpoint loaders (Loader; `src/tensorfold/xpu/quant/**`, the family `QUANT_METHODS["xpu"]` entries, `tests/test_xpu_quant_*.py`)
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
   - Loaders take a `bf16` path for those. All exactness-path projections, including small modules and lm_head,
     use K4's row-invariant BF16 kernel or another explicitly qualified XPU kernel. Never use `F.linear` or
     oneDNN matmul on these paths; deterministic torch mode does not waive that prohibition.
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
never stored. K4 uses canonical fp32 `xs` sums over 64 inputs. g128 weights combine adjacent sums with one
fixed-order fp32 add; fused producers share this definition. K3 must document and test its matching grouping.

Kernel selection lives in one place per op. Add `tensorfold/xpu/select.py` with `TF_XPU_KERNEL_<OP>=triton|native`
env overrides. The analyst can then A/B any op without code changes.

### K0 — Native build system for XPU (Kernel agent K0, first)
- `src/tensorfold/xpu/build.py::build_aot` runs native compilation through `tools/xpu/build_ext.sh` in the
  isolated build environment. Its runtime `load` reads compatible prebuilt artifacts; it never runs icpx in the
  server/container. Use the pinned PyTorch SYCL extension API in the build process only.
  - Flags: `-fsycl-targets=intel_gpu_bmg_g31`, `-fp-model=precise -ffp-contract=off` (the equivalent of
    `--fmad=false`, required because icpx defaults device code to fast-math), and a per-extension GRF option.
  - ESIMD extensions add `-Xsycl-target-backend=intel_gpu_bmg_g31 "-options '-vc-codegen'"`.
  - Keep the same lock and announcement behaviour. `TORCH_EXTENSIONS_DIR` is per toolchain hash, so a driver or
    compiler bump forces a rebuild.
  - **Builds run AOT in a separate shell** (`tools/xpu/build_ext.sh`, which sources DLE in a subshell). The server and
    test process never has oneAPI on PATH (Triton SIGSEGV, Appendix B). `xpu/build.py::load` imports the prebuilt
    `.so` for the current toolchain hash; a missing/incompatible artifact fails with build instructions. An explicit
  host development build may invoke the script in a separate process, outside the runtime container.
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
48×128×128×4 = 3.146 decimal MB of state per layer; one read plus write across 48 layers is 0.302 GB per
serial step, before tree scratch and replay traffic.
- **K1.T0 Triton:**
  - `_gdn_step` per (head, value-row block). Each program holds a `[RB,128]` fp32 state tile and iterates the window's
    nodes in schedule order.
  - Branching trees use a slot scratch in global memory (an SLM equivalent is not available in Triton). The CUDA
    `schedule()` host table is reused as-is.
  - `replay` uses the same body over the accepted path. Its multi-layer pointer table becomes a stacked state tensor
    plus a layer index, avoiding pointer casts.
  - `chain` (prefill) runs a sequential scan over chunk rows with the state tile resident.
  - **Arithmetic contract:** identify CUDA's op order, FMA points and stored roundings, then define one explicit
    XPU reduction over dk=128 with pinned lane mapping. `tl.sum` alone does not prove equivalence to CUDA's
    shuffle tree. XPU serial, tree, replay and chain use one shared step/reduction body; prove the required
    invariances and reference tolerance. CUDA bit parity is not required. Do not substitute a reassociated
    history/parallel-scan decomposition without re-establishing every relevant contract.
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
- **T0 format matrix qualified on `da2a34f`:** SYM g64/g128, fp16/bf16 scales and BF16 GEMV for lm_head and
  `in_proj_a/b`. Kernels live in `xpu/kernels/qmm/`; canonical `xs` uses 64-wide sums. Preserve the affine CUDA
  path. See [the card](kernels/qmm.md) for arithmetic, coverage and measurements.
- **T1:** one shape-only `XPU_CONFIG` per weight shape/format, tested across M ∈ {1, 2–12, 16} and the invariance
  sweep. BM, BN, split-K, subgroup width, GRF mode and num_warps never depend on runtime M. Run a bounded
  offline tuning pass only after current-head correctness and timing probes are qualified.
  - Large Qwen INT4 shapes report 5.8–9.8% of peak and 5120–9472 spill bytes on `da2a34f`; reduce register
    pressure and measure tile/split trade-offs. The BF16 Qwen head reports 356.2 GB/s, 7140 us and zero spills.
  - Small shapes share a 117–125 us wrapper timing floor. Separate host submission, event/device execution,
    allocations, group-sum and reduction costs before calling it GPU launch latency. Retain cold/warm cache
    tests and report the bytes model; GB/s is an effective rate, not measured DRAM traffic.
  - Measure realistic verify windows as well as M=1: total window time, time per verified row, launch count,
    state/replay costs for recurrent ops. End-to-end time per emitted token belongs to WS5/WS6 once acceptance
    and draft/commit costs can be measured; do not infer it from window size.
- **Head-row adapters (before WS5):** add stored-INT4 and BF16 row selection for DFlash2/MTP without CUDA `tile()`.
  `qmm_fast.rows` and `matmul_rows` currently assume tiled weights. Test contiguous, disjoint and boundary spans
  against corresponding full-head outputs, retaining the parent head's arithmetic plan when selecting rows.
  Single-GPU XPU must reject unsupported `matmul_partial` clearly instead of entering a CUDA extension.
- **N0/N1:** a SYCL/ESIMD GEMV. Build a recipe traffic model from loaded tensors, including scales, the full
  2.543 GB BF16 head, recurrent state, KV and workspace. A bandwidth-only tokens/s ceiling is a labeled
  scenario, not a prediction from checkpoint size. Rank native work by measured decode share after WS5.
  - Pack weights into an XPU SoA layout (`pack_xpu`) for 2D block loads.
  - Decode nibbles with the exact bf16 trick `(0x4300|q)-128`.
  - Window rows (≤ 16) go on DPAS **N=16**; weight output rows go on DPAS **M=8**. P_g comes from one K=16 DPAS chain per
    64-group (4 steps), then `fma(xs,b,fma(P,s,acc))` on the vector engine.
  - Benchmark vector-FMA/subgroup-reduce and padded DPAS as separate whole-contract candidates. Do not
    dispatch between them by runtime row count unless serial/window bitwise equivalence is established;
    launch constants remain shape-only.
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
- **T0:** Triton dequant-to-BF16 + `tl.dot` with one fixed fp32 accumulation chain over K. Define prompt weights
  as `bf16_rne(fma(fp32(q), fp32(s), -8*fp32(s)))`, preserving stored fp16/bf16 scales before conversion to fp32.
  This is a separate contract from decode's group-dot-plus-bias arithmetic; prompt bits need chunk/resume
  invariance, not equality with decode. Pin tiles by weight shape and assert DPAS in compiled TTGIR.
- `lane_matmul` can be a temporary correctness fallback only if explicitly selected and separately qualified;
  never switch prompt arithmetic based on chunk size.
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
K4.T0 verified → current-head coverage/timing → bounded K4.T1 → head-row adapters
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
- Measure host submission, device time, allocations and launch counts independently. Fuse glue and reuse
  canonical activation sums if profiles show a benefit; preserve stored roundings and producer equality.
- Candidate ideas from the v0.6.3 atlas: grouped projection launches and accepted replay fused into verification.
  Implement XPU-specific equivalents only after the unfused baseline and state/commit tests are green. No
  automatic upstream merge is implied; CUDA dependent-launch/cluster mechanisms are not Xe2 APIs.
- Preserve recurrent state residency and compare full window cost against accepted/emitted tokens. Report
  serial token latency, draft cost, verify cost, commit/replay cost, acceptance, TTFT and time per emitted token.
  A lower per-row verify cost alone does not prove an end-to-end speedup.
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

## WS10 — EXL3 on XPU (later, after P4)
Full analysis and phases: [EXL3_PORT.md](EXL3_PORT.md).
- **Why:** better quality per bit. 27B at 3–4 bpw is about 11–14 GB, which frees device and host RAM. Upstream already
  has the EXL3 CUDA path, and the user's `exl3xpu-b70-lab` proves bit-exact codebook decode on B70 (M=1 ≈ 481 GB/s).
- **Key constraint:** exl3xpu is deterministic but **not row-invariant**. It switches kernel and split-K by M and uses
  oneDNN/int8 for prefill. The port keeps exl3xpu's decode (bit-exact W_q) and rebuilds the GEMM around TensorFold's
  shape-only plan, fixed-order sums and fixed-tile prompt GEMM.
- **Phases:**

  | Phase | Work |
  |---|---|
  | E0 | prereqs (WS1/WS2/K0 with AOT+JIT validated) |
  | E1 | loader gating (`exl3` for `qwen3_5` on XPU, `layout="stored"`) |
  | E2 | bit-exact decode/reconstruct (port from exl3xpu, MIT attribution, no `-ffast-math`) |
  | E3 | row-invariant ESIMD DPAS linear (fixed MB, shape-only split) |
  | E4 | chunk-invariant prompt GEMM (Triton-XPU, then ESIMD) |
  | E5 | recipe A-EXL3: `turboderp/Qwen3.8-27B-exl3` 3.00/4.00bpw + DFlash2 |
  | E6 | optional EXL3 experts |

- **Rule:** never push, open PRs or file issues to `0xSero/exl3xpu` (or `ashhart/TensorFold`). Learn from and copy
  MIT code with attribution only.

---

## 8. Milestones and parallel schedule

| Phase | Work items (in any order on `xpu/main`) | Exit gate (B70 bundle) |
|---|---|---|
| **P0 Setup** | Docs: guides plus AGENTS.md · Harness: bootstrap, runner, suites `env` · K0: build plus hello-SYCL/DPAS · Infra: accel and CLI | `env` suite green; SYCL and Triton smoke pass on the B70 |
| **P1 Triton bring-up** | Harness: device fixture, test rewrite · WS3 Triton port (A and B modules, smoke ladder S0–S7) · K1.T0 **and K1.N0 spike in parallel** (Triton recurrence DEVICE_LOST risk) · K2.T0 + K2.N0 spike · K3.T0 (plan, pack, decode, prefill) · K4.T0 format matrix (g128, fp16, SYM, bf16 GEMV) · WS3b loaders (detect, map, repack) · Infra: engines take a device | `unit-xpu` for A and B kernels green; invariance tests green |
| **P2 Recipes correct** | Engine-A M-A1..A3 · Engine-B M-B1..B2, on the **AutoRound primary** checkpoints (synthetic SYM tiny models before that) | `e2e:*-smoke` on the AutoRound checkpoints: drafted == serial token_sha for both recipes, plus the WS3b quality gate |
| **P3 Native kernels** | K4.N · K1.N · K3.N · K5.N, each one topic with kbench | each passes its promotion rule |
| **P4 Perf and polish** | WS6 graphs, fusion, `--parallel`, docs | roofline targets; full bench bundle published |
| **P5 Nemotron DFlash** | WS9 | drafted == serial; speedup reported |
| **P6 Two GPUs** | WS7 | when the second B70 arrives |
| **P7 EXL3** | WS10 (E0–E6, see EXL3_PORT.md) | A-EXL3: drafted == serial; row/chunk invariance; quality gate |

---

## 9. Verification (end to end)
- **On the B70, per pushed head** via `.b70/run.yml`:
  - `env` → `unit-xpu` → relevant `kernels:*` → `e2e:27b-smoke` and `e2e:nemotron-smoke`.
  - Smoke = start `tensorfold serve <ckpt> --backend xpu`, `curl /health`, `/v1/models`, and a chat completion.
  - Then `python tools/bench_concurrent.py ... --serial` asserts drafted token_sha == serial for fixed prompts and seeds.
- **Exactness tests (bitwise):** alone == in-window, chunked == one-shot, drafted == serial, multi-stream == solo, and
  native == Triton where the card says so.
- **Quality tests (tolerance):** vs `reference.py` fp32/fp64, and greedy argmax agreement ≥ 0.9 vs reference on the
  tiny models.
- **CUDA non-regression:** every PR keeps `tests/` host-side green. The CUDA path's behaviour must not change; the
  owner spot-checks on an NVIDIA box if one is available, or at least checks that `tests/cuda` collects unchanged.
- **Perf:** `kbench` and e2e numbers in each bundle; `compare.py` flags >3% regressions against matching baseline
  cases. Missing required cases block qualification but are reported separately from observed bitwise failures.
  Preserve timing methodology, cache conditions, byte estimates and `n_regs_source`; inferred GRF budget is
  not compiler-measured register usage. Requalify changed arithmetic/toolchain contracts before perf comparison.

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
- Benchmark vector-FMA/subgroup reduction and padded DPAS as separate invariant candidates. The selected
  weight-shape plan must serve both serial and verify rows; do not choose arithmetic by runtime M.

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
- **Single allocations > 4 GB fail** even with `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`. Audit the loaders: no
  tensor ≥ 4 GB. The primary Qwen head is BF16, 2.543 decimal GB; check expert concatenations, stacked states
  and flattened staging buffers in `direct_read`/`weights` as well.
- Host RAM may shadow XPU allocations (torch-xpu-ops #5428: torch 2.14 / kernel 7.1.8). **The box has 32 GB**, so
  measure it with the `host_ram_shadow` probe.
  - Commit-only: swap or overcommit settings cover it.
  - Resident: `capacity.py` must subtract the shadow from the host budget, and the server caps its XPU budget
    (context/KV) to fit.
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
