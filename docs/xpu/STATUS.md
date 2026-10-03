# XPU port status

This snapshot is based on inspected result bundles through `da2a34f`, not on an end-to-end recipe run.
Refresh it from `origin/results:index.jsonl` before starting kernel work. The
[execution prompt](prompts/kernel-engineer.txt), [plan](PORT_PLAN.md) and
[upstream atlas](reports/tensorfold-kernel-atlas-2026-10-02.html) describe the next work.

## Verified evidence

| Commit | Bundle | Executed suites | Qualification |
|---|---|---|---|
| `c32b417` | `runs/xpu--main/c32b417-20261003T050422Z` | env, triton-smoke, unit-host, unit-xpu, kernels:glue, kernels:prefill-attention | pass; register reporting fix and selected migrated checks |
| `da2a34f` | `runs/xpu--main/da2a34f-20261003T054251Z` | env, unit-host, unit-xpu, kernels:qmm | pass on executed suites; unit-xpu 37 passed, QMM 31 passed |

K4.T0 supports symmetric INT4 g64/g128, fp16/bf16 scales and row-invariant BF16 GEMV. The inspected QMM
results report bitwise checks passing, DPAS and 16 compiled lanes. Cross-producer canonical 64-wide sums
passed. The [QMM card](kernels/qmm.md) records arithmetic and exact coverage.

| K4 case, M=1 | Wrapper time (us) | Effective GB/s | % of 608 GB/s | Spill bytes |
|---|---:|---:|---:|---:|
| Qwen down, 5120 x 17408 SYM | 1302 | 35.3 | 5.8 | 9472 |
| Qwen gate/up, 17408 x 5120 SYM | 775 | 59.3 | 9.8 | 5120 |
| Qwen qkv, 10240 x 5120 SYM | 764 | 35.4 | 5.8 | 9472 |
| Qwen head, 248320 x 5120 BF16 | 7140 | 356.2 | 58.6 | 0 |
| Nemotron head, 131072 x 2688 BF16 | 2023 | 348.4 | 57.3 | 0 |

Times are per-wrapper-call means of 20 queued calls per timing sample; SYM includes the split reduction.
Effective GB/s uses estimated weights/scales/input/output bytes, not measured DRAM traffic. Small projections
share a 117–125 us floor. Its attribution to host submission, allocations or GPU launch/execution is
**[UNVERIFIED]** pending inspection of the timing probe. BF16 head register counts are inferred GRF budgets,
not measured compiler usage; preserve `n_regs_source`.

## Coverage limits and pending changes

- `55d2095` adds recipe-shape invariance tests, and `f569957` adds host-submit/device timing probes. No bundle
  for either appears in the result index inspected for this snapshot. Later heads are not qualified by ancestry.
- `da2a34f` did not request glue/prompt-attention suites. Its summary labels their absent JSON artifacts
  "Bitwise break" because `compare.py` treats missing files as failures. Those entries are missing coverage,
  not observed numerical mismatches. Separate these categories in the harness and rerun required suites.
- The migrated host/device suites do not cover every recipe path. Neither recipe has an inspected e2e bundle;
  serial/drafted token equality, checkpoint quality and model throughput remain unqualified.
- The atlas is upstream source research: v0.6.3 `9356df5` compared with v0.6.0 `c464617`. It does not authorize
  an upstream merge, new families or new checkpoint formats.

## Next work, in order

1. Qualify current-head tests/timing and correct missing-versus-failed coverage reporting.
2. One bounded K4.T1 pass: reduce large Qwen INT4 spills; measure head bandwidth and realistic verify windows.
3. K0 isolated AOT/spir64 build and native loader/smoke; no compiler in the Triton runtime.
4. K1 GDN tree/replay/chain T0; prepare N0 alternative if recurrence instability requires it.
5. Recipe A portability, deterministic BF16/stored-INT4 head-row adapters and K5 prompt GEMM.
6. K3 expert plan/pack/decode/prompt, then K2 prompt scan and Nemotron portability.
7. Hand off exact-SHA kernel coverage and loader/adapter prerequisites. WS5/N1 require a separate request.

K0 Python build/loader, K1, K2 prompt scan and K3 have no XPU implementation in the inspected tree.
Head-row selection in `qmm_fast.rows`/`matmul_rows` still assumes CUDA tiled weights; ordinary XPU matmul
support does not fix those paths. Preserve the parent arithmetic plan in selected-row tests.

Native priorities remain provisional until WS5 supplies a per-op decode profile. Future WS6 candidates are
projection grouping, producer fusion, state residency and replay fused into verification, gated by their
arithmetic/state tests. Measure draft/verify/commit cost and time per emitted token before claiming speedup.

## Recovery rule

On DEVICE_LOST or a runner STOP flag, stop GPU submissions and record the failing SHA, logs and minimal repro.
Independent host work can continue. The operator must restore device health before new GPU experiments;
never clear STOP automatically or continue GPU work on another kernel while the device is wedged.

## Documentation validation

This documentation revision changes no executable source or runner configuration. Local checks in an isolated
Python 3.11 environment: `pytest tests --host-only -q` passed (1074 passed, 349 skipped); the full `pytest tests -q`
could not collect two MLX-dependent modules because MLX is unavailable. Ruff 0.16.10 reported 1117 findings,
identical to the untouched source commit, with zero new findings. Documentation links, staged whitespace and
atlas Chromium interactions/desktop/mobile layout passed. These checks are not a new B70 qualification.
