# XPU port status

This snapshot is based on inspected result bundles through `44a106d`, not on an end-to-end recipe run.
Refresh it from `origin/results:index.jsonl` before starting kernel work. The
[execution prompt](prompts/kernel-engineer.txt), [plan](PORT_PLAN.md) and
[upstream atlas](reports/tensorfold-kernel-atlas-2026-10-02.html) describe the next work.

## Verified evidence

| Commit | Bundle | Executed suites | Qualification |
|---|---|---|---|
| `c32b417` | `runs/xpu--main/c32b417-20261003T050422Z` | env, triton-smoke, unit-host, unit-xpu, kernels:glue, kernels:prefill-attention | pass; register reporting fix and selected migrated checks |
| `da2a34f` | `runs/xpu--main/da2a34f-20261003T054251Z` | env, unit-host, unit-xpu, kernels:qmm | pass on executed suites; unit-xpu 37 passed, QMM 31 passed |
| `44a106d` | `runs/xpu--main/44a106d-20261003T071529Z` | env, triton-smoke, unit-host, unit-xpu, kernels:glue, kernels:prefill-attention, kernels:qmm | pass; unit-host 1479 passed / 30 skipped, unit-xpu 51, glue 3, prefill-attention 3, QMM 45; qualifies `55d2095`, `f569957`, `d3008cb`, K4.T1 `f54f223`, compare fix `f1aefb4` |

K4.T0 supports symmetric INT4 g64/g128, fp16/bf16 scales and row-invariant BF16 GEMV. The inspected QMM
results report bitwise checks passing, DPAS and 16 compiled lanes. Cross-producer canonical 64-wide sums
passed. The [QMM card](kernels/qmm.md) records arithmetic and exact coverage.

K4.T1 (`44a106d`): per-shape single-warp tiles and chained sub-dots remove every spill. Device-only bandwidth at
M=1 / M=16 (calls queued behind a long matmul, so no host gaps); details and all windows in the [QMM card](kernels/qmm.md).

| K4 case | Call us M=1 / 16 | Device us M=1 / 16 | Device % of 608 GB/s | T0 call % (da2a34f) | Spill bytes |
|---|---:|---:|---:|---:|---:|
| Qwen down, 5120 x 17408 SYM | 120 / 140 | 115 / 137 | 66 / 56 | 5.8 | 0 |
| Qwen gate/up, 17408 x 5120 SYM | 138 / 212 | 136 / 205 | 56 / 38 | 9.8 | 0 |
| Qwen qkv, 10240 x 5120 SYM | 121 / 120 | 83 / 115 | 54 / 39 | 5.8 | 0 |
| Qwen head, 248320 x 5120 BF16 | 4959 / 5115 | 4977 / 5112 | 84 / 82 | 58.6 | 0 |
| Nemotron head, 131072 x 2688 BF16 | 1351 / 1386 | 1364 / 1391 | 85 / 84 | 57.3 | 0 |

The 117-125 us floor is host submission, measured on `44a106d` (`kernels/qmm.json` `host_breakdown`, M=1 B out_proj):
the wrapper takes 98.5 us of host time; one Triton JIT launch is 32.3 us (11.4 us of it the driver launch), the
split-K reduce is a second launch, `group_sums` alone is 41.9 us, an allocation 2.1 us. Calls under ~115 us of device
time therefore run at the host's pace. `triton-smoke` `launch_latency`: Triton add 38.5 us host vs 18.1 us `torch.add`.
Launch reduction is WS6 work. Effective GB/s uses the bytes model, not measured DRAM traffic.

## Coverage limits and pending changes

- `compare.py` (`f1aefb4`) now lists a baseline artifact whose suite the run did not request as "not covered" and one
  missing from a suite that ran as a blocking "missing artifact"; only `bitwise_ok: false` is a bitwise break.
- `44a106d`'s compare flags 3-9% slower host-bound cases (Nemotron in_proj, in_proj_a/b): their device time is
  45-57 us and host submission moved by a few us; they are host timing noise, not kernel changes.
- The migrated host/device suites do not cover every recipe path. Neither recipe has an inspected e2e bundle;
  serial/drafted token equality, checkpoint quality and model throughput remain unqualified.
- The atlas is upstream source research: v0.6.3 `9356df5` compared with v0.6.0 `c464617`. It does not authorize
  an upstream merge, new families or new checkpoint formats.

## Next work, in order

1. Done on `44a106d`: current-head qualification, missing-versus-failed coverage, the launch-floor breakdown.
2. Done on `44a106d`: the bounded K4.T1 pass (zero spills, windows 1-16, host/device split).
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
