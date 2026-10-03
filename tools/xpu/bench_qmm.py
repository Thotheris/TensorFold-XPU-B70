"""The qmm kernel suite: invariance checks and kbench timings at the recipe shapes (docs/xpu/kernels/qmm.md)."""

from __future__ import annotations

import json
from pathlib import Path

from .kbench import bench
from .kernel_benchmarks import _Capture

__all__ = ["CASES", "run"]

# (id, N, K, group size; 0 = bf16 weights)
CASES = (
    ("qwen-down", 5120, 17408, 128),
    ("qwen-gate-up", 17408, 5120, 128),
    ("qwen-qkv", 10240, 5120, 128),
    ("qwen-head", 248320, 5120, 0),
    ("qwen-in-proj-ab", 48, 5120, 0),
    ("nemotron-in-proj", 10304, 2688, 64),
    ("nemotron-out-proj", 2688, 4096, 64),
    ("nemotron-head", 131072, 2688, 0),
)
ROWS = (1, 16)
HEADLINE = ("qwen-down", 1)
CACHE_BYTES = 128 << 20      # weights are cycled over copies of at least this size: the last-level cache is 24 MB
BATCH = 20                   # launches queued back to back per timing sample, as a decode step queues them


def _bits(tensor):
    import torch

    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def _make_weights(torch, n: int, k: int, gs: int):
    gen = torch.Generator(device="xpu").manual_seed(n + k)
    if gs == 0:
        return (torch.randn((n, k), generator=gen, device="xpu").bfloat16(),)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="xpu", dtype=torch.int64)
    scales = (torch.rand((n, k // gs), generator=gen, device="xpu") * 0.02 + 0.001).half()
    return words.to(torch.int32), scales


def _one(torch, case: str, n: int, k: int, gs: int, m: int, out_dir: Path) -> dict:
    from tensorfold.families.qwen3_5.cuda import qmm
    from tensorfold.xpu.kernels.qmm import bf16 as bf16_module
    from tensorfold.xpu.kernels.qmm import bf16_matmul, lane_config, slices, sym_matmul
    from tensorfold.xpu.kernels.qmm import lane as lane_module

    weights = _make_weights(torch, n, k, gs)
    nbytes_weights = sum(t.numel() * t.element_size() for t in weights)
    copies = [weights] + [tuple(t.clone() for t in weights)
                          for _ in range(min(31, max(1, -(-CACHE_BYTES // nbytes_weights)) - 1))]
    gen = torch.Generator(device="xpu").manual_seed(m)
    x = torch.randn((m, k), generator=gen, device="xpu").bfloat16()
    xs = qmm.group_sums(x) if gs else None

    def launch(copy, rows=slice(None), f32=False):
        if gs == 0:
            return bf16_matmul(x[rows], copy[0], f32=f32)
        return sym_matmul(x[rows], copy[0], copy[1], xs[rows], gs=gs, f32=f32)

    # invariance: 20 repeats of the same launch, and every row alone against the window, on unrounded fp32 sums
    first = launch(weights, f32=True).clone()
    torch.xpu.synchronize()
    repeats_equal = True
    for _ in range(20):
        again = launch(weights, f32=True)
        torch.xpu.synchronize()
        repeats_equal = repeats_equal and torch.equal(_bits(first), _bits(again))
    rows_equal = all(torch.equal(_bits(first[r:r + 1]), _bits(launch(weights, slice(r, r + 1), f32=True)))
                     for r in range(m))
    copies_equal = all(torch.equal(_bits(first), _bits(launch(copy, f32=True))) for copy in copies[1:])
    torch.xpu.synchronize()
    equal = repeats_equal and rows_equal and copies_equal
    if not equal:
        raise RuntimeError(f"qmm {case} M={m} failed invariance (repeats={repeats_equal}, rows={rows_equal}, "
                           f"copies={copies_equal})")

    module, name = (bf16_module, "_gemv") if gs == 0 else (lane_module, "_qmm_sym")
    original = getattr(module, name)
    capture = _Capture(original)
    setattr(module, name, capture)
    counter = [0]

    def timed():
        counter[0] += 1
        return launch(copies[counter[0] % len(copies)])

    try:
        timed()
        torch.xpu.synchronize()
        nbytes = nbytes_weights + x.numel() * 2 + m * n * 2
        metrics = bench(fn=timed, nbytes=nbytes, flops=2.0 * m * n * k, name=f"qmm-{case}-m{m}",
                        out_dir=out_dir / "kernels", triton_kernel=capture.compiled, bitwise_ok=equal, batch=BATCH)
    finally:
        setattr(module, name, original)
    cfg = lane_config(n, k, gs)
    metrics.update(shape={"n": n, "k": k, "gs": gs, "m": m, "dtype": "bf16" if gs == 0 else "sym-int4"},
                   config={**vars(cfg), "sk": slices(n, k, gs)},
                   repeats_checked=20, rows_checked=m, weight_copies=len(copies), status="pass",
                   bytes_model="weights + scales + x + out, one pass; weights cycled over copies larger than the LLC",
                   timing=f"{BATCH} launches queued back to back per sample, per-launch mean")
    (out_dir / "kernels" / f"qmm-{case}-m{m}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def run(out_dir: Path) -> dict:
    """Every recipe shape is checked for invariance and timed; the headline file is the Qwen down projection at M=1."""
    import torch

    if not torch.xpu.is_available():
        raise RuntimeError("requested XPU is unavailable")
    results = []
    for case, n, k, gs in CASES:
        for m in ROWS:
            results.append((case, m, _one(torch, case, n, k, gs, m, out_dir)))
            torch.xpu.empty_cache()
    headline = next(r for case, m, r in results if (case, m) == HEADLINE)
    summary = {
        **headline, "name": "qmm", "headline": f"{HEADLINE[0]} M={HEADLINE[1]}",
        "cases": [{"name": r["name"], "median_us": r["median_us"], "gbps": r["gbps"],
                   "pct_peak_gbps": r["pct_peak_gbps"], "n_spills": r["n_spills"], "n_regs": r["n_regs"],
                   "threads_per_warp": r["threads_per_warp"], "dpas": r["dpas"], "bitwise_ok": r["bitwise_ok"]}
                  for _, _, r in results],
        "bitwise_ok": all(r["bitwise_ok"] for _, _, r in results),
    }
    (out_dir / "kernels" / "qmm.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
