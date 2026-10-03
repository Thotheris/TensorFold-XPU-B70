"""The gdn kernel suite: invariance checks and timings of trees, replays and prompt chains (kernels/gdn.md)."""

from __future__ import annotations

import json
from pathlib import Path

from .kbench import bench
from .kernel_benchmarks import _Capture

__all__ = ["run"]

HK, HV, DV, DK = 16, 48, 128, 128            # Qwen3.8-27B's linear-attention heads
STATE_BYTES = HV * DV * DK * 4               # one stream's fp32 state for one layer (3 MiB)
BATCH = 10
TREES = {
    "tree-w1": [-1],
    "tree-w4": [-1, 0, 1, 2],                                           # a chain: no slots
    "tree-w12": [-1, 0, 1, 2, 3, 1, 5, 0, 7, 8, 2, 10],                 # three branches off the spine
}


def _inputs(torch, rows: int, seed: int):
    gen = torch.Generator(device="xpu").manual_seed(seed)
    q = (torch.randn((rows, HK, DK), generator=gen, device="xpu") * 0.01).bfloat16()
    k = (torch.randn((rows, HK, DK), generator=gen, device="xpu") * 0.01).bfloat16()
    v = torch.randn((rows, HV, DV), generator=gen, device="xpu").bfloat16()
    g = torch.rand((rows, HV), generator=gen, device="xpu") * 0.4 + 0.5
    beta = torch.rand((rows, HV), generator=gen, device="xpu")
    state = torch.randn((HV, DV, DK), generator=gen, device="xpu") * 0.05
    return q, k, v, g, beta, state


def _window_bytes(rows: int) -> int:
    """q, k (bf16), v (bf16), g, beta (fp32) and y (bf16) of a window."""
    return rows * (2 * HK * DK * 2 + HV * DV * 2 + 2 * HV * 4 + HV * DV * 2)


def _bits(torch, t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _measure(torch, name: str, launch, check, nbytes: int, flops: float, kernel_module, kernel_name: str,
             out_dir: Path, extra: dict) -> dict:
    first = launch()
    torch.xpu.synchronize()
    first = first.clone()
    repeats = True
    for _ in range(20):
        again = launch()
        torch.xpu.synchronize()
        repeats = repeats and torch.equal(_bits(torch, again), _bits(torch, first))
    checked = check(first)
    equal = repeats and checked
    if not equal:
        raise RuntimeError(f"gdn {name} failed invariance (repeats={repeats}, serial={checked})")
    original = getattr(kernel_module, kernel_name)
    capture = _Capture(original)
    setattr(kernel_module, kernel_name, capture)
    try:
        launch()
        torch.xpu.synchronize()
        metrics = bench(fn=launch, nbytes=nbytes, flops=flops, name=f"gdn-{name}", out_dir=out_dir / "kernels",
                        triton_kernel=capture.compiled, bitwise_ok=equal, batch=BATCH)
    finally:
        setattr(kernel_module, kernel_name, original)
    metrics.update(repeats_checked=20, status="pass", **extra,
                   bytes_model="states read and written, slot writes and reads, window inputs and outputs",
                   timing=f"{BATCH} launches queued back to back per sample, per-launch mean")
    (out_dir / "kernels" / f"gdn-{name}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def run(out_dir: Path) -> dict:
    """Trees of 1, 4 and 12 rows, a 48-layer replay and a 512-row prompt chain; the headline is the 12-row tree."""
    import torch

    from tensorfold.cuda.kernels import gdn as host
    from tensorfold.xpu.kernels import gdn

    if not torch.xpu.is_available():
        raise RuntimeError("requested XPU is unavailable")
    results = {}
    for name, parents in TREES.items():
        args = _inputs(torch, len(parents), len(parents))
        plan = host.plan([parents], "xpu")
        entries = plan.entries.view(-1, 3).tolist()
        slot_moves = sum(source >= 0 for _, source, _ in entries) + sum(dest >= 0 for _, _, dest in entries)

        def launch(args=args, plan=plan):
            return gdn.tree(*args[:5], plan, state=args[5])

        def check(out, args=args, parents=parents):
            node = len(parents) - 1
            path = []
            while node >= 0:
                path.append(node)
                node = parents[node]
            state = args[5]
            for row in reversed(path):
                one = [x[row:row + 1].contiguous() for x in args[:5]]
                y = gdn.tree(*one, host.plan([[-1]], "xpu"), state=state)
                state = gdn.replay(*(x[None] for x in one[1:5]), state[None, None].contiguous(),
                                   torch.zeros((1, 1), dtype=torch.int32, device="xpu"),
                                   torch.ones(1, dtype=torch.int32, device="xpu"))[0, 0]
            return bool(torch.equal(_bits(torch, y[0]), _bits(torch, out[len(parents) - 1])))

        rows = len(parents)
        results[name] = _measure(torch, name, launch, check, STATE_BYTES * (1 + slot_moves) + _window_bytes(rows),
                                 rows * HV * DV * DK * 8.0, gdn, "_tree", out_dir,
                                 {"shape": {"rows": rows, "slots": plan.slots, "slot_moves": slot_moves,
                                            "heads": [HK, HV], "dv": DV, "dk": DK}})

    layers, accepted = 48, 4
    stack = [_inputs(torch, 12, 100 + i) for i in range(layers)]
    k, v, g, beta = (torch.stack([a[j] for a in stack]).contiguous() for j in (1, 2, 3, 4))
    states = torch.stack([a[5] for a in stack])[None].contiguous()
    path = torch.tensor([[0, 1, 3, 7]], dtype=torch.int32, device="xpu")
    counts = torch.tensor([accepted], dtype=torch.int32, device="xpu")

    def replay():
        return gdn.replay(k, v, g, beta, states, path, counts)

    def check_replay(out):
        one = gdn.replay(k[:1], v[:1], g[:1], beta[:1], states[:, :1].contiguous(), path, counts)
        return bool(torch.equal(one[0, 0], out[0, 0]))

    results["replay-48x4"] = _measure(torch, "replay-48x4", replay, check_replay,
                                      layers * 2 * STATE_BYTES, layers * accepted * HV * DV * DK * 6.0, gdn,
                                      "_replay", out_dir, {"shape": {"layers": layers, "rows": accepted, "streams": 1}})

    rows = 512
    args = _inputs(torch, rows, 512)
    final = torch.empty_like(args[5])

    def chain():
        return gdn.chain(*args[:5], args[5], final)

    def check_chain(out):
        mid = torch.empty_like(args[5])
        first = gdn.chain(*(x[:200].contiguous() for x in args[:5]), args[5], mid)
        rest = gdn.chain(*(x[200:].contiguous() for x in args[:5]), mid, torch.empty_like(mid))
        return bool(torch.equal(torch.cat([first, rest]), out))

    results["chain-512"] = _measure(torch, "chain-512", chain, check_chain, 2 * STATE_BYTES + _window_bytes(rows),
                                    rows * HV * DV * DK * 8.0, gdn, "_tree", out_dir,
                                    {"shape": {"rows": rows, "heads": [HK, HV], "dv": DV, "dk": DK}})
    headline = results["tree-w12"]
    summary = {**headline, "name": "gdn", "headline": "tree-w12",
               "cases": [{key: r.get(key) for key in ("name", "median_us", "gbps", "pct_peak_gbps", "n_spills",
                                                      "n_regs", "n_regs_source", "threads_per_warp", "bitwise_ok")}
                         for r in results.values()],
               "bitwise_ok": all(r["bitwise_ok"] for r in results.values())}
    (out_dir / "kernels" / "gdn.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
