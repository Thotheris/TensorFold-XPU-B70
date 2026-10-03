"""kbench suites for the prompt GEMM (K5), grouped experts (K3) and the Mamba prompt scan (K2) at recipe shapes."""

from __future__ import annotations

import json
from pathlib import Path

from .kbench import bench
from .kernel_benchmarks import _Capture

__all__ = ["run_experts", "run_mamba", "run_prompt"]


def _bits(torch, t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _measure(torch, name, launch, check, nbytes, flops, module, kernel_name, out_dir: Path, extra: dict) -> dict:
    first = launch().clone()
    torch.xpu.synchronize()
    repeats = all(torch.equal(_bits(torch, launch()), _bits(torch, first)) for _ in range(20))
    checked = bool(check(first))
    if not (repeats and checked):
        raise RuntimeError(f"{name} failed invariance (repeats={repeats}, check={checked})")
    original = getattr(module, kernel_name)
    capture = _Capture(original)
    setattr(module, kernel_name, capture)
    try:
        launch()
        torch.xpu.synchronize()
        metrics = bench(fn=launch, nbytes=nbytes, flops=flops, name=name, out_dir=out_dir / "kernels",
                        triton_kernel=capture.compiled, bitwise_ok=True, batch=5)
    finally:
        setattr(module, kernel_name, original)
    metrics.update(repeats_checked=20, status="pass", timing="5 launches queued back to back per sample", **extra)
    (out_dir / "kernels" / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _sym(torch, e, n, k, gs, seed):
    g = torch.Generator(device="xpu").manual_seed(seed)
    shape = (e, n, k // 8) if e else (n, k // 8)
    words = torch.randint(-(2**31), 2**31 - 1, shape, generator=g, device="xpu", dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(shape[:-1] + (k // gs,), generator=g, device="xpu") * 0.01 + 0.0005).half()
    return words, scales


def _summary(name: str, results: list[dict], out_dir: Path) -> dict:
    head = results[0]
    summary = {**head, "name": name, "headline": head["name"],
               "cases": [{key: r.get(key) for key in ("name", "median_us", "gbps", "pct_peak_gbps", "tflops",
                                                      "pct_peak_tflops", "n_spills", "n_regs", "n_regs_source",
                                                      "threads_per_warp", "dpas", "bitwise_ok")} for r in results],
               "bitwise_ok": all(r["bitwise_ok"] for r in results)}
    (out_dir / "kernels" / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_prompt(out_dir: Path) -> dict:
    """K5 at a 1024-row prompt chunk: Qwen gate/up (g128) and Nemotron in_proj (g64); rows checked against halves."""
    import torch

    from tensorfold.xpu.kernels.qmm import prompt as P

    results = []
    for case, n, k, gs in (("prompt-qwen-gate-up", 17408, 5120, 128), ("prompt-nemotron-in-proj", 10304, 2688, 64)):
        words, scales = _sym(torch, 0, n, k, gs, n)
        x = torch.randn((1024, k), device="xpu").bfloat16()

        def launch(words=words, scales=scales, x=x, gs=gs):
            return P.prompt_matmul(x, words, scales, gs=gs)

        def check(out, words=words, scales=scales, x=x, gs=gs):
            halves = [P.prompt_matmul(x[a:b], words, scales, gs=gs) for a, b in ((0, 300), (300, 1024))]
            return torch.equal(_bits(torch, torch.cat(halves)), _bits(torch, out))

        nbytes = words.numel() * 4 + scales.numel() * 2 + x.numel() * 2 + 1024 * n * 2
        results.append(_measure(torch, case, launch, check, nbytes, 2.0 * 1024 * n * k, P, "_prompt", out_dir,
                                {"shape": {"m": 1024, "n": n, "k": k, "gs": gs}}))
    return _summary("prompt", results, out_dir)


def run_experts(out_dir: Path) -> dict:
    """K3 at Nemotron's shapes (130 experts, 8 slots): decode for 16 tokens, prompt for 1024; up relu^2 and down."""
    import torch

    from tensorfold.xpu.kernels import experts as X

    experts, slots, d, ni, gs = 130, 8, 2688, 1856, 64
    up = _sym(torch, experts, ni, d, gs, 1)
    down = _sym(torch, experts, d, ni, gs, 2)
    results = []
    for mode, rows, tile, run in (("decode", 16, X.TILE, X.decode), ("prompt", 1024, X.PREFILL_TILE, X.prompt)):
        g = torch.Generator(device="xpu").manual_seed(rows)
        picks = torch.cat([torch.randint(0, experts - 2, (rows, slots - 2), generator=g, device="xpu"),
                           torch.tensor([[experts - 2, experts - 1]], device="xpu").expand(rows, 2)], 1).int()
        p = X.plan(picks.contiguous(), experts, tile)
        x = torch.randn((rows, d), generator=g, device="xpu").bfloat16()
        act = torch.randn((rows * slots, ni), generator=g, device="xpu").bfloat16()
        used = int((torch.bincount(picks.flatten().long(), minlength=experts) > 0).sum())
        for proj, (words, scales), inp, tokens, epi, n, k in (
                ("up", up, x, True, X.EPI_RELU2, ni, d),
                ("down", down, act, False, X.EPI_FP32 if mode == "decode" else X.EPI_BF16, d, ni)):
            def launch(words=words, scales=scales, inp=inp, tokens=tokens, epi=epi, run=run, p=p):
                return run(inp, words, scales, p, from_tokens=tokens, gs=gs, epi=epi)

            def check(out, words=words, scales=scales, inp=inp, tokens=tokens, epi=epi, run=run, picks=picks,
                      tile=tile):
                pair = 13
                row = pair // slots if tokens else pair
                one = X.plan(picks.flatten()[pair:pair + 1].view(1, 1).contiguous(), experts, tile)
                alone = run(inp[row:row + 1], words, scales, one, from_tokens=tokens, gs=gs, epi=epi)
                return torch.equal(_bits(torch, alone[0]), _bits(torch, out[pair]))

            weight_bytes = used * (words[0].numel() * 4 + scales[0].numel() * 2)
            kernel = "_decode" if mode == "decode" else "_prompt"
            results.append(_measure(torch, f"experts-{mode}-{proj}", launch, check, weight_bytes + inp.numel() * 2,
                                    2.0 * rows * slots * n * k, X, kernel, out_dir,
                                    {"shape": {"rows": rows, "slots": slots, "experts_used": used, "n": n, "k": k},
                                     "bytes_model": "weights of the experts used, once, plus inputs"}))
    return _summary("experts", results, out_dir)


def run_mamba(out_dir: Path) -> dict:
    """K2 at Nemotron's Mamba-2 layer over a 1024-row chunk; checked against two chunks."""
    import torch

    from tensorfold.xpu.kernels import mamba as M

    heads, dh, groups, ds, rows = 64, 64, 8, 128, 1024
    xd, cd = heads * dh, heads * dh + 2 * groups * ds
    g = torch.Generator(device="xpu").manual_seed(4)
    proj = (torch.randn((rows, xd + cd + heads), generator=g, device="xpu") * 0.5).bfloat16()
    xc = (torch.randn((rows, cd), generator=g, device="xpu") * 0.5).bfloat16()
    state = torch.randn((heads, dh, ds), generator=g, device="xpu") * 0.1
    a = -torch.rand(heads, generator=g, device="xpu") - 0.1
    d = torch.randn(heads, generator=g, device="xpu")
    dtb = torch.randn(heads, generator=g, device="xpu") * 0.5

    def scan(p, x, s):
        s = s.clone()
        return M.scan_rows(p, x, s, a, d, dtb, p.shape[0], heads=heads, head_dim=dh, groups=groups, state_dim=ds,
                           lo=0.001, hi=100.0), s

    def launch():
        return scan(proj, xc, state)[0]

    def check(out):
        y1, s1 = scan(proj[:300], xc[:300], state)
        y2, _ = scan(proj[300:], xc[300:], s1)
        return torch.equal(_bits(torch, torch.cat([y1, y2])), _bits(torch, out))

    nbytes = proj.numel() * 2 + xc.numel() * 2 + 2 * state.numel() * 4 + rows * xd * 2
    result = _measure(torch, "mamba-scan-1024", launch, check, nbytes, rows * heads * dh * ds * 6.0, M, "_scan_rows",
                      out_dir, {"shape": {"rows": rows, "heads": heads, "head_dim": dh, "states": ds}})
    return _summary("mamba", [result], out_dir)
