"""The first migrated Triton kernels have measured, repeat-checked microbenchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

from .kbench import bench

__all__ = ["run_benchmark"]


class _Capture:
    """A wrapper records the compiled handle while preserving the production launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.compiled = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.compiled = self.kernel[grid](*args, **kwargs)
            return self.compiled
        return launch


def run_benchmark(name: str, out_dir: Path) -> dict:
    """A named benchmark checks 20 repeats before measuring the same production launch."""
    import torch

    if not torch.xpu.is_available():
        raise RuntimeError("requested XPU is unavailable")
    if name == "qmm":
        from .bench_qmm import run

        return run(out_dir)
    if name == "glue":
        from tensorfold.families.qwen3_5.cuda import glue as module

        rows, width = 17, 17408
        gen = torch.Generator(device="xpu").manual_seed(27)
        gate = torch.randn(rows, width, generator=gen, device="xpu").bfloat16()
        up = torch.randn(rows, width, generator=gen, device="xpu").bfloat16()
        kernel_name = "_swiglu"
        # Preallocate outputs and launch exactly as swiglu() does, so allocation is not timed.
        output = torch.empty_like(gate)
        sums = torch.empty((rows, width // 64), device="xpu", dtype=torch.float32)

        def launch():
            module._swiglu[(rows, (width + 1023) // 1024)](gate, up, output, sums, N=width, BLOCK=1024, num_warps=4)
            return output, sums

        nbytes = rows * width * 6 + rows * (width // 64) * 4
        flops = rows * width * 4  # exp/sigmoid transcendental cost excluded
        shape = {"rows": rows, "width": width}
    elif name == "prefill-attention":
        from tensorfold.cuda.kernels import prefill_attention as module

        rows, heads, kv_heads, dim, prefix = 129, 24, 4, 256, 17
        gen = torch.Generator(device="xpu").manual_seed(27)
        q = torch.randn(rows, heads, dim, generator=gen, device="xpu").bfloat16()
        k = torch.randn(prefix + rows, kv_heads, dim, generator=gen, device="xpu").bfloat16()
        v = torch.randn(prefix + rows, kv_heads, dim, generator=gen, device="xpu").bfloat16()
        output = torch.empty_like(q)
        kernel_name = "_attend"

        def launch():
            return (module.triton_attention(q, k, v, prefix, scale=dim ** -0.5, out=output),)

        nbytes = (q.numel() + k.numel() + v.numel() + output.numel()) * 2
        flops = 4 * heads * dim * sum(prefix + row + 1 for row in range(rows))
        shape = {"rows": rows, "heads": heads, "kv_heads": kv_heads, "dim": dim, "prefix": prefix}
    else:
        raise ValueError(f"no microbenchmark registered for {name}; add its tests and benchmark together")

    original = getattr(module, kernel_name)
    capture = _Capture(original)
    setattr(module, kernel_name, capture)
    try:
        baseline = tuple(tensor.clone() for tensor in launch())
        torch.xpu.synchronize()
        equal = True
        for _ in range(20):
            current = launch()
            torch.xpu.synchronize()
            equal = equal and all(torch.equal(a.view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32),
                                             b.view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32))
                                  for a, b in zip(baseline, current))
        if not equal:
            raise RuntimeError(f"{name} failed repeat bitwise equality")
        metrics = bench(fn=launch, nbytes=nbytes, flops=flops, name=name, out_dir=out_dir / "kernels",
                        triton_kernel=capture.compiled, bitwise_ok=equal)
        metrics.update(shape=shape, repeats_checked=20, status="pass", bytes_model="unique tensor bytes",
                       flops_model="dense arithmetic estimate; transcendental operations excluded")
        import json

        (out_dir / "kernels" / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics
    finally:
        setattr(module, kernel_name, original)


if __name__ == "__main__":
    run_benchmark(sys.argv[1], Path(sys.argv[2]))
