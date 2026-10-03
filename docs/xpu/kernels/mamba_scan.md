# mamba_scan  (K2, kernel engineer, status: T0)

Op: Nemotron 3.5's Mamba-2 prompt scan: per head and value row, a 128-state recurrence over a prompt chunk.
CUDA: `families/nemotron_h/cuda/scan_rows.cu` (not on XPU). XPU: `src/tensorfold/xpu/kernels/mamba/__init__.py`
(Triton T0), reached from `nemotron_h/cuda/mamba.py::scan_rows` on XPU devices.

## Shapes

64 heads x 64 value rows (head_dim), 128 states, 8 groups (8 heads a group share B and C). proj `(W, xd | cd | heads)`
bf16 (z, then xBC, then dt), xc `(W, cd)` bf16 (x, then B for each group, then C), state `(heads, dh, 128)` fp32 in
place, y `(W, xd)` bf16. Chunks up to 4096 rows.

## Arithmetic contract (one step; CUDA's op order and FMA points, the XPU reduction pinned)

```
v    = fp32(dt_raw[t, h]) + dt_bias[h]
dt   = clamp(max(v, 0) + log(1 + exp(-|v|)), lo, hi)          # softplus, then the clamp
da   = exp(a[h] * dt)
xdt  = x * dt                                                    # x = fp32(xc[t, h*dh + r])
s    = fma(xdt, B, s * da)                                       # 128 states: multiply, then one FMA
out  = halving_sum(s * C)                                        # 128 values: +64, +32, ... +1, two-value adds
gz   = bf16(z / (1 + exp(-z)))                                   # silu of the gate z
y    = bf16(gz * bf16(fma(x, D[h], out)))
```

`halving_sum` is the GDN card's: its bits do not depend on tile, layout or lanes. CUDA sums 4 values a lane in FMA
chains, then a 4-lane butterfly; XPU bits need not equal CUDA's. Prompt bits need not equal the decode `_scan`'s.
`enable_fp_fusion=False`; exp/log are the toolchain's (pinned by the toolchain hash).

## Invariances required

chunked == one chunk (sizes 1, 7, 64, 300) in y and final state | row tile changes no bits | 20 repeats | tolerance
against an fp64 loop of the same recurrence.

## Tests

`tests/cuda/test_xpu_mamba_scan.py` (`xpu_kernel("mamba")`).

## Measurements

T0 qualified on `3a28fb4`, bundle `runs/xpu--main/3a28fb4-20261003T101551Z` (toolchain hash 858a0a59). Per-launch means
of 5 launches queued back to back; every case `bitwise_ok` (20 repeats plus the in-bench chunk / alone check); 0 spill
bytes. % of 608 GB/s or 183 TFLOPS on the stated bytes / flops models. `kernels:mamba` 6 tests passed. 32 lanes, 1 warp,
R = 8 rows a program.

| Case | us | per step |
|---|---|---|
| 1024-row chunk, 64 heads x 64 x 128 | 4127.5 | about 4.0 us |

Latency-bound (sequential steps, 512 programs). A chunked (SSD) form would change the arithmetic contract and needs
its own card; 23 Mamba layers x 4.1 ms is about 95 ms a 1024-token chunk.
