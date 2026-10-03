# prompt_gemm  (K5, kernel engineer, status: T0)

Op: prompt (prefill) projections `y = x @ W.T` for 4-bit symmetric weights, M up to a prompt chunk (4096 rows).
CUDA source: `cuda/kernels/qmm_prefill.cu` (not on XPU; `qmm_prefill8.cu`, FP8, is refused on XPU).
XPU: `src/tensorfold/xpu/kernels/qmm/prompt.py::prompt_matmul` (Triton T0).

## Shapes and dtypes

The K4 weight shapes (`docs/xpu/kernels/qmm.md`): Qwen 5120 x 17408, 17408 x 5120, 10240 x 5120 (g128); Nemotron
10304 x 2688, 2688 x 4096 (g64). Words `(N, K/8)` int32 low nibble first, scales `(N, K/gs)` fp16 or bf16 as stored.
x bf16 `(M, K)`, output bf16 (or fp32 with `f32=True`).

## Arithmetic contract

```
w[n, k] = bf16_rne( fma(fp32(q[n, k]), fp32(s[n, k // gs]), -8 * fp32(s[n, k // gs])) )   # rounded once
acc     = 0
for t in 0 .. K/64 - 1:                       # one fp32 chain over K, 64 columns a step, ascending
    acc = dot(x[:, 64t : 64t+64], w[:, 64t : 64t+64]^T, acc)        # tl.dot (DPAS), bf16 operands, fp32 accumulate
y = bf16_rne(acc)
```

- `fma(q, s, -8s)` equals `s * (q - 8)` exactly (the product of a 4-bit integer and an fp16/bf16 scale is exact in
  fp32); the only roundings are the bf16 weight and the fp32 dot chain.
- No split-K and no M-dependent choice: BM, BN, warps, stages and GRF mode are functions of the weight shape. A row's
  bits therefore do not depend on the chunk it is in, its place in the chunk, or a resumed prompt.
- Prompt bits are not decode bits (decode folds `xs * b` per group; prompt rounds the weight to bf16). The engine must
  never switch arithmetic by chunk size: a prompt row always takes this kernel.
- `enable_fp_fusion=False`; the one FMA is the explicit `tl.fma` in the dequant. Integer to bf16 goes through fp32.

## Invariances required

row alone == row in any chunk (M = 1, 17, 300, 1024 and resumed halves) | 20 repeats | one-hot rows read back
`bf16(s * (q - 8))` exactly | fp64 tolerance | DPAS in the TTGIR.

## References and tests

`tests/cuda/test_xpu_prompt_gemm.py` (`xpu_kernel("prompt")`); fp64 reference `x.double() @ w.double().T` with `w` the
rounded weight.

## Roofline target

Compute-bound for large M: 2 M N K flops against ~183 TFLOPS bf16 DPAS. T0 aims at correctness; tuning per shape is T1.

## Measurements

T0 qualified on `3a28fb4`, bundle `runs/xpu--main/3a28fb4-20261003T101551Z` (toolchain hash 858a0a59). Per-launch means
of 5 launches queued back to back; every case `bitwise_ok` (20 repeats plus the in-bench chunk / alone check); 0 spill
bytes. % of 608 GB/s or 183 TFLOPS on the stated bytes / flops models. `kernels:prompt` 14 tests passed. 16 lanes, DPAS
in the TTGIR, `n_regs` 256 (driver: automatic large GRF).

| Case (M = 1024) | us | TFLOPS | % of 183 |
|---|---|---|---|
| Qwen gate/up 17408 x 5120 g128 | 8962.9 | 20.37 | 11.1 |
| Nemotron in_proj 10304 x 2688 g64 | 2868.7 | 19.77 | 10.8 |

T1 candidates: per-shape BM/BN/warps (the default 64 x 64, 8 warps is untuned), tensor descriptors for the x loads,
dequantising the weight tile once per K step for several row tiles.
