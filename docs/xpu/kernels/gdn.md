# gdn  (K1, kernel engineer, status: T0)

Op: the gated delta rule of Qwen3.8-27B's linear-attention layers: verify-window trees, commit replays and prompt
chains. CUDA sources: `cuda/kernels/gdn.cu` (`tree_kernel`, `replay_kernel`), `gdn_prefill.cu` (`chain_kernel`),
wrapper `cuda/kernels/gdn.py`. fp32 reference: `families/qwen3_5/cuda/reference.py::_gdn`.
XPU: `src/tensorfold/xpu/kernels/gdn/` (Triton T0), selected by `TF_XPU_KERNEL_GDN=triton|native` (`xpu/select.py`).

## Shapes and dtypes

- 48 value heads, 16 key heads (`hv / hk = 3` value heads share a key head), dk = dv = 128, 48 GDN layers.
- State fp32 `(Hv, Dv, Dk) = (48, 128, 128)` per stream and layer (3 MiB), row-major in dk.
- q, k: `(W, Hk, 128)` bf16 (fp32 accepted); v: `(W, Hv, Dv)` bf16; g, beta: `(W, Hv)` fp32. Output y `(W, Hv, Dv)` bf16.
- Verify windows: up to 16 rows a stream, trees from `gdn.schedule` (CUDA's host plan, reused unchanged). Prompt
  chains: up to 4096 rows a chunk.

## Arithmetic contract (one step; shared by tree, chain and replay)

For each value row `r` of head `h` (state row `s = S[h, r, :]`, 128 fp32), at node `t` with `key = k[t, h // 3]`:

```
s     = s * g[t, h]                                  # 128 fp32 multiplies
mem   = halving_sum(s * key)                         # 128 multiplies, then the tree below
delta = (fp32(v[t, h, r]) - mem) * beta[t, h]
s     = s + key * delta                              # multiply, then add (no FMA)
y     = bf16_rne(halving_sum(s * q[t, h // 3]))      # tree, chain only; replay keeps s
```

`halving_sum(x)` over 128 values: `x[i] += x[i + 64]` for i < 64, then `+32`, `+16`, `+8`, `+4`, `+2`, `+1`; the
result is `x[0]`. Every step adds exactly two values, so the sum does not depend on tile shape, layout, lane count,
`num_warps` or which kernel runs it. All ops are separate IEEE fp32 operations (`enable_fp_fusion=False`); q and k are
read as fp32 (bf16 widening is exact).

- Serial == tree == replay == chain by construction: every node runs this step on its parent's state, whichever
  kernel or slot it came from.
- XPU bits need not equal CUDA's (CUDA sums 4 values a lane, then a 32-lane xor butterfly; its prompt chain uses FMA).
  On XPU the prompt chain runs the same step as verify, so prompt and verify bits are equal too.
- Native N0 mapping (SYCL, SG16): lane `l` holds `s[l + 16 j]`, j = 0..7. Levels +64, +32, +16 are in-lane adds of
  `(j, j+4)`, `(j, j+2)`, `(j, j+1)`; +8, +4, +2, +1 are `permute_group_by_xor` 8, 4, 2, 1. That is the same tree.

## Layouts and scheduling

- Tree plan: `(node, source, dest)` int32 triples per row from `gdn.plan` (source -1 committed state, -2 the node just
  run, else a slot; dest a slot or -1), and stream row ranges `starts`.
- Multi-stream states are one stacked fp32 tensor `(S, Hv, Dv, 128)` plus an int32 state index per stream (no pointer
  tables). Slots are global scratch `(streams, slots, Hv, Dv, 128)` fp32; a program writes and reads only its own rows.
- Replay: stacked per-layer inputs `k (L, W, Hk, 128)`, `v (L, W, Hv, Dv)`, `g, beta (L, W, Hv)` and stacked states
  `(S, L, Hv, Dv, 128)`, with per-stream accepted rows and counts. In place or into a new stack.

## Invariances required

tree node == its serial path of one-row steps | replay of a path == the path's serial steps | several streams in one
launch == each alone | pending rows folded into the next tree == replay then tree | chain chunked anywhere == one chain
| chain == tree over the same path | 20 repeats | R and num_warps change no bits.

## References and tests that pin bits

`tests/cuda/test_gdn.py` (migrated to DEV; CUDA-only cases stay `cuda_only`), new XPU cases in the same file
(`xpu_kernel("gdn")`); tolerance against an fp64 torch loop of the step. Suite `kernels:gdn`.

## Roofline target on B70

Bandwidth-bound per verify window: each program reads its state rows once (3 MiB a layer and stream), the window's
q/k/v/g/beta, and writes y; slot traffic only at branching nodes. 48 layers x 3 MiB = 151 MiB a stream per window; at
608 GB/s that is about 0.25 ms. The T0 aim is correctness without DEVICE_LOST; the N0 SYCL kernel is the speed path.

## Risks

- Persistent-state recurrence loops have hit DEVICE_LOST on B70 (intel-xpu-backend-for-triton #6658). Mitigations:
  small R tiles, synchronise after every launch in tests, short bounded tests first.

## Measurements

(none yet)

## Open issues

- The explicit halving tree costs reshapes per step; measure against `tl.sum` (which is not pinned) at T1.
