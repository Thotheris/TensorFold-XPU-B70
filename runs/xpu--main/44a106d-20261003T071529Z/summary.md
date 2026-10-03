# xpu/main 44a106d

Status: **pass**

## Suites
- `env`: pass
- `triton-smoke`: pass
- `unit-host`: pass
- `unit-xpu`: pass
- `kernels:glue`: pass
- `kernels:prefill-attention`: pass
- `kernels:qmm`: pass

## Baseline regressions
- Performance: `kernels/qmm-nemotron-in-proj-m1.json` gbps 118.243 -> 112.609
- Performance: `kernels/qmm-nemotron-in-proj-m1.json` median_us 124.659 -> 130.896
- Performance: `kernels/qmm-nemotron-in-proj-m16.json` gbps 121.53 -> 115.451
- Performance: `kernels/qmm-nemotron-in-proj-m16.json` median_us 124.495 -> 131.049
- Performance: `kernels/qmm-qwen-in-proj-ab-m1.json` gbps 4.27519 -> 4.1063
- Performance: `kernels/qmm-qwen-in-proj-ab-m1.json` median_us 117.388 -> 122.216
- Performance: `kernels/qmm-qwen-in-proj-ab-m16.json` gbps 5.5337 -> 5.07215
- Performance: `kernels/qmm-qwen-in-proj-ab-m16.json` median_us 118.708 -> 129.51
