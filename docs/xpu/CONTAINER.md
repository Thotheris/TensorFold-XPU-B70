# The B70 container

Status: **plan**. Owner: Harness (WS2), on one branch `xpu/harness/container` cut from `xpu/main`. Paths below
(`tools/xpu/**`, `docs/xpu/HARNESS.md`) are as they stand on `xpu/main`.

The B70's userspace toolchain moves from the host (`tools/xpu/bootstrap.sh --apply --system` plus a host venv) into an
Ubuntu 26.04 image. The image runs tests and serves. It never contains a SYCL compiler. Native kernels are compiled on
the host and mounted read-only. The bundle that gates a merge comes from the container.

## 1. The split

| Host (cannot live in a container) | Image (runtime only) |
|---|---|
| kernel ≥ 6.17 with `xe`, GuC/HuC firmware, ReBAR | pinned compute-runtime, IGC, Level Zero, `ocloc` |
| Docker Engine | venv with torch 2.14.1+xpu and triton-xpu 3.8.0 |
| `xpu-smi` (used by `tools/xpu/health.py`) | TensorFold's runtime Python deps |
| DLE 2026.1 at `/opt/intel/dle-2026.1`, for native builds only | the TensorFold source (`serve` target) or a mounted worktree |
| model snapshots (`HF_HOME`) | |

Workflow:
- **Dev loop on the host:** edit, build native kernels with `build_ext.sh` (DLE sourced only in its subshell), quick
  tests in the host venv.
- **Verification in the container:** the runner runs suites in the image; that bundle is the one that counts.

Keeping DLE out of the image makes the setvars rule (AGENTS.md §4) impossible to break: Triton XPU segfaults when a
second SYCL runtime from oneAPI is on the library path, and the image has none. The torch wheel already brings Intel's
SYCL runtime and oneMKL as pip packages, and Triton compiles through the driver's IGC, so nothing at runtime needs
`icpx`.

exl3xpu ships `icpx` in its image (oneAPI 2025.3 inside `intel/llm-scaler-vllm`) and builds there. That works for it
because Intel pairs the compiler and torch in one image and its hot path is ESIMD plus oneDNN, not Triton XPU. We run
nearly everything through Triton and pin torch ourselves, so we pin the compiler on the host instead.

## 2. Dependencies

| Item | Image | Host | Why |
|---|---|---|---|
| torch `2.14.1+xpu` (XPU index) | ✅ | dev venv | brings `intel-sycl-rt`, oneMKL and the other runtime wheels |
| triton-xpu `3.8.0` | ✅ | dev venv | pinned by torch; `tools/xpu/constraints.txt` |
| `libze-intel-gpu1` 26.31.39395.13 | ✅ | ✅ | the Level Zero GPU driver |
| `libigdgmm12` 22.10.0 | ✅ | ✅ | compute-runtime dependency |
| `intel-igc-core-2`, `intel-igc-opencl-2` 2.40.13 | ✅ | ✅ | the driver finalises Triton's SPIR-V at runtime |
| `intel-ocloc` 26.31.39395.13 | ✅ | ✅ | without it torch silently reports no DPAS / 2D block I/O (B70 guide §1.6) |
| `libze1`, `libze-dev` 1.32.0 | ✅ | ✅ | loader ≥ 1.32; Triton's launcher needs `level_zero/ze_api.h` |
| gcc, g++, python3-dev | ✅ | ✅ | Triton compiles its launcher with the host compiler; which of these it needs is `[UNVERIFIED]` |
| numpy, huggingface-hub, tokenizers, safetensors, jinja2, pytest | ✅ | dev venv | TensorFold installs `--no-deps`; these go in under the constraint file |
| `intel-opencl-icd`, `ocl-icd-libopencl1` | probe | ✅ | torch and Triton use Level Zero; drop if the §6 probe passes |
| DLE 2026.1 / `icpx` | **never** | ✅ | native builds only; must match torch's SYCL runtime (`torch.version.xpu == 20260100`) |
| `xpu-smi`, `libmetee` 6.2.5 | never | ✅ | health checks run on the host |
| model weights | never (mount) | ✅ | `HF_HOME` mounted read-only |

Ubuntu 26.04's own archive versions (compute-runtime 26.05, IGC 1.0.17791, `libze1` 1.28.2) are too old, so the image
installs the pinned debs. The `+u24.04` IGC and loader debs already install on the 26.04 box.

## 3. The image (`tools/xpu/docker/`)

- **`Dockerfile`**, `FROM ubuntu:26.04`, two targets:
  - `toolchain`:
    1. apt: `ca-certificates curl python3 python3-venv python3-dev gcc g++ ocl-icd-libopencl1`;
    2. COPY `tools/xpu/debs.lock`, download each deb, `sha256sum --check`, then one `dpkg -i`; no Intel apt repo;
    3. venv at `/opt/tf-venv`: `pip install torch==2.14.1+xpu --index-url https://download.pytorch.org/whl/xpu`;
    4. `pip install --constraint constraints.txt numpy huggingface-hub tokenizers safetensors jinja2 pytest`, then
       `pip check`;
    5. `pip freeze > /opt/tf-venv/freeze.txt`.
  - `serve`: FROM `toolchain`; COPY the source; `pip install . --no-deps --constraint …`; `ENTRYPOINT ["tensorfold"]`.
- **`Dockerfile.dockerignore`** (BuildKit's per-Dockerfile ignore; keeps everything in Harness paths): `.git`, `build/`,
  `.claude/`, `**/__pycache__`, `*.so`, `*.safetensors`, caches.
- **`tools/xpu/debs.lock`**: the deb table now inline in `bootstrap.sh` (`package|version|url|sha256`, one per line).
  `bootstrap.sh` reads it too, so there is one source of pins.
- **`build.sh`**: tags `tensorfold-xpu:tc-<hash12>`, hash over the Dockerfile, `debs.lock` and `constraints.txt`; prints
  the image ID. It never runs with oneAPI sourced.
- Python is whatever 26.04's `python3` is, expected 3.14 like the box venv `[UNVERIFIED]`; the `env` suite records it.

## 4. Runner integration (`tools/xpu/b70_runner.py`, new `tools/xpu/container.py`)

Opt-in with `TF_XPU_IMAGE` / `--image`; the venv mode stays the default until both agree (§7).

`container.py` builds the `docker run` argv (reusing `refuse_shell_meta`):

| Concern | Argument |
|---|---|
| lifetime | `--rm --name tf-<sha7>-<suite>`; on timeout the runner calls `docker kill <name>` |
| GPU | `--device /dev/dri --group-add <numeric gid of /dev/dri/renderD*>` (group names differ inside the image) |
| user | `--user <uid>:<gid>`, `HOME=/tmp/home`, so bundle files belong to the runner user |
| network | `--network none`: branch code is untrusted and needs no network |
| harness | runner checkout at `/harness:ro`; suites import `tools.xpu.suites` from there, never from the branch |
| code | branch worktree at `/src` (read-write: `pip install -e` writes egg-info) |
| output | suite out_dir at `/out` |
| Triton cache | `--triton-cache` dir at `/cache/triton`, `TRITON_CACHE_DIR=/cache/triton`; the existing fingerprint purge applies |
| models | `HF_HOME` at `/models:ro`, `HF_HOME=/models`, `HF_HUB_OFFLINE=1` |
| native ext | `build/xpu-ext/<hash>` at `/opt/tf-ext:ro`, `TF_XPU_EXT_DIR=/opt/tf-ext`, only when present (§5) |
| env | only `knob_env()` values (`TRITON_INTEL_*`, `IGC_*`); never the host environment |

Per suite the container runs `pip install -e /src --no-deps`, then `run_suite` with the same JSON stdin/stdout
protocol `invoke_suite` uses today.

`_collect_versions` splits:
- in the image: `dpkg-query` for the driver packages, torch / `torch.version.xpu` / triton, Python;
- on the host: `uname -r`, the `xe` binding, `xpu-smi discovery`, `icpx --version`;
- new `versions["image"]` = image ID, part of `toolchain_hash`. Update `tools/xpu/schema.json` and `schema_check.py`.

`health.check_gpu`, the STOP marker and publishing to `results` stay on the host, unchanged.

## 5. Native kernels (after K0)

- `tools/xpu/build_ext.sh` runs on the host, sources DLE 2026.1 in a subshell, and writes
  `build/xpu-ext/<hash>/` with a manifest. The hash covers the DLE version, the torch version and the source SHA.
- The runner mounts that directory (§4); `src/tensorfold/xpu/build.py` loads `.so` files from `TF_XPU_EXT_DIR` and never
  compiles inside the container.
- `env` fails if the manifest's torch / SYCL versions differ from the image's.
- Host and image share Ubuntu 26.04, so glibc and libstdc++ match. A self-contained DLE build stage is deferred until a
  second box needs one.

## 6. New `env` probes (run inside the container)

- Level Zero only: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` with the OpenCL ICD absent still passes `triton_add` and the
  DPAS flags. Decides whether `intel-opencl-icd` / `ocl-icd-libopencl1` leave the image.
- Triton's launcher builds with gcc alone vs gcc plus `python3-dev`.
- The image's compute-runtime 26.31 works against the host `xe` driver.
- Python version and image ID in `env.json`.

## 7. Rollout

1. Operator installs Docker on the host (`docker.io` from the 26.04 archive) and adds the runner user to `docker`, or
   uses rootless Docker.
2. Build `toolchain`; run `env` by hand with the §4 arguments; compare with the latest venv `env` bundle.
3. Set `TF_XPU_IMAGE` in `runner.env`; push a branch whose `.b70/run.yml` asks for `suites: [env, unit-host]`.
4. Iterate until a container bundle and a venv bundle agree on torch, triton, `torch.version.xpu`, the DPAS flags and
   `triton_add`.
5. Make container mode the default. In HARNESS.md mark `bootstrap.sh --apply` and `--system` deprecated, keep
   `--install-kernel` and the read-only report, and move `--swap` and `--models` to host steps. Remove the venv mode in
   a later PR.

## 8. Done when

- Host `pytest` and `ruff` are clean; tests in `tests/xpu/` cover the argv builder with fakes (no Docker needed).
- A green B70 bundle from container mode exists for the head SHA, with the image ID in `env.json`.
- HARNESS.md documents the host prerequisites, `build.sh`, and serving:

```bash
docker run --rm -it --device /dev/dri --group-add <render gid> -p 8000:8000 -v <HF_HOME>:/models:ro -e HF_HOME=/models tensorfold-xpu:<tag> serve ...
```

No CUDA path changes. Every file is Harness-owned except the AGENTS.md pointer, which already exists.
