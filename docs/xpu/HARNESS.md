# B70 harness

## What this is

The harness prepares a user-run Ubuntu box with one Intel Arc Pro B70 (PCI 8086:E223,
BMG-G31), checks new `xpu/*` heads in isolated worktrees, and publishes result bundles on
the fork's orphan `results` branch. This host is expected to have 32 GB of RAM. Local host
tests are not GPU results. See [PORT_PLAN.md](PORT_PLAN.md) for ownership and milestones,
and [B70_NATIVE_KERNEL_GUIDE.md](B70_NATIVE_KERNEL_GUIDE.md) for hardware and build rules.

## Box setup

From the repository root, start with the read-only report:

```bash
bash tools/xpu/bootstrap.sh
bash tools/xpu/bootstrap.sh --self-test
```

Missing commands, firmware or hardware produce `MISSING` lines, not a failing dry-run.
The report reads package metadata using `dpkg-query`, not `dpkg` installation commands.
It never invokes sudo, wget, pip or downloads. `xpu-smi discovery` has a 30 second deadline;
without `timeout` it is skipped. Arc standalone builds do not have `xpu-smi diag`.

Create the venv and install the constrained Python environment:

```bash
bash tools/xpu/bootstrap.sh --apply
```

The default venv is `~/.local/share/tensorfold-xpu/venv`. `--venv /absolute/path` changes it.
Python 3.11 or newer with `venv` support must already be available. This mode does not
change system packages, call sudo, install MLX, or install the `grammar` extra. The latter
pulls torch through xgrammar and is not allowed in the pinned runtime.

Bootstrap queries the XPU wheel index, prefers `torch==2.14.1+xpu`, and otherwise accepts
only an available `2.14.*+xpu` release. It refuses an existing CPU, CUDA or 2.15 wheel rather
than replacing it. After installing torch first, it discovers the exact bundled Triton
distribution names and versions from installed metadata, cross-checks `pip freeze`, and
prints `pip show` and torch's Triton requirement. The committed
`pytorch-triton-xpu==3.8.0` line is an unconfirmed starting placeholder, not an authoritative
package name. Bootstrap rewrites it from `pip freeze` for the installed bundle before any
later constrained dependency install.

The exact pins are saved to both `tools/xpu/constraints.txt` and the venv's
`constraints.txt`. If the repository copy is not writable, the venv copy is used instead.
The editable project install uses `--no-deps`; numpy, huggingface-hub, tokenizers,
safetensors, jinja2 and pytest are installed separately under those constraints.
`pip check` and an exact torch/Triton pin comparison run afterward. Never upgrade either
protected wheel, and pass the generated constraints file to subsequent pip installs.

For explicit system installation:

```bash
bash tools/xpu/bootstrap.sh --apply --system
```

Every sudo command is printed before execution. No Intel apt repository is needed.
Bootstrap downloads the pinned, non-dbgsym debs into a new directory under `/tmp`, verifies
each supplied SHA256 before `sudo dpkg -i`, and installs `ocl-icd-libopencl1` with apt.
Already pinned packages are left alone. A different compute-runtime or IGC version is
replaced only in this explicitly requested mode, with a warning before a downgrade.
Temporary installers remain available for inspection, including on failure.

The pins are compute-runtime 26.31.39395.13, IGC 2.40.13 (build 22418), libigdgmm12 22.10.0,
`intel-ocloc` 26.31.39395.13, and Level Zero loader >= 1.32.0. Ubuntu 26.04's archive
copies (compute-runtime 26.05, IGC 1.0.17791, `libze1` 1.28.2) are too old. There is no
`+u26.04` loader deb. Bootstrap installs `libze1_1.32.0+u24.04` when the installed loader
is below 1.32.0, on both 24.04 and 26.04, and installs matching `libze-dev` headers.
Triton needs `level_zero/ze_api.h` from that package. An already newer loader is left
alone. Those
debs, including the IGC packages labeled Ubuntu 24.04, unpacked on the 26.04 test box.
`dpkg` errors stop the step. There is no automatic `apt-get -f` repair.

DLE 2026.1 is installed from Intel's 2026.1.2.25 offline installer into
`/opt/intel/dle-2026.1`, or an absolute `--dle-prefix` outside `/opt/intel/oneapi`.
The installer must advertise `--install-dir` in `--help`; otherwise bootstrap stops.
The standard apt package `intel-deep-learning-essentials-2026.1` installs into
`/opt/intel/oneapi`, so bootstrap does not use it. The offline command is:

```bash
sudo sh ./intel-deep-learning-essentials-2026.1.2.25_offline.sh \
  -a --silent --eula accept --install-dir /opt/intel/dle-2026.1
```

Never source DLE or oneAPI in `~/.bashrc`, `/etc/profile`, the runner, or a shell running
binary torch wheels. That mixture causes Triton SIGSEGV. Native builds later use only a
separate subshell such as `bash -lc 'source <prefix>/compiler/<version>/env/vars.sh; ...'`.
`build_ext.sh` is separate work. Bootstrap does not source any vars script or edit profiles.

The running kernel must be >= 6.17 with `xe` bound to the B70. The test box
`5950x-server` runs `7.0.0-34-generic`. The operator confirmed ReBAR and Above 4G
Decoding in the BIOS. Firmware should match `bmg_guc*` and `bmg_huc*` in
`/lib/firmware/xe`; exact filenames are not yet confirmed.
ReBAR should expose about 32G in Region 2 or the largest 64-bit prefetchable BAR, not 256M.
Fix firmware, driver binding and BIOS settings manually. Kernel changes require:

```bash
bash tools/xpu/bootstrap.sh --apply --system --install-kernel
```

On Ubuntu 24.04 this prints and runs `sudo apt-get install -y linux-generic-hwe-24.04`.
That package is only a suggestion, its guarantee of >= 6.17 is unverified. Confirm the
installed kernel version before a manual reboot. Ubuntu 26.04 stock is expected to
already satisfy the requirement, so no kernel package is installed there. Bootstrap
never reboots and never installs a kernel under `--apply` or `--apply --system` alone.

RAM below 64 GiB warns rather than fails. If swap totals less than 32 GiB, the report says
to re-run with `--apply --swap`. This is separate, explicit permission for sudo:

```bash
bash tools/xpu/bootstrap.sh --apply --swap
```

It creates a 32 GiB `/swapfile` only if that path does not exist, sets mode 600, runs
`mkswap` and `swapon`, then appends an fstab entry only if absent. It uses `fallocate`,
or `dd` when fallocate is unavailable. Filesystem failures stop the step; existing swap
files are never overwritten. Dry-run and `--apply` alone never create swap.

Optionally populate model snapshots:

```bash
bash tools/xpu/bootstrap.sh --apply --models --hf-home /models
```

The five targets are:

- `devan-carlin/Qwen3.8-27B-int4-AutoRound`
- `RedHatAI/Qwen3.8-27B-INT4`
- `letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound`
- `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym`
- `z-lab/Qwen3.8-27B-DFlash2`

`hf download` (or `huggingface-cli download`) uses `HF_HOME`. Bootstrap checks that the
snapshot directory name is a commit SHA and records `{"repo": "sha"}` entries in
`$HF_HOME/tensorfold-xpu-revisions.json`. Downloads are opt-in and can consume substantial
disk space. `--hf-home` defaults to `HF_HOME`, or `~/.cache/huggingface`.

## Runner install and enable

Check out this fork at `~/.local/share/tensorfold-xpu/repo` for the supplied service:

```bash
mkdir -p "$HOME/.local/share/tensorfold-xpu"
git clone https://github.com/Thotheris/TensorFold-XPU-B70.git \
  "$HOME/.local/share/tensorfold-xpu/repo"
cd "$HOME/.local/share/tensorfold-xpu/repo"
mkdir -p "$HOME/.config/systemd/user" "$HOME/.config/tensorfold-xpu"
cp tools/xpu/systemd/tensorfold-b70-runner.service "$HOME/.config/systemd/user/"
cp tools/xpu/systemd/tensorfold-b70-runner.timer "$HOME/.config/systemd/user/"
cp tools/xpu/runner.env.example "$HOME/.config/tensorfold-xpu/runner.env"
```

For an existing checkout anywhere else, simply copy the units, then edit the copied
service's `ExecStart` to the absolute Python and runner script paths. Also edit its
`Environment=TF_XPU_PYTHON=...` default if the venv is elsewhere. No wrapper is needed.
Edit `runner.env` to actual absolute Linux paths, replacing `/home/operator`.
EnvironmentFile does not expand `$HOME`, `~`, or other variables. Its leading `-` in
the unit means a missing file is allowed, but using a configured file is recommended.
`TF_XPU_REPO` does not redirect `ExecStart`; the service path must match the checkout.

The environment keys are `TF_XPU_REPO`, `TF_XPU_PYTHON`, `TF_XPU_VENV`, `TF_XPU_WORK`,
`TF_XPU_STATE_DIR`, `HF_HOME` and `TF_XPU_REMOTE=origin`. Do not put tokens or oneAPI
variables in this file. The runner scrubs secrets from bundles, but the box should not
need tokens in environment files. Configure any required fork Git access separately,
using least privilege and no credentials embedded in remote URLs.

Start polling:

```bash
systemctl --user daemon-reload
systemctl --user enable --now tensorfold-b70-runner.timer
systemctl --user status tensorfold-b70-runner.timer
journalctl --user -u tensorfold-b70-runner.service
```

The service is oneshot and never sources oneAPI. The timer runs two minutes after boot,
then every five minutes after activation, with 30 second accuracy and `Persistent=true`.
It does not run concurrent instances of the same service. Each suite is a separate
process, killed at its deadline. Systemd's start timeout is four hours, only as a
backstop if that process cannot be killed.

Optionally ask an administrator to run `loginctl enable-linger "$USER"` to keep user
timers running after logout. This is an operator choice, not a bootstrap action.

## Reset the stop flag

The default stop flag is `$HOME/.local/state/tensorfold-xpu/STOP`, or `STOP` beneath
`TF_XPU_STATE_DIR` when overridden. A wedged device or an xpu-smi timeout creates it.
A missing `xpu-smi` binary fails that run and does not latch the queue. The queue does
not continue until the flag is manually removed.
Inspect the logs and restore GPU health first, then reset:

```bash
rm "$HOME/.local/state/tensorfold-xpu/STOP"
```

Use the overridden state directory when configured. Removing the flag does not repair
the device, and bootstrap never resets it or reboots the host.

## Suites and `.b70/run.yml`

A requested `xpu/*` head carries `.b70/run.yml`:

```yaml
suites: [env, unit-xpu, kernels:gdn, e2e:27b-smoke]
baseline: xpu/main
timeout_min: 90
model_cache: /models
```

`suites` selects the comma-separated suite names. `baseline` identifies the comparison
branch, `timeout_min` bounds the run, and `model_cache` supplies the box's model-cache
path. Keep this file small and only request implemented tiers.

`env` is implemented. `unit-xpu`, `kernels:<name>`, `e2e:27b-smoke`,
`e2e:nemotron-smoke` and `e2e:*-bench` are stubs (TODO). The sample describes a future
full smoke run, not a claim that those stubs currently execute GPU tests. A green bundle
must contain real checks for the changed kernel before a port milestone can be done.

## Bundle layout

The `results` branch holds `runs/<branch-slug>/<sha7>-<utc>/`:

```text
env.json
pytest.xml
pytest.txt
kernels/*.json
e2e/*.json
logs/
summary.md
```

Only outputs from the requested, implemented suites can supply real measurements.
The root `index.jsonl` has one entry per run. The branch slug replaces `/` with `--`,
for example `xpu/k1/gdn` becomes `xpu--k1--gdn`. Environment metadata records toolchain
versions, device probes and model revisions; summary and JSON metrics describe check
outcomes and performance. Never infer a GPU pass from a host test or a skipped stub.

## How agents read results

In a Bash shell, run exactly:

```bash
git fetch origin results && git show origin/results:index.jsonl | tail
```

Then select the exact head SHA and inspect its bundle, for example:

```bash
git show origin/results:runs/xpu--k1--gdn/<sha7>-<utc>/summary.md
git show origin/results:runs/xpu--k1--gdn/<sha7>-<utc>/env.json
```

In Windows PowerShell 5.1 use separate commands rather than `&&`, and
`Select-Object -Last 10` rather than `tail`. Read the bundle for the exact SHA being
evaluated, including failures and bitwise checks, before reporting any hardware result.

## Safety

The runner only considers `xpu/*` refs. Branch execution is isolated in disposable
worktrees and runs as the unprivileged user, not root. Suite processes keep the harness
checkout as their working directory and import `tools.xpu` from that checkout, so a
branch cannot shadow the runner. `pip install -e` of the worktree does execute that
commit's build. Only `origin results` may be pushed; code branches and upstream must
never be pushed by the runner. Never merge the results branch into code. Restrict
branch authors to trusted collaborators because worktree isolation is not a security
sandbox for executing Python.

No secrets, model weights or native binaries belong in result bundles. Redaction is a
defense in depth, not permission to expose credentials. Keep the machine free of tokens
in runner configuration and avoid credential-bearing URLs.

The runner purges its Triton cache when the toolchain hash or any `TRITON_INTEL_*` or
`IGC_*` environment value changes. These knobs are not reliably part of Triton's cache
key. Keep DLE out of runtime PATH and use a separate shell for native builds.

## Confirmed on 5950x-server

The venv is `~/.local/share/tensorfold-xpu/venv` (Python 3.14). It has
`torch==2.14.1+xpu` and `triton-xpu==3.8.0`. `torch.version.xpu` is `20260100`.
`torch.xpu` sees Intel Arc Pro B70. Both
`has_subgroup_matrix_multiply_accumulate` and `has_subgroup_2d_block_io` are true,
so `ocloc` is visible. A 128-element int32 Triton add matched. `data_ptr()` was
above 2^63. Triton failed to compile until `libze-dev` 1.32.0 supplied
`level_zero/ze_api.h`. `xpu-smi` is not installed.

Do not treat the host-RAM deltas as an idle shadow result. The GPU was already in
use. `mem_get_info()` reported about 112 MiB free while a 2 GiB allocation still
succeeded. A later 2 GiB step, then another 2 GiB, moved `MemAvailable` by about
-1.1 GiB and then -1.7 GiB, and `Committed_AS` by about +2.7 GiB and then +2.0 GiB.

## Not yet confirmed on this box

- DLE offline installer support for `--install-dir`, and the script's SHA256 (none is published here).
- A host-RAM shadow measurement taken while the GPU is idle.
- Swapfile creation and activation on the box's filesystem.
- The systemd user timer and results push using the box's configured Git access.
- Exact `bmg_guc*` and `bmg_huc*` firmware filenames.
