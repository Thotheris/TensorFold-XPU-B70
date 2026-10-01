#!/usr/bin/env bash
# Usage: bash tools/xpu/bootstrap.sh [--apply] [--system] [--swap] [--models]
#        [--install-kernel] [--venv PATH] [--dle-prefix PATH] [--hf-home PATH]
#        [--help] [--self-test]
# Default: read-only report. No kernel installation or reboot without explicit consent.
# Never source oneAPI here or in a profile. Native builds later use a subshell:
# bash -lc 'source <prefix>/compiler/<version>/env/vars.sh; ...'
# build_ext.sh is separate work. DLE variables in a binary torch wheel's runtime shell
# can cause Triton SIGSEGV. This script never sources setvars.sh or compiler vars.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
XPU_INDEX=https://download.pytorch.org/whl/xpu
DLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/c109e1ae-e02c-48a6-917b-b03b90d33f77/intel-deep-learning-essentials-2026.1.2.25_offline.sh
MODELS=(
    devan-carlin/Qwen3.8-27B-int4-AutoRound
    RedHatAI/Qwen3.8-27B-INT4
    letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound
    SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym
    z-lab/Qwen3.8-27B-DFlash2
)

usage() {
    head -n 11 "$SCRIPT_DIR/bootstrap.sh"
    printf '%s\n' \
        '--apply creates the venv and freezes torch/triton. Without it, nothing is changed.' \
        '--system permits explicit sudo for pinned debs, prerequisites and isolated DLE.' \
        '--swap separately permits explicit sudo for a 32 GiB swap file (requires --apply).' \
        '--models downloads the five snapshots and records their revisions (requires --apply).' \
        '--install-kernel requires --apply --system, never reboots, and is Ubuntu 24.04 only.' \
        '--self-test runs embedded, system-independent checks only.'
}

parse_args() {
    APPLY=0 SYSTEM=0 SWAP=0 DOWNLOAD_MODELS=0 INSTALL_KERNEL=0 SELF_TEST=0 HELP=0
    VENV="$HOME/.local/share/tensorfold-xpu/venv"
    DLE_PREFIX=/opt/intel/dle-2026.1
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    while (($#)); do
        case "$1" in
            --apply) APPLY=1 ;;
            --system) SYSTEM=1 ;;
            --swap) SWAP=1 ;;
            --models) DOWNLOAD_MODELS=1 ;;
            --install-kernel) INSTALL_KERNEL=1 ;;
            --self-test) SELF_TEST=1 ;;
            --help|-h) HELP=1 ;;
            --venv|--dle-prefix|--hf-home)
                if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                    printf 'MISSING value for %s\n' "$1" >&2
                    return 2
                fi
                case "$1" in
                    --venv) VENV="$2" ;;
                    --dle-prefix) DLE_PREFIX="$2" ;;
                    --hf-home) HF_HOME="$2" ;;
                esac
                shift
                ;;
            *) printf 'MISSING unknown argument: %s\n' "$1" >&2; return 2 ;;
        esac
        shift
    done
}

version_ge() {
    local left="${1%%[-+~]*}" right="${2%%[-+~]*}" i a b
    local -a lhs rhs
    [[ "$left" =~ ^[0-9]+(\.[0-9]+)*$ && "$right" =~ ^[0-9]+(\.[0-9]+)*$ ]] || return 1
    IFS=. read -r -a lhs <<< "$left"
    IFS=. read -r -a rhs <<< "$right"
    for ((i=0; i<${#lhs[@]} || i<${#rhs[@]}; i++)); do
        a="${lhs[i]:-0}" b="${rhs[i]:-0}"
        ((10#$a > 10#$b)) && return 0
        ((10#$a < 10#$b)) && return 1
    done
    return 0
}

parse_meminfo_kib() {
    local key="$1" field value units
    while read -r field value units; do
        if [[ "$field" == "$key:" && "$value" =~ ^[0-9]+$ && "$units" == kB ]]; then
            printf '%s\n' "$value"
            return 0
        fi
    done
    return 1
}

run_cmd() {
    if ((APPLY == 0)); then
        printf 'WARN WOULD-RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    printf 'OK RUN:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

section() { printf '\n[%s]\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
missing() { printf 'MISSING %s\n' "$*" >&2; }
report_output() { local line; while IFS= read -r line; do printf 'OK %s\n' "$line"; done; }

self_test() {
    local sample mem
    ! version_ge 6.17.0 6.18 || return 1
    ! version_ge 6.16 6.17.0 || return 1
    version_ge 6.17.0 6.16 || return 1
    version_ge 6.17.0 6.17 || return 1
    version_ge 1.32.0 1.31.9 || return 1
    version_ge 2.40.13 2.40 || return 1
    version_ge 26.31.39395.13 26.31 || return 1
    ! version_ge 6.17 6.17.1 || return 1
    sample=$'MemTotal:       32768000 kB\nMemAvailable:    4000000 kB\nCommitted_AS:   10000000 kB\nSwapTotal:             0 kB'
    mem="$(parse_meminfo_kib MemTotal <<< "$sample")" || return 1
    [[ "$mem" == 32768000 ]] || return 1
    ((mem < 64 * 1024 * 1024)) || return 1
    printf 'WARN embedded MemTotal %s KiB is below 64 GiB\n' "$mem"
    [[ "$(parse_meminfo_kib SwapTotal <<< "$sample")" == 0 ]] || return 1
    parse_args || return 1
    ((APPLY == 0 && SYSTEM == 0 && SWAP == 0 && DOWNLOAD_MODELS == 0)) || return 1
    [[ "$VENV" == "$HOME/.local/share/tensorfold-xpu/venv" ]] || return 1
    # A shell function is the sentinel, so even a broken dry-run cannot call a real sudo.
    sudo() { printf 'MISSING self-test executed sudo\n' >&2; return 97; }
    run_cmd sudo false || { unset -f sudo; return 1; }
    unset -f sudo
    printf 'OK bootstrap self-test passed\n'
}

package_version() {
    local result
    have dpkg-query || return 1
    result="$(dpkg-query -W -f='${Status}|${Version}\n' "$1" 2>/dev/null)" || return 1
    [[ "$result" == 'install ok installed|'* ]] || return 1
    printf '%s\n' "${result#*|}"
}

report_package() {
    local package="$1" pin="$2" mode="${3:-exact}" installed
    installed="$(package_version "$package")" || installed=""
    if [[ -z "$installed" ]]; then
        printf 'MISSING %s, --apply --system would install %s\n' "$package" "$pin"
    elif [[ "$mode" == minimum ]] && version_ge "$installed" "$pin"; then
        printf 'OK %s %s (minimum %s, leave installed)\n' "$package" "$installed" "$pin"
    elif [[ "$installed" == "$pin" ]]; then
        printf 'OK %s %s\n' "$package" "$installed"
    else
        printf 'WARN %s %s differs from pin %s, --apply --system would replace it' "$package" "$installed" "$pin"
        if version_ge "$installed" "$pin"; then printf ' (possible downgrade)'; fi
        printf '\n'
    fi
}

ubuntu_version() {
    # Read only the two needed fields, do not execute os-release as shell code.
    [[ -r /etc/os-release ]] || return 1
    local id version
    id="$(awk -F= '$1=="ID" {gsub(/"/,"",$2); print $2}' /etc/os-release)"
    version="$(awk -F= '$1=="VERSION_ID" {gsub(/"/,"",$2); print $2}' /etc/os-release)"
    [[ "$id" == ubuntu ]] || return 1
    printf '%s\n' "$version"
}

report() {
    local kernel pci="" line largest=0 size=0 mem swap firmware installed output distro
    section kernel
    kernel="$(uname -r 2>/dev/null)" || kernel=unknown
    if version_ge "$kernel" 6.17; then printf 'OK kernel %s >= 6.17\n' "$kernel"
    else printf 'WARN kernel %s, need >= 6.17, no automatic reboot\n' "$kernel"; fi
    printf 'WARN --apply --system --install-kernel permits linux-generic-hwe-24.04 only on Ubuntu 24.04\n'
    printf 'WARN exact HWE package version >= 6.17 is UNVERIFIED, inspect it before any manual reboot\n'
    section 'xe + PCI'
    if have lspci; then
        pci="$(lspci -Dnnk -d 8086:e223 2>/dev/null)" || pci=""
        if [[ -n "$pci" ]]; then
            printf 'OK PCI 8086:e223 (BMG-G31)\n'
            report_output <<< "$pci"
            if [[ "$pci" == *'Kernel driver in use: xe'* ]]; then printf 'OK xe bound to B70\n'
            else printf 'MISSING xe binding for B70\n'; fi
        else printf 'MISSING PCI 8086:e223\n'; fi
    else printf 'MISSING lspci, install pciutils manually if needed\n'; fi
    section firmware
    for firmware in bmg_guc bmg_huc; do
        if compgen -G "/lib/firmware/xe/${firmware}*" >/dev/null; then
            printf 'OK firmware matches /lib/firmware/xe/%s*\n' "$firmware"
        else printf 'MISSING /lib/firmware/xe/%s*, exact filename UNVERIFIED on box\n' "$firmware"; fi
    done
    section ReBAR
    if [[ -n "$pci" ]]; then
        output="$(lspci -Dvv -d 8086:e223 2>/dev/null)" || output=""
        while IFS= read -r line; do
            if [[ "$line" == *'64-bit'* && "$line" == *'prefetchable'* && "$line" != *'non-prefetchable'* ]]; then
                if [[ "$line" =~ size=([0-9]+)([GMK]) ]]; then
                    size="${BASH_REMATCH[1]}"
                    case "${BASH_REMATCH[2]}" in G) size=$((size * 1024));; K) size=$((size / 1024));; esac
                    if ((size > largest)); then largest=$size; fi
                fi
            fi
        done <<< "$output"
        if ((largest >= 32768)); then printf 'OK largest 64-bit prefetchable BAR is %s MiB\n' "$largest"
        elif ((largest > 0)); then printf 'WARN BAR is %s MiB, expected about 32G, not 256M; enable ReBAR in BIOS\n' "$largest"
        else printf 'MISSING readable BAR size, check lspci Region 2 and ReBAR sysfs on the box\n'; fi
    else printf 'MISSING B70 BAR information\n'; fi
    section compute-runtime
    report_package intel-opencl-icd 26.31.39395.13-0
    report_package libze-intel-gpu1 26.31.39395.13-0
    report_package libigdgmm12 22.10.0
    section IGC
    report_package intel-igc-core-2 2.40.13+22418
    report_package intel-igc-opencl-2 2.40.13+22418
    printf 'OK IGC 2.40.13 debs are labeled Ubuntu 24.04; dpkg accepted them on Ubuntu 26.04\n'
    section level-zero
    report_package libze1 1.32.0 minimum
    distro="$(ubuntu_version)" || distro=unknown
    if [[ "$distro" == 26.04 ]]; then
        printf 'WARN no u26.04 libze1 deb; --apply --system installs 1.32.0+u24.04 when the loader is below 1.32.0\n'
    fi
    section ocloc
    report_package intel-ocloc 26.31.39395.13-0
    if have ocloc; then printf 'OK ocloc at %s\n' "$(command -v ocloc)"
    else printf 'MISSING ocloc, --apply --system would install intel-ocloc\n'; fi
    section 'xpu-smi discovery'
    if have xpu-smi; then
        if have timeout; then
            if output="$(timeout 30s xpu-smi discovery 2>&1)"; then report_output <<< "$output"
            else printf 'WARN xpu-smi discovery failed or timed out: %s\n' "$output"; fi
        else printf 'WARN timeout missing, skip xpu-smi discovery to avoid a hung report\n'; fi
    else printf 'MISSING xpu-smi, Arc standalone builds do not provide diag; never call diag\n'; fi
    section DLE
    if [[ -d "$DLE_PREFIX/compiler" ]]; then printf 'OK DLE compiler tree at %s, verify version 2026.1 on box\n' "$DLE_PREFIX"
    else printf 'MISSING DLE 2026.1, --apply --system would install into %s only\n' "$DLE_PREFIX"; fi
    printf 'WARN installer must advertise --install-dir, otherwise installation stops\n'
    printf 'WARN do not source DLE in profiles or torch/Triton runtime shells (Triton SIGSEGV)\n'
    section venv
    if [[ -x "$VENV/bin/python" ]]; then
        if output="$("$VENV/bin/python" -B -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null)"; then
            if [[ "$output" == 2.14.*+xpu ]]; then printf 'OK torch %s in %s\n' "$output" "$VENV"
            else printf 'WARN torch %s is not 2.14.x+xpu, --apply refuses to replace it\n' "$output"; fi
        else printf 'MISSING torch in %s\n' "$VENV"; fi
    else printf 'MISSING venv %s\n' "$VENV"; fi
    printf 'WARN --apply would create venv, prefer torch 2.14.1+xpu, freeze bundled triton, install constrained pure deps\n'
    section RAM/swap
    if [[ -r /proc/meminfo ]]; then
        mem="$(parse_meminfo_kib MemTotal < /proc/meminfo)" || mem=""
        swap="$(parse_meminfo_kib SwapTotal < /proc/meminfo)" || swap=""
        if [[ -n "$mem" ]] && ((mem >= 64 * 1024 * 1024)); then printf 'OK host RAM %s KiB >= 64 GiB\n' "$mem"
        else printf 'WARN host RAM %s KiB < 64 GiB, this box is expected to have 32 GB\n' "${mem:-unknown}"; fi
        if [[ -n "$swap" ]] && ((swap >= 32 * 1024 * 1024)); then printf 'OK swap %s KiB >= 32 GiB\n' "$swap"
        else printf 'WARN swap %s KiB, re-run with --apply --swap to create a 32 GiB swap file\n' "${swap:-unknown}"; fi
    else printf 'MISSING /proc/meminfo, expected host RAM 32 GB; re-run with --apply --swap on Ubuntu\n'; fi
    section models
    printf 'WARN --apply --models downloads into HF_HOME=%s and records snapshot commit SHAs\n' "$HF_HOME"
    for line in "${MODELS[@]}"; do printf 'OK model target %s\n' "$line"; done
    if ((APPLY == 0)); then
        run_cmd python3 -m venv "$VENV"
        printf '\nOK dry-run complete, no sudo, wget, dpkg install, pip, downloads or writes performed\n'
    fi
}

install_venv() {
    local python="$VENV/bin/python" torch_version versions selected freeze pins names constraint
    have python3 || { missing python3; return 1; }
    if [[ ! -x "$python" ]]; then run_cmd python3 -m venv "$VENV" || return 1; fi
    # Refuse a mismatched existing wheel before upgrading even pip.
    torch_version="$("$python" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null)" || torch_version=""
    if [[ -n "$torch_version" && "$torch_version" != 2.14.*+xpu ]]; then
        missing "existing torch $torch_version, use a separate --venv instead of replacing it"
        return 1
    fi
    run_cmd "$python" -m pip install --upgrade pip || return 1
    if [[ -z "$torch_version" ]]; then
        versions="$("$python" -m pip index versions torch --index-url "$XPU_INDEX")" || return 1
        selected="$("$python" -c '
import re, sys
versions = set(re.findall(r"(?<![\w.])2\.14\.\d+\+xpu(?![\w.])", sys.stdin.read()))
if not versions:
    raise SystemExit("No torch 2.14.*+xpu wheel found, refusing CPU/CUDA/2.15")
print("2.14.1+xpu" if "2.14.1+xpu" in versions else max(versions, key=lambda v: int(v.split(".")[2].split("+")[0])))
' <<< "$versions")" || return 1
        run_cmd "$python" -m pip install "torch==$selected" --index-url "$XPU_INDEX" || return 1
    fi
    # Freeze the installed bundle, not the unconfirmed placeholder distribution.
    freeze="$("$python" -m pip freeze)" || return 1
    pins="$("$python" - "$freeze" <<'PY'
import importlib.metadata as m
import re
import sys

frozen = {}
for line in sys.argv[1].splitlines():
    if "==" in line and not line.startswith("#"):
        name, version = line.split("==", 1)
        frozen[re.sub(r"[-_.]+", "-", name).lower()] = (name, version)

def freeze_pin(name, expected):
    key = re.sub(r"[-_.]+", "-", name).lower()
    pin = frozen.get(key)
    if pin is None or pin[1] != expected:
        raise SystemExit(f"Missing or mismatched pip freeze entry: {name}=={expected}")
    return "==".join(pin)

torch = m.distribution("torch")
if not re.fullmatch(r"2\.14\.\d+\+xpu", torch.version):
    raise SystemExit(f"Refusing torch {torch.version}, require 2.14.x+xpu")
triton = sorted(
    (d.metadata["Name"], d.version) for d in m.distributions()
    if "triton" in d.metadata.get("Name", "").lower()
)
if not triton or any(not re.fullmatch(r"3\.8\.\d+(?:\+[\w.]+)?", version) for _, version in triton):
    raise SystemExit(f"Missing or unexpected bundled triton (expected 3.8.x): {triton}")
print("# Pass --constraint to every later pip install. Never upgrade torch or triton.")
print(freeze_pin("torch", torch.version))
for name, version in triton:
    print(freeze_pin(name, version))
for requirement in torch.requires or []:
    if "triton" in requirement.lower():
        print(f"# torch requirement: {requirement}")
PY
)" || return 1
    constraint="$VENV/constraints.txt"
    printf '%s\n' "$pins" > "$constraint" || return 1
    if { [[ -e "$SCRIPT_DIR/constraints.txt" && -w "$SCRIPT_DIR/constraints.txt" ]] ||
         [[ ! -e "$SCRIPT_DIR/constraints.txt" && -w "$SCRIPT_DIR" ]]; }; then
        run_cmd cp "$constraint" "$SCRIPT_DIR/constraints.txt" || return 1
        constraint="$SCRIPT_DIR/constraints.txt"
    else
        printf 'WARN repo constraints not writable, using and keeping %s\n' "$constraint"
    fi
    names="$(printf '%s\n' "$pins" | awk '!/^#/ && !/^torch==/ {sub(/==.*/, ""); print}')"
    local -a distributions
    read -r -a distributions <<< "$(printf '%s\n' "$names" | tr '\n' ' ')"
    run_cmd "$python" -m pip show torch "${distributions[@]}" || return 1
    printf 'OK installed torch/triton pins:\n%s\n' "$pins"
    (
        cd -- "$REPO_ROOT" || exit 1
        run_cmd "$python" -m pip install -e . --no-deps --constraint "$constraint" || exit 1
        run_cmd "$python" -m pip install --constraint "$constraint" \
            numpy huggingface-hub tokenizers safetensors jinja2 pytest || exit 1
    ) || return 1
    run_cmd "$python" -m pip check || return 1
    "$python" - "$constraint" <<'PY'
import importlib.metadata as m
from pathlib import Path
import re
import sys

for line in Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("#") or "==" not in line:
        continue
    name, expected = line.split("==", 1)
    actual = m.version(name)
    if actual != expected:
        raise SystemExit(f"Protected wheel changed: {name} {expected} -> {actual}")
if not re.fullmatch(r"2\.14\.\d+\+xpu", m.version("torch")):
    raise SystemExit("torch must remain 2.14.x+xpu")
print("OK protected torch/triton wheels unchanged")
PY
}

install_system() {
    local distro installed temp file url digest package version help_text
    local -a debs=()
    distro="$(ubuntu_version)" || { missing 'Ubuntu 24.04 or 26.04 required for --system'; return 1; }
    [[ "$distro" == 24.04 || "$distro" == 26.04 ]] || { missing "unsupported Ubuntu $distro"; return 1; }
    for file in curl sha256sum dpkg-query dpkg sudo apt-get; do
        have "$file" || { missing "$file needed by --system"; return 1; }
    done
    [[ "$DLE_PREFIX" == /* && "$DLE_PREFIX" != / && "$DLE_PREFIX" != /opt/intel/oneapi* ]] ||
        { missing 'DLE prefix must be absolute and outside /opt/intel/oneapi'; return 1; }
    if have readlink; then
        DLE_PREFIX="$(readlink -m -- "$DLE_PREFIX")" || return 1
        [[ "$DLE_PREFIX" != / && "$DLE_PREFIX" != /opt/intel/oneapi* ]] ||
            { missing 'DLE prefix resolves into the default oneAPI tree'; return 1; }
    fi
    temp="$(mktemp -d /tmp/tensorfold-b70-bootstrap.XXXXXXXX)" || return 1
    # Keep verified installers for inspection on failure; do not delete operator files.
    printf 'OK system downloads use %s (remove manually after inspection)\n' "$temp"
    while IFS='|' read -r package version url digest; do
        installed="$(package_version "$package")" || installed=""
        if [[ "$package" == libze1 ]] && version_ge "${installed:-0}" 1.32.0; then
            printf 'OK libze1 %s >= 1.32.0, left unchanged\n' "$installed"
            continue
        fi
        if [[ "$installed" == "$version" ]]; then
            printf 'OK %s already pinned at %s\n' "$package" "$version"
            continue
        fi
        if [[ -n "$installed" ]] && version_ge "$installed" "$version"; then
            printf 'WARN explicit --system downgrade: %s %s -> %s\n' "$package" "$installed" "$version"
        fi
        if [[ "$package" == libze1 && "$distro" == 26.04 ]]; then
            printf 'WARN no u26.04 libze1 deb; installing the 1.32.0+u24.04 pin\n'
        fi
        file="$temp/${url##*/}"
        run_cmd curl --fail --location --proto '=https' --tlsv1.2 --output "$file" "$url" || return 1
        printf '%s  %s\n' "$digest" "$file" | sha256sum --check - || return 1
        debs+=("$file")
    done <<'DEBS'
libigdgmm12|22.10.0|https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/libigdgmm12_22.10.0_amd64.deb|6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54
intel-igc-core-2|2.40.13+22418|https://github.com/intel/intel-graphics-compiler/releases/download/v2.40.13/intel-igc-core-2_2.40.13+22418_amd64.deb|ebd795e9fddf303a9b24b7f04545d8ddd9ad1f85b3d0cb1166476fab24da6d44
intel-igc-opencl-2|2.40.13+22418|https://github.com/intel/intel-graphics-compiler/releases/download/v2.40.13/intel-igc-opencl-2_2.40.13+22418_amd64.deb|4f990874efc11c3f6091a663b08aef576c4af592dcd8f12e116f8c2fc92d34d9
libze1|1.32.0+u24.04|https://github.com/oneapi-src/level-zero/releases/download/v1.32.0/libze1_1.32.0+u24.04_amd64.deb|3c846af24f84a89150f6a4c6adcb4ea4ebef74dc119fe44f4e269bfaa72c7ba6
libze-intel-gpu1|26.31.39395.13-0|https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/libze-intel-gpu1_26.31.39395.13-0_amd64.deb|1722943f81b576b9bb8d61016464208f48ce533dc3bf24ad39605293115cc289
intel-opencl-icd|26.31.39395.13-0|https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/intel-opencl-icd_26.31.39395.13-0_amd64.deb|5a9c9e8fdca8a2f9e22754b1a4618c7babf21d7c3ab3503c680005007c7a8c44
intel-ocloc|26.31.39395.13-0|https://github.com/intel/compute-runtime/releases/download/26.31.39395.13/intel-ocloc_26.31.39395.13-0_amd64.deb|12c5e61ed1dca5cbf38494e280abf88100a451580d57c44f601a17d9727e465e
DEBS
    run_cmd sudo apt-get install -y ocl-icd-libopencl1 || return 1
    if ((${#debs[@]})); then
        run_cmd sudo dpkg -i "${debs[@]}" || {
            missing 'deb dependency or compatibility failure, inspect before any manual apt repair'
            return 1
        }
    fi
    if [[ ! -d "$DLE_PREFIX/compiler" ]]; then
        file="$temp/${DLE_URL##*/}"
        printf 'WARN DLE offline script has no published SHA256 in Appendix A; the URL is Intel registration center\n'
        run_cmd curl --fail --location --proto '=https' --tlsv1.2 --output "$file" "$DLE_URL" || return 1
        if ! grep -a -q -- '--install-dir' "$file"; then
            help_text=""
            if have timeout; then help_text="$(timeout 30s sh "$file" --help 2>&1)" || true; fi
            [[ "$help_text" == *--install-dir* ]] || {
                missing 'DLE installer does not advertise --install-dir; refusing default /opt/intel/oneapi'
                return 1
            }
        fi
        run_cmd sudo sh "$file" -a --silent --eula accept --install-dir "$DLE_PREFIX" || return 1
    else printf 'OK existing isolated DLE compiler tree, no reinstall\n'; fi
    if ((INSTALL_KERNEL)); then
        if [[ "$distro" == 24.04 ]]; then
            printf 'WARN HWE candidate is not guaranteed >= 6.17; inspect installed version before manual reboot\n'
            run_cmd sudo apt-get install -y linux-generic-hwe-24.04 || return 1
        else printf 'OK Ubuntu 26.04 stock kernel is expected >= 6.17, no kernel package change\n'; fi
    fi
    printf 'OK system step complete, no profiles changed and no reboot requested\n'
}

install_swap() {
    local swap
    [[ -r /proc/meminfo ]] || { missing '/proc/meminfo required for --swap'; return 1; }
    swap="$(parse_meminfo_kib SwapTotal < /proc/meminfo)" || return 1
    if ((swap >= 32 * 1024 * 1024)); then printf 'OK swap >= 32 GiB, no change\n'; return 0; fi
    [[ ! -e /swapfile && ! -L /swapfile ]] || {
        missing '/swapfile already exists, refusing to overwrite it; inspect and configure swap manually'
        return 1
    }
    for swap in sudo chmod mkswap swapon grep tee; do
        have "$swap" || { missing "$swap required for --swap"; return 1; }
    done
    [[ -r /etc/fstab ]] || { missing '/etc/fstab must be readable before --swap'; return 1; }
    if grep -Eq '^[[:space:]]*/swapfile[[:space:]]' /etc/fstab; then
        missing 'fstab references missing /swapfile, inspect its options manually before creation'
        return 1
    fi
    printf 'WARN explicit --apply --swap creates /swapfile; filesystem support is UNVERIFIED on this box\n'
    if have fallocate; then run_cmd sudo fallocate -l 32G /swapfile || return 1
    else
        have dd || { missing 'fallocate and dd'; return 1; }
        run_cmd sudo dd if=/dev/zero of=/swapfile bs=1M count=32768 status=progress || return 1
    fi
    run_cmd sudo chmod 600 /swapfile || return 1
    run_cmd sudo mkswap /swapfile || return 1
    run_cmd sudo swapon /swapfile || return 1
    if ! grep -Eq '^[[:space:]]*/swapfile[[:space:]]' /etc/fstab; then
        printf '/swapfile none swap sw 0 0\n' | run_cmd sudo tee -a /etc/fstab || return 1
    fi
}

download_models() {
    local repo output snapshot sha python="$VENV/bin/python"
    local -a downloader
    if [[ -x "$VENV/bin/hf" ]]; then downloader=("$VENV/bin/hf" download)
    elif have hf; then downloader=("$(command -v hf)" download)
    elif [[ -x "$VENV/bin/huggingface-cli" ]]; then downloader=("$VENV/bin/huggingface-cli" download)
    elif have huggingface-cli; then downloader=("$(command -v huggingface-cli)" download)
    else missing 'hf or huggingface-cli required for --models'; return 1; fi
    run_cmd mkdir -p "$HF_HOME" || return 1
    export HF_HOME
    for repo in "${MODELS[@]}"; do
        printf 'OK RUN:'; printf ' %q' "${downloader[@]}" "$repo"; printf '\n'
        output="$("${downloader[@]}" "$repo")" || return 1
        snapshot="$(printf '%s\n' "$output" | tail -n 1)"
        sha="${snapshot##*/}"
        [[ "$snapshot" == */snapshots/* && "$sha" =~ ^[0-9a-f]{40}$ && -d "$snapshot" ]] || {
            missing "download output for $repo is not a snapshot commit directory: $snapshot"
            return 1
        }
        "$python" - "$HF_HOME/tensorfold-xpu-revisions.json" "$repo" "$sha" <<'PY' || return 1
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}
data[sys.argv[2]] = sys.argv[3]
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
        printf 'OK snapshot %s %s\n' "$repo" "$sha"
    done
}

main() {
    local failed=0
    parse_args "$@" || return $?
    if ((HELP)); then usage; return 0; fi
    if ((SELF_TEST)); then self_test; return $?; fi
    report
    ((APPLY)) || return 0
    [[ "$VENV" == /* && "$HF_HOME" == /* ]] || {
        missing '--venv and --hf-home must be absolute paths'
        return 1
    }
    if ((INSTALL_KERNEL && ! SYSTEM)); then
        missing '--install-kernel requires --apply --system'
        return 1
    fi
    if ((SYSTEM)); then install_system || return 1; fi
    install_venv || return 1
    if ((SWAP)); then install_swap || failed=1; fi
    if ((DOWNLOAD_MODELS)); then download_models || failed=1; fi
    return "$failed"
}

main "$@"
