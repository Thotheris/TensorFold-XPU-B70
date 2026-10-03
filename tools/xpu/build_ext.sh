#!/usr/bin/env bash
# Build K0's AOT extension entry point in an isolated DLE subshell.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
prefix="${DLE_PREFIX:-/opt/intel/dle-2026.1}"
python="${TF_XPU_PYTHON:-python3}"
[[ "$prefix" == /* && "$prefix" != /opt/intel/oneapi* ]] || { echo 'Use isolated DLE 2026.1.' >&2; exit 1; }
# Query binary wheels before sourcing the compiler environment.
metadata="$("$python" -c 'import json,torch; print(json.dumps({"torch":str(torch.__version__),"sycl":str(torch.version.xpu)}))')"
"$python" -c 'import json,sys; v=json.loads(sys.argv[1]); assert v["torch"].startswith("2.14.") and v["torch"].endswith("+xpu") and v["sycl"]=="20260100"' "$metadata"
sha="$(git -C "$root" rev-parse HEAD)"
[[ -z "$(git -C "$root" status --porcelain -- src/tensorfold/xpu)" ]] || { echo 'Commit native sources first.' >&2; exit 1; }
# K0 owns this module and its build_aot(output_dir) implementation.
[[ -f "$root/src/tensorfold/xpu/build.py" ]] || { echo 'K0 build.py is not implemented yet.' >&2; exit 1; }
[[ -f "$prefix/compiler/latest/env/vars.sh" ]] || { echo 'DLE compiler vars.sh is missing.' >&2; exit 1; }
compiler="$("$prefix/compiler/latest/bin/icpx" --version)"
[[ "$compiler" == *2026.1* ]] || { echo 'DLE 2026.1 is required.' >&2; exit 1; }
hash="$(printf '%s\n' "$metadata" "$compiler" "$sha" | sha256sum)"
out="$root/build/xpu-ext/${hash%% *}"
[[ ! -e "$out" ]] || { echo 'Build directory already exists; use its manifest or inspect it manually.' >&2; exit 1; }
mkdir -p "$out"
(
    # Intel's vars.sh reads unset variables; nounset applies again once it is sourced.
    set +u
    source "$prefix/compiler/latest/env/vars.sh"
    set -u
    export PYTHONPATH="$root/src" TORCH_XPU_ARCH_LIST=bmg TORCH_EXTENSIONS_DIR="$out/cache"
    "$python" -c 'from pathlib import Path; from tensorfold.xpu.build import build_aot; import sys; build_aot(Path(sys.argv[1]))' "$out"
)
"$python" -c 'import json,sys; from pathlib import Path; v=json.loads(sys.argv[2]); v.update(dle="2026.1",source_sha=sys.argv[3],compiler=sys.argv[4]); p=Path(sys.argv[1]); assert list(p.glob("*.so")),"build produced no libraries"; (p/"manifest.json").write_text(json.dumps(v,indent=2)+"\n")' "$out" "$metadata" "$sha" "$compiler"
printf 'TF_XPU_EXT_DIR=%s\n' "$out"
