#!/usr/bin/env bash
# Build the native XPU extensions with build_ext.sh inside the DLE build image; the host needs only Docker.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${TF_XPU_BUILD_IMAGE:-}"
[[ -n "$image" ]] || { echo 'Set TF_XPU_BUILD_IMAGE (tools/xpu/docker/build.sh build prints it).' >&2; exit 2; }
output="$(docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e GIT_CONFIG_COUNT=1 \
    -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/src -v "$root:/src" -w /src "$image" \
    bash tools/xpu/build_ext.sh)"
printf '%s\n' "$output" | sed "s#^TF_XPU_EXT_DIR=/src/#TF_XPU_EXT_DIR=$root/#"
