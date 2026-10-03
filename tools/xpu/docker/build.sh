#!/usr/bin/env bash
# Build the pinned runtime or serving image without a compiler environment.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if command -v icpx >/dev/null || [[ -n "${ONEAPI_ROOT:-}${CMPLR_ROOT:-}${DLE_ROOT:-}" ]] ||
    [[ "${LD_LIBRARY_PATH:-}" == *oneapi* || "${LD_LIBRARY_PATH:-}" == *dle-2026* ]]; then
    echo 'Run build.sh from a shell without oneAPI/DLE sourced.' >&2
    exit 1
fi
target="${1:-toolchain}"
[[ "$target" == toolchain || "$target" == serve || "$target" == build ]] || {
    echo 'Usage: build.sh [toolchain|serve|build]' >&2; exit 2; }
cd "$root"
if [[ "$target" == build ]]; then
    # The native-kernel build image holds the compiler, so it has its own tag and is never the runtime image.
    hash="$(cat tools/xpu/docker/Dockerfile.build tools/xpu/debs.lock tools/xpu/constraints.txt | sha256sum)"
    tag="tensorfold-xpu-build:dle-2026.1-${hash:0:12}"
    docker build --file tools/xpu/docker/Dockerfile.build --tag "$tag" .
    printf 'Image: %s\n' "$tag"
    docker image inspect --format '{{.Id}}' "$tag"
    exit 0
fi
hash="$(cat tools/xpu/docker/Dockerfile tools/xpu/debs.lock tools/xpu/constraints.txt | sha256sum)"
tag="tensorfold-xpu:tc-${hash:0:12}"
if [[ "$target" == serve ]]; then
    # A serving tag also identifies the source, including local edits.
    source_hash="$(git diff HEAD -- src pyproject.toml | sha256sum)"
    tag="tensorfold-xpu:serve-${hash:0:12}-$(git rev-parse --short=12 HEAD)-${source_hash:0:12}"
fi
docker build --file tools/xpu/docker/Dockerfile --target "$target" --tag "$tag" .
printf 'Image: %s\n' "$tag"
docker image inspect --format '{{.Id}}' "$tag"
