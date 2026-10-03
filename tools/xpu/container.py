"""Container suites use an immutable runtime image and the trusted host harness."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

__all__ = ["container_name", "docker_argv", "image_id", "render_gids", "runtime_versions"]


def container_name(sha: str, suite: str) -> str:
    """Docker names identify a head and a shell-free suite name."""
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        raise ValueError("invalid head SHA")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:_-]*", suite):
        raise ValueError("invalid suite name")
    return f"tf-{sha[:7]}-{suite.replace(':', '-')}"


def render_gids(dri: Path = Path("/dev/dri")) -> list[int]:
    """Numeric render-node groups work across host and image group databases."""
    groups = sorted({node.stat().st_gid for node in dri.glob("renderD*")})
    if not groups:
        raise ValueError("no /dev/dri/renderD* nodes")
    return groups


def image_id(image: str, *, repo: Path) -> str:
    """A local image reference resolves once; the runner never pulls an image."""
    from .b70_runner import run_cmd

    if not image or image.startswith("-") or any(c.isspace() for c in image):
        raise ValueError("invalid image reference")
    completed = run_cmd(["docker", "image", "inspect", "--format", "{{.Id}}", image], cwd=repo, timeout=20)
    identifier = completed.stdout.strip()
    if completed.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", identifier):
        raise RuntimeError(completed.stderr.strip() or "Docker image is not available locally")
    return identifier


def docker_argv(
    *, image: str, name: str, repo: Path, worktree: Path, out_dir: Path, cache: Path,
    model_cache: Path | None, gids: list[int], knobs: dict[str, str], mode: str = "suite",
    native_ext: Path | None = None, uid: int | None = None, gid: int | None = None,
) -> list[str]:
    """Only explicit mounts and Intel compiler knobs enter the offline runtime."""
    from .b70_runner import knob_env, refuse_shell_meta

    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("container requires an immutable image ID")
    if not re.fullmatch(r"tf-[A-Za-z0-9_-]+", name) or mode not in {"suite", "versions"}:
        raise ValueError("invalid container name or mode")
    argv = ["docker", "run", "--rm", "--pull=never", "--name", name, "--network", "none", "--device", "/dev/dri",
            "--user", f"{os.getuid() if uid is None else uid}:{os.getgid() if gid is None else gid}",
            "--workdir", "/harness", "--interactive", "--entrypoint", "/opt/tf-venv/bin/python"]
    for group in gids:
        if not isinstance(group, int) or group < 0:
            raise ValueError("invalid render group")
        argv.extend(["--group-add", str(group)])

    def mount(source: Path, target: str, readonly: bool = False) -> None:
        source = source.resolve()
        if not source.is_dir() or any(c in str(source) for c in ",\n\r"):
            raise ValueError(f"invalid mount directory: {source}")
        argv.extend(["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")])

    mount(repo, "/harness", True)
    mount(worktree, "/src")
    mount(out_dir, "/out")
    mount(cache, "/cache/triton")
    environment = {"HOME": "/tmp/home", "PYTHONPATH": "/harness", "PYTHONDONTWRITEBYTECODE": "1",
                   "TRITON_CACHE_DIR": "/cache/triton", "HF_HOME": "/models", "HF_HUB_OFFLINE": "1",
                   "TF_XPU_IMAGE_ID": image}
    if model_cache is not None:
        mount(model_cache, "/models", True)
    if native_ext is not None:
        mount(native_ext, "/opt/tf-ext", True)
        environment["TF_XPU_EXT_DIR"] = "/opt/tf-ext"
    environment.update(knob_env(knobs))
    for key, value in environment.items():
        argv.extend(["--env", f"{key}={value}"])
    argv.extend([image, "-m", "tools.xpu.container", mode])
    refuse_shell_meta(argv)
    return argv


def runtime_versions() -> dict[str, str | None]:
    """Userspace versions come from the runtime that executes the suites."""
    import torch
    import triton

    versions = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                "torch_xpu": str(torch.version.xpu), "triton": str(triton.__version__),
                "image": os.environ.get("TF_XPU_IMAGE_ID")}
    for key, package in {"compute_runtime": "libze-intel-gpu1", "level_zero_gpu": "libze-intel-gpu1",
                         "ocloc": "intel-ocloc",
                         "igc": "intel-igc-core-2", "level_zero_loader": "libze1"}.items():
        completed = subprocess.run(["dpkg-query", "-W", "-f=${Version}", package],
                                   text=True, capture_output=True, check=False)
        versions[key] = completed.stdout.strip() if completed.returncode == 0 else None
    return versions


def main() -> int:
    """The trusted container worker emits the runner's JSON protocol on stdout."""
    if sys.argv[1:] == ["versions"]:
        json.dump(runtime_versions(), sys.stdout)
        return 0
    if sys.argv[1:] != ["suite"]:
        return 2
    from .suites import run_suite

    request = json.load(sys.stdin)
    Path("/tmp/home").mkdir(exist_ok=True)
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/src", "--no-deps", "--no-build-isolation",
         "--constraint", "/opt/constraints.txt"], text=True, capture_output=True, check=False,
    )
    if installed.returncode:
        body = {"status": "error", "detail": "pip install failed", "payload": {},
                "log": installed.stdout + installed.stderr}
    else:
        result = run_suite(request["name"], worktree="/src", out_dir="/out",
                           timeout_s=max(1, int(float(request["timeout_s"]))), model_cache="/models")
        body = {"status": result.status, "detail": result.detail, "payload": result.payload,
                "log": installed.stdout + installed.stderr + result.log}
    json.dump(body, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
