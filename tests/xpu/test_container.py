"""Container argv, metadata and timeout behavior are testable without Docker or a GPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.xpu import b70_runner as runner
from tools.xpu import container
from tools.xpu.container_probes import validate_manifest
from tools.xpu.schema_check import validate_document
from tools.xpu.suites import run_suite

IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def config(tmp_path):
    directories = {}
    for key in ("repo", "worktree", "out_dir", "cache", "model_cache", "native_ext"):
        directories[key] = tmp_path / key
        directories[key].mkdir()
    return {**directories, "image": IMAGE, "name": "tf-abcdef0-env", "gids": [109, 110],
            "knobs": {"IGC_Test": "1", "HF_TOKEN": "secret", "PATH": "/opt/oneapi", "IGC_SECRET": "secret"},
            "uid": 1000, "gid": 1000}


def test_argv_is_offline_and_mounts_only_requested_paths(config):
    argv = container.docker_argv(**config)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--user") + 1] == "1000:1000"
    assert argv[argv.index("--device") + 1] == "/dev/dri"
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == "--group-add"] == ["109", "110"]
    mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--mount"]
    assert any("dst=/harness,readonly" in mount for mount in mounts)
    assert any("dst=/src" in mount and "readonly" not in mount for mount in mounts)
    assert any("dst=/models,readonly" in mount for mount in mounts)
    assert any("dst=/opt/tf-ext,readonly" in mount for mount in mounts)
    env = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--env"]
    assert "IGC_Test=1" in env and "PYTHONPATH=/harness" in env
    assert "HF_HUB_OFFLINE=1" in env and "HOME=/tmp/home" in env
    assert not any("secret" in value or "oneapi" in value for value in env)
    assert argv[-4:] == [IMAGE, "-m", "tools.xpu.container", "suite"]
    runner.refuse_shell_meta(argv)


def test_optional_mounts_and_bad_inputs(config, tmp_path):
    argv = container.docker_argv(**{**config, "model_cache": None, "native_ext": None})
    assert not any("dst=/models" in value or "dst=/opt/tf-ext" in value for value in argv)
    for bad in ("--privileged", "tag", "sha256:abc"):
        with pytest.raises(ValueError):
            container.docker_argv(**{**config, "image": bad})
    with pytest.raises(ValueError):
        container.docker_argv(**{**config, "repo": tmp_path / "missing"})
    bad_path = tmp_path / "with,comma"
    bad_path.mkdir()
    with pytest.raises(ValueError):
        container.docker_argv(**{**config, "repo": bad_path})
    with pytest.raises(ValueError):
        container.container_name("abc", "env")
    assert container.container_name("b" * 40, "kernels:gdn") == "tf-bbbbbbb-kernels-gdn"


def test_image_is_local_and_immutable(monkeypatch, tmp_path):
    seen = []

    def command(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, IMAGE + "\n", "")

    monkeypatch.setattr(runner, "run_cmd", command)
    assert container.image_id("tensorfold-xpu:tc-test", repo=tmp_path) == IMAGE
    assert seen == [["docker", "image", "inspect", "--format", "{{.Id}}", "tensorfold-xpu:tc-test"]]
    with pytest.raises(ValueError):
        container.image_id("--privileged", repo=tmp_path)


def test_suite_timeout_kills_container(config, monkeypatch):
    killed = []

    def spawn(argv, **kwargs):
        assert argv[0] == "docker"
        assert json.loads(kwargs["stdin"])["name"] == "env"
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(runner, "_spawn_suite", spawn)
    monkeypatch.setattr(runner, "run_cmd", lambda argv, **kwargs: killed.append(argv))
    result = runner.invoke_suite("env", repo=config["repo"], worktree=config["worktree"],
                                 out_dir=config["out_dir"], timeout_s=1, model_cache=None, container=config)
    assert result.status == "fail"
    assert killed == [["docker", "kill", "tf-abcdef0-env"]]


def test_versions_split_and_image_changes_hash(config, monkeypatch):
    commands = []

    def command(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "host", "")

    def spawn(argv, **kwargs):
        assert argv[-1] == "versions"
        return subprocess.CompletedProcess(argv, 0, json.dumps({"image": IMAGE, "python": "3.14",
            "torch": "2.14.1+xpu", "torch_xpu": "20260100", "triton": "3.8.0",
            "compute_runtime": "26.31.39395.13-0"}), "")

    monkeypatch.setattr(runner, "run_cmd", command)
    monkeypatch.setattr(runner, "_spawn_suite", spawn)
    versions, probes = runner._collect_versions(config["worktree"], Path("/missing"), [], container=config)
    assert versions["kernel"] == "host" and probes["device"] == "host"
    assert versions["image"] == IMAGE and versions["python"] == "3.14"
    assert not any(argv[0] == "dpkg-query" for argv in commands)
    assert runner.toolchain_hash(versions) != runner.toolchain_hash({**versions, "image": "different"})
    env = runner.merge_env(versions=versions, probes=probes, models={}, fingerprint_hash="test")
    assert env["image"] == IMAGE and validate_document(env) == []
    env["versions"]["image"] = 12
    assert validate_document(env)


def test_native_manifest_must_match(tmp_path):
    assert not validate_manifest(tmp_path, torch="2.14.1+xpu", sycl="20260100")["ok"]
    manifest = {"torch": "2.14.1+xpu", "sycl": "20260100", "dle": "2026.1"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert validate_manifest(tmp_path, torch="2.14.1+xpu", sycl="20260100")["ok"]
    assert not validate_manifest(tmp_path, torch="2.14.2+xpu", sycl="20260100")["ok"]
    assert not validate_manifest(tmp_path, torch="2.14.1+xpu", sycl="20250100")["ok"]


def test_unit_host_executes_pytest(monkeypatch, tmp_path):
    seen = []

    def command(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 1, "failed", "")

    monkeypatch.setattr(subprocess, "run", command)
    result = run_suite("unit-host", worktree=tmp_path, out_dir=tmp_path, timeout_s=30)
    assert result.status == "fail" and (tmp_path / "unit-host.txt").read_text() == "failed"
    assert "--host-only" in seen[0][0]
    assert seen[0][1]["cwd"] == tmp_path and seen[0][1]["timeout"] == 30


def test_env_image_and_one_cycle_fallback_parser(monkeypatch):
    monkeypatch.setenv("TF_XPU_IMAGE", "toolchain:test")
    monkeypatch.delenv("TF_XPU_MODE", raising=False)
    args = runner._parser().parse_args(["--once"])
    assert args.image == "toolchain:test" and args.mode is None
    monkeypatch.setenv("TF_XPU_MODE", "venv")
    assert runner._parser().parse_args(["--once"]).mode == "venv"
    assert runner._parser().parse_args(["--once", "--mode", "container"]).mode == "container"


def test_container_requirement_probes_record_variants(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tools.xpu import container_probes as probes

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.1+xpu",
                                                            version=SimpleNamespace(xpu="20260100")))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.8.0"))
    monkeypatch.setenv("TF_XPU_IMAGE_ID", IMAGE)
    monkeypatch.delenv("TF_XPU_EXT_DIR", raising=False)
    seen = []

    def launch(env):
        seen.append(env)
        assert not Path(env["TRITON_CACHE_DIR"]).exists()
        return {"ok": env["CC"] == "/usr/bin/gcc"}

    monkeypatch.setattr(probes, "_launcher_probe", launch)
    results = probes.container_probes()
    assert results["ok"] and not results["gcc_without_python_headers"]["ok"]
    assert results["level_zero_only"]["ok"]
    assert seen[2]["ONEAPI_DEVICE_SELECTOR"] == "level_zero:gpu"
    assert len({env["TRITON_CACHE_DIR"] for env in seen}) == 3
    monkeypatch.setenv("TF_XPU_EXT_DIR", str(tmp_path))
    assert not probes.container_probes()["ok"]


@pytest.mark.parametrize("mode", ["container", "venv"])
def test_runner_image_selects_container_without_host_install(mode, tmp_path, monkeypatch):
    sha = "c" * 40
    spec = "suites: [env, unit-host]\n"
    repo = tmp_path / "repo"
    repo.mkdir()
    work = tmp_path / "work"
    results = tmp_path / "results"
    seen = {"pip": [], "suites": [], "push": []}
    monkeypatch.setenv("TF_XPU_IMAGE", "toolchain:test")
    monkeypatch.setenv("TF_XPU_MODE", mode)
    monkeypatch.setattr(runner, "_fetch", lambda *args: False)
    monkeypatch.setattr(runner, "_index_text", lambda *args: "")
    monkeypatch.setattr(runner, "_heads", lambda *args: [runner.Head("xpu/main", sha, 1)])
    monkeypatch.setattr(container, "image_id", lambda *args, **kwargs: IMAGE)
    monkeypatch.setattr(container, "render_gids", lambda: [109])
    monkeypatch.setattr(runner, "check_gpu", lambda *args: type("Health", (), {"ok": True})())
    monkeypatch.setattr(runner, "ensure_results_worktree", lambda *args: results.mkdir())
    monkeypatch.setattr(runner, "push_results", lambda *args: seen["push"].append(args))

    def git(args, **kwargs):
        if args[:3] == ["worktree", "add", "--detach"]:
            branch = Path(args[3])
            (branch / ".b70").mkdir(parents=True)
            (branch / ".b70/run.yml").write_text(spec)
        return subprocess.CompletedProcess(args, 0, spec if args[0] == "show" else "", "")

    def command(argv, **kwargs):
        assert "pip" in argv
        seen["pip"].append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def versions(*args, **kwargs):
        assert bool(kwargs.get("container")) == (mode == "container")
        return ({"image": IMAGE} if mode == "container" else {}), {}

    def suite(name, **kwargs):
        assert bool(kwargs.get("container")) == (mode == "container")
        seen["suites"].append(name)
        return runner.SuiteResult(name, "pass", "", {}, "probe  \n\t\n")

    monkeypatch.setattr(runner, "run_git", git)
    monkeypatch.setattr(runner, "run_cmd", command)
    monkeypatch.setattr(runner, "_collect_versions", versions)
    monkeypatch.setattr(runner, "invoke_suite", suite)
    venv = tmp_path / "venv"
    if mode == "venv":
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python").touch()
    assert runner.main(["--once", "--repo", str(repo), "--work-dir", str(work),
                        "--results-dir", str(results), "--state-dir", str(tmp_path / "state"),
                        "--venv", str(venv), "--triton-cache", str(tmp_path / "cache")]) == 0
    assert bool(seen["pip"]) == (mode == "venv")
    assert seen["suites"] == ["env", "unit-host"] and seen["push"]
    env_doc = json.loads(next(results.glob("runs/*/*/env.json")).read_text())
    assert env_doc["image"] == (IMAGE if mode == "container" else None)
    assert next(results.glob("runs/*/*/logs/runner.log")).read_text() == "probe\n\n\nprobe\n"


def test_host_only_keeps_mlx_sources_and_filters_test_dependencies(tmp_path):
    from tools.xpu.host_tests import mlx_modules

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "mlx_helper.py").write_text('def forward():\n    import mlx.core as mx\n')
    (tests / "test_lane.py").write_text('from tests.mlx_helper import forward\n')
    (tests / "test_shared.py").write_text('from tests.test_lane import forward\n')
    (tests / "test_cli.py").write_text('from tensorfold import cli\n')
    (tests / "test_mlx_cli.py").write_text('from tensorfold import cli\ncli._serve_mlx(None)\n')
    excluded = mlx_modules(tests)
    assert {path.name for path in excluded} == {"mlx_helper.py", "test_lane.py", "test_shared.py", "test_mlx_cli.py"}
    assert all(path.exists() for path in excluded)


def test_persistent_venv_fallback_is_consumed_once(tmp_path):
    assert not runner._container_mode("image:test", "venv", tmp_path, "a" * 40)
    assert runner._container_mode("image:test", "venv", tmp_path, "b" * 40)
    assert runner._container_mode("image:test", None, tmp_path, "c" * 40)
    assert not runner._container_mode("image:test", "venv", tmp_path, "d" * 40)
    assert not runner._container_mode(None, None, tmp_path, "e" * 40)


def test_suite_python_uses_writable_venv_and_image_packages(tmp_path, monkeypatch):
    import sys
    import sysconfig

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    runtime = tmp_path / "runtime"
    python = container._suite_python(runtime)
    assert python == runtime / "bin/python"
    assert calls[0][0] == [sys.executable, "-m", "venv", "--without-pip", str(runtime)]
    assert calls[0][1]["check"] is True
    assert next(runtime.glob("lib/python*/site-packages/image-runtime.pth")).read_text().strip() == (
        sysconfig.get_path("purelib")
    )
    monkeypatch.setattr(sys, "prefix", str(runtime))
    assert container._suite_python(runtime) == Path(sys.executable)
    assert len(calls) == 1


def test_result_logs_strip_traceback_whitespace():
    from tools.xpu.b70_runner import _log_text

    assert _log_text("traceback  \n  failure\t\n\n") == "traceback\n  failure\n"
