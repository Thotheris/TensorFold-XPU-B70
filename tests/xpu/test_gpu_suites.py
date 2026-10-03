"""GPU suite results require executed tests and preserve failed pytest artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tools.xpu import suites


def _junit(argv: list[str], outcomes: tuple[str, ...]) -> Path:
    path = Path(next(arg.partition("=")[2] for arg in argv if arg.startswith("--junitxml=")))
    document = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(document, "testsuite")
    for number, outcome in enumerate(outcomes):
        case = ElementTree.SubElement(suite, "testcase", name=f"probe_{number}")
        if outcome != "passed":
            ElementTree.SubElement(case, outcome)
    ElementTree.ElementTree(document).write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.fixture
def available_xpu(monkeypatch):
    monkeypatch.setattr(suites, "_xpu_available", lambda: True)


def test_unit_xpu_executes_selected_device_and_retains_junit(available_xpu, monkeypatch, tmp_path):
    seen = []

    def command(argv, **kwargs):
        seen.append((argv, kwargs))
        _junit(argv, ("passed", "passed", "skipped"))
        return subprocess.CompletedProcess(argv, 0, "2 passed, 1 skipped\n", "warning\n")

    monkeypatch.setattr(subprocess, "run", command)
    result = suites.run_suite("unit-xpu", worktree=tmp_path, out_dir=tmp_path, timeout_s=31)
    assert result.status == "pass"
    argv, kwargs = seen[0]
    assert argv[1:5] == ["-m", "pytest", "tests/cuda", "-q"]
    assert "--host-only" not in argv
    assert kwargs["env"]["TF_TEST_DEVICE"] == "xpu"
    assert kwargs["cwd"] == tmp_path and kwargs["timeout"] == 31
    assert result.payload["pytest"] == {"tests": 3, "passed": 2, "failed": 0, "errors": 0, "skipped": 1}
    assert (tmp_path / "unit-xpu.xml").is_file()
    assert (tmp_path / "unit-xpu.txt").read_text() == result.log
    assert "warning" in result.log


def test_requested_xpu_unavailable_fails_before_pytest(monkeypatch, tmp_path):
    monkeypatch.setattr(suites, "_xpu_available", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("pytest must not turn unavailable requested XPU into skipped success")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = suites.run_suite("unit-xpu", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "fail"
    assert "xpu" in result.detail.lower()


@pytest.mark.parametrize("outcomes", [(), ("skipped",), ("skipped", "skipped")])
def test_no_executed_gpu_checks_is_not_green(available_xpu, monkeypatch, tmp_path, outcomes):
    def command(argv, **kwargs):
        _junit(argv, outcomes)
        return subprocess.CompletedProcess(argv, 0, "all checks skipped\n", "")

    monkeypatch.setattr(subprocess, "run", command)
    result = suites.run_suite("unit-xpu", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "fail"
    assert result.payload["pytest"]["passed"] == 0


def test_missing_junit_is_not_green(available_xpu, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "", ""))
    result = suites.run_suite("unit-xpu", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "fail"


@pytest.mark.parametrize("outcome", ["failure", "error"])
def test_nonzero_pytest_retains_failure_counts(available_xpu, monkeypatch, tmp_path, outcome):
    def command(argv, **kwargs):
        _junit(argv, ("passed", outcome, "skipped"))
        return subprocess.CompletedProcess(argv, 1, "failure traceback\n", "")

    monkeypatch.setattr(subprocess, "run", command)
    result = suites.run_suite("unit-xpu", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "fail"
    counts = result.payload["pytest"]
    assert counts["passed"] == 1 and counts["skipped"] == 1
    assert counts["failed" if outcome == "failure" else "errors"] == 1
    assert "failure traceback" in (tmp_path / "unit-xpu.txt").read_text()


def test_kernel_suite_filters_checks_and_records_payload(available_xpu, monkeypatch, tmp_path):
    monkeypatch.setattr(suites, "_kernel_benchmark", lambda *args, **kwargs: {"kind": "kernel", "name": "glue"})
    def command(argv, **kwargs):
        assert "--xpu-kernel=glue" in argv
        assert kwargs["env"]["TF_TEST_DEVICE"] == "xpu"
        _junit(argv, ("passed",))
        return subprocess.CompletedProcess(argv, 0, "glue checks passed\n", "")

    monkeypatch.setattr(subprocess, "run", command)
    result = suites.run_suite("kernels:glue", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "pass"
    assert result.payload["kind"] == "kernel" and result.payload["name"] == "glue"
    assert result.payload["pytest"]["passed"] == 1
    assert (tmp_path / "kernels--glue.xml").is_file()
    assert (tmp_path / "kernels--glue.txt").is_file()


def test_unknown_kernel_selection_is_not_green(available_xpu, monkeypatch, tmp_path):
    def command(argv, **kwargs):
        assert "--xpu-kernel=not_implemented" in argv
        _junit(argv, ())
        return subprocess.CompletedProcess(argv, 5, "no tests ran\n", "")

    monkeypatch.setattr(subprocess, "run", command)
    result = suites.run_suite("kernels:not_implemented", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "fail"
    assert result.payload["pytest"]["passed"] == 0


def test_pytest_timeout_keeps_partial_output(available_xpu, monkeypatch, tmp_path):
    def command(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 5, output=b"last completed check\n", stderr=b"diagnostic\n")

    monkeypatch.setattr(subprocess, "run", command)
    result = suites.run_suite("unit-xpu", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "error"
    assert "last completed check" in result.log and "diagnostic" in result.log
    assert (tmp_path / "unit-xpu.txt").read_text() == result.log


def test_stale_junit_cannot_make_new_run_green(available_xpu, monkeypatch, tmp_path):
    _junit([f"--junitxml={tmp_path / 'unit-xpu.xml'}"], ("passed",))
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "", ""))
    result = suites.run_suite("unit-xpu", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)
    assert result.status == "fail"


@pytest.mark.parametrize("encoding", ["#triton_intel_gpu.dpas", "#ttig.dpas"])
def test_compiled_kernel_stats_preserve_intel_metadata_aliases(encoding):
    from types import SimpleNamespace

    from tools.xpu.kbench import triton_kernel_stats

    kernel = SimpleNamespace(metadata={"warp_size": 16, "num_warps": 4, "n_spills": 0},
                             asm={"ttgir": f"#mma = {encoding}<{{warpsPerCTA = [4, 1]}}>"})
    stats = triton_kernel_stats(kernel)
    assert stats["threads_per_warp"] == 16 and stats["num_warps"] == 4
    assert stats["dpas"] is True and stats["n_spills"] == 0
    assert stats["n_regs"] is None


def test_explicit_threads_per_warp_takes_precedence_over_alias():
    from types import SimpleNamespace

    from tools.xpu.kbench import triton_kernel_stats

    kernel = SimpleNamespace(metadata={"warp_size": 16, "threads_per_warp": 32}, asm={"ttgir": "plain FMA"})
    assert triton_kernel_stats(kernel)["threads_per_warp"] == 32
    assert triton_kernel_stats(kernel)["dpas"] is False


def _metrics():
    return {"kind": "kernel", "name": "glue", "status": "pass", "median_us": 10.0,
            "p10_us": 9.0, "p90_us": 11.0, "gbps": 30.0, "tflops": 0.1,
            "pct_peak_gbps": 5.0, "pct_peak_tflops": 0.05, "n_regs": None,
            "n_spills": None, "threads_per_warp": 32, "dpas": False, "bitwise_ok": True}


def test_stale_metrics_cannot_make_benchmark_green(monkeypatch, tmp_path):
    import json

    kernels = tmp_path / "kernels"
    kernels.mkdir()
    (kernels / "glue.json").write_text(json.dumps(_metrics()))
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "", ""))
    with pytest.raises((OSError, RuntimeError, ValueError)):
        suites._kernel_benchmark("glue", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)


@pytest.mark.parametrize("mutation", [{"status": "fail"}, {"bitwise_ok": False}, {"name": "other"},
                                      {"median_us": "unknown"}, {"kind": "e2e"}])
def test_failed_or_invalid_measured_metrics_cannot_make_benchmark_green(monkeypatch, tmp_path, mutation):
    import json

    def command(argv, **kwargs):
        kernels = tmp_path / "kernels"
        kernels.mkdir(exist_ok=True)
        (kernels / "glue.json").write_text(json.dumps({**_metrics(), **mutation}))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises((RuntimeError, ValueError)):
        suites._kernel_benchmark("glue", worktree=tmp_path, out_dir=tmp_path, timeout_s=5)


def test_valid_fresh_measured_metrics_are_returned(monkeypatch, tmp_path):
    import json

    def command(argv, **kwargs):
        kernels = tmp_path / "kernels"
        kernels.mkdir(exist_ok=True)
        (kernels / "glue.json").write_text(json.dumps(_metrics()))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", command)
    assert suites._kernel_benchmark("glue", worktree=tmp_path, out_dir=tmp_path, timeout_s=5) == _metrics()
