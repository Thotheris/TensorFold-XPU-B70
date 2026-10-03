"""The smoke ladder preserves pointer bits and reports failures without a GPU."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

from tools.xpu.triton_smoke import _numeric, _s64, run_probes


def test_unsigned_addresses_survive_signed_tables():
    for pointer in (0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1):
        signed = _s64(pointer)
        assert -(1 << 63) <= signed < 1 << 63
        assert signed & ((1 << 64) - 1) == pointer


def test_special_values_have_strict_json_representation():
    values = _numeric([0.0, float("inf"), float("-inf"), float("nan")])
    assert values == [0.0, "inf", "-inf", "nan"]
    json.dumps(values, allow_nan=False)


def test_unavailable_device_stops_before_kernel_definition(tmp_path, monkeypatch):
    torch = ModuleType("torch")
    torch.__version__ = "test"
    torch.xpu = SimpleNamespace(is_available=lambda: False)
    triton = ModuleType("triton")
    triton.__version__ = "test"
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    result = run_probes(tmp_path)
    assert result["ok"] is False
    assert result["stopped_at"] == "available"
    assert result["probes"] == {"available": {"ok": False}}
    assert json.loads((tmp_path / "triton-smoke.json").read_text()) == result


def test_missing_runtime_import_leaves_failure_artifact(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    result = run_probes(tmp_path)
    assert not result["ok"]
    assert result["stopped_at"] == "imports"
    assert result["probes"]["imports"]["error"]
    assert json.loads((tmp_path / "triton-smoke.json").read_text()) == result
