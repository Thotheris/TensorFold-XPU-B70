"""Device selection keeps host tests on CPU and prefers CUDA when both GPUs exist."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.devices import select_device


@pytest.mark.parametrize('cuda,xpu,expected', [(True, True, 'cuda'), (False, True, 'xpu'), (False, False, None)])
def test_auto_device_prefers_cuda(monkeypatch, cuda, xpu, expected):
    monkeypatch.delenv('TF_TEST_DEVICE', raising=False)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                            xpu=SimpleNamespace(is_available=lambda: xpu))
    assert select_device(torch) == expected


@pytest.mark.parametrize('device,expected', [('cuda', 'cuda'), ('xpu', 'xpu'), ('cpu', None)])
def test_explicit_device_and_host_suite(monkeypatch, device, expected):
    monkeypatch.setenv('TF_TEST_DEVICE', device)
    assert select_device(object()) == expected
