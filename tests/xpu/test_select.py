"""XPU ops run Triton by default; an override must name an implementation that exists."""

from __future__ import annotations

import pytest

from tensorfold.xpu import select


def test_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("TF_XPU_KERNEL_GDN", raising=False)
    assert select.choice("gdn") == "triton"
    monkeypatch.setenv("TF_XPU_KERNEL_GDN", "Triton")
    assert select.choice("gdn") == "triton"
    monkeypatch.setenv("TF_XPU_KERNEL_GDN", "native")
    with pytest.raises(ValueError, match="no native"):
        select.choice("gdn")
    monkeypatch.setenv("TF_XPU_KERNEL_GDN", "cuda")
    with pytest.raises(ValueError, match="triton or native"):
        select.choice("gdn")
    with pytest.raises(ValueError, match="unknown"):
        select.choice("attention")
