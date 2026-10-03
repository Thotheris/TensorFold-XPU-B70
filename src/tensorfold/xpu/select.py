"""Which implementation each XPU op runs: Triton unless TF_XPU_KERNEL_<OP>=native picks a promoted native kernel."""

from __future__ import annotations

import importlib
import os
from typing import Any

__all__ = ["CHOICES", "IMPLEMENTATIONS", "choice", "module"]

CHOICES = ("triton", "native")
# op -> {choice: module}; a native entry appears only once it passes every invariance test and the promotion rule
IMPLEMENTATIONS: dict[str, dict[str, str]] = {
    "gdn": {"triton": "tensorfold.xpu.kernels.gdn"},
    "qmm": {"triton": "tensorfold.xpu.kernels.qmm"},
}


def choice(op: str) -> str:
    """The implementation ``op`` runs: the TF_XPU_KERNEL_<OP> override, else Triton."""

    if op not in IMPLEMENTATIONS:
        raise ValueError(f"unknown XPU op {op!r}")
    value = os.environ.get(f"TF_XPU_KERNEL_{op.upper()}", "triton").strip().lower()
    if value not in CHOICES:
        raise ValueError(f"TF_XPU_KERNEL_{op.upper()} must be triton or native, not {value!r}")
    if value not in IMPLEMENTATIONS[op]:
        raise ValueError(f"no {value} XPU kernel for {op} yet")
    return value


def module(op: str) -> Any:
    """The module implementing ``op`` under the current choice."""

    return importlib.import_module(IMPLEMENTATIONS[op][choice(op)])
