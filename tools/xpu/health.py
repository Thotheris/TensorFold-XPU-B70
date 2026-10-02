"""GPU discovery failures stop the queue until an operator clears STOP."""

from __future__ import annotations

import errno
import inspect
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

STOP_NAME = "STOP"


def stop_path(state_dir: Path) -> Path:
    """The queue stop marker lives in the runner state directory."""
    return state_dir / STOP_NAME


def queue_stopped(state_dir: Path) -> bool:
    """An existing STOP marker disables new runs."""
    return stop_path(state_dir).exists()


def raise_stop(state_dir: Path, reason: str) -> None:
    """GPU failures persist as a single-line stop marker."""
    state_dir.mkdir(parents=True, exist_ok=True)
    stop_path(state_dir).write_text(" ".join(reason.split()) + "\n", encoding="utf-8")


def clear_stop(state_dir: Path) -> None:
    """Operators can explicitly re-enable a stopped queue."""
    stop_path(state_dir).unlink(missing_ok=True)


@dataclass(frozen=True)
class GpuHealth:
    """Discovery identifies a healthy, hung, or wedged GPU."""

    ok: bool
    hung: bool
    wedged: bool
    detail: str


def parse_discovery(text: str, returncode: int, timed_out: bool) -> GpuHealth:
    """A successful discovery table identifies a responsive GPU."""
    if timed_out:
        return GpuHealth(False, True, False, "timeout")
    if returncode != 0:
        return GpuHealth(False, False, True, f"discovery exit {returncode}")
    normalized = re.sub(r"[\s_-]+", " ", text.lower()).strip()
    if not normalized or re.search(r"\bno\s*devices?(?:\s*found)?\b", normalized):
        return GpuHealth(False, False, True, "no device")
    wedge = re.search(r"\b(?:wedged|device\s*lost|i915\s*reset|xe\s*reset|gpu\s*hang|reset in progress)\b", normalized)
    if wedge:
        return GpuHealth(False, False, True, wedge.group(0))
    if re.search(r"\bdevice\s*(?:name|id)\b", normalized):
        return GpuHealth(True, False, False, "discovery healthy")
    return GpuHealth(False, False, True, "missing discovery device table")


def check_gpu(runner: Callable[..., object], *, timeout_s: float = 20.0) -> GpuHealth:
    """Only xpu-smi discovery is used to determine queue health."""
    argv = ["xpu-smi", "discovery"]
    try:
        parameters = inspect.signature(runner).parameters
        if "timeout_s" in parameters:
            completed = runner(argv, timeout_s=timeout_s)
        elif "timeout" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            completed = runner(argv, timeout=timeout_s)
        else:
            completed = runner(argv)
    except subprocess.TimeoutExpired:
        return parse_discovery("", 0, True)
    except FileNotFoundError:
        return GpuHealth(False, False, False, "xpu-smi not installed")
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return GpuHealth(False, False, False, "xpu-smi not installed")
        return GpuHealth(False, False, True, f"discovery unavailable: {exc}")
    except ValueError as exc:
        return GpuHealth(False, False, True, f"discovery unavailable: {exc}")
    text = f"{getattr(completed, 'stdout', '') or ''}\n{getattr(completed, 'stderr', '') or ''}"
    return parse_discovery(text, getattr(completed, "returncode", 1), False)
