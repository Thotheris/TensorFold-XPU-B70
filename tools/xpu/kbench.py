"""Kernel timings and hardware counters are recorded against the B70 peaks."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .pins import PEAK_BF16_TFLOPS, PEAK_GBPS


def percentiles_us(samples_us: list[float]) -> dict[str, float]:
    """Median, p10, and p90 use nearest-rank selection."""
    if not samples_us:
        raise ValueError("timing samples must not be empty")
    samples = sorted(samples_us)
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("timing samples must be finite and nonnegative")
    return {
        name: samples[max(0, math.ceil(percentile * len(samples)) - 1)]
        for name, percentile in (("median_us", 0.5), ("p10_us", 0.1), ("p90_us", 0.9))
    }


def rates(*, median_us: float, nbytes: int, flops: float) -> dict[str, float]:
    """Throughput and peak percentages use decimal GB and tera operations."""
    if not math.isfinite(median_us) or median_us <= 0:
        raise ValueError("median_us must be finite and positive")
    if nbytes < 0 or flops < 0 or not math.isfinite(flops):
        raise ValueError("nbytes and flops must be nonnegative and finite")
    seconds = median_us / 1e6
    gbps = nbytes / 1e9 / seconds
    tflops = flops / 1e12 / seconds
    return {
        "gbps": gbps,
        "tflops": tflops,
        "pct_peak_gbps": 100 * gbps / PEAK_GBPS,
        "pct_peak_tflops": 100 * tflops / PEAK_BF16_TFLOPS,
    }


def locates_dpas(ir_text: str) -> bool:
    """The Intel Triton DPAS encoding is present only when its marker appears."""
    return "#triton_intel_gpu.dpas" in ir_text or "#ttig.dpas" in ir_text


def _get(obj: object, name: str) -> Any:
    try:
        return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - compiled-kernel attributes vary by backend.
        return None


def _counter(kernel: object, metadata: object, name: str) -> int | float | None:
    for obj in (kernel, metadata):
        value = _get(obj, name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return value
    return None


def triton_kernel_stats(kernel: object) -> dict[str, Any]:
    """Unavailable or unreadable compiled-kernel statistics remain null."""
    result = {"n_regs": None, "n_spills": None, "threads_per_warp": None, "dpas": None}
    try:
        metadata = _get(kernel, "metadata")
        for key in ("n_regs", "n_spills", "threads_per_warp"):
            result[key] = _counter(kernel, metadata, key)
        if result["threads_per_warp"] is None:
            result["threads_per_warp"] = _counter(kernel, metadata, "warp_size")
        num_warps = _counter(kernel, metadata, "num_warps")
        if num_warps is not None:
            result["num_warps"] = num_warps
        texts = []
        asm = _get(kernel, "asm")
        if isinstance(asm, Mapping):
            try:
                texts.extend(value for value in asm.values() if isinstance(value, str))
            except Exception:  # noqa: BLE001, S110 - unreadable IR is reported as null.
                pass
        ttgir = _get(kernel, "ttgir")
        if isinstance(ttgir, str):
            texts.append(ttgir)
        if texts:
            result["dpas"] = any(locates_dpas(text) for text in texts)
    except Exception:  # noqa: BLE001, S110 - missing metadata is reported as null.
        pass
    return result


class ManualTimer:
    """A manual timer uses perf_counter or scripted elapsed milliseconds."""

    def __init__(self, samples_ms: list[float] | None = None) -> None:
        self._samples = iter(samples_ms) if samples_ms is not None else None
        self._start: float | None = None

    def start(self) -> None:
        """The elapsed span starts at the current monotonic counter."""
        self._start = time.perf_counter()

    def stop_ms(self) -> float:
        """The elapsed span is returned in milliseconds."""
        if self._start is None:
            raise RuntimeError("timer has not started")
        elapsed = (time.perf_counter() - self._start) * 1e3
        self._start = None
        return float(next(self._samples)) if self._samples is not None else elapsed


class _XPUTimer:
    def __init__(self) -> None:
        import torch

        self._xpu = torch.xpu
        self._start = self._xpu.Event(enable_timing=True)
        self._stop = self._xpu.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop_ms(self) -> float:
        self._stop.record()
        self._xpu.synchronize()
        return float(self._start.elapsed_time(self._stop))


def bench(
    *,
    fn: Callable[[], Any],
    nbytes: int,
    flops: float,
    name: str,
    out_dir: str | Path,
    warmup: int = 5,
    repeats: int = 20,
    timer: Any = None,
    triton_kernel: object = None,
    bitwise_ok: bool | None = None,
    sync: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Warmup calls are untimed and each kernel result is written as JSON."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    timer = timer if timer is not None else _XPUTimer()
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    samples = []
    for _ in range(repeats):
        timer.start()
        fn()
        if sync is not None:
            sync()
        samples.append(timer.stop_ms() * 1e3)
    percentiles = percentiles_us(samples)
    payload = {
        "kind": "kernel",
        "name": name,
        **percentiles,
        **rates(median_us=percentiles["median_us"], nbytes=nbytes, flops=flops),
        **triton_kernel_stats(triton_kernel),
        "bitwise_ok": bitwise_ok,
    }
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", name.replace("/", "_").replace("\\", "_")).strip(".")
    if not filename:
        raise ValueError("kernel name must contain a safe filename character")
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{filename}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload
