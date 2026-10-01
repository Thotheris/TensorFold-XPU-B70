"""Host memory shadow probes hold cumulative XPU allocations below the allocation ceiling."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .pins import CHUNK_BYTES, HOST_RAM_WARN_GIB, MAX_ALLOC_BYTES, MEM_AVAILABLE_FLOOR_KIB, SHADOW_SIZES_GIB


def parse_meminfo(text: str) -> dict[str, int]:
    """Well-formed proc meminfo values are returned in KiB."""
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_()]*):\s*([0-9]+)\s*kB\s*", line)
        if match:
            result[match[1]] = int(match[2])
    return result


def mem_available_kib(info: Mapping[str, int]) -> int | None:
    """MemAvailable is unknown when its proc field is absent."""
    return info.get("MemAvailable")


def committed_as_kib(info: Mapping[str, int]) -> int | None:
    """Committed_AS is unknown when its proc field is absent."""
    return info.get("Committed_AS")


def host_ram_gib(info: Mapping[str, int]) -> float | None:
    """MemTotal is converted from KiB to GiB when present."""
    value = info.get("MemTotal")
    return value / 1024**2 if value is not None else None


def swap_gib(info: Mapping[str, int]) -> float | None:
    """SwapTotal is converted from KiB to GiB when present."""
    value = info.get("SwapTotal")
    return value / 1024**2 if value is not None else None


def below_ram_warn(info: Mapping[str, int], warn_gib: int = HOST_RAM_WARN_GIB) -> bool | None:
    """The RAM warning is unknown when MemTotal is absent."""
    value = host_ram_gib(info)
    return value < warn_gib if value is not None else None


def shadow_step(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int | None]:
    """Shadow deltas are after minus before, or unknown when either field is absent."""
    result = {}
    for key, label in (("MemAvailable", "delta_mem_available_kib"), ("Committed_AS", "delta_committed_as_kib")):
        left, right = before.get(key), after.get(key)
        result[label] = right - left if left is not None and right is not None else None
    return result


def chunk_plan(total_gib: int) -> list[int]:
    """Every chunk is strictly smaller than the B70 single-allocation ceiling."""
    if isinstance(total_gib, bool) or not isinstance(total_gib, int) or total_gib < 0:
        raise ValueError("total_gib must be a nonnegative integer")
    if not 0 < CHUNK_BYTES < MAX_ALLOC_BYTES:
        raise ValueError("CHUNK_BYTES must be positive and below MAX_ALLOC_BYTES")
    whole, tail = divmod(total_gib * 1024**3, CHUNK_BYTES)
    return [CHUNK_BYTES] * whole + ([tail] if tail else [])


def run_host_ram_shadow(
    read_meminfo: Callable[[], dict[str, int]],
    allocate: Callable[[int], Any],
    release: Callable[[list[Any]], None],
    *,
    sizes: Sequence[int] = SHADOW_SIZES_GIB,
    floor_kib: int = MEM_AVAILABLE_FLOOR_KIB,
) -> dict[str, Any]:
    """Cumulative shadow steps always release held allocations, including on errors."""
    result = {"baseline": {}, "steps": [], "stopped_early": False, "error": None}
    tokens = []
    held_gib = 0
    try:
        baseline = dict(read_meminfo())
        result["baseline"] = baseline
        current = baseline
        for size in sizes:
            if isinstance(size, bool) or not isinstance(size, int) or size < held_gib:
                raise ValueError("shadow sizes must be nonnegative, nondecreasing integers")
            available = mem_available_kib(current)
            if available is not None and available < floor_kib:
                result["stopped_early"] = True
                break
            for nbytes in chunk_plan(size - held_gib):
                tokens.append(allocate(nbytes))
            held_gib = size
            current = dict(read_meminfo())
            result["steps"].append({
                "size_gib": size,
                "meminfo": current,
                **shadow_step(baseline, current),
            })
    except MemoryError as exc:
        result["stopped_early"] = True
        result["stop_reason"] = f"allocation refused: {exc}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["stopped_early"] = True
    finally:
        try:
            release(tokens)
        except Exception as exc:
            error = f"release: {type(exc).__name__}: {exc}"
            result["error"] = f"{result['error']}; {error}" if result["error"] else error
    return result
