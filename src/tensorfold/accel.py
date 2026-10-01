"""One device API over torch.cuda and torch.xpu: the B70 backend calls these where the CUDA code calls torch.cuda."""

from __future__ import annotations


def _torch(torch=None):
    if torch is not None:
        return torch
    try:
        import torch
    except ImportError:
        return None
    return torch


def _up(ns) -> bool:
    try:
        return bool(ns.is_available())
    except Exception:
        return False


def device_type(torch=None) -> str:
    """``"xpu"`` when only XPU is available; ``"cuda"`` otherwise, so CPU-only hosts and fakes keep the CUDA path."""

    t = _torch(torch)
    if t is None:
        return "cuda"
    return "xpu" if _up(getattr(t, "xpu", None)) and not _up(getattr(t, "cuda", None)) else "cuda"


def is_available(kind: str = "cuda", torch=None) -> bool:
    """Whether torch is importable and reports ``kind`` ("cuda" or "xpu") available."""

    t = _torch(torch)
    return t is not None and _up(getattr(t, kind, None))


def _split(device, torch=None) -> tuple[str, int | None]:
    if device is None:
        return device_type(torch), None
    if isinstance(device, str):
        kind, _, idx = device.partition(":")
        index = int(idx) if idx.isdigit() else None
    else:
        kind, index = getattr(device, "type", None), getattr(device, "index", None)
    if kind not in ("cuda", "xpu"):
        raise ValueError(f"unsupported device {device!r}: expected cuda or xpu")
    return kind, index


def api(device=None, torch=None):
    """``torch.cuda`` or ``torch.xpu`` for ``device`` (a string, a torch.device or None for the available one)."""

    kind, _ = _split(device, torch)
    t = _torch(torch)
    if t is None:
        raise ImportError("torch is not installed")
    return getattr(t, kind)


def _call(name: str, device=None, *args, **kwargs):
    fn = getattr(api(device), name)
    return fn(*args, **kwargs) if device is None else fn(device, *args, **kwargs)


def synchronize(device=None) -> None:
    _call("synchronize", device)


def empty_cache(device=None) -> None:
    api(device).empty_cache()


def Event(device=None, **kwargs):
    return api(device).Event(**kwargs)


def Stream(device=None, **kwargs):
    return api(device).Stream(**kwargs) if device is None else api(device).Stream(device, **kwargs)


def stream(s):
    """Context manager making ``s`` (a torch.cuda or torch.xpu stream) current."""

    return api(s.device).stream(s)


def device_guard(device):
    """Context manager making ``device`` current."""

    return api(device).device(device)


def current_stream(device=None):
    return _call("current_stream", device)


def set_device(device) -> None:
    api(device).set_device(device)


def mem_get_info(device=None) -> tuple[int, int]:
    return _call("mem_get_info", device)


def memory_allocated(device=None) -> int:
    return _call("memory_allocated", device)


def memory_reserved(device=None) -> int:
    return _call("memory_reserved", device)


def name(device=None) -> str:
    return _call("get_device_name", device)


def is_discrete(device=None) -> bool:
    """False for an integrated (unified-memory) GPU; the B70 reports no integrated flag, which means discrete."""

    ns, (_, index) = api(device), _split(device)
    try:
        props = ns.get_device_properties(index or 0 if device is not None else ns.current_device())
    except Exception:  # [UNVERIFIED] the xpu property object's fields; an unreadable one is treated as discrete
        return True
    return not getattr(props, "is_integrated", False)


def pinnable() -> bool:
    """Whether host pinned staging is worth asking for: a CUDA or XPU device is present."""

    return is_available("cuda") or is_available("xpu")


def graphs_supported(device=None) -> bool:
    """CUDA graphs only: XPU graphs stay off until a graphs == eager bitwise test passes on the B70."""

    return _split(device)[0] == "cuda"


def set_allocator_settings(settings: str, device=None) -> None:
    if _split(device)[0] == "cuda":
        api(device).memory._set_allocator_settings(settings)


def host_empty_cache(device=None) -> None:
    if _split(device)[0] == "cuda":
        import torch

        getattr(torch._C, "_host_emptyCache", lambda: None)()


__all__ = [
    "Event", "Stream", "api", "current_stream", "device_guard", "device_type", "empty_cache", "graphs_supported",
    "host_empty_cache", "is_available", "is_discrete", "mem_get_info", "memory_allocated", "memory_reserved", "name",
    "pinnable", "set_allocator_settings", "set_device", "stream", "synchronize",
]
