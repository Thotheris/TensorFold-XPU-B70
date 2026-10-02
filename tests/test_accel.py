"""accel routes to torch.cuda or torch.xpu, imports torch lazily and keeps the CUDA path on hosts with neither."""

import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from tensorfold import accel


class Ns:
    """A fake torch.cuda / torch.xpu that records calls."""

    def __init__(self, tag, up=True, integrated=None):
        self.tag, self.up, self.calls = tag, up, []
        self.memory = NS(_set_allocator_settings=lambda s: self.calls.append(("alloc", s)))
        self.props = NS() if integrated is None else NS(is_integrated=integrated)

    def is_available(self):
        return self.up

    def mem_get_info(self, *a):
        self.calls.append(("mem", a))
        return (1, 2)

    def synchronize(self, *a):
        self.calls.append(("sync", a))

    def current_device(self):
        return 0

    def get_device_properties(self, i):
        self.calls.append(("props", i))
        return self.props

    def Event(self, **kw):
        return (self.tag, "event", kw)

    def Stream(self, *a, **kw):
        return (self.tag, "stream", a, kw)


def fake(cuda=False, xpu=False, **kw):
    return NS(cuda=Ns("cuda", cuda, **kw), xpu=Ns("xpu", xpu, **kw))


def test_device_type_rules():
    assert accel.device_type(fake(cuda=True)) == "cuda"
    assert accel.device_type(fake(xpu=True)) == "xpu"
    assert accel.device_type(fake(cuda=True, xpu=True)) == "cuda"
    assert accel.device_type(fake()) == "cuda"


def test_device_type_tolerates_odd_fakes():
    assert accel.device_type(NS(cuda=NS(is_available=lambda: True))) == "cuda"
    assert accel.device_type(NS(cuda=NS())) == "cuda"
    assert accel.device_type(NS()) == "cuda"

    def boom():
        raise RuntimeError("no driver")

    assert accel.device_type(NS(cuda=NS(is_available=boom), xpu=NS(is_available=lambda: True))) == "xpu"
    assert accel.device_type(NS(cuda=NS(is_available=lambda: True), xpu=NS(is_available=boom))) == "cuda"


def test_is_available():
    t = fake(xpu=True)
    assert accel.is_available("xpu", t) and not accel.is_available("cuda", t)
    assert not accel.is_available("xpu", NS())


def test_api_routing():
    t = fake(cuda=True, xpu=True)
    assert accel.api("cuda", t) is t.cuda and accel.api("cuda:1", t) is t.cuda
    assert accel.api("xpu", t) is t.xpu and accel.api("xpu:0", t) is t.xpu
    assert accel.api(NS(type="xpu", index=1), t) is t.xpu
    assert accel.api(None, t) is t.cuda
    for bad in ("cpu", "mps", NS(type="cpu", index=None)):
        with pytest.raises(ValueError, match="cpu|mps"):
            accel.api(bad, t)


def test_api_with_real_torch_devices():
    torch = pytest.importorskip("torch")
    assert accel.api(torch.device("cuda", 0)) is torch.cuda
    assert accel.api("xpu") is torch.xpu
    with pytest.raises(ValueError):
        accel.api(torch.device("cpu"))


def test_graphs_supported():
    assert accel.graphs_supported("cuda") and not accel.graphs_supported("xpu:0")


def test_allocator_and_host_cache(monkeypatch):
    t = fake(cuda=True, xpu=True)
    monkeypatch.setitem(sys.modules, "torch", t)
    accel.set_allocator_settings("expandable_segments:True", "xpu")
    accel.host_empty_cache("xpu")
    assert t.xpu.calls == []
    accel.set_allocator_settings("expandable_segments:True", "cuda")
    assert t.cuda.calls == [("alloc", "expandable_segments:True")]
    t._C = NS(_host_emptyCache=lambda: t.cuda.calls.append("host"))
    accel.host_empty_cache("cuda")
    assert "host" in t.cuda.calls


def test_host_empty_cache_missing_hook(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", NS(_C=NS()))
    accel.host_empty_cache("cuda")


def test_mem_get_info_routes(monkeypatch):
    t = fake(cuda=True, xpu=True)
    monkeypatch.setitem(sys.modules, "torch", t)
    assert accel.mem_get_info("xpu:0") == (1, 2)
    assert t.xpu.calls == [("mem", ("xpu:0",))]
    assert accel.mem_get_info("cuda") == (1, 2)
    assert t.cuda.calls == [("mem", ("cuda",))]
    t.cuda.calls.clear()
    assert accel.mem_get_info() == (1, 2)
    assert t.cuda.calls == [("mem", ())]


def test_event_and_stream_args(monkeypatch):
    t = fake(cuda=True, xpu=True)
    monkeypatch.setitem(sys.modules, "torch", t)
    assert accel.Event("xpu", enable_timing=True) == ("xpu", "event", {"enable_timing": True})
    assert accel.Stream("cuda:0", priority=1) == ("cuda", "stream", ("cuda:0",), {"priority": 1})
    assert accel.Stream(priority=1) == ("cuda", "stream", (), {"priority": 1})


def test_is_discrete(monkeypatch):
    t = fake(cuda=True, xpu=True, integrated=True)
    monkeypatch.setitem(sys.modules, "torch", t)
    assert accel.is_discrete("cuda:1") is False
    assert t.cuda.calls == [("props", 1)]
    t = fake(cuda=True, xpu=True)
    monkeypatch.setitem(sys.modules, "torch", t)
    assert accel.is_discrete("xpu") is True and accel.is_discrete() is True
    t = fake(cuda=True, xpu=True, integrated=False)
    monkeypatch.setitem(sys.modules, "torch", t)
    assert accel.is_discrete("cuda") is True


def test_import_is_lazy():
    code = "import sys, tensorfold.accel, tensorfold.xpu; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
