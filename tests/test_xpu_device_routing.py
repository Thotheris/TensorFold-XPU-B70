"""Shared CUDA modules route device calls through accel: the xpu namespace on an XPU-only torch, torch.cuda otherwise."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold import accel  # noqa: E402
from tensorfold.cuda import capacity, direct_read, memory_gate, sampling  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

GB = 10**9
MEMINFO = "MemTotal: 64000000 kB\nMemFree: 60000000 kB\nMemAvailable: 60000000 kB\n"


def ns(calls: list, tag: str, integrated: bool = False, free: int = 20 * GB, total: int = 32 * GB):
    def mem_get_info(*args):
        calls.append((tag, "mem_get_info", args))
        return free, total

    def props(index):
        calls.append((tag, "props", (index,)))
        return SimpleNamespace(is_integrated=integrated)

    return SimpleNamespace(is_available=lambda: True, mem_get_info=mem_get_info, get_device_properties=props,
                           current_device=lambda: 0,
                           memory_reserved=lambda *a: calls.append((tag, "reserved", a)) or 5,
                           memory_allocated=lambda *a: calls.append((tag, "allocated", a)) or 2)


def xpu_only(calls):
    return SimpleNamespace(xpu=ns(calls, "xpu"), cuda=SimpleNamespace(is_available=lambda: False))


def cuda_only(calls, **kw):
    return SimpleNamespace(cuda=ns(calls, "cuda", **kw))


@pytest.fixture(autouse=True)
def meminfo(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: MEMINFO)


def test_xpu_memory_reads_the_xpu_namespace_and_is_discrete():
    calls: list = []
    fake = xpu_only(calls)
    assert capacity.total_bytes(fake) == 32 * GB
    assert capacity.available_bytes(fake) == 20 * GB - capacity.reserve_bytes(32 * GB)
    assert capacity.unified(fake) is False
    assert capacity.page_room(fake) is None
    assert {c[0] for c in calls} == {"xpu"}


def test_cuda_memory_reads_cuda_exactly_as_before():
    calls: list = []
    fake = cuda_only(calls, integrated=True)
    assert capacity.total_bytes(fake) == 32 * GB
    assert calls == [("cuda", "mem_get_info", ())]
    calls.clear()
    assert capacity.unified(fake) is True
    assert calls == [("cuda", "props", (0,))]
    assert capacity.page_room(fake) == 60000000 * 1024
    calls.clear()
    assert capacity.available_bytes(cuda_only(calls, integrated=True)) == 60000000 * 1024 - capacity.reserve_bytes(
        64000000 * 1024, host=True)
    assert calls[0] == ("cuda", "mem_get_info", ())


def test_torch_live_reads_reserved_and_allocated_of_the_device():
    for fake, tag in ((xpu_only(calls := []), "xpu"), (cuda_only(calls2 := []), "cuda")):
        got = memory_gate.torch_live(fake, lambda t: 100)()
        assert got == 103
        assert [c[:2] for c in (calls if tag == "xpu" else calls2)] == [(tag, "reserved"), (tag, "allocated")]


def test_gather_ints_places_tensors_on_the_given_or_derived_device():
    seen: list = []

    class Torch(SimpleNamespace):
        int64 = "i64"

        def tensor(self, values, dtype, device):
            seen.append(("send", device))
            return SimpleNamespace(values=values)

        def empty(self, shape, dtype, device):
            seen.append(("recv", device))
            return SimpleNamespace(view=lambda w, n: SimpleNamespace(tolist=lambda: [[1, 2]] * w))

    capacity.gather_ints(Torch(cuda=SimpleNamespace()), lambda s, r: None, [1, 2])
    assert seen == [("send", "cuda"), ("recv", "cuda")]
    seen.clear()
    capacity.gather_ints(Torch(xpu=ns([], "xpu"), cuda=SimpleNamespace(is_available=lambda: False)),
                         lambda s, r: None, [1, 2])
    assert seen == [("send", "xpu"), ("recv", "xpu")]
    seen.clear()
    capacity.gather_ints(Torch(cuda=SimpleNamespace()), lambda s, r: None, [1, 2], device="xpu")
    assert seen == [("send", "xpu"), ("recv", "xpu")]


class FakeDevice:
    def __init__(self, type_: str) -> None:
        self.type = type_


class FakeLogits:
    ndim, shape = 2, (1, 4)

    def __init__(self, kind: str) -> None:
        self.device = FakeDevice(kind)

    def argmax(self, dim):
        return torch.tensor([2])


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_sampling_accepts_gpu_logits(kind):
    assert sampling.sample_rows(FakeLogits(kind), [0], None) == [2]
    assert sampling.sample_rows(FakeLogits(kind), [0], Sampling(temperature=0.0, seed=1)) == [2]


def test_sampling_rejects_cpu_logits():
    with pytest.raises(ValueError, match="logits"):
        sampling.sample_rows(torch.zeros(1, 4), [0], None)


def recorder(monkeypatch):
    calls: list = []

    class Handle:
        device = "xpu"

        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append(("record", self.name))

        def synchronize(self):
            calls.append(("sync", self.name))

    monkeypatch.setattr(accel, "Stream", lambda device=None, **kw: calls.append(("Stream", str(device))) or Handle("s"))
    monkeypatch.setattr(accel, "Event", lambda device=None, **kw: calls.append(("Event", str(device))) or Handle("e"))
    monkeypatch.setattr(accel, "current_stream", lambda device=None: calls.append(("current", str(device))))
    monkeypatch.setattr(accel, "host_empty_cache", lambda device=None: calls.append(("host_empty_cache", device)))
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append(("raw_host_empty",)), raising=False)
    return calls, Handle


def test_reader_classifies_xpu_as_gpu_and_frees_staging_through_accel(monkeypatch, tmp_path):
    calls, _ = recorder(monkeypatch)
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(256)) * 4)
    reader = direct_read.Reader()
    hit: list = []
    monkeypatch.setattr(reader, "_to_device", lambda p, o, n, d: hit.append(d) or torch.zeros(n, dtype=torch.uint8))
    monkeypatch.setattr(reader, "direct", True)
    reader.read(path, 0, 16, "xpu")
    assert hit == ["xpu"] and reader.device_type == "xpu"
    reader.staging.append([None, None])
    reader.close()
    assert calls == [("host_empty_cache", "xpu")]


def test_reader_close_defaults_to_cuda_and_cpu_reads_stay_host(monkeypatch, tmp_path):
    calls, _ = recorder(monkeypatch)
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 64)
    reader = direct_read.Reader()
    reader.direct = False
    assert reader.read(path, 0, 8).device.type == "cpu"
    reader.staging.append([None, None])
    reader.close()
    assert calls == [("host_empty_cache", "cuda")]


def test_read_ahead_routes_stream_event_and_close_through_accel(monkeypatch, tmp_path):
    calls, Handle = recorder(monkeypatch)
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(64)))

    class Host:
        def to(self, device, non_blocking=False):
            return torch.zeros(8, dtype=torch.uint8)

    reader = SimpleNamespace(read=lambda *a, **k: Host(), close=lambda: None)
    ahead = direct_read.ReadAhead(reader, threads=1)
    monkeypatch.setattr(accel, "device_guard", lambda device: __import__("contextlib").nullcontext())
    monkeypatch.setattr(accel, "stream", lambda s: __import__("contextlib").nullcontext())
    ahead.queue([("k", path, 0, 8, None)], device="xpu", cut=lambda raw, meta: raw)
    uploaded, tensors = ahead.ahead["k"].result()
    assert ahead.device_type == "xpu" and uploaded.name == "e" and "k" in tensors
    assert [c[0] for c in calls[:2]] == ["Stream", "Event"] and calls[1][1] == "xpu"
    calls.clear()
    ahead.close()
    assert ("raw_host_empty",) not in calls and calls[-1] == ("host_empty_cache", "xpu")


def test_read_ahead_ignores_cpu_device(monkeypatch):
    recorder(monkeypatch)
    ahead = direct_read.ReadAhead(SimpleNamespace(), threads=1)
    ahead.queue([], device="cpu")
    assert ahead.stream is None
    ahead.close()
