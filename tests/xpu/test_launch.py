"""The direct-launch cache: entries hold their function, cached launches pass this call's grid, args and stream."""

from __future__ import annotations

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from tensorfold.xpu.kernels.launch import Launcher, _spec  # noqa: E402
from tools.xpu.kernel_benchmarks import _Capture  # noqa: E402

SIGNATURE = {"x": "*bf16", "n": "i32", "BLOCK": "constexpr"}


class _Src:
    def __init__(self, signature):
        self.signature = signature


class _Compiled:
    """A compiled-kernel fake: ``run`` records the direct launches made with it."""

    def __init__(self, tag, signature=SIGNATURE):
        self.tag, self.src, self.calls, self.inits = tag, _Src(signature), [], 0
        self.function, self.packed_metadata = f"function-{tag}", f"metadata-{tag}"

    def _init_handles(self):
        self.inits += 1

    def run(self, *launch):
        self.calls.append(launch)


class _Jit:
    """A jit-function fake: each ``fn[grid](...)`` is a JIT dispatch that returns its kernel for that specialisation."""

    def __init__(self, tag="jit", signature=SIGNATURE):
        self.tag, self.signature, self.dispatches, self.kernels = tag, signature, [], {}

    def __getitem__(self, grid):
        def launch(*args, **options):
            self.dispatches.append((grid, args, options))
            key = (tuple(_spec(a) for a in args), tuple(options.items()))
            return self.kernels.setdefault(key, _Compiled(f"{self.tag}-{len(self.kernels)}", self.signature))
        return launch


def _launcher(kernel, streams=None):
    launcher = Launcher(kernel)
    streams = streams if streams is not None else ["stream-0"]
    launcher.stream = lambda index: streams[-1]
    launcher.device = lambda: None        # CPU tensors have no device index
    return launcher


def _x(n=64):
    return torch.zeros(n, dtype=torch.bfloat16)


def test_a_repeat_launches_the_cached_kernel_with_this_calls_grid_args_and_stream():
    fn, streams = _Jit(), ["stream-0"]
    launcher = _launcher(lambda: fn, streams)
    x, y = _x(), _x()
    first = launcher((4, 2), x, 7, BLOCK=64)
    assert len(fn.dispatches) == 1 and first.inits == 1 and first.calls == []
    streams.append("stream-1")
    again = launcher((5,), y, 9, BLOCK=64)
    assert again is first and len(fn.dispatches) == 1
    (call,) = first.calls
    assert call[:9] == (5, 1, 1, "stream-1", "function-jit-0", "metadata-jit-0", None, None, None)
    assert call[9] is y and call[10:] == (9,)
    launcher((3, 2, 6), x, 11, BLOCK=64)
    assert first.calls[1][:4] == (3, 2, 6, "stream-1") and first.calls[1][9] is x and first.calls[1][10:] == (11,)


def test_a_new_specialisation_or_option_goes_through_the_jit():
    fn = _Jit()
    launcher = _launcher(lambda: fn)
    x = _x()
    launcher((1,), x, 7, BLOCK=64)
    launcher((1,), x, 16, BLOCK=64)            # n % 16 == 0
    launcher((1,), x, 1, BLOCK=64)             # n == 1 is a constexpr
    launcher((1,), x[1:], 7, BLOCK=64)         # 2-byte offset: not 16-byte aligned
    launcher((1,), x.float(), 7, BLOCK=64)
    launcher((1,), x, 7, BLOCK=128)
    launcher((1,), x, 2**31, BLOCK=64)
    assert len(fn.dispatches) == 7 and len(launcher.cache) == 7


def test_ints_key_on_the_width_triton_picks():
    assert len({_spec(2**31 - 16), _spec(2**31), _spec(2**63)}) == 3
    assert _spec(-(2**31)) != _spec(-(2**31) - 16)


def test_a_replaced_function_never_reuses_the_old_compiled_kernel():
    old, new = _Jit("old"), _Jit("new")
    current = [old]
    launcher = _launcher(lambda: current[0])
    x = _x()
    first = launcher((1,), x, 7, BLOCK=64)
    current[0] = new
    second = launcher((1,), x, 7, BLOCK=64)
    assert second is not first and second.tag == "new-0" and len(new.dispatches) == 1 and first.calls == []
    current[0] = old
    assert launcher((1,), x, 7, BLOCK=64) is first and len(old.dispatches) == 1 and len(first.calls) == 1


def test_a_cache_entry_keeps_its_function_alive_so_its_id_is_never_reused():
    launcher = _launcher(lambda: current[0])
    current = [_Jit("gone")]
    launcher((1,), _x(), 7, BLOCK=64)
    gone = weakref.ref(current[0])
    current[0] = None
    gc.collect()
    assert gone() is not None
    for i in range(200):                       # fresh temporaries: each one dispatches once, none hits another's entry
        current[0] = _Jit(f"t{i}")
        launcher((1,), _x(), 7, BLOCK=64)
        assert len(current[0].dispatches) == 1


def test_an_entry_under_a_reused_id_with_another_function_is_a_miss():
    fn, stale = _Jit("fn"), _Compiled("stale")
    launcher = _launcher(lambda: fn)
    x = _x()
    key = (id(fn), x.device.index, (_spec(x), _spec(7)), (("BLOCK", 64),))
    launcher.cache[key] = (object(), stale, stale.run)
    assert launcher((1,), x, 7, BLOCK=64).tag == "fn-0" and stale.calls == [] and len(fn.dispatches) == 1


def test_capture_records_the_compiled_kernel_of_fresh_and_cached_launches():
    fn = _Jit()
    launcher = _launcher(lambda: fn)
    x = _x()
    for _ in range(50):                        # temporary captures around the Launcher, as bench_qmm makes them
        capture = _Capture(launcher)
        compiled = capture((1,), x, 7, BLOCK=64)
        assert capture.compiled is compiled is fn.kernels[next(iter(fn.kernels))]
    assert len(fn.dispatches) == 1


def test_temporary_captures_of_the_jit_function_each_see_a_dispatch():
    fn = _Jit()
    current = [fn]
    launcher = _launcher(lambda: current[0])
    x = _x()
    for _ in range(50):                        # the c155552 failure: a capture under a reused id hit a stale entry
        capture = _Capture(fn)
        current[0] = capture
        launcher((1,), x, 7, BLOCK=64)
        launcher((1,), x, 7, BLOCK=64)
        current[0] = fn
        assert capture.compiled is not None and capture.compiled.calls
        del capture


def test_runtime_arguments_must_all_come_positionally():
    fn = _Jit(signature={"x": "*bf16", "n": "i32", "m": "i32", "BLOCK": "constexpr"})
    launcher = _launcher(lambda: fn)
    with pytest.raises(TypeError, match="positionally"):
        launcher((1,), _x(), 7, BLOCK=64)
    assert launcher.cache == {}


def test_a_tensor_off_the_current_device_never_launches():
    fn = _Jit()
    launcher = _launcher(lambda: fn)
    launcher.device = lambda: 1
    with pytest.raises(ValueError, match="current device"):
        launcher((1,), _x(), 7, BLOCK=64)
    assert fn.dispatches == [] and launcher.cache == {}
