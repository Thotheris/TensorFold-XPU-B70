"""Direct launches of compiled Triton kernels: the JIT dispatch runs once per specialisation, later calls skip it."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["Launcher"]


def _spec(arg: Any) -> Any:
    """What Triton specialises a runtime argument on: dtype and 16-byte alignment, or an int's 1 / %16 / width."""

    if isinstance(arg, torch.Tensor):
        return (arg.dtype, arg.data_ptr() % 16 == 0)
    if isinstance(arg, bool):
        return ("bool", arg)
    if isinstance(arg, int):
        return ("int", arg == 1, arg % 16 == 0, -(2**31) <= arg < 2**31)
    if isinstance(arg, float):
        return "float"
    raise TypeError(f"no launch key for {type(arg).__name__}")


class Launcher:
    """``launcher(grid, *args, **compile_options)`` launches the kernel with the bits of ``kernel[grid](*args, ...)``.

    ``args[0]`` must be a tensor on the launch device; keyword options must come in the same order at a call site.

    The first call per key goes through Triton's JIT (compiling if needed); later calls with the same specialisation
    call the compiled kernel's ``run`` directly on the current stream (about 11 us of host time instead of 32 us).
    """

    def __init__(self, kernel) -> None:
        self.kernel = kernel          # a zero-argument callable returning the jit function (patchable by tests)
        self.cache: dict[tuple, Any] = {}
        self.stream = None

    def __call__(self, grid: tuple[int, ...], *args: Any, **options: Any) -> Any:
        fn = self.kernel()
        device = args[0].device
        key = (id(fn), device.index, tuple([_spec(a) for a in args]), tuple(options.items()))
        compiled = self.cache.get(key)
        if compiled is None:
            compiled = fn[grid](*args, **options)
            compiled._init_handles()
            self.cache[key] = compiled
            return compiled
        if self.stream is None:
            from triton.runtime.driver import driver

            self.stream = driver.active.get_current_stream
        g = tuple(grid) + (1,) * (3 - len(grid))
        compiled.run(g[0], g[1], g[2], self.stream(device.index), compiled.function, compiled.packed_metadata, None,
                     None, None, *args)
        return compiled
