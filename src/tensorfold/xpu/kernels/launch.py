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
        width = "i32" if -(2**31) <= arg < 2**31 else "i64" if arg < 2**63 else "u64"
        return ("int", arg == 1, arg % 16 == 0, width)
    if isinstance(arg, float):
        return "float"
    raise TypeError(f"no launch key for {type(arg).__name__}")


def _check_arity(compiled: Any, nargs: int) -> None:
    """Triton's launcher reads one positional value per non-constexpr parameter: all of them must come in ``args``."""

    signature = getattr(getattr(compiled, "src", None), "signature", None)
    if not isinstance(signature, dict):
        return
    kinds = list(signature.values())
    if nargs > len(kinds) or any(kind != "constexpr" for kind in kinds[nargs:]):
        names = [name for name, kind in list(signature.items())[nargs:] if kind != "constexpr"]
        raise TypeError(f"Launcher: {nargs} positional args for {len(kinds)} parameters; runtime parameters "
                        f"{names} must be passed positionally, before every keyword constexpr")


class Launcher:
    """``launcher(grid, *args, **compile_options)`` launches the kernel with the bits of ``kernel[grid](*args, ...)``.

    ``args[0]`` must be a tensor on the current device; every runtime parameter comes positionally in ``args``, and
    keyword options must come in the same order at a call site.

    The first call per key goes through Triton's JIT (compiling if needed); later calls with the same specialisation
    call the compiled kernel's ``run`` directly with this call's grid and args on the device's current stream (about
    11 us of host time instead of 32 us), skipping Triton's launch hooks. Both paths return the compiled kernel they
    launched. A cache entry holds its function, so a key's ``id`` cannot be reused by another object; ``kernel()``
    must return a long-lived object (a patched one gets its own entries).
    """

    def __init__(self, kernel) -> None:
        self.kernel = kernel          # a zero-argument callable returning the jit function (patchable by tests)
        self.cache: dict[tuple, tuple] = {}
        self.stream = None            # driver hooks, bound at the first compile
        self.device = None

    def __call__(self, grid: tuple[int, ...], *args: Any, **options: Any) -> Any:
        fn = self.kernel()
        device = args[0].device
        key = (id(fn), device.index, tuple([_spec(a) for a in args]), tuple(options.items()))
        entry = self.cache.get(key)
        if entry is None or entry[0] is not fn:
            return self._compile(key, fn, grid, args, options)
        _, compiled, run = entry
        g = tuple(grid) + (1,) * (3 - len(grid))
        run(g[0], g[1], g[2], self.stream(device.index), compiled.function, compiled.packed_metadata, None, None, None,
            *args)
        return compiled

    def _compile(self, key: tuple, fn: Any, grid: tuple[int, ...], args: tuple, options: dict) -> Any:
        if self.stream is None:
            from triton.runtime.driver import driver

            self.stream, self.device = driver.active.get_current_stream, driver.active.get_current_device
        if self.device() != args[0].device.index:
            raise ValueError(f"Launcher: args[0] is on {args[0].device}, not the current device {self.device()}")
        compiled = fn[grid](*args, **options)
        compiled._init_handles()
        _check_arity(compiled, len(args))
        self.cache[key] = (fn, compiled, compiled.run)
        return compiled
