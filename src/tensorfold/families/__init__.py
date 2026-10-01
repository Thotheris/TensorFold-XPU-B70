"""Discover family packages by model_type and inspect checkpoint compatibility before loading weights or MLX."""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class Family:
    model_type: str
    title: str
    module: str
    lanes: bool  # served by the lane engine

    @property
    def package(self) -> ModuleType:
        return importlib.import_module(self.module)


_found: dict[str, Family] | None = None


def families() -> dict[str, Family]:
    """Every family package in this folder, by the model_type values it serves."""

    global _found
    if _found is None:
        found: dict[str, Family] = {}
        for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
            if not info.ispkg:
                continue
            module = f"{__name__}.{info.name}"
            package = importlib.import_module(module)
            for kind in getattr(package, "MODEL_TYPES", ()):
                if kind in found:
                    raise ValueError(f"model_type {kind!r} claimed by {found[kind].module} and {module}")
                found[kind] = Family(kind, str(getattr(package, "TITLE", info.name)), module,
                                     bool(getattr(package, "LANES", False)))
        _found = found
    return _found


def read_config(model_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "config.json").read_text())


RECIPES_URL = "https://github.com/ashhart/TensorFold/blob/main/docs/recipes/README.md"
RUNBOOK_URL = "https://github.com/ashhart/TensorFold/blob/main/RUNBOOK.md"
OWN_MODEL_HELP = (f"To run a model or checkpoint TensorFold has no recipe for, write one with the recipe book "
                  f"({RECIPES_URL}: adding a family on a Mac, adding a CUDA family on NVIDIA GPUs), and read the "
                  f"runbook first ({RUNBOOK_URL}).")
MLX_QUANT = "mlx"
EXL3_QUANT = "exl3"
XPU_QUANTS = ("auto-round", "gptq", "compressed-tensors")   # the symmetric INT4 weight-only formats XPU reads
EXL3_VARIANT_ANY = "any"          # a family declaring EXL3_VARIANT = EXL3_VARIANT_ANY reads every codebook and width


def _quantization_block(config: dict[str, Any]) -> dict[str, Any] | None:
    for source in (config, config.get("text_config") or {}):
        for key in ("quantization", "quantization_config"):
            found = source.get(key)
            if isinstance(found, dict) and found:
                return found
    return None


def quant_method(config: dict[str, Any]) -> str | None:
    """Return the storage format: mlx for affine quantization, mlx-<mode> for other MLX modes, quant_method otherwise, or None."""

    found = _quantization_block(config)
    if found is None:
        return None
    method = found.get("quant_method")
    if method:
        return str(method).lower()
    if "bits" in found:
        mode = str(found.get("mode") or "affine").lower()
        return MLX_QUANT if mode == "affine" else f"{MLX_QUANT}-{mode}"
    return None


def quantization(config: dict[str, Any]) -> tuple[int | None, int | None]:
    """(bits, group size) of MLX affine-quantized weights; (None, None) for any other format or none."""

    found = _quantization_block(config)
    if found is None or quant_method(config) != MLX_QUANT:
        return None, None
    return int(found["bits"]), int(found.get("group_size", 64))


MLX_MODE_DEFAULTS = {"affine": (64, 4), "mxfp4": (32, 4), "nvfp4": (16, 4), "mxfp8": (32, 8)}   # to_quantized's


def layer_quantization(config: dict[str, Any]) -> dict[str, tuple[int, int, str]]:
    """config.json's per-layer MLX entries: path -> (bits, group size, mode), a missing key at MLX's mode default."""

    found = _quantization_block(config) or {}
    layers = {}
    for path, entry in found.items():
        if isinstance(entry, dict) and entry:
            mode = str(entry.get("mode") or "affine").lower()
            group, bits = MLX_MODE_DEFAULTS.get(mode, (64, 4))
            layers[path] = (int(entry.get("bits") or bits), int(entry.get("group_size") or group), mode)
    return layers


def describe_quantization(config: dict[str, Any]) -> str:
    method = quant_method(config)
    if method is None:
        return "none (unquantized weights)"
    if method == MLX_QUANT:
        bits, group = quantization(config)
        return f"MLX {bits}-bit, groups of {group}"
    block = _quantization_block(config) or {}
    bits, algo = block.get("bits"), block.get("quant_algo") or block.get("format")    # ModelOpt / compressed-tensors
    return f"{method}" + (f" {algo}" if algo else "") + (f" ({bits}-bit)" if bits else "")


def backends_of(family: Family) -> tuple[str, ...]:
    """The backends a family has an engine for: ``mlx`` (``load``), ``cuda`` (``cuda_engine``), ``xpu`` (``xpu_engine``)."""

    package = family.package
    return tuple(b for b, member in (("mlx", "load"), ("cuda", "cuda_engine"), ("xpu", "xpu_engine"))
                 if hasattr(package, member))


def readable_quants(family: Family, backend: str) -> tuple[str | None, ...]:
    """Return formats supported by the backend, using the family QUANT_METHODS override when present."""

    declared = getattr(family.package, "QUANT_METHODS", {}) or {}
    default = (MLX_QUANT, None) if backend == "mlx" else XPU_QUANTS if backend == "xpu" else (MLX_QUANT,)
    return tuple(declared.get(backend, default))


def require_readable(family: Family, config: dict[str, Any], backend: str) -> None:
    """Reject unsupported storage formats or MLX quantization dimensions before downloading weights."""

    method = quant_method(config)
    where = {"cuda": "NVIDIA GPUs (CUDA)", "xpu": "Intel GPUs (XPU)"}.get(backend, "Apple Silicon (MLX)")
    tested = ", ".join(getattr(family.package, "MODELS", ())) or "none listed"
    if backend == "xpu":
        _require_xpu_readable(family, config, method, where, tested)
        return
    accepted = readable_quants(family, backend)
    if method not in accepted:
        names = {None: "unquantized weights", MLX_QUANT: "MLX-quantized weights"}
        reads = " or ".join(names.get(m, str(m)) for m in accepted)
        raise ValueError(f"{family.title} on {where} does not read this checkpoint's weights "
                         f"({describe_quantization(config)}); it reads {reads}. Tested checkpoints: {tested}. "
                         f"{OWN_MODEL_HELP}")
    if backend == "cuda" and method in ("modelopt", "compressed-tensors"):
        # NVFP4 / FP8 / MXFP8 weights: the config's schemes here, each linear's tensors by the loader
        from tensorfold.cuda.nvfp4 import format as nvfp4_format

        nvfp4_format.require_config(config, where=where, tested=tested, help=OWN_MODEL_HELP)
    if backend == "cuda" and method == EXL3_QUANT and getattr(family.package, "EXL3_VARIANT", None) == EXL3_VARIANT_ANY:
        # every EXL3 codebook and width: the config is checked here, the tensors by the loader's scan
        from tensorfold.cuda.exl3 import format as exl3_format

        exl3_format.require_config(config, where=where, tested=tested, help=OWN_MODEL_HELP)
    check = getattr(family.package, "check_quantization", None)
    if method == MLX_QUANT and check is not None:
        check(config, backend)                   # the family reads its own affine widths, groups and mixed layers
        return
    expected = getattr(family.package, "CUDA_QUANTIZATION", None) if backend == "cuda" else None
    if expected is not None and method == MLX_QUANT and quantization(config) != tuple(expected):
        bits, group = expected
        raise ValueError(f"{family.title}'s CUDA kernels read MLX {bits}-bit weights in groups of {group}; this "
                         f"checkpoint has {describe_quantization(config)}. Tested checkpoints: {tested}. "
                         f"{OWN_MODEL_HELP}")


XPU_READS = ("XPU reads only symmetric INT4 weight-only checkpoints (W4A16): AutoRound auto_round:auto_gptq, GPTQ v1, "
             "compressed-tensors pack-quantized (docs/xpu/QUANT_FORMATS.md)")


def _require_xpu_readable(family: Family, config: dict[str, Any], method: str | None, where: str, tested: str) -> None:
    """Refuse by name what XPU never reads, then check the W4A16 header; tensors are the loader's."""

    tail = f"Tested checkpoints: {tested}. {OWN_MODEL_HELP}"
    found = describe_quantization(config)
    never = False                                  # formats that share a quant_method with an accepted one
    if method == "compressed-tensors":
        block = _quantization_block(config) or {}
        schemes = [str(g.get("weights", {}).get("type", "")) for g in (block.get("config_groups") or {}).values()
                   if isinstance(g, dict) and isinstance(g.get("weights"), dict)]
        never = any(t in ("float", "nvfp4") for t in schemes) or "float" in str(block.get("format", ""))
        found += f", weight types {'/'.join(schemes) or 'none'}"
    accepted = getattr(family.package, "QUANT_METHODS", {}).get("xpu", XPU_QUANTS)
    if never or method not in accepted:
        raise ValueError(f"{family.title} on {where} does not read this checkpoint's weights. "
                         f"{XPU_READS}; this checkpoint is {found}. {tail}")
    if method is not None:
        _require_xpu_w4a16(config, where, tested)


def _require_xpu_w4a16(config: dict[str, Any], where: str, tested: str) -> None:
    """Check a W4A16 checkpoint's header for what XPU reads: 4-bit symmetric, groups of 64 or 128, no act-order."""

    block = _quantization_block(config) or {}
    method = quant_method(config)
    tail = f"Tested checkpoints: {tested}. {OWN_MODEL_HELP}"

    def refuse(found: str, accepted: str) -> None:
        raise ValueError(f"{where} reads {accepted}; this checkpoint has {found}. {XPU_READS}. {tail}")

    if method == "compressed-tensors":
        formats = {str(block.get("format") or "")} | {str(g.get("format") or "") for g in
                                                      (block.get("config_groups") or {}).values()
                                                      if isinstance(g, dict)}
        formats.discard("")
        if formats != {"pack-quantized"}:
            refuse(f"compressed-tensors format {', '.join(sorted(formats)) or 'none'}", "compressed-tensors pack-quantized")
        groups = [g for g in (block.get("config_groups") or {}).values() if isinstance(g, dict)]
        if not groups:
            refuse("no config_groups", "compressed-tensors config_groups with 4-bit weights")
        for group in groups:
            w = group.get("weights") or {}
            found = (f"weights num_bits {w.get('num_bits')}, type {w.get('type')}, symmetric {w.get('symmetric')}, "
                     f"strategy {w.get('strategy')}, group_size {w.get('group_size')}, actorder {w.get('actorder')}")
            if w.get("num_bits") != 4 or w.get("type") != "int":
                refuse(found, "4-bit int weights")
            if w.get("symmetric") is not True:
                refuse(found, "symmetric weights")
            if w.get("strategy") != "group" or w.get("group_size") not in (64, 128):
                refuse(found, "group strategy with group_size 64 or 128")
            if w.get("actorder"):
                refuse(found, "weights without act-order")
            if group.get("input_activations") is not None:
                refuse(f"input_activations {group.get('input_activations')}", "weight-only W4A16 (input_activations null)")
        return
    packing = block.get("packing_format")
    if method == "auto-round" or method == "auto_round":
        if packing is not None and not (str(packing).startswith("auto_round:auto_gptq") or packing == "auto_gptq"):
            refuse(f"packing_format {packing}", "packing_format auto_round:auto_gptq")
    elif block.get("checkpoint_format") not in (None, "gptq"):
        refuse(f"checkpoint_format {block.get('checkpoint_format')}", "GPTQ v1 (checkpoint_format gptq)")
    found = (f"bits {block.get('bits')}, sym {block.get('sym')}, group_size {block.get('group_size')}, "
             f"desc_act {block.get('desc_act')}")
    if block.get("bits") != 4:
        refuse(found, "4-bit weights")
    if block.get("sym") is not True:
        refuse(found, "symmetric weights (sym true)")
    if block.get("group_size") not in (64, 128):
        refuse(found, "group_size 64 or 128")
    if block.get("desc_act"):
        refuse(found, "weights without act-order (desc_act false)")
    act_bits, data_type = block.get("act_bits"), block.get("data_type")
    if act_bits not in (None, 16, 32) or str(data_type or "int").lower() != "int":
        refuse(f"data_type {data_type}, act_bits {act_bits}", "int weights with 16-bit activations (W4A16)")
    extra = block.get("extra_config")
    for layer, entry in (extra.items() if isinstance(extra, dict) else ()):
        if isinstance(entry, dict) and (entry.get("bits", 4) not in (4, 16) or entry.get("sym", True) is False):
            refuse(f"extra_config {layer}: {entry}", "4-bit symmetric layers (mixed bits)")


def model_type(model_dir: str | Path) -> str:
    config = read_config(model_dir)
    return str(config.get("model_type") or config.get("text_config", {}).get("model_type") or "")


def detect(model_dir: str | Path) -> Family:
    kind = model_type(model_dir)
    family = families().get(kind)
    if family is None:
        raise ValueError(f"TensorFold has no recipe for model_type {kind!r} yet (it has: "
                         f"{', '.join(sorted(families()))}; `tensorfold models` lists the tested checkpoints). "
                         f"{OWN_MODEL_HELP}")
    return family


def load(model_dir: str | Path, **options: Any) -> tuple[Any, Any]:
    family = detect(model_dir)
    return family.package.load(Path(model_dir), **options)


def kernel_version(family: Family, model: Any) -> str:
    """Fingerprint the active family and versioned kernels for safe prefix-snapshot reuse."""

    hook = getattr(family.package, "kernel_version", None)
    if hook is not None:
        return str(hook(model))
    source = kernel_source_version(family)
    prefill_key = getattr(model, "prefill_key", None)              # how prompts are prefilled changes the bits too
    return f"{source}|{prefill_key}" if prefill_key else source


def kernel_source_version(family: Family) -> str:
    """Hash a family's source and kernel dependencies without invoking its runtime hook."""

    digest = hashlib.sha256()
    modules = (family.module, getattr(family.package, "KERNEL_PACKAGE", ""),
               *getattr(family.package, "KERNEL_DEPENDENCIES", ()))
    for module_name in filter(None, modules):
        digest.update(module_name.encode())
        module = importlib.import_module(module_name)
        source = Path(str(module.__file__))
        paths = sorted(source.parent.rglob("*.py")) if source.name == "__init__.py" else [source]
        for path in paths:
            digest.update(path.relative_to(source.parent).as_posix().encode())
            digest.update(path.read_bytes())
    version = getattr(family.package, "KERNEL_VERSION", "")
    prefix = f"{family.model_type}-{version}-" if version else ""
    return prefix + digest.hexdigest()[:12]
