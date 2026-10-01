"""Run requests use a small, explicit YAML subset without external dependencies."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunSpec:
    """A run specifies suites, a baseline, and bounded timeouts."""

    suites: tuple[str, ...]
    baseline: str = "xpu/main"
    timeout_min: int = 90
    model_cache: str | None = None
    suite_timeouts_min: dict[str, int] = field(default_factory=dict)


def known_suite(name: str) -> bool:
    """Only named harness suites are executable requests."""
    return isinstance(name, str) and (
        name in {"env", "unit-host", "unit-xpu", "e2e:27b-smoke", "e2e:nemotron-smoke"}
        or re.fullmatch(r"kernels:[A-Za-z0-9_-]+", name) is not None
        or re.fullmatch(r"e2e:[A-Za-z0-9_-]+-bench", name) is not None
    )


def _uncomment(line: str) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif quote and char == "\\":
            escaped = True
        elif char == quote:
            quote = None
        elif not quote and char in "\"'":
            quote = char
        elif not quote and char == "#":
            return line[:index].strip()
    if quote:
        raise ValueError("unterminated quoted value")
    return line.strip()


def _string(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty value")
    if value[0] in "\"'":
        try:
            result = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid quoted value") from exc
        if not isinstance(result, str):
            raise ValueError("quoted value must be a string")
        return result
    if any(char in value for char in "[]{}\"'"):
        raise ValueError("unsupported YAML value")
    return value


def _positive_int(value: str) -> int:
    if not re.fullmatch(r"[0-9]+", value.strip()) or int(value) < 1:
        raise ValueError("timeout must be a positive integer")
    return int(value)


def _suites(value: str) -> tuple[str, ...]:
    if value.startswith("["):
        if not value.endswith("]"):
            raise ValueError("unterminated suite list")
        value = value[1:-1]
    suites = tuple(_string(item) for item in value.split(","))
    if not suites or any(not known_suite(name) for name in suites):
        raise ValueError("suites must be a nonempty list of known suite names")
    return suites


def _timeouts(value: str) -> dict[str, int]:
    if not value.startswith("{") or not value.endswith("}"):
        raise ValueError("suite_timeouts_min must be a one-level flow mapping")
    result = {}
    if not value[1:-1].strip():
        return result
    for item in value[1:-1].split(","):
        key, separator, minutes = item.rpartition(":")
        if not separator:
            raise ValueError("invalid suite timeout entry")
        name = _string(key)
        if not known_suite(name) or name in result:
            raise ValueError(f"unknown or duplicate suite timeout: {name}")
        result[name] = _positive_int(minutes)
    return result


def parse_run_yml(text: str) -> RunSpec:
    """Only supported top-level run fields are parsed."""
    allowed = {"suites", "baseline", "timeout_min", "model_cache", "suite_timeouts_min"}
    values = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = _uncomment(raw)
        if not line:
            continue
        key, separator, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not separator or key not in allowed:
            raise ValueError(f"line {number}: unknown run key {key!r}")
        if key in values:
            raise ValueError(f"line {number}: duplicate run key {key!r}")
        values[key] = value
    if "suites" not in values:
        raise ValueError("required run key is missing: suites")
    return RunSpec(
        suites=_suites(values["suites"]),
        baseline=_string(values.get("baseline", "xpu/main")),
        timeout_min=_positive_int(values.get("timeout_min", "90")),
        model_cache=_string(values["model_cache"]) if "model_cache" in values else None,
        suite_timeouts_min=_timeouts(values.get("suite_timeouts_min", "{}")),
    )


def suite_timeout_min(spec: RunSpec, name: str) -> int:
    """Per-suite limits never exceed the whole-run cap."""
    if not known_suite(name):
        raise ValueError(f"unknown suite: {name}")
    defaults = {"env": 10, "unit-host": 30, "unit-xpu": 60}
    default = defaults.get(name, 20 if name.startswith("kernels:") else 90 if name.endswith("-bench") else 45)
    return min(spec.timeout_min, spec.suite_timeouts_min.get(name, default))
