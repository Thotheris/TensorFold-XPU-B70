"""Bundle comparisons identify new failures, bitwise breaks, and performance regressions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PERF_THRESHOLD = 0.03
_FAILED = {"fail", "failed", "failure", "error"}


def _read(path: Path) -> dict:
    """Missing or malformed JSON documents have no comparable fields."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _failures(bundle: Path) -> set[str]:
    doc = _read(bundle / ("summary.json" if (bundle / "summary.json").exists() else "status.json"))
    suites = doc.get("suites", {})
    failures = set()
    if isinstance(suites, dict):
        for name, status in suites.items():
            if isinstance(status, dict):
                status = status.get("status")
            if isinstance(status, str) and status.lower() in _FAILED:
                failures.add(name)
    cases = doc.get("pytest_failures", [])
    if isinstance(cases, list):
        failures.update(name for name in cases if isinstance(name, str))
    return failures


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_bundles(current: Path, baseline: Path, *, threshold: float = PERF_THRESHOLD) -> dict:
    """Only newly broken checks and slower comparable measurements are regressions."""
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("threshold must be finite and in [0, 1)")
    report = {
        "new_failures": sorted(_failures(current) - _failures(baseline)),
        "bitwise_breaks": [],
        "perf_regressions": [],
        "ok": True,
    }
    for category in ("kernels", "e2e"):
        for old_path in sorted((baseline / category).glob("*.json")):
            relative = old_path.relative_to(baseline)
            new_path = current / relative
            old, new = _read(old_path), _read(new_path)
            if (
                category == "kernels" and old.get("bitwise_ok") is True
                and (not new_path.exists() or new.get("bitwise_ok") is False)
            ):
                report["bitwise_breaks"].append(
                    {"file": relative.as_posix(), "baseline": True, "current": new.get("bitwise_ok")}
                )
            metrics = ("gbps", "median_us") if category == "kernels" else ("tok_s",)
            for metric in metrics:
                before, after = old.get(metric), new.get(metric)
                if not (_number(before) and _number(after)):
                    continue
                if metric == "median_us":
                    regressed = after > before * (1 + threshold)
                else:
                    regressed = after < before * (1 - threshold)
                if regressed:
                    report["perf_regressions"].append(
                        {"file": relative.as_posix(), "metric": metric, "baseline": before, "current": after}
                    )
    report["ok"] = not any(report[key] for key in ("new_failures", "bitwise_breaks", "perf_regressions"))
    return report


def render_diff(report: dict) -> str:
    """The regression summary is short Markdown."""
    if not any(report.get(key) for key in ("new_failures", "bitwise_breaks", "perf_regressions")):
        return "No regressions."
    lines = ["## Baseline regressions"]
    lines.extend(f"- New failure: `{name}`" for name in report.get("new_failures", []))
    lines.extend(f"- Bitwise break: `{item['file']}`" for item in report.get("bitwise_breaks", []))
    lines.extend(
        f"- Performance: `{item['file']}` {item['metric']} {item['baseline']:g} -> {item['current']:g}"
        for item in report.get("perf_regressions", [])
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """The command compares two result bundle directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--threshold", type=float, default=PERF_THRESHOLD)
    args = parser.parse_args(argv)
    report = compare_bundles(args.current, args.baseline, threshold=args.threshold)
    print(render_diff(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
