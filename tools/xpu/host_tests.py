"""Host collection excludes MLX-dependent modules while retaining their source tests."""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

__all__ = ["mlx_modules"]


def _uses_mlx(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in {"mlx", "mlx_lm"}
                                                for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"mlx", "mlx_lm"}:
            return True
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.split(".")[0] in {"mlx", "mlx_lm"}):
            return True
        if isinstance(node, ast.Attribute) and node.attr == "_serve_mlx":
            return True
    return False


@cache
def mlx_modules(tests: Path) -> frozenset[Path]:
    """Direct MLX use and imports of MLX test helpers propagate at module granularity."""
    modules: dict[str, Path] = {}
    trees = {}
    for path in tests.rglob("*.py"):
        module = ".".join(("tests", *path.relative_to(tests).with_suffix("").parts))
        modules[module] = path.resolve()
        trees[module] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    excluded = {name for name, tree in trees.items() if _uses_mlx(tree)}
    dependencies = {}
    for name, tree in trees.items():
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    parent = name.split(".")[:-node.level]
                    base = ".".join((*parent, base))
                imports.add(base)
                imports.update(f"{base}.{alias.name}" for alias in node.names)
        dependencies[name] = imports
    while True:
        added = {name for name, imports in dependencies.items() if imports & excluded} - excluded
        if not added:
            return frozenset(modules[name] for name in excluded)
        excluded.update(added)
