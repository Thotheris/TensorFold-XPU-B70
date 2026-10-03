"""Bundle validation implements only the JSON Schema subset shipped with the harness."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_KEYWORDS = {
    "$schema", "$defs", "$ref", "oneOf", "const", "type", "required", "properties", "additionalProperties", "enum",
}
_TYPES = {"null", "boolean", "object", "array", "number", "integer", "string"}


def _check_schema(schema: Any) -> None:
    if not isinstance(schema, dict):
        raise RuntimeError("schema nodes must be objects")  # noqa: TRY004 - malformed schemas are runtime errors
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise RuntimeError(f"unsupported schema keywords: {', '.join(sorted(unknown))}")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if any(not isinstance(kind, str) or kind not in _TYPES for kind in types):
            raise RuntimeError(f"unsupported schema types: {types!r}")
    for keyword in ("$defs", "properties"):
        for child in schema.get(keyword, {}).values():
            _check_schema(child)
    for child in schema.get("oneOf", []):
        _check_schema(child)
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, dict):
        _check_schema(additional)
    elif not isinstance(additional, bool):
        raise RuntimeError("additionalProperties must be a boolean or schema")  # noqa: TRY004


def _matches_type(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict) and all(isinstance(key, str) for key in value)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return kind == "number" or kind == "integer" and (isinstance(value, int) or value.is_integer())


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _resolve(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise RuntimeError(f"unsupported schema reference: {reference!r}")
    node = root
    try:
        for piece in reference[2:].split("/"):
            node = node[piece.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"missing schema reference: {reference}") from exc
    if not isinstance(node, dict):
        raise RuntimeError(f"schema reference does not name an object: {reference}")  # noqa: TRY004
    return node


def _validate(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> list[str]:
    errors = []
    if "$ref" in schema:
        errors.extend(_validate(value, _resolve(root, schema["$ref"]), root, path))
    if "oneOf" in schema:
        alternatives = [_validate(value, child, root, path) for child in schema["oneOf"]]
        matches = sum(not alternative for alternative in alternatives)
        if matches != 1:
            errors.append(f"{path}: expected exactly one schema match, found {matches}")
            if matches == 0:
                errors.extend(error for alternative in alternatives for error in alternative)
    if "const" in schema and not _equal(value, schema["const"]):
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(_equal(value, choice) for choice in schema["enum"]):
        errors.append(f"{path}: expected one of {schema['enum']!r}")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_matches_type(value, kind) for kind in types):
            errors.append(f"{path}: expected type {' or '.join(types)}")
            return errors
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            if key in properties:
                errors.extend(_validate(child, properties[key], root, f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}: unexpected field {key!r}")
            elif isinstance(additional, dict):
                errors.extend(_validate(child, additional, root, f"{path}.{key}"))
    return errors


def validate_document(doc: object) -> list[str]:
    """An empty error list means the document matches exactly one bundle kind."""
    schema = json.loads(Path(__file__).with_name("schema.json").read_text(encoding="utf-8"))
    _check_schema(schema)
    errors = _validate(doc, schema, schema, "$")
    if isinstance(doc, dict) and doc.get("kind") == "env" and doc.get("image") is not None:
        image = doc["image"]
        if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            errors.append("$.image: expected an immutable Docker image ID")
        versions = doc.get("versions")
        if not isinstance(versions, dict) or versions.get("image") != image:
            errors.append("$.versions.image: differs from runtime image")
    return errors
