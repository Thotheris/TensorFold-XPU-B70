"""Bundle paths contain only validated branch, revision, and timestamp components."""

from __future__ import annotations

import re


def branch_slug(branch: str) -> str:
    """XPU branch components become a safe double-hyphen slug."""
    if not isinstance(branch, str) or not branch.startswith("xpu/"):
        raise ValueError("branch must start with xpu/")
    pieces = branch.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", piece) or ".." in piece for piece in pieces):
        raise ValueError("branch has an empty, unsafe, or parent-directory component")
    if any(piece == "." for piece in pieces):
        raise ValueError("branch has a current-directory component")
    return "--".join(pieces)


def sha7(sha: str) -> str:
    """A hexadecimal revision has at least seven digits."""
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{7,}", sha):
        raise ValueError("sha must contain at least seven hexadecimal digits")
    return sha[:7]


def bundle_rel(branch: str, sha: str, utc: str) -> str:
    """A bundle lives under runs with a compact UTC timestamp."""
    if not isinstance(utc, str) or not re.fullmatch(r"\d{8}T\d{6}Z", utc, flags=re.ASCII):
        raise ValueError("utc must match YYYYMMDDTHHMMSSZ")
    return f"runs/{branch_slug(branch)}/{sha7(sha)}-{utc}"
