"""Streaming file hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
