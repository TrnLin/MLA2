"""Streaming file hashing helpers."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any


def compute_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_header_and_id_fingerprint(path: str | Path) -> dict[str, Any]:
    """Hash only a CSV header and ordered IDs, never the other cell values."""
    source = Path(path)
    digest = hashlib.sha256()
    row_count = 0
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"CSV has no ID header: {source}")
        digest.update(
            json.dumps(reader.fieldnames, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
        for row in reader:
            digest.update(f"{int(row['id'])}\n".encode("ascii"))
            row_count += 1
    return {
        "rows": row_count,
        "header_and_id_sha256": digest.hexdigest(),
        "protected_target_values_hashed": 0,
    }


def write_deterministic_csv(frame: Any, path: str | Path, **kwargs: Any) -> Path:
    """Write CSV, using path-independent gzip bytes with ``mtime=0`` for ``.gz``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".gz":
        frame.to_csv(output, **kwargs)
        return output
    with output.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=1,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, **kwargs)
    return output
