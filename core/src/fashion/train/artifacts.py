"""Atomic artifact writes and SHA-256 verification for experiment outputs."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from fashion.data.hashing import compute_sha256


class ArtifactVerificationError(RuntimeError):
    """Raised when an artifact is missing or its content digest is unexpected."""


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value in a stable, whitespace-free form."""
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of a value's canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Replace a file only after its complete payload is durable on disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return output


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write text using an explicit encoding."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Atomically write deterministic, human-readable JSON."""
    payload = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    return atomic_write_text(path, f"{payload}\n")


def atomic_write_csv(path: str | Path, frame: Any, **kwargs: Any) -> Path:
    """Atomically write a DataFrame-like object as deterministic CSV text."""
    kwargs.setdefault("index", False)
    kwargs.setdefault("lineterminator", "\n")
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, **kwargs)
    return atomic_write_text(path, buffer.getvalue())


def verify_artifact(path: str | Path, expected_sha256: str) -> str:
    """Return the actual digest, or raise when a file is absent or changed."""
    artifact = Path(path)
    if not artifact.is_file():
        raise ArtifactVerificationError(f"artifact does not exist: {artifact}")
    actual = compute_sha256(artifact)
    if actual != expected_sha256.lower():
        raise ArtifactVerificationError(
            f"artifact SHA-256 mismatch for {artifact}: "
            f"expected {expected_sha256.lower()}, got {actual}"
        )
    return actual
