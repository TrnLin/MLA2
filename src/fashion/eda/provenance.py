"""Traceable input, runtime, and output records for EDA evidence."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from fashion.data.hashing import compute_sha256


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_record(
    path: str | Path,
    root: str | Path,
    rows: int | None = None,
) -> dict[str, Any]:
    """Describe a file by path, bytes, digest, and optional row count."""
    path = Path(path)
    record: dict[str, Any] = {
        "path": relative_path(path, Path(root)),
        "sha256": compute_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def csv_record(path: str | Path, root: str | Path) -> dict[str, Any]:
    return file_record(path, root, rows=len(pd.read_csv(path)))


def runtime_record() -> dict[str, Any]:
    """Capture the versions that materially affect calculations and rendering."""
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "Pillow": "Pillow",
        "matplotlib": "matplotlib",
    }
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "matplotlib_backend": str(matplotlib.get_backend()),
        "libraries": {
            label: importlib.metadata.version(package) for label, package in packages.items()
        },
    }
