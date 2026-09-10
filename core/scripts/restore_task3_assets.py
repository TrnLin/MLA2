"""Restore ignored Task 3 assets from a retained local source checkout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(source: Path, *, verify_only: bool = False) -> int:
    """Check receipt hashes and never overwrite a different destination file."""
    count = 0
    for inventory in ("asset-inventory.csv", "local-assets.csv"):
        with (ROOT / "reports/task3" / inventory).open() as stream:
            for row in csv.DictReader(stream):
                if not row["disposition"].startswith("local"):
                    continue
                origin, target = source / row["source_path"], ROOT / row["current_path"]
                if not target.resolve().is_relative_to(ROOT):
                    raise ValueError(f"Destination escapes this checkout: {target}")
                expected = row["source_sha256"]
                if target.is_file():
                    if digest(target) != expected:
                        raise ValueError(f"Existing asset differs; left untouched: {target}")
                elif verify_only:
                    raise FileNotFoundError(f"Missing asset: {target}")
                else:
                    if digest(origin) != expected:
                        raise ValueError(f"Source asset differs: {origin}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin, target)
                    if digest(target) != expected:
                        raise ValueError(f"Copied asset differs: {target}")
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Retained source checkout")
    parser.add_argument(
        "--verify-only", action="store_true", help="Check local bytes without copying"
    )
    args = parser.parse_args()
    print(f"Verified {restore(args.source.resolve(), verify_only=args.verify_only)} local assets.")
