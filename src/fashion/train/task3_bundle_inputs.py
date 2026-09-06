"""Reuse verified Drive inputs with a fresh checkout of the training code."""

import hashlib
import shutil
from pathlib import Path


def install_bundle_inputs(bundle_root: Path, root: Path) -> int:
    """Copy data and reference results only; reject conflicting local files."""
    count = 0
    for folder in ("data", "reference", "results"):
        source = bundle_root / folder
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.is_symlink() or path.suffix in {".py", ".pyc", ".ipynb"}:
                raise ValueError(f"Bundle input is not a data file: {path}")
            target = root / path.relative_to(bundle_root)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Local path escapes the checkout: {target}")
            if target.exists():
                if (
                    hashlib.sha256(target.read_bytes()).digest()
                    != hashlib.sha256(path.read_bytes()).digest()
                ):
                    raise ValueError(f"Local input differs; use a fresh runtime: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            count += 1
    return count
