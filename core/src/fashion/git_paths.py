"""Translate model-relative paths to paths in the enclosing Git checkout."""
import subprocess
from pathlib import Path


def git_project_prefix(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()
