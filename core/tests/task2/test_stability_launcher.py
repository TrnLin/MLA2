from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/run_task2_stability.py"


def test_stability_launcher_has_windows_safe_main_guard_and_paired_runner() -> None:
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "__main__"
            for comparator in node.test.comparators
        )
    ]
    imports = {
        alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
    }

    assert len(guards) == 1
    assert "run_stability_matrix" in imports


def test_stability_launcher_help_names_seed_and_excludes_pstar() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=Path(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    help_text = " ".join(completed.stdout.split())
    assert "seed 2026" in help_text
    assert "P* is excluded" in help_text
