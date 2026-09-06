from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/run_task2_experiment.py"


def test_launcher_has_windows_safe_main_guard() -> None:
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

    assert len(guards) == 1


def test_launcher_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=Path(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "run_or_load" in completed.stdout
    assert "configs/task2" in completed.stdout
