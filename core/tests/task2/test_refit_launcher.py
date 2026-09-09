from __future__ import annotations

import ast
import subprocess
import sys

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/refit_task2_season.py"


def test_refit_launcher_has_one_windows_safe_main_guard() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
    assert "run_or_load_development_refit" in source


def test_refit_launcher_help_states_fixed_epoch_and_holdout_boundaries() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "exactly 24 epochs" in completed.stdout
    assert "No validation selection or holdout labels" in completed.stdout
