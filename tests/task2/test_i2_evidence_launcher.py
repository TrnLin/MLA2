from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/build_task2_i2_evidence.py"


def test_i2_evidence_launcher_has_safe_main_guard_and_load_only_runner() -> None:
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
    source = LAUNCHER.read_text(encoding="utf-8")

    assert len(guards) == 1
    assert 'mode="load"' in source
    assert "build_experiment_evidence" in source
    assert "build_i2_transfer_evidence" in source
    assert 'mode="run"' not in source
    assert 'mode="run_or_load"' not in source


def test_i2_evidence_launcher_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=Path(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "aligned/conflict" in completed.stdout
    assert "lambda 0.1" in completed.stdout
    assert "lambda 0.3" in completed.stdout
