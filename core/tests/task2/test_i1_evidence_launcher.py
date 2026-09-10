from __future__ import annotations

import ast
import subprocess
import sys

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/build_task2_i1_evidence.py"


def test_i1_evidence_launcher_is_load_only_and_windows_safe() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    guarded = [
        node
        for node in module.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]

    assert "run_or_load_i1_experiment" in imports
    assert "build_experiment_evidence" in imports
    assert "build_i1_class_balance_evidence" in imports
    assert 'mode="load"' in source
    assert "holdout" in source
    assert "quarantine" in source
    assert guarded


def test_i1_evidence_launcher_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "five verified I1 folds" in completed.stdout
    assert "g4_i1_effective_number_c1.json" in completed.stdout
