from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from fashion.config import ROOT
from fashion.task2.class_balance import validate_i1_config
from fashion.task2.experiments import load_experiment_config

LAUNCHER = ROOT / "scripts/run_task2_i1_experiment.py"
CONFIG = ROOT / "configs/task2/g4_i1_effective_number_c1.json"


def test_i1_launcher_has_windows_safe_main_guard_and_dedicated_runner() -> None:
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
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert len(guards) == 1
    assert "run_or_load_i1_experiment" in imports


def test_i1_launcher_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=Path(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "effective-number" in completed.stdout
    assert "run_or_load" in completed.stdout
    assert "g4_i1_effective_number_c1.json" in completed.stdout


def test_i1_launcher_default_config_satisfies_frozen_protocol() -> None:
    validate_i1_config(load_experiment_config(CONFIG))
