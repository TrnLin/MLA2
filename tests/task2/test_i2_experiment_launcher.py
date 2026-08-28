from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from fashion.config import ROOT
from fashion.task2.multitask import load_i2_config

LAUNCHER = ROOT / "scripts/run_task2_i2_experiments.py"
CONFIGS = (
    ROOT / "configs/task2/g4_i2_article_type_lambda_0_1_c1.json",
    ROOT / "configs/task2/g4_i2_article_type_lambda_0_3_c1.json",
)


def test_i2_launcher_has_windows_safe_main_guard_and_matrix_runner() -> None:
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
    assert "load_i2_config" in imports
    assert "run_i2_matrix" in imports


def test_i2_launcher_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=Path(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "ArticleType auxiliary-loss" in completed.stdout
    assert "g4_i2_article_type_lambda_0_1_c1.json" in completed.stdout
    assert "g4_i2_article_type_lambda_0_3_c1.json" in completed.stdout


def test_i2_launcher_defaults_are_the_two_frozen_weights() -> None:
    configs = [load_i2_config(path) for path in CONFIGS]

    assert [config.auxiliary.loss_weight for config in configs] == [0.1, 0.3]
    assert all(config.folds == tuple(range(5)) for config in configs)
    assert all(config.seeds == (2753,) for config in configs)
