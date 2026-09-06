from __future__ import annotations

import ast
import subprocess
import sys

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/build_task2_gradcam_evidence.py"


def test_gradcam_launcher_is_post_training_and_windows_safe() -> None:
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
    assert "build_gradcam_failure_evidence" in source
    assert "run_or_load_experiment" not in source
    assert "train_fold" not in source


def test_gradcam_launcher_help_explains_oof_and_holdout_boundary() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Grad-CAM" in completed.stdout
    assert "OOF" in completed.stdout
    assert "holdout" in completed.stdout
