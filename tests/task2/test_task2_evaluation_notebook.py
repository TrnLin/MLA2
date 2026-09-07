"""Contracts for Kai's locked Task 2 evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/06_task2_season_evaluation.ipynb"
PLAN = ROOT / "docs/plans/260908-task2-assessment3-evaluation/plan.md"


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
    )


def test_task2_owner_notebook_is_replay_safe_and_explicit() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = _source()
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]

    assert payload["metadata"]["owner"] == "Kai"
    assert len(code_cells) >= 10
    assert "EVALUATION_MODE = 'replay'" in source
    assert "require_group_unlock" in source
    assert "HOLDOUT_ROWS_EXPECTED = 5778" in source
    assert "B0" in source
    assert "B1" in source
    assert "I2" in source
    assert "train_test_split" not in source
    assert "load_splits_for_final_evaluation" not in source


def test_task2_plan_freezes_baseline_and_assessment3_link() -> None:
    source = PLAN.read_text(encoding="utf-8")

    assert "Use **B0 as the primary holdout baseline**." in source
    assert "B1 HOG + HSV LinearSVC" in source
    assert "Assessment 3 is directly connected to Assignment 2." in source
    assert "At least two additional peer-reviewed papers" in source
    assert "Success criteria" in source
