"""Contracts for Kai's completed Task 2 evaluation notebook."""

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
    markdown_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "markdown"]

    assert payload["metadata"]["owner"] == "Kai"
    assert len(code_cells) == 15
    assert 'EVALUATION_MODE = "artifact_replay"' in source
    assert "HOLDOUT_ROWS_EXPECTED = 5778" in source
    assert "B0" in source
    assert "B1" in source
    assert "I2" in source
    assert "train_test_split" not in source
    assert "load_splits_for_final_evaluation" not in source
    assert "torch.load" not in source
    assert "fit_temperature" not in source
    assert "NotImplementedError" not in source
    assert "TODO" not in source
    assert "awaiting" not in source.lower()
    assert "holdout_scorecard.csv" in source
    assert "holdout_bootstrap_intervals.csv" in source
    assert "holdout_calibration_summary.csv" in source
    assert "holdout_slice_metrics.csv" in source
    assert "holdout_robustness_metrics.csv" in source
    assert "season_test_predictions.csv" in source
    assert sum(
        line.startswith("## ")
        for cell in markdown_cells
        for line in "".join(cell["source"]).splitlines()
    ) == 15
    assert sum(
        line.startswith("### ")
        for cell in markdown_cells
        for line in "".join(cell["source"]).splitlines()
    ) == 15


def test_task2_evaluation_notebook_is_executed_and_each_leaf_is_interpreted() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = payload["cells"]
    code_indices = [index for index, cell in enumerate(cells) if cell["cell_type"] == "code"]

    assert [cells[index]["execution_count"] for index in code_indices] == list(range(1, 16))
    for index in code_indices:
        assert cells[index]["outputs"]
        assert not any(output.get("output_type") == "error" for output in cells[index]["outputs"])
        assert index + 1 < len(cells)
        interpretation = cells[index + 1]
        assert interpretation["cell_type"] == "markdown"
        assert "".join(interpretation["source"]).startswith("**Interpretation.**")


def test_task2_plan_freezes_baseline_and_assessment3_link() -> None:
    source = PLAN.read_text(encoding="utf-8")

    assert "Use **B0 as the primary holdout baseline**." in source
    assert "B1 HOG + HSV LinearSVC" in source
    assert "Assessment 3 is directly connected to Assignment 2." in source
    assert "At least two additional peer-reviewed papers" in source
    assert "Success criteria" in source
