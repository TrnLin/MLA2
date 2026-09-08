"""Contracts for Kai's completed Task 2 evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/06_task2_season_evaluation.ipynb"
HTML = ROOT / "results/notebooks/06_task2_season_evaluation.html"
PLAN = ROOT / "docs/plans/260908-task2-assessment3-evaluation/plan.md"
REPORT = ROOT / "docs/task2-season-execution-report.md"
NOTEBOOK_README = ROOT / "notebooks/README.md"


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_task2_owner_notebook_is_replay_safe_and_explicit() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = _source()
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    markdown_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "markdown"]

    assert payload["metadata"]["owner"] == "Kai"
    assert len(code_cells) == 16
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
    assert "Trace the one-way Task 2 file execution flow" in source
    assert "results/figures/task2/file_impact_flow.png" in source
    assert "full floating-point precision" in source
    assert (
        sum(
            line.startswith("## ")
            for cell in markdown_cells
            for line in "".join(cell["source"]).splitlines()
        )
        == 15
    )
    assert (
        sum(
            line.startswith("### ")
            for cell in markdown_cells
            for line in "".join(cell["source"]).splitlines()
        )
        == 16
    )


def test_task2_evaluation_notebook_is_executed_and_each_leaf_is_interpreted() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = payload["cells"]
    code_indices = [index for index, cell in enumerate(cells) if cell["cell_type"] == "code"]

    assert [cells[index]["execution_count"] for index in code_indices] == list(range(1, 17))
    for index in code_indices:
        assert cells[index]["outputs"]
        assert not any(output.get("output_type") == "error" for output in cells[index]["outputs"])
        assert index + 1 < len(cells)
        interpretation = cells[index + 1]
        assert interpretation["cell_type"] == "markdown"
        assert "".join(interpretation["source"]).startswith("**Interpretation.**")


def test_task2_evaluation_html_contains_the_final_replay() -> None:
    assert HTML.stat().st_size > 1_000_000
    source = HTML.read_text(encoding="utf-8")

    assert "Task 2 — Independent Season Evaluation" in source
    assert "I2 obtains 0.7534 macro-F1" in source
    assert "results/season_test_predictions.csv" in source
    assert "Trace-the-one-way-Task-2-file-execution-flow" in source
    assert "15. Ultimate judgement and artifact audit" in source


def test_task2_plan_freezes_baseline_and_assessment3_link() -> None:
    source = PLAN.read_text(encoding="utf-8")

    assert "Use **B0 as the primary holdout baseline**." in source
    assert "B1 HOG + HSV LinearSVC" in source
    assert "Assessment 3 is directly connected to Assignment 2." in source
    assert "At least two additional peer-reviewed papers" in source
    assert "Success criteria" in source


def test_task2_outputs_have_a_plain_language_glossary() -> None:
    source = _source()
    html = HTML.read_text(encoding="utf-8")

    for term in (
        "Plain-language output glossary",
        "Macro-F1",
        "Balanced accuracy",
        "95% bootstrap interval",
        "KDE",
        "NLL",
        "Brier score",
        "ECE",
        "Coverage",
        "Selective risk",
        "SHA-256",
    ):
        assert term in source
        assert term in html


def test_task2_completion_documents_match_the_final_artifacts() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")
    readme = NOTEBOOK_README.read_text(encoding="utf-8")

    assert "status: complete" in plan
    assert "5,778" in plan
    assert "5,829" in plan
    assert "0.7533847968563716" in plan
    assert "6fda3688" in plan
    assert "final evaluation remains locked" not in report
    assert "one independent\nholdout evaluation after group freeze" not in report
    assert "0.7533847968563716" in report
    assert "[0.5666774590701896, 0.6072286677590603]" in report
    assert "complete evaluation replay" in readme
