from __future__ import annotations

import nbformat

from fashion.config import ROOT

NOTEBOOK = ROOT / "notebooks/02_task1_article_type.ipynb"


def _notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return notebook


def _notebook_source() -> str:
    return "\n".join(cell.source for cell in _notebook().cells)


def _notebook_code_source() -> str:
    return "\n".join(
        cell.source for cell in _notebook().cells if cell.cell_type == "code"
    )


def test_task1_notebook_is_evidence_led_and_run_all_safe() -> None:
    """Keep Notebook 02 as a narrative controller, not a second trainer."""
    source = _notebook_source()
    code_source = _notebook_code_source()

    expected_headings = [
        "## 1. Problem and output",
        "## 2. EDA evidence",
        "## 3. Safety contract",
        "## 4. Evaluation",
        "## 5. Candidate hypotheses",
        "## 6. Controlled preprocessing",
        "## 7. Classical baselines and scratch CNN controllers",
        "## 8. Learning-curve diagnosis",
        "## 9. Balanced class-weighted loss",
        "## 10. Combined five-fold and OOF comparison",
        "## 11. Weak-class/confusion analysis",
        "## 12. Development decision and Notebook 06 handoff",
    ]
    positions = [source.index(heading) for heading in expected_headings]
    assert positions == sorted(positions)

    required = (
        "build_task1_problem_profile",
        "build_task1_decision_evidence",
        "build_task1_weak_class_table",
        "build_task1_confusion_pairs",
        'RUN_MODE = "smoke"',
        'CLASSICAL_STAGE = "smoke"',
        'WEIGHTED_MODE = "smoke"',
        "run_task1_experiment",
        "run_task1_classical_experiment",
        "run_task1_weighted_experiment",
        "write_task1_learning_curve_figure",
        "0.5315 ± 0.0331",
        "0.5218 ± 0.0158",
        "0.4564 ± 0.0181",
        "practical stability decision",
        "task1_cnn_mild_aug_unweighted_v1",
        "results/runs.csv",
    )
    assert all(token in source for token in required)

    for forbidden in (
        "train_test_split",
        "pretrained=True",
        "optimizer.step()",
        "KNeighborsClassifier(",
        "LinearSVC(",
    ):
        assert forbidden not in code_source

    decision = source[source.index("## 12. Development decision and Notebook 06 handoff") :]
    assert "task1_cnn_mild_aug_unweighted_v1" in decision
    assert "practical stability decision" in decision
    assert "did not pass" in decision
    assert "not ready" not in decision.lower()


def test_task1_notebook_orders_diagnosis_before_weighted_experiment() -> None:
    notebook = _notebook()
    headings = [
        line.strip()
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    assert headings.index("## 8. Learning-curve diagnosis") < headings.index(
        "## 9. Balanced class-weighted loss"
    )


def test_task1_notebook_defaults_weighted_controller_to_smoke() -> None:
    source = _notebook_code_source()
    assert 'WEIGHTED_MODE = "smoke"' in source
    assert "run_task1_weighted_experiment(" in source
    assert "train_test_split" not in source
