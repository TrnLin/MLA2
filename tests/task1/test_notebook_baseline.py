from __future__ import annotations

import nbformat

from fashion.config import ROOT


def _notebook_source() -> str:
    notebook = nbformat.read(ROOT / "notebooks/02_task1_article_type.ipynb", as_version=4)
    nbformat.validate(notebook)
    return "\n".join(cell.source for cell in notebook.cells)


def test_task1_notebook_is_evidence_led_and_run_all_safe() -> None:
    """Keep Notebook 02 as a narrative controller, not a second trainer."""
    source = _notebook_source()

    expected_headings = [
        "## 1. Problem and output",
        "## 2. EDA evidence",
        "## 3. Safety contract",
        "## 4. Evaluation",
        "## 5. Candidate hypotheses",
        "## 6. Controlled preprocessing",
        "## 7. Run plan",
        "## 8. Run controllers",
        "## 9. Results",
        "## 10. Failure analysis",
        "## 11. Decision",
        "## 12. Final handoff",
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
        "run_task1_experiment",
        "run_task1_classical_experiment",
        "macro-F1",
        "124",
        "HOG",
        "scratch CNN",
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
        assert forbidden not in source

    decision = source[source.index("## 11. Decision") : source.index("## 12. Final handoff")]
    for word in ("hypothesis", "evidence", "passed", "failed", "not ready"):
        assert word in decision.lower()
