from __future__ import annotations

import nbformat

from fashion.config import ROOT


def _notebook_source() -> str:
    notebook = nbformat.read(ROOT / "notebooks/02_task1_article_type.ipynb", as_version=4)
    nbformat.validate(notebook)
    return "\n".join(cell.source for cell in notebook.cells)


def _notebook_cells() -> list[dict[str, object]]:
    notebook = nbformat.read(ROOT / "notebooks/02_task1_article_type.ipynb", as_version=4)
    nbformat.validate(notebook)
    return list(notebook.cells)


def test_task1_notebook_controls_registered_cnn_experiments() -> None:
    """Catch a notebook that stops describing the CNN experiment controller."""
    source = _notebook_source()

    required = (
        'RUN_MODE = "smoke"',
        "run_task1_experiment",
        "Task1SmallCNN",
        "macro-F1",
        "results/runs.csv",
        "task1_rgb_60x80_no_aug_v1",
        "task1_rgb_60x80_mild_aug_v1",
    )

    assert all(token in source for token in required)
    required_classical = (
        'CLASSICAL_STAGE = "smoke"',
        "run_task1_classical_experiment",
        "TASK1_HOG_COARSE",
        "TASK1_HOG_FINE",
        "KNeighborsClassifier",
        "LinearSVC",
        "16 x 16",
        "10 x 10",
        "classical_tuning.csv",
        "results/runs.csv",
    )

    assert all(token in source for token in required_classical)
    assert "shared Task 1 framework owns the metrics for every candidate family" in source
    assert "train_test_split" not in source
    assert "pretrained=True" not in source
    assert "optimizer.step()" not in source
    assert "StandardScaler" not in source
    assert "PCA(" not in source

    cells = _notebook_cells()
    controller = next(
        cell for cell in cells
        if cell.cell_type == "code" and "run_task1_classical_experiment" in cell.source
    )
    assert 'CLASSICAL_STAGE = "smoke"' in controller.source
    assert "from fashion.task1 import" in controller.source
    assert "TASK1_HOG_COARSE" in controller.source
    assert "TASK1_HOG_FINE" in controller.source
    assert "display(classic_experiment.oof_metrics)" in controller.source

    candidate_narrative = next(
        cell for cell in cells
        if cell.cell_type == "markdown" and "## 9. Candidate model comparisons" in cell.source
    )
    assert (
        "shared Task 1 framework owns the metrics for every candidate family"
        in candidate_narrative.source
    )
    assert "KNeighborsClassifier" in candidate_narrative.source
    assert "LinearSVC" in candidate_narrative.source
    assert "scikit-image" in candidate_narrative.source
    assert "scikit-learn" in candidate_narrative.source

    matrix_narrative = next(
        cell for cell in cells
        if cell.cell_type == "markdown" and "## 10. Experiment matrix" in cell.source
    )
    assert "Classic smoke" in matrix_narrative.source
    assert "Classic HOG selection" in matrix_narrative.source
    assert "Classic tuning" in matrix_narrative.source
    assert "Classic final evidence" in matrix_narrative.source
