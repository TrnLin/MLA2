from __future__ import annotations

import nbformat

from fashion.config import ROOT


def _notebook_source() -> str:
    notebook = nbformat.read(ROOT / "notebooks/02_task1_article_type.ipynb", as_version=4)
    nbformat.validate(notebook)
    return "\n".join(cell.source for cell in notebook.cells)


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
