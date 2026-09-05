from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fashion.task1.plotting import (
    write_task1_comparison_figure,
    write_task1_confusion_figure,
    write_task1_learning_curve_figure,
)


def test_comparison_figure_groups_five_folds_per_candidate(tmp_path: Path) -> None:
    """Grouping by preprocessing instead of candidate must fail this figure contract."""
    metrics = pd.DataFrame(
        {
            "candidate_id": ["task1_cnn_no_aug_unweighted_v1"] * 5
            + ["task1_cnn_no_aug_gentle_weighted_v1"] * 5,
            "preprocessing_id": ["no_augmentation"] * 10,
            "loss_id": ["unweighted"] * 5 + ["gentle_weighted"] * 5,
            "fold": list(range(5)) * 2,
            "macro_f1": np.linspace(0.1, 0.5, 10),
        }
    )

    output = write_task1_comparison_figure(metrics, output=tmp_path / "comparison.png")

    assert output == tmp_path / "comparison.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_learning_curve_figure_accepts_existing_history_schema(tmp_path: Path) -> None:
    """Omitting a history-series writer must prevent the report artifact."""
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [2.0, 1.0, 0.5],
            "validation_loss": [2.1, 1.2, 1.4],
            "macro_f1": [0.1, 0.3, 0.25],
        }
    )

    output = write_task1_learning_curve_figure(
        {"task1_cnn_no_aug_unweighted_v1": [history] * 5},
        output=tmp_path / "learning.png",
    )

    assert output == tmp_path / "learning.png"
    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_confusion_figure_writes_one_png(tmp_path: Path) -> None:
    """Removing the report figure must make the confusion artifact disappear."""
    predictions = pd.DataFrame({"true_index": [0, 1], "predicted_index": [0, 1]})

    output = write_task1_confusion_figure(
        predictions,
        [f"class-{index}" for index in range(124)],
        output=tmp_path / "confusion.png",
    )

    assert output == tmp_path / "confusion.png"
    assert output.is_file()
    assert output.stat().st_size > 0
