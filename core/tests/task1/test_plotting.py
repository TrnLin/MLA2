from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fashion.task1.plotting import (
    write_task1_comparison_figure,
    write_task1_confusion_example_figure,
    write_task1_confusion_figure,
    write_task1_confusion_pair_figure,
    write_task1_focused_confusion_figure,
    write_task1_learning_curve_figure,
)


def test_comparison_figure_groups_five_folds_per_candidate(tmp_path: Path) -> None:
    """Grouping by preprocessing instead of candidate must fail this figure contract."""
    metrics = pd.DataFrame(
        {
            "candidate_id": ["task1_cnn_mild_aug_unweighted_v1"] * 5
            + ["task1_cnn_mild_aug_balanced_weighted_v1"] * 5,
            "preprocessing_id": ["mild_augmentation"] * 10,
            "loss_id": ["unweighted"] * 5 + ["balanced_weighted"] * 5,
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


def _confusion_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "true_label": ["Shoe", "Shoe", "Shoe", "Top", "Top", "Top"],
            "predicted_label": ["Bag", "Bag", "Shoe", "Shirt", "Other class", "Top"],
        }
    )


def _confusion_detail() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [1, 2],
            "candidate_id": ["scratch-cnn", "scratch-cnn"],
            "true_label": ["Shoe", "Top"],
            "predicted_label": ["Bag", "Shirt"],
            "error_count": [2, 1],
            "true_support": [3, 3],
            "error_rate": [2 / 3, 1 / 3],
            "example_ids": ["1,2", "4"],
        }
    )


def test_confusion_pair_figure_writes_png(tmp_path: Path) -> None:
    output = write_task1_confusion_pair_figure(
        _confusion_detail(), output=tmp_path / "pairs.png"
    )

    assert output == tmp_path / "pairs.png"
    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_focused_confusion_figure_writes_png_with_other_predictions(tmp_path: Path) -> None:
    output = write_task1_focused_confusion_figure(
        _confusion_predictions(),
        _confusion_detail(),
        output=tmp_path / "focused.png",
        max_classes=4,
    )

    assert output == tmp_path / "focused.png"
    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_confusion_example_figure_uses_development_images(tmp_path: Path) -> None:
    for product_id in (1, 2, 4):
        Image.new("RGB", (12, 16), color=(product_id * 20, 30, 40)).save(
            tmp_path / f"{product_id}.jpg"
        )
    splits = pd.DataFrame(
        {
            "id": [1, 2, 4],
            "partition": ["development"] * 3,
            "path": ["1.jpg", "2.jpg", "4.jpg"],
        }
    )

    output = write_task1_confusion_example_figure(
        _confusion_predictions(),
        splits,
        _confusion_detail(),
        output=tmp_path / "examples.png",
        pair_limit=2,
        examples_per_pair=2,
        root=tmp_path,
    )

    assert output == tmp_path / "examples.png"
    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_confusion_example_figure_rejects_protected_image(tmp_path: Path) -> None:
    Image.new("RGB", (12, 16)).save(tmp_path / "1.jpg")
    splits = pd.DataFrame(
        {"id": [1], "partition": ["holdout"], "path": ["1.jpg"]}
    )
    detail = _confusion_detail().iloc[[0]].assign(example_ids="1")

    with np.testing.assert_raises_regex(ValueError, "development"):
        write_task1_confusion_example_figure(
            _confusion_predictions(),
            splits,
            detail,
            output=tmp_path / "examples.png",
            root=tmp_path,
        )
