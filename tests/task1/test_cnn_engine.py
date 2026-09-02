from __future__ import annotations

from pathlib import Path

import torch
from test_training import _label_map, _splits_with_images

from fashion.task1.cnn_engine import train_task1_cnn
from fashion.task1.dataset import get_task1_fold_rows
from fashion.task1.preprocessing import TASK1_CONTROL_PREPROCESSING, fit_task1_normalization
from fashion.task1.training import Task1TrainConfig


def test_train_task1_cnn_returns_one_epoch_fixed_class_evidence(tmp_path: Path) -> None:
    """Engine extraction must retain one-epoch evidence and fixed 124-class output."""
    splits = _splits_with_images(tmp_path)
    label_map = _label_map()
    label_to_index = label_map["label_to_index"]
    class_names = label_map["classes"]
    training_rows, validation_rows = get_task1_fold_rows(splits, validation_fold=0)
    normalization = fit_task1_normalization(
        training_rows,
        validation_fold=0,
        root=tmp_path,
        config=TASK1_CONTROL_PREPROCESSING,
    )

    result = train_task1_cnn(
        training_rows,
        validation_rows,
        label_to_index,
        class_names,
        normalization=normalization,
        preprocessing=TASK1_CONTROL_PREPROCESSING,
        config=Task1TrainConfig.smoke(),
        root=tmp_path,
        device=torch.device("cpu"),
    )

    assert result.best_epoch == 1
    assert list(result.history.columns) == [
        "epoch",
        "train_loss",
        "macro_f1",
        "weighted_f1",
        "top1_accuracy",
        "top5_accuracy",
        "validation_loss",
    ]
    assert len(result.predictions.filter(like="prob_", axis=1).columns) == 124
    assert result.metrics["macro_f1"] == result.history.iloc[0]["macro_f1"]
