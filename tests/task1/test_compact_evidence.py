from __future__ import annotations

import pandas as pd

from fashion.config import SPLITS_CSV, TASK1_EVIDENCE_DIR
from fashion.task1.analysis import build_task1_confusion_detail


def test_compact_histories_cover_the_fifteen_saved_cnn_folds() -> None:
    fold_metrics = pd.read_csv(TASK1_EVIDENCE_DIR / "fold_metrics.csv")
    histories = pd.read_csv(TASK1_EVIDENCE_DIR / "cnn_learning_histories.csv")

    assert set(histories["run_id"]) == set(fold_metrics["run_id"])
    groups = histories.groupby(["candidate_id", "fold"], sort=False)
    assert len(groups) == 15
    assert set(groups.size()) == {20}
    assert set(histories["epoch"]) == set(range(1, 21))


def test_compact_selected_predictions_cover_each_development_product_once() -> None:
    predictions = pd.read_csv(TASK1_EVIDENCE_DIR / "selected_oof_predictions.csv")
    splits = pd.read_csv(SPLITS_CSV, usecols=["id", "partition"])
    development_ids = set(
        splits.loc[splits["partition"].eq("development"), "id"].astype(int)
    )

    assert not predictions["id"].duplicated().any()
    assert set(predictions["id"].astype(int)) == development_ids
    assert set(predictions["fold"].astype(int)) == set(range(5))
    assert not any(column.startswith("prob_") for column in predictions.columns)
    fold_metrics = pd.read_csv(TASK1_EVIDENCE_DIR / "fold_metrics.csv")
    selected_runs = fold_metrics.loc[
        fold_metrics["candidate_id"].eq("task1_cnn_no_aug_unweighted_v1"), "run_id"
    ]
    assert len(selected_runs) == 5
    assert set(predictions["run_id"]) == set(selected_runs)


def test_compact_predictions_rebuild_the_saved_confusion_detail() -> None:
    predictions = pd.read_csv(TASK1_EVIDENCE_DIR / "selected_oof_predictions.csv")
    expected = pd.read_csv(TASK1_EVIDENCE_DIR / "top_confusion_pairs.csv")

    actual = build_task1_confusion_detail(
        predictions,
        candidate_id="task1_cnn_no_aug_unweighted_v1",
        limit=10,
    )

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
