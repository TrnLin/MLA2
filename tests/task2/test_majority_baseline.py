from __future__ import annotations

import pandas as pd
import pytest

from fashion.data.dataset import get_cv_split, get_samples, iter_cv_folds, load_splits
from fashion.task2.baselines import evaluate_majority_fold, fit_training_fold_majority
from fashion.train.metrics import SEASON_LABELS, validate_oof


def _frame(ids: list[int], labels: list[str], *, fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ids,
            "partition": "development",
            "cv_fold": fold,
            "season": labels,
            "has_season_label": True,
        }
    )


def test_majority_fit_uses_training_counts_and_class_order_for_ties() -> None:
    training = _frame([1, 2, 3, 4], ["Fall", "Winter", "Fall", "Winter"], fold=1)

    model = fit_training_fold_majority(training)

    assert model.labels == SEASON_LABELS
    assert model.class_counts == (2, 0, 0, 2)
    assert model.class_probabilities == (0.5, 0.0, 0.0, 0.5)
    assert model.majority_label == "Fall"
    assert len(model.training_id_sha256) == 64


def test_majority_evaluation_does_not_refit_on_validation_labels() -> None:
    training = _frame([1, 2, 3], ["Summer", "Summer", "Fall"], fold=1)
    validation = _frame([10, 11], ["Winter", "Winter"], fold=0)

    result = evaluate_majority_fold(training, validation, validation_fold=0)

    assert result.model.majority_label == "Summer"
    assert result.oof["y_pred"].eq("Summer").all()
    assert result.oof["prob_Summer"].eq(2 / 3).all()
    assert result.metrics["accuracy"] == 0.0
    assert result.metrics["macro_f1"] == 0.0


def test_majority_baseline_rejects_protected_or_overlapping_rows() -> None:
    training = _frame([1, 2], ["Fall", "Summer"], fold=1)
    protected = _frame([3], ["Winter"], fold=0).assign(partition="holdout")
    with pytest.raises(ValueError, match="development rows only"):
        evaluate_majority_fold(training, protected, validation_fold=0)

    overlapping = _frame([2], ["Winter"], fold=0)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_majority_fold(training, overlapping, validation_fold=0)


def test_repository_majority_builds_exact_five_fold_oof_contract() -> None:
    splits = load_splits()
    valid_development = get_samples(splits, partition="development", target="season")
    fold_outputs = []
    fold_majorities = []
    for fold, training, validation in iter_cv_folds(splits):
        result = evaluate_majority_fold(
            training,
            validation,
            validation_fold=fold,
        )
        fold_outputs.append(result.oof)
        fold_majorities.append(result.model.majority_label)

    pooled = pd.concat(fold_outputs, ignore_index=True)
    audit = validate_oof(
        pooled,
        expected_ids=valid_development["id"],
        labels=SEASON_LABELS,
    )

    assert audit["row_count"] == 32_753
    assert fold_majorities == ["Summer"] * 5


def test_evaluation_rejects_rows_from_the_wrong_fold() -> None:
    splits = load_splits()
    training, validation = get_cv_split(splits, 0)

    with pytest.raises(ValueError, match="requested canonical fold"):
        evaluate_majority_fold(training, validation, validation_fold=1)
