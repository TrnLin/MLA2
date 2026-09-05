from __future__ import annotations

import pandas as pd
import pytest

from fashion.task1.losses import (
    TASK1_GENTLE_WEIGHTED_LOSS,
    TASK1_UNWEIGHTED_LOSS,
    build_task1_loss_weights,
)


def _training_rows() -> pd.DataFrame:
    labels = ["a"] + ["b"] * 4 + ["c"] * 9
    return pd.DataFrame(
        {
            "id": range(1, 15),
            "partition": ["development"] * 14,
            "cv_fold": [1] * 14,
            "articleType": labels,
        }
    )


def test_gentle_weights_use_sqrt_mean_normalisation_and_absent_zero() -> None:
    result = build_task1_loss_weights(
        _training_rows(),
        {"a": 0, "b": 1, "c": 2, "d": 3},
        validation_fold=0,
        config=TASK1_GENTLE_WEIGHTED_LOSS,
    )

    assert result.class_counts == (1, 4, 9, 0)
    assert result.class_weights == pytest.approx((1.6363636, 0.8181818, 0.5454545, 0.0))


def test_unweighted_loss_returns_no_training_tensor() -> None:
    result = build_task1_loss_weights(
        _training_rows(),
        {"a": 0, "b": 1, "c": 2, "d": 3},
        validation_fold=0,
        config=TASK1_UNWEIGHTED_LOSS,
    )
    assert result.tensor is None


@pytest.mark.parametrize("mutation", [{"partition": "holdout"}, {"cv_fold": 0}])
def test_weights_reject_rows_outside_the_fold_training_set(mutation: dict[str, object]) -> None:
    rows = _training_rows()
    for column, value in mutation.items():
        rows.loc[rows.index[0], column] = value
    with pytest.raises(ValueError, match="development training rows"):
        build_task1_loss_weights(
            rows,
            {"a": 0, "b": 1, "c": 2, "d": 3},
            validation_fold=0,
            config=TASK1_GENTLE_WEIGHTED_LOSS,
        )
