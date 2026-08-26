from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fashion.train.metrics import OOFValidationError, multiclass_metrics, validate_oof

LABELS = ("Fall", "Spring", "Summer", "Winter")


def _oof_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [11, 12, 13, 14],
            "y_true": list(LABELS),
            "y_pred": list(LABELS),
            "prob_Fall": [0.7, 0.1, 0.1, 0.1],
            "prob_Spring": [0.1, 0.7, 0.1, 0.1],
            "prob_Summer": [0.1, 0.1, 0.7, 0.1],
            "prob_Winter": [0.1, 0.1, 0.1, 0.7],
        }
    )


def test_validate_oof_accepts_exact_development_coverage() -> None:
    audit = validate_oof(
        _oof_frame(),
        expected_ids={11, 12, 13, 14},
        protected_ids={999},
    )

    assert audit["row_count"] == 4
    assert audit["unique_id_count"] == 4
    assert len(audit["id_set_sha256"]) == 64
    assert audit["labels"] == list(LABELS)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicates"),
        (lambda frame: frame.iloc[:-1], "coverage"),
        (lambda frame: frame.assign(prob_Fall=0.9), "sum to one"),
        (lambda frame: frame.assign(y_pred="Fall"), "argmax"),
    ],
)
def test_validate_oof_rejects_invalid_evidence(mutation, message: str) -> None:
    with pytest.raises(OOFValidationError, match=message):
        validate_oof(mutation(_oof_frame()), expected_ids={11, 12, 13, 14})


def test_validate_oof_rejects_protected_id() -> None:
    with pytest.raises(OOFValidationError, match="protected"):
        validate_oof(
            _oof_frame(),
            expected_ids={11, 12, 13, 14},
            protected_ids={14},
        )


def test_multiclass_metrics_returns_fixed_label_evidence() -> None:
    frame = _oof_frame()
    metrics = multiclass_metrics(
        frame["y_true"].to_numpy(),
        probabilities=frame.loc[:, [f"prob_{label}" for label in LABELS]].to_numpy(),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["Spring"]["support"] == 1
    assert metrics["confusion_matrix"] == np.eye(4, dtype=int).tolist()
    assert metrics["nll"] == pytest.approx(-np.log(0.7))
    assert metrics["brier"] == pytest.approx(0.12)
    assert metrics["ece"] == pytest.approx(0.3)


def test_multiclass_metrics_keeps_absent_classes_in_macro_f1() -> None:
    metrics = multiclass_metrics(
        ["Fall", "Fall"],
        probabilities=[[0.9, 0.05, 0.03, 0.02], [0.8, 0.1, 0.05, 0.05]],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(0.25)
    assert set(metrics["per_class"]) == set(LABELS)


def test_multiclass_metrics_rejects_wrong_probability_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        multiclass_metrics(["Fall"], probabilities=[[1.0, 0.0]])
