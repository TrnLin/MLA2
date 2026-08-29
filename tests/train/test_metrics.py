from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fashion.train.metrics import (
    OOFValidationError,
    cross_fit_temperature,
    fit_temperature,
    multiclass_metrics,
    temperature_scale_probabilities,
    validate_oof,
)

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


def test_multiclass_metrics_accepts_float32_softmax_without_warning() -> None:
    probabilities = np.array([[0.7310586, 0.26894143]], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        metrics = multiclass_metrics(
            ["Fall"],
            probabilities=probabilities,
            labels=("Fall", "Winter"),
        )

    assert metrics["nll"] > 0


def _overconfident_probabilities() -> tuple[np.ndarray, np.ndarray]:
    true = np.tile(np.asarray(LABELS, dtype=object), 10)
    probabilities = np.full((len(true), len(LABELS)), 0.01, dtype=float)
    true_indices = np.tile(np.arange(len(LABELS)), 10)
    wrong_indices = (true_indices + 1) % len(LABELS)
    for row, (true_index, wrong_index) in enumerate(zip(true_indices, wrong_indices, strict=True)):
        if row % 4 == 0:
            probabilities[row, wrong_index] = 0.94
            probabilities[row, true_index] = 0.03
        else:
            probabilities[row, true_index] = 0.94
            probabilities[row, wrong_index] = 0.03
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return true, probabilities


def test_temperature_scaling_softens_confidence_without_changing_argmax() -> None:
    _, probabilities = _overconfident_probabilities()

    scaled = temperature_scale_probabilities(probabilities, 2.0)

    np.testing.assert_allclose(scaled.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(scaled.argmax(axis=1), probabilities.argmax(axis=1))
    assert np.all(scaled.max(axis=1) < probabilities.max(axis=1))


def test_fit_temperature_reduces_nll_for_overconfident_predictions() -> None:
    true, probabilities = _overconfident_probabilities()
    before = multiclass_metrics(true, probabilities=probabilities, labels=LABELS)

    temperature = fit_temperature(probabilities, true, labels=LABELS)
    scaled = temperature_scale_probabilities(probabilities, temperature)
    after = multiclass_metrics(true, probabilities=scaled, labels=LABELS)

    assert temperature > 1.0
    assert after["nll"] < before["nll"]
    assert after["accuracy"] == before["accuracy"]


def test_cross_fit_temperature_never_fits_on_the_evaluation_fold(monkeypatch) -> None:
    true, probabilities = _overconfident_probabilities()
    folds = np.repeat(np.arange(5), 8)
    observed_fit_markers: list[set[float]] = []
    probabilities[:, 0] += np.repeat(np.arange(5), 8) * 1e-4
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    def recording_fit(matrix, targets, **kwargs):
        observed_fit_markers.append(set(np.round(matrix[:, 0], 8)))
        return 1.5

    monkeypatch.setattr("fashion.train.metrics.fit_temperature", recording_fit)
    calibrated, audit = cross_fit_temperature(probabilities, true, folds, labels=LABELS)

    assert len(audit) == 5
    assert audit["fit_fold_count"].eq(4).all()
    assert audit["calibration_rows"].eq(32).all()
    assert audit["evaluation_rows"].eq(8).all()
    for evaluation_fold, seen in enumerate(observed_fit_markers):
        held_out = set(np.round(probabilities[folds == evaluation_fold, 0], 8))
        assert seen.isdisjoint(held_out)
    np.testing.assert_array_equal(calibrated.argmax(axis=1), probabilities.argmax(axis=1))


def test_cross_fit_temperature_rejects_incomplete_fold_set() -> None:
    true, probabilities = _overconfident_probabilities()

    with pytest.raises(ValueError, match="expected_folds"):
        cross_fit_temperature(probabilities, true, np.zeros(len(true), dtype=int), labels=LABELS)
