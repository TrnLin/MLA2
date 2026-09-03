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
    paired_group_bootstrap,
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


def test_temperature_scaling_matches_direct_logit_scaling() -> None:
    logits = np.array([[2.0, -0.5, 1.2, 0.1], [-1.0, 0.7, 2.4, 0.3]])
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    temperature = 1.7
    expected = np.exp(logits / temperature - (logits / temperature).max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)

    actual = temperature_scale_probabilities(probabilities, temperature)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-16)


def test_temperature_scaling_rejects_floor_that_can_flatten_class_ranking() -> None:
    probabilities = np.array([[0.1, 0.4, 0.3, 0.2]])

    with pytest.raises(ValueError, match="reciprocal"):
        temperature_scale_probabilities(
            probabilities,
            1.0,
            probability_floor=0.26,
        )


def test_temperature_scaling_handles_subnormal_positive_temperature() -> None:
    probabilities = np.array([[0.1, 0.4, 0.3, 0.2]])

    scaled = temperature_scale_probabilities(probabilities, np.nextafter(0.0, 1.0))

    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled.sum(axis=1), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(scaled.argmax(axis=1), probabilities.argmax(axis=1))


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
    fold_targets = [
        np.roll(np.tile(np.asarray(LABELS, dtype=object), 2), fold) for fold in range(5)
    ]
    true = np.concatenate(fold_targets)
    observed_fit_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    probabilities[:, 0] += np.repeat(np.arange(5), 8) * 1e-4
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    def recording_fit(matrix, targets, **kwargs):
        observed_fit_inputs.append((matrix.copy(), np.asarray(targets).copy()))
        return 1.5

    monkeypatch.setattr("fashion.train.metrics.fit_temperature", recording_fit)
    calibrated, audit = cross_fit_temperature(probabilities, true, folds, labels=LABELS)

    assert len(audit) == 5
    assert audit["fit_fold_count"].eq(4).all()
    assert audit["calibration_rows"].eq(32).all()
    assert audit["evaluation_rows"].eq(8).all()
    for evaluation_fold, (fit_probabilities, fit_targets) in enumerate(observed_fit_inputs):
        fit_mask = folds != evaluation_fold
        np.testing.assert_array_equal(fit_probabilities, probabilities[fit_mask])
        np.testing.assert_array_equal(fit_targets, true[fit_mask])
        assert audit.loc[evaluation_fold, "fit_folds"] == "|".join(
            str(fold) for fold in range(5) if fold != evaluation_fold
        )
    np.testing.assert_array_equal(calibrated.argmax(axis=1), probabilities.argmax(axis=1))


def test_cross_fit_temperature_rejects_incomplete_fold_set() -> None:
    true, probabilities = _overconfident_probabilities()

    with pytest.raises(ValueError, match="expected_folds"):
        cross_fit_temperature(probabilities, true, np.zeros(len(true), dtype=int), labels=LABELS)


def test_cross_fit_temperature_rejects_fractional_expected_folds() -> None:
    true, probabilities = _overconfident_probabilities()
    folds = np.repeat(np.arange(5), 8)

    with pytest.raises(ValueError, match="integers"):
        cross_fit_temperature(
            probabilities,
            true,
            folds,
            labels=LABELS,
            expected_folds=(0.0, 1.0, 2.0, 3.0, 4.5),
        )


def test_fit_temperature_rejects_non_finite_bounds() -> None:
    true, probabilities = _overconfident_probabilities()

    with pytest.raises(ValueError, match="finite"):
        fit_temperature(
            probabilities,
            true,
            labels=LABELS,
            temperature_bounds=(0.05, np.inf),
        )


def _bootstrap_inputs():
    true = np.asarray(["Fall", "Fall", "Spring", "Summer", "Winter"], dtype=object)
    groups = np.asarray(["family-b", "family-b", "family-a", "family-c", "family-c"])
    model_a = np.asarray(["Fall", "Spring", "Spring", "Summer", "Fall"], dtype=object)
    model_b = np.asarray(["Fall", "Fall", "Spring", "Winter", "Winter"], dtype=object)
    return true, groups, model_a, model_b


def test_paired_group_bootstrap_is_deterministic_and_batch_invariant() -> None:
    true, groups, model_a, model_b = _bootstrap_inputs()
    comparisons = {"primary": (model_a, model_b)}

    first = paired_group_bootstrap(
        true,
        groups,
        comparisons,
        labels=LABELS,
        replicates=37,
        random_seed=19,
        batch_size=1,
    )
    second = paired_group_bootstrap(
        true,
        groups,
        comparisons,
        labels=LABELS,
        replicates=37,
        random_seed=19,
        batch_size=7,
    )

    pd.testing.assert_frame_equal(first, second)


def test_paired_group_bootstrap_sorts_comparisons_and_reuses_draws() -> None:
    true, groups, model_a, model_b = _bootstrap_inputs()
    forward = {
        "seed-2753": (model_a, model_b),
        "seed-2026": (model_a.copy(), model_b.copy()),
    }
    reversed_mapping = dict(reversed(list(forward.items())))

    first = paired_group_bootstrap(
        true,
        groups,
        forward,
        labels=LABELS,
        replicates=25,
        random_seed=7,
        batch_size=4,
    )
    second = paired_group_bootstrap(
        true,
        groups,
        reversed_mapping,
        labels=LABELS,
        replicates=25,
        random_seed=7,
        batch_size=4,
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first["comparison_id"].drop_duplicates()) == ["seed-2026", "seed-2753"]
    by_comparison = {
        comparison_id: rows.reset_index(drop=True)
        for comparison_id, rows in first.groupby("comparison_id", sort=False)
    }
    metric_columns = [column for column in first if column != "comparison_id"]
    pd.testing.assert_frame_equal(
        by_comparison["seed-2026"][metric_columns],
        by_comparison["seed-2753"][metric_columns],
    )


def test_paired_group_bootstrap_resamples_whole_groups() -> None:
    true = np.asarray(["Fall", "Spring", "Winter"], dtype=object)
    groups = np.asarray(["large", "large", "small"], dtype=object)
    model_a = np.asarray(["Fall", "Fall", "Winter"], dtype=object)
    model_b = np.asarray(["Fall", "Spring", "Fall"], dtype=object)

    result = paired_group_bootstrap(
        true,
        groups,
        {"pair": (model_a, model_b)},
        labels=LABELS,
        replicates=500,
        random_seed=11,
        batch_size=17,
    )

    assert set(result["sampled_row_count"]) == {2, 3, 4}
    assert result["sampled_group_count"].eq(2).all()

    def direct_scores(row_indices: list[int]) -> tuple[float, ...]:
        selected_true = true[row_indices]
        values: list[float] = []
        for predictions in (model_a[row_indices], model_b[row_indices]):
            class_f1 = []
            for label in LABELS:
                true_positive = np.count_nonzero((selected_true == label) & (predictions == label))
                false_positive = np.count_nonzero((selected_true != label) & (predictions == label))
                false_negative = np.count_nonzero((selected_true == label) & (predictions != label))
                denominator = 2 * true_positive + false_positive + false_negative
                class_f1.append(2 * true_positive / denominator if denominator else 0.0)
            values.extend((float(np.mean(class_f1)), float(np.mean(selected_true == predictions))))
        return (
            values[0],
            values[2],
            values[2] - values[0],
            values[1],
            values[3],
            values[3] - values[1],
        )

    exhaustive_scores = []
    for large_count in range(3):
        row_indices = [0, 1] * large_count + [2] * (2 - large_count)
        exhaustive_scores.append(direct_scores(row_indices))
    metric_columns = [
        "model_a_macro_f1",
        "model_b_macro_f1",
        "b_minus_a_macro_f1",
        "model_a_accuracy",
        "model_b_accuracy",
        "b_minus_a_accuracy",
    ]
    expected = {tuple(round(float(value), 12) for value in row) for row in exhaustive_scores}
    observed = {
        tuple(round(float(row[column]), 12) for column in metric_columns)
        for _, row in result.iterrows()
    }
    assert observed == expected


def test_paired_group_bootstrap_keeps_absent_classes_with_zero_f1() -> None:
    result = paired_group_bootstrap(
        ["Fall", "Fall", "Spring"],
        ["a", "b", "c"],
        {"pair": (["Fall", "Fall", "Spring"], ["Fall", "Fall", "Spring"])},
        labels=LABELS,
        replicates=12,
        random_seed=3,
    )

    assert result["model_a_f1_summer"].eq(0.0).all()
    assert result["model_b_f1_winter"].eq(0.0).all()
    assert result["model_a_macro_f1"].le(0.5).all()
    assert not result.isna().any().any()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"comparisons": {}}, "non-empty mapping"),
        ({"comparisons": {1: (["Fall"] * 3, ["Fall"] * 3)}}, "comparison IDs"),
        ({"comparisons": {"pair": ("Fall",)}}, "model A and model B"),
        ({"groups": ["a"]}, "match y_true length"),
        ({"groups": ["same", "same", "same"]}, "at least two unique"),
        ({"groups": ["a", "", "b"]}, "non-empty strings"),
        ({"groups": ["a", None, "b"]}, "non-empty strings"),
        ({"y_true": ["Fall", "Unknown", "Spring"]}, "unknown labels"),
        (
            {"comparisons": {"pair": (["Fall"], ["Fall", "Fall", "Spring"])}},
            "match y_true length",
        ),
        (
            {"comparisons": {"pair": (["Fall", "Fall", "Spring"], ["Bad"] * 3)}},
            "unknown labels",
        ),
        ({"replicates": 1.5}, "positive integer"),
        ({"replicates": True}, "positive integer"),
        ({"batch_size": 0}, "positive integer"),
        ({"random_seed": -1}, "non-negative integer"),
        ({"random_seed": 2.0}, "non-negative integer"),
    ],
)
def test_paired_group_bootstrap_rejects_invalid_inputs(overrides, message: str) -> None:
    inputs = {
        "y_true": ["Fall", "Fall", "Spring"],
        "groups": ["a", "b", "c"],
        "comparisons": {"pair": (["Fall", "Fall", "Spring"], ["Fall", "Spring", "Spring"])},
        "labels": LABELS,
        "replicates": 4,
        "random_seed": 0,
        "batch_size": 2,
    }
    inputs.update(overrides)

    with pytest.raises(ValueError, match=message):
        paired_group_bootstrap(**inputs)


def test_paired_group_bootstrap_returns_one_row_per_pair_and_replicate() -> None:
    true, groups, model_a, model_b = _bootstrap_inputs()

    result = paired_group_bootstrap(
        true,
        groups,
        {"primary": (model_a, model_b), "stability": (model_b, model_a)},
        labels=LABELS,
        replicates=13,
        random_seed=5,
        batch_size=6,
    )

    assert len(result) == 26
    assert result.groupby("comparison_id")["replicate"].nunique().eq(13).all()
    assert not result.isna().any().any()

    primary = result.loc[result["comparison_id"] == "primary"].reset_index(drop=True)
    stability = result.loc[result["comparison_id"] == "stability"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        primary["sampled_row_count"],
        stability["sampled_row_count"],
        check_names=False,
    )
    metric_suffixes = [
        "macro_f1",
        "accuracy",
        *(f"f1_{label.lower()}" for label in LABELS),
    ]
    for suffix in metric_suffixes:
        np.testing.assert_allclose(
            stability[f"model_a_{suffix}"],
            primary[f"model_b_{suffix}"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            stability[f"model_b_{suffix}"],
            primary[f"model_a_{suffix}"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            stability[f"b_minus_a_{suffix}"],
            -primary[f"b_minus_a_{suffix}"],
            rtol=0.0,
            atol=0.0,
        )
