"""Leakage checks and fixed-label multiclass metrics for pooled OOF evidence."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from fashion.train.artifacts import canonical_sha256

SEASON_LABELS = ("Fall", "Spring", "Summer", "Winter")


class OOFValidationError(ValueError):
    """Raised when out-of-fold predictions are incomplete, duplicated, or unsafe."""


def _unique_labels(labels: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(str(label) for label in labels)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("labels must contain at least two unique values in class order")
    return ordered


def _probability_matrix(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != class_count:
        raise ValueError(
            "probabilities must be a non-empty matrix with "
            f"{class_count} columns, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities must all be finite")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("each probability row must sum to one")
    return matrix


def temperature_scale_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    temperature: float,
    *,
    probability_floor: float = 1e-12,
) -> np.ndarray:
    """Apply scalar temperature scaling to softmax probabilities."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix")
    matrix = _probability_matrix(matrix, class_count=matrix.shape[1])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    class_reciprocal = 1.0 / matrix.shape[1]
    if not 0 < probability_floor < class_reciprocal:
        raise ValueError("probability_floor must be below the reciprocal of class count")
    log_probabilities = np.log(np.clip(matrix, probability_floor, 1.0))
    centred = log_probabilities - log_probabilities.max(axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore"):
        scaled = np.exp(centred / float(temperature))
    row_sums = scaled.sum(axis=1, keepdims=True)
    if not np.isfinite(scaled).all() or not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        raise ValueError("temperature scaling produced invalid probabilities")
    scaled /= row_sums
    return scaled


def _target_indices(y_true: Sequence[str] | np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    true = np.asarray(y_true, dtype=object)
    if true.ndim != 1 or len(true) == 0:
        raise ValueError("y_true must be a non-empty one-dimensional sequence")
    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown = sorted(set(true.astype(str)) - set(labels))
    if unknown:
        raise ValueError(f"y_true contains unknown labels: {unknown}")
    if set(true.astype(str)) != set(labels):
        raise ValueError("temperature fitting requires every declared class")
    return np.fromiter(
        (label_to_index[str(label)] for label in true),
        dtype=np.int64,
        count=len(true),
    )


def _temperature_nll(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    temperature: float,
    *,
    probability_floor: float,
) -> float:
    scaled = temperature_scale_probabilities(
        probabilities,
        temperature,
        probability_floor=probability_floor,
    )
    selected = scaled[np.arange(len(target_indices)), target_indices]
    return float(-np.log(np.clip(selected, probability_floor, 1.0)).mean())


def fit_temperature(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    y_true: Sequence[str] | np.ndarray,
    *,
    labels: Sequence[str] = SEASON_LABELS,
    temperature_bounds: tuple[float, float] = (0.05, 10.0),
    probability_floor: float = 1e-12,
    optimizer_tolerance: float = 1e-8,
) -> float:
    """Fit one scalar by bounded NLL minimisation; fitting data must contain every class."""
    ordered_labels = _unique_labels(labels)
    matrix = _probability_matrix(probabilities, class_count=len(ordered_labels))
    targets = _target_indices(y_true, ordered_labels)
    if len(targets) != len(matrix):
        raise ValueError("probabilities and y_true row counts differ")
    lower, upper = (float(value) for value in temperature_bounds)
    if not np.isfinite((lower, upper)).all() or not 0 < lower < upper:
        raise ValueError("temperature_bounds must be finite and satisfy 0 < lower < upper")
    if not np.isfinite(optimizer_tolerance) or optimizer_tolerance <= 0:
        raise ValueError("optimizer_tolerance must be finite and positive")

    result = minimize_scalar(
        lambda log_temperature: _temperature_nll(
            matrix,
            targets,
            math.exp(float(log_temperature)),
            probability_floor=probability_floor,
        ),
        bounds=(math.log(lower), math.log(upper)),
        method="bounded",
        options={"xatol": optimizer_tolerance},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"temperature optimisation failed: {result.message}")
    return float(math.exp(float(result.x)))


def cross_fit_temperature(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    y_true: Sequence[str] | np.ndarray,
    fold_ids: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[str] = SEASON_LABELS,
    expected_folds: Sequence[int] = tuple(range(5)),
    temperature_bounds: tuple[float, float] = (0.05, 10.0),
    probability_floor: float = 1e-12,
    optimizer_tolerance: float = 1e-8,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit on other OOF folds and calibrate each row only with an unseen-fold scalar."""
    ordered_labels = _unique_labels(labels)
    matrix = _probability_matrix(probabilities, class_count=len(ordered_labels))
    true = np.asarray(y_true, dtype=object)
    folds = np.asarray(fold_ids)
    if true.ndim != 1 or folds.ndim != 1 or len(true) != len(matrix) or len(folds) != len(matrix):
        raise ValueError("probabilities, y_true, and fold_ids must have equal row counts")
    if folds.dtype.kind not in "iu" or not np.isfinite(folds.astype(float)).all():
        raise ValueError("fold_ids must contain finite integers")
    try:
        expected_values = np.asarray(expected_folds, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("expected_folds must contain finite integers") from error
    if (
        expected_values.ndim != 1
        or not np.isfinite(expected_values).all()
        or not np.equal(expected_values, np.floor(expected_values)).all()
    ):
        raise ValueError("expected_folds must contain finite integers")
    expected = tuple(int(fold) for fold in expected_values)
    if len(expected) < 2 or len(set(expected)) != len(expected):
        raise ValueError("expected_folds must contain at least two unique folds")
    if set(folds.astype(int)) != set(expected):
        raise ValueError("fold_ids do not match expected_folds")

    calibrated = np.empty_like(matrix)
    audit_rows: list[dict[str, Any]] = []
    original_argmax = matrix.argmax(axis=1)
    for evaluation_fold in expected:
        evaluation_mask = folds.astype(int) == evaluation_fold
        fit_mask = ~evaluation_mask
        fit_folds = tuple(fold for fold in expected if fold != evaluation_fold)
        temperature = fit_temperature(
            matrix[fit_mask],
            true[fit_mask],
            labels=ordered_labels,
            temperature_bounds=temperature_bounds,
            probability_floor=probability_floor,
            optimizer_tolerance=optimizer_tolerance,
        )
        calibrated[evaluation_mask] = temperature_scale_probabilities(
            matrix[evaluation_mask],
            temperature,
            probability_floor=probability_floor,
        )
        audit_rows.append(
            {
                "evaluation_fold": evaluation_fold,
                "fit_folds": "|".join(str(fold) for fold in fit_folds),
                "fit_fold_count": len(fit_folds),
                "calibration_rows": int(fit_mask.sum()),
                "evaluation_rows": int(evaluation_mask.sum()),
                "temperature": temperature,
                "fit_nll_before": _temperature_nll(
                    matrix[fit_mask],
                    _target_indices(true[fit_mask], ordered_labels),
                    1.0,
                    probability_floor=probability_floor,
                ),
                "fit_nll_after": _temperature_nll(
                    matrix[fit_mask],
                    _target_indices(true[fit_mask], ordered_labels),
                    temperature,
                    probability_floor=probability_floor,
                ),
            }
        )
    if not np.array_equal(original_argmax, calibrated.argmax(axis=1)):
        raise RuntimeError("temperature scaling changed class ranking")
    return calibrated, pd.DataFrame(audit_rows)


def _exact_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _label_column_slug(label: str) -> str:
    slug = "_".join(
        part
        for part in "".join(
            character.lower() if character.isalnum() else " " for character in label
        ).split()
    )
    if not slug:
        raise ValueError("labels must produce non-empty output column names")
    return slug


def _confusion_scores(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_positive = np.diagonal(confusion, axis1=1, axis2=2).astype(np.float64)
    true_support = confusion.sum(axis=2, dtype=np.int64).astype(np.float64)
    predicted_support = confusion.sum(axis=1, dtype=np.int64).astype(np.float64)
    denominator = true_support + predicted_support
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = confusion.sum(axis=(1, 2), dtype=np.int64).astype(np.float64)
    accuracy = np.divide(
        true_positive.sum(axis=1),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    return per_class_f1.mean(axis=1), accuracy, per_class_f1


def paired_group_bootstrap(
    y_true: Sequence[str] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    comparisons: Mapping[
        str,
        tuple[Sequence[str] | np.ndarray, Sequence[str] | np.ndarray],
    ],
    *,
    labels: Sequence[str] = SEASON_LABELS,
    replicates: int = 10_000,
    random_seed: int = 2753,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Resample whole groups and compare paired fixed-label predictions.

    ``y_true``, ``groups``, and every prediction vector are positional: callers must
    align them to the same audited row order before invoking this metric-only helper.
    Groups with identical joint confusion signatures are compressed before sampling.
    Sampling those signatures with their observed frequencies is distributionally exact,
    while avoiding one probability category per product family. One multiplicity draw is
    shared by every comparison, so candidate and seed-pair differences stay paired.
    """
    ordered_labels = _unique_labels(labels)
    label_slugs = tuple(_label_column_slug(label) for label in ordered_labels)
    if len(set(label_slugs)) != len(label_slugs):
        raise ValueError("labels must produce unique output column names")
    replicate_count = _exact_integer(replicates, name="replicates", minimum=1)
    seed = _exact_integer(random_seed, name="random_seed", minimum=0)
    draw_batch_size = _exact_integer(batch_size, name="batch_size", minimum=1)

    true = np.asarray(y_true, dtype=object)
    group_values = np.asarray(groups, dtype=object)
    if true.ndim != 1 or len(true) == 0:
        raise ValueError("y_true must be a non-empty one-dimensional sequence")
    if group_values.ndim != 1 or len(group_values) != len(true):
        raise ValueError("groups must be one-dimensional and match y_true length")
    if any(not isinstance(value, str) or not value.strip() for value in group_values):
        raise ValueError("groups must contain non-empty strings")

    allowed = set(ordered_labels)
    true_strings = np.asarray([str(value) for value in true], dtype=object)
    unknown_true = sorted(set(true_strings) - allowed)
    if unknown_true:
        raise ValueError(f"y_true contains unknown labels: {unknown_true}")
    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    true_indices = np.fromiter(
        (label_to_index[value] for value in true_strings),
        dtype=np.int64,
        count=len(true_strings),
    )

    if not isinstance(comparisons, Mapping) or not comparisons:
        raise ValueError("comparisons must be a non-empty mapping")
    raw_comparison_ids = list(comparisons)
    if any(
        not isinstance(comparison_id, str) or not comparison_id.strip()
        for comparison_id in raw_comparison_ids
    ):
        raise ValueError("comparison IDs must be non-empty strings")
    comparison_ids = sorted(raw_comparison_ids)
    prediction_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for comparison_id in comparison_ids:
        pair = comparisons[comparison_id]
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(
                f"comparison {comparison_id!r} must contain model A and model B predictions"
            )
        encoded_pair: list[np.ndarray] = []
        for model_name, predictions in zip(("model A", "model B"), pair, strict=True):
            predicted = np.asarray(predictions, dtype=object)
            if predicted.ndim != 1 or len(predicted) != len(true):
                raise ValueError(
                    f"{comparison_id!r} {model_name} predictions must match y_true length"
                )
            predicted_strings = np.asarray([str(value) for value in predicted], dtype=object)
            unknown_predictions = sorted(set(predicted_strings) - allowed)
            if unknown_predictions:
                raise ValueError(
                    f"{comparison_id!r} {model_name} contains unknown labels: {unknown_predictions}"
                )
            encoded_pair.append(
                np.fromiter(
                    (label_to_index[value] for value in predicted_strings),
                    dtype=np.int64,
                    count=len(predicted_strings),
                )
            )
        prediction_pairs.append((encoded_pair[0], encoded_pair[1]))

    ordered_groups = sorted(set(group_values.tolist()))
    group_count = len(ordered_groups)
    if group_count < 2:
        raise ValueError("groups must contain at least two unique values")
    group_to_index = {group: index for index, group in enumerate(ordered_groups)}
    group_indices = np.fromiter(
        (group_to_index[value] for value in group_values),
        dtype=np.int64,
        count=len(group_values),
    )

    class_count = len(ordered_labels)
    confusion_width = class_count * class_count
    pair_width = 2 * confusion_width
    group_signatures = np.zeros(
        (group_count, 1 + len(prediction_pairs) * pair_width),
        dtype=np.int64,
    )
    np.add.at(group_signatures[:, 0], group_indices, 1)
    for pair_index, (model_a, model_b) in enumerate(prediction_pairs):
        pair_offset = 1 + pair_index * pair_width
        model_a_cells = true_indices * class_count + model_a
        model_b_cells = true_indices * class_count + model_b
        np.add.at(
            group_signatures,
            (group_indices, pair_offset + model_a_cells),
            1,
        )
        np.add.at(
            group_signatures,
            (group_indices, pair_offset + confusion_width + model_b_cells),
            1,
        )

    signatures, signature_group_counts = np.unique(
        group_signatures,
        axis=0,
        return_counts=True,
    )
    signature_probabilities = signature_group_counts.astype(np.float64) / group_count
    signature_probabilities[-1] = 1.0 - signature_probabilities[:-1].sum()

    sampled_row_counts = np.empty(replicate_count, dtype=np.int64)
    macro_a = np.empty((replicate_count, len(prediction_pairs)), dtype=np.float64)
    macro_b = np.empty_like(macro_a)
    accuracy_a = np.empty_like(macro_a)
    accuracy_b = np.empty_like(macro_a)
    per_class_a = np.empty(
        (replicate_count, len(prediction_pairs), class_count),
        dtype=np.float64,
    )
    per_class_b = np.empty_like(per_class_a)

    generator = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, replicate_count, draw_batch_size):
        stop = min(start + draw_batch_size, replicate_count)
        multiplicities = generator.multinomial(
            group_count,
            signature_probabilities,
            size=stop - start,
        )
        totals = multiplicities @ signatures
        sampled_row_counts[start:stop] = totals[:, 0]
        for pair_index in range(len(prediction_pairs)):
            pair_offset = 1 + pair_index * pair_width
            model_a_confusion = totals[:, pair_offset : pair_offset + confusion_width].reshape(
                -1, class_count, class_count
            )
            model_b_confusion = totals[
                :,
                pair_offset + confusion_width : pair_offset + pair_width,
            ].reshape(-1, class_count, class_count)
            (
                macro_a[start:stop, pair_index],
                accuracy_a[start:stop, pair_index],
                per_class_a[start:stop, pair_index, :],
            ) = _confusion_scores(model_a_confusion)
            (
                macro_b[start:stop, pair_index],
                accuracy_b[start:stop, pair_index],
                per_class_b[start:stop, pair_index, :],
            ) = _confusion_scores(model_b_confusion)

    frames: list[pd.DataFrame] = []
    replicate_ids = np.arange(1, replicate_count + 1, dtype=np.int64)
    for pair_index, comparison_id in enumerate(comparison_ids):
        payload: dict[str, Any] = {
            "comparison_id": comparison_id,
            "replicate": replicate_ids,
            "sampled_group_count": np.full(replicate_count, group_count, dtype=np.int64),
            "sampled_row_count": sampled_row_counts,
            "model_a_macro_f1": macro_a[:, pair_index],
            "model_b_macro_f1": macro_b[:, pair_index],
            "b_minus_a_macro_f1": macro_b[:, pair_index] - macro_a[:, pair_index],
            "model_a_accuracy": accuracy_a[:, pair_index],
            "model_b_accuracy": accuracy_b[:, pair_index],
            "b_minus_a_accuracy": accuracy_b[:, pair_index] - accuracy_a[:, pair_index],
        }
        for label_index, slug in enumerate(label_slugs):
            payload[f"model_a_f1_{slug}"] = per_class_a[:, pair_index, label_index]
            payload[f"model_b_f1_{slug}"] = per_class_b[:, pair_index, label_index]
            payload[f"b_minus_a_f1_{slug}"] = (
                per_class_b[:, pair_index, label_index] - per_class_a[:, pair_index, label_index]
            )
        frames.append(pd.DataFrame(payload))
    return pd.concat(frames, ignore_index=True)


def validate_oof(
    frame: pd.DataFrame,
    *,
    expected_ids: Collection[Any],
    labels: Sequence[str] = SEASON_LABELS,
    protected_ids: Collection[Any] = (),
    expected_targets: Mapping[Any, str] | None = None,
    id_column: str = "id",
    true_column: str = "y_true",
    prediction_column: str = "y_pred",
    probability_prefix: str = "prob_",
) -> dict[str, Any]:
    """Validate exactly-once OOF coverage, labels, and probability consistency."""
    ordered_labels = _unique_labels(labels)
    probability_columns = tuple(f"{probability_prefix}{label}" for label in ordered_labels)
    required = {id_column, true_column, prediction_column, *probability_columns}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise OOFValidationError(f"OOF frame is missing columns: {missing_columns}")

    relevant = frame.loc[
        :, [id_column, true_column, prediction_column, *probability_columns]
    ].copy()
    if relevant.isna().any().any():
        raise OOFValidationError("OOF frame contains missing IDs, labels, or probabilities")

    ids = relevant[id_column].astype(str)
    duplicate_ids = sorted(ids.loc[ids.duplicated(keep=False)].unique())
    if duplicate_ids:
        raise OOFValidationError(f"OOF IDs must appear once; duplicates: {duplicate_ids[:10]}")

    actual_ids = set(ids)
    expected = {str(value) for value in expected_ids}
    protected = {str(value) for value in protected_ids}
    missing_ids = sorted(expected - actual_ids)
    extra_ids = sorted(actual_ids - expected)
    if missing_ids or extra_ids:
        raise OOFValidationError(
            "OOF ID coverage differs from the canonical development set; "
            f"missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )
    leaked_ids = sorted(actual_ids & protected)
    if leaked_ids:
        raise OOFValidationError(f"OOF frame contains protected IDs: {leaked_ids[:10]}")

    if expected_targets is not None:
        canonical_truth = {str(key): str(value) for key, value in expected_targets.items()}
        if len(canonical_truth) != len(expected_targets) or set(canonical_truth) != expected:
            raise OOFValidationError("canonical true-label IDs differ from the expected OOF IDs")
        recorded_truth = dict(zip(ids, relevant[true_column].astype(str), strict=True))
        mismatches = sorted(
            identifier
            for identifier in expected
            if recorded_truth[identifier] != canonical_truth[identifier]
        )
        if mismatches:
            raise OOFValidationError(
                "OOF y_true values disagree with canonical true labels; "
                f"mismatched_ids={mismatches[:10]}"
            )

    allowed = set(ordered_labels)
    for column in (true_column, prediction_column):
        unknown = sorted(set(relevant[column].astype(str)) - allowed)
        if unknown:
            raise OOFValidationError(f"{column} contains unknown labels: {unknown}")

    probabilities = relevant.loc[:, probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise OOFValidationError("OOF probabilities must all be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise OOFValidationError("OOF probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise OOFValidationError("each OOF probability row must sum to one")

    predicted = np.asarray(ordered_labels, dtype=object)[probabilities.argmax(axis=1)]
    recorded = relevant[prediction_column].astype(str).to_numpy()
    if not np.array_equal(predicted, recorded):
        mismatch_count = int(np.count_nonzero(predicted != recorded))
        raise OOFValidationError(
            f"recorded predictions disagree with probability argmax in {mismatch_count} rows"
        )

    return {
        "row_count": len(relevant),
        "expected_row_count": len(expected),
        "unique_id_count": len(actual_ids),
        "id_set_sha256": canonical_sha256(sorted(actual_ids)),
        "labels": list(ordered_labels),
        "protected_id_count": 0,
        "probability_sum_tolerance": 1e-5,
    }


def validate_oof_identity(
    frame: pd.DataFrame,
    *,
    run_id: str | None = None,
    experiment_id: str | None = None,
    fold: int | None = None,
    seed: int | None = None,
) -> None:
    """Require cached OOF identity columns to match the requested run exactly."""
    expected = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "fold": fold,
        "seed": seed,
    }
    for column, value in expected.items():
        if value is None:
            continue
        if column not in frame:
            raise OOFValidationError(f"OOF identity is missing column: {column}")
        observed = set(frame[column].astype(str))
        if observed != {str(value)}:
            raise OOFValidationError(
                f"OOF identity mismatch for {column}: "
                f"expected={value!r}, observed={sorted(observed)!r}"
            )


def multiclass_metrics(
    y_true: Sequence[str] | np.ndarray,
    *,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[str] = SEASON_LABELS,
    y_pred: Sequence[str] | np.ndarray | None = None,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Compute fixed-label discrimination and probability-quality metrics."""
    ordered_labels = _unique_labels(labels)
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least 2")
    true = np.asarray(y_true, dtype=object)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if true.ndim != 1 or len(true) == 0:
        raise ValueError("y_true must be a non-empty one-dimensional sequence")
    if probability_array.shape != (len(true), len(ordered_labels)):
        raise ValueError(
            "probabilities must have shape "
            f"({len(true)}, {len(ordered_labels)}), got {probability_array.shape}"
        )
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must all be finite")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    probability_sums = probability_array.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("each probability row must sum to one")
    probability_array = probability_array / probability_sums[:, np.newaxis]

    allowed = set(ordered_labels)
    unknown_true = sorted(set(true.astype(str)) - allowed)
    if unknown_true:
        raise ValueError(f"y_true contains unknown labels: {unknown_true}")
    if y_pred is None:
        predicted = np.asarray(ordered_labels, dtype=object)[probability_array.argmax(axis=1)]
    else:
        predicted = np.asarray(y_pred, dtype=object)
        if predicted.shape != true.shape:
            raise ValueError("y_pred must have the same shape as y_true")
        unknown_predicted = sorted(set(predicted.astype(str)) - allowed)
        if unknown_predicted:
            raise ValueError(f"y_pred contains unknown labels: {unknown_predicted}")

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=ordered_labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        labels=ordered_labels,
        average="weighted",
        zero_division=0,
    )

    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    true_indices = np.fromiter(
        (label_to_index[str(label)] for label in true),
        dtype=np.int64,
        count=len(true),
    )
    one_hot = np.eye(len(ordered_labels), dtype=np.float64)[true_indices]
    brier = float(np.mean(np.sum((probability_array - one_hot) ** 2, axis=1)))

    confidence = probability_array.max(axis=1)
    correct = predicted.astype(str) == true.astype(str)
    bin_indices = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
    ece = 0.0
    calibration_bins: list[dict[str, Any]] = []
    for index in range(ece_bins):
        mask = bin_indices == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        fraction = count / len(true)
        ece += fraction * abs(accuracy - mean_confidence)
        calibration_bins.append(
            {
                "bin": index,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return {
        "n_samples": len(true),
        "labels": list(ordered_labels),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "nll": float(log_loss(true, probability_array, labels=ordered_labels)),
        "brier": brier,
        "ece": float(ece),
        "ece_bins": calibration_bins,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(ordered_labels)
        },
        "confusion_matrix": confusion_matrix(
            true,
            predicted,
            labels=ordered_labels,
        )
        .astype(int)
        .tolist(),
    }


def _assert_metric_payload_matches(
    recorded: Any,
    recomputed: Any,
    *,
    path: str,
) -> None:
    if isinstance(recomputed, Mapping):
        if not isinstance(recorded, Mapping):
            raise OOFValidationError(f"cached metrics type mismatch at {path}")
        missing = sorted(set(recomputed) - set(recorded))
        if missing:
            raise OOFValidationError(f"cached metrics are missing keys at {path}: {missing}")
        for key, value in recomputed.items():
            _assert_metric_payload_matches(
                recorded[key],
                value,
                path=f"{path}.{key}",
            )
        return
    if isinstance(recomputed, Sequence) and not isinstance(recomputed, (str, bytes)):
        if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
            raise OOFValidationError(f"cached metrics type mismatch at {path}")
        if len(recorded) != len(recomputed):
            raise OOFValidationError(f"cached metrics length mismatch at {path}")
        for index, value in enumerate(recomputed):
            _assert_metric_payload_matches(
                recorded[index],
                value,
                path=f"{path}[{index}]",
            )
        return
    if (
        isinstance(recomputed, Real)
        and not isinstance(recomputed, bool)
        and isinstance(recorded, Real)
        and not isinstance(recorded, bool)
    ):
        if not math.isclose(
            float(recorded),
            float(recomputed),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise OOFValidationError(
                f"cached metrics disagree with OOF predictions at {path}: "
                f"recorded={recorded!r}, recomputed={recomputed!r}"
            )
        return
    if recorded != recomputed:
        raise OOFValidationError(
            f"cached metrics disagree with OOF predictions at {path}: "
            f"recorded={recorded!r}, recomputed={recomputed!r}"
        )


def validate_metrics_match_oof(
    frame: pd.DataFrame,
    recorded_metrics: Mapping[str, Any],
    *,
    labels: Sequence[str] = SEASON_LABELS,
    true_column: str = "y_true",
    prediction_column: str = "y_pred",
    probability_prefix: str = "prob_",
) -> dict[str, Any]:
    """Recompute metrics from cached probabilities and reject registry drift."""
    ordered_labels = _unique_labels(labels)
    probability_columns = [f"{probability_prefix}{label}" for label in ordered_labels]
    recomputed = multiclass_metrics(
        frame[true_column].astype(str),
        probabilities=frame[probability_columns].to_numpy(dtype=np.float64),
        labels=ordered_labels,
        y_pred=frame[prediction_column].astype(str),
    )
    _assert_metric_payload_matches(recorded_metrics, recomputed, path="metrics")
    return recomputed
