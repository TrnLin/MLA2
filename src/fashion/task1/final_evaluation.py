"""Fixed, label-aware reports; callers must freeze predictions before calling these."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fashion.task1.evaluation import classification_metrics, per_class_metrics

PROBABILITY_COLUMNS = [f"prob_{i:03d}" for i in range(124)]


def checked_ids(values) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not (numeric == np.floor(numeric)).all():
        raise ValueError("IDs must be finite integers")
    if len(numeric) == 0 or len(np.unique(numeric)) != len(numeric):
        raise ValueError("IDs must be nonempty and unique")
    return numeric.astype(np.int64)


def blind_prediction_frame(ids, probabilities, class_names) -> pd.DataFrame:
    ids = checked_ids(ids)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(ids), 124) or len(set(class_names)) != 124:
        raise ValueError("predictions require the ordered 124-class map")
    frame = pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)
    frame.insert(0, "predicted_label", np.asarray(class_names)[probabilities.argmax(axis=1)])
    frame.insert(0, "id", ids)
    validate_predictions(frame, ids, class_names)
    return frame


def validate_predictions(frame, expected_ids, class_names) -> np.ndarray:
    if list(frame.columns) != ["id", "predicted_label", *PROBABILITY_COLUMNS]:
        raise ValueError("blind prediction columns differ from the contract")
    if not np.array_equal(checked_ids(frame.id), checked_ids(expected_ids)):
        raise ValueError("prediction IDs/order differ from the frozen image list")
    p = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("probability rows must sum to one")
    if len(class_names) != 124 or len(set(class_names)) != 124:
        raise ValueError("exactly 124 unique class names are required")
    if not np.array_equal(
        frame.predicted_label.astype(str), np.asarray(class_names)[p.argmax(axis=1)]
    ):
        raise ValueError("predicted labels do not match probability argmax")
    return p


def grouped_intervals(y, p, groups, *, replicates=10000, seed=2753) -> pd.DataFrame:
    """Percentile intervals using whole-family samples and explicit absent-class zeros."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(p).argmax(axis=1)
    _, group = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    if len(group) != len(y) or replicates < 1:
        raise ValueError("bootstrap requires aligned groups and positive replicates")
    # Only true/predicted/TP class counts are needed, avoiding G x 124 x 124 matrices.
    n_groups = int(group.max()) + 1
    counts = np.zeros((n_groups, 3 * 124 + 3), dtype=float)
    np.add.at(counts, (group, y), 1)
    np.add.at(counts, (group, 124 + pred), 1)
    correct = pred == y
    np.add.at(counts, (group[correct], 248 + y[correct]), 1)
    top5 = (np.argpartition(p, -5, axis=1)[:, -5:] == y[:, None]).any(axis=1)
    np.add.at(counts[:, -3], group, 1)
    np.add.at(counts[:, -2], group, correct)
    np.add.at(counts[:, -1], group, top5)
    random = np.random.default_rng(seed)
    samples = np.empty((replicates, 4))
    for i in range(replicates):
        weights = np.bincount(random.integers(0, n_groups, size=n_groups), minlength=n_groups)
        total = weights @ counts
        support, predicted, tp = total[:124], total[124:248], total[248:372]
        denom = support + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros(124), where=denom > 0)
        samples[i] = [
            f1.mean(),
            np.sum(f1 * support) / total[-3],
            total[-2] / total[-3],
            total[-1] / total[-3],
        ]
    names = ["macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy"]
    point = classification_metrics(y, p)
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "metric": names,
            "estimate": [point[n] for n in names],
            "lower_95": intervals[0],
            "upper_95": intervals[1],
            "families": n_groups,
            "replicates": replicates,
        }
    )


def score_tables(rows, probabilities, class_names, development_counts) -> dict[str, pd.DataFrame]:
    lookup = {label: i for i, label in enumerate(class_names)}
    if not rows.articleType.isin(lookup).all():
        raise ValueError("holdout contains missing/unknown articleType labels")
    y = rows.articleType.map(lookup).to_numpy(dtype=int)
    p = np.asarray(probabilities)
    pred = p.argmax(axis=1)
    per_class = per_class_metrics(y, p, class_names)
    per_class["development_count"] = (
        per_class.class_name.map(development_counts).fillna(0).astype(int)
    )
    detail = rows[["id", "articleType", "product_family_group", "mode"]].copy()
    detail["predicted_label"] = np.asarray(class_names)[pred]
    detail["confidence"] = p.max(axis=1)
    detail["correct"] = pred == y
    errors = detail.loc[~detail.correct].copy()
    pairs = errors.groupby(["articleType", "predicted_label"]).size().rename("errors").reset_index()
    pairs["true_class_support"] = pairs.articleType.map(rows.articleType.value_counts())
    pairs["share_of_true_class"] = pairs.errors / pairs.true_class_support
    pairs = pairs.sort_values("errors", ascending=False)
    counts = rows.articleType.map(development_counts).fillna(0).to_numpy()
    gray = rows["mode"].isin(["1", "L", "LA", "I", "F", "I;16"]).to_numpy()
    masks = {
        "development_count_1_20": (counts >= 1) & (counts <= 20),
        "development_count_21_100": (counts >= 21) & (counts <= 100),
        "development_count_over_100": counts > 100,
        "grayscale": gray,
        "colour": ~gray,
    }
    slices = [
        {"slice": name, "rows": int(mask.sum()), **classification_metrics(y[mask], p[mask])}
        for name, mask in masks.items()
        if mask.any()
    ]
    bins = np.minimum((p.max(axis=1) * 10).astype(int), 9)
    reliability = pd.DataFrame({"bin": bins, "confidence": p.max(axis=1), "correct": pred == y})
    reliability = (
        reliability.groupby("bin")
        .agg(
            rows=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    # Fixed coverage values, with stable ID-order tie handling. Diagnostic only.
    order = np.argsort(-p.max(axis=1), kind="stable")
    risk = [
        {
            "coverage": k / len(y),
            "rows": k,
            "error_rate": float(np.mean(pred[order[:k]] != y[order[:k]])),
        }
        for k in sorted(set(max(1, int(np.ceil(c * len(y)))) for c in np.linspace(0.1, 1, 10)))
    ]
    return {
        "metrics": pd.DataFrame([{"rows": len(y), **classification_metrics(y, p)}]),
        "per_class": per_class,
        "errors": errors,
        "confusion_pairs": pairs,
        "slices": pd.DataFrame(slices),
        "reliability": reliability,
        "risk_coverage": pd.DataFrame(risk),
    }
