"""Read-only, fixed-scope checks for the G2 and U2 decisions.

These checks never train a model, choose a threshold, or replace the canonical split.
The family bootstrap keeps related rows together and retains every fixed class.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fashion.train.metrics import classification_metrics

CORE_CORRUPTIONS = ("jpeg_75", "brightness_085", "brightness_115", "translation_003", "grayscale")
DECISION_BOOTSTRAP_REPETITIONS = 10_000
DECISION_BOOTSTRAP_SEED = 2753


def probability_columns(classes: Sequence[str]) -> list[str]:
    return [f"probability_{index}_{name}" for index, name in enumerate(classes)]


def _integers(values: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integers")
    return numeric.astype(np.int64)


def validate_oof(
    predictions: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    target: str,
    classes: Sequence[str],
    run_ids_by_fold: Mapping[int, str] | None = None,
    allow_legacy_na: bool = False,
) -> pd.DataFrame:
    """Require exact IDs, canonical labels/folds/families, and honest probabilities.

    An old CSV writer lost literal NA strings in some E2 label columns. Repair
    only that representation, and only after its numeric index agrees with the
    canonical label. Never repair a missing prediction, ID, or probability.
    """
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("fixed classes must be non-empty and unique")
    result = predictions.copy()
    required = {
        "id",
        "cv_fold",
        "product_family_group",
        "path",
        "run_id",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_label",
        "confidence",
        *probability_columns(classes),
    }
    if required.difference(result.columns):
        raise ValueError(f"OOF lacks columns: {sorted(required.difference(result.columns))}")
    if set(c for c in result if c.startswith("probability_")) != set(probability_columns(classes)):
        raise ValueError("OOF probability columns disagree with the fixed classes")
    for frame in (result, expected):
        if frame.empty or frame["id"].duplicated().any():
            raise ValueError("OOF and expected scope must be non-empty with unique IDs")
    for column in ("id", "cv_fold", "true_index", "predicted_index"):
        result[column] = _integers(result[column], column)
    if result["id"].duplicated().any():
        raise ValueError("OOF contains duplicate numeric IDs")
    canonical = expected.copy()
    canonical["id"] = _integers(canonical["id"], "expected id")
    if set(result["id"]) != set(canonical["id"]):
        raise ValueError("OOF IDs do not exactly cover the canonical evaluation scope")
    result = result.sort_values("id").reset_index(drop=True)
    canonical = canonical.set_index("id").loc[result["id"]].reset_index()
    if "partition" in canonical and not canonical["partition"].eq("development").all():
        raise ValueError("only development rows belong in a Task 3 decision")
    for column in ("cv_fold", "product_family_group", "path"):
        left, right = result[column], canonical[column]
        matches = (
            left.eq(_integers(right, "expected cv_fold"))
            if column == "cv_fold"
            else left.astype(str).eq(right.astype(str))
        )
        if not matches.all():
            raise ValueError(f"OOF {column} disagrees with the canonical split")
    if result.groupby("product_family_group")["cv_fold"].nunique().gt(1).any():
        raise ValueError("an OOF family crosses canonical folds")
    lookup = dict(zip(classes, range(len(classes)), strict=True))
    truth = canonical[target].astype(str).map(lookup)
    if truth.isna().any() or not result["true_index"].eq(truth.to_numpy()).all():
        raise ValueError("OOF true indices disagree with canonical labels")
    probabilities = result[probability_columns(classes)].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError("OOF probabilities must be finite, in [0,1], and sum to one")
    indices = probabilities.argmax(axis=1)
    if not result["predicted_index"].eq(indices).all():
        raise ValueError("OOF predicted indices disagree with probability argmax")
    repairs = 0
    for column, indices in (
        ("true_label", result["true_index"]),
        ("predicted_label", result["predicted_index"]),
    ):
        labels = indices.map(dict(enumerate(classes)))
        if allow_legacy_na:
            repair = result[column].eq("") & labels.eq("NA")
            repairs += int(repair.sum())
            result.loc[repair, column] = "NA"
        if not result[column].eq(labels).all():
            raise ValueError(f"OOF {column} disagrees with the fixed numeric index")
    confidence = pd.to_numeric(result["confidence"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or not np.allclose(
        confidence, probabilities.max(axis=1), atol=1e-6, rtol=0
    ):
        raise ValueError("OOF confidence disagrees with probability maximum")
    if result["run_id"].isna().any() or result["run_id"].eq("").any():
        raise ValueError("OOF rows must identify their run")
    if run_ids_by_fold is not None:
        wanted = result["cv_fold"].map(run_ids_by_fold)
        if wanted.isna().any() or not result["run_id"].eq(wanted).all():
            raise ValueError("OOF run IDs disagree with the required fold lineage")
    result.attrs["legacy_na_label_repairs"] = repairs
    return result


def oof_metrics(predictions: pd.DataFrame, classes: Sequence[str]) -> dict[str, Any]:
    return classification_metrics(
        predictions["true_index"].to_numpy(dtype=np.int64),
        predictions[probability_columns(classes)].to_numpy(dtype=float),
        classes,
    )


def paired_family_bootstrap(
    candidate: pd.DataFrame,
    parent: pd.DataFrame,
    *,
    classes: Sequence[str],
    repetitions: int = DECISION_BOOTSTRAP_REPETITIONS,
    seed: int = DECISION_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired macro-F1 difference; resample whole families within each saved fold.

    Family sufficient counts avoid rebuilding data frames in each draw. Zero-
    support classes still contribute zero F1, matching the official metric.
    """
    if repetitions < 2:
        raise ValueError("bootstrap needs at least two repetitions")
    candidate = candidate.sort_values("id").reset_index(drop=True)
    parent = parent.sort_values("id").reset_index(drop=True)
    if candidate.empty or candidate["id"].duplicated().any() or parent["id"].duplicated().any():
        raise ValueError("bootstrap needs non-empty unique paired IDs")
    for field in ("id", "cv_fold", "product_family_group", "true_index"):
        if not candidate[field].equals(parent[field]):
            raise ValueError(f"bootstrap pairing disagrees on {field}")
    if candidate.groupby("product_family_group")["cv_fold"].nunique().gt(1).any():
        raise ValueError("bootstrap family crosses folds")
    k = len(classes)
    truth = _integers(candidate["true_index"], "true_index")
    child = _integers(candidate["predicted_index"], "candidate predicted_index")
    anchor = _integers(parent["predicted_index"], "parent predicted_index")
    if any((a < 0).any() or (a >= k).any() for a in (truth, child, anchor)):
        raise ValueError("bootstrap labels exceed the fixed class range")
    one_hot = np.eye(k, dtype=np.int64)
    counts = np.concatenate(
        [
            one_hot[truth],
            one_hot[child],
            one_hot[anchor],
            one_hot[truth] * (truth == child)[:, None],
            one_hot[truth] * (truth == anchor)[:, None],
        ],
        axis=1,
    )
    grouped: list[np.ndarray] = []
    family_counts: dict[str, int] = {}
    for fold in sorted(candidate["cv_fold"].unique()):
        selected = candidate["cv_fold"].eq(fold).to_numpy()
        families, inverse = np.unique(
            candidate.loc[selected, "product_family_group"].astype(str), return_inverse=True
        )
        if len(families) < 2:
            raise ValueError("bootstrap needs at least two families in every fold")
        block = np.zeros((len(families), 5 * k), dtype=np.int64)
        np.add.at(block, inverse, counts[selected])
        grouped.append(block)
        family_counts[str(int(fold))] = len(families)

    def difference(total: np.ndarray) -> float:
        support, child_count, parent_count, child_tp, parent_tp = np.split(total, 5)

        def score(tp: np.ndarray, predicted: np.ndarray) -> float:
            denominator = support + predicted
            return float(
                np.divide(
                    2 * tp, denominator, out=np.zeros(k, dtype=float), where=denominator > 0
                ).mean()
            )

        return score(child_tp, child_count) - score(parent_tp, parent_count)

    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for repeat in range(repetitions):
        total = np.zeros(5 * k, dtype=np.int64)
        for block in grouped:
            total += block[rng.integers(0, len(block), size=len(block))].sum(axis=0)
        draws[repeat] = difference(total)
    return {
        "point": difference(counts.sum(axis=0)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "repetitions": repetitions,
        "seed": seed,
        "method": "paired_whole_family_percentile_stratified_by_canonical_fold",
        "families_by_fold": family_counts,
        "classes": list(classes),
        "selection_bias_removed": False,
    }


def check(name: str, value: Any, rule: str, passed: bool | None) -> dict[str, Any]:
    return {
        "gate": name,
        "value": value,
        "rule": rule,
        "status": "not_evaluated" if passed is None else "pass" if passed else "fail",
    }


def decision(checks: Sequence[Mapping[str, Any]]) -> str:
    """A known failure rejects; otherwise missing evidence blocks acceptance."""
    statuses = [row["status"] for row in checks]
    if "fail" in statuses:
        return "fail"
    return "pass" if statuses and all(s == "pass" for s in statuses) else "not_evaluated"


def robustness_changes(
    frame: pd.DataFrame,
    *,
    clean_by_fold: Mapping[int, float],
    run_ids_by_fold: Mapping[int, str],
) -> pd.Series:
    """Validate the complete frozen corruption grid, then average fold deltas."""
    required = {"validation_fold", "corruption", "macro_f1", "macro_f1_change", "run_id"}
    if required.difference(frame.columns):
        raise ValueError("robustness evidence lacks required columns")
    rows = frame.copy()
    rows["validation_fold"] = _integers(rows["validation_fold"], "robustness fold")
    wanted = {(int(fold), c) for fold in clean_by_fold for c in CORE_CORRUPTIONS}
    actual = list(zip(rows["validation_fold"], rows["corruption"], strict=True))
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise ValueError("robustness must contain each required fold/corruption exactly once")
    if not rows["run_id"].eq(rows["validation_fold"].map(run_ids_by_fold)).all():
        raise ValueError("robustness run IDs disagree with fold lineage")
    scores = pd.to_numeric(rows["macro_f1"], errors="raise").to_numpy(dtype=float)
    changes = pd.to_numeric(rows["macro_f1_change"], errors="raise").to_numpy(dtype=float)
    clean = rows["validation_fold"].map(clean_by_fold).to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(changes).all():
        raise ValueError("robustness evidence must be finite")
    if (
        (scores < 0).any()
        or (scores > 1).any()
        or not np.allclose(changes, scores - clean, atol=1e-8, rtol=0)
    ):
        raise ValueError("robustness changes disagree with the matching clean scores")
    rows["macro_f1_change"] = changes
    return rows.groupby("corruption")["macro_f1_change"].mean()
