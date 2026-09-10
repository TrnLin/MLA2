"""Read-only Task 1 EDA and failure-analysis evidence helpers."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from fashion.config import ROOT
from fashion.task1.evaluation import validate_oof_predictions
from fashion.task1.image_contract import TASK1_IMAGE_SIZE


@dataclass(frozen=True)
class Task1ProblemProfile:
    """Small, safe summary of development-only Task 1 data conditions."""

    development_products: int
    class_count: int
    minimum_class_products: int
    maximum_class_products: int
    grayscale_images: int
    unusual_geometry_images: int
    classes_with_fold_warnings: int
    classes_untrainable_in_any_fold: int


def build_task1_problem_profile(
    splits: pd.DataFrame,
    class_summary: pd.DataFrame,
) -> Task1ProblemProfile:
    """Summarise only development rows for Task 1 modelling decisions."""
    required = {"id", "partition", "articleType", "width", "height", "mode"}
    if missing := required.difference(splits.columns):
        raise ValueError(f"Task 1 EDA rows are missing columns: {sorted(missing)}")
    development = splits.loc[splits["partition"].eq("development")].copy()
    if development.empty or development["id"].duplicated().any():
        raise ValueError("Task 1 EDA requires unique development products")
    if not set(splits["partition"]).issubset({"development", "holdout", "quarantine"}):
        raise ValueError("Task 1 EDA contains an unknown partition")
    summary_required = {"target", "rare_warning", "untrainable_fold_count"}
    if missing := summary_required.difference(class_summary.columns):
        raise ValueError(f"Task 1 class summary is missing columns: {sorted(missing)}")
    summary = class_summary.loc[class_summary["target"].eq("articleType")].copy()
    if summary.empty:
        raise ValueError("Task 1 EDA requires the articleType class summary")
    untrainable = pd.to_numeric(summary["untrainable_fold_count"], errors="raise")
    if untrainable.lt(0).any():
        raise ValueError("untrainable fold counts must be non-negative")
    support = development["articleType"].astype(str).value_counts()
    unusual = ~(
        pd.to_numeric(development["width"], errors="raise").eq(TASK1_IMAGE_SIZE[1])
        & pd.to_numeric(development["height"], errors="raise").eq(TASK1_IMAGE_SIZE[0])
    )
    return Task1ProblemProfile(
        development_products=len(development),
        class_count=len(support),
        minimum_class_products=int(support.min()),
        maximum_class_products=int(support.max()),
        grayscale_images=int(development["mode"].astype(str).eq("L").sum()),
        unusual_geometry_images=int(unusual.sum()),
        classes_with_fold_warnings=int(
            summary["rare_warning"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        classes_untrainable_in_any_fold=int(untrainable.gt(0).sum()),
    )


def build_task1_decision_evidence(profile: Task1ProblemProfile) -> pd.DataFrame:
    """Connect observed Task 1 risks to the experiments that answer them."""
    rows = [
        {
            "evidence": (
                f"{profile.unusual_geometry_images} development images differ from the usual "
                "60-by-80 geometry"
            ),
            "problem": "stretching changes shape and centre crops can remove product edges",
            "choice_to_test": "aspect-preserving 60-by-80 white canvas",
        },
        {
            "evidence": f"{profile.grayscale_images} development images are grayscale",
            "problem": "input channel formats are inconsistent",
            "choice_to_test": "deterministic RGB conversion for every model family",
        },
        {
            "evidence": (
                f"{profile.class_count} classes range from "
                f"{profile.minimum_class_products} to {profile.maximum_class_products} products"
            ),
            "problem": "accuracy can hide failure on rare classes",
            "choice_to_test": "fixed-class macro-F1 plus per-class evidence",
        },
        {
            "evidence": (
                f"{profile.classes_untrainable_in_any_fold} classes are untrainable in at least "
                "one fold"
            ),
            "problem": "some validation folds cannot represent every class",
            "choice_to_test": "fixed-class macro-F1 with zero-support classes retained",
        },
        {
            "evidence": "product shape and edge detail may separate visually similar classes",
            "problem": "raw pixels may not provide a strong low-data baseline",
            "choice_to_test": "HOG with k-NN and linear SVM baselines",
        },
        {
            "evidence": (
                f"{profile.development_products} development products support a small benchmark"
            ),
            "problem": "a large network can overfit a limited labelled set",
            "choice_to_test": "small scratch CNN against the classical baselines",
        },
        {
            "evidence": (
                f"{profile.classes_with_fold_warnings} classes have rare-class fold warnings"
            ),
            "problem": "rare classes see too few distinct training examples",
            "choice_to_test": "training-only augmentation for the scratch CNN",
        },
    ]
    return pd.DataFrame(rows, columns=["evidence", "problem", "choice_to_test"])


def build_task1_weak_class_table(
    per_class: Mapping[str, pd.DataFrame], *, limit: int = 10
) -> pd.DataFrame:
    """Return each candidate's lowest-F1 classes without mixing identities."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {"class_index", "class_name", "support", "precision", "recall", "f1"}
    columns = [
        "candidate_id",
        "class_index",
        "class_name",
        "support",
        "precision",
        "recall",
        "f1",
    ]
    rows = []
    for candidate_id, frame in per_class.items():
        if missing := required.difference(frame.columns):
            raise ValueError(f"per-class evidence is missing columns: {sorted(missing)}")
        weakest = (
            frame.sort_values(["f1", "support", "class_index"], kind="stable")
            .head(limit)
            .copy()
        )
        weakest.insert(0, "candidate_id", candidate_id)
        rows.append(weakest.loc[:, columns])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def build_task1_confusion_pairs(
    oof_predictions: Mapping[str, pd.DataFrame], *, limit: int = 10
) -> pd.DataFrame:
    """Return each candidate's most common out-of-fold confusion pairs."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {"id", "true_label", "predicted_label"}
    columns = ["candidate_id", "true_label", "predicted_label", "error_count", "example_ids"]
    output = []
    for candidate_id, frame in oof_predictions.items():
        if missing := required.difference(frame.columns):
            raise ValueError(f"OOF evidence is missing columns: {sorted(missing)}")
        errors = frame.loc[frame["true_label"].ne(frame["predicted_label"])].copy()
        grouped = (
            errors.groupby(["true_label", "predicted_label"], as_index=False)
            .agg(
                error_count=("id", "size"),
                example_ids=(
                    "id",
                    lambda ids: ",".join(str(value) for value in sorted(map(int, ids))[:3]),
                ),
            )
            .sort_values(
                ["error_count", "true_label", "predicted_label"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(limit)
        )
        grouped.insert(0, "candidate_id", candidate_id)
        output.append(grouped)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=columns)


def load_task1_oof_predictions(
    fold_metrics: pd.DataFrame,
    registry_rows: pd.DataFrame,
    *,
    candidate_id: str,
    expected_ids: Collection[int],
    root: str | Path = ROOT,
) -> pd.DataFrame:
    """Load one candidate's five registered development OOF prediction files."""
    fold_required = {"run_id", "fold", "candidate_id"}
    if missing := fold_required.difference(fold_metrics.columns):
        raise ValueError(f"fold metrics are missing columns: {sorted(missing)}")
    registry_required = {"run_id", "status", "prediction_path"}
    if missing := registry_required.difference(registry_rows.columns):
        raise ValueError(f"registry rows are missing columns: {sorted(missing)}")

    candidate = fold_metrics.loc[fold_metrics["candidate_id"].eq(candidate_id)].copy()
    folds = pd.to_numeric(candidate["fold"], errors="raise").astype(int)
    if len(candidate) != 5 or set(folds) != set(range(5)) or folds.duplicated().any():
        raise ValueError("OOF loading requires exactly five folds labelled 0,1,2,3,4")
    if candidate["run_id"].astype(str).duplicated().any():
        raise ValueError("OOF loading requires one unique run ID per fold")

    root_path = Path(root)
    prediction_frames: list[pd.DataFrame] = []
    for row in candidate.assign(fold=folds).sort_values("fold").itertuples(index=False):
        matches = registry_rows.loc[
            registry_rows["run_id"].astype(str).eq(str(row.run_id))
            & registry_rows["status"].astype(str).eq("completed")
        ]
        if len(matches) != 1:
            raise ValueError(f"run {row.run_id!r} must have exactly one completed registry row")
        raw_path = str(matches.iloc[0]["prediction_path"]).strip()
        if not raw_path:
            raise ValueError(f"run {row.run_id!r} has no registered prediction path")
        prediction_path = Path(raw_path)
        if not prediction_path.is_absolute():
            prediction_path = root_path / prediction_path
        if not prediction_path.is_file():
            raise ValueError(
                f"registered prediction file does not exist for run {row.run_id!r}: "
                f"{prediction_path}"
            )
        prediction_frames.append(pd.read_csv(prediction_path))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    validate_oof_predictions(predictions, expected_ids)
    return predictions.sort_values("id", kind="stable").reset_index(drop=True)


def build_task1_confusion_detail(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Describe one candidate's largest directed OOF confusion pairs."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {"id", "true_label", "predicted_label"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"OOF evidence is missing columns: {sorted(missing)}")
    columns = [
        "rank",
        "candidate_id",
        "true_label",
        "predicted_label",
        "error_count",
        "true_support",
        "error_rate",
        "example_ids",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    support = predictions.groupby("true_label").size().rename("true_support")
    errors = predictions.loc[predictions["true_label"].ne(predictions["predicted_label"])]
    if errors.empty:
        return pd.DataFrame(columns=columns)
    detail = (
        errors.groupby(["true_label", "predicted_label"], as_index=False)
        .agg(
            error_count=("id", "size"),
            example_ids=(
                "id",
                lambda ids: ",".join(str(value) for value in sorted(map(int, ids))[:3]),
            ),
        )
        .join(support, on="true_label")
        .sort_values(
            ["error_count", "true_label", "predicted_label"],
            ascending=[False, True, True],
            kind="stable",
        )
        .head(limit)
        .reset_index(drop=True)
    )
    detail.insert(0, "candidate_id", str(candidate_id))
    detail.insert(0, "rank", range(1, len(detail) + 1))
    detail["error_rate"] = detail["error_count"] / detail["true_support"]
    return detail.loc[:, columns]
