"""Smoke and five-fold CNN experiment orchestration for Task 1."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fashion.config import ROOT, TASK1_EVIDENCE_DIR, TASK1_RESULT_DIR
from fashion.data.dataset import get_samples
from fashion.task1.evaluation import (
    aggregate_fold_metrics,
    classification_metrics,
    per_class_metrics,
    validate_oof_predictions,
)
from fashion.task1.candidates import (
    TASK1_MILD_AUG_CANDIDATE,
    TASK1_NO_AUG_CANDIDATE,
    Task1CnnCandidate,
)
from fashion.task1.training import Task1FoldResult, Task1TrainConfig, train_task1_fold
from fashion.train.artifacts import atomic_write_csv
from fashion.train.registry import RunRegistry


@dataclass(frozen=True)
class Task1ExperimentResult:
    """Evidence collected from a smoke run or complete preprocessing comparison."""

    mode: Literal["smoke", "full"]
    fold_results: tuple[Task1FoldResult, ...]
    fold_metrics: pd.DataFrame
    comparison: pd.DataFrame
    oof_metrics: pd.DataFrame
    oof_predictions: dict[str, pd.DataFrame]
    per_class: dict[str, pd.DataFrame]


def _fold_metrics_frame(results: Sequence[Task1FoldResult]) -> pd.DataFrame:
    """Make one auditable metrics row for each physical fold run."""
    return pd.DataFrame(
        [
            {
                "run_id": result.run_id,
                "fold": result.fold,
                "candidate_id": result.candidate_id,
                "preprocessing_id": result.preprocessing_id,
                "loss_id": result.loss_id,
                **result.metrics,
            }
            for result in results
        ]
    )


def _class_names(label_map: Mapping[str, object]) -> list[str]:
    """Read the fixed label ordering needed for report per-class evidence."""
    try:
        classes = [str(value) for value in label_map["classes"]]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("Task 1 label map must define classes") from error
    if len(classes) != 124:
        raise ValueError("Task 1 label map must contain exactly 124 classes")
    return classes


def _aggregate_comparison(
    fold_metrics: pd.DataFrame,
    candidate_ids: Sequence[str],
) -> pd.DataFrame:
    """Aggregate each candidate's five physical folds into one comparison row."""
    metric_columns = ["macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy"]
    missing = {"run_id", "fold", "candidate_id", "preprocessing_id", "loss_id", *metric_columns}
    missing = missing.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"Task 1 fold metrics are missing columns: {sorted(missing)}")
    rows: list[dict[str, float | str]] = []
    for candidate_id in candidate_ids:
        candidate = fold_metrics.loc[fold_metrics["candidate_id"].eq(candidate_id)].copy()
        candidate_folds = fold_metrics.loc[
            fold_metrics["candidate_id"].eq(candidate_id), "fold"
        ]
        if set(candidate_folds.astype(int)) != set(range(5)) or len(candidate_folds) != 5:
            raise ValueError(
                "full Task 1 evidence requires exactly five folds with unique labels 0,1,2,3,4"
            )
        if candidate["preprocessing_id"].nunique() != 1 or candidate["loss_id"].nunique() != 1:
            raise ValueError("Task 1 candidate identity must stay consistent across final folds")
        summary = aggregate_fold_metrics(candidate.loc[:, metric_columns].to_dict(orient="records"))
        row: dict[str, float | str] = {
            "candidate_id": candidate_id,
            "preprocessing_id": str(candidate.iloc[0]["preprocessing_id"]),
            "loss_id": str(candidate.iloc[0]["loss_id"]),
        }
        for metric, values in summary.iterrows():
            row[f"{metric}_mean"] = float(values["mean"])
            row[f"{metric}_std"] = float(values["std"])
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction_probabilities(predictions: pd.DataFrame) -> np.ndarray:
    """Extract the ordered 124-way probability matrix from a fold artifact."""
    columns = [f"prob_{index:03d}" for index in range(124)]
    missing = [column for column in columns if column not in predictions]
    if missing:
        raise ValueError("prediction artifacts must contain all 124 class probabilities")
    return predictions.loc[:, columns].to_numpy(dtype=np.float64)


def _write_full_evidence(
    fold_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    oof_metrics: pd.DataFrame,
    per_class: Mapping[str, pd.DataFrame],
) -> None:
    """Persist report inputs only after all candidate evidence validates."""
    atomic_write_csv(TASK1_EVIDENCE_DIR / "fold_metrics.csv", fold_metrics)
    atomic_write_csv(TASK1_EVIDENCE_DIR / "comparison.csv", comparison)
    atomic_write_csv(TASK1_EVIDENCE_DIR / "oof_metrics.csv", oof_metrics)
    for candidate_id, frame in per_class.items():
        atomic_write_csv(TASK1_EVIDENCE_DIR / f"per_class_{candidate_id}.csv", frame)


def run_task1_experiment(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    mode: Literal["smoke", "full"],
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR,
    fold_runner: Callable[..., Task1FoldResult] = train_task1_fold,
) -> Task1ExperimentResult:
    """Run the fixed smoke check or two complete five-fold preprocessing candidates."""
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")

    candidates: tuple[Task1CnnCandidate, ...] = (
        TASK1_NO_AUG_CANDIDATE,
        TASK1_MILD_AUG_CANDIDATE,
    )
    if mode == "smoke":
        schedule = ((TASK1_NO_AUG_CANDIDATE, 0, Task1TrainConfig.smoke()),)
    else:
        schedule = tuple(
            (candidate, fold, Task1TrainConfig.full())
            for candidate in candidates
            for fold in range(5)
        )

    fold_results = tuple(
        fold_runner(
            splits,
            label_map,
            validation_fold=fold,
            candidate=candidate,
            config=config,
            registry=registry,
            root=root,
            result_root=result_root,
        )
        for candidate, fold, config in schedule
    )
    fold_metrics = _fold_metrics_frame(fold_results)
    if mode == "smoke":
        return Task1ExperimentResult(
            mode=mode,
            fold_results=fold_results,
            fold_metrics=fold_metrics,
            comparison=pd.DataFrame(),
            oof_metrics=pd.DataFrame(),
            oof_predictions={},
            per_class={},
        )

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    comparison = _aggregate_comparison(fold_metrics, candidate_ids)
    expected_ids = get_samples(splits, partition="development", target="articleType")["id"].tolist()
    class_names = _class_names(label_map)
    oof_predictions: dict[str, pd.DataFrame] = {}
    oof_metrics_rows: list[dict[str, float | str]] = []
    per_class: dict[str, pd.DataFrame] = {}
    for candidate_id in candidate_ids:
        candidate_results = [
            result for result in fold_results if result.candidate_id == candidate_id
        ]
        predictions = pd.concat(
            [pd.read_csv(result.prediction_path) for result in candidate_results],
            ignore_index=True,
        )
        validate_oof_predictions(predictions, expected_ids)
        probabilities = _prediction_probabilities(predictions)
        pooled = classification_metrics(
            predictions["true_index"].to_numpy(dtype=np.int64), probabilities
        )
        oof_metrics_rows.append(
            {
                "candidate_id": candidate_id,
                "preprocessing_id": candidate_results[0].preprocessing_id,
                "loss_id": candidate_results[0].loss_id,
                "macro_f1_124": pooled["macro_f1"],
                "weighted_f1": pooled["weighted_f1"],
                "top1_accuracy": pooled["top1_accuracy"],
                "top5_accuracy": pooled["top5_accuracy"],
            }
        )
        oof_predictions[candidate_id] = predictions
        per_class[candidate_id] = per_class_metrics(
            predictions["true_index"].to_numpy(dtype=np.int64), probabilities, class_names
        )

    oof_metrics = pd.DataFrame(oof_metrics_rows)
    _write_full_evidence(fold_metrics, comparison, oof_metrics, per_class)
    return Task1ExperimentResult(
        mode=mode,
        fold_results=fold_results,
        fold_metrics=fold_metrics,
        comparison=comparison,
        oof_metrics=oof_metrics,
        oof_predictions=oof_predictions,
        per_class=per_class,
    )
