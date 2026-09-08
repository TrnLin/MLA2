"""Staged classical-model experiment orchestration for Task 1."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from fashion.config import ROOT, TASK1_EVIDENCE_DIR, TASK1_HOG_CACHE_DIR, TASK1_RESULT_DIR
from fashion.data.dataset import get_samples
from fashion.data.hashing import compute_sha256
from fashion.task1.classical_features import TASK1_HOG_COARSE, TASK1_HOG_SPECS, Task1HogSpec
from fashion.task1.classical_models import (
    TASK1_DEFAULT_KNN,
    TASK1_DEFAULT_SVM,
    TASK1_KNN_GRID,
    TASK1_SVM_GRID,
    Task1ClassicalModelConfig,
    Task1LinearSVMConfig,
)
from fashion.task1.classical_training import (
    Task1ClassicalFoldResult,
    Task1ClassicalRunConfig,
    _classical_implementation_sha256,
    run_task1_classical_fold,
)
from fashion.task1.cnn_experiments import _class_names, _prediction_probabilities
from fashion.task1.evaluation import (
    aggregate_fold_metrics,
    classification_metrics,
    per_class_metrics,
    validate_oof_predictions,
)
from fashion.train.artifacts import atomic_write_csv, atomic_write_json, canonical_sha256
from fashion.train.registry import RunRegistry

__all__ = [
    "Task1ClassicalSelection",
    "Task1ClassicalExperimentResult",
    "run_task1_classical_experiment",
]


@dataclass(frozen=True)
class Task1ClassicalSelection:
    """The one frozen HOG setting and one selected configuration per model family."""

    hog_id: str
    knn_config_id: str
    svm_config_id: str


@dataclass(frozen=True)
class Task1ClassicalExperimentResult:
    """Evidence from the staged fold-0 classic-model controller."""

    stage: Literal["smoke", "tune", "final"]
    fold_results: tuple[Task1ClassicalFoldResult, ...]
    tuning: pd.DataFrame
    selection: Task1ClassicalSelection | None
    fold_metrics: pd.DataFrame
    comparison: pd.DataFrame
    oof_metrics: pd.DataFrame
    oof_predictions: dict[str, pd.DataFrame]
    per_class: dict[str, pd.DataFrame]


def _classical_metrics_frame(results: Sequence[Task1ClassicalFoldResult]) -> pd.DataFrame:
    """Make one auditable row for every classic fold candidate."""
    return pd.DataFrame(
        [
            {
                "run_id": r.run_id,
                "fold": r.fold,
                "candidate_id": r.candidate_id,
                "hog_id": r.hog_id,
                "model_family": r.model_family,
                **r.metrics,
            }
            for r in results
        ]
    )


def _rank_classical_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return candidates in the fixed, deterministic model-selection order."""
    ranking = ["macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy"]
    missing = {"candidate_id", *ranking}.difference(frame.columns)
    if missing:
        raise ValueError(f"classic candidate metrics are missing columns: {sorted(missing)}")
    return frame.sort_values(
        ranking + ["candidate_id"], ascending=[False, False, False, False, True], kind="stable"
    ).reset_index(drop=True)


def _select_shared_hog(default_metrics: pd.DataFrame) -> Task1HogSpec:
    """Select a HOG setting from the two default-model macro-F1 means."""
    missing = {"hog_id", "macro_f1"}.difference(default_metrics.columns)
    if missing:
        raise ValueError(f"default HOG metrics are missing columns: {sorted(missing)}")
    means = default_metrics.groupby("hog_id", sort=False)["macro_f1"].mean()
    scores: list[tuple[float, int, Task1HogSpec]] = []
    for position, spec in enumerate(TASK1_HOG_SPECS):
        if spec.hog_id not in means.index:
            raise ValueError(f"missing default metrics for HOG candidate {spec.hog_id}")
        scores.append((float(means.loc[spec.hog_id]), -position, spec))
    return max(scores, key=lambda item: (item[0], item[1]))[2]


def _select_model_config(metrics: pd.DataFrame, config_ids: set[str]) -> str:
    """Choose one configuration using the project-wide deterministic ranking rule."""
    candidates = metrics.loc[
        metrics["candidate_id"].map(
            lambda candidate: any(
                str(candidate).endswith(f"-{config_id}") for config_id in config_ids
            )
        )
    ]
    if candidates.empty:
        raise ValueError("no completed classic candidates match the requested model family")
    candidate_id = str(_rank_classical_candidates(candidates).iloc[0]["candidate_id"])
    for config_id in config_ids:
        if candidate_id.endswith(f"-{config_id}"):
            return config_id
    raise ValueError("ranked classic candidate has an unrecognised configuration ID")


def _split_sha256(splits: pd.DataFrame) -> str:
    """Hash the supplied fixed split table in ID order for selection provenance."""
    if "id" not in splits:
        raise ValueError("splits must contain id before selection provenance can be written")
    ordered = splits.sort_values("id", kind="stable").reset_index(drop=True)
    normalized = ordered.astype(object).where(pd.notna(ordered), None)
    return canonical_sha256(normalized.to_dict(orient="records"))


def _classical_selection_implementation_sha256() -> str:
    """Seal tuning to the dedicated controller and physical-run implementation."""
    return canonical_sha256(
        {
            "classical_experiments_sha256": compute_sha256(Path(__file__)),
            "classical_sha256": _classical_implementation_sha256(),
        }
    )


def _classical_selection_payload(
    selection: Task1ClassicalSelection,
    *,
    tuning_path: Path,
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
) -> dict[str, str]:
    return {
        "hog_id": selection.hog_id,
        "knn_config_id": selection.knn_config_id,
        "svm_config_id": selection.svm_config_id,
        "tuning_sha256": compute_sha256(tuning_path),
        "split_sha256": _split_sha256(splits),
        "label_map_sha256": canonical_sha256(label_map),
        "implementation_sha256": _classical_selection_implementation_sha256(),
    }


def _load_classical_selection(
    evidence_root: Path, *, splits: pd.DataFrame, label_map: Mapping[str, object]
) -> Task1ClassicalSelection:
    """Load a sealed tuning decision only when its inputs and CSV still match."""
    selection_path = evidence_root / "classical_selection.json"
    try:
        with selection_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as error:
        raise ValueError(
            "classical_selection.json is required for the final classic stage"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("classical selection is malformed") from error
    required = {
        "hog_id",
        "knn_config_id",
        "svm_config_id",
        "tuning_sha256",
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or not all(isinstance(payload[key], str) for key in required)
    ):
        raise ValueError("classical selection is malformed")
    if payload["split_sha256"] != _split_sha256(splits):
        raise ValueError("classical selection split provenance is stale")
    if payload["label_map_sha256"] != canonical_sha256(label_map):
        raise ValueError("classical selection label-map provenance is stale")
    if payload["implementation_sha256"] != _classical_selection_implementation_sha256():
        raise ValueError("classical selection implementation provenance is stale")
    try:
        tuning_sha256 = compute_sha256(evidence_root / "classical_tuning.csv")
    except OSError as error:
        raise ValueError("classical selection tuning evidence is missing") from error
    if payload["tuning_sha256"] != tuning_sha256:
        raise ValueError("classical selection tuning provenance is stale")
    return Task1ClassicalSelection(
        payload["hog_id"], payload["knn_config_id"], payload["svm_config_id"]
    )


def _resolve_classical_selection(
    selection: Task1ClassicalSelection,
) -> tuple[Task1HogSpec, Task1ClassicalModelConfig, Task1ClassicalModelConfig]:
    hogs = {spec.hog_id: spec for spec in TASK1_HOG_SPECS}
    knn = {config.config_id: config for config in TASK1_KNN_GRID}
    svm = {config.config_id: config for config in TASK1_SVM_GRID}
    try:
        hog_spec = hogs[selection.hog_id]
    except KeyError as error:
        raise ValueError(f"unknown HOG spec in classical selection: {selection.hog_id}") from error
    try:
        knn_config = knn[selection.knn_config_id]
    except KeyError as error:
        raise ValueError(
            f"unknown KNN config in classical selection: {selection.knn_config_id}"
        ) from error
    try:
        svm_config = svm[selection.svm_config_id]
    except KeyError as error:
        raise ValueError(
            f"unknown Linear SVM config in classical selection: {selection.svm_config_id}"
        ) from error
    return hog_spec, knn_config, svm_config


def _aggregate_classical_comparison(
    fold_metrics: pd.DataFrame, candidate_ids: Sequence[str]
) -> pd.DataFrame:
    metrics = ["macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy"]
    missing = {"run_id", "fold", "candidate_id", "hog_id", "model_family", *metrics}.difference(
        fold_metrics.columns
    )
    if missing:
        raise ValueError(f"classic fold metrics are missing columns: {sorted(missing)}")
    rows: list[dict[str, float | str]] = []
    for candidate_id in candidate_ids:
        candidate = fold_metrics.loc[fold_metrics["candidate_id"].eq(candidate_id)].copy()
        if set(candidate["fold"].astype(int)) != set(range(5)) or len(candidate) != 5:
            raise ValueError(
                "final classic evidence requires exactly five folds with unique labels 0,1,2,3,4"
            )
        if candidate["hog_id"].nunique() != 1 or candidate["model_family"].nunique() != 1:
            raise ValueError("classic candidate identity must stay consistent across final folds")
        summary = aggregate_fold_metrics(candidate.loc[:, metrics].to_dict(orient="records"))
        row: dict[str, float | str] = {
            "candidate_id": candidate_id,
            "hog_id": str(candidate.iloc[0]["hog_id"]),
            "model_family": str(candidate.iloc[0]["model_family"]),
        }
        for metric, values in summary.iterrows():
            row[f"{metric}_mean"] = float(values["mean"])
            row[f"{metric}_std"] = float(values["std"])
        rows.append(row)
    return pd.DataFrame(rows)


def _classical_final_oof_evidence(
    results: Sequence[Task1ClassicalFoldResult],
    candidate_ids: Sequence[str],
    *,
    expected_ids: Sequence[int],
    class_names: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    oof_predictions: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, float | str]] = []
    per_class: dict[str, pd.DataFrame] = {}
    for candidate_id in candidate_ids:
        candidate_results = sorted(
            (r for r in results if r.candidate_id == candidate_id), key=lambda r: r.fold
        )
        predictions = pd.concat(
            [pd.read_csv(r.prediction_path, keep_default_na=False) for r in candidate_results],
            ignore_index=True,
        )
        validate_oof_predictions(predictions, expected_ids)
        probabilities = _prediction_probabilities(predictions)
        rows.append(
            {
                "candidate_id": candidate_id,
                **classification_metrics(
                    predictions["true_index"].to_numpy(dtype=np.int64), probabilities
                ),
            }
        )
        oof_predictions[candidate_id] = predictions
        per_class[candidate_id] = per_class_metrics(
            predictions["true_index"].to_numpy(dtype=np.int64), probabilities, class_names
        )
    return oof_predictions, pd.DataFrame(rows), per_class


def _write_classical_final_evidence(
    evidence_root: Path,
    fold_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    oof_metrics: pd.DataFrame,
    per_class: Mapping[str, pd.DataFrame],
) -> None:
    atomic_write_csv(evidence_root / "classical_fold_metrics.csv", fold_metrics)
    atomic_write_csv(evidence_root / "classical_comparison.csv", comparison)
    atomic_write_csv(evidence_root / "classical_oof_metrics.csv", oof_metrics)
    for candidate_id, frame in per_class.items():
        atomic_write_csv(evidence_root / f"per_class_classical_{candidate_id}.csv", frame)


def run_task1_classical_experiment(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    stage: Literal["smoke", "tune", "final"],
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR / "classical",
    cache_root: str | Path = TASK1_HOG_CACHE_DIR,
    evidence_root: str | Path = TASK1_EVIDENCE_DIR,
    selection: Task1ClassicalSelection | None = None,
    fold_runner: Callable[..., Task1ClassicalFoldResult] = run_task1_classical_fold,
) -> Task1ClassicalExperimentResult:
    """Run classic smoke, tuning, or selected-model five-fold evidence."""
    if stage not in {"smoke", "tune", "final"}:
        raise ValueError("stage must be 'smoke', 'tune', or 'final'")

    def run_one(
        hog_spec: Task1HogSpec,
        model_config: Task1ClassicalModelConfig,
        run_config: Task1ClassicalRunConfig,
        validation_fold: int = 0,
    ) -> Task1ClassicalFoldResult:
        return fold_runner(
            splits,
            label_map,
            validation_fold=validation_fold,
            hog_spec=hog_spec,
            model_config=model_config,
            run_config=run_config,
            registry=registry,
            root=root,
            result_root=result_root,
            cache_root=cache_root,
        )

    if stage == "smoke":
        results = tuple(
            run_one(TASK1_HOG_COARSE, config, Task1ClassicalRunConfig.smoke())
            for config in (TASK1_DEFAULT_KNN, TASK1_DEFAULT_SVM)
        )
        return Task1ClassicalExperimentResult(
            stage,
            results,
            pd.DataFrame(),
            None,
            _classical_metrics_frame(results),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            {},
        )

    output_root = Path(evidence_root)
    if stage == "final":
        selected = selection or _load_classical_selection(
            output_root, splits=splits, label_map=label_map
        )
        hog_spec, knn_config, svm_config = _resolve_classical_selection(selected)
        configs = (knn_config, svm_config)
        results = tuple(
            run_one(hog_spec, config, Task1ClassicalRunConfig.full(), validation_fold=fold)
            for config in configs
            for fold in range(5)
        )
        candidate_ids = tuple(f"{hog_spec.hog_id}-{config.config_id}" for config in configs)
        fold_metrics = _classical_metrics_frame(results).loc[
            :,
            [
                "run_id",
                "fold",
                "candidate_id",
                "hog_id",
                "model_family",
                "macro_f1",
                "weighted_f1",
                "top1_accuracy",
                "top5_accuracy",
            ],
        ]
        comparison = _aggregate_classical_comparison(fold_metrics, candidate_ids)
        expected_ids = get_samples(splits, partition="development", target="articleType")[
            "id"
        ].tolist()
        oof_predictions, oof_metrics, per_class = _classical_final_oof_evidence(
            results, candidate_ids, expected_ids=expected_ids, class_names=_class_names(label_map)
        )
        _write_classical_final_evidence(
            output_root, fold_metrics, comparison, oof_metrics, per_class
        )
        return Task1ClassicalExperimentResult(
            stage,
            results,
            pd.DataFrame(),
            selected,
            fold_metrics,
            comparison,
            oof_metrics,
            oof_predictions,
            per_class,
        )

    defaults = tuple(
        run_one(hog, config, Task1ClassicalRunConfig.full())
        for hog in TASK1_HOG_SPECS
        for config in (TASK1_DEFAULT_KNN, TASK1_DEFAULT_SVM)
    )
    selected_hog = _select_shared_hog(_classical_metrics_frame(defaults))
    completed = {result.candidate_id for result in defaults}
    tuned = list(defaults)
    for config in (*TASK1_KNN_GRID, *TASK1_SVM_GRID):
        candidate_id = f"{selected_hog.hog_id}-{config.config_id}"
        if candidate_id not in completed:
            try:
                result = run_one(selected_hog, config, Task1ClassicalRunConfig.full())
            except ConvergenceWarning:
                if not isinstance(config, Task1LinearSVMConfig) or config == TASK1_DEFAULT_SVM:
                    raise
                continue
            tuned.append(result)
            completed.add(candidate_id)
    results = tuple(tuned)
    tuning = _classical_metrics_frame(results)
    selected_metrics = tuning.loc[tuning["hog_id"].eq(selected_hog.hog_id)]
    chosen = Task1ClassicalSelection(
        selected_hog.hog_id,
        _select_model_config(selected_metrics, {c.config_id for c in TASK1_KNN_GRID}),
        _select_model_config(selected_metrics, {c.config_id for c in TASK1_SVM_GRID}),
    )
    tuning_path = atomic_write_csv(output_root / "classical_tuning.csv", tuning)
    atomic_write_json(
        output_root / "classical_selection.json",
        _classical_selection_payload(
            chosen, tuning_path=tuning_path, splits=splits, label_map=label_map
        ),
    )
    return Task1ClassicalExperimentResult(
        stage, results, tuning, chosen, tuning, pd.DataFrame(), pd.DataFrame(), {}, {}
    )
