"""Smoke and five-fold experiment orchestration for Task 1."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from fashion.config import (
    ROOT,
    TASK1_EVIDENCE_DIR,
    TASK1_FIGURE_DIR,
    TASK1_HOG_CACHE_DIR,
    TASK1_RESULT_DIR,
)
from fashion.data.dataset import get_samples
from fashion.data.hashing import compute_sha256
from fashion.task1.classical import (
    TASK1_DEFAULT_KNN,
    TASK1_DEFAULT_SVM,
    TASK1_HOG_COARSE,
    TASK1_HOG_SPECS,
    TASK1_KNN_GRID,
    TASK1_SVM_GRID,
    Task1ClassicalFoldResult,
    Task1ClassicalModelConfig,
    Task1ClassicalRunConfig,
    Task1HogSpec,
    run_task1_classical_fold,
)
from fashion.task1.evaluation import (
    aggregate_fold_metrics,
    classification_metrics,
    per_class_metrics,
    validate_oof_predictions,
)
from fashion.task1.preprocessing import DEFAULT_TASK1_PREPROCESSING, TASK1_CONTROL_PREPROCESSING
from fashion.task1.training import Task1FoldResult, Task1TrainConfig, train_task1_fold
from fashion.train.artifacts import atomic_write_csv, atomic_write_json, canonical_sha256
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


def _fold_metrics_frame(results: Sequence[Task1FoldResult]) -> pd.DataFrame:
    """Make one auditable metrics row for each physical fold run."""
    return pd.DataFrame(
        [
            {
                "run_id": result.run_id,
                "fold": result.fold,
                "preprocessing_id": result.preprocessing_id,
                **result.metrics,
            }
            for result in results
        ]
    )


def _classical_metrics_frame(results: Sequence[Task1ClassicalFoldResult]) -> pd.DataFrame:
    """Make one auditable row for every classic fold candidate."""
    return pd.DataFrame(
        [
            {
                "run_id": result.run_id,
                "fold": result.fold,
                "candidate_id": result.candidate_id,
                "hog_id": result.hog_id,
                "model_family": result.model_family,
                **result.metrics,
            }
            for result in results
        ]
    )


def _rank_classical_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return candidates in the fixed, deterministic model-selection order."""
    ranking = ["macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy"]
    required = {"candidate_id", *ranking}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"classic candidate metrics are missing columns: {sorted(missing)}")
    return frame.sort_values(
        ranking + ["candidate_id"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _select_shared_hog(default_metrics: pd.DataFrame) -> Task1HogSpec:
    """Select a HOG setting from the two default-model macro-F1 means."""
    required = {"hog_id", "macro_f1"}
    missing = required.difference(default_metrics.columns)
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
    """Choose one configuration using the project-wide, deterministic ranking rule."""
    candidate_metrics = metrics.loc[
        metrics["candidate_id"].map(
            lambda candidate_id: any(
                str(candidate_id).endswith(f"-{config_id}") for config_id in config_ids
            )
        )
    ]
    if candidate_metrics.empty:
        raise ValueError("no completed classic candidates match the requested model family")
    candidate_id = str(_rank_classical_candidates(candidate_metrics).iloc[0]["candidate_id"])
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


def _classical_selection_payload(
    selection: Task1ClassicalSelection,
    *,
    tuning_path: Path,
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
) -> dict[str, str]:
    """Record selected IDs and immutable input/output provenance for final execution."""
    return {
        "hog_id": selection.hog_id,
        "knn_config_id": selection.knn_config_id,
        "svm_config_id": selection.svm_config_id,
        "tuning_sha256": compute_sha256(tuning_path),
        "split_sha256": _split_sha256(splits),
        "label_map_sha256": canonical_sha256(label_map),
        "implementation_sha256": compute_sha256(Path(__file__)),
    }


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
    """Run classic fold-0 smoke checks or the staged HOG/model tuning schedule."""
    if stage not in {"smoke", "tune", "final"}:
        raise ValueError("stage must be 'smoke', 'tune', or 'final'")
    if stage == "final":
        raise ValueError("stage='final' is not implemented until Task 6")

    def run_one(
        hog_spec: Task1HogSpec,
        model_config: Task1ClassicalModelConfig,
        run_config: Task1ClassicalRunConfig,
    ) -> Task1ClassicalFoldResult:
        return fold_runner(
            splits,
            label_map,
            validation_fold=0,
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
        metrics = _classical_metrics_frame(results)
        return Task1ClassicalExperimentResult(
            stage=stage,
            fold_results=results,
            tuning=pd.DataFrame(),
            selection=None,
            fold_metrics=metrics,
            comparison=pd.DataFrame(),
            oof_metrics=pd.DataFrame(),
            oof_predictions={},
            per_class={},
        )

    default_results = tuple(
        run_one(hog_spec, model_config, Task1ClassicalRunConfig.full())
        for hog_spec in TASK1_HOG_SPECS
        for model_config in (TASK1_DEFAULT_KNN, TASK1_DEFAULT_SVM)
    )
    default_metrics = _classical_metrics_frame(default_results)
    selected_hog = _select_shared_hog(default_metrics)
    completed_ids = {result.candidate_id for result in default_results}
    tuned_results = list(default_results)
    for model_config in (*TASK1_KNN_GRID, *TASK1_SVM_GRID):
        candidate_id = f"{selected_hog.hog_id}-{model_config.config_id}"
        if candidate_id not in completed_ids:
            tuned_results.append(
                run_one(selected_hog, model_config, Task1ClassicalRunConfig.full())
            )
            completed_ids.add(candidate_id)

    results = tuple(tuned_results)
    tuning = _classical_metrics_frame(results)
    selected_hog_metrics = tuning.loc[tuning["hog_id"].eq(selected_hog.hog_id)]
    chosen = Task1ClassicalSelection(
        hog_id=selected_hog.hog_id,
        knn_config_id=_select_model_config(
            selected_hog_metrics, {config.config_id for config in TASK1_KNN_GRID}
        ),
        svm_config_id=_select_model_config(
            selected_hog_metrics, {config.config_id for config in TASK1_SVM_GRID}
        ),
    )
    output_root = Path(evidence_root)
    tuning_path = atomic_write_csv(output_root / "classical_tuning.csv", tuning)
    atomic_write_json(
        output_root / "classical_selection.json",
        _classical_selection_payload(
            chosen,
            tuning_path=tuning_path,
            splits=splits,
            label_map=label_map,
        ),
    )
    return Task1ClassicalExperimentResult(
        stage=stage,
        fold_results=results,
        tuning=tuning,
        selection=chosen,
        fold_metrics=tuning,
        comparison=pd.DataFrame(),
        oof_metrics=pd.DataFrame(),
        oof_predictions={},
        per_class={},
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
    preprocessing_ids: Sequence[str],
) -> pd.DataFrame:
    """Aggregate each candidate's five physical folds into one comparison row."""
    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in {"run_id", "fold", "preprocessing_id"}
    ]
    rows: list[dict[str, float | str]] = []
    for preprocessing_id in preprocessing_ids:
        candidate_rows = fold_metrics.loc[
            fold_metrics["preprocessing_id"].eq(preprocessing_id), metric_columns
        ]
        candidate_folds = fold_metrics.loc[
            fold_metrics["preprocessing_id"].eq(preprocessing_id), "fold"
        ]
        if set(candidate_folds.astype(int)) != set(range(5)) or len(candidate_folds) != 5:
            raise ValueError(
                "full Task 1 evidence requires exactly five folds with unique labels 0,1,2,3,4"
            )
        summary = aggregate_fold_metrics(candidate_rows.to_dict(orient="records"))
        row: dict[str, float | str] = {"preprocessing_id": preprocessing_id}
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
    for preprocessing_id, frame in per_class.items():
        atomic_write_csv(TASK1_EVIDENCE_DIR / f"per_class_{preprocessing_id}.csv", frame)


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

    candidates = (TASK1_CONTROL_PREPROCESSING, DEFAULT_TASK1_PREPROCESSING)
    if mode == "smoke":
        schedule = ((TASK1_CONTROL_PREPROCESSING, 0, Task1TrainConfig.smoke()),)
    else:
        schedule = tuple(
            (preprocessing, fold, Task1TrainConfig.full())
            for preprocessing in candidates
            for fold in range(5)
        )

    fold_results = tuple(
        fold_runner(
            splits,
            label_map,
            validation_fold=fold,
            preprocessing=preprocessing,
            config=config,
            registry=registry,
            root=root,
            result_root=result_root,
        )
        for preprocessing, fold, config in schedule
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

    preprocessing_ids = [candidate.preprocessing_id for candidate in candidates]
    comparison = _aggregate_comparison(fold_metrics, preprocessing_ids)
    expected_ids = get_samples(splits, partition="development", target="articleType")["id"].tolist()
    class_names = _class_names(label_map)
    oof_predictions: dict[str, pd.DataFrame] = {}
    oof_metrics_rows: list[dict[str, float | str]] = []
    per_class: dict[str, pd.DataFrame] = {}
    for preprocessing_id in preprocessing_ids:
        candidate_results = [
            result for result in fold_results if result.preprocessing_id == preprocessing_id
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
                "preprocessing_id": preprocessing_id,
                "macro_f1_124": pooled["macro_f1"],
                "weighted_f1": pooled["weighted_f1"],
                "top1_accuracy": pooled["top1_accuracy"],
                "top5_accuracy": pooled["top5_accuracy"],
            }
        )
        oof_predictions[preprocessing_id] = predictions
        per_class[preprocessing_id] = per_class_metrics(
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


def write_task1_comparison_figure(
    fold_metrics: pd.DataFrame,
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_preprocessing_macro_f1.png",
) -> Path:
    """Write a fold-level macro-F1 comparison with sample-standard-deviation bars."""
    required = {"preprocessing_id", "fold", "macro_f1"}
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold_metrics are missing columns: {sorted(missing)}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    grouped = list(fold_metrics.groupby("preprocessing_id", sort=False))
    for position, (preprocessing_id, candidate) in enumerate(grouped):
        values = candidate["macro_f1"].to_numpy(dtype=float)
        if len(values) != 5:
            raise ValueError("comparison figure requires exactly five folds per preprocessing ID")
        jitter = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            alpha=0.8,
            label=preprocessing_id,
        )
        axis.errorbar(
            position,
            values.mean(),
            yerr=values.std(ddof=1),
            color="black",
            capsize=6,
            fmt="D",
            zorder=3,
        )
    axis.set_xticks(range(len(grouped)), [name for name, _ in grouped], rotation=15, ha="right")
    axis.set_xlabel("Preprocessing candidate")
    axis.set_ylabel("Validation macro-F1 (124 classes)")
    axis.set_title("Task 1 five-fold preprocessing comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_task1_confusion_figure(
    predictions: pd.DataFrame,
    class_names: Sequence[str],
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_oof_confusion_matrix.png",
) -> Path:
    """Write a normalized 124-class OOF confusion matrix after full CV is available."""
    if len(class_names) != 124:
        raise ValueError("Task 1 confusion evidence requires exactly 124 class names")
    required = {"true_index", "predicted_index"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    matrix = confusion_matrix(
        predictions["true_index"],
        predictions["predicted_index"],
        labels=np.arange(124),
        normalize="true",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(22, 20))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Within-class proportion")
    ticks = np.arange(124)
    axis.set_xticks(ticks, class_names, rotation=90, fontsize=4)
    axis.set_yticks(ticks, class_names, fontsize=4)
    axis.set_xlabel("Predicted article type")
    axis.set_ylabel("True article type")
    axis.set_title("Task 1 out-of-fold normalized confusion matrix")
    figure.text(0.5, 0.01, "Rows with no true examples are shown as zeros.", ha="center")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return output_path
