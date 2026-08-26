"""Traceable Task 2 file-impact evidence and an HTML-safe Matplotlib flow diagram."""

from __future__ import annotations

import io
import os
import re
from collections.abc import Collection, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.hashing import compute_sha256
from fashion.task2.smoke import G0SmokeResult
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
)
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics, validate_oof
from fashion.train.registry import RunRegistry

if TYPE_CHECKING:
    from fashion.task2.experiments import ExperimentFoldOutput

FILE_IMPACT_COLUMNS = ("producer", "artifact", "consumer", "effect", "phase")
FROZEN_BUNDLE = "Frozen Season bundle"
DEPLOYMENT_CONSUMERS = frozenset(
    {"Notebook 06 holdout", "Prediction CLI", "Streamlit app"}
)
TRAINING_NODES = frozenset(
    {"fashion.data.torch", "fashion.models.season", "fashion.train.engine", "Task 2 runner"}
)
EXPERIMENT_REGISTRY_COLUMNS = (
    "run_id",
    "stage",
    "experiment_id",
    "model_family",
    "benchmark_only",
    "final_eligible",
    "scratch",
    "fold",
    "seed",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "split_sha256",
    "label_map_sha256",
    "implementation_sha256",
    "transform_id",
    "loss_id",
    "epochs_requested",
    "epochs_completed",
    "best_epoch",
    "primary_metric_name",
    "primary_metric_value",
    "runtime_seconds",
    "peak_vram_mb",
    "parameter_count",
    "checkpoint_sha256",
    "prediction_sha256",
    "history_sha256",
    "status",
)
SCALAR_METRICS = (
    "n_samples",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "nll",
    "brier",
    "ece",
)


def build_file_impact_edges() -> pd.DataFrame:
    """Return the declared producer-artifact-consumer contract in execution order."""
    rows = [
        (
            "data/processed/splits.csv",
            "canonical partitions and cv_fold",
            "fashion.data.torch",
            "Changing membership or folds invalidates every comparable run.",
            "inputs",
        ),
        (
            "data/processed/label_maps.json",
            "Fall, Spring, Summer, Winter class order",
            "fashion.data.torch",
            "Changing class indices invalidates models, metrics, and predictions.",
            "inputs",
        ),
        (
            "fashion.data.torch",
            "fold-fitted tensors and loaders",
            "Task 2 runner",
            "Transform changes create a new implementation and transform hash.",
            "training",
        ),
        (
            "configs/task2/*.json",
            "immutable experiment choices",
            "Task 2 runner",
            "A changed scientific choice creates a new config hash and run.",
            "training",
        ),
        (
            "fashion.models.season",
            "scratch logits and feature maps",
            "Task 2 runner",
            "Architecture changes invalidate checkpoints and OOF evidence.",
            "training",
        ),
        (
            "fashion.train.engine",
            "best checkpoint, OOF predictions, and history",
            "Task 2 runner",
            "Optimisation changes invalidate the implementation hash.",
            "training",
        ),
        (
            "Task 2 runner",
            "run lifecycle and provenance row",
            "results/runs.csv",
            "Every physical run is retained as completed, failed, or interrupted.",
            "artifacts",
        ),
        (
            "Task 2 runner",
            "checkpoint, OOF, and history files",
            "tmp/task2 artifacts",
            "Bytes are reused only when their declared SHA-256 values match.",
            "artifacts",
        ),
        (
            "results/runs.csv",
            "run IDs, hashes, metrics, and status",
            "results/evidence/task2",
            "Registry changes rebuild evidence but do not alter trained weights.",
            "evidence",
        ),
        (
            "tmp/task2 artifacts",
            "verified OOF and histories",
            "results/evidence/task2",
            "Only hash-verified outputs may support a measured claim.",
            "evidence",
        ),
        (
            "results/evidence/task2",
            "tables, figures, run IDs, and limitations",
            "Notebook 03",
            "Notebook claims must trace to measured run IDs.",
            "narrative",
        ),
        (
            "results/evidence/task2",
            "selected tables, figures, and limitations",
            "Task 2 report",
            "Report text changes interpretation only, never model bytes.",
            "narrative",
        ),
        (
            "Notebook 03",
            "predeclared scorecard decision",
            "selection_freeze.json",
            "The selected run and rules become immutable before holdout access.",
            "freeze",
        ),
        (
            "selection_freeze.json",
            "winner config, hashes, and median epoch rule",
            "Development refit",
            "Refit must reproduce the frozen method on all valid development rows.",
            "freeze",
        ),
        (
            "Development refit",
            "task2_season.pt and manifest",
            FROZEN_BUNDLE,
            "The manifest binds model bytes, labels, transform, and config by SHA-256.",
            "bundle",
        ),
        (
            FROZEN_BUNDLE,
            "immutable model plus manifest",
            "Notebook 06 holdout",
            "Holdout evaluates the bundle once and cannot tune it.",
            "deployment",
        ),
        (
            FROZEN_BUNDLE,
            "immutable model plus manifest",
            "Prediction CLI",
            "Official Season predictions must come from the frozen bytes.",
            "deployment",
        ),
        (
            FROZEN_BUNDLE,
            "immutable model plus manifest",
            "Streamlit app",
            "The app reports the same model/config hash as offline inference.",
            "deployment",
        ),
    ]
    frame = pd.DataFrame(rows, columns=FILE_IMPACT_COLUMNS)
    validate_file_impact_edges(frame)
    return frame


def validate_file_impact_edges(frame: pd.DataFrame) -> None:
    """Fail closed on missing nodes, unsafe feedback, or alternative deployment inputs."""
    if tuple(frame.columns) != FILE_IMPACT_COLUMNS:
        raise ValueError(f"file-impact columns must be {list(FILE_IMPACT_COLUMNS)}")
    if frame.empty or frame.isna().any().any():
        raise ValueError("file-impact edges must be non-empty and complete")
    if frame.duplicated(["producer", "artifact", "consumer"]).any():
        raise ValueError("file-impact edges must be unique")
    nodes = set(frame["producer"]) | set(frame["consumer"])
    required = {
        "data/processed/splits.csv",
        "data/processed/label_maps.json",
        "fashion.data.torch",
        "configs/task2/*.json",
        "fashion.models.season",
        "fashion.train.engine",
        "Task 2 runner",
        "results/runs.csv",
        "tmp/task2 artifacts",
        "results/evidence/task2",
        "Notebook 03",
        "Task 2 report",
        "selection_freeze.json",
        "Development refit",
        FROZEN_BUNDLE,
        *DEPLOYMENT_CONSUMERS,
    }
    missing = sorted(required - nodes)
    if missing:
        raise ValueError(f"file-impact flow is missing nodes: {missing}")
    unsafe_holdout = frame.loc[
        frame["producer"].astype(str).str.contains("holdout", case=False)
        & frame["consumer"].isin(TRAINING_NODES)
    ]
    if not unsafe_holdout.empty:
        raise ValueError("holdout information cannot flow into training")
    for consumer in DEPLOYMENT_CONSUMERS:
        producers = set(frame.loc[frame["consumer"].eq(consumer), "producer"])
        if producers != {FROZEN_BUNDLE}:
            raise ValueError(f"{consumer} must consume only the frozen bundle")
    evidence_feedback = frame.loc[
        frame["producer"].eq("results/evidence/task2")
        & frame["consumer"].isin(TRAINING_NODES | {"Development refit"})
    ]
    if not evidence_feedback.empty:
        raise ValueError("descriptive evidence cannot modify training or refit")


def _flow_positions() -> dict[str, tuple[float, float]]:
    return {
        "data/processed/splits.csv": (0.7, 8.8),
        "data/processed/label_maps.json": (0.7, 7.1),
        "configs/task2/*.json": (0.7, 5.4),
        "fashion.data.torch": (3.3, 8.8),
        "fashion.models.season": (3.3, 7.1),
        "fashion.train.engine": (3.3, 5.4),
        "Task 2 runner": (6.0, 7.1),
        "results/runs.csv": (8.7, 8.3),
        "tmp/task2 artifacts": (8.7, 6.0),
        "results/evidence/task2": (11.4, 7.1),
        "Notebook 03": (14.0, 8.3),
        "Task 2 report": (14.0, 6.0),
        "selection_freeze.json": (16.6, 8.3),
        "Development refit": (16.6, 6.0),
        FROZEN_BUNDLE: (16.6, 3.7),
        "Notebook 06 holdout": (14.0, 1.2),
        "Prediction CLI": (16.6, 1.2),
        "Streamlit app": (19.2, 1.2),
    }


def _node_colour(node: str) -> str:
    if node in DEPLOYMENT_CONSUMERS:
        return "#FDE68A"
    if node == FROZEN_BUNDLE:
        return "#86EFAC"
    if node in {"selection_freeze.json", "Development refit"}:
        return "#C4B5FD"
    if node in {"Notebook 03", "Task 2 report", "results/evidence/task2"}:
        return "#BAE6FD"
    if node in {"results/runs.csv", "tmp/task2 artifacts"}:
        return "#FED7AA"
    return "#E2E8F0"


def _portable_artifact_path(path: Path, *, fallback_root: Path) -> str:
    resolved = path.resolve()
    for anchor in (ROOT.resolve(), fallback_root.resolve()):
        try:
            return resolved.relative_to(anchor).as_posix()
        except ValueError:
            continue
    return path.name


def plot_file_impact_flow(
    edges: pd.DataFrame | None = None,
    *,
    output_path: str | Path = TASK2_FIGURE_DIR / "file_impact_flow.png",
) -> tuple[Figure, Axes]:
    """Render and atomically save the notebook-safe file-impact flow as PNG."""
    frame = build_file_impact_edges() if edges is None else edges.copy()
    validate_file_impact_edges(frame)
    positions = _flow_positions()
    unknown = (set(frame["producer"]) | set(frame["consumer"])) - set(positions)
    if unknown:
        raise ValueError(f"file-impact layout is missing positions: {sorted(unknown)}")

    figure = Figure(figsize=(20, 10.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.set_xlim(-0.7, 20.7)
    axis.set_ylim(-0.2, 10.2)
    axis.axis("off")
    axis.set_title(
        "Task 2 file impact: data and code create runs; evidence explains frozen bytes",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    for row in frame.itertuples(index=False):
        start = positions[row.producer]
        end = positions[row.consumer]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#64748B",
            connectionstyle="arc3,rad=0.04",
            shrinkA=46,
            shrinkB=46,
            zorder=1,
        )
        axis.add_patch(arrow)
    for node, (x, y) in positions.items():
        width = 2.15 if node not in {FROZEN_BUNDLE, "results/evidence/task2"} else 2.35
        box = FancyBboxPatch(
            (x - width / 2, y - 0.42),
            width,
            0.84,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=_node_colour(node),
            edgecolor="#334155",
            linewidth=1.2,
            zorder=2,
        )
        axis.add_patch(box)
        axis.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="semibold",
            wrap=True,
            zorder=3,
        )
    axis.text(
        0.0,
        0.05,
        "Rule: evidence and narrative may explain a model, but cannot flow back into training.",
        fontsize=10,
        color="#475569",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    return figure, axis


def build_task2_evidence(
    *,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Build the static traceability evidence used before measured experiments exist."""
    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    edges_path = evidence_root / "file_impact_edges.csv"
    figure_path = figure_root / "file_impact_flow.png"
    manifest_path = evidence_root / "file_impact_manifest.json"
    common_root = Path(
        os.path.commonpath([evidence_root.resolve(), figure_root.resolve()])
    )
    edges = build_file_impact_edges()
    atomic_write_csv(edges_path, edges)
    figure, _ = plot_file_impact_flow(edges, output_path=figure_path)
    figure.clear()
    manifest = {
        "schema_version": "1.0.0",
        "scope": "task2_file_impact",
        "edge_count": len(edges),
        "node_count": len(set(edges["producer"]) | set(edges["consumer"])),
        "edges_path": _portable_artifact_path(edges_path, fallback_root=common_root),
        "edges_sha256": compute_sha256(edges_path),
        "figure_path": _portable_artifact_path(figure_path, fallback_root=common_root),
        "figure_sha256": compute_sha256(figure_path),
        "holdout_to_training_edges": 0,
        "deployment_input": FROZEN_BUNDLE,
    }
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


def _validate_experiment_outputs(
    outputs: Sequence[ExperimentFoldOutput],
    *,
    registry: RunRegistry,
    expected_folds: tuple[int, ...],
) -> tuple[str, int, pd.DataFrame]:
    if not outputs:
        raise ValueError("experiment evidence requires at least one fold output")
    if len(set(expected_folds)) != len(expected_folds):
        raise ValueError("expected_folds must be unique")
    experiment_ids = {output.experiment_id for output in outputs}
    seeds = {output.seed for output in outputs}
    folds = [output.fold for output in outputs]
    run_ids = [output.run_id for output in outputs]
    if len(experiment_ids) != 1 or len(seeds) != 1:
        raise ValueError("one evidence pack may contain one experiment and one seed only")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("experiment evidence contains duplicate run IDs")
    if len(set(folds)) != len(folds) or set(folds) != set(expected_folds):
        raise ValueError(
            "experiment evidence must contain each expected fold exactly once; "
            f"expected={list(expected_folds)}, actual={sorted(folds)}"
        )

    registry_rows = registry.read()
    selected = registry_rows.loc[registry_rows["run_id"].isin(run_ids)].copy()
    if len(selected) != len(run_ids) or set(selected["run_id"]) != set(run_ids):
        raise ValueError("every experiment output must map to one registry row")
    if not selected["status"].eq("completed").all():
        raise ValueError("experiment evidence may use completed registry rows only")

    by_run_id = selected.set_index("run_id", drop=False)
    for output in outputs:
        row = by_run_id.loc[output.run_id]
        expected_identity = {
            "experiment_id": output.experiment_id,
            "fold": str(output.fold),
            "seed": str(output.seed),
            "config_sha256": output.cache_key.config_sha256,
            "split_sha256": output.cache_key.split_sha256,
            "label_map_sha256": output.cache_key.label_map_sha256,
            "implementation_sha256": output.cache_key.implementation_sha256,
        }
        mismatches = [
            name for name, value in expected_identity.items() if row[name] != value
        ]
        for artifact, digest in output.artifacts.items():
            hash_column = f"{artifact}_sha256"
            if hash_column not in row or row[hash_column] != digest:
                mismatches.append(hash_column)
        if mismatches:
            raise ValueError(
                f"registry identity or artifact mismatch for {output.run_id}: "
                f"{sorted(set(mismatches))}"
            )
    return experiment_ids.pop(), seeds.pop(), selected


def _pool_experiment_oof(
    outputs: Sequence[ExperimentFoldOutput],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for output in sorted(outputs, key=lambda item: item.fold):
        frame = output.oof.copy()
        for column, expected in (
            ("run_id", output.run_id),
            ("experiment_id", output.experiment_id),
            ("fold", output.fold),
            ("seed", output.seed),
        ):
            if column in frame:
                observed = set(frame[column].astype(str))
                if observed != {str(expected)}:
                    raise ValueError(
                        f"OOF {column} disagrees with output {output.run_id}: {observed}"
                    )
            else:
                frame.insert(min(len(frame.columns), 3), column, expected)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _experiment_figure(
    *,
    experiment_id: str,
    labels: tuple[str, ...],
    pooled_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    matrix = np.asarray(pooled_metrics["confusion_matrix"], dtype=np.int64)
    class_f1 = np.asarray(
        [pooled_metrics["per_class"][label]["f1"] for label in labels],
        dtype=np.float64,
    )
    figure = Figure(figsize=(12, 5.4), constrained_layout=True)
    FigureCanvasAgg(figure)
    matrix_axis, class_axis = figure.subplots(1, 2)
    image = matrix_axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
    matrix_axis.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    matrix_axis.set_yticks(range(len(labels)), labels=labels)
    matrix_axis.set_xlabel("Predicted Season")
    matrix_axis.set_ylabel("True Season")
    matrix_axis.set_title("Pooled five-fold OOF confusion matrix")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(len(labels)):
        for column in range(len(labels)):
            matrix_axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#0F172A",
                fontsize=9,
            )

    positions = np.arange(len(labels))
    class_axis.barh(positions, class_f1, color="#2563EB")
    class_axis.axvline(
        pooled_metrics["macro_f1"],
        color="#DC2626",
        linestyle="--",
        label=f"macro-F1 = {pooled_metrics['macro_f1']:.3f}",
    )
    class_axis.set_yticks(positions, labels=labels)
    class_axis.set_xlim(0.0, 1.0)
    class_axis.set_xlabel("F1 score")
    class_axis.set_title("Pooled OOF per-class F1")
    class_axis.legend(loc="lower right")
    for index, value in enumerate(class_f1):
        class_axis.text(min(value + 0.02, 0.96), index, f"{value:.3f}", va="center")
    figure.suptitle(
        f"{experiment_id}: discrimination evidence from canonical OOF predictions",
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    figure.clear()


def build_experiment_evidence(
    outputs: Sequence[ExperimentFoldOutput],
    *,
    registry_path: str | Path,
    expected_ids: Collection[Any],
    protected_ids: Collection[Any] = (),
    labels: Sequence[str] = SEASON_LABELS,
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    probability_note: str,
    calibration_claim_allowed: bool,
    evidence_directory: str | Path,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Build a hash-linked OOF evidence pack after strict coverage and registry checks."""
    if not probability_note.strip():
        raise ValueError("probability_note must explain how probabilities were produced")
    ordered_labels = tuple(str(label) for label in labels)
    experiment_id, seed, registry_rows = _validate_experiment_outputs(
        outputs,
        registry=RunRegistry(registry_path),
        expected_folds=expected_folds,
    )
    pooled_oof = _pool_experiment_oof(outputs)
    coverage = validate_oof(
        pooled_oof,
        expected_ids=expected_ids,
        protected_ids=protected_ids,
        labels=ordered_labels,
    )
    probability_columns = [f"prob_{label}" for label in ordered_labels]
    pooled_metrics = multiclass_metrics(
        pooled_oof["y_true"].astype(str).to_numpy(),
        probabilities=pooled_oof.loc[:, probability_columns].to_numpy(dtype=float),
        labels=ordered_labels,
        y_pred=pooled_oof["y_pred"].astype(str).to_numpy(),
    )

    fold_rows = []
    for output in sorted(outputs, key=lambda item: item.fold):
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "run_id": output.run_id,
            "fold": output.fold,
            "seed": output.seed,
            "source": output.source,
        }
        row.update({name: output.metrics[name] for name in SCALAR_METRICS})
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for metric in SCALAR_METRICS:
        values = pd.to_numeric(fold_metrics[metric], errors="raise")
        summary_rows.append(
            {
                "metric": metric,
                "fold_mean": float(values.mean()),
                "fold_sd": float(values.std(ddof=1)),
                "fold_min": float(values.min()),
                "fold_max": float(values.max()),
                "pooled_value": pooled_metrics.get(metric, ""),
            }
        )
    fold_summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame.from_dict(pooled_metrics["per_class"], orient="index")
    per_class.insert(0, "label", per_class.index)
    per_class = per_class.reset_index(drop=True)
    confusion = pd.DataFrame(
        pooled_metrics["confusion_matrix"],
        columns=[f"predicted_{label}" for label in ordered_labels],
    )
    confusion.insert(0, "true_label", ordered_labels)

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(os.path.commonpath([evidence_root.resolve(), figure_root.resolve()]))
    slug = re.sub(r"[^a-z0-9]+", "_", experiment_id.lower()).strip("_")
    if not slug:
        raise ValueError("experiment_id must produce a non-empty evidence slug")
    paths = {
        "registry_snapshot": evidence_root / "registry_snapshot.csv",
        "fold_metrics": evidence_root / "fold_metrics.csv",
        "fold_summary": evidence_root / "fold_summary.csv",
        "pooled_metrics": evidence_root / "pooled_metrics.json",
        "per_class_metrics": evidence_root / "per_class_metrics.csv",
        "confusion_matrix": evidence_root / "confusion_matrix.csv",
        "figure": figure_root / f"{slug}.png",
    }
    snapshot = registry_rows.loc[:, EXPERIMENT_REGISTRY_COLUMNS].copy()
    snapshot["fold"] = pd.to_numeric(snapshot["fold"], errors="raise")
    snapshot = snapshot.sort_values(["seed", "fold", "run_id"]).reset_index(drop=True)
    atomic_write_csv(paths["registry_snapshot"], snapshot)
    atomic_write_csv(paths["fold_metrics"], fold_metrics)
    atomic_write_csv(paths["fold_summary"], fold_summary)
    atomic_write_json(paths["pooled_metrics"], pooled_metrics)
    atomic_write_csv(paths["per_class_metrics"], per_class)
    atomic_write_csv(paths["confusion_matrix"], confusion)
    _experiment_figure(
        experiment_id=experiment_id,
        labels=ordered_labels,
        pooled_metrics=pooled_metrics,
        output_path=paths["figure"],
    )

    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=common_root),
            "sha256": compute_sha256(path),
        }
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "seed": seed,
        "folds": sorted(expected_folds),
        "run_ids": [output.run_id for output in sorted(outputs, key=lambda item: item.fold)],
        "sources": [output.source for output in sorted(outputs, key=lambda item: item.fold)],
        "coverage": coverage,
        "pooled_macro_f1": pooled_metrics["macro_f1"],
        "probability_note": probability_note,
        "calibration_claim_allowed": calibration_claim_allowed,
        "oof_artifact_set_sha256": canonical_sha256(
            {
                output.run_id: output.artifacts["prediction"]
                for output in sorted(outputs, key=lambda item: item.fold)
            }
        ),
        "artifacts": artifact_manifest,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


def build_g0_evidence(
    result: G0SmokeResult,
    *,
    registry_path: str | Path,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR / "g0",
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Write a portable, non-comparison evidence pack for one verified G0 pass."""
    if not result.passed:
        raise ValueError("cannot build passing G0 evidence from a failed result")
    registry_rows = RunRegistry(registry_path).find(run_id=result.run_id)
    if len(registry_rows) != 1:
        raise ValueError(f"G0 run ID must map to one registry row: {result.run_id}")
    registry_row = registry_rows.iloc[0]
    if registry_row["status"] != "completed" or registry_row["stage"] != "g0_smoke":
        raise ValueError("G0 evidence requires one completed g0_smoke registry row")

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(os.path.commonpath([evidence_root.resolve(), figure_root.resolve()]))
    registry_snapshot_path = evidence_root / "registry_snapshot.csv"
    tiny_trace_path = evidence_root / "tiny_loss_trace.csv"
    integration_history_path = evidence_root / "integration_history.csv"
    figure_path = figure_root / "g0_pipeline_smoke.png"
    manifest_path = evidence_root / "manifest.json"

    snapshot_columns = [
        "run_id",
        "stage",
        "experiment_id",
        "model_family",
        "benchmark_only",
        "final_eligible",
        "scratch",
        "fold",
        "seed",
        "git_commit",
        "config_sha256",
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
        "transform_id",
        "loss_id",
        "epochs_requested",
        "epochs_completed",
        "best_epoch",
        "primary_metric_name",
        "primary_metric_value",
        "runtime_seconds",
        "peak_vram_mb",
        "parameter_count",
        "checkpoint_sha256",
        "prediction_sha256",
        "history_sha256",
        "status",
    ]
    registry_snapshot = registry_rows.loc[:, snapshot_columns].copy()
    tiny_trace = pd.DataFrame(result.tiny_overfit["loss_trace"])
    integration_history = pd.DataFrame(result.integration["history"])
    if tiny_trace.empty or integration_history.empty:
        raise ValueError("G0 evidence requires non-empty tiny and integration histories")
    atomic_write_csv(registry_snapshot_path, registry_snapshot)
    atomic_write_csv(tiny_trace_path, tiny_trace)
    atomic_write_csv(integration_history_path, integration_history)

    figure = Figure(figsize=(12, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    tiny_axis, integration_axis = figure.subplots(1, 2)
    tiny_axis.plot(tiny_trace["step"], tiny_trace["train_loss"], marker="o", color="#2563EB")
    tiny_axis.axhline(
        result.tiny_overfit["initial_loss"]
        * result.tiny_overfit["maximum_loss_ratio"],
        linestyle="--",
        color="#DC2626",
        label="maximum passing loss",
    )
    tiny_axis.set_yscale("log")
    tiny_axis.set_title("G0 fixed-batch memorisation")
    tiny_axis.set_xlabel("optimizer step")
    tiny_axis.set_ylabel("cross-entropy loss (log scale)")
    tiny_axis.legend()

    integration_axis.plot(
        integration_history["epoch"],
        integration_history["train_loss"],
        marker="o",
        label="training loss",
        color="#0F766E",
    )
    integration_axis.plot(
        integration_history["epoch"],
        integration_history["validation_loss"],
        marker="o",
        label="validation loss",
        color="#D97706",
    )
    integration_axis.set_title("G0 shared-engine integration")
    integration_axis.set_xlabel("epoch")
    integration_axis.set_ylabel("cross-entropy loss")
    integration_axis.set_xticks(integration_history["epoch"])
    integration_axis.legend()
    figure.suptitle("G0 is a pipeline pass, not model-comparison evidence", fontweight="bold")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(figure_path, buffer.getvalue())
    figure.clear()

    manifest = {
        "schema_version": "1.0.0",
        "gate": "G0",
        "passed": True,
        "comparison_eligible": False,
        "run_id": result.run_id,
        "source": result.source,
        "cache_key": asdict(result.cache_key),
        "tiny_overfit": {
            key: result.tiny_overfit[key]
            for key in (
                "products",
                "steps",
                "device",
                "initial_loss",
                "final_loss",
                "final_accuracy",
                "loss_ratio",
                "minimum_accuracy",
                "maximum_loss_ratio",
                "gradients_finite",
                "passed",
            )
        },
        "integration": {
            key: result.integration[key]
            for key in (
                "training_products",
                "validation_products",
                "epochs_completed",
                "best_epoch",
                "device",
                "peak_vram_mb",
                "runtime_seconds",
            )
        },
        "integration_macro_f1": result.integration["best_metrics"]["macro_f1"],
        "artifact_sha256": dict(result.artifacts),
        "registry_snapshot_path": _portable_artifact_path(
            registry_snapshot_path,
            fallback_root=common_root,
        ),
        "registry_snapshot_sha256": compute_sha256(registry_snapshot_path),
        "tiny_trace_path": _portable_artifact_path(tiny_trace_path, fallback_root=common_root),
        "tiny_trace_sha256": compute_sha256(tiny_trace_path),
        "integration_history_path": _portable_artifact_path(
            integration_history_path,
            fallback_root=common_root,
        ),
        "integration_history_sha256": compute_sha256(integration_history_path),
        "figure_path": _portable_artifact_path(figure_path, fallback_root=common_root),
        "figure_sha256": compute_sha256(figure_path),
    }
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest
