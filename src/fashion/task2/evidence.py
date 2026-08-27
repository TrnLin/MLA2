"""Traceable Task 2 file-impact evidence and an HTML-safe Matplotlib flow diagram."""

from __future__ import annotations

import io
import json
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
G1_FAMILY_EXPERIMENTS = frozenset(
    {
        "g1-c1-smallcnn",
        "g1-c2-resnet18",
        "g1-c3-mobilenetv3",
    }
)
G1_EXPECTED_FOLDS = (0, 1, 2, 3, 4)
G2_SIZE_EXPERIMENTS = {
    "P0": "g1-c2-resnet18",
    "P1": "g2-p1-c2-resnet18",
}
G2_SIZE_EXPECTED_PIXELS = {
    "P0": (80, 60),
    "P1": (128, 96),
}
G2_SIZE_MINIMUM_GAIN = 0.005
G2_AUGMENTATION_EXPERIMENTS = {
    "A0": "g1-c2-resnet18",
    "A1": "g2-a1-c2-resnet18",
}
G2_AUGMENTATION_VALUES = {
    "A0": "a0",
    "A1": "a1",
}
G2_AUGMENTATION_MINIMUM_GAIN = 0.003
G2_AUGMENTATION_MAX_ROBUSTNESS_LOSS = 0.010


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


def _resolve_evidence_path(path: str | Path, *, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _verified_manifest_artifact(
    manifest: dict[str, Any],
    name: str,
    *,
    project_root: Path,
) -> Path:
    artifacts = manifest.get("artifacts", {})
    if name not in artifacts:
        raise ValueError(f"experiment manifest is missing required artifact: {name}")
    declaration = artifacts[name]
    artifact_path = _resolve_evidence_path(
        declaration["path"], project_root=project_root
    )
    if not artifact_path.is_file():
        raise ValueError(f"declared {name} artifact does not exist: {artifact_path}")
    if compute_sha256(artifact_path) != declaration["sha256"]:
        raise ValueError(f"declared {name} artifact hash does not match its bytes")
    return artifact_path


def _load_verified_experiment_manifest(
    manifest_path: str | Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], Path]:
    resolved_manifest = _resolve_evidence_path(manifest_path, project_root=project_root)
    with resolved_manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required_artifacts = {"pooled_metrics", "fold_summary", "registry_snapshot"}
    artifacts = manifest.get("artifacts", {})
    missing = required_artifacts - set(artifacts)
    if missing:
        raise ValueError(
            f"experiment manifest is missing required artifacts: {sorted(missing)}"
        )
    for name in required_artifacts:
        _verified_manifest_artifact(manifest, name, project_root=project_root)
    return manifest, resolved_manifest


def _g1_family_row(
    manifest: dict[str, Any],
    *,
    project_root: Path,
    expected_coverage_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    experiment_id = str(manifest.get("experiment_id", ""))
    if experiment_id not in G1_FAMILY_EXPERIMENTS:
        raise ValueError(f"unexpected G1 family experiment: {experiment_id}")
    if tuple(manifest.get("folds", ())) != G1_EXPECTED_FOLDS:
        raise ValueError(f"{experiment_id} must contain canonical folds 0-4")
    if int(manifest.get("seed", -1)) != 2753:
        raise ValueError(f"{experiment_id} must use the primary seed 2753")
    coverage = manifest.get("coverage", {})
    if not (
        coverage.get("row_count")
        == coverage.get("unique_id_count")
        == coverage.get("expected_row_count")
    ):
        raise ValueError(f"{experiment_id} has incomplete OOF coverage")
    if int(coverage.get("protected_id_count", -1)) != 0:
        raise ValueError(f"{experiment_id} includes protected IDs")
    coverage_sha256 = canonical_sha256(
        {
            "row_count": coverage.get("row_count"),
            "id_set_sha256": coverage.get("id_set_sha256"),
            "labels": coverage.get("labels"),
        }
    )
    if expected_coverage_sha256 and coverage_sha256 != expected_coverage_sha256:
        raise ValueError("G1 manifests do not describe the same OOF products and labels")

    artifacts = manifest["artifacts"]
    with _resolve_evidence_path(
        artifacts["pooled_metrics"]["path"], project_root=project_root
    ).open(encoding="utf-8") as handle:
        pooled = json.load(handle)
    fold_summary = pd.read_csv(
        _resolve_evidence_path(
            artifacts["fold_summary"]["path"], project_root=project_root
        )
    )
    registry = pd.read_csv(
        _resolve_evidence_path(
            artifacts["registry_snapshot"]["path"], project_root=project_root
        ),
        dtype=str,
        keep_default_na=False,
    )
    run_ids = [str(run_id) for run_id in manifest.get("run_ids", ())]
    if len(registry) != 5 or set(registry["run_id"]) != set(run_ids):
        raise ValueError(f"{experiment_id} registry snapshot must match five run IDs")
    if set(pd.to_numeric(registry["fold"], errors="raise")) != set(G1_EXPECTED_FOLDS):
        raise ValueError(f"{experiment_id} registry snapshot has invalid folds")
    required_registry_values = {
        "experiment_id": experiment_id,
        "benchmark_only": "false",
        "final_eligible": "true",
        "scratch": "true",
        "status": "completed",
    }
    for column, expected in required_registry_values.items():
        observed = set(registry[column].astype(str).str.lower())
        if observed != {expected}:
            raise ValueError(
                f"{experiment_id} registry {column} must be {expected}: {observed}"
            )
    model_families = set(registry["model_family"].astype(str))
    parameter_counts = set(
        pd.to_numeric(registry["parameter_count"], errors="raise").astype(int)
    )
    if len(model_families) != 1 or len(parameter_counts) != 1:
        raise ValueError(f"{experiment_id} changed model identity across folds")
    macro_summary = fold_summary.loc[fold_summary["metric"].eq("macro_f1")]
    if len(macro_summary) != 1:
        raise ValueError(f"{experiment_id} requires one macro-F1 fold summary row")
    pooled_macro_f1 = float(pooled["macro_f1"])
    if not np.isclose(pooled_macro_f1, float(manifest["pooled_macro_f1"])):
        raise ValueError(f"{experiment_id} pooled macro-F1 disagrees with its manifest")
    spring_f1 = float(pooled["per_class"]["Spring"]["f1"])
    summary = macro_summary.iloc[0]
    return (
        {
            "experiment_id": experiment_id,
            "model_family": model_families.pop(),
            "pooled_macro_f1": pooled_macro_f1,
            "fold_mean_macro_f1": float(summary["fold_mean"]),
            "fold_sd_macro_f1": float(summary["fold_sd"]),
            "spring_f1": spring_f1,
            "parameter_count": parameter_counts.pop(),
            "five_fold_runtime_minutes": float(
                pd.to_numeric(registry["runtime_seconds"], errors="raise").sum() / 60.0
            ),
            "peak_vram_mb": float(
                pd.to_numeric(registry["peak_vram_mb"], errors="raise").max()
            ),
        },
        coverage_sha256,
    )


def _plot_g1_family_screen(
    leaderboard: pd.DataFrame,
    *,
    reference_macro_f1: float | None,
    output_path: Path,
) -> None:
    figure = Figure(figsize=(10.5, 6.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    colours = [
        "#16A34A" if value else "#94A3B8" for value in leaderboard["shortlisted"]
    ]
    sizes = 100.0 + 12.0 * leaderboard["five_fold_runtime_minutes"].to_numpy()
    axis.scatter(
        leaderboard["parameter_count"],
        leaderboard["pooled_macro_f1"],
        s=sizes,
        c=colours,
        edgecolor="#0F172A",
        linewidth=0.8,
        alpha=0.9,
    )
    maximum_parameters = int(leaderboard["parameter_count"].max())
    for row in leaderboard.itertuples(index=False):
        is_rightmost = int(row.parameter_count) == maximum_parameters
        axis.annotate(
            f"{row.experiment_id}\n{row.pooled_macro_f1:.3f}",
            (row.parameter_count, row.pooled_macro_f1),
            xytext=(-7, 7) if is_rightmost else (7, 7),
            textcoords="offset points",
            ha="right" if is_rightmost else "left",
            fontsize=9,
        )
    if reference_macro_f1 is not None:
        axis.axhline(
            reference_macro_f1,
            color="#DC2626",
            linestyle="--",
            linewidth=1.3,
            label=f"B1 pooled macro-F1 = {reference_macro_f1:.3f}",
        )
        axis.legend(loc="lower right")
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Pooled five-fold OOF macro-F1")
    axis.set_title("G1 scratch family screen: quality, size, and training cost")
    axis.grid(alpha=0.2)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    figure.clear()


def build_g1_family_screen_evidence(
    experiment_manifest_paths: Sequence[str | Path],
    *,
    reference_manifest_path: str | Path | None = None,
    project_root: str | Path = ROOT,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR / "g1_family_screen",
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Build the G1 leaderboard and select the top two deep families by pooled OOF F1."""
    root = Path(project_root)
    loaded: list[tuple[dict[str, Any], Path]] = [
        _load_verified_experiment_manifest(path, project_root=root)
        for path in experiment_manifest_paths
    ]
    experiment_ids = [str(manifest.get("experiment_id", "")) for manifest, _ in loaded]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("G1 family manifest paths must be unique")
    if set(experiment_ids) != set(G1_FAMILY_EXPERIMENTS):
        raise ValueError(
            "G1 family screen requires exactly C1 SmallCNN, C2 ResNet18, and "
            "C3 MobileNetV3-Small"
        )

    rows: list[dict[str, Any]] = []
    coverage_sha256: str | None = None
    for manifest, _ in loaded:
        row, observed_coverage_sha256 = _g1_family_row(
            manifest,
            project_root=root,
            expected_coverage_sha256=coverage_sha256,
        )
        coverage_sha256 = coverage_sha256 or observed_coverage_sha256
        rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values(
        ["pooled_macro_f1", "parameter_count"], ascending=[False, True]
    )
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    leaderboard["shortlisted"] = leaderboard["rank"].le(2)
    best_score = float(leaderboard.loc[0, "pooled_macro_f1"])
    leaderboard["delta_vs_best_macro_f1"] = (
        leaderboard["pooled_macro_f1"] - best_score
    )

    reference_macro_f1: float | None = None
    reference_manifest_sha256: str | None = None
    if reference_manifest_path is not None:
        reference, resolved_reference = _load_verified_experiment_manifest(
            reference_manifest_path, project_root=root
        )
        if reference.get("experiment_id") != "b1-hog-hsv-svm":
            raise ValueError("G1 reference manifest must be B1 HOG-HSV LinearSVC")
        reference_coverage = reference.get("coverage", {})
        reference_coverage_sha256 = canonical_sha256(
            {
                "row_count": reference_coverage.get("row_count"),
                "id_set_sha256": reference_coverage.get("id_set_sha256"),
                "labels": reference_coverage.get("labels"),
            }
        )
        if reference_coverage_sha256 != coverage_sha256:
            raise ValueError("B1 and G1 evidence do not cover the same OOF products")
        reference_macro_f1 = float(reference["pooled_macro_f1"])
        reference_manifest_sha256 = compute_sha256(resolved_reference)
        leaderboard["gain_over_b1_macro_f1"] = (
            leaderboard["pooled_macro_f1"] - reference_macro_f1
        )

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(
        os.path.commonpath([evidence_root.resolve(), figure_root.resolve()])
    )
    leaderboard_path = evidence_root / "leaderboard.csv"
    shortlist_path = evidence_root / "shortlist.json"
    figure_path = figure_root / "g1_family_screen.png"
    manifest_path = evidence_root / "manifest.json"
    selected = leaderboard.loc[leaderboard["shortlisted"], "experiment_id"].tolist()
    rejected = leaderboard.loc[~leaderboard["shortlisted"], "experiment_id"].tolist()
    shortlist = {
        "schema_version": "1.0.0",
        "gate": "G1",
        "decision_status": "closed",
        "primary_metric": "pooled_five_fold_oof_macro_f1",
        "selection_rule": (
            "Select the two deep families with the highest pooled OOF macro-F1."
        ),
        "selected_experiment_ids": selected,
        "rejected_experiment_ids": rejected,
        "next_question": (
            "Run P0-P1 and A0-A1 transform ablations on leading C2 while retaining "
            "C1 for the full-budget G3 comparison."
        ),
    }
    atomic_write_csv(leaderboard_path, leaderboard)
    atomic_write_json(shortlist_path, shortlist)
    _plot_g1_family_screen(
        leaderboard,
        reference_macro_f1=reference_macro_f1,
        output_path=figure_path,
    )
    artifacts = {
        "leaderboard": leaderboard_path,
        "shortlist": shortlist_path,
        "figure": figure_path,
    }
    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=common_root),
            "sha256": compute_sha256(path),
        }
        for name, path in artifacts.items()
    }
    input_manifests = {
        str(manifest["experiment_id"]): {
            "path": _portable_artifact_path(path, fallback_root=root),
            "sha256": compute_sha256(path),
        }
        for manifest, path in loaded
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G1",
        "decision_status": "closed",
        "coverage_sha256": coverage_sha256,
        "primary_metric": "pooled_five_fold_oof_macro_f1",
        "selected_experiment_ids": selected,
        "rejected_experiment_ids": rejected,
        "input_manifests": input_manifests,
        "reference": {
            "experiment_id": "b1-hog-hsv-svm" if reference_manifest_path else None,
            "pooled_macro_f1": reference_macro_f1,
            "manifest_sha256": reference_manifest_sha256,
        },
        "artifacts": artifact_manifest,
    }
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


def _g2_transform_row(
    manifest: dict[str, Any],
    *,
    variant: str,
    config_path: str | Path,
    project_root: Path,
    expected_coverage_sha256: str | None,
    expected_experiment_id: str,
    expected_stage: str,
    expected_image_size: tuple[int, int],
    expected_augmentation: str,
    changed_data_field: str,
    comparison_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, str, str, Path]:
    """Load one matched G2 transform row and verify every reusable artifact."""
    from fashion.task2.experiments import load_experiment_config

    if changed_data_field not in {"image_size", "augmentation"}:
        raise ValueError(f"unsupported G2 transform field: {changed_data_field}")
    experiment_id = str(manifest.get("experiment_id", ""))
    if experiment_id != expected_experiment_id:
        raise ValueError(
            f"{variant} manifest must be {expected_experiment_id}: {experiment_id}"
        )
    if tuple(manifest.get("folds", ())) != G1_EXPECTED_FOLDS:
        raise ValueError(f"{experiment_id} must contain canonical folds 0-4")
    if int(manifest.get("seed", -1)) != 2753:
        raise ValueError(f"{experiment_id} must use the primary seed 2753")

    coverage = manifest.get("coverage", {})
    if not (
        coverage.get("row_count")
        == coverage.get("unique_id_count")
        == coverage.get("expected_row_count")
    ):
        raise ValueError(f"{experiment_id} has incomplete OOF coverage")
    if int(coverage.get("protected_id_count", -1)) != 0:
        raise ValueError(f"{experiment_id} includes protected IDs")
    coverage_sha256 = canonical_sha256(
        {
            "row_count": coverage.get("row_count"),
            "id_set_sha256": coverage.get("id_set_sha256"),
            "labels": coverage.get("labels"),
        }
    )
    if expected_coverage_sha256 and coverage_sha256 != expected_coverage_sha256:
        raise ValueError(
            f"{comparison_name} do not cover the same OOF products and labels"
        )

    resolved_config_path = _resolve_evidence_path(
        config_path, project_root=project_root
    )
    config = load_experiment_config(resolved_config_path)
    if config.experiment_id != experiment_id:
        raise ValueError(f"{variant} config and evidence experiment IDs do not match")
    if config.stage != expected_stage:
        raise ValueError(f"{experiment_id} config stage must be {expected_stage}")
    if config.folds != G1_EXPECTED_FOLDS or config.seeds != (2753,):
        raise ValueError(f"{experiment_id} config must use folds 0-4 and seed 2753")
    if config.model_family != "resnet18_small_stem" or config.loss_id != "cross_entropy":
        raise ValueError(
            f"{comparison_name} must use scratch small-stem ResNet18 and cross-entropy"
        )
    if config.data.image_size != expected_image_size:
        raise ValueError(f"{variant} image size must be {expected_image_size}")
    if config.data.augmentation != expected_augmentation:
        raise ValueError(f"{variant} augmentation must be {expected_augmentation}")
    comparison_payload = config.to_dict()
    comparison_payload.pop("experiment_id")
    comparison_payload.pop("stage")
    comparison_payload["data"].pop(changed_data_field)
    matched_config_sha256 = canonical_sha256(comparison_payload)
    config_sha256 = canonical_sha256(config.to_dict())

    pooled_path = _verified_manifest_artifact(
        manifest, "pooled_metrics", project_root=project_root
    )
    fold_summary_path = _verified_manifest_artifact(
        manifest, "fold_summary", project_root=project_root
    )
    fold_metrics_path = _verified_manifest_artifact(
        manifest, "fold_metrics", project_root=project_root
    )
    registry_path = _verified_manifest_artifact(
        manifest, "registry_snapshot", project_root=project_root
    )
    with pooled_path.open(encoding="utf-8") as handle:
        pooled = json.load(handle)
    fold_summary = pd.read_csv(fold_summary_path)
    fold_metrics = pd.read_csv(fold_metrics_path, dtype={"run_id": str})
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)

    run_ids = [str(run_id) for run_id in manifest.get("run_ids", ())]
    if len(registry) != 5 or set(registry["run_id"]) != set(run_ids):
        raise ValueError(f"{experiment_id} registry snapshot must match five run IDs")
    observed_folds = set(pd.to_numeric(registry["fold"], errors="raise").astype(int))
    if observed_folds != set(G1_EXPECTED_FOLDS):
        raise ValueError(f"{experiment_id} registry snapshot has invalid folds")
    required_registry_values = {
        "stage": expected_stage,
        "experiment_id": experiment_id,
        "model_family": "resnet18_small_stem",
        "benchmark_only": "false",
        "final_eligible": "true",
        "scratch": "true",
        "seed": "2753",
        "loss_id": "cross_entropy",
        "git_dirty": "false",
        "status": "completed",
    }
    for column, expected in required_registry_values.items():
        observed = set(registry[column].astype(str).str.lower())
        if observed != {expected}:
            raise ValueError(
                f"{experiment_id} registry {column} must be {expected}: {observed}"
            )
    unique_registry_fields: dict[str, str] = {}
    for column in (
        "config_sha256",
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
        "transform_id",
        "parameter_count",
    ):
        observed = set(registry[column].astype(str))
        if len(observed) != 1:
            raise ValueError(f"{experiment_id} changed {column} across folds")
        unique_registry_fields[column] = observed.pop()
    if unique_registry_fields["config_sha256"] != config_sha256:
        raise ValueError(f"{experiment_id} config hash does not match its registry rows")

    if len(fold_metrics) != 5 or set(fold_metrics["run_id"].astype(str)) != set(run_ids):
        raise ValueError(f"{experiment_id} fold metrics must match five run IDs")
    fold_metrics["fold"] = pd.to_numeric(fold_metrics["fold"], errors="raise").astype(int)
    if set(fold_metrics["fold"]) != set(G1_EXPECTED_FOLDS):
        raise ValueError(f"{experiment_id} fold metrics have invalid folds")
    registry_cost = registry.loc[
        :, ["run_id", "runtime_seconds", "peak_vram_mb", "best_epoch"]
    ].copy()
    registry_cost["runtime_seconds"] = pd.to_numeric(
        registry_cost["runtime_seconds"], errors="raise"
    )
    registry_cost["peak_vram_mb"] = pd.to_numeric(
        registry_cost["peak_vram_mb"], errors="raise"
    )
    registry_cost["best_epoch"] = pd.to_numeric(
        registry_cost["best_epoch"], errors="raise"
    ).astype(int)
    fold_detail = fold_metrics.loc[:, ["run_id", "fold", "macro_f1"]].merge(
        registry_cost,
        on="run_id",
        how="inner",
        validate="one_to_one",
    )
    fold_detail["macro_f1"] = pd.to_numeric(
        fold_detail["macro_f1"], errors="raise"
    )
    macro_summary = fold_summary.loc[fold_summary["metric"].eq("macro_f1")]
    if len(macro_summary) != 1:
        raise ValueError(f"{experiment_id} requires one macro-F1 fold summary row")
    pooled_macro_f1 = float(pooled["macro_f1"])
    if not np.isclose(pooled_macro_f1, float(manifest["pooled_macro_f1"])):
        raise ValueError(f"{experiment_id} pooled macro-F1 disagrees with its manifest")
    pooled_per_class = pooled.get("per_class", {})
    if set(pooled_per_class) != set(SEASON_LABELS):
        raise ValueError(f"{experiment_id} requires all four Season class metrics")
    per_class_rows = []
    for label in SEASON_LABELS:
        metrics = pooled_per_class[label]
        required_metrics = {"precision", "recall", "f1", "support"}
        if set(metrics) < required_metrics:
            raise ValueError(f"{experiment_id} {label} metrics are incomplete")
        scores = {
            name: float(metrics[name]) for name in ("precision", "recall", "f1")
        }
        support = int(metrics["support"])
        if (
            not np.isfinite(np.fromiter(scores.values(), dtype=float)).all()
            or any(value < 0.0 or value > 1.0 for value in scores.values())
            or support < 0
        ):
            raise ValueError(f"{experiment_id} {label} metrics are invalid")
        per_class_rows.append(
            {
                "variant": variant,
                "label": label,
                **scores,
                "support": support,
            }
        )
    summary = macro_summary.iloc[0]
    row = {
        "variant": variant,
        "experiment_id": experiment_id,
        "image_height": expected_image_size[0],
        "image_width": expected_image_size[1],
        "augmentation": expected_augmentation,
        "pooled_macro_f1": pooled_macro_f1,
        "fold_mean_macro_f1": float(summary["fold_mean"]),
        "fold_sd_macro_f1": float(summary["fold_sd"]),
        "spring_f1": float(pooled["per_class"]["Spring"]["f1"]),
        "five_fold_runtime_minutes": float(
            pd.to_numeric(registry["runtime_seconds"], errors="raise").sum() / 60.0
        ),
        "peak_vram_mb": float(
            pd.to_numeric(registry["peak_vram_mb"], errors="raise").max()
        ),
        "parameter_count": int(unique_registry_fields["parameter_count"]),
        "config_sha256": config_sha256,
        "split_sha256": unique_registry_fields["split_sha256"],
        "label_map_sha256": unique_registry_fields["label_map_sha256"],
        "implementation_sha256": unique_registry_fields["implementation_sha256"],
        "transform_id": unique_registry_fields["transform_id"],
    }
    return (
        row,
        fold_detail,
        pd.DataFrame(per_class_rows),
        coverage_sha256,
        matched_config_sha256,
        resolved_config_path,
    )


def _plot_g2_input_size_ablation(
    comparison: pd.DataFrame,
    paired_folds: pd.DataFrame,
    *,
    minimum_gain: float,
    selected_variant: str,
    output_path: Path,
) -> None:
    figure = Figure(figsize=(12, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    quality_axis, fold_axis = figure.subplots(1, 2)
    variants = comparison["variant"].tolist()
    positions = np.arange(len(variants))
    quality_values: list[float] = []
    for column, label, colour, offset in (
        ("pooled_macro_f1", "Pooled macro-F1", "#2563EB", -0.04),
        ("spring_f1", "Spring F1", "#F59E0B", 0.04),
    ):
        values = comparison[column].to_numpy(dtype=float)
        quality_values.extend(values.tolist())
        metric_positions = positions + offset
        quality_axis.plot(
            metric_positions,
            values,
            marker="o",
            markersize=8,
            linewidth=1.8,
            label=label,
            color=colour,
        )
        for position, value in zip(metric_positions, values, strict=True):
            quality_axis.annotate(
                f"{value:.4f}",
                (position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    p0_score = float(
        comparison.loc[comparison["variant"].eq("P0"), "pooled_macro_f1"].iloc[0]
    )
    threshold = p0_score + minimum_gain
    quality_axis.axhline(
        threshold,
        color="#DC2626",
        linestyle="--",
        label=f"P1 selection threshold = P0 + {minimum_gain:.3f}",
    )
    quality_axis.set_xticks(positions, labels=variants)
    quality_axis.set_xlim(-0.25, len(variants) - 0.75)
    quality_axis.set_ylim(
        max(0.0, min(quality_values + [threshold]) - 0.01),
        min(1.0, max(quality_values + [threshold]) + 0.01),
    )
    quality_axis.set_ylabel("OOF F1")
    quality_axis.set_title(f"Pooled quality; selected {selected_variant}")
    quality_axis.legend(loc="lower right", fontsize=8)
    quality_axis.grid(axis="y", alpha=0.2)

    fold_colours = [
        "#16A34A" if value >= 0.0 else "#DC2626"
        for value in paired_folds["delta_p1_minus_p0_macro_f1"]
    ]
    fold_axis.bar(
        paired_folds["fold"].astype(int),
        paired_folds["delta_p1_minus_p0_macro_f1"],
        color=fold_colours,
    )
    pooled_delta = float(
        comparison.loc[
            comparison["variant"].eq("P1"), "delta_vs_p0_macro_f1"
        ].iloc[0]
    )
    fold_axis.axhline(0.0, color="#0F172A", linewidth=1.0)
    fold_axis.axhline(
        pooled_delta,
        color="#7C3AED",
        linestyle="--",
        label=f"pooled delta = {pooled_delta:+.4f}",
    )
    fold_axis.set_xticks(G1_EXPECTED_FOLDS)
    fold_axis.set_xlabel("Canonical validation fold")
    fold_axis.set_ylabel("P1 - P0 macro-F1")
    fold_axis.set_title("Paired fold direction")
    fold_axis.legend(loc="best")
    fold_axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "G2-P input-size ablation: moderate upscaling versus source-like resolution",
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    figure.clear()


def build_g2_input_size_evidence(
    *,
    p0_manifest_path: str | Path,
    p1_manifest_path: str | Path,
    p0_config_path: str | Path,
    p1_config_path: str | Path,
    project_root: str | Path = ROOT,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR / "g2_input_size_ablation",
    figure_directory: str | Path = TASK2_FIGURE_DIR,
    minimum_gain: float = G2_SIZE_MINIMUM_GAIN,
) -> dict[str, Any]:
    """Audit P0/P1 matching and apply the frozen pooled OOF size-selection rule."""
    if minimum_gain < 0.0:
        raise ValueError("minimum_gain must be non-negative")
    root = Path(project_root)
    manifest_paths = {"P0": p0_manifest_path, "P1": p1_manifest_path}
    config_paths = {"P0": p0_config_path, "P1": p1_config_path}
    loaded = {
        variant: _load_verified_experiment_manifest(path, project_root=root)
        for variant, path in manifest_paths.items()
    }
    rows: list[dict[str, Any]] = []
    details: dict[str, pd.DataFrame] = {}
    coverage_sha256: str | None = None
    matched_config_sha256: str | None = None
    resolved_configs: dict[str, Path] = {}
    for variant in ("P0", "P1"):
        (
            row,
            fold_detail,
            _,
            observed_coverage,
            observed_config,
            resolved_config,
        ) = (
            _g2_transform_row(
                loaded[variant][0],
                variant=variant,
                config_path=config_paths[variant],
                project_root=root,
                expected_coverage_sha256=coverage_sha256,
                expected_experiment_id=G2_SIZE_EXPERIMENTS[variant],
                expected_stage={
                    "P0": "g1_family_screen",
                    "P1": "g2_input_size_ablation",
                }[variant],
                expected_image_size=G2_SIZE_EXPECTED_PIXELS[variant],
                expected_augmentation="a0",
                changed_data_field="image_size",
                comparison_name="P0 and P1",
            )
        )
        coverage_sha256 = coverage_sha256 or observed_coverage
        matched_config_sha256 = matched_config_sha256 or observed_config
        if observed_config != matched_config_sha256:
            raise ValueError(
                "P0 and P1 configs differ outside experiment ID, stage, and image size"
            )
        rows.append(row)
        details[variant] = fold_detail
        resolved_configs[variant] = resolved_config

    comparison = pd.DataFrame(rows).drop(columns="augmentation")
    for column in (
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
        "parameter_count",
    ):
        if comparison[column].nunique() != 1:
            raise ValueError(f"P0 and P1 must share the same {column}")
    p0_score = float(
        comparison.loc[comparison["variant"].eq("P0"), "pooled_macro_f1"].iloc[0]
    )
    p0_spring = float(
        comparison.loc[comparison["variant"].eq("P0"), "spring_f1"].iloc[0]
    )
    comparison["delta_vs_p0_macro_f1"] = comparison["pooled_macro_f1"] - p0_score
    comparison["delta_vs_p0_spring_f1"] = comparison["spring_f1"] - p0_spring
    p0_runtime = float(
        comparison.loc[
            comparison["variant"].eq("P0"), "five_fold_runtime_minutes"
        ].iloc[0]
    )
    p0_vram = float(
        comparison.loc[comparison["variant"].eq("P0"), "peak_vram_mb"].iloc[0]
    )
    comparison["runtime_ratio_vs_p0"] = (
        comparison["five_fold_runtime_minutes"] / p0_runtime
    )
    comparison["peak_vram_ratio_vs_p0"] = comparison["peak_vram_mb"] / p0_vram

    paired_folds = details["P0"].rename(
        columns={
            "run_id": "p0_run_id",
            "macro_f1": "p0_macro_f1",
            "runtime_seconds": "p0_runtime_seconds",
            "peak_vram_mb": "p0_peak_vram_mb",
            "best_epoch": "p0_best_epoch",
        }
    ).merge(
        details["P1"].rename(
            columns={
                "run_id": "p1_run_id",
                "macro_f1": "p1_macro_f1",
                "runtime_seconds": "p1_runtime_seconds",
                "peak_vram_mb": "p1_peak_vram_mb",
                "best_epoch": "p1_best_epoch",
            }
        ),
        on="fold",
        how="inner",
        validate="one_to_one",
    )
    if len(paired_folds) != 5:
        raise ValueError("P0/P1 paired comparison requires exactly five folds")
    paired_folds["delta_p1_minus_p0_macro_f1"] = (
        paired_folds["p1_macro_f1"] - paired_folds["p0_macro_f1"]
    )
    paired_folds["runtime_ratio_p1_vs_p0"] = (
        paired_folds["p1_runtime_seconds"] / paired_folds["p0_runtime_seconds"]
    )

    p1_delta = float(
        comparison.loc[
            comparison["variant"].eq("P1"), "delta_vs_p0_macro_f1"
        ].iloc[0]
    )
    selected_variant = "P1" if p1_delta >= minimum_gain else "P0"
    rejected_variant = "P0" if selected_variant == "P1" else "P1"
    selected_experiment_id = G2_SIZE_EXPERIMENTS[selected_variant]
    decision = {
        "schema_version": "1.0.0",
        "gate": "G2-P",
        "decision_status": "closed",
        "primary_metric": "pooled_five_fold_oof_macro_f1",
        "selection_rule": (
            f"Select P1 only when P1 minus P0 is at least {minimum_gain:.3f} "
            "absolute macro-F1; otherwise retain P0."
        ),
        "minimum_gain": minimum_gain,
        "observed_p1_minus_p0_macro_f1": p1_delta,
        "selected_variant": selected_variant,
        "selected_experiment_id": selected_experiment_id,
        "rejected_variant": rejected_variant,
        "next_question": (
            f"Compare A0 with A1 on {selected_variant} while holding the selected "
            "image size, C2, folds, seed, optimiser, and budget fixed."
        ),
    }

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(
        os.path.commonpath([evidence_root.resolve(), figure_root.resolve()])
    )
    comparison_path = evidence_root / "comparison.csv"
    paired_folds_path = evidence_root / "paired_fold_metrics.csv"
    decision_path = evidence_root / "decision.json"
    figure_path = figure_root / "g2_input_size_ablation.png"
    manifest_path = evidence_root / "manifest.json"
    atomic_write_csv(comparison_path, comparison)
    atomic_write_csv(paired_folds_path, paired_folds.sort_values("fold"))
    atomic_write_json(decision_path, decision)
    _plot_g2_input_size_ablation(
        comparison,
        paired_folds,
        minimum_gain=minimum_gain,
        selected_variant=selected_variant,
        output_path=figure_path,
    )
    artifacts = {
        "comparison": comparison_path,
        "paired_fold_metrics": paired_folds_path,
        "decision": decision_path,
        "figure": figure_path,
    }
    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=common_root),
            "sha256": compute_sha256(path),
        }
        for name, path in artifacts.items()
    }
    input_manifests = {
        variant: {
            "experiment_id": loaded[variant][0]["experiment_id"],
            "path": _portable_artifact_path(loaded[variant][1], fallback_root=root),
            "sha256": compute_sha256(loaded[variant][1]),
        }
        for variant in ("P0", "P1")
    }
    input_configs = {
        variant: {
            "path": _portable_artifact_path(resolved_configs[variant], fallback_root=root),
            "sha256": compute_sha256(resolved_configs[variant]),
        }
        for variant in ("P0", "P1")
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G2-P",
        "decision_status": "closed",
        "coverage_sha256": coverage_sha256,
        "matched_config_sha256": matched_config_sha256,
        "minimum_gain": minimum_gain,
        "observed_p1_minus_p0_macro_f1": p1_delta,
        "selected_variant": selected_variant,
        "selected_experiment_id": selected_experiment_id,
        "input_manifests": input_manifests,
        "input_configs": input_configs,
        "artifacts": artifact_manifest,
    }
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


def _load_g2_augmentation_robustness(
    path: str | Path,
    *,
    project_root: Path,
) -> tuple[pd.DataFrame, Path, float]:
    """Load matched A1-minus-A0 robustness deltas from one auditable CSV."""
    resolved = _resolve_evidence_path(path, project_root=project_root)
    if not resolved.is_file():
        raise ValueError(f"robustness evidence does not exist: {resolved}")
    evidence = pd.read_csv(resolved)
    required = {"probe", "delta_a1_minus_a0_macro_f1"}
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise ValueError(f"robustness evidence is missing columns: {missing}")
    if evidence.empty:
        raise ValueError("robustness evidence must contain at least one probe")
    probes = evidence["probe"].astype(str).str.strip()
    if probes.eq("").any() or probes.duplicated().any():
        raise ValueError("robustness probe names must be non-empty and unique")
    deltas = pd.to_numeric(
        evidence["delta_a1_minus_a0_macro_f1"], errors="raise"
    )
    if not np.isfinite(deltas.to_numpy(dtype=float)).all():
        raise ValueError("robustness deltas must be finite")
    evidence = evidence.copy()
    evidence["probe"] = probes
    evidence["delta_a1_minus_a0_macro_f1"] = deltas
    worst_loss = max(0.0, float((-deltas).max()))
    return evidence, resolved, worst_loss


def _plot_g2_augmentation_ablation(
    comparison: pd.DataFrame,
    per_class: pd.DataFrame,
    *,
    minimum_gain: float,
    decision_status: str,
    selected_variant: str | None,
    output_path: Path,
) -> None:
    """Render quality as points and signed per-class changes as zero-based bars."""
    figure = Figure(figsize=(12, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    quality_axis, class_axis = figure.subplots(1, 2)
    variants = comparison["variant"].tolist()
    positions = np.arange(len(variants))
    quality_values: list[float] = []
    for column, label, colour, offset in (
        ("pooled_macro_f1", "Pooled macro-F1", "#2563EB", -0.04),
        ("spring_f1", "Spring F1", "#F59E0B", 0.04),
    ):
        values = comparison[column].to_numpy(dtype=float)
        quality_values.extend(values.tolist())
        metric_positions = positions + offset
        quality_axis.plot(
            metric_positions,
            values,
            marker="o",
            markersize=8,
            linewidth=1.8,
            label=label,
            color=colour,
        )
        for position, value in zip(metric_positions, values, strict=True):
            quality_axis.annotate(
                f"{value:.4f}",
                (position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    a0_score = float(
        comparison.loc[comparison["variant"].eq("A0"), "pooled_macro_f1"].iloc[0]
    )
    threshold = a0_score + minimum_gain
    quality_axis.axhline(
        threshold,
        color="#DC2626",
        linestyle="--",
        label=f"A1 quality threshold = A0 + {minimum_gain:.3f}",
    )
    quality_axis.set_xticks(positions, labels=variants)
    quality_axis.set_xlim(-0.25, len(variants) - 0.75)
    quality_axis.set_ylim(
        max(0.0, min(quality_values + [threshold]) - 0.015),
        min(1.0, max(quality_values + [threshold]) + 0.015),
    )
    decision_label = (
        f"selected {selected_variant}"
        if selected_variant is not None
        else "pending robustness"
    )
    quality_axis.set_ylabel("OOF F1")
    quality_axis.set_title(f"Pooled quality; {decision_status}: {decision_label}")
    quality_axis.legend(loc="best", fontsize=8)
    quality_axis.grid(axis="y", alpha=0.2)

    deltas = per_class["delta_a1_minus_a0_f1"].to_numpy(dtype=float)
    colours = ["#16A34A" if value >= 0.0 else "#DC2626" for value in deltas]
    class_axis.bar(per_class["label"], deltas, color=colours)
    class_axis.axhline(0.0, color="#0F172A", linewidth=1.0)
    for position, value in enumerate(deltas):
        class_axis.annotate(
            f"{value:+.4f}",
            (position, value),
            xytext=(0, 5 if value >= 0.0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    class_axis.set_xlabel("Season class")
    class_axis.set_ylabel("A1 - A0 F1")
    class_axis.set_title("Pooled OOF per-class direction")
    class_axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "G2-A augmentation ablation: A0 geometry versus A1 mild colour jitter",
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    figure.clear()


def build_g2_augmentation_evidence(
    *,
    a0_manifest_path: str | Path,
    a1_manifest_path: str | Path,
    a0_config_path: str | Path,
    a1_config_path: str | Path,
    project_root: str | Path = ROOT,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR
    / "g2_augmentation_ablation",
    figure_directory: str | Path = TASK2_FIGURE_DIR,
    minimum_gain: float = G2_AUGMENTATION_MINIMUM_GAIN,
    maximum_robustness_loss: float = G2_AUGMENTATION_MAX_ROBUSTNESS_LOSS,
    robustness_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit A0/A1 matching and apply the frozen quality-plus-robustness rule."""
    if (
        not np.isfinite(minimum_gain)
        or not np.isfinite(maximum_robustness_loss)
        or minimum_gain < 0.0
        or maximum_robustness_loss < 0.0
    ):
        raise ValueError("augmentation thresholds must be non-negative")
    root = Path(project_root)
    manifest_paths = {"A0": a0_manifest_path, "A1": a1_manifest_path}
    config_paths = {"A0": a0_config_path, "A1": a1_config_path}
    loaded = {
        variant: _load_verified_experiment_manifest(path, project_root=root)
        for variant, path in manifest_paths.items()
    }
    rows: list[dict[str, Any]] = []
    fold_details: dict[str, pd.DataFrame] = {}
    class_details: dict[str, pd.DataFrame] = {}
    coverage_sha256: str | None = None
    matched_config_sha256: str | None = None
    resolved_configs: dict[str, Path] = {}
    for variant in ("A0", "A1"):
        (
            row,
            fold_detail,
            per_class,
            observed_coverage,
            observed_config,
            resolved_config,
        ) = _g2_transform_row(
            loaded[variant][0],
            variant=variant,
            config_path=config_paths[variant],
            project_root=root,
            expected_coverage_sha256=coverage_sha256,
            expected_experiment_id=G2_AUGMENTATION_EXPERIMENTS[variant],
            expected_stage={
                "A0": "g1_family_screen",
                "A1": "g2_augmentation_ablation",
            }[variant],
            expected_image_size=G2_SIZE_EXPECTED_PIXELS["P0"],
            expected_augmentation=G2_AUGMENTATION_VALUES[variant],
            changed_data_field="augmentation",
            comparison_name="A0 and A1",
        )
        coverage_sha256 = coverage_sha256 or observed_coverage
        matched_config_sha256 = matched_config_sha256 or observed_config
        if observed_config != matched_config_sha256:
            raise ValueError(
                "A0 and A1 configs differ outside experiment ID, stage, and augmentation"
            )
        rows.append(row)
        fold_details[variant] = fold_detail
        class_details[variant] = per_class
        resolved_configs[variant] = resolved_config

    comparison = pd.DataFrame(rows)
    for column in (
        "image_height",
        "image_width",
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
        "parameter_count",
    ):
        if comparison[column].nunique() != 1:
            raise ValueError(f"A0 and A1 must share the same {column}")
    if comparison["transform_id"].nunique() != 2:
        raise ValueError("A0 and A1 must produce distinct transform IDs")

    a0_score = float(
        comparison.loc[comparison["variant"].eq("A0"), "pooled_macro_f1"].iloc[0]
    )
    a0_spring = float(
        comparison.loc[comparison["variant"].eq("A0"), "spring_f1"].iloc[0]
    )
    comparison["delta_vs_a0_macro_f1"] = comparison["pooled_macro_f1"] - a0_score
    comparison["delta_vs_a0_spring_f1"] = comparison["spring_f1"] - a0_spring
    a0_runtime = float(
        comparison.loc[
            comparison["variant"].eq("A0"), "five_fold_runtime_minutes"
        ].iloc[0]
    )
    a0_vram = float(
        comparison.loc[comparison["variant"].eq("A0"), "peak_vram_mb"].iloc[0]
    )
    comparison["runtime_ratio_vs_a0"] = (
        comparison["five_fold_runtime_minutes"] / a0_runtime
    )
    comparison["peak_vram_ratio_vs_a0"] = comparison["peak_vram_mb"] / a0_vram

    paired_folds = fold_details["A0"].rename(
        columns={
            "run_id": "a0_run_id",
            "macro_f1": "a0_macro_f1",
            "runtime_seconds": "a0_runtime_seconds",
            "peak_vram_mb": "a0_peak_vram_mb",
            "best_epoch": "a0_best_epoch",
        }
    ).merge(
        fold_details["A1"].rename(
            columns={
                "run_id": "a1_run_id",
                "macro_f1": "a1_macro_f1",
                "runtime_seconds": "a1_runtime_seconds",
                "peak_vram_mb": "a1_peak_vram_mb",
                "best_epoch": "a1_best_epoch",
            }
        ),
        on="fold",
        how="inner",
        validate="one_to_one",
    )
    if len(paired_folds) != 5:
        raise ValueError("A0/A1 paired comparison requires exactly five folds")
    paired_folds["delta_a1_minus_a0_macro_f1"] = (
        paired_folds["a1_macro_f1"] - paired_folds["a0_macro_f1"]
    )
    paired_folds["runtime_ratio_a1_vs_a0"] = (
        paired_folds["a1_runtime_seconds"] / paired_folds["a0_runtime_seconds"]
    )

    a0_classes = class_details["A0"].drop(columns="variant").rename(
        columns={name: f"a0_{name}" for name in ("precision", "recall", "f1", "support")}
    )
    a1_classes = class_details["A1"].drop(columns="variant").rename(
        columns={name: f"a1_{name}" for name in ("precision", "recall", "f1", "support")}
    )
    per_class = a0_classes.merge(
        a1_classes,
        on="label",
        how="inner",
        validate="one_to_one",
    )
    if per_class["label"].tolist() != list(SEASON_LABELS):
        raise ValueError("A0/A1 per-class evidence must follow canonical label order")
    if not per_class["a0_support"].equals(per_class["a1_support"]):
        raise ValueError("A0 and A1 per-class supports must match")
    per_class.insert(1, "support", per_class.pop("a0_support"))
    per_class = per_class.drop(columns="a1_support")
    for metric in ("precision", "recall", "f1"):
        per_class[f"delta_a1_minus_a0_{metric}"] = (
            per_class[f"a1_{metric}"] - per_class[f"a0_{metric}"]
        )

    a1_delta = float(
        comparison.loc[
            comparison["variant"].eq("A1"), "delta_vs_a0_macro_f1"
        ].iloc[0]
    )
    quality_gate_passed = bool(
        a1_delta > minimum_gain
        or np.isclose(a1_delta, minimum_gain, rtol=0.0, atol=1e-12)
    )
    robustness_frame: pd.DataFrame | None = None
    resolved_robustness: Path | None = None
    worst_robustness_loss: float | None = None
    if robustness_evidence_path is not None:
        robustness_frame, resolved_robustness, worst_robustness_loss = (
            _load_g2_augmentation_robustness(
                robustness_evidence_path,
                project_root=root,
            )
        )

    if not quality_gate_passed:
        decision_status = "closed"
        robustness_status = "not_required"
        robustness_gate_passed: bool | None = None
        selected_variant: str | None = "A0"
    elif robustness_frame is None:
        decision_status = "pending"
        robustness_status = "required"
        robustness_gate_passed = None
        selected_variant = None
    else:
        decision_status = "closed"
        robustness_status = "available"
        robustness_gate_passed = bool(
            worst_robustness_loss is not None
            and (
                worst_robustness_loss < maximum_robustness_loss
                or np.isclose(
                    worst_robustness_loss,
                    maximum_robustness_loss,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        )
        selected_variant = "A1" if robustness_gate_passed else "A0"
    selected_experiment_id = (
        G2_AUGMENTATION_EXPERIMENTS[selected_variant]
        if selected_variant is not None
        else None
    )
    rejected_variant = {"A0": "A1", "A1": "A0", None: None}[selected_variant]
    next_question = (
        "Generate matched A0/A1 robustness probes before selecting A1."
        if decision_status == "pending"
        else (
            f"Tune T0, T1, and T2 on {selected_variant} while holding the retained "
            "transform, folds, seed, loss, and budget fixed."
        )
    )
    decision = {
        "schema_version": "1.0.0",
        "gate": "G2-A",
        "decision_status": decision_status,
        "primary_metric": "pooled_five_fold_oof_macro_f1",
        "selection_rule": (
            f"Select A1 only when A1 minus A0 is at least {minimum_gain:.3f} "
            "absolute macro-F1 and its worst matched robustness loss is no more "
            f"than {maximum_robustness_loss:.3f}; otherwise retain A0."
        ),
        "minimum_gain": minimum_gain,
        "maximum_robustness_loss": maximum_robustness_loss,
        "observed_a1_minus_a0_macro_f1": a1_delta,
        "quality_gate_passed": quality_gate_passed,
        "robustness_evidence_status": robustness_status,
        "observed_worst_robustness_loss": worst_robustness_loss,
        "robustness_gate_passed": robustness_gate_passed,
        "selected_variant": selected_variant,
        "selected_experiment_id": selected_experiment_id,
        "rejected_variant": rejected_variant,
        "next_question": next_question,
    }

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(
        os.path.commonpath([evidence_root.resolve(), figure_root.resolve()])
    )
    comparison_path = evidence_root / "comparison.csv"
    paired_folds_path = evidence_root / "paired_fold_metrics.csv"
    per_class_path = evidence_root / "per_class_comparison.csv"
    decision_path = evidence_root / "decision.json"
    figure_path = figure_root / "g2_augmentation_ablation.png"
    manifest_path = evidence_root / "manifest.json"
    atomic_write_csv(comparison_path, comparison)
    atomic_write_csv(paired_folds_path, paired_folds.sort_values("fold"))
    atomic_write_csv(per_class_path, per_class)
    atomic_write_json(decision_path, decision)
    _plot_g2_augmentation_ablation(
        comparison,
        per_class,
        minimum_gain=minimum_gain,
        decision_status=decision_status,
        selected_variant=selected_variant,
        output_path=figure_path,
    )
    artifacts = {
        "comparison": comparison_path,
        "paired_fold_metrics": paired_folds_path,
        "per_class_comparison": per_class_path,
        "decision": decision_path,
        "figure": figure_path,
    }
    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=common_root),
            "sha256": compute_sha256(path),
        }
        for name, path in artifacts.items()
    }
    input_manifests = {
        variant: {
            "experiment_id": loaded[variant][0]["experiment_id"],
            "path": _portable_artifact_path(loaded[variant][1], fallback_root=root),
            "sha256": compute_sha256(loaded[variant][1]),
        }
        for variant in ("A0", "A1")
    }
    input_configs = {
        variant: {
            "path": _portable_artifact_path(resolved_configs[variant], fallback_root=root),
            "sha256": compute_sha256(resolved_configs[variant]),
        }
        for variant in ("A0", "A1")
    }
    robustness_input = (
        {
            "path": _portable_artifact_path(resolved_robustness, fallback_root=root),
            "sha256": compute_sha256(resolved_robustness),
        }
        if resolved_robustness is not None
        else None
    )
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G2-A",
        "decision_status": decision_status,
        "coverage_sha256": coverage_sha256,
        "matched_config_sha256": matched_config_sha256,
        "minimum_gain": minimum_gain,
        "maximum_robustness_loss": maximum_robustness_loss,
        "observed_a1_minus_a0_macro_f1": a1_delta,
        "selected_variant": selected_variant,
        "selected_experiment_id": selected_experiment_id,
        "input_manifests": input_manifests,
        "input_configs": input_configs,
        "robustness_input": robustness_input,
        "artifacts": artifact_manifest,
    }
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
