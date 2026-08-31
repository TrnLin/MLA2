"""Hash-linked evidence for the eligible Task 2 seed-stability pair."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import (
    EXPERIMENT_REGISTRY_COLUMNS,
    _load_verified_experiment_manifest,
    _portable_artifact_path,
    _resolve_evidence_path,
    _verified_manifest_artifact,
)
from fashion.task2.experiments import ExperimentConfig, load_experiment_config
from fashion.task2.multitask import I2ExperimentConfig, load_i2_config
from fashion.task2.stability import (
    C2_PRIMARY_EXPERIMENT_ID,
    C2_STABILITY_EXPERIMENT_ID,
    G5_SEED,
    I2_PRIMARY_EXPERIMENT_ID,
    I2_STABILITY_EXPERIMENT_ID,
    StabilityConfigPair,
    load_stability_i2_config,
    validate_stability_pair,
)
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
)
from fashion.train.metrics import SEASON_LABELS

STABILITY_EXPERIMENT_IDS = (
    C2_PRIMARY_EXPERIMENT_ID,
    C2_STABILITY_EXPERIMENT_ID,
    I2_PRIMARY_EXPERIMENT_ID,
    I2_STABILITY_EXPERIMENT_ID,
)
EXPECTED_EXPERIMENTS = {
    C2_PRIMARY_EXPERIMENT_ID: {
        "candidate": "C2",
        "variant": "C2 ResNet18",
        "seed": 2753,
        "stage": "g3_full_budget",
        "model_family": "resnet18_small_stem",
        "loss_id": "cross_entropy",
        "loss_fields": ("train_loss", "validation_loss"),
    },
    C2_STABILITY_EXPERIMENT_ID: {
        "candidate": "C2",
        "variant": "C2 ResNet18",
        "seed": G5_SEED,
        "stage": "g5_seed_stability",
        "model_family": "resnet18_small_stem",
        "loss_id": "cross_entropy",
        "loss_fields": ("train_loss", "validation_loss"),
    },
    I2_PRIMARY_EXPERIMENT_ID: {
        "candidate": "I2",
        "variant": "I2 SmallCNN",
        "seed": 2753,
        "stage": "g4_i2_multitask",
        "model_family": "smallcnn",
        "loss_id": "season_ce_plus_masked_article_type_ce_lambda_0_3",
        "loss_fields": ("train_total_loss", "validation_total_loss"),
    },
    I2_STABILITY_EXPERIMENT_ID: {
        "candidate": "I2",
        "variant": "I2 SmallCNN",
        "seed": G5_SEED,
        "stage": "g5_seed_stability",
        "model_family": "smallcnn",
        "loss_id": "season_ce_plus_masked_article_type_ce_lambda_0_3",
        "loss_fields": ("train_total_loss", "validation_total_loss"),
    },
}


@dataclass(frozen=True)
class _StabilityAudit:
    experiment_id: str
    candidate: str
    variant: str
    seed: int
    config: ExperimentConfig | I2ExperimentConfig
    config_path: Path
    manifest: dict[str, Any]
    manifest_path: Path
    registry: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_metrics: dict[str, Any]
    histories: pd.DataFrame


def _load_declared_config(
    path: str | Path,
    *,
    project_root: Path,
) -> tuple[ExperimentConfig | I2ExperimentConfig, Path]:
    resolved = _resolve_evidence_path(path, project_root=project_root)
    with resolved.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"stability config must be a JSON object: {resolved}")
    experiment_id = str(raw.get("experiment_id", ""))
    if experiment_id in {C2_PRIMARY_EXPERIMENT_ID, C2_STABILITY_EXPERIMENT_ID}:
        config = load_experiment_config(resolved)
    elif experiment_id == I2_PRIMARY_EXPERIMENT_ID:
        config = load_i2_config(resolved)
    elif experiment_id == I2_STABILITY_EXPERIMENT_ID:
        config = load_stability_i2_config(resolved)
    else:
        raise ValueError(f"unexpected stability experiment_id: {experiment_id}")
    return config, resolved


def _load_histories(
    registry: pd.DataFrame,
    *,
    config: ExperimentConfig | I2ExperimentConfig,
    project_root: Path,
    expected: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_loss_field, validation_loss_field = expected["loss_fields"]
    required_metrics = {
        train_loss_field,
        validation_loss_field,
        "validation_accuracy",
        "validation_macro_f1",
    }
    for registry_row in registry.sort_values("fold").to_dict(orient="records"):
        history_path = _resolve_evidence_path(
            str(registry_row["history_path"]),
            project_root=project_root,
        )
        if not history_path.is_file():
            raise ValueError(f"stability history does not exist: {history_path}")
        if compute_sha256(history_path) != str(registry_row["history_sha256"]):
            raise ValueError(f"stability history hash mismatch for {registry_row['run_id']}")
        with history_path.open(encoding="utf-8") as handle:
            history = json.load(handle)
        identity = {
            "run_id": str(registry_row["run_id"]),
            "experiment_id": config.experiment_id,
            "fold": int(registry_row["fold"]),
            "seed": expected["seed"],
        }
        for field, value in identity.items():
            if history.get(field) != value:
                raise ValueError(f"stability history {field} mismatch for {identity['run_id']}")
        if canonical_sha256(history.get("config", {})) != canonical_sha256(config.to_dict()):
            raise ValueError(f"stability history config mismatch for {identity['run_id']}")
        boundary = history.get("model_boundary", {})
        required_boundary = {
            "benchmark_only": False,
            "final_eligible": True,
            "training_origin": "scratch",
            "weights": None,
        }
        mismatches = [
            field for field, value in required_boundary.items() if boundary.get(field) != value
        ]
        if mismatches:
            raise ValueError(
                f"stability model boundary mismatch for {identity['run_id']}: {mismatches}"
            )
        epochs = history.get("epoch_history", [])
        if not isinstance(epochs, list) or not epochs:
            raise ValueError(f"stability history is empty for {identity['run_id']}")
        if len(epochs) != int(registry_row["epochs_completed"]):
            raise ValueError(f"stability epoch count mismatch for {identity['run_id']}")
        if [int(epoch.get("epoch", -1)) for epoch in epochs] != list(range(1, len(epochs) + 1)):
            raise ValueError(f"stability epochs are not contiguous for {identity['run_id']}")
        for epoch in epochs:
            missing = sorted(required_metrics - set(epoch))
            if missing:
                raise ValueError(
                    f"stability history is missing metrics for {identity['run_id']}: {missing}"
                )
            rows.append(
                {
                    "candidate": expected["candidate"],
                    "variant": expected["variant"],
                    "experiment_id": config.experiment_id,
                    "run_id": identity["run_id"],
                    "fold": identity["fold"],
                    "seed": identity["seed"],
                    "epoch": int(epoch["epoch"]),
                    "train_loss": float(epoch[train_loss_field]),
                    "validation_loss": float(epoch[validation_loss_field]),
                    "validation_accuracy": float(epoch["validation_accuracy"]),
                    "validation_macro_f1": float(epoch["validation_macro_f1"]),
                }
            )
    return pd.DataFrame(rows)


def _load_audit(
    *,
    manifest_path: str | Path,
    config: ExperimentConfig | I2ExperimentConfig,
    config_path: Path,
    project_root: Path,
    expected_row_count: int,
) -> _StabilityAudit:
    manifest, resolved_manifest = _load_verified_experiment_manifest(
        manifest_path,
        project_root=project_root,
    )
    experiment_id = config.experiment_id
    if experiment_id not in EXPECTED_EXPERIMENTS:
        raise ValueError(f"unexpected stability experiment: {experiment_id}")
    expected = EXPECTED_EXPERIMENTS[experiment_id]
    if str(manifest.get("experiment_id", "")) != experiment_id:
        raise ValueError("stability config and manifest identities differ")
    if tuple(manifest.get("folds", ())) != tuple(range(5)):
        raise ValueError(f"{experiment_id} must contain folds 0-4")
    if int(manifest.get("seed", -1)) != expected["seed"]:
        raise ValueError(f"{experiment_id} has the wrong seed")
    coverage = manifest.get("coverage", {})
    if not (
        coverage.get("row_count")
        == coverage.get("unique_id_count")
        == coverage.get("expected_row_count")
        == expected_row_count
    ):
        raise ValueError(f"{experiment_id} has incomplete OOF coverage")
    if int(coverage.get("protected_id_count", -1)) != 0:
        raise ValueError(f"{experiment_id} includes protected IDs")
    if tuple(coverage.get("labels", ())) != tuple(SEASON_LABELS):
        raise ValueError(f"{experiment_id} has an invalid Season class order")

    registry_path = _verified_manifest_artifact(
        manifest,
        "registry_snapshot",
        project_root=project_root,
    )
    fold_metrics_path = _verified_manifest_artifact(
        manifest,
        "fold_metrics",
        project_root=project_root,
    )
    pooled_path = _verified_manifest_artifact(
        manifest,
        "pooled_metrics",
        project_root=project_root,
    )
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    fold_metrics = pd.read_csv(fold_metrics_path)
    with pooled_path.open(encoding="utf-8") as handle:
        pooled_metrics = json.load(handle)
    missing_columns = sorted(set(EXPERIMENT_REGISTRY_COLUMNS) - set(registry))
    if missing_columns:
        raise ValueError(f"{experiment_id} registry is missing columns: {missing_columns}")
    run_ids = [str(value) for value in manifest.get("run_ids", ())]
    if len(run_ids) != 5 or len(set(run_ids)) != 5 or set(registry["run_id"]) != set(run_ids):
        raise ValueError(f"{experiment_id} registry must match five unique run IDs")
    expected_registry = {
        "stage": expected["stage"],
        "experiment_id": experiment_id,
        "model_family": expected["model_family"],
        "benchmark_only": "false",
        "final_eligible": "true",
        "scratch": "true",
        "seed": str(expected["seed"]),
        "git_dirty": "false",
        "loss_id": expected["loss_id"],
        "primary_metric_name": "macro_f1",
        "status": "completed",
    }
    for column, value in expected_registry.items():
        observed = set(registry[column].astype(str).str.lower())
        if observed != {value.lower()}:
            raise ValueError(f"{experiment_id} registry mismatch for {column}")
    if set(registry["fold"].astype(int)) != set(range(5)):
        raise ValueError(f"{experiment_id} registry has incomplete folds")
    if registry["git_commit"].nunique() != 1:
        raise ValueError(f"{experiment_id} spans multiple Git commits")
    if set(registry["config_sha256"]) != {canonical_sha256(config.to_dict())}:
        raise ValueError(f"{experiment_id} config hash mismatch")
    if set(fold_metrics["run_id"].astype(str)) != set(run_ids):
        raise ValueError(f"{experiment_id} fold metrics do not match the registry")
    if set(pd.to_numeric(fold_metrics["seed"], errors="raise").astype(int)) != {expected["seed"]}:
        raise ValueError(f"{experiment_id} fold metrics have the wrong seed")
    if tuple(pooled_metrics.get("per_class", {})) != tuple(SEASON_LABELS):
        raise ValueError(f"{experiment_id} pooled class order is invalid")
    histories = _load_histories(
        registry,
        config=config,
        project_root=project_root,
        expected=expected,
    )
    return _StabilityAudit(
        experiment_id=experiment_id,
        candidate=expected["candidate"],
        variant=expected["variant"],
        seed=expected["seed"],
        config=config,
        config_path=config_path,
        manifest=manifest,
        manifest_path=resolved_manifest,
        registry=registry,
        fold_metrics=fold_metrics,
        pooled_metrics=pooled_metrics,
        histories=histories,
    )


def _learning_curve_summary(audits: Sequence[_StabilityAudit]) -> pd.DataFrame:
    histories = pd.concat([audit.histories for audit in audits], ignore_index=True)
    horizon_by_candidate = (
        histories.groupby(["candidate", "run_id"], sort=False)["epoch"]
        .max()
        .groupby("candidate")
        .min()
        .astype(int)
        .to_dict()
    )
    histories["common_horizon"] = histories["candidate"].map(horizon_by_candidate)
    histories = histories.loc[histories["epoch"].le(histories["common_horizon"])].copy()
    summary = (
        histories.groupby(
            ["candidate", "variant", "experiment_id", "seed", "epoch"],
            sort=False,
        )
        .agg(
            fold_count=("fold", "nunique"),
            train_loss_mean=("train_loss", "mean"),
            train_loss_sd=("train_loss", "std"),
            validation_loss_mean=("validation_loss", "mean"),
            validation_loss_sd=("validation_loss", "std"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_accuracy_sd=("validation_accuracy", "std"),
            validation_macro_f1_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_sd=("validation_macro_f1", "std"),
        )
        .reset_index()
    )
    summary["common_five_fold_horizon"] = summary["candidate"].map(horizon_by_candidate)
    if not summary["fold_count"].eq(5).all():
        raise ValueError("stability learning curves lost a fold")
    return summary


def plot_seed_stability_learning_curves(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Draw teacher-style loss and validation-score panels for both candidates."""
    required = {
        "candidate",
        "variant",
        "seed",
        "epoch",
        "fold_count",
        "train_loss_mean",
        "train_loss_sd",
        "validation_loss_mean",
        "validation_loss_sd",
        "validation_accuracy_mean",
        "validation_accuracy_sd",
        "validation_macro_f1_mean",
        "validation_macro_f1_sd",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"stability learning summary is missing columns: {missing}")
    if set(summary["candidate"]) != {"C2", "I2"}:
        raise ValueError("stability learning curves require C2 and I2")
    if set(pd.to_numeric(summary["seed"], errors="raise").astype(int)) != {
        2753,
        G5_SEED,
    }:
        raise ValueError("stability learning curves require seeds 2753 and 2026")
    if not summary["fold_count"].eq(5).all():
        raise ValueError("stability learning curves require five folds per point")

    figure = Figure(figsize=(14, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, squeeze=False)
    line_styles = {2753: "-", G5_SEED: "--"}
    colors = {
        "train_loss": "#2563eb",
        "validation_loss": "#f97316",
        "validation_accuracy": "#7c3aed",
        "validation_macro_f1": "#16a34a",
    }
    for row_index, candidate in enumerate(("C2", "I2")):
        candidate_rows = summary.loc[summary["candidate"].eq(candidate)].copy()
        variant = str(candidate_rows["variant"].iloc[0])
        loss_axis = axes[row_index, 0]
        score_axis = axes[row_index, 1]
        for seed in (2753, G5_SEED):
            rows = candidate_rows.loc[candidate_rows["seed"].eq(seed)].sort_values("epoch")
            epochs = rows["epoch"].to_numpy(dtype=float)
            style = line_styles[seed]
            for metric, label in (
                ("train_loss", f"Train s{seed}"),
                ("validation_loss", f"Validation s{seed}"),
            ):
                mean = rows[f"{metric}_mean"].to_numpy(dtype=float)
                sd = rows[f"{metric}_sd"].fillna(0).to_numpy(dtype=float)
                loss_axis.plot(
                    epochs,
                    mean,
                    linestyle=style,
                    color=colors[metric],
                    linewidth=2,
                    label=label,
                )
                loss_axis.fill_between(
                    epochs,
                    mean - sd,
                    mean + sd,
                    color=colors[metric],
                    alpha=0.08,
                )
            for metric, label in (
                ("validation_accuracy", f"Accuracy s{seed}"),
                ("validation_macro_f1", f"Macro-F1 s{seed}"),
            ):
                mean = rows[f"{metric}_mean"].to_numpy(dtype=float)
                sd = rows[f"{metric}_sd"].fillna(0).to_numpy(dtype=float)
                score_axis.plot(
                    epochs,
                    mean,
                    linestyle=style,
                    color=colors[metric],
                    linewidth=2,
                    label=label,
                )
                score_axis.fill_between(
                    epochs,
                    np.clip(mean - sd, 0, 1),
                    np.clip(mean + sd, 0, 1),
                    color=colors[metric],
                    alpha=0.08,
                )
        loss_axis.set_title(f"{variant}: loss by seed")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")
        loss_axis.grid(alpha=0.2)
        loss_axis.legend(fontsize=8)
        score_axis.set_title(f"{variant}: validation scores by seed")
        score_axis.set_xlabel("Epoch")
        score_axis.set_ylabel("Score")
        score_axis.set_ylim(0, 1)
        score_axis.grid(alpha=0.2)
        score_axis.legend(fontsize=8)
    destination = Path(output_path)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(destination, buffer.getvalue())
    figure.clear()
    return destination


def _plot_seed_stability_comparison(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    metrics = (
        ("pooled_macro_f1", "Pooled macro-F1"),
        ("spring_recall", "Spring recall"),
        ("spring_f1", "Spring F1"),
    )
    candidates = ("C2", "I2")
    seeds = (2753, G5_SEED)
    figure = Figure(figsize=(11, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(metrics), dtype=float)
    width = 0.18
    colors = {"C2": "#2563eb", "I2": "#16a34a"}
    offsets = {
        ("C2", 2753): -1.5 * width,
        ("C2", G5_SEED): -0.5 * width,
        ("I2", 2753): 0.5 * width,
        ("I2", G5_SEED): 1.5 * width,
    }
    for candidate in candidates:
        for seed in seeds:
            row = summary.loc[summary["candidate"].eq(candidate) & summary["seed"].eq(seed)].iloc[0]
            values = [float(row[column]) for column, _ in metrics]
            bars = axis.bar(
                x + offsets[(candidate, seed)],
                values,
                width,
                color=colors[candidate],
                alpha=1.0 if seed == G5_SEED else 0.55,
                label=f"{candidate} seed {seed}",
            )
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Eligible finalist stability across two seeds")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncols=2, fontsize=8)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    figure.clear()


def build_seed_stability_evidence(
    *,
    experiment_manifest_paths: Sequence[str | Path],
    experiment_config_paths: Sequence[str | Path],
    project_root: str | Path = ROOT,
    expected_row_count: int = 32_753,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR / "seed_stability",
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Audit two candidates at two seeds and report an honest ordering result."""
    if expected_row_count < 1:
        raise ValueError("expected_row_count must be positive")
    if len(experiment_manifest_paths) != 4 or len(experiment_config_paths) != 4:
        raise ValueError("stability evidence requires four manifests and configs")
    root = Path(project_root)
    configs_by_id: dict[str, tuple[ExperimentConfig | I2ExperimentConfig, Path]] = {}
    for path in experiment_config_paths:
        config, resolved = _load_declared_config(path, project_root=root)
        if config.experiment_id in configs_by_id:
            raise ValueError("stability evidence contains a duplicate config identity")
        configs_by_id[config.experiment_id] = (config, resolved)
    if set(configs_by_id) != set(STABILITY_EXPERIMENT_IDS):
        raise ValueError("stability evidence requires the exact frozen config set")
    validate_stability_pair(
        StabilityConfigPair(
            c2=configs_by_id[C2_STABILITY_EXPERIMENT_ID][0],
            i2=configs_by_id[I2_STABILITY_EXPERIMENT_ID][0],
        )
    )

    manifests_by_id: dict[str, str | Path] = {}
    for path in experiment_manifest_paths:
        resolved = _resolve_evidence_path(path, project_root=root)
        with resolved.open(encoding="utf-8") as handle:
            experiment_id = str(json.load(handle).get("experiment_id", ""))
        if experiment_id in manifests_by_id:
            raise ValueError("stability evidence contains a duplicate manifest identity")
        manifests_by_id[experiment_id] = path
    if set(manifests_by_id) != set(STABILITY_EXPERIMENT_IDS):
        raise ValueError("stability evidence requires the exact frozen manifest set")

    audits = [
        _load_audit(
            manifest_path=manifests_by_id[experiment_id],
            config=configs_by_id[experiment_id][0],
            config_path=configs_by_id[experiment_id][1],
            project_root=root,
            expected_row_count=expected_row_count,
        )
        for experiment_id in STABILITY_EXPERIMENT_IDS
    ]
    coverage_hashes = {canonical_sha256(audit.manifest["coverage"]) for audit in audits}
    if len(coverage_hashes) != 1:
        raise ValueError("stability runs do not cover the same OOF products")
    combined_registry = pd.concat(
        [audit.registry.assign(candidate=audit.candidate) for audit in audits],
        ignore_index=True,
    )
    for field in ("split_sha256", "label_map_sha256", "transform_id"):
        if combined_registry[field].nunique() != 1:
            raise ValueError(f"stability runs changed shared field {field}")
    for candidate in ("C2", "I2"):
        candidate_registry = combined_registry.loc[combined_registry["candidate"].eq(candidate)]
        for field in (
            "implementation_sha256",
            "loss_id",
            "model_family",
            "parameter_count",
        ):
            if candidate_registry[field].nunique() != 1:
                raise ValueError(f"{candidate} changed {field} between stability seeds")

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for audit in audits:
        folds = pd.to_numeric(audit.fold_metrics["macro_f1"], errors="raise")
        spring = audit.pooled_metrics["per_class"]["Spring"]
        summary_rows.append(
            {
                "candidate": audit.candidate,
                "variant": audit.variant,
                "experiment_id": audit.experiment_id,
                "seed": audit.seed,
                "pooled_macro_f1": float(audit.pooled_metrics["macro_f1"]),
                "fold_mean_macro_f1": float(folds.mean()),
                "fold_sd_macro_f1": float(folds.std(ddof=1)),
                "accuracy": float(audit.pooled_metrics["accuracy"]),
                "balanced_accuracy": float(audit.pooled_metrics["balanced_accuracy"]),
                "spring_precision": float(spring["precision"]),
                "spring_recall": float(spring["recall"]),
                "spring_f1": float(spring["f1"]),
                "five_fold_runtime_minutes": float(
                    pd.to_numeric(
                        audit.registry["runtime_seconds"],
                        errors="raise",
                    ).sum()
                    / 60
                ),
                "peak_vram_mb": float(
                    pd.to_numeric(
                        audit.registry["peak_vram_mb"],
                        errors="coerce",
                    ).max()
                ),
                "parameter_count": int(
                    pd.to_numeric(
                        audit.registry["parameter_count"],
                        errors="raise",
                    ).iloc[0]
                ),
                "median_best_epoch": float(
                    pd.to_numeric(
                        audit.registry["best_epoch"],
                        errors="raise",
                    ).median()
                ),
                "implementation_sha256": str(audit.registry["implementation_sha256"].iloc[0]),
                "config_sha256": str(audit.registry["config_sha256"].iloc[0]),
            }
        )
        for label in SEASON_LABELS:
            metrics = audit.pooled_metrics["per_class"][label]
            per_class_rows.append(
                {
                    "candidate": audit.candidate,
                    "experiment_id": audit.experiment_id,
                    "seed": audit.seed,
                    "label": label,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "support": int(metrics["support"]),
                }
            )
    seed_stability = pd.DataFrame(summary_rows).sort_values(["candidate", "seed"])
    per_class = pd.DataFrame(per_class_rows).sort_values(["candidate", "seed", "label"])
    if per_class.groupby(["candidate", "label"])["support"].nunique().gt(1).any():
        raise ValueError("stability per-class supports changed between seeds")

    by_experiment = {audit.experiment_id: audit for audit in audits}
    paired_rows = []
    for fold in range(5):
        row: dict[str, Any] = {"fold": fold}
        for candidate, primary_id, stability_id in (
            ("c2", C2_PRIMARY_EXPERIMENT_ID, C2_STABILITY_EXPERIMENT_ID),
            ("i2", I2_PRIMARY_EXPERIMENT_ID, I2_STABILITY_EXPERIMENT_ID),
        ):
            for seed_name, experiment_id in (
                ("s2753", primary_id),
                ("s2026", stability_id),
            ):
                audit = by_experiment[experiment_id]
                fold_row = audit.fold_metrics.loc[
                    pd.to_numeric(
                        audit.fold_metrics["fold"],
                        errors="raise",
                    )
                    .astype(int)
                    .eq(fold)
                ].iloc[0]
                row[f"{candidate}_{seed_name}_run_id"] = str(fold_row["run_id"])
                row[f"{candidate}_{seed_name}_macro_f1"] = float(fold_row["macro_f1"])
        row["i2_minus_c2_macro_f1_s2753"] = row["i2_s2753_macro_f1"] - row["c2_s2753_macro_f1"]
        row["i2_minus_c2_macro_f1_s2026"] = row["i2_s2026_macro_f1"] - row["c2_s2026_macro_f1"]
        paired_rows.append(row)
    paired_folds = pd.DataFrame(paired_rows)

    summary_index = seed_stability.set_index(["candidate", "seed"])
    drift_rows = []
    for candidate in ("C2", "I2"):
        primary = summary_index.loc[(candidate, 2753)]
        stability = summary_index.loc[(candidate, G5_SEED)]
        drift_rows.append(
            {
                "candidate": candidate,
                "macro_f1_s2753": float(primary["pooled_macro_f1"]),
                "macro_f1_s2026": float(stability["pooled_macro_f1"]),
                "macro_f1_s2026_minus_s2753": float(
                    stability["pooled_macro_f1"] - primary["pooled_macro_f1"]
                ),
                "spring_recall_s2753": float(primary["spring_recall"]),
                "spring_recall_s2026": float(stability["spring_recall"]),
                "spring_recall_s2026_minus_s2753": float(
                    stability["spring_recall"] - primary["spring_recall"]
                ),
                "spring_f1_s2753": float(primary["spring_f1"]),
                "spring_f1_s2026": float(stability["spring_f1"]),
                "spring_f1_s2026_minus_s2753": float(stability["spring_f1"] - primary["spring_f1"]),
            }
        )
    seed_drift = pd.DataFrame(drift_rows)
    primary_delta = float(
        summary_index.loc[("I2", 2753), "pooled_macro_f1"]
        - summary_index.loc[("C2", 2753), "pooled_macro_f1"]
    )
    stability_delta = float(
        summary_index.loc[("I2", G5_SEED), "pooled_macro_f1"]
        - summary_index.loc[("C2", G5_SEED), "pooled_macro_f1"]
    )
    if primary_delta <= 0:
        raise ValueError("primary-seed inputs do not preserve the selected I2 ordering")
    ordering_stable = stability_delta > 0
    decision = {
        "schema_version": "1.0.0",
        "gate": "G5-SEED",
        "decision_status": "closed",
        "primary_metric": "pooled_five_fold_oof_macro_f1",
        "candidate_experiment_id_seed_2753": I2_PRIMARY_EXPERIMENT_ID,
        "candidate_experiment_id_seed_2026": I2_STABILITY_EXPERIMENT_ID,
        "comparator_experiment_id_seed_2753": C2_PRIMARY_EXPERIMENT_ID,
        "comparator_experiment_id_seed_2026": C2_STABILITY_EXPERIMENT_ID,
        "i2_minus_c2_macro_f1_seed_2753": primary_delta,
        "i2_minus_c2_macro_f1_seed_2026": stability_delta,
        "ordering_stable": ordering_stable,
        "candidate_selection_affected": not ordering_stable,
        "candidate_status": (
            "supported_across_two_seeds" if ordering_stable else "ordering_reversed_unresolved"
        ),
        "current_candidate": "I2" if ordering_stable else None,
        "ultimate_winner_frozen": False,
        "interpretation_rule": (
            "The primary I2 candidate is seed-stable only if it remains above the "
            "retained C2 comparator at both complete five-fold seeds. A reversal "
            "weakens the claim and does not trigger post-hoc tuning."
        ),
        "limitations": [
            "Two seeds support a stability check but do not describe every random start.",
            "Grouped bootstrap uncertainty is still pending.",
            "I2 and C2 differ in architecture and loss, so ordering is not a causal ablation.",
            "Current softmax probabilities are uncalibrated.",
        ],
        "next_question": (
            "Keep the frozen candidates and run shortcut, error, robustness, cost, "
            "calibration, grouped-bootstrap, and Grad-CAM analysis."
        ),
    }

    histories = pd.concat([audit.histories for audit in audits], ignore_index=True)
    learning_summary = _learning_curve_summary(audits)
    registry_snapshot = pd.concat(
        [
            audit.registry.assign(
                candidate=audit.candidate,
                variant=audit.variant,
            )
            for audit in audits
        ],
        ignore_index=True,
    ).loc[:, ["candidate", "variant", *EXPERIMENT_REGISTRY_COLUMNS]]
    registry_snapshot = registry_snapshot.sort_values(["candidate", "seed", "fold", "run_id"])

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(os.path.commonpath([evidence_root.resolve(), figure_root.resolve()]))
    paths = {
        "seed_stability": evidence_root / "seed_stability.csv",
        "paired_fold_metrics": evidence_root / "paired_fold_metrics.csv",
        "seed_drift": evidence_root / "seed_drift.csv",
        "per_class_by_seed": evidence_root / "per_class_by_seed.csv",
        "learning_curves_by_fold": evidence_root / "learning_curves_by_fold.csv",
        "learning_curve_summary": evidence_root / "learning_curve_summary.csv",
        "registry_snapshot": evidence_root / "registry_snapshot.csv",
        "decision": evidence_root / "decision.json",
        "learning_curves": figure_root / "seed_stability_learning_curves.png",
        "comparison_figure": figure_root / "seed_stability_comparison.png",
    }
    atomic_write_csv(paths["seed_stability"], seed_stability)
    atomic_write_csv(paths["paired_fold_metrics"], paired_folds)
    atomic_write_csv(paths["seed_drift"], seed_drift)
    atomic_write_csv(paths["per_class_by_seed"], per_class)
    atomic_write_csv(paths["learning_curves_by_fold"], histories)
    atomic_write_csv(paths["learning_curve_summary"], learning_summary)
    atomic_write_csv(paths["registry_snapshot"], registry_snapshot)
    atomic_write_json(paths["decision"], decision)
    plot_seed_stability_learning_curves(
        learning_summary,
        paths["learning_curves"],
    )
    _plot_seed_stability_comparison(
        seed_stability,
        paths["comparison_figure"],
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
        "gate": "G5-SEED",
        "decision_status": "closed",
        "coverage_sha256": coverage_hashes.pop(),
        "ordering_stable": ordering_stable,
        "candidate_selection_affected": not ordering_stable,
        "ultimate_winner_frozen": False,
        "implementation_hashes_by_candidate": {
            candidate: str(
                combined_registry.loc[
                    combined_registry["candidate"].eq(candidate),
                    "implementation_sha256",
                ].iloc[0]
            )
            for candidate in ("C2", "I2")
        },
        "input_manifests": {
            audit.experiment_id: {
                "path": _portable_artifact_path(
                    audit.manifest_path,
                    fallback_root=root,
                ),
                "sha256": compute_sha256(audit.manifest_path),
            }
            for audit in audits
        },
        "input_configs": {
            audit.experiment_id: {
                "path": _portable_artifact_path(
                    audit.config_path,
                    fallback_root=root,
                ),
                "sha256": compute_sha256(audit.config_path),
            }
            for audit in audits
        },
        "artifacts": artifact_manifest,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


__all__ = [
    "build_seed_stability_evidence",
    "plot_seed_stability_learning_curves",
]
