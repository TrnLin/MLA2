"""Hash-linked evidence for the matched P0S/P* pretraining benchmark."""

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
from torchvision.models import ResNet18_Weights

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
from fashion.task2.pretraining import (
    P0S_EXPERIMENT_ID,
    PSTAR_EXPERIMENT_ID,
    validate_pretraining_pair,
)
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
)
from fashion.train.metrics import SEASON_LABELS

PRETRAINING_EXPERIMENT_IDS = (P0S_EXPERIMENT_ID, PSTAR_EXPERIMENT_ID)


@dataclass(frozen=True)
class _BenchmarkAudit:
    experiment_id: str
    variant: str
    config: ExperimentConfig
    config_path: Path
    manifest: dict[str, Any]
    manifest_path: Path
    registry: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_metrics: dict[str, Any]
    histories: pd.DataFrame
    training_origin: str
    weights: str | None


def _expected_boundary(experiment_id: str) -> dict[str, Any]:
    if experiment_id == P0S_EXPERIMENT_ID:
        return {
            "variant": "P0S standard-stem scratch control",
            "model_family": "resnet18_standard_scratch",
            "scratch": "true",
            "training_origin": "scratch",
            "weights": None,
        }
    if experiment_id == PSTAR_EXPERIMENT_ID:
        return {
            "variant": "P* standard-stem ImageNet benchmark",
            "model_family": "resnet18_standard_pretrained",
            "scratch": "false",
            "training_origin": "imagenet_pretrained",
            "weights": "ResNet18_Weights.DEFAULT",
        }
    raise ValueError(f"unexpected pretraining benchmark experiment: {experiment_id}")


def _load_histories(
    registry: pd.DataFrame,
    *,
    config: ExperimentConfig,
    project_root: Path,
    expected_boundary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required_metrics = {
        "train_loss",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    }
    for registry_row in registry.sort_values("fold").to_dict(orient="records"):
        history_path = _resolve_evidence_path(
            str(registry_row["history_path"]), project_root=project_root
        )
        if not history_path.is_file():
            raise ValueError(f"benchmark history does not exist: {history_path}")
        if compute_sha256(history_path) != str(registry_row["history_sha256"]):
            raise ValueError(f"benchmark history hash mismatch for {registry_row['run_id']}")
        with history_path.open(encoding="utf-8") as handle:
            history = json.load(handle)
        expected_identity = {
            "run_id": str(registry_row["run_id"]),
            "experiment_id": config.experiment_id,
            "fold": int(registry_row["fold"]),
            "seed": 2753,
        }
        for field, expected in expected_identity.items():
            if history.get(field) != expected:
                raise ValueError(
                    f"benchmark history {field} mismatch for {expected_identity['run_id']}"
                )
        if canonical_sha256(history.get("config", {})) != canonical_sha256(config.to_dict()):
            raise ValueError(f"benchmark history config mismatch for {expected_identity['run_id']}")
        boundary = history.get("model_boundary", {})
        required_boundary = {
            "benchmark_only": True,
            "final_eligible": False,
            "training_origin": expected_boundary["training_origin"],
            "weights": expected_boundary["weights"],
        }
        mismatches = [
            field
            for field, expected in required_boundary.items()
            if boundary.get(field) != expected
        ]
        if mismatches:
            raise ValueError(
                f"benchmark model boundary mismatch for {expected_identity['run_id']}: {mismatches}"
            )
        epochs = history.get("epoch_history", [])
        if not isinstance(epochs, list) or not epochs:
            raise ValueError(f"benchmark history is empty for {expected_identity['run_id']}")
        if len(epochs) != int(registry_row["epochs_completed"]):
            raise ValueError(f"benchmark epoch count mismatch for {expected_identity['run_id']}")
        observed_epochs = [int(epoch.get("epoch", -1)) for epoch in epochs]
        if observed_epochs != list(range(1, len(epochs) + 1)):
            raise ValueError(
                f"benchmark epochs are not contiguous for {expected_identity['run_id']}"
            )
        for epoch in epochs:
            missing = sorted(required_metrics - set(epoch))
            if missing:
                raise ValueError(
                    "benchmark history is missing metrics for "
                    f"{expected_identity['run_id']}: {missing}"
                )
            rows.append(
                {
                    "variant": expected_boundary["variant"],
                    "experiment_id": config.experiment_id,
                    "run_id": expected_identity["run_id"],
                    "fold": expected_identity["fold"],
                    "epoch": int(epoch["epoch"]),
                    **{name: float(epoch[name]) for name in sorted(required_metrics)},
                }
            )
    return pd.DataFrame(rows)


def _load_benchmark_audit(
    *,
    manifest_path: str | Path,
    config_path: str | Path,
    project_root: Path,
    expected_row_count: int,
) -> _BenchmarkAudit:
    manifest, resolved_manifest = _load_verified_experiment_manifest(
        manifest_path,
        project_root=project_root,
    )
    resolved_config = _resolve_evidence_path(config_path, project_root=project_root)
    config = load_experiment_config(resolved_config)
    boundary = _expected_boundary(config.experiment_id)
    if str(manifest.get("experiment_id", "")) != config.experiment_id:
        raise ValueError("benchmark config and manifest experiment identities differ")
    if tuple(manifest.get("folds", ())) != tuple(range(5)) or int(manifest.get("seed", -1)) != 2753:
        raise ValueError(f"{config.experiment_id} must contain folds 0-4 and seed 2753")
    coverage = manifest.get("coverage", {})
    if not (
        coverage.get("row_count")
        == coverage.get("unique_id_count")
        == coverage.get("expected_row_count")
        == expected_row_count
    ):
        raise ValueError(f"{config.experiment_id} has incomplete OOF coverage")
    if int(coverage.get("protected_id_count", -1)) != 0:
        raise ValueError(f"{config.experiment_id} includes protected IDs")
    if tuple(coverage.get("labels", ())) != tuple(SEASON_LABELS):
        raise ValueError(f"{config.experiment_id} has an invalid Season class order")

    registry_path = _verified_manifest_artifact(
        manifest, "registry_snapshot", project_root=project_root
    )
    fold_metrics_path = _verified_manifest_artifact(
        manifest, "fold_metrics", project_root=project_root
    )
    pooled_path = _verified_manifest_artifact(manifest, "pooled_metrics", project_root=project_root)
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    fold_metrics = pd.read_csv(fold_metrics_path)
    with pooled_path.open(encoding="utf-8") as handle:
        pooled_metrics = json.load(handle)
    missing_registry_columns = sorted(set(EXPERIMENT_REGISTRY_COLUMNS) - set(registry))
    if missing_registry_columns:
        raise ValueError(f"benchmark registry is missing columns: {missing_registry_columns}")
    run_ids = [str(value) for value in manifest.get("run_ids", ())]
    if len(run_ids) != 5 or len(set(run_ids)) != 5 or set(registry["run_id"]) != set(run_ids):
        raise ValueError(f"{config.experiment_id} registry must match five unique run IDs")
    expected_registry = {
        "stage": "g4_pretraining_benchmark",
        "experiment_id": config.experiment_id,
        "model_family": boundary["model_family"],
        "benchmark_only": "true",
        "final_eligible": "false",
        "scratch": boundary["scratch"],
        "seed": "2753",
        "git_dirty": "false",
        "loss_id": "cross_entropy",
        "primary_metric_name": "macro_f1",
        "status": "completed",
    }
    for column, expected in expected_registry.items():
        observed = set(registry[column].astype(str).str.lower())
        if observed != {expected.lower()}:
            raise ValueError(f"{config.experiment_id} registry {column} mismatch: {observed}")
    if set(pd.to_numeric(registry["fold"], errors="raise").astype(int)) != set(range(5)):
        raise ValueError(f"{config.experiment_id} registry folds are invalid")
    if set(registry["config_sha256"]) != {canonical_sha256(config.to_dict())}:
        raise ValueError(f"{config.experiment_id} registry config hash is invalid")
    for field in (
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
        "transform_id",
    ):
        if len(set(registry[field])) != 1:
            raise ValueError(f"{config.experiment_id} changed {field} across folds")

    required_fold_columns = {"run_id", "fold", "macro_f1", "accuracy"}
    missing_fold_columns = sorted(required_fold_columns - set(fold_metrics))
    if missing_fold_columns:
        raise ValueError(f"benchmark fold metrics are missing columns: {missing_fold_columns}")
    if len(fold_metrics) != 5 or set(fold_metrics["run_id"].astype(str)) != set(run_ids):
        raise ValueError(f"{config.experiment_id} fold metrics do not match five runs")
    if set(pd.to_numeric(fold_metrics["fold"], errors="raise").astype(int)) != set(range(5)):
        raise ValueError(f"{config.experiment_id} fold metrics have invalid folds")
    primary_by_run = registry.set_index("run_id")["primary_metric_value"].astype(float)
    for row in fold_metrics.to_dict(orient="records"):
        if not np.isclose(
            float(row["macro_f1"]),
            float(primary_by_run.loc[str(row["run_id"])]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{config.experiment_id} fold metric differs from registry")
    pooled_macro_f1 = float(pooled_metrics.get("macro_f1", np.nan))
    if not np.isfinite(pooled_macro_f1) or not np.isclose(
        pooled_macro_f1,
        float(manifest.get("pooled_macro_f1", np.nan)),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{config.experiment_id} pooled macro-F1 is invalid")
    if tuple(pooled_metrics.get("per_class", {})) != tuple(SEASON_LABELS):
        raise ValueError(f"{config.experiment_id} pooled per-class order is invalid")

    histories = _load_histories(
        registry,
        config=config,
        project_root=project_root,
        expected_boundary=boundary,
    )
    return _BenchmarkAudit(
        experiment_id=config.experiment_id,
        variant=boundary["variant"],
        config=config,
        config_path=resolved_config,
        manifest=manifest,
        manifest_path=resolved_manifest,
        registry=registry,
        fold_metrics=fold_metrics,
        pooled_metrics=pooled_metrics,
        histories=histories,
        training_origin=boundary["training_origin"],
        weights=boundary["weights"],
    )


def _learning_curve_summary(audits: Sequence[_BenchmarkAudit]) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    metrics = (
        "train_loss",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    )
    for audit in audits:
        fold_horizons = audit.histories.groupby("fold")["epoch"].max()
        common_horizon = int(fold_horizons.min())
        source = audit.histories.loc[audit.histories["epoch"].le(common_horizon)]
        aggregations: dict[str, tuple[str, str]] = {"fold_count": ("fold", "nunique")}
        for metric in metrics:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_sd"] = (metric, "std")
        summary = (
            source.groupby(["variant", "experiment_id", "epoch"], as_index=False)
            .agg(**aggregations)
            .fillna(0.0)
        )
        if not summary["fold_count"].eq(5).all():
            raise ValueError(f"{audit.experiment_id} learning means require five folds")
        summary["common_five_fold_horizon"] = common_horizon
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def plot_pretraining_learning_curves(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Render teacher-style loss and validation-score curves for P0S and P*."""
    figure = Figure(figsize=(12.0, 8.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, squeeze=False)
    for row_index, experiment_id in enumerate(PRETRAINING_EXPERIMENT_IDS):
        subset = summary.loc[summary["experiment_id"].eq(experiment_id)].sort_values("epoch")
        if subset.empty:
            raise ValueError(f"learning summary is missing {experiment_id}")
        epochs = subset["epoch"].to_numpy(dtype=float)
        panel_label = "P0S scratch" if experiment_id == P0S_EXPERIMENT_ID else "P* ImageNet"
        loss_axis = axes[row_index, 0]
        metric_axis = axes[row_index, 1]
        for prefix, label, colour in (
            ("train_loss", "Train loss", "#2563EB"),
            ("validation_loss", "Validation loss", "#F97316"),
        ):
            mean = subset[f"{prefix}_mean"].to_numpy(dtype=float)
            sd = subset[f"{prefix}_sd"].to_numpy(dtype=float)
            loss_axis.plot(epochs, mean, label=label, color=colour, linewidth=2.0)
            loss_axis.fill_between(epochs, mean - sd, mean + sd, color=colour, alpha=0.15)
        for prefix, label, colour in (
            ("validation_accuracy", "Validation accuracy", "#7C3AED"),
            ("validation_macro_f1", "Validation macro-F1", "#16A34A"),
        ):
            mean = subset[f"{prefix}_mean"].to_numpy(dtype=float)
            sd = subset[f"{prefix}_sd"].to_numpy(dtype=float)
            metric_axis.plot(epochs, mean, label=label, color=colour, linewidth=2.0)
            metric_axis.fill_between(epochs, mean - sd, mean + sd, color=colour, alpha=0.15)
        loss_axis.set_title(f"{panel_label}: loss (5-fold mean ± SD)")
        metric_axis.set_title(f"{panel_label}: validation scores (5-fold mean ± SD)")
        loss_axis.set_ylabel("Loss")
        metric_axis.set_ylabel("Score")
        metric_axis.set_ylim(0.0, 1.0)
        for axis in (loss_axis, metric_axis):
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
    output = Path(output_path)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output, buffer.getvalue())
    figure.clear()
    return output


def _plot_pretraining_effect(effect: pd.DataFrame, output_path: Path) -> None:
    values = effect["pstar_minus_p0s"].to_numpy(dtype=float)
    colours = np.where(values >= 0.0, "#16A34A", "#F97316")
    figure = Figure(figsize=(10.0, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    bars = axis.bar(effect["metric"], values, color=colours, edgecolor="#334155")
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.bar_label(bars, fmt="%+.4f", padding=3)
    axis.set_ylabel("P* minus P0S score")
    axis.set_title("Matched pretraining effect under the fixed Task 2 pipeline")
    axis.grid(axis="y", alpha=0.2)
    axis.tick_params(axis="x", rotation=20)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    figure.clear()


def build_pretraining_benchmark_evidence(
    *,
    experiment_manifest_paths: Sequence[str | Path],
    experiment_config_paths: Sequence[str | Path],
    project_root: str | Path = ROOT,
    expected_row_count: int = 32_753,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR / "pretraining_benchmark",
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Audit the matched pair, quantify P* minus P0S, and keep P* non-selectable."""
    if expected_row_count < 1:
        raise ValueError("expected_row_count must be positive")
    if len(experiment_manifest_paths) != 2 or len(experiment_config_paths) != 2:
        raise ValueError("pretraining evidence requires exactly two manifests and configs")
    root = Path(project_root)
    configs = [
        load_experiment_config(_resolve_evidence_path(path, project_root=root))
        for path in experiment_config_paths
    ]
    scratch_config, pretrained_config = validate_pretraining_pair(configs)
    config_paths_by_id = {
        config.experiment_id: _resolve_evidence_path(path, project_root=root)
        for config, path in zip(configs, experiment_config_paths, strict=True)
    }
    manifests_by_id: dict[str, str | Path] = {}
    for path in experiment_manifest_paths:
        resolved = _resolve_evidence_path(path, project_root=root)
        with resolved.open(encoding="utf-8") as handle:
            experiment_id = str(json.load(handle).get("experiment_id", ""))
        if experiment_id in manifests_by_id:
            raise ValueError("pretraining evidence contains a duplicate manifest identity")
        manifests_by_id[experiment_id] = path
    if set(manifests_by_id) != set(PRETRAINING_EXPERIMENT_IDS):
        raise ValueError("pretraining evidence requires exactly the P0S and P* manifests")

    audits = [
        _load_benchmark_audit(
            manifest_path=manifests_by_id[config.experiment_id],
            config_path=config_paths_by_id[config.experiment_id],
            project_root=root,
            expected_row_count=expected_row_count,
        )
        for config in (scratch_config, pretrained_config)
    ]
    coverage_hashes = {canonical_sha256(audit.manifest["coverage"]) for audit in audits}
    if len(coverage_hashes) != 1:
        raise ValueError("P0S and P* do not cover the same canonical OOF products")
    shared_fields = (
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
        "transform_id",
        "loss_id",
    )
    combined_registry = pd.concat([audit.registry for audit in audits], ignore_index=True)
    for field in shared_fields:
        if len(set(combined_registry[field].astype(str))) != 1:
            raise ValueError(f"P0S and P* changed matched field {field}")

    comparison_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for audit in audits:
        fold_scores = pd.to_numeric(audit.fold_metrics["macro_f1"], errors="raise")
        registry = audit.registry
        comparison_rows.append(
            {
                "variant": audit.variant,
                "experiment_id": audit.experiment_id,
                "model_family": audit.config.model_family,
                "training_origin": audit.training_origin,
                "weights": audit.weights or "none",
                "scratch": audit.experiment_id == P0S_EXPERIMENT_ID,
                "benchmark_only": True,
                "final_eligible": False,
                "pooled_macro_f1": float(audit.pooled_metrics["macro_f1"]),
                "fold_mean_macro_f1": float(fold_scores.mean()),
                "fold_sd_macro_f1": float(fold_scores.std(ddof=1)),
                "spring_f1": float(audit.pooled_metrics["per_class"]["Spring"]["f1"]),
                "five_fold_runtime_minutes": float(
                    pd.to_numeric(registry["runtime_seconds"], errors="raise").sum() / 60.0
                ),
                "peak_vram_mb": float(
                    pd.to_numeric(registry["peak_vram_mb"], errors="coerce").max()
                ),
                "parameter_count": int(
                    pd.to_numeric(registry["parameter_count"], errors="raise").iloc[0]
                ),
                "median_best_epoch": float(
                    pd.to_numeric(registry["best_epoch"], errors="raise").median()
                ),
                "config_sha256": str(registry["config_sha256"].iloc[0]),
                "implementation_sha256": str(registry["implementation_sha256"].iloc[0]),
            }
        )
        for label in SEASON_LABELS:
            metrics = audit.pooled_metrics["per_class"][label]
            per_class_rows.append(
                {
                    "variant": audit.variant,
                    "experiment_id": audit.experiment_id,
                    "label": label,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "support": int(metrics["support"]),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    per_class_long = pd.DataFrame(per_class_rows)
    per_class = per_class_long.pivot(
        index="label", columns="experiment_id", values=["precision", "recall", "f1", "support"]
    ).reindex(SEASON_LABELS)
    per_class.columns = [
        f"{metric}_{'p0s' if experiment_id == P0S_EXPERIMENT_ID else 'pstar'}"
        for metric, experiment_id in per_class.columns
    ]
    per_class = per_class.reset_index()
    per_class["pstar_minus_p0s_f1"] = per_class["f1_pstar"] - per_class["f1_p0s"]

    fold_parts = []
    for audit, prefix in zip(audits, ("p0s", "pstar"), strict=True):
        part = audit.fold_metrics.loc[:, ["fold", "run_id", "macro_f1", "accuracy"]].copy()
        part = part.rename(columns={name: f"{prefix}_{name}" for name in part if name != "fold"})
        fold_parts.append(part)
    paired_folds = fold_parts[0].merge(fold_parts[1], on="fold", validate="one_to_one")
    paired_folds["pstar_minus_p0s_macro_f1"] = (
        paired_folds["pstar_macro_f1"] - paired_folds["p0s_macro_f1"]
    )
    paired_folds["pstar_minus_p0s_accuracy"] = (
        paired_folds["pstar_accuracy"] - paired_folds["p0s_accuracy"]
    )
    overall_delta = float(
        comparison.set_index("experiment_id").loc[PSTAR_EXPERIMENT_ID, "pooled_macro_f1"]
        - comparison.set_index("experiment_id").loc[P0S_EXPERIMENT_ID, "pooled_macro_f1"]
    )
    spring_delta = float(
        comparison.set_index("experiment_id").loc[PSTAR_EXPERIMENT_ID, "spring_f1"]
        - comparison.set_index("experiment_id").loc[P0S_EXPERIMENT_ID, "spring_f1"]
    )
    effect = pd.concat(
        [
            pd.DataFrame([{"metric": "Overall macro-F1", "pstar_minus_p0s": overall_delta}]),
            per_class.loc[:, ["label", "pstar_minus_p0s_f1"]]
            .rename(columns={"label": "metric", "pstar_minus_p0s_f1": "pstar_minus_p0s"})
            .assign(metric=lambda frame: frame["metric"] + " F1"),
        ],
        ignore_index=True,
    )
    histories = pd.concat([audit.histories for audit in audits], ignore_index=True)
    learning_summary = _learning_curve_summary(audits)
    registry_snapshot = pd.concat(
        [audit.registry.assign(variant=audit.variant) for audit in audits],
        ignore_index=True,
    ).loc[:, ["variant", *EXPERIMENT_REGISTRY_COLUMNS]]
    registry_snapshot = registry_snapshot.sort_values(["experiment_id", "fold", "run_id"])

    resolved_weights = ResNet18_Weights.DEFAULT
    weight_provenance = {
        "declaration": "ResNet18_Weights.DEFAULT",
        "resolved_enum": f"ResNet18_Weights.{resolved_weights.name}",
        "url": resolved_weights.url,
    }
    decision = {
        "schema_version": "1.0.0",
        "gate": "G4-PSTAR",
        "decision_status": "closed",
        "scientific_question": (
            "Under the same folds, transforms, optimiser, loss, and budget, how much "
            "does ImageNet initialisation change standard-stem ResNet18 Season performance?"
        ),
        "control_experiment_id": P0S_EXPERIMENT_ID,
        "benchmark_experiment_id": PSTAR_EXPERIMENT_ID,
        "observed_pstar_minus_p0s_macro_f1": overall_delta,
        "observed_pstar_minus_p0s_spring_f1": spring_delta,
        "paired_fold_mean_delta": float(paired_folds["pstar_minus_p0s_macro_f1"].mean()),
        "pretrained_weight_provenance": weight_provenance,
        "p0s_benchmark_only": True,
        "pstar_benchmark_only": True,
        "pstar_final_eligible": False,
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "interpretation_rule": (
            "P* minus P0S estimates only the initialisation effect inside this fixed "
            "80x60 Task 2 pipeline. P* is a comparison ceiling, never a candidate."
        ),
        "limitations": [
            "This gate uses the primary seed only.",
            "The fixed 80x60 project transform is not TorchVision's 224x224 ImageNet recipe.",
            "Softmax outputs are uncalibrated and support no calibration claim here.",
            "P* cannot be compared causally with I2 because architecture and loss differ.",
        ],
        "next_question": "Run the frozen scratch finalist with seed 2026 for stability.",
    }

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(os.path.commonpath([evidence_root.resolve(), figure_root.resolve()]))
    paths = {
        "comparison": evidence_root / "comparison.csv",
        "paired_fold_metrics": evidence_root / "paired_fold_metrics.csv",
        "per_class_comparison": evidence_root / "per_class_comparison.csv",
        "pretraining_effect": evidence_root / "pretraining_effect.csv",
        "learning_curves_by_fold": evidence_root / "learning_curves_by_fold.csv",
        "learning_curve_summary": evidence_root / "learning_curve_summary.csv",
        "registry_snapshot": evidence_root / "registry_snapshot.csv",
        "decision": evidence_root / "decision.json",
        "learning_curves": figure_root / "pretraining_benchmark_learning_curves.png",
        "effect_figure": figure_root / "pretraining_benchmark_effect.png",
    }
    atomic_write_csv(paths["comparison"], comparison)
    atomic_write_csv(paths["paired_fold_metrics"], paired_folds)
    atomic_write_csv(paths["per_class_comparison"], per_class)
    atomic_write_csv(paths["pretraining_effect"], effect)
    atomic_write_csv(paths["learning_curves_by_fold"], histories)
    atomic_write_csv(paths["learning_curve_summary"], learning_summary)
    atomic_write_csv(paths["registry_snapshot"], registry_snapshot)
    atomic_write_json(paths["decision"], decision)
    plot_pretraining_learning_curves(learning_summary, paths["learning_curves"])
    _plot_pretraining_effect(effect, paths["effect_figure"])

    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=common_root),
            "sha256": compute_sha256(path),
        }
        for name, path in paths.items()
    }
    matched_payload = scratch_config.to_dict()
    for field in ("experiment_id", "model_family"):
        matched_payload.pop(field)
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G4-PSTAR",
        "decision_status": "closed",
        "coverage_sha256": coverage_hashes.pop(),
        "matched_protocol_sha256": canonical_sha256(matched_payload),
        "shared_run_contract_sha256": canonical_sha256(
            {field: str(combined_registry[field].iloc[0]) for field in shared_fields}
        ),
        "observed_pstar_minus_p0s_macro_f1": overall_delta,
        "pretrained_weight_provenance": weight_provenance,
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "input_manifests": {
            audit.experiment_id: {
                "path": _portable_artifact_path(audit.manifest_path, fallback_root=root),
                "sha256": compute_sha256(audit.manifest_path),
            }
            for audit in audits
        },
        "input_configs": {
            audit.experiment_id: {
                "path": _portable_artifact_path(audit.config_path, fallback_root=root),
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
    "build_pretraining_benchmark_evidence",
    "plot_pretraining_learning_curves",
]
