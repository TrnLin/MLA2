"""Hash-linked I2 transfer evidence and training-fold ArticleType shortcut slices."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fashion.config import (
    ROOT,
    SPLITS_CSV,
    TASK2_EVIDENCE_DIR,
    TASK2_FIGURE_DIR,
)
from fashion.data.dataset import get_cv_split, get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import (
    EXPERIMENT_REGISTRY_COLUMNS,
    _load_verified_experiment_manifest,
    _portable_artifact_path,
    _resolve_evidence_path,
    _verified_manifest_artifact,
)
from fashion.task2.experiments import load_experiment_config
from fashion.task2.multitask import load_i2_config
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
)
from fashion.train.metrics import (
    SEASON_LABELS,
    multiclass_metrics,
    validate_metrics_match_oof,
    validate_oof,
    validate_oof_identity,
)

I2_REFERENCE_EXPERIMENT_ID = "g3-c1-t1-smallcnn"
I2_EXPERIMENT_IDS = (
    "g4-i2-article-type-lambda-0-1-c1",
    "g4-i2-article-type-lambda-0-3-c1",
)
I2_MINIMUM_OVERALL_GAIN = 0.003
I2_MINIMUM_CONFLICT_GAIN = 0.010
I2_MAXIMUM_OVERALL_LOSS = 0.002
SLICE_ORDER = (
    "aligned",
    "conflict",
    "unseen_article_type",
    "missing_article_type",
)


@dataclass(frozen=True)
class _ExperimentAudit:
    experiment_id: str
    variant: str
    auxiliary_weight: float
    manifest_path: Path
    config_path: Path
    manifest: dict[str, Any]
    registry: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof: pd.DataFrame
    metrics: dict[str, Any]
    histories: pd.DataFrame


def fit_article_type_majorities(training_frame: pd.DataFrame) -> pd.DataFrame:
    """Fit deterministic ArticleType-to-Season majorities on training rows only."""
    required = {"id", "season", "articleType", "has_articleType_label"}
    missing = sorted(required - set(training_frame.columns))
    if missing:
        raise ValueError(f"training frame is missing shortcut columns: {missing}")
    valid = training_frame.loc[training_frame["has_articleType_label"].astype(bool)].copy()
    if valid.empty:
        raise ValueError("shortcut mapping requires at least one auxiliary-labelled row")
    unknown_seasons = sorted(set(valid["season"].astype(str)) - set(SEASON_LABELS))
    if unknown_seasons:
        raise ValueError(f"shortcut mapping contains unknown Season labels: {unknown_seasons}")
    counts = (
        valid.groupby(["articleType", "season"], observed=True)
        .size()
        .rename("majority_count")
        .reset_index()
    )
    label_order = {label: index for index, label in enumerate(SEASON_LABELS)}
    counts["label_order"] = counts["season"].map(label_order)
    counts = counts.sort_values(
        ["articleType", "majority_count", "label_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    majority = counts.drop_duplicates("articleType", keep="first").copy()
    totals = (
        counts.groupby("articleType", observed=True)["majority_count"]
        .sum()
        .rename("training_labeled_count")
    )
    majority = majority.join(totals, on="articleType")
    majority["majority_share"] = (
        majority["majority_count"] / majority["training_labeled_count"]
    )
    majority = majority.rename(columns={"season": "shortcut_majority_season"})
    return majority.loc[
        :,
        [
            "articleType",
            "shortcut_majority_season",
            "majority_count",
            "training_labeled_count",
            "majority_share",
        ],
    ].reset_index(drop=True)


def build_article_type_shortcut_audit(
    splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create fold-fitted aligned/conflict assignments for every valid Season row."""
    development = get_samples(splits, partition="development")
    expected = get_samples(development, target="season").reset_index(drop=True)
    mapping_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for fold in range(5):
        training, validation = get_cv_split(splits, fold)
        training = get_samples(training, target="season").reset_index(drop=True)
        validation = get_samples(validation, target="season").reset_index(drop=True)
        mapping = fit_article_type_majorities(training)
        auxiliary_training = training.loc[training["has_articleType_label"].astype(bool)]
        training_ids = sorted(int(value) for value in auxiliary_training["id"])
        mapping.insert(0, "fold", fold)
        mapping["training_id_sha256"] = canonical_sha256(training_ids)
        mapping_parts.append(mapping)

        assignments = validation.loc[
            :, ["id", "cv_fold", "season", "articleType", "has_articleType_label"]
        ].copy()
        assignments = assignments.rename(columns={"cv_fold": "fold"})
        assignments["fold"] = pd.to_numeric(assignments["fold"], errors="raise").astype(int)
        assignments = assignments.merge(
            mapping.loc[:, ["articleType", "shortcut_majority_season"]],
            on="articleType",
            how="left",
            validate="many_to_one",
        )
        assignments["shortcut_slice"] = np.select(
            [
                ~assignments["has_articleType_label"].astype(bool),
                assignments["shortcut_majority_season"].isna(),
                assignments["season"].eq(assignments["shortcut_majority_season"]),
            ],
            ["missing_article_type", "unseen_article_type", "aligned"],
            default="conflict",
        )
        assignments["articleType"] = (
            assignments["articleType"].astype("string").fillna("<missing>")
        )
        assignments["shortcut_majority_season"] = (
            assignments["shortcut_majority_season"]
            .astype("string")
            .fillna("<unmapped>")
        )
        assignment_parts.append(assignments)
        fold_audits.append(
            {
                "fold": fold,
                "training_products": len(training),
                "auxiliary_training_products": len(auxiliary_training),
                "validation_products": len(validation),
                "mapped_article_types": len(mapping),
                "training_id_sha256": canonical_sha256(training_ids),
            }
        )

    mappings = pd.concat(mapping_parts, ignore_index=True)
    assignments = pd.concat(assignment_parts, ignore_index=True)
    assignments["id"] = pd.to_numeric(assignments["id"], errors="raise").astype(int)
    if assignments["id"].duplicated().any():
        raise ValueError("shortcut audit assigned a development ID more than once")
    if set(assignments["id"]) != set(expected["id"].astype(int)):
        raise ValueError("shortcut audit does not cover every valid Season development ID")
    if not assignments["fold"].isin(range(5)).all():
        raise ValueError("shortcut audit contains a non-canonical fold")
    if set(assignments["shortcut_slice"]) - set(SLICE_ORDER):
        raise ValueError("shortcut audit produced an unknown slice")
    fold_summary = pd.DataFrame(fold_audits)
    return mappings, assignments, fold_summary


def _load_registry_oof(
    registry: pd.DataFrame,
    *,
    experiment_id: str,
    project_root: Path,
    expected_ids: pd.Series,
    expected_targets: dict[int, str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for row in registry.sort_values("fold").to_dict(orient="records"):
        prediction_path = _resolve_evidence_path(
            str(row["prediction_path"]), project_root=project_root
        )
        if not prediction_path.is_file():
            raise ValueError(f"OOF artifact does not exist: {prediction_path}")
        if compute_sha256(prediction_path) != str(row["prediction_sha256"]):
            raise ValueError(f"OOF artifact hash mismatch for {row['run_id']}")
        oof = pd.read_csv(prediction_path)
        validate_oof_identity(
            oof,
            run_id=str(row["run_id"]),
            experiment_id=experiment_id,
            fold=int(row["fold"]),
            seed=int(row["seed"]),
        )
        parts.append(oof)
    pooled = pd.concat(parts, ignore_index=True)
    validate_oof(
        pooled,
        expected_ids=expected_ids,
        labels=SEASON_LABELS,
        expected_targets=expected_targets,
    )
    return pooled


def _load_histories(
    registry: pd.DataFrame,
    *,
    project_root: Path,
    auxiliary_weight: float,
) -> pd.DataFrame:
    if auxiliary_weight == 0.0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for registry_row in registry.sort_values("fold").to_dict(orient="records"):
        history_path = _resolve_evidence_path(
            str(registry_row["history_path"]), project_root=project_root
        )
        if not history_path.is_file():
            raise ValueError(f"I2 history does not exist: {history_path}")
        if compute_sha256(history_path) != str(registry_row["history_sha256"]):
            raise ValueError(f"I2 history hash mismatch for {registry_row['run_id']}")
        with history_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        auxiliary = payload.get("auxiliary", {})
        if auxiliary.get("target") != "articleType" or not np.isclose(
            float(auxiliary.get("loss_weight", np.nan)),
            auxiliary_weight,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"I2 auxiliary audit mismatch for {registry_row['run_id']}")
        history = payload.get("epoch_history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"I2 epoch history is missing for {registry_row['run_id']}")
        for epoch_row in history:
            required = {
                "epoch",
                "train_total_loss",
                "validation_total_loss",
                "validation_accuracy",
                "validation_macro_f1",
            }
            if required - set(epoch_row):
                raise ValueError(f"I2 epoch history is incomplete for {registry_row['run_id']}")
            rows.append(
                {
                    "experiment_id": str(registry_row["experiment_id"]),
                    "auxiliary_weight": auxiliary_weight,
                    "fold": int(registry_row["fold"]),
                    "run_id": str(registry_row["run_id"]),
                    **epoch_row,
                }
            )
    histories = pd.DataFrame(rows)
    if set(histories["fold"]) != set(range(5)):
        raise ValueError("I2 histories require all five folds")
    return histories


def _load_experiment_audit(
    *,
    manifest_path: str | Path,
    config_path: str | Path,
    variant: str,
    auxiliary_weight: float,
    project_root: Path,
    expected_ids: pd.Series,
    expected_targets: dict[int, str],
) -> _ExperimentAudit:
    manifest, resolved_manifest = _load_verified_experiment_manifest(
        manifest_path,
        project_root=project_root,
    )
    resolved_config = _resolve_evidence_path(config_path, project_root=project_root)
    if auxiliary_weight == 0.0:
        config: Any = load_experiment_config(resolved_config)
        canonical_config = config.to_dict()
    else:
        config = load_i2_config(resolved_config)
        canonical_config = config.to_dict()
    experiment_id = str(manifest.get("experiment_id", ""))
    if config.experiment_id != experiment_id:
        raise ValueError("I2 config and manifest experiment identities differ")
    if tuple(manifest.get("folds", ())) != tuple(range(5)) or int(
        manifest.get("seed", -1)
    ) != 2753:
        raise ValueError(f"{experiment_id} must contain folds 0-4 and seed 2753")
    coverage = manifest.get("coverage", {})
    if not (
        coverage.get("row_count")
        == coverage.get("unique_id_count")
        == coverage.get("expected_row_count")
        == len(expected_ids)
    ):
        raise ValueError(f"{experiment_id} has incomplete OOF coverage")
    if int(coverage.get("protected_id_count", -1)) != 0:
        raise ValueError(f"{experiment_id} includes protected IDs")

    registry_path = _verified_manifest_artifact(
        manifest, "registry_snapshot", project_root=project_root
    )
    fold_metrics_path = _verified_manifest_artifact(
        manifest, "fold_metrics", project_root=project_root
    )
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    fold_metrics = pd.read_csv(fold_metrics_path)
    run_ids = [str(value) for value in manifest.get("run_ids", ())]
    if len(registry) != 5 or set(registry["run_id"]) != set(run_ids):
        raise ValueError(f"{experiment_id} registry does not match five manifest runs")
    expected_registry = {
        "experiment_id": experiment_id,
        "model_family": "smallcnn",
        "benchmark_only": "false",
        "final_eligible": "true",
        "scratch": "true",
        "seed": "2753",
        "git_dirty": "false",
        "status": "completed",
    }
    for column, expected in expected_registry.items():
        observed = set(registry[column].astype(str).str.lower())
        if observed != {expected.lower()}:
            raise ValueError(f"{experiment_id} registry {column} mismatch: {observed}")
    if set(pd.to_numeric(registry["fold"], errors="raise").astype(int)) != set(range(5)):
        raise ValueError(f"{experiment_id} registry folds are invalid")
    if set(registry["config_sha256"]) != {canonical_sha256(canonical_config)}:
        raise ValueError(f"{experiment_id} registry config hash is invalid")
    for field in ("split_sha256", "label_map_sha256", "implementation_sha256"):
        if len(set(registry[field])) != 1:
            raise ValueError(f"{experiment_id} changed {field} across folds")

    oof = _load_registry_oof(
        registry,
        experiment_id=experiment_id,
        project_root=project_root,
        expected_ids=expected_ids,
        expected_targets=expected_targets,
    )
    probability_columns = [f"prob_{label}" for label in SEASON_LABELS]
    metrics = multiclass_metrics(
        oof["y_true"].astype(str),
        probabilities=oof.loc[:, probability_columns].to_numpy(dtype=float),
        labels=SEASON_LABELS,
        y_pred=oof["y_pred"].astype(str),
    )
    validate_metrics_match_oof(oof, metrics, labels=SEASON_LABELS)
    if not np.isclose(
        float(manifest.get("pooled_macro_f1", np.nan)),
        float(metrics["macro_f1"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{experiment_id} pooled metric differs from its OOF bytes")
    histories = _load_histories(
        registry,
        project_root=project_root,
        auxiliary_weight=auxiliary_weight,
    )
    return _ExperimentAudit(
        experiment_id=experiment_id,
        variant=variant,
        auxiliary_weight=auxiliary_weight,
        manifest_path=resolved_manifest,
        config_path=resolved_config,
        manifest=manifest,
        registry=registry,
        fold_metrics=fold_metrics,
        oof=oof,
        metrics=metrics,
        histories=histories,
    )


def _slice_metrics(audit: _ExperimentAudit, assignments: pd.DataFrame) -> pd.DataFrame:
    merged = audit.oof.merge(
        assignments.loc[
            :,
            [
                "id",
                "fold",
                "season",
                "articleType",
                "shortcut_majority_season",
                "shortcut_slice",
            ],
        ],
        on="id",
        how="left",
        suffixes=("_oof", "_slice"),
        validate="one_to_one",
    )
    if merged["shortcut_slice"].isna().any():
        raise ValueError(f"{audit.experiment_id} has OOF rows without shortcut assignments")
    if not merged["y_true"].astype(str).eq(merged["season"].astype(str)).all():
        raise ValueError(f"{audit.experiment_id} OOF truth differs from shortcut audit")
    if not pd.to_numeric(merged["fold_oof"], errors="raise").astype(int).eq(
        pd.to_numeric(merged["fold_slice"], errors="raise").astype(int)
    ).all():
        raise ValueError(f"{audit.experiment_id} OOF folds differ from shortcut audit")
    probability_columns = [f"prob_{label}" for label in SEASON_LABELS]
    rows: list[dict[str, Any]] = []
    for slice_name in SLICE_ORDER:
        subset = merged.loc[merged["shortcut_slice"].eq(slice_name)]
        if subset.empty:
            rows.append(
                {
                    "variant": audit.variant,
                    "experiment_id": audit.experiment_id,
                    "auxiliary_weight": audit.auxiliary_weight,
                    "shortcut_slice": slice_name,
                    "support": 0,
                    "macro_f1": np.nan,
                    "accuracy": np.nan,
                }
            )
            continue
        metrics = multiclass_metrics(
            subset["y_true"].astype(str),
            probabilities=subset.loc[:, probability_columns].to_numpy(dtype=float),
            labels=SEASON_LABELS,
            y_pred=subset["y_pred"].astype(str),
        )
        rows.append(
            {
                "variant": audit.variant,
                "experiment_id": audit.experiment_id,
                "auxiliary_weight": audit.auxiliary_weight,
                "shortcut_slice": slice_name,
                "support": len(subset),
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def apply_i2_selection_rule(
    comparison: pd.DataFrame,
    *,
    minimum_overall_gain: float = I2_MINIMUM_OVERALL_GAIN,
    minimum_conflict_gain: float = I2_MINIMUM_CONFLICT_GAIN,
    maximum_overall_loss: float = I2_MAXIMUM_OVERALL_LOSS,
) -> tuple[pd.DataFrame, str]:
    """Apply the frozen I2 transfer rule, prioritising pooled Season macro-F1."""
    thresholds = (minimum_overall_gain, minimum_conflict_gain, maximum_overall_loss)
    if any(not np.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("I2 thresholds must be finite and non-negative")
    required = {
        "experiment_id",
        "auxiliary_weight",
        "delta_macro_f1_vs_reference",
        "delta_conflict_macro_f1_vs_reference",
    }
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"I2 comparison is missing selection columns: {missing}")
    decided = comparison.copy()
    decided["passes_overall_gain"] = (
        decided["delta_macro_f1_vs_reference"] >= minimum_overall_gain
    )
    decided["passes_conflict_gain"] = (
        decided["delta_conflict_macro_f1_vs_reference"] >= minimum_conflict_gain
    ) & (decided["delta_macro_f1_vs_reference"] >= -maximum_overall_loss)
    decided["passes_i2_gate"] = (
        decided["passes_overall_gain"] | decided["passes_conflict_gain"]
    ) & decided["auxiliary_weight"].gt(0.0)
    passing = decided.loc[decided["passes_i2_gate"]].sort_values(
        ["pooled_macro_f1", "auxiliary_weight"],
        ascending=[False, True],
        kind="stable",
    )
    selected = (
        str(passing.iloc[0]["experiment_id"])
        if not passing.empty
        else I2_REFERENCE_EXPERIMENT_ID
    )
    decided["selected_by_i2_gate"] = decided["experiment_id"].eq(selected)
    return decided, selected


def _learning_curve_summary(audits: list[_ExperimentAudit]) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    metrics = (
        "train_total_loss",
        "validation_total_loss",
        "validation_accuracy",
        "validation_macro_f1",
    )
    for audit in audits:
        histories = audit.histories.copy()
        fold_horizons = histories.groupby("fold")["epoch"].max()
        common_horizon = int(fold_horizons.min())
        source = histories.loc[histories["epoch"].le(common_horizon)]
        aggregations: dict[str, tuple[str, str]] = {"fold_count": ("fold", "nunique")}
        for metric in metrics:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_sd"] = (metric, "std")
        summary = (
            source.groupby(
                ["experiment_id", "auxiliary_weight", "epoch"],
                as_index=False,
                sort=True,
            )
            .agg(**aggregations)
            .fillna(0.0)
        )
        if not summary["fold_count"].eq(5).all():
            raise ValueError(f"{audit.experiment_id} learning means require five folds")
        summary["common_five_fold_horizon"] = common_horizon
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def plot_i2_learning_curves(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Render teacher-style loss and validation-performance curves for both lambdas."""
    figure = Figure(figsize=(12.0, 8.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, squeeze=False)
    for row_index, experiment_id in enumerate(I2_EXPERIMENT_IDS):
        subset = summary.loc[summary["experiment_id"].eq(experiment_id)].sort_values("epoch")
        if subset.empty:
            raise ValueError(f"learning summary is missing {experiment_id}")
        epochs = subset["epoch"].to_numpy(dtype=float)
        weight = float(subset["auxiliary_weight"].iloc[0])
        loss_axis = axes[row_index, 0]
        metric_axis = axes[row_index, 1]
        for prefix, label, colour in (
            ("train_total_loss", "Train total loss", "#2563EB"),
            ("validation_total_loss", "Validation total loss", "#F97316"),
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
        loss_axis.set_title(f"I2 lambda={weight:.1f}: total loss (five-fold mean ± SD)")
        metric_axis.set_title(
            f"I2 lambda={weight:.1f}: Season validation metrics (five-fold mean ± SD)"
        )
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


def _plot_transfer_deltas(comparison: pd.DataFrame, output_path: Path) -> None:
    candidates = comparison.loc[comparison["auxiliary_weight"].gt(0.0)].sort_values(
        "auxiliary_weight"
    )
    labels = [f"lambda={value:.1f}" for value in candidates["auxiliary_weight"]]
    metrics = (
        ("delta_macro_f1_vs_reference", "Overall", "#2563EB"),
        ("delta_aligned_macro_f1_vs_reference", "Aligned", "#16A34A"),
        ("delta_conflict_macro_f1_vs_reference", "Conflict", "#F97316"),
    )
    figure = Figure(figsize=(9.0, 5.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = np.arange(len(candidates), dtype=float)
    width = 0.23
    all_values: list[float] = []
    for offset, (column, label, colour) in zip((-width, 0.0, width), metrics, strict=True):
        values = candidates[column].to_numpy(dtype=float)
        all_values.extend(values.tolist())
        bars = axis.bar(positions + offset, values, width=width, label=label, color=colour)
        axis.bar_label(bars, fmt="%+.4f", padding=3, fontsize=8)
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.axhline(
        I2_MINIMUM_OVERALL_GAIN,
        color="#2563EB",
        linestyle="--",
        linewidth=1.0,
        label=f"Overall gate +{I2_MINIMUM_OVERALL_GAIN:.3f}",
    )
    axis.axhline(
        I2_MINIMUM_CONFLICT_GAIN,
        color="#F97316",
        linestyle=":",
        linewidth=1.2,
        label=f"Conflict gate +{I2_MINIMUM_CONFLICT_GAIN:.3f}",
    )
    axis.set_xticks(positions, labels=labels)
    axis.set_ylabel("OOF macro-F1 change versus G3-C1")
    axis.set_title("I2 auxiliary supervision: overall and shortcut-slice transfer")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="best")
    span = max(0.01, max(abs(value) for value in all_values))
    axis.set_ylim(
        min(min(all_values), -I2_MAXIMUM_OVERALL_LOSS) - span * 0.20,
        max(max(all_values), I2_MINIMUM_CONFLICT_GAIN) + span * 0.28,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    figure.clear()


def build_i2_transfer_evidence(
    *,
    reference_manifest_path: str | Path = TASK2_EVIDENCE_DIR
    / "g3_c1_t1_smallcnn/manifest.json",
    i2_manifest_paths: tuple[str | Path, str | Path] = (
        TASK2_EVIDENCE_DIR / "g4_i2_article_type_lambda_0_1_c1/manifest.json",
        TASK2_EVIDENCE_DIR / "g4_i2_article_type_lambda_0_3_c1/manifest.json",
    ),
    reference_config_path: str | Path = ROOT / "configs/task2/g3_c1_t1_smallcnn.json",
    i2_config_paths: tuple[str | Path, str | Path] = (
        ROOT / "configs/task2/g4_i2_article_type_lambda_0_1_c1.json",
        ROOT / "configs/task2/g4_i2_article_type_lambda_0_3_c1.json",
    ),
    splits_path: str | Path = SPLITS_CSV,
    project_root: str | Path = ROOT,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR / "i2_multitask",
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Compare both I2 lambdas with G3-C1 and close the frozen transfer gate."""
    root = Path(project_root)
    splits = load_splits(splits_path)
    development = get_samples(splits, partition="development")
    expected = get_samples(development, target="season").reset_index(drop=True)
    expected_ids = expected["id"].astype(int)
    expected_targets = dict(
        zip(expected_ids, expected["season"].astype(str), strict=True)
    )
    mappings, assignments, fold_audit = build_article_type_shortcut_audit(splits)
    assignment_sha256 = canonical_sha256(
        assignments.loc[
            :,
            ["id", "fold", "season", "articleType", "shortcut_majority_season", "shortcut_slice"],
        ]
        .sort_values("id", kind="stable")
        .to_dict(orient="records")
    )

    i2_configs = [load_i2_config(path) for path in i2_config_paths]
    if tuple(config.experiment_id for config in i2_configs) != I2_EXPERIMENT_IDS:
        raise ValueError("I2 evidence requires the frozen lambda 0.1 then lambda 0.3 configs")
    reference_config = load_experiment_config(reference_config_path)
    if reference_config.experiment_id != I2_REFERENCE_EXPERIMENT_ID:
        raise ValueError("I2 reference config must be corrected G3-C1")
    reference_payload = reference_config.to_dict()
    for field in ("experiment_id", "stage", "loss_id"):
        reference_payload.pop(field)
    for config in i2_configs:
        candidate = config.base.to_dict()
        for field in ("experiment_id", "stage", "loss_id"):
            candidate.pop(field)
        if candidate != reference_payload:
            raise ValueError("I2 must preserve every G3-C1 field outside its declared changes")
    matched_protocol_sha256 = canonical_sha256(reference_payload)

    audits = [
        _load_experiment_audit(
            manifest_path=reference_manifest_path,
            config_path=reference_config_path,
            variant="G3-C1 reference",
            auxiliary_weight=0.0,
            project_root=root,
            expected_ids=expected_ids,
            expected_targets=expected_targets,
        )
    ]
    for manifest_path, config_path, config in zip(
        i2_manifest_paths, i2_config_paths, i2_configs, strict=True
    ):
        audits.append(
            _load_experiment_audit(
                manifest_path=manifest_path,
                config_path=config_path,
                variant=f"I2 lambda={config.auxiliary.loss_weight:.1f}",
                auxiliary_weight=config.auxiliary.loss_weight,
                project_root=root,
                expected_ids=expected_ids,
                expected_targets=expected_targets,
            )
        )
    shared_fields = ("split_sha256", "label_map_sha256", "transform_id")
    for field in shared_fields:
        values = {str(audit.registry[field].iloc[0]) for audit in audits}
        if len(values) != 1:
            raise ValueError(f"I2 comparison changed shared field {field}")

    slice_metrics = pd.concat(
        [_slice_metrics(audit, assignments) for audit in audits],
        ignore_index=True,
    )
    reference_slices = (
        slice_metrics.loc[
            slice_metrics["experiment_id"].eq(I2_REFERENCE_EXPERIMENT_ID),
            ["shortcut_slice", "macro_f1"],
        ]
        .set_index("shortcut_slice")["macro_f1"]
        .to_dict()
    )
    comparison_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    reference_macro = float(audits[0].metrics["macro_f1"])
    reference_spring = float(audits[0].metrics["per_class"]["Spring"]["f1"])
    for audit in audits:
        audit_slices = slice_metrics.loc[
            slice_metrics["experiment_id"].eq(audit.experiment_id)
        ].set_index("shortcut_slice")
        overall_macro = float(audit.metrics["macro_f1"])
        spring_f1 = float(audit.metrics["per_class"]["Spring"]["f1"])
        aligned_macro = float(audit_slices.loc["aligned", "macro_f1"])
        conflict_macro = float(audit_slices.loc["conflict", "macro_f1"])
        fold_scores = pd.to_numeric(audit.fold_metrics["macro_f1"], errors="raise")
        registry = audit.registry
        comparison_rows.append(
            {
                "variant": audit.variant,
                "experiment_id": audit.experiment_id,
                "auxiliary_weight": audit.auxiliary_weight,
                "pooled_macro_f1": overall_macro,
                "fold_mean_macro_f1": float(fold_scores.mean()),
                "fold_sd_macro_f1": float(fold_scores.std(ddof=1)),
                "spring_f1": spring_f1,
                "aligned_macro_f1": aligned_macro,
                "conflict_macro_f1": conflict_macro,
                "aligned_support": int(audit_slices.loc["aligned", "support"]),
                "conflict_support": int(audit_slices.loc["conflict", "support"]),
                "unseen_article_type_support": int(
                    audit_slices.loc["unseen_article_type", "support"]
                ),
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
                "delta_macro_f1_vs_reference": overall_macro - reference_macro,
                "delta_spring_f1_vs_reference": spring_f1 - reference_spring,
                "delta_aligned_macro_f1_vs_reference": (
                    aligned_macro - float(reference_slices["aligned"])
                ),
                "delta_conflict_macro_f1_vs_reference": (
                    conflict_macro - float(reference_slices["conflict"])
                ),
                "config_sha256": str(registry["config_sha256"].iloc[0]),
                "split_sha256": str(registry["split_sha256"].iloc[0]),
                "label_map_sha256": str(registry["label_map_sha256"].iloc[0]),
                "implementation_sha256": str(registry["implementation_sha256"].iloc[0]),
                "transform_id": str(registry["transform_id"].iloc[0]),
            }
        )
        for label in SEASON_LABELS:
            metrics = audit.metrics["per_class"][label]
            reference_f1 = float(audits[0].metrics["per_class"][label]["f1"])
            per_class_rows.append(
                {
                    "variant": audit.variant,
                    "experiment_id": audit.experiment_id,
                    "auxiliary_weight": audit.auxiliary_weight,
                    "label": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "support": metrics["support"],
                    "delta_f1_vs_reference": float(metrics["f1"]) - reference_f1,
                }
            )
    comparison, selected_experiment_id = apply_i2_selection_rule(
        pd.DataFrame(comparison_rows)
    )
    per_class = pd.DataFrame(per_class_rows)

    paired_fold_parts = []
    for audit in audits:
        part = audit.fold_metrics.loc[
            :, ["fold", "run_id", "macro_f1", "accuracy"]
        ].copy()
        slug = (
            "reference"
            if audit.auxiliary_weight == 0.0
            else f"lambda_{audit.auxiliary_weight:.1f}"
        )
        part = part.rename(
            columns={name: f"{slug}_{name}" for name in part if name != "fold"}
        )
        paired_fold_parts.append(part)
    paired_folds = paired_fold_parts[0]
    for part in paired_fold_parts[1:]:
        paired_folds = paired_folds.merge(part, on="fold", validate="one_to_one")
    learning_summary = _learning_curve_summary(audits[1:])
    registry_snapshot = pd.concat(
        [
            audit.registry.assign(
                variant=audit.variant,
                auxiliary_weight=audit.auxiliary_weight,
            )
            for audit in audits
        ],
        ignore_index=True,
    )
    registry_snapshot = registry_snapshot.loc[
        :, ["variant", "auxiliary_weight", *EXPERIMENT_REGISTRY_COLUMNS]
    ].sort_values(["auxiliary_weight", "fold", "run_id"])
    slice_summary = (
        assignments.groupby(["fold", "shortcut_slice"], observed=True)
        .size()
        .rename("support")
        .reset_index()
    )
    decision_candidates = comparison.loc[comparison["auxiliary_weight"].gt(0.0)]
    decision = {
        "schema_version": "1.0.0",
        "gate": "G4-I2",
        "decision_status": "closed",
        "primary_metric": "pooled_five_fold_oof_macro_f1",
        "reference_experiment_id": I2_REFERENCE_EXPERIMENT_ID,
        "candidate_criteria": {
            str(row["experiment_id"]): {
                "auxiliary_weight": float(row["auxiliary_weight"]),
                "observed_overall_delta": float(row["delta_macro_f1_vs_reference"]),
                "observed_conflict_delta": float(
                    row["delta_conflict_macro_f1_vs_reference"]
                ),
                "passes_overall_gain": bool(row["passes_overall_gain"]),
                "passes_conflict_gain_with_overall_floor": bool(
                    row["passes_conflict_gain"]
                ),
                "passes_i2_gate": bool(row["passes_i2_gate"]),
            }
            for row in decision_candidates.to_dict(orient="records")
        },
        "minimum_overall_gain": I2_MINIMUM_OVERALL_GAIN,
        "minimum_conflict_gain": I2_MINIMUM_CONFLICT_GAIN,
        "maximum_overall_loss": I2_MAXIMUM_OVERALL_LOSS,
        "selected_experiment_id": selected_experiment_id,
        "keep_i2": selected_experiment_id in I2_EXPERIMENT_IDS,
        "selection_priority": (
            "Among passing candidates, choose highest pooled Season macro-F1; exact ties "
            "choose the lower auxiliary weight."
        ),
        "shortcut_mapping_scope": "four training folds only for each validation fold",
        "shortcut_assignment_sha256": assignment_sha256,
        "loss_values_comparable_to_reference": False,
        "ultimate_winner_frozen": False,
        "limitations": [
            "This gate uses the primary seed only.",
            "ArticleType aligned/conflict is one declared shortcut slice, not proof of causality.",
            "Current softmax probabilities are not calibrated claims.",
        ],
        "next_question": (
            "Run the matched pretrained benchmark boundary and the second-seed stability gate, "
            "then analyse robustness, cost, calibration, uncertainty, and failures."
        ),
    }

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    common_root = Path(os.path.commonpath([evidence_root.resolve(), figure_root.resolve()]))
    paths = {
        "comparison": evidence_root / "comparison.csv",
        "paired_fold_metrics": evidence_root / "paired_fold_metrics.csv",
        "per_class_comparison": evidence_root / "per_class_comparison.csv",
        "slice_metrics": evidence_root / "slice_metrics.csv",
        "slice_summary": evidence_root / "slice_summary.csv",
        "article_type_majorities_by_fold": evidence_root
        / "article_type_majorities_by_fold.csv",
        "fold_shortcut_audit": evidence_root / "fold_shortcut_audit.csv",
        "learning_curve_summary": evidence_root / "learning_curve_summary.csv",
        "registry_snapshot": evidence_root / "registry_snapshot.csv",
        "decision": evidence_root / "decision.json",
        "learning_curves": figure_root / "i2_multitask_learning_curves.png",
        "transfer_deltas": figure_root / "i2_multitask_transfer_deltas.png",
    }
    atomic_write_csv(paths["comparison"], comparison)
    atomic_write_csv(paths["paired_fold_metrics"], paired_folds)
    atomic_write_csv(paths["per_class_comparison"], per_class)
    atomic_write_csv(paths["slice_metrics"], slice_metrics)
    atomic_write_csv(paths["slice_summary"], slice_summary)
    atomic_write_csv(paths["article_type_majorities_by_fold"], mappings)
    atomic_write_csv(paths["fold_shortcut_audit"], fold_audit)
    atomic_write_csv(paths["learning_curve_summary"], learning_summary)
    atomic_write_csv(paths["registry_snapshot"], registry_snapshot)
    atomic_write_json(paths["decision"], decision)
    plot_i2_learning_curves(learning_summary, paths["learning_curves"])
    _plot_transfer_deltas(comparison, paths["transfer_deltas"])

    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=common_root),
            "sha256": compute_sha256(path),
        }
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G4-I2",
        "decision_status": "closed",
        "coverage_sha256": canonical_sha256(
            {
                "row_count": len(expected_ids),
                "id_set_sha256": canonical_sha256(sorted(expected_ids.tolist())),
                "labels": list(SEASON_LABELS),
            }
        ),
        "matched_protocol_sha256": matched_protocol_sha256,
        "shortcut_assignment_sha256": assignment_sha256,
        "selected_experiment_id": selected_experiment_id,
        "keep_i2": selected_experiment_id in I2_EXPERIMENT_IDS,
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
        "split_input": {
            "path": _portable_artifact_path(Path(splits_path), fallback_root=root),
            "sha256": compute_sha256(splits_path),
        },
        "artifacts": artifact_manifest,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest
