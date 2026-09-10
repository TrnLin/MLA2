"""Hash-verified overview of every Task 2 development OOF experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from fashion.config import (
    ROOT,
    TASK2_EVIDENCE_DIR,
    TASK2_FIGURE_DIR,
    TASK2_SELECTION_FREEZE_JSON,
)
from fashion.data.hashing import compute_sha256
from fashion.train.artifacts import atomic_write_csv, atomic_write_json

EXPECTED_DEVELOPMENT_ROWS: Final = 32_753
EXPECTED_FOLDS: Final = (0, 1, 2, 3, 4)
EXPECTED_LABELS: Final = ("Fall", "Spring", "Summer", "Winter")


@dataclass(frozen=True)
class DevelopmentModelSpec:
    """One expected experiment in the chronological Task 2 development record."""

    evidence_folder: str
    experiment_id: str
    display_name: str
    phase: str
    comparison_role: str
    expected_seed: int = 2753


DEVELOPMENT_MODEL_SPECS: Final = (
    DevelopmentModelSpec(
        "b0_majority",
        "b0-majority",
        "B0 Majority",
        "Baselines",
        "Prior-only floor",
    ),
    DevelopmentModelSpec(
        "b1_hog_hsv_svm",
        "b1-hog-hsv-svm",
        "B1 HOG + HSV SVM",
        "Baselines",
        "Classical baseline",
    ),
    DevelopmentModelSpec(
        "g1_c1_smallcnn",
        "g1-c1-smallcnn",
        "G1 C1 SmallCNN (T0)",
        "Family screen",
        "Eight-epoch screen",
    ),
    DevelopmentModelSpec(
        "g1_c2_resnet18",
        "g1-c2-resnet18",
        "G1 C2 ResNet18 (T0)",
        "Family screen",
        "Eight-epoch screen",
    ),
    DevelopmentModelSpec(
        "g1_c3_mobilenetv3",
        "g1-c3-mobilenetv3",
        "G1 C3 MobileNetV3",
        "Family screen",
        "Efficiency alternative",
    ),
    DevelopmentModelSpec(
        "g2_p1_c2_resnet18",
        "g2-p1-c2-resnet18",
        "G2 C2 P1 input size",
        "Controlled ablations",
        "Input-size ablation",
    ),
    DevelopmentModelSpec(
        "g2_a1_c2_resnet18",
        "g2-a1-c2-resnet18",
        "G2 C2 A1 augmentation",
        "Controlled ablations",
        "Augmentation ablation",
    ),
    DevelopmentModelSpec(
        "g2_t1_c1_smallcnn",
        "g2-t1-c1-smallcnn",
        "G2 C1 T1",
        "Compact tuning",
        "Learning-rate tuning",
    ),
    DevelopmentModelSpec(
        "g2_t1_c2_resnet18",
        "g2-t1-c2-resnet18",
        "G2 C2 T1",
        "Compact tuning",
        "Learning-rate tuning",
    ),
    DevelopmentModelSpec(
        "g2_t2_c1_smallcnn",
        "g2-t2-c1-smallcnn",
        "G2 C1 T2",
        "Compact tuning",
        "Weight-decay tuning",
    ),
    DevelopmentModelSpec(
        "g2_t2_c2_resnet18",
        "g2-t2-c2-resnet18",
        "G2 C2 T2",
        "Compact tuning",
        "Weight-decay tuning",
    ),
    DevelopmentModelSpec(
        "g3_c1_t1_smallcnn",
        "g3-c1-t1-smallcnn",
        "G3 C1-T1 full budget",
        "Full-budget finalists",
        "Eligible finalist",
    ),
    DevelopmentModelSpec(
        "g3_c2_t0_resnet18",
        "g3-c2-t0-resnet18",
        "G3 C2-T0 full budget",
        "Full-budget finalists",
        "Eligible finalist",
    ),
    DevelopmentModelSpec(
        "g4_i1_effective_number_c1",
        "g4-i1-effective-number-c1",
        "G4 I1 class-balanced C1",
        "Targeted interventions",
        "Class-balance intervention",
    ),
    DevelopmentModelSpec(
        "g4_i2_article_type_lambda_0_1_c1",
        "g4-i2-article-type-lambda-0-1-c1",
        "G4 I2 multi-task (lambda=0.1)",
        "Targeted interventions",
        "Auxiliary-task ablation",
    ),
    DevelopmentModelSpec(
        "g4_i2_article_type_lambda_0_3_c1",
        "g4-i2-article-type-lambda-0-3-c1",
        "G4 I2 multi-task (lambda=0.3)",
        "Targeted interventions",
        "Selected eligible model",
    ),
    DevelopmentModelSpec(
        "g4_p0s_resnet18_standard_scratch",
        "g4-p0s-resnet18-standard-scratch",
        "G4 P0S standard-stem scratch",
        "Pretraining benchmark",
        "Matched scratch control",
    ),
    DevelopmentModelSpec(
        "g4_pstar_resnet18_standard_pretrained",
        "g4-pstar-resnet18-standard-pretrained",
        "G4 P* pretrained [benchmark only]",
        "Pretraining benchmark",
        "Final-ineligible benchmark",
    ),
    DevelopmentModelSpec(
        "g5_c2_t0_resnet18_s2026",
        "g5-c2-t0-resnet18-s2026",
        "G5 C2-T0 [seed 2026]",
        "Seed stability",
        "Second-seed check",
        expected_seed=2026,
    ),
    DevelopmentModelSpec(
        "g5_i2_article_type_lambda_0_3_c1_s2026",
        "g5-i2-article-type-lambda-0-3-c1-s2026",
        "G5 I2 lambda=0.3 [seed 2026]",
        "Seed stability",
        "Second-seed check",
        expected_seed=2026,
    ),
)


def _resolve(path: str | Path, *, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _portable(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _discover_experiment_ids(project_root: Path) -> set[str]:
    evidence_root = project_root / "results/evidence/task2"
    discovered: set[str] = set()
    for directory in evidence_root.iterdir():
        manifest_path = directory / "manifest.json"
        if not (
            directory.is_dir()
            and manifest_path.is_file()
            and (directory / "pooled_metrics.json").is_file()
            and (directory / "registry_snapshot.csv").is_file()
        ):
            continue
        with manifest_path.open(encoding="utf-8") as handle:
            experiment_id = str(json.load(handle).get("experiment_id", ""))
        if not experiment_id or experiment_id in discovered:
            raise ValueError("development evidence has a missing or duplicate experiment ID")
        discovered.add(experiment_id)
    return discovered


def _verified_artifact(
    manifest: dict[str, Any],
    name: str,
    *,
    project_root: Path,
) -> Path:
    declaration = manifest.get("artifacts", {}).get(name)
    if not isinstance(declaration, dict):
        raise ValueError(f"experiment manifest is missing required artifact: {name}")
    artifact = _resolve(declaration.get("path", ""), project_root=project_root)
    if not artifact.is_file():
        raise ValueError(f"declared {name} artifact does not exist: {artifact}")
    if compute_sha256(artifact) != declaration.get("sha256"):
        raise ValueError(f"declared {name} artifact hash does not match its bytes")
    return artifact


def _one_registry_value(registry: pd.DataFrame, column: str, experiment_id: str) -> str:
    if column not in registry:
        raise ValueError(f"{experiment_id} registry is missing {column}")
    values = set(registry[column].astype(str))
    if len(values) != 1:
        raise ValueError(f"{experiment_id} changed {column} across folds")
    return values.pop()


def _bool_value(value: str, *, field: str, experiment_id: str) -> bool:
    normalised = value.strip().lower()
    if normalised not in {"true", "false"}:
        raise ValueError(f"{experiment_id} registry {field} is not boolean")
    return normalised == "true"


def _load_experiment_row(
    spec: DevelopmentModelSpec,
    *,
    workflow_order: int,
    project_root: Path,
    selected_experiment_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = (
        project_root / "results/evidence/task2" / spec.evidence_folder / "manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError(f"missing development experiment manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("experiment_id") != spec.experiment_id:
        raise ValueError(f"unexpected experiment ID in {manifest_path}")
    if int(manifest.get("seed", -1)) != spec.expected_seed:
        raise ValueError(f"{spec.experiment_id} has an unexpected seed")
    if tuple(manifest.get("folds", ())) != EXPECTED_FOLDS:
        raise ValueError(f"{spec.experiment_id} must contain canonical folds 0-4")

    coverage = manifest.get("coverage", {})
    counts = {
        int(coverage.get("row_count", -1)),
        int(coverage.get("unique_id_count", -1)),
        int(coverage.get("expected_row_count", -1)),
    }
    if counts != {EXPECTED_DEVELOPMENT_ROWS}:
        raise ValueError(f"{spec.experiment_id} has incomplete development OOF coverage")
    if int(coverage.get("protected_id_count", -1)) != 0:
        raise ValueError(f"{spec.experiment_id} includes protected IDs")
    if tuple(coverage.get("labels", ())) != EXPECTED_LABELS:
        raise ValueError(f"{spec.experiment_id} changed the Season label order")
    coverage_id_set_sha256 = str(coverage.get("id_set_sha256", ""))
    if len(coverage_id_set_sha256) != 64:
        raise ValueError(f"{spec.experiment_id} has an invalid OOF ID-set hash")

    pooled_path = _verified_artifact(manifest, "pooled_metrics", project_root=project_root)
    registry_path = _verified_artifact(
        manifest, "registry_snapshot", project_root=project_root
    )
    _verified_artifact(manifest, "fold_summary", project_root=project_root)
    with pooled_path.open(encoding="utf-8") as handle:
        pooled = json.load(handle)
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    run_ids = {str(run_id) for run_id in manifest.get("run_ids", ())}
    if len(registry) != 5 or set(registry["run_id"]) != run_ids:
        raise ValueError(f"{spec.experiment_id} registry must match five run IDs")
    if set(pd.to_numeric(registry["fold"], errors="raise").astype(int)) != set(
        EXPECTED_FOLDS
    ):
        raise ValueError(f"{spec.experiment_id} registry has invalid folds")
    for column, expected in (
        ("experiment_id", spec.experiment_id),
        ("seed", str(spec.expected_seed)),
        ("status", "completed"),
    ):
        if _one_registry_value(registry, column, spec.experiment_id).lower() != expected:
            raise ValueError(f"{spec.experiment_id} registry {column} must be {expected}")

    metric_names = (
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "nll",
        "brier",
        "ece",
    )
    metrics = {name: float(pooled[name]) for name in metric_names}
    for name, value in metrics.items():
        if not np.isfinite(value):
            raise ValueError(f"{spec.experiment_id} has non-finite {name}")
    for name in metric_names[:6] + ("brier", "ece"):
        if not 0.0 <= metrics[name] <= 1.0:
            raise ValueError(f"{spec.experiment_id} has invalid {name}")
    if metrics["nll"] < 0.0:
        raise ValueError(f"{spec.experiment_id} has invalid nll")
    if int(pooled.get("n_samples", -1)) != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError(f"{spec.experiment_id} pooled metrics have invalid n_samples")
    if tuple(pooled.get("labels", ())) != EXPECTED_LABELS:
        raise ValueError(f"{spec.experiment_id} pooled metrics changed label order")
    if not np.isclose(metrics["macro_f1"], float(manifest["pooled_macro_f1"])):
        raise ValueError(f"{spec.experiment_id} pooled macro-F1 disagrees with manifest")

    benchmark_only = _bool_value(
        _one_registry_value(registry, "benchmark_only", spec.experiment_id),
        field="benchmark_only",
        experiment_id=spec.experiment_id,
    )
    final_eligible = _bool_value(
        _one_registry_value(registry, "final_eligible", spec.experiment_id),
        field="final_eligible",
        experiment_id=spec.experiment_id,
    )
    scratch = _bool_value(
        _one_registry_value(registry, "scratch", spec.experiment_id),
        field="scratch",
        experiment_id=spec.experiment_id,
    )
    if benchmark_only != (spec.phase == "Pretraining benchmark"):
        raise ValueError(f"{spec.experiment_id} has an unexpected benchmark boundary")
    if benchmark_only and final_eligible:
        raise ValueError(f"{spec.experiment_id} cannot be benchmark-only and final-eligible")
    selected_for_freeze = spec.experiment_id == selected_experiment_id
    if selected_for_freeze and not (scratch and final_eligible and not benchmark_only):
        raise ValueError(f"{spec.experiment_id} is not an eligible scratch selection")
    split_sha256 = _one_registry_value(registry, "split_sha256", spec.experiment_id)
    label_map_sha256 = _one_registry_value(registry, "label_map_sha256", spec.experiment_id)
    if len(split_sha256) != 64 or len(label_map_sha256) != 64:
        raise ValueError(f"{spec.experiment_id} has invalid canonical-input hashes")

    row: dict[str, Any] = {
        "workflow_order": workflow_order,
        "display_name": spec.display_name,
        "experiment_id": spec.experiment_id,
        "phase": spec.phase,
        "comparison_role": spec.comparison_role,
        "seed": spec.expected_seed,
        "model_family": _one_registry_value(registry, "model_family", spec.experiment_id),
        "loss_id": _one_registry_value(registry, "loss_id", spec.experiment_id),
        "parameter_count": int(
            _one_registry_value(registry, "parameter_count", spec.experiment_id)
        ),
        "scratch": scratch,
        "benchmark_only": benchmark_only,
        "final_eligible": final_eligible,
        "selected_for_freeze": selected_for_freeze,
        "n_samples": EXPECTED_DEVELOPMENT_ROWS,
        "coverage_id_set_sha256": coverage_id_set_sha256,
        "split_sha256": split_sha256,
        "label_map_sha256": label_map_sha256,
        **metrics,
        "manifest_path": _portable(manifest_path, project_root=project_root),
        "manifest_sha256": compute_sha256(manifest_path),
        "pooled_metrics_path": _portable(pooled_path, project_root=project_root),
        "pooled_metrics_sha256": compute_sha256(pooled_path),
    }
    input_declaration = {
        "path": row["manifest_path"],
        "sha256": row["manifest_sha256"],
    }
    return row, input_declaration


def plot_development_model_comparison(catalog: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot all OOF configurations without implying they are holdout scores."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked = catalog.sort_values(["macro_f1", "workflow_order"], ascending=[True, True])
    y = np.arange(len(ranked))
    palette = {
        "Baselines": "#6B7280",
        "Family screen": "#4C78A8",
        "Controlled ablations": "#F58518",
        "Compact tuning": "#ECA82C",
        "Full-budget finalists": "#72B7B2",
        "Targeted interventions": "#54A24B",
        "Pretraining benchmark": "#B279A2",
        "Seed stability": "#E45756",
    }
    colours = [palette[phase] for phase in ranked["phase"]]

    fig, (rank_axis, metric_axis) = plt.subplots(
        1,
        2,
        figsize=(18, 12),
        sharey=True,
        gridspec_kw={"width_ratios": (1.05, 1.0), "wspace": 0.04},
    )
    bars = rank_axis.barh(y, ranked["macro_f1"], color=colours, alpha=0.88)
    for bar, (_, row) in zip(bars, ranked.iterrows(), strict=True):
        if bool(row["benchmark_only"]):
            bar.set_hatch("//")
            bar.set_alpha(0.6)
        if bool(row["selected_for_freeze"]):
            bar.set_edgecolor("#111827")
            bar.set_linewidth(2.4)
        rank_axis.text(
            min(float(row["macro_f1"]) + 0.008, 0.785),
            bar.get_y() + bar.get_height() / 2,
            f"{float(row['macro_f1']):.3f}",
            va="center",
            fontsize=8.5,
        )
    rank_axis.set_yticks(y, ranked["display_name"])
    rank_axis.set_xlim(0.0, 0.81)
    rank_axis.set_xlabel("Pooled five-fold OOF macro-F1")
    rank_axis.set_title("A. Macro-F1 ranking", loc="left", fontweight="bold")
    rank_axis.grid(axis="x", alpha=0.22)

    metric_specs = (
        ("macro_f1", "Macro-F1", "o", "#111827"),
        ("accuracy", "Accuracy", "s", "#2563EB"),
        ("balanced_accuracy", "Balanced accuracy", "^", "#D97706"),
    )
    row_min = ranked[[item[0] for item in metric_specs]].min(axis=1)
    row_max = ranked[[item[0] for item in metric_specs]].max(axis=1)
    metric_axis.hlines(y, row_min, row_max, color="#D1D5DB", linewidth=1.2, zorder=1)
    for column, label, marker, colour in metric_specs:
        metric_axis.scatter(
            ranked[column],
            y,
            label=label,
            marker=marker,
            color=colour,
            s=46,
            zorder=2,
        )
    metric_axis.set_xlim(0.0, 0.81)
    metric_axis.set_xlabel("Score (higher is better)")
    metric_axis.set_title("B. Three performance views", loc="left", fontweight="bold")
    metric_axis.grid(axis="x", alpha=0.22)
    metric_axis.tick_params(axis="y", labelleft=False)
    metric_axis.legend(loc="lower right", frameon=True)

    phase_handles = [Patch(facecolor=colour, label=phase) for phase, colour in palette.items()]
    fig.legend(
        handles=phase_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Task 2: all 20 registered development OOF configurations",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.952,
        (
            "Each row uses the same 32,753 development IDs and canonical folds. "
            "These are not internal-holdout scores."
        ),
        ha="center",
        fontsize=10,
        color="#374151",
    )
    fig.subplots_adjust(left=0.29, right=0.98, top=0.92, bottom=0.10)
    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "MLA2 Task 2 development comparison"},
    )
    plt.close(fig)
    return output


def build_development_model_comparison_evidence(
    *,
    project_root: str | Path = ROOT,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR / "development_model_comparison",
    figure_path: str | Path = TASK2_FIGURE_DIR / "development_model_comparison.png",
    selection_freeze_path: str | Path = TASK2_SELECTION_FREEZE_JSON,
) -> dict[str, Any]:
    """Build a complete, hash-linked development comparison table and figure."""
    root = Path(project_root)
    from fashion.task2.ultimate_judgement import load_verified_selection_freeze

    selection_freeze, resolved_selection_freeze = load_verified_selection_freeze(
        selection_freeze_path,
        project_root=root,
    )
    selected_experiment_id = str(selection_freeze["selected_model"]["experiment_id"])
    expected_ids = {spec.experiment_id for spec in DEVELOPMENT_MODEL_SPECS}
    if selected_experiment_id not in expected_ids:
        raise ValueError("selection freeze chose an experiment outside the comparison catalog")
    discovered_ids = _discover_experiment_ids(root)
    if discovered_ids != expected_ids:
        missing = sorted(discovered_ids - expected_ids)
        stale = sorted(expected_ids - discovered_ids)
        raise ValueError(
            f"development comparison inventory mismatch; unlisted={missing}, missing={stale}"
        )
    rows: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, str]] = {}
    for order, spec in enumerate(DEVELOPMENT_MODEL_SPECS, start=1):
        row, declaration = _load_experiment_row(
            spec,
            workflow_order=order,
            project_root=root,
            selected_experiment_id=selected_experiment_id,
        )
        rows.append(row)
        inputs[spec.experiment_id] = declaration
    catalog = pd.DataFrame(rows)
    if catalog["experiment_id"].duplicated().any():
        raise ValueError("development comparison contains duplicate experiment IDs")
    if len(set(catalog["n_samples"])) != 1:
        raise ValueError("development comparison does not use common OOF coverage")
    if catalog["coverage_id_set_sha256"].nunique() != 1:
        raise ValueError("development comparison does not use the same OOF IDs")
    if catalog["split_sha256"].nunique() != 1:
        raise ValueError("development comparison does not use one canonical split")
    if catalog["label_map_sha256"].nunique() != 1:
        raise ValueError("development comparison does not use one label map")

    evidence_root = Path(evidence_directory)
    catalog_path = evidence_root / "all_model_oof_metrics.csv"
    figure = plot_development_model_comparison(catalog, figure_path)
    atomic_write_csv(catalog_path, catalog)
    artifacts = {
        "catalog": {
            "path": _portable(catalog_path, project_root=root),
            "sha256": compute_sha256(catalog_path),
        },
        "figure": {
            "path": _portable(figure, project_root=root),
            "sha256": compute_sha256(figure),
        },
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "Task2-development-model-comparison",
        "status": "complete",
        "source_partition": "development_oof",
        "model_count": len(catalog),
        "expected_row_count_per_experiment": EXPECTED_DEVELOPMENT_ROWS,
        "metric_columns": ["macro_f1", "accuracy", "balanced_accuracy"],
        "canonical_inputs": {
            "coverage_id_set_sha256": str(catalog["coverage_id_set_sha256"].iloc[0]),
            "split_sha256": str(catalog["split_sha256"].iloc[0]),
            "label_map_sha256": str(catalog["label_map_sha256"].iloc[0]),
        },
        "selection_freeze": {
            "path": _portable(resolved_selection_freeze, project_root=root),
            "sha256": compute_sha256(resolved_selection_freeze),
            "selected_experiment_id": selected_experiment_id,
        },
        "claim_boundary": (
            "This inventory compares pooled five-fold development OOF artifacts. "
            "It does not create, replace, or compare internal-holdout predictions; "
            "stage-specific causal claims still require their matched gate evidence."
        ),
        "input_manifests": inputs,
        "artifacts": artifacts,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


def load_verified_development_model_comparison(
    manifest_path: str | Path = TASK2_EVIDENCE_DIR / "development_model_comparison/manifest.json",
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load the complete comparison only after verifying outputs and upstream manifests."""
    root = Path(project_root)
    resolved_manifest = _resolve(manifest_path, project_root=root)
    with resolved_manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("gate") != "Task2-development-model-comparison":
        raise ValueError("unexpected Task 2 development comparison gate")
    if manifest.get("status") != "complete":
        raise ValueError("Task 2 development comparison is not complete")
    if int(manifest.get("model_count", -1)) != len(DEVELOPMENT_MODEL_SPECS):
        raise ValueError("Task 2 development comparison has an incomplete model count")

    from fashion.task2.ultimate_judgement import load_verified_selection_freeze

    freeze_declaration = manifest.get("selection_freeze", {})
    freeze_path = _resolve(freeze_declaration.get("path", ""), project_root=root)
    if not freeze_path.is_file() or compute_sha256(freeze_path) != freeze_declaration.get(
        "sha256"
    ):
        raise ValueError("development comparison selection freeze changed")
    selection_freeze, _ = load_verified_selection_freeze(freeze_path, project_root=root)
    selected_experiment_id = str(selection_freeze["selected_model"]["experiment_id"])
    if freeze_declaration.get("selected_experiment_id") != selected_experiment_id:
        raise ValueError("development comparison disagrees with the selection freeze")

    input_manifests = manifest.get("input_manifests", {})
    expected_ids = [spec.experiment_id for spec in DEVELOPMENT_MODEL_SPECS]
    if len(input_manifests) != len(expected_ids) or set(input_manifests) != set(expected_ids):
        raise ValueError("development comparison has incomplete upstream manifests")
    if _discover_experiment_ids(root) != set(expected_ids):
        raise ValueError("development comparison no longer covers every experiment artifact")
    for experiment_id, declaration in input_manifests.items():
        upstream = _resolve(declaration.get("path", ""), project_root=root)
        if not upstream.is_file() or compute_sha256(upstream) != declaration.get("sha256"):
            raise ValueError(f"upstream manifest changed for {experiment_id}")
    catalog_path = _verified_artifact(manifest, "catalog", project_root=root)
    _verified_artifact(manifest, "figure", project_root=root)
    catalog = pd.read_csv(catalog_path)
    for column in ("scratch", "benchmark_only", "final_eligible", "selected_for_freeze"):
        normalised = catalog[column].map(lambda value: str(value).strip().lower())
        if not normalised.isin({"true", "false"}).all():
            raise ValueError(f"development comparison catalog has invalid {column}")
        catalog[column] = normalised.eq("true")
    if catalog["experiment_id"].tolist() != expected_ids:
        raise ValueError("development comparison catalog is incomplete or out of order")
    if set(catalog["n_samples"].astype(int)) != {EXPECTED_DEVELOPMENT_ROWS}:
        raise ValueError("development comparison catalog has inconsistent coverage")
    if catalog["coverage_id_set_sha256"].nunique() != 1:
        raise ValueError("development comparison catalog has inconsistent OOF IDs")
    canonical_inputs = manifest.get("canonical_inputs", {})
    for column in ("coverage_id_set_sha256", "split_sha256", "label_map_sha256"):
        if catalog[column].nunique() != 1:
            raise ValueError(f"development comparison catalog has inconsistent {column}")
        if str(catalog[column].iloc[0]) != canonical_inputs.get(column):
            raise ValueError(f"development comparison manifest changed {column}")
    selected_ids = catalog.loc[catalog["selected_for_freeze"], "experiment_id"].tolist()
    if selected_ids != [selected_experiment_id]:
        raise ValueError("development comparison selected row disagrees with the freeze")
    return manifest, catalog
