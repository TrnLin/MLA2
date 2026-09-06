"""Hash-linked G6 shortcut-slice and error evidence for frozen Task 2 OOF packs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection, Mapping

import numpy as np
import pandas as pd

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import (
    EXPERIMENT_REGISTRY_COLUMNS,
    _portable_artifact_path,
    _resolve_evidence_path,
)
from fashion.task2.experiments import load_experiment_config
from fashion.task2.multitask import load_i2_config
from fashion.task2.slices import (
    CandidateOOFPack,
    SliceAnalysisSpec,
    SliceAnalysisTables,
    analyse_slice_packs,
    build_slice_assignments,
    load_slice_analysis_spec,
    plot_slice_macro_f1,
    plot_spring_destinations,
)
from fashion.task2.stability import (
    C2_PRIMARY_EXPERIMENT_ID,
    C2_STABILITY_EXPERIMENT_ID,
    I2_PRIMARY_EXPERIMENT_ID,
    I2_STABILITY_EXPERIMENT_ID,
    load_stability_i2_config,
)
from fashion.task2.stability_evidence import EXPECTED_EXPERIMENTS
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.metrics import (
    SEASON_LABELS,
    multiclass_metrics,
    validate_oof,
    validate_oof_identity,
)

DEFAULT_ANALYSIS_CONFIG = Path("configs/task2/g6_shortcut_error_slices.json")
DEFAULT_STABILITY_MANIFEST = Path("results/evidence/task2/seed_stability/manifest.json")
DEFAULT_EVIDENCE_DIRECTORY = TASK2_EVIDENCE_DIR / "shortcut_error_slices"


def _verify_declared_files(
    declarations: Mapping[str, Any],
    *,
    project_root: Path,
    section: str,
) -> dict[str, Path]:
    if not isinstance(declarations, Mapping) or not declarations:
        raise ValueError(f"stability manifest has no {section}")
    resolved: dict[str, Path] = {}
    for name, declaration in declarations.items():
        if not isinstance(declaration, Mapping):
            raise ValueError(f"stability {section}.{name} is not an artifact declaration")
        path = _resolve_evidence_path(str(declaration.get("path", "")), project_root=project_root)
        digest = str(declaration.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError(f"stability {section}.{name} has an invalid SHA-256")
        verify_artifact(path, digest)
        resolved[str(name)] = path
    return resolved


def load_verified_stability_manifest(
    path: str | Path = DEFAULT_STABILITY_MANIFEST,
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path, dict[str, dict[str, Path]]]:
    """Verify the complete G5 evidence boundary before consuming its OOF registry."""
    root = Path(project_root)
    resolved_manifest = _resolve_evidence_path(path, project_root=root)
    if not resolved_manifest.is_file():
        raise ValueError(f"seed-stability manifest does not exist: {resolved_manifest}")
    with resolved_manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required_identity = {
        "schema_version": "1.0.0",
        "gate": "G5-SEED",
        "decision_status": "closed",
        "ordering_stable": True,
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
    }
    mismatches = [
        name for name, expected in required_identity.items() if manifest.get(name) != expected
    ]
    if mismatches:
        raise ValueError(f"seed-stability manifest boundary changed: {mismatches}")

    expected_ids = set(EXPECTED_EXPERIMENTS)
    if set(manifest.get("input_configs", {})) != expected_ids:
        raise ValueError("seed-stability manifest changed the frozen config set")
    if set(manifest.get("input_manifests", {})) != expected_ids:
        raise ValueError("seed-stability manifest changed the frozen experiment manifests")
    resolved_sections = {
        section: _verify_declared_files(
            manifest.get(section, {}),
            project_root=root,
            section=section,
        )
        for section in ("artifacts", "input_configs", "input_manifests")
    }
    required_artifacts = {"registry_snapshot", "seed_stability", "decision"}
    missing = required_artifacts - set(resolved_sections["artifacts"])
    if missing:
        raise ValueError(f"seed-stability manifest lacks G6 inputs: {sorted(missing)}")
    return manifest, resolved_manifest, resolved_sections


def _normalised_boolean_values(frame: pd.DataFrame, column: str) -> set[str]:
    return set(frame[column].astype(str).str.strip().str.lower())


def load_declared_config_hashes(
    config_paths: Mapping[str, str | Path],
) -> dict[str, str]:
    """Hash loader-normalised configs exactly as the run registry does."""
    if set(config_paths) != set(EXPECTED_EXPERIMENTS):
        raise ValueError("config hash audit requires the exact frozen experiment set")
    hashes: dict[str, str] = {}
    for experiment_id, path in config_paths.items():
        if experiment_id in {C2_PRIMARY_EXPERIMENT_ID, C2_STABILITY_EXPERIMENT_ID}:
            config = load_experiment_config(path)
        elif experiment_id == I2_PRIMARY_EXPERIMENT_ID:
            config = load_i2_config(path)
        elif experiment_id == I2_STABILITY_EXPERIMENT_ID:
            config = load_stability_i2_config(path)
        else:  # pragma: no cover - exact set check above keeps this defensive.
            raise ValueError(f"unexpected G6 experiment config: {experiment_id}")
        if config.experiment_id != experiment_id:
            raise ValueError(f"declared config identity differs for {experiment_id}")
        hashes[experiment_id] = canonical_sha256(config.to_dict())
    return hashes


def load_candidate_oof_packs(
    registry: pd.DataFrame,
    spec: SliceAnalysisSpec,
    *,
    project_root: str | Path,
    expected_ids: Collection[Any],
    expected_targets: Mapping[Any, str],
    protected_ids: Collection[Any],
    split_sha256: str,
    label_map_sha256: str,
    config_sha256_by_experiment: Mapping[str, str],
    stability_summary: pd.DataFrame | None = None,
) -> tuple[list[CandidateOOFPack], dict[str, Any], dict[str, dict[str, str]]]:
    """Load and independently validate four complete five-fold OOF packs."""
    root = Path(project_root)
    missing = sorted(set(EXPERIMENT_REGISTRY_COLUMNS) - set(registry))
    if missing:
        raise ValueError(f"seed-stability registry is missing columns: {missing}")
    if len(registry) != 20 or registry["run_id"].nunique() != 20:
        raise ValueError("slice analysis requires exactly 20 unique frozen registry rows")
    if set(registry["experiment_id"]) != {candidate.experiment_id for candidate in spec.candidates}:
        raise ValueError("slice registry changed the frozen experiment set")
    if set(registry["split_sha256"]) != {split_sha256}:
        raise ValueError("slice registry split hash differs from canonical splits.csv")
    if set(registry["label_map_sha256"]) != {label_map_sha256}:
        raise ValueError("slice registry label-map hash differs from canonical label_maps.json")
    if registry["transform_id"].nunique() != 1:
        raise ValueError("slice candidates do not share the same image transform")

    expected_id_values = list(expected_ids)
    packs: list[CandidateOOFPack] = []
    coverage_by_experiment: dict[str, Any] = {}
    prediction_artifacts: dict[str, dict[str, str]] = {}
    for candidate in spec.candidates:
        expected = EXPECTED_EXPERIMENTS[candidate.experiment_id]
        rows = registry.loc[
            registry["experiment_id"].eq(candidate.experiment_id)
            & pd.to_numeric(registry["seed"], errors="raise").astype(int).eq(candidate.seed)
        ].copy()
        if len(rows) != 5 or set(pd.to_numeric(rows["fold"], errors="raise").astype(int)) != set(
            range(5)
        ):
            raise ValueError(f"{candidate.experiment_id} does not contain folds 0-4")
        required_values = {
            "candidate": candidate.candidate,
            "stage": expected["stage"],
            "model_family": expected["model_family"],
            "loss_id": expected["loss_id"],
            "primary_metric_name": "macro_f1",
            "status": "completed",
        }
        for column, value in required_values.items():
            if set(rows[column].astype(str)) != {str(value)}:
                raise ValueError(f"{candidate.experiment_id} registry mismatch for {column}")
        for column, value in {
            "benchmark_only": "false",
            "final_eligible": "true",
            "scratch": "true",
            "git_dirty": "false",
        }.items():
            if _normalised_boolean_values(rows, column) != {value}:
                raise ValueError(f"{candidate.experiment_id} registry mismatch for {column}")
        expected_config_sha256 = config_sha256_by_experiment.get(candidate.experiment_id)
        if set(rows["config_sha256"]) != {expected_config_sha256}:
            raise ValueError(f"{candidate.experiment_id} registry config hash changed")
        for field in (
            "implementation_sha256",
            "parameter_count",
            "split_sha256",
            "label_map_sha256",
            "transform_id",
        ):
            if rows[field].nunique() != 1:
                raise ValueError(f"{candidate.experiment_id} changed {field} across folds")

        fold_frames = []
        for row in rows.sort_values("fold", kind="stable").to_dict(orient="records"):
            prediction_path = _resolve_evidence_path(
                str(row["prediction_path"]),
                project_root=root,
            )
            prediction_sha256 = str(row["prediction_sha256"])
            verify_artifact(prediction_path, prediction_sha256)
            prediction_artifacts[str(row["run_id"])] = {
                "path": _portable_artifact_path(prediction_path, fallback_root=root),
                "sha256": prediction_sha256,
            }
            fold_frame = pd.read_csv(prediction_path)
            validate_oof_identity(
                fold_frame,
                run_id=str(row["run_id"]),
                experiment_id=candidate.experiment_id,
                fold=int(row["fold"]),
                seed=candidate.seed,
            )
            fold_frames.append(fold_frame)
        oof = pd.concat(fold_frames, ignore_index=True)
        coverage = validate_oof(
            oof,
            expected_ids=expected_id_values,
            expected_targets=expected_targets,
            protected_ids=protected_ids,
            labels=SEASON_LABELS,
        )
        coverage_by_experiment[candidate.experiment_id] = coverage
        metrics = multiclass_metrics(
            oof["y_true"].astype(str),
            probabilities=oof.loc[:, [f"prob_{label}" for label in SEASON_LABELS]].to_numpy(
                dtype=float
            ),
            labels=SEASON_LABELS,
            y_pred=oof["y_pred"].astype(str),
        )
        if stability_summary is not None:
            summary_row = stability_summary.loc[
                stability_summary["experiment_id"].eq(candidate.experiment_id)
                & pd.to_numeric(stability_summary["seed"], errors="raise")
                .astype(int)
                .eq(candidate.seed)
            ]
            if len(summary_row) != 1:
                raise ValueError(f"{candidate.experiment_id} stability summary is incomplete")
            recorded = summary_row.iloc[0]
            comparisons = {
                "pooled_macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "spring_precision": metrics["per_class"]["Spring"]["precision"],
                "spring_recall": metrics["per_class"]["Spring"]["recall"],
                "spring_f1": metrics["per_class"]["Spring"]["f1"],
            }
            for column, recomputed in comparisons.items():
                if not np.isclose(
                    float(recorded[column]),
                    float(recomputed),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"{candidate.experiment_id} stability {column} differs from OOF bytes"
                    )
        packs.append(
            CandidateOOFPack(
                candidate=candidate.candidate,
                experiment_id=candidate.experiment_id,
                seed=candidate.seed,
                oof=oof,
                registry=rows,
            )
        )

    coverage_hashes = {canonical_sha256(value) for value in coverage_by_experiment.values()}
    if len(coverage_hashes) != 1:
        raise ValueError("slice candidates do not cover the same canonical OOF products")
    combined = pd.concat(
        [pack.registry.assign(candidate=pack.candidate) for pack in packs],
        ignore_index=True,
    )
    for candidate in ("C2", "I2"):
        candidate_rows = combined.loc[combined["candidate"].eq(candidate)]
        for field in ("implementation_sha256", "loss_id", "model_family", "parameter_count"):
            if candidate_rows[field].nunique() != 1:
                raise ValueError(f"{candidate} changed {field} between seeds")
    return packs, coverage_by_experiment, prediction_artifacts


def build_slice_decision(
    tables: SliceAnalysisTables,
    *,
    stability_decision: Mapping[str, Any],
    low_support_threshold: int = 100,
) -> dict[str, Any]:
    """Describe measured weaknesses without using post-hoc slices to select a model."""
    if stability_decision.get("current_candidate") != "I2":
        raise ValueError("G6 requires the seed-stable I2 candidate from G5")
    if low_support_threshold < 1:
        raise ValueError("low-support threshold must be positive")
    deltas = tables.candidate_slice_deltas.copy()
    if not deltas["c2_support"].eq(deltas["i2_support"]).all():
        raise ValueError("C2 and I2 slice supports differ")
    finite = deltas.loc[np.isfinite(pd.to_numeric(deltas["i2_minus_c2_macro_f1"]))].copy()
    finite["support"] = pd.to_numeric(finite["c2_support"], errors="raise").astype(int)
    finite["low_support"] = finite["support"].lt(low_support_threshold)
    negative = finite.loc[finite["i2_minus_c2_macro_f1"].lt(0)]
    adequately_supported_negative = negative.loc[~negative["low_support"]]
    pivot = finite.pivot_table(
        index=["slice_family", "slice_name"],
        columns="seed",
        values="i2_minus_c2_macro_f1",
        aggfunc="first",
    )
    if {2753, 2026} <= set(pivot.columns):
        sign_reversals = int((pivot[2753] * pivot[2026]).lt(0).sum())
    else:
        sign_reversals = 0
    worst_rows = (
        finite.sort_values("i2_minus_c2_macro_f1", kind="stable")
        .head(5)
        .loc[
            :,
            [
                "seed",
                "slice_family",
                "slice_name",
                "support",
                "low_support",
                "i2_minus_c2_macro_f1",
            ],
        ]
        .to_dict(orient="records")
    )
    spring = tables.spring_metrics.set_index(["candidate", "seed"])
    spring_deltas = {
        str(seed): {
            "i2_minus_c2_recall": float(
                spring.loc[("I2", seed), "recall"] - spring.loc[("C2", seed), "recall"]
            ),
            "i2_minus_c2_f1": float(
                spring.loc[("I2", seed), "f1"] - spring.loc[("C2", seed), "f1"]
            ),
        }
        for seed in (2753, 2026)
    }
    return {
        "schema_version": "1.0.0",
        "gate": "G6-SLICE",
        "decision_status": "closed",
        "analysis_role": "development_oof_diagnosis_only",
        "current_candidate": "I2",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "evaluated_candidate_seed_pairs": 4,
        "finite_candidate_slice_comparisons": len(finite),
        "low_support_threshold": low_support_threshold,
        "i2_below_c2_slice_count": len(negative),
        "adequately_supported_i2_below_c2_slice_count": len(adequately_supported_negative),
        "low_support_i2_below_c2_slice_count": int(negative["low_support"].sum()),
        "slice_delta_sign_reversals_between_seeds": sign_reversals,
        "spring_deltas_by_seed": spring_deltas,
        "worst_i2_minus_c2_slice_rows": worst_rows,
        "interpretation_rule": (
            "Slices diagnose where frozen models fail. They do not create a new selection "
            "rule after results are visible and therefore cannot change the G5 candidate."
        ),
        "confidence_warning": (
            "Confidence is raw softmax confidence. Calibration is still pending, so it is "
            "used to rank errors only and is not a deployment probability claim."
        ),
        "metadata_boundary": (
            "Year, file size, product-family size, and image mode are joined after OOF "
            "prediction and are never inference features."
        ),
        "next_question": (
            "Probe robustness and deployment cost before calibration, grouped bootstrap, "
            "and deterministic Grad-CAM."
        ),
    }


def build_shortcut_error_slice_evidence(
    *,
    project_root: str | Path = ROOT,
    analysis_config_path: str | Path = DEFAULT_ANALYSIS_CONFIG,
    stability_manifest_path: str | Path = DEFAULT_STABILITY_MANIFEST,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Build deterministic G6 evidence from verified cached OOF predictions only."""
    root = Path(project_root)
    spec_path = _resolve_evidence_path(analysis_config_path, project_root=root)
    spec = load_slice_analysis_spec(spec_path)
    stability, resolved_stability, sections = load_verified_stability_manifest(
        stability_manifest_path,
        project_root=root,
    )

    splits_path = root / "data/processed/splits.csv"
    label_maps_path = root / "data/processed/label_maps.json"
    splits = load_splits(splits_path)
    with label_maps_path.open(encoding="utf-8") as handle:
        label_maps = json.load(handle)
    if tuple(label_maps.get("season", {}).get("classes", ())) != tuple(SEASON_LABELS):
        raise ValueError("canonical Season label-map order changed")
    development = get_samples(splits, partition="development", target="season").reset_index(
        drop=True
    )
    if len(development) != spec.expected_row_count:
        raise ValueError("canonical development Season row count changed")
    expected_ids = development["id"].astype(int).tolist()
    expected_targets = dict(
        zip(development["id"].astype(int), development["season"].astype(str), strict=True)
    )
    protected_ids = set(
        splits.loc[splits["partition"].isin(["holdout", "quarantine"]), "id"].astype(int)
    )

    registry = pd.read_csv(
        sections["artifacts"]["registry_snapshot"],
        dtype=str,
        keep_default_na=False,
    )
    stability_summary = pd.read_csv(sections["artifacts"]["seed_stability"])
    with sections["artifacts"]["decision"].open(encoding="utf-8") as handle:
        stability_decision = json.load(handle)
    config_hashes = load_declared_config_hashes(sections["input_configs"])

    packs, coverage_by_experiment, prediction_artifacts = load_candidate_oof_packs(
        registry,
        spec,
        project_root=root,
        expected_ids=expected_ids,
        expected_targets=expected_targets,
        protected_ids=protected_ids,
        split_sha256=compute_sha256(splits_path),
        label_map_sha256=compute_sha256(label_maps_path),
        config_sha256_by_experiment=config_hashes,
        stability_summary=stability_summary,
    )
    coverage_sha256 = canonical_sha256(next(iter(coverage_by_experiment.values())))
    if coverage_sha256 != str(stability.get("coverage_sha256", "")):
        raise ValueError("G6 OOF coverage differs from the verified G5 coverage")

    bundle = build_slice_assignments(splits, spec)
    tables = analyse_slice_packs(packs, bundle, spec)
    decision = build_slice_decision(
        tables,
        stability_decision=stability_decision,
        low_support_threshold=spec.low_support_threshold,
    )

    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    paths = {
        "slice_metrics": evidence_root / "slice_metrics.csv",
        "candidate_slice_deltas": evidence_root / "candidate_slice_deltas.csv",
        "slice_contrasts": evidence_root / "slice_contrasts.csv",
        "spring_metrics": evidence_root / "spring_metrics.csv",
        "spring_destinations": evidence_root / "spring_destinations.csv",
        "error_confusions": evidence_root / "error_confusions.csv",
        "error_examples": evidence_root / "error_examples.csv",
        "file_size_boundaries": evidence_root / "file_size_boundaries.csv",
        "article_type_mappings": evidence_root / "article_type_mappings.csv",
        "article_type_fold_audit": evidence_root / "article_type_fold_audit.csv",
        "slice_support": evidence_root / "slice_support.csv",
        "registry_snapshot": evidence_root / "registry_snapshot.csv",
        "decision": evidence_root / "decision.json",
        "slice_macro_f1_figure": figure_root / "shortcut_slice_macro_f1.png",
        "spring_destinations_figure": figure_root / "spring_error_destinations.png",
    }
    frames = {
        "slice_metrics": tables.slice_metrics,
        "candidate_slice_deltas": tables.candidate_slice_deltas,
        "slice_contrasts": tables.slice_contrasts,
        "spring_metrics": tables.spring_metrics,
        "spring_destinations": tables.spring_destinations,
        "error_confusions": tables.error_confusions,
        "error_examples": tables.error_examples,
        "file_size_boundaries": bundle.file_size_boundaries,
        "article_type_mappings": bundle.article_type_mappings,
        "article_type_fold_audit": bundle.article_type_fold_audit,
        "slice_support": bundle.slice_support,
        "registry_snapshot": pd.concat(
            [pack.registry.assign(candidate=pack.candidate) for pack in packs],
            ignore_index=True,
        ).sort_values(["candidate", "seed", "fold", "run_id"], kind="stable"),
    }
    for name, frame in frames.items():
        atomic_write_csv(paths[name], frame)
    atomic_write_json(paths["decision"], decision)
    plot_slice_macro_f1(tables.slice_metrics, paths["slice_macro_f1_figure"])
    plot_spring_destinations(
        tables.spring_destinations,
        paths["spring_destinations_figure"],
    )

    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=root),
            "sha256": compute_sha256(path),
        }
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G6-SLICE",
        "decision_status": "closed",
        "analysis_role": "development_oof_diagnosis_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "coverage_sha256": coverage_sha256,
        "slice_assignment_sha256": bundle.assignment_sha256,
        "analysis_config": {
            "path": _portable_artifact_path(spec_path, fallback_root=root),
            "sha256": compute_sha256(spec_path),
        },
        "stability_manifest": {
            "path": _portable_artifact_path(resolved_stability, fallback_root=root),
            "sha256": compute_sha256(resolved_stability),
        },
        "canonical_inputs": {
            "splits": {
                "path": _portable_artifact_path(splits_path, fallback_root=root),
                "sha256": compute_sha256(splits_path),
            },
            "label_maps": {
                "path": _portable_artifact_path(label_maps_path, fallback_root=root),
                "sha256": compute_sha256(label_maps_path),
            },
        },
        "input_predictions": prediction_artifacts,
        "artifacts": artifact_manifest,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


__all__ = [
    "build_shortcut_error_slice_evidence",
    "build_slice_decision",
    "load_candidate_oof_packs",
    "load_declared_config_hashes",
    "load_verified_stability_manifest",
]
