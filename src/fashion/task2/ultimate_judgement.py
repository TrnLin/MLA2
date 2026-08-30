"""Audited Task 2 model judgement and immutable pre-holdout freeze."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil

from fashion.config import (
    ROOT,
    TASK2_CONFIG_DIR,
    TASK2_EVIDENCE_DIR,
    TASK2_SELECTION_FREEZE_JSON,
)
from fashion.data.hashing import compute_sha256
from fashion.task2.bootstrap_evidence import load_verified_calibration_manifest
from fashion.task2.calibration_evidence import load_verified_robustness_manifest
from fashion.task2.evidence import _portable_artifact_path, _resolve_evidence_path
from fashion.task2.experiments import load_experiment_config
from fashion.task2.gradcam_evidence import load_verified_bootstrap_manifest
from fashion.task2.multitask import load_i2_config
from fashion.task2.robustness_evidence import load_verified_slice_manifest
from fashion.task2.slice_evidence import load_verified_stability_manifest
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import implementation_sha256, verify_implementation_at_head
from fashion.train.metrics import SEASON_LABELS
from fashion.train.reproducibility import capture_git_state, capture_runtime

ULTIMATE_JUDGEMENT_CONFIG_PATH = TASK2_CONFIG_DIR / "g7_ultimate_judgement.json"
DEFAULT_GRADCAM_MANIFEST = Path("results/evidence/task2/gradcam_failure_review/manifest.json")
DEFAULT_EVIDENCE_DIRECTORY = TASK2_EVIDENCE_DIR / "ultimate_judgement"
ULTIMATE_JUDGEMENT_IMPLEMENTATION_PATHS = (
    "src/fashion/task2/ultimate_judgement.py",
    "src/fashion/train/artifacts.py",
)

_GRADCAM_MANIFEST_FIELDS = {
    "analysis_config",
    "analysis_role",
    "artifacts",
    "bootstrap_manifest",
    "calibration_manifest",
    "candidate_selection_affected",
    "canonical_inputs",
    "causal_failure_claim_allowed",
    "decision_status",
    "gate",
    "git_commit",
    "git_dirty",
    "holdout_opened",
    "implementation_files_at_head",
    "implementation_sha256",
    "input_checkpoints",
    "input_images",
    "input_predictions",
    "labels",
    "robustness_manifest",
    "runtime_sha256",
    "schema_version",
    "slice_assignment_sha256",
    "slice_manifest",
    "stability_coverage_sha256",
    "stability_manifest",
    "temporary_heatmaps",
    "ultimate_winner_frozen",
}

_FREEZE_FIELDS = {
    "analysis_role",
    "calibration",
    "canonical_inputs",
    "decision_evidence",
    "decision_provenance",
    "decision_recorded_at_utc",
    "freeze_id",
    "gate",
    "holdout_metrics_present",
    "holdout_opened",
    "immutability",
    "limitations",
    "primary_development_evidence",
    "refit_rule",
    "schema_version",
    "selected_model",
    "selection_checks",
    "stability_evidence",
    "status",
    "upstream_manifests",
}
_ULTIMATE_MANIFEST_FIELDS = {
    "analysis_config",
    "analysis_role",
    "artifacts",
    "candidate_selection_affected",
    "canonical_inputs",
    "decision_status",
    "gate",
    "git_commit",
    "git_dirty",
    "holdout_metrics_present",
    "holdout_opened",
    "implementation_files_at_head",
    "implementation_sha256",
    "labels",
    "runtime_sha256",
    "schema_version",
    "selected_candidate",
    "selected_experiment_id",
    "ultimate_winner_frozen",
    "upstream_manifests",
}
_ULTIMATE_ARTIFACTS = {
    "decision",
    "rejected_alternatives",
    "runtime",
    "scorecard",
    "selection_freeze",
}
_SELECTION_CHECK_FIELDS = {
    "above_reference_in_every_robustness_condition",
    "article_type_conflict_guard",
    "both_grouped_bootstrap_intervals_above_zero",
    "jpeg_drop_guard",
    "primary_seed_macro_f1_lead",
    "stability_seed_macro_f1_lead",
}
_CANDIDATE_PRIMARY_EXPERIMENTS = {
    "C2": "g3-c2-t0-resnet18",
    "I2": "g4-i2-article-type-lambda-0-3-c1",
}
_CANDIDATE_STABILITY_EXPERIMENTS = {
    "C2": "g5-c2-t0-resnet18-s2026",
    "I2": "g5-i2-article-type-lambda-0-3-c1-s2026",
}
_REFIT_CONFIG_CONTRACT = {
    "dataset": "all_development_rows_with_valid_season",
    "epoch_rule": "median_primary_seed_cv_best_epoch",
    "normalisation_scope": "all_valid_development_content_pixels_only",
    "seed": 2753,
    "validation_or_holdout_early_stopping": False,
}
_REFIT_FREEZE_FIELDS = {
    "article_type_missing_labels",
    "checkpoint_rule",
    "dataset",
    "epoch_rule",
    "epochs",
    "normalisation_scope",
    "season_class_weights",
    "seed",
    "source_best_epochs",
    "validation_or_holdout_early_stopping",
}
_SELECTED_MODEL_FIELDS = {
    "auxiliary_target_used_at_inference",
    "auxiliary_training_target",
    "benchmark_only",
    "candidate",
    "experiment_id",
    "final_eligible",
    "inference_inputs",
    "model_family",
    "parameter_count",
    "scratch",
    "weights",
}
_CANONICAL_INPUT_FIELDS = {"label_maps", "splits"}
_DECISION_EVIDENCE_FIELDS = {
    "analysis_config",
    "decision",
    "rejected_alternatives",
    "scorecard",
}
_UPSTREAM_MANIFEST_FIELDS = {
    "bootstrap",
    "calibration",
    "gradcam",
    "robustness",
    "selected_experiment",
    "slices",
    "stability",
}
_PRIMARY_EVIDENCE_FIELDS = {
    "augmentation",
    "config",
    "config_semantic_sha256",
    "folds",
    "image_size",
    "implementation_sha256",
    "loss_id",
    "metric",
    "pooled_oof_macro_f1",
    "run_ids",
    "seed",
    "transform_id",
    "valid_development_rows",
}
_STABILITY_EVIDENCE_FIELDS = {
    "experiment_id",
    "pooled_oof_macro_f1",
    "seed",
    "two_seeds_cover_all_randomness",
}
_CALIBRATION_FREEZE_FIELDS = {
    "app_review_threshold",
    "evaluation_claim_allowed",
    "fit_rows",
    "fit_scope",
    "method",
    "purpose",
    "temperature",
}
_IMMUTABILITY_FIELDS = {
    "different_payload_overwrite_allowed",
    "identical_retry_allowed",
    "model_change_after_holdout_allowed",
}


@dataclass(frozen=True)
class CandidateSpec:
    """One final-eligible candidate and its two frozen experiment identities."""

    candidate: str
    role: str
    experiment_id_seed_2753: str
    experiment_id_seed_2026: str


@dataclass(frozen=True)
class DecisionRules:
    """Numeric rules that convert measured evidence into one winner."""

    minimum_macro_f1_lead_at_each_seed: float
    maximum_article_type_conflict_disadvantage: float
    maximum_excess_jpeg_macro_f1_drop: float
    maximum_robustness_disadvantage_for_cost_tie_break: float
    practical_tie_threshold_macro_f1: float
    require_both_grouped_bootstrap_intervals_above_zero: bool
    require_challenger_above_reference_in_every_robustness_condition: bool


@dataclass(frozen=True)
class UltimateJudgementSpec:
    """Validated G7 decision, calibration, and refit contract."""

    analysis_id: str
    candidates: tuple[CandidateSpec, ...]
    rules: DecisionRules
    primary_metric: str
    refit_seed: int
    refit_epoch_rule: str
    refit_dataset: str
    refit_normalisation_scope: str
    calibration_source: str
    calibration_purpose: str

    def candidate_for_role(self, role: str) -> CandidateSpec:
        matches = [candidate for candidate in self.candidates if candidate.role == role]
        if len(matches) != 1:
            raise ValueError(f"ultimate judgement requires exactly one {role} candidate")
        return matches[0]


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _load_json_object(path: Path, scope: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{scope} must be a JSON object")
    return dict(payload)


def load_ultimate_judgement_spec(
    path: str | Path = ULTIMATE_JUDGEMENT_CONFIG_PATH,
) -> UltimateJudgementSpec:
    """Load the exact two-candidate G7 contract without reading results."""
    payload = _load_json_object(Path(path), "ultimate judgement config")
    _require_exact_keys(
        payload,
        {
            "analysis_id",
            "calibration",
            "candidates",
            "decision_rules",
            "labels",
            "primary_metric",
            "refit",
            "schema_version",
            "stage",
            "target",
            "warnings",
        },
        "ultimate judgement config",
    )
    expected_identity = {
        "schema_version": "1.0.0",
        "analysis_id": "g7-ultimate-season-judgement",
        "stage": "g7_ultimate_judgement_and_pre_holdout_freeze",
        "target": "season",
        "primary_metric": "pooled_five_fold_oof_macro_f1",
        "labels": list(SEASON_LABELS),
    }
    mismatches = [name for name, expected in expected_identity.items() if payload[name] != expected]
    if mismatches:
        raise ValueError(f"ultimate judgement identity changed: {mismatches}")

    candidate_payloads = payload["candidates"]
    if not isinstance(candidate_payloads, list) or len(candidate_payloads) != 2:
        raise ValueError("ultimate judgement requires exactly two candidates")
    candidates = []
    for item in candidate_payloads:
        if not isinstance(item, Mapping):
            raise ValueError("ultimate judgement candidate must be an object")
        _require_exact_keys(
            item,
            {
                "candidate",
                "experiment_id_seed_2026",
                "experiment_id_seed_2753",
                "role",
            },
            "ultimate judgement candidate",
        )
        candidates.append(
            CandidateSpec(
                candidate=str(item["candidate"]),
                role=str(item["role"]),
                experiment_id_seed_2753=str(item["experiment_id_seed_2753"]),
                experiment_id_seed_2026=str(item["experiment_id_seed_2026"]),
            )
        )
    if {(candidate.candidate, candidate.role) for candidate in candidates} != {
        ("C2", "reference"),
        ("I2", "challenger"),
    }:
        raise ValueError("ultimate judgement changed the frozen C2/I2 candidate roles")
    for candidate in candidates:
        if (
            candidate.experiment_id_seed_2753 != _CANDIDATE_PRIMARY_EXPERIMENTS[candidate.candidate]
            or candidate.experiment_id_seed_2026
            != _CANDIDATE_STABILITY_EXPERIMENTS[candidate.candidate]
        ):
            raise ValueError(
                f"ultimate judgement changed the audited {candidate.candidate} experiments"
            )

    rules_payload = payload["decision_rules"]
    if not isinstance(rules_payload, Mapping):
        raise ValueError("ultimate judgement decision_rules must be an object")
    _require_exact_keys(
        rules_payload,
        {
            "maximum_article_type_conflict_disadvantage",
            "maximum_excess_jpeg_macro_f1_drop",
            "maximum_robustness_disadvantage_for_cost_tie_break",
            "minimum_macro_f1_lead_at_each_seed",
            "practical_tie_threshold_macro_f1",
            "require_both_grouped_bootstrap_intervals_above_zero",
            "require_challenger_above_reference_in_every_robustness_condition",
        },
        "ultimate judgement decision_rules",
    )
    for field in (
        "require_both_grouped_bootstrap_intervals_above_zero",
        "require_challenger_above_reference_in_every_robustness_condition",
    ):
        if type(rules_payload[field]) is not bool:
            raise ValueError(f"ultimate judgement decision_rules.{field} must be boolean")
    numeric_rule_fields = set(rules_payload) - {
        "require_both_grouped_bootstrap_intervals_above_zero",
        "require_challenger_above_reference_in_every_robustness_condition",
    }
    if any(
        type(rules_payload[field]) not in (int, float)
        or not np.isfinite(float(rules_payload[field]))
        for field in numeric_rule_fields
    ):
        raise ValueError("ultimate judgement numeric thresholds must be finite JSON numbers")
    rules = DecisionRules(
        minimum_macro_f1_lead_at_each_seed=float(
            rules_payload["minimum_macro_f1_lead_at_each_seed"]
        ),
        maximum_article_type_conflict_disadvantage=float(
            rules_payload["maximum_article_type_conflict_disadvantage"]
        ),
        maximum_excess_jpeg_macro_f1_drop=float(rules_payload["maximum_excess_jpeg_macro_f1_drop"]),
        maximum_robustness_disadvantage_for_cost_tie_break=float(
            rules_payload["maximum_robustness_disadvantage_for_cost_tie_break"]
        ),
        practical_tie_threshold_macro_f1=float(rules_payload["practical_tie_threshold_macro_f1"]),
        require_both_grouped_bootstrap_intervals_above_zero=rules_payload[
            "require_both_grouped_bootstrap_intervals_above_zero"
        ],
        require_challenger_above_reference_in_every_robustness_condition=rules_payload[
            "require_challenger_above_reference_in_every_robustness_condition"
        ],
    )
    if (
        min(
            rules.minimum_macro_f1_lead_at_each_seed,
            rules.maximum_article_type_conflict_disadvantage,
            rules.maximum_excess_jpeg_macro_f1_drop,
            rules.maximum_robustness_disadvantage_for_cost_tie_break,
            rules.practical_tie_threshold_macro_f1,
        )
        < 0
    ):
        raise ValueError("ultimate judgement thresholds must be non-negative")
    if not (
        rules.require_both_grouped_bootstrap_intervals_above_zero
        and rules.require_challenger_above_reference_in_every_robustness_condition
    ):
        raise ValueError("ultimate judgement weakened a required final safety check")

    refit = payload["refit"]
    calibration = payload["calibration"]
    warnings = payload["warnings"]
    if not all(isinstance(section, Mapping) for section in (refit, calibration, warnings)):
        raise ValueError("ultimate judgement refit/calibration/warnings must be objects")
    _require_exact_keys(refit, set(_REFIT_CONFIG_CONTRACT), "ultimate judgement refit")
    refit_mismatches = [
        name for name, expected in _REFIT_CONFIG_CONTRACT.items() if refit[name] != expected
    ]
    if refit_mismatches:
        raise ValueError(
            f"ultimate judgement changed the development-only refit contract: {refit_mismatches}"
        )
    _require_exact_keys(
        calibration,
        {
            "app_review_threshold",
            "deployment_temperature_evaluation_claim_allowed",
            "purpose",
            "source",
        },
        "ultimate judgement calibration",
    )
    if (
        calibration["source"] != "all_primary_seed_oof_rows"
        or calibration["purpose"] != "future_frozen_bundle_confidence_only"
    ):
        raise ValueError("ultimate judgement changed the calibration claim boundary")
    if calibration.get("app_review_threshold") is not None:
        raise ValueError("G7 cannot freeze an app threshold without a business cost")
    if calibration.get("deployment_temperature_evaluation_claim_allowed") is not False:
        raise ValueError("deployment temperature cannot become evaluation evidence")
    _require_exact_keys(
        warnings,
        {
            "analysis_cannot_add_or_retune_candidates",
            "final_model_must_be_scratch_trained",
            "holdout_is_forbidden",
            "pretrained_benchmark_is_not_final_eligible",
            "two_seeds_do_not_cover_all_training_randomness",
        },
        "ultimate judgement warnings",
    )
    if any(value is not True for value in warnings.values()):
        raise ValueError("ultimate judgement safety warnings must all remain enabled")
    return UltimateJudgementSpec(
        analysis_id=str(payload["analysis_id"]),
        candidates=tuple(candidates),
        rules=rules,
        primary_metric=str(payload["primary_metric"]),
        refit_seed=int(refit["seed"]),
        refit_epoch_rule=str(refit["epoch_rule"]),
        refit_dataset=str(refit["dataset"]),
        refit_normalisation_scope=str(refit["normalisation_scope"]),
        calibration_source=str(calibration["source"]),
        calibration_purpose=str(calibration["purpose"]),
    )


def _verify_declaration(
    declaration: Any,
    *,
    project_root: Path,
    name: str,
) -> Path:
    if not isinstance(declaration, Mapping):
        raise ValueError(f"{name} declaration must be an object")
    _require_exact_keys(declaration, {"path", "sha256"}, f"{name} declaration")
    path = _resolve_evidence_path(str(declaration["path"]), project_root=project_root)
    digest = str(declaration["sha256"])
    if len(digest) != 64:
        raise ValueError(f"{name} declaration has an invalid SHA-256")
    verify_artifact(path, digest)
    return path


def _portable_declaration(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "path": _portable_artifact_path(path, fallback_root=root),
        "sha256": compute_sha256(path),
    }


def _load_verified_gradcam_boundary(
    path: str | Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    resolved = _resolve_evidence_path(path, project_root=project_root)
    manifest = _load_json_object(resolved, "Grad-CAM manifest")
    _require_exact_keys(manifest, _GRADCAM_MANIFEST_FIELDS, "Grad-CAM manifest")
    identity = {
        "schema_version": "1.0.0",
        "gate": "G6-GRADCAM-FAILURE-REVIEW",
        "decision_status": "closed",
        "analysis_role": "development_oof_explainability_and_failure_diagnosis_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "causal_failure_claim_allowed": False,
        "git_dirty": False,
        "labels": list(SEASON_LABELS),
    }
    mismatches = [name for name, expected in identity.items() if manifest[name] != expected]
    if mismatches:
        raise ValueError(f"Grad-CAM evidence boundary changed: {mismatches}")
    for field in (
        "implementation_sha256",
        "runtime_sha256",
        "slice_assignment_sha256",
        "stability_coverage_sha256",
    ):
        if len(str(manifest[field])) != 64:
            raise ValueError(f"Grad-CAM manifest has invalid {field}")
    if len(str(manifest["git_commit"])) != 40:
        raise ValueError("Grad-CAM manifest has invalid git_commit")
    artifacts = {
        str(name): _verify_declaration(
            declaration,
            project_root=project_root,
            name=f"Grad-CAM artifact {name}",
        )
        for name, declaration in dict(manifest["artifacts"]).items()
    }
    required_artifacts = {
        "attention_metrics",
        "c2_contact_sheet",
        "decision",
        "failure_taxonomy",
        "failure_taxonomy_summary",
        "heatmap_index",
        "i2_contact_sheet",
        "registry_snapshot",
        "runtime",
        "selected_examples",
    }
    if not required_artifacts <= set(artifacts):
        raise ValueError("Grad-CAM manifest lacks final-judgement evidence")
    canonical_inputs = {
        str(name): _verify_declaration(
            declaration,
            project_root=project_root,
            name=f"Grad-CAM canonical input {name}",
        )
        for name, declaration in dict(manifest["canonical_inputs"]).items()
    }
    if set(canonical_inputs) != {"splits", "label_maps"}:
        raise ValueError("Grad-CAM manifest changed canonical inputs")
    upstream = {
        name: _verify_declaration(
            manifest[f"{name}_manifest"],
            project_root=project_root,
            name=f"Grad-CAM {name} manifest",
        )
        for name in ("bootstrap", "calibration", "robustness", "slice", "stability")
    }
    analysis_config = _verify_declaration(
        manifest["analysis_config"],
        project_root=project_root,
        name="Grad-CAM analysis config",
    )
    declared_inputs: dict[str, dict[str, Path]] = {}
    for field, expected_count in (
        ("input_checkpoints", 8),
        ("input_images", 44),
        ("input_predictions", 10),
    ):
        declarations = manifest[field]
        if not isinstance(declarations, Mapping) or len(declarations) != expected_count:
            raise ValueError(f"Grad-CAM {field} must retain exactly {expected_count} declarations")
        declared_inputs[field] = {
            str(name): _verify_declaration(
                declaration,
                project_root=project_root,
                name=f"Grad-CAM {field}.{name}",
            )
            for name, declaration in declarations.items()
        }
    temporary = manifest["temporary_heatmaps"]
    if not isinstance(temporary, Mapping):
        raise ValueError("Grad-CAM temporary heatmaps declaration must be an object")
    _require_exact_keys(
        temporary,
        {"dtype", "index_sha256", "path", "sha256", "shape"},
        "Grad-CAM temporary heatmaps",
    )
    temporary_path = _resolve_evidence_path(str(temporary["path"]), project_root=project_root)
    verify_artifact(temporary_path, str(temporary["sha256"]))
    if (
        temporary["shape"] != [48, 80, 60]
        or temporary["dtype"] != "float32"
        or str(temporary["index_sha256"]) != compute_sha256(artifacts["heatmap_index"])
    ):
        raise ValueError("Grad-CAM temporary heatmap boundary changed")
    decision = _load_json_object(artifacts["decision"], "Grad-CAM decision")
    if (
        decision.get("current_candidate") != "I2"
        or decision.get("zero_heatmap_count") != 0
        or decision.get("causal_failure_claim_allowed") is not False
        or decision.get("ultimate_winner_frozen") is not False
    ):
        raise ValueError("Grad-CAM decision no longer preserves the non-causal G6 boundary")
    runtime = _load_json_object(artifacts["runtime"], "Grad-CAM runtime")
    if canonical_sha256(runtime) != str(manifest["runtime_sha256"]):
        raise ValueError("Grad-CAM runtime semantic hash changed")
    return (
        manifest,
        resolved,
        {
            "analysis_config": analysis_config,
            "artifacts": artifacts,
            "canonical_inputs": canonical_inputs,
            "decision": decision,
            "declared_inputs": declared_inputs,
            "temporary_heatmaps": temporary_path,
            "upstream": upstream,
        },
    )


def _one_row(frame: pd.DataFrame, mask: pd.Series, scope: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"{scope} requires exactly one row; observed {len(rows)}")
    return rows.iloc[0]


def _finite_float(value: Any, scope: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{scope} must be finite")
    return result


def _strict_bool(value: Any, scope: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalised = str(value).strip().lower()
    if normalised in {"true", "false"}:
        return normalised == "true"
    raise ValueError(f"{scope} must be boolean")


def build_candidate_scorecard(
    *,
    spec: UltimateJudgementSpec,
    stability: pd.DataFrame,
    slice_deltas: pd.DataFrame,
    robustness: pd.DataFrame,
    deployment_cost: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Join quality, shortcut, stress, calibration, and cost evidence by candidate."""
    required_stability = {
        "candidate",
        "experiment_id",
        "seed",
        "pooled_macro_f1",
        "spring_f1",
        "five_fold_runtime_minutes",
        "parameter_count",
        "median_best_epoch",
    }
    if not required_stability <= set(stability):
        raise ValueError("seed-stability table lacks final scorecard columns")
    conditions = set(robustness["condition"].astype(str))
    expected_conditions = {
        "clean",
        "jpeg_quality_85",
        "brightness_0_85",
        "brightness_1_15",
        "gaussian_blur_radius_1",
    }
    if conditions != expected_conditions:
        raise ValueError("robustness scorecard condition set changed")
    if set(calibration["calibration_method"].astype(str)) != {
        "uncalibrated",
        "cross_fitted_temperature",
    }:
        raise ValueError("calibration scorecard requires before/after rows")

    rows: list[dict[str, Any]] = []
    for candidate in spec.candidates:
        primary = _one_row(
            stability,
            stability["candidate"].eq(candidate.candidate)
            & stability["experiment_id"].eq(candidate.experiment_id_seed_2753)
            & pd.to_numeric(stability["seed"], errors="raise").eq(2753),
            f"{candidate.candidate} primary stability",
        )
        second = _one_row(
            stability,
            stability["candidate"].eq(candidate.candidate)
            & stability["experiment_id"].eq(candidate.experiment_id_seed_2026)
            & pd.to_numeric(stability["seed"], errors="raise").eq(2026),
            f"{candidate.candidate} second-seed stability",
        )
        conflict_by_seed: dict[int, float] = {}
        for seed in (2753, 2026):
            conflict = _one_row(
                slice_deltas,
                pd.to_numeric(slice_deltas["seed"], errors="raise").eq(seed)
                & slice_deltas["slice_family"].eq("article_type_shortcut")
                & slice_deltas["slice_name"].eq("conflict"),
                f"ArticleType-conflict seed {seed}",
            )
            conflict_by_seed[seed] = _finite_float(
                conflict[f"{candidate.candidate.lower()}_macro_f1"],
                f"{candidate.candidate} conflict macro-F1 seed {seed}",
            )

        clean = _one_row(
            robustness,
            robustness["condition"].eq("clean"),
            "clean robustness comparison",
        )
        jpeg = _one_row(
            robustness,
            robustness["condition"].eq("jpeg_quality_85"),
            "JPEG robustness comparison",
        )
        macro_column = f"{candidate.candidate.lower()}_macro_f1"
        clean_macro = _finite_float(clean[macro_column], f"{candidate.candidate} clean macro-F1")
        primary_macro = _finite_float(
            primary["pooled_macro_f1"], f"{candidate.candidate} primary macro-F1"
        )
        if not np.isclose(clean_macro, primary_macro, rtol=0.0, atol=1e-12):
            raise ValueError(f"{candidate.candidate} clean robustness score differs from OOF")
        stress_rows = robustness.loc[~robustness["condition"].eq("clean")].copy()
        worst_index = pd.to_numeric(stress_rows[macro_column], errors="raise").idxmin()
        worst = stress_rows.loc[worst_index]
        calibrated = _one_row(
            calibration,
            calibration["candidate"].eq(candidate.candidate)
            & calibration["calibration_method"].eq("cross_fitted_temperature"),
            f"{candidate.candidate} cross-fitted calibration",
        )
        cpu = _one_row(
            deployment_cost,
            deployment_cost["candidate"].eq(candidate.candidate)
            & deployment_cost["device"].eq("cpu")
            & deployment_cost["available"].astype(str).str.lower().eq("true"),
            f"{candidate.candidate} CPU deployment cost",
        )
        cuda = _one_row(
            deployment_cost,
            deployment_cost["candidate"].eq(candidate.candidate)
            & deployment_cost["device"].eq("cuda")
            & deployment_cost["available"].astype(str).str.lower().eq("true"),
            f"{candidate.candidate} CUDA deployment cost",
        )
        if int(primary["parameter_count"]) != int(cpu["parameter_count"]):
            raise ValueError(f"{candidate.candidate} parameter count changed across evidence")
        rows.append(
            {
                "candidate": candidate.candidate,
                "role": candidate.role,
                "experiment_id_seed_2753": candidate.experiment_id_seed_2753,
                "experiment_id_seed_2026": candidate.experiment_id_seed_2026,
                "primary_macro_f1": primary_macro,
                "stability_macro_f1": _finite_float(
                    second["pooled_macro_f1"],
                    f"{candidate.candidate} stability macro-F1",
                ),
                "primary_spring_f1": _finite_float(
                    primary["spring_f1"], f"{candidate.candidate} primary Spring F1"
                ),
                "stability_spring_f1": _finite_float(
                    second["spring_f1"], f"{candidate.candidate} stability Spring F1"
                ),
                "article_type_conflict_macro_f1_seed_2753": conflict_by_seed[2753],
                "article_type_conflict_macro_f1_seed_2026": conflict_by_seed[2026],
                "jpeg_macro_f1": _finite_float(
                    jpeg[macro_column], f"{candidate.candidate} JPEG macro-F1"
                ),
                "jpeg_drop_vs_clean": clean_macro
                - _finite_float(jpeg[macro_column], f"{candidate.candidate} JPEG macro-F1"),
                "worst_stress_condition": str(worst["condition"]),
                "worst_stress_macro_f1": _finite_float(
                    worst[macro_column], f"{candidate.candidate} worst stress macro-F1"
                ),
                "worst_stress_spring_recall": _finite_float(
                    worst[f"{candidate.candidate.lower()}_spring_recall"],
                    f"{candidate.candidate} worst stress Spring recall",
                ),
                "cross_fitted_nll": _finite_float(
                    calibrated["nll"], f"{candidate.candidate} calibrated NLL"
                ),
                "cross_fitted_brier": _finite_float(
                    calibrated["brier"], f"{candidate.candidate} calibrated Brier"
                ),
                "cross_fitted_ece": _finite_float(
                    calibrated["ece"], f"{candidate.candidate} calibrated ECE"
                ),
                "five_fold_runtime_minutes_primary": _finite_float(
                    primary["five_fold_runtime_minutes"],
                    f"{candidate.candidate} training runtime",
                ),
                "parameter_count": int(primary["parameter_count"]),
                "parameter_and_buffer_bytes": int(cpu["parameter_and_buffer_bytes"]),
                "cpu_end_to_end_median_ms": _finite_float(
                    cpu["end_to_end_median_ms"], f"{candidate.candidate} CPU latency"
                ),
                "cuda_end_to_end_median_ms": _finite_float(
                    cuda["end_to_end_median_ms"], f"{candidate.candidate} CUDA latency"
                ),
                "median_best_epoch_primary": int(primary["median_best_epoch"]),
                "selected": False,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("role", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def apply_ultimate_judgement(
    scorecard: pd.DataFrame,
    *,
    interval_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    spec: UltimateJudgementSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen G7 rule and return one selected scratch candidate."""
    reference_spec = spec.candidate_for_role("reference")
    challenger_spec = spec.candidate_for_role("challenger")
    reference = _one_row(
        scorecard, scorecard["candidate"].eq(reference_spec.candidate), "reference scorecard"
    )
    challenger = _one_row(
        scorecard,
        scorecard["candidate"].eq(challenger_spec.candidate),
        "challenger scorecard",
    )
    primary_lead = float(challenger["primary_macro_f1"] - reference["primary_macro_f1"])
    stability_lead = float(challenger["stability_macro_f1"] - reference["stability_macro_f1"])
    conflict_deltas = {
        str(seed): float(
            challenger[f"article_type_conflict_macro_f1_seed_{seed}"]
            - reference[f"article_type_conflict_macro_f1_seed_{seed}"]
        )
        for seed in (2753, 2026)
    }
    excess_jpeg_drop = float(challenger["jpeg_drop_vs_clean"] - reference["jpeg_drop_vs_clean"])

    macro_intervals = interval_summary.loc[
        interval_summary["metric_scope"].eq("overall") & interval_summary["metric"].eq("macro_f1")
    ].copy()
    if set(pd.to_numeric(macro_intervals["seed"], errors="raise").astype(int)) != {2753, 2026}:
        raise ValueError("ultimate judgement requires one overall macro-F1 interval per seed")
    if len(macro_intervals) != 2 or not macro_intervals["replicates"].eq(10_000).all():
        raise ValueError("ultimate judgement requires two complete 10,000-draw intervals")
    bootstrap_by_seed = {
        str(int(row.seed)): {
            "observed_delta": float(row.observed_delta),
            "ci_lower": float(row.ci_lower),
            "ci_upper": float(row.ci_upper),
            "interval_contains_zero": _strict_bool(
                row.interval_contains_zero,
                f"bootstrap interval_contains_zero seed {int(row.seed)}",
            ),
            "replicates": int(row.replicates),
        }
        for row in macro_intervals.itertuples(index=False)
    }
    bootstrap_positive = all(
        value["ci_lower"] > 0 and not value["interval_contains_zero"]
        for value in bootstrap_by_seed.values()
    )
    robustness_deltas = pd.to_numeric(robustness["i2_minus_c2_macro_f1"], errors="raise").to_numpy(
        dtype=float
    )
    if len(robustness_deltas) != 5 or not np.isfinite(robustness_deltas).all():
        raise ValueError("ultimate judgement robustness comparison is incomplete")
    challenger_above_every_condition = bool((robustness_deltas > 0).all())
    worst_robustness_disadvantage = float(max(0.0, -robustness_deltas.min()))

    checks = {
        "primary_seed_macro_f1_lead": primary_lead >= spec.rules.minimum_macro_f1_lead_at_each_seed,
        "stability_seed_macro_f1_lead": stability_lead
        >= spec.rules.minimum_macro_f1_lead_at_each_seed,
        "both_grouped_bootstrap_intervals_above_zero": bootstrap_positive,
        "article_type_conflict_guard": min(conflict_deltas.values())
        >= -spec.rules.maximum_article_type_conflict_disadvantage,
        "jpeg_drop_guard": excess_jpeg_drop <= spec.rules.maximum_excess_jpeg_macro_f1_drop,
        "above_reference_in_every_robustness_condition": challenger_above_every_condition,
    }
    direct_selection = all(checks.values())
    primary_interval_contains_zero = bool(bootstrap_by_seed["2753"]["interval_contains_zero"])
    near_tie = (
        abs(primary_lead) < spec.rules.practical_tie_threshold_macro_f1
        and primary_interval_contains_zero
    )
    challenger_smaller = int(challenger["parameter_and_buffer_bytes"]) < int(
        reference["parameter_and_buffer_bytes"]
    )
    challenger_faster = float(challenger["cpu_end_to_end_median_ms"]) < float(
        reference["cpu_end_to_end_median_ms"]
    )
    tie_break_safety_passed = all(
        checks[name]
        for name in (
            "stability_seed_macro_f1_lead",
            "article_type_conflict_guard",
            "jpeg_drop_guard",
        )
    )
    cost_tie_break = bool(
        near_tie
        and (challenger_smaller or challenger_faster)
        and tie_break_safety_passed
        and worst_robustness_disadvantage
        <= spec.rules.maximum_robustness_disadvantage_for_cost_tie_break
    )
    winner = challenger_spec if direct_selection or cost_tie_break else reference_spec
    winner_row = challenger if winner.candidate == challenger_spec.candidate else reference
    selected = scorecard.copy()
    selected["selected"] = selected["candidate"].eq(winner.candidate)
    if int(selected["selected"].sum()) != 1:
        raise ValueError("ultimate judgement did not produce exactly one winner")
    decision = {
        "schema_version": "1.0.0",
        "gate": "G7-ULTIMATE-JUDGEMENT",
        "decision_status": "closed",
        "analysis_role": "development_evidence_model_selection_and_pre_holdout_freeze",
        "selected_candidate": winner.candidate,
        "selected_experiment_id": winner.experiment_id_seed_2753,
        "reference_candidate": reference_spec.candidate,
        "challenger_candidate": challenger_spec.candidate,
        "candidate_selection_affected": True,
        "ultimate_winner_frozen": True,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "primary_metric": spec.primary_metric,
        "observed_i2_minus_c2_macro_f1": {
            "2753": primary_lead,
            "2026": stability_lead,
        },
        "article_type_conflict_i2_minus_c2_macro_f1": conflict_deltas,
        "i2_excess_jpeg_macro_f1_drop": excess_jpeg_drop,
        "i2_above_c2_in_every_robustness_condition": challenger_above_every_condition,
        "worst_i2_robustness_disadvantage": worst_robustness_disadvantage,
        "bootstrap_intervals": bootstrap_by_seed,
        "selection_checks": checks,
        "direct_selection_rule_passed": direct_selection,
        "near_tie_rule_triggered": near_tie,
        "cost_tie_break_safety_passed": tie_break_safety_passed,
        "cost_tie_break_used": cost_tie_break,
        "challenger_smaller": challenger_smaller,
        "challenger_faster_on_cpu": challenger_faster,
        "decision_summary": (
            "Select I2 because it leads C2 at both seeds, both grouped-bootstrap "
            "intervals remain above zero, declared shortcut and JPEG guards pass, and "
            "I2 remains above C2 in every robustness condition. The cost tie-break was "
            "not needed; I2 is also smaller and faster."
            if direct_selection
            else (
                "Select I2 under the declared practical-tie rule: the primary interval "
                "contains zero, the stability, shortcut, JPEG, and robustness guards pass, "
                "and I2 is smaller or faster."
                if cost_tie_break
                else "Retain C2 because I2 did not pass the direct rule or the safe "
                "practical-tie rule."
            )
        ),
        "strongest_limitation": (
            "Brightness 0.85 remains a severe shift: the selected model's Spring recall "
            f"falls to {float(winner_row['worst_stress_spring_recall']):.6f}."
        ),
        "claim_boundary": (
            "The judgement covers two fixed training seeds and development OOF evidence; "
            "it does not prove superiority under every random start or real-world shift."
        ),
    }
    return selected, decision


def _read_verified_gate_decision(
    manifest_path: str | Path,
    *,
    project_root: Path,
    expected_gate: str,
) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_evidence_path(manifest_path, project_root=project_root)
    manifest = _load_json_object(resolved, f"{expected_gate} manifest")
    if manifest.get("gate") != expected_gate or manifest.get("decision_status") != "closed":
        raise ValueError(f"{expected_gate} decision source is not closed")
    decision_path = _verify_declaration(
        dict(manifest.get("artifacts", {})).get("decision"),
        project_root=project_root,
        name=f"{expected_gate} decision",
    )
    decision = _load_json_object(decision_path, f"{expected_gate} decision")
    if decision.get("gate") != expected_gate or decision.get("decision_status") != "closed":
        raise ValueError(f"{expected_gate} decision artifact changed identity")
    return decision, resolved


def build_rejected_alternatives(
    *,
    project_root: str | Path,
    scorecard: pd.DataFrame,
) -> pd.DataFrame:
    """Trace every baseline, finalist, intervention, and benchmark that did not win."""
    root = Path(project_root)
    sources = {
        "G2-P": "results/evidence/task2/g2_input_size_ablation/manifest.json",
        "G2-A": "results/evidence/task2/g2_augmentation_ablation/manifest.json",
        "G2-T": "results/evidence/task2/g2_compact_tuning/manifest.json",
        "G3-F": "results/evidence/task2/g3_full_budget/manifest.json",
        "G4-I1": "results/evidence/task2/i1_class_balance/manifest.json",
        "G4-I2": "results/evidence/task2/i2_multitask/manifest.json",
        "G4-PSTAR": "results/evidence/task2/pretraining_benchmark/manifest.json",
    }
    decisions: dict[str, dict[str, Any]] = {}
    manifests: dict[str, Path] = {}
    for gate, path in sources.items():
        decisions[gate], manifests[gate] = _read_verified_gate_decision(
            path,
            project_root=root,
            expected_gate=gate,
        )
    i2 = _one_row(scorecard, scorecard["candidate"].eq("I2"), "I2 scorecard")
    c2 = _one_row(scorecard, scorecard["candidate"].eq("C2"), "C2 scorecard")
    if "selected" not in scorecard:
        raise ValueError("rejected alternatives require the applied ultimate judgement")
    selected_rows = scorecard.loc[scorecard["selected"]]
    if len(selected_rows) != 1:
        raise ValueError("rejected alternatives require exactly one selected candidate")
    selected_candidate = str(selected_rows["candidate"].item())

    def source(gate: str) -> str:
        return _portable_artifact_path(manifests[gate], fallback_root=root)

    rows = [
        {
            "alternative": "B0 majority",
            "role": "comparison_anchor",
            "final_eligible": False,
            "reason": "Measures imbalance only; it ignores image content and has zero Spring F1.",
            "source_manifest": "results/evidence/task2/b0_majority/manifest.json",
        },
        {
            "alternative": "B1 HOG + HSV LinearSVC",
            "role": "serious_classical_baseline",
            "final_eligible": False,
            "reason": (
                "Useful fixed features, but it trails learned models and scores are uncalibrated."
            ),
            "source_manifest": "results/evidence/task2/b1_hog_hsv_svm/manifest.json",
        },
        {
            "alternative": "C3 MobileNetV3-Small",
            "role": "screened_deep_family",
            "final_eligible": True,
            "reason": "Rejected after G1 because its equal-budget macro-F1 ranked third.",
            "source_manifest": "results/evidence/task2/g1_family_screen/manifest.json",
        },
        {
            "alternative": "P1 128x96 input",
            "role": "input_size_ablation",
            "final_eligible": True,
            "reason": (
                "Rejected: P1 minus P0 macro-F1 was "
                f"{float(decisions['G2-P']['observed_p1_minus_p0_macro_f1']):+.6f}, "
                "below the +0.005 gate."
            ),
            "source_manifest": source("G2-P"),
        },
        {
            "alternative": "A1 colour-jitter augmentation",
            "role": "augmentation_ablation",
            "final_eligible": True,
            "reason": (
                "Rejected: A1 minus A0 macro-F1 was "
                f"{float(decisions['G2-A']['observed_a1_minus_a0_macro_f1']):+.6f}."
            ),
            "source_manifest": source("G2-A"),
        },
        {
            "alternative": "Unselected compact tuning settings",
            "role": "learning_rate_weight_decay_ablation",
            "final_eligible": True,
            "reason": "C1 retained T1; C2 retained T0 because no alternative passed +0.003.",
            "source_manifest": source("G2-T"),
        },
        {
            "alternative": "G3 C1-T1 SmallCNN",
            "role": "full_budget_finalist",
            "final_eligible": True,
            "reason": (
                "It was only a provisional near-tie leader and did not remain the winner "
                "after the declared G4-G7 evidence chain."
            ),
            "source_manifest": source("G3-F"),
        },
        {
            "alternative": "I1 effective-number class balancing",
            "role": "minority_class_intervention",
            "final_eligible": True,
            "reason": (
                "Rejected: macro-F1 delta "
                f"{float(decisions['G4-I1']['observed_i1_minus_reference_macro_f1']):+.6f}; "
                "the Spring and other-class guards also failed."
            ),
            "source_manifest": source("G4-I1"),
        },
        {
            "alternative": "I2 ArticleType auxiliary lambda 0.1",
            "role": "multitask_weight_ablation",
            "final_eligible": True,
            "reason": "It passed I2 gates but lambda 0.3 had the higher pooled Season macro-F1.",
            "source_manifest": source("G4-I2"),
        },
        {
            "alternative": "P0S standard-stem scratch ResNet18",
            "role": "benchmark_control",
            "final_eligible": False,
            "reason": "Benchmark-only control; it was never allowed into final selection.",
            "source_manifest": source("G4-PSTAR"),
        },
        {
            "alternative": "P* pretrained standard-stem ResNet18",
            "role": "pretraining_benchmark",
            "final_eligible": False,
            "reason": "Pretrained benchmark-only model; assignment final models must be scratch.",
            "source_manifest": source("G4-PSTAR"),
        },
    ]
    if selected_candidate == "I2":
        rows.append(
            {
                "alternative": "C2 small-stem ResNet18",
                "role": "final_reference",
                "final_eligible": True,
                "reason": (
                    "I2 led primary macro-F1 by "
                    f"{float(i2['primary_macro_f1'] - c2['primary_macro_f1']):+.6f}, "
                    "kept both grouped intervals above zero, and was smaller and faster."
                ),
                "source_manifest": "results/evidence/task2/paired_bootstrap/manifest.json",
            }
        )
    elif selected_candidate == "C2":
        rows.append(
            {
                "alternative": "I2 ArticleType auxiliary lambda 0.3",
                "role": "final_challenger",
                "final_eligible": True,
                "reason": (
                    "Rejected because I2 did not pass the direct rule or the safe "
                    "practical-tie rule; C2 therefore remained the frozen reference."
                ),
                "source_manifest": "results/evidence/task2/paired_bootstrap/manifest.json",
            }
        )
    else:
        raise ValueError(f"unsupported selected candidate: {selected_candidate}")
    frame = pd.DataFrame(rows)
    frame["source_manifest_sha256"] = frame["source_manifest"].map(
        lambda value: compute_sha256(_resolve_evidence_path(value, project_root=root))
    )
    return frame


def _normalised_boolean_values(series: pd.Series) -> set[str]:
    return set(series.astype(str).str.strip().str.lower())


def _selected_registry_identity(
    registry: pd.DataFrame,
    *,
    selected: CandidateSpec,
) -> dict[str, Any]:
    rows = registry.loc[
        registry["experiment_id"].eq(selected.experiment_id_seed_2753)
        & pd.to_numeric(registry["seed"], errors="raise").eq(2753)
    ].copy()
    if len(rows) != 5 or set(pd.to_numeric(rows["fold"], errors="raise").astype(int)) != set(
        range(5)
    ):
        raise ValueError("selected primary experiment does not contain folds 0-4")
    for column, expected in {
        "benchmark_only": "false",
        "final_eligible": "true",
        "scratch": "true",
        "git_dirty": "false",
    }.items():
        if _normalised_boolean_values(rows[column]) != {expected}:
            raise ValueError(f"selected primary registry changed {column}")
    if set(rows["status"].astype(str)) != {"completed"}:
        raise ValueError("selected primary registry contains a non-completed run")
    invariant_fields = (
        "config_sha256",
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
        "transform_id",
        "loss_id",
        "model_family",
        "parameter_count",
    )
    for field in invariant_fields:
        if rows[field].nunique() != 1:
            raise ValueError(f"selected primary registry changed {field} across folds")
    ordered = rows.sort_values("fold", kind="stable")
    best_epochs = pd.to_numeric(ordered["best_epoch"], errors="raise").astype(int)
    median_best_epoch = int(best_epochs.median())
    return {
        "rows": ordered,
        "run_ids": ordered["run_id"].astype(str).tolist(),
        "config_sha256": str(ordered["config_sha256"].iloc[0]),
        "split_sha256": str(ordered["split_sha256"].iloc[0]),
        "label_map_sha256": str(ordered["label_map_sha256"].iloc[0]),
        "implementation_sha256": str(ordered["implementation_sha256"].iloc[0]),
        "transform_id": str(ordered["transform_id"].iloc[0]),
        "loss_id": str(ordered["loss_id"].iloc[0]),
        "model_family": str(ordered["model_family"].iloc[0]),
        "parameter_count": int(ordered["parameter_count"].iloc[0]),
        "median_best_epoch": median_best_epoch,
        "best_epochs": best_epochs.tolist(),
    }


def _validate_timestamp(value: str) -> str:
    normalised = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalised)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("selection freeze timestamp must include UTC timezone")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("selection freeze timestamp must be UTC")
    return value


def _write_immutable_freeze(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create one freeze or accept an identical retry; never overwrite a changed choice."""
    if path.exists():
        existing = _load_json_object(path, "selection freeze")
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise ValueError(
                "selection freeze already exists with different content; create no overwrite"
            )
        return path
    atomic_write_json(path, payload)
    return path


def _lock_owner_is_alive(path: Path) -> bool:
    try:
        owner = _load_json_object(path, "ultimate judgement build lock")
        process_id = int(owner["pid"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    return psutil.pid_exists(process_id)


@contextmanager
def _exclusive_build_lock(path: Path):
    """Serialise evidence writes and recover a lock left by a dead process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid()}
    acquired = False
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_owner_is_alive(path):
                raise RuntimeError(
                    f"ultimate judgement build is already running; lock={path}"
                ) from None
            path.unlink(missing_ok=True)
            continue
        try:
            with os.fdopen(descriptor, mode="w", encoding="utf-8") as handle:
                json.dump(owner, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        acquired = True
        break
    if not acquired:
        raise RuntimeError(f"could not acquire ultimate judgement build lock: {path}")
    try:
        yield
    finally:
        try:
            current_owner = _load_json_object(path, "ultimate judgement build lock")
        except (json.JSONDecodeError, OSError, ValueError):
            current_owner = None
        if current_owner == owner:
            path.unlink(missing_ok=True)


def load_verified_selection_freeze(
    path: str | Path = TASK2_SELECTION_FREEZE_JSON,
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path]:
    """Verify the immutable G7 record before refit or holdout handoff consumes it."""
    root = Path(project_root)
    resolved = _resolve_evidence_path(path, project_root=root)
    payload = _load_json_object(resolved, "selection freeze")
    _require_exact_keys(payload, _FREEZE_FIELDS, "selection freeze")
    identity = {
        "schema_version": "1.0.0",
        "gate": "G7-ULTIMATE-JUDGEMENT",
        "status": "frozen",
        "analysis_role": "development_evidence_model_selection_and_pre_holdout_freeze",
        "holdout_opened": False,
        "holdout_metrics_present": False,
    }
    mismatches = [name for name, expected in identity.items() if payload[name] != expected]
    if mismatches:
        raise ValueError(f"selection freeze boundary changed: {mismatches}")
    selected_model = payload["selected_model"]
    if not isinstance(selected_model, Mapping):
        raise ValueError("selection freeze selected_model must be an object")
    _require_exact_keys(selected_model, _SELECTED_MODEL_FIELDS, "selection freeze selected_model")
    selected_candidate = str(selected_model["candidate"])
    if selected_candidate not in _CANDIDATE_PRIMARY_EXPERIMENTS:
        raise ValueError("selection freeze selected an unsupported candidate")
    if (
        selected_model["experiment_id"] != _CANDIDATE_PRIMARY_EXPERIMENTS[selected_candidate]
        or payload["freeze_id"] != f"task2-season-{selected_candidate.lower()}-development-v1"
    ):
        raise ValueError("selection freeze changed the audited candidate identity")
    expected_auxiliary_target = "articleType" if selected_candidate == "I2" else None
    if (
        selected_model.get("scratch") is not True
        or selected_model.get("weights") is not None
        or selected_model.get("benchmark_only") is not False
        or selected_model.get("final_eligible") is not True
        or selected_model.get("inference_inputs") != ["image"]
        or selected_model.get("auxiliary_training_target") != expected_auxiliary_target
        or selected_model.get("auxiliary_target_used_at_inference") is not False
        or type(selected_model.get("parameter_count")) is not int
        or selected_model["parameter_count"] <= 0
    ):
        raise ValueError("selection freeze changed the scratch image-only model boundary")
    checks = payload["selection_checks"]
    if not isinstance(checks, Mapping):
        raise ValueError("selection freeze selection_checks must be an object")
    _require_exact_keys(checks, _SELECTION_CHECK_FIELDS, "selection freeze selection_checks")
    if any(type(value) is not bool for value in checks.values()):
        raise ValueError("selection freeze selection checks must be strict booleans")
    refit = payload["refit_rule"]
    if not isinstance(refit, Mapping):
        raise ValueError("selection freeze refit_rule must be an object")
    _require_exact_keys(refit, _REFIT_FREEZE_FIELDS, "selection freeze refit_rule")
    refit_mismatches = [
        name for name, expected in _REFIT_CONFIG_CONTRACT.items() if refit[name] != expected
    ]
    source_best_epochs = refit["source_best_epochs"]
    valid_source_epochs = (
        isinstance(source_best_epochs, list)
        and len(source_best_epochs) == 5
        and all(type(epoch) is int and epoch > 0 for epoch in source_best_epochs)
    )
    expected_epochs = int(np.median(source_best_epochs)) if valid_source_epochs else -1
    expected_missing_label_rule = (
        "masked_not_dropped" if selected_candidate == "I2" else "not_applicable_no_auxiliary_target"
    )
    if (
        refit_mismatches
        or type(refit["epochs"]) is not int
        or refit["epochs"] != expected_epochs
        or refit["season_class_weights"] is not None
        or refit["article_type_missing_labels"] != expected_missing_label_rule
        or refit["checkpoint_rule"] != "save_the_declared_final_epoch_state"
    ):
        raise ValueError("selection freeze changed the development-only refit rule")
    calibration = payload["calibration"]
    if not isinstance(calibration, Mapping):
        raise ValueError("selection freeze calibration must be an object")
    _require_exact_keys(
        calibration,
        _CALIBRATION_FREEZE_FIELDS,
        "selection freeze calibration",
    )
    if (
        calibration.get("method") != "scalar_temperature_scaling"
        or calibration.get("fit_rows") != 32_753
        or calibration.get("fit_scope") != "all_primary_seed_oof_rows"
        or calibration.get("purpose") != "future_frozen_bundle_confidence_only"
        or type(calibration.get("temperature")) not in (int, float)
        or not np.isfinite(float(calibration["temperature"]))
        or float(calibration["temperature"]) <= 0
        or calibration.get("evaluation_claim_allowed") is not False
        or calibration.get("app_review_threshold") is not None
    ):
        raise ValueError("selection freeze crossed the calibration claim boundary")
    immutability = payload["immutability"]
    if not isinstance(immutability, Mapping):
        raise ValueError("selection freeze immutability must be an object")
    _require_exact_keys(
        immutability,
        _IMMUTABILITY_FIELDS,
        "selection freeze immutability",
    )
    if immutability != {
        "different_payload_overwrite_allowed": False,
        "identical_retry_allowed": True,
        "model_change_after_holdout_allowed": False,
    }:
        raise ValueError("selection freeze weakened its immutability boundary")
    limitations = payload["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ValueError("selection freeze requires explicit non-empty limitations")
    _validate_timestamp(str(payload["decision_recorded_at_utc"]))
    declared_paths: dict[str, dict[str, Path]] = {}
    for section in ("canonical_inputs", "decision_evidence", "upstream_manifests"):
        declarations = payload[section]
        if not isinstance(declarations, Mapping) or not declarations:
            raise ValueError(f"selection freeze lacks {section}")
        expected_fields = {
            "canonical_inputs": _CANONICAL_INPUT_FIELDS,
            "decision_evidence": _DECISION_EVIDENCE_FIELDS,
            "upstream_manifests": _UPSTREAM_MANIFEST_FIELDS,
        }[section]
        _require_exact_keys(
            declarations,
            expected_fields,
            f"selection freeze {section}",
        )
        declared_paths[section] = {}
        for name, declaration in declarations.items():
            declared_paths[section][str(name)] = _verify_declaration(
                declaration,
                project_root=root,
                name=f"selection freeze {section}.{name}",
            )
    decision = _load_json_object(
        declared_paths["decision_evidence"]["decision"],
        "selection freeze decision evidence",
    )
    if decision.get("selection_checks") != dict(checks):
        raise ValueError("selection freeze checks disagree with the frozen decision")
    direct_selection = decision.get("direct_selection_rule_passed")
    cost_tie_break = decision.get("cost_tie_break_used")
    if type(direct_selection) is not bool or type(cost_tie_break) is not bool:
        raise ValueError("selection freeze decision routes must be strict booleans")
    if direct_selection is not all(checks.values()):
        raise ValueError("selection freeze direct-selection route is inconsistent")
    decision_spec = load_ultimate_judgement_spec(
        declared_paths["decision_evidence"]["analysis_config"]
    )
    safe_cost_tie_break = (
        decision.get("near_tie_rule_triggered") is True
        and decision.get("cost_tie_break_safety_passed") is True
        and (
            decision.get("challenger_smaller") is True
            or decision.get("challenger_faster_on_cpu") is True
        )
        and type(decision.get("worst_i2_robustness_disadvantage")) in (int, float)
        and float(decision["worst_i2_robustness_disadvantage"])
        <= decision_spec.rules.maximum_robustness_disadvantage_for_cost_tie_break
    )
    if cost_tie_break is True and not safe_cost_tie_break:
        raise ValueError("selection freeze contains an unsafe cost tie-break")
    if (
        decision.get("selected_candidate") != selected_candidate
        or decision.get("selected_experiment_id")
        != _CANDIDATE_PRIMARY_EXPERIMENTS[selected_candidate]
        or (selected_candidate == "I2" and not (direct_selection or cost_tie_break))
        or (selected_candidate == "C2" and (direct_selection or cost_tie_break))
    ):
        raise ValueError("selection freeze disagrees with the frozen decision outcome")
    provenance = payload["decision_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("selection freeze decision_provenance must be an object")
    _require_exact_keys(
        provenance,
        {
            "git_commit",
            "git_dirty",
            "implementation_files_at_head",
            "implementation_sha256",
        },
        "selection freeze decision_provenance",
    )
    if (
        len(str(provenance["git_commit"])) != 40
        or provenance["git_dirty"] is not False
        or len(str(provenance["implementation_sha256"])) != 64
    ):
        raise ValueError("selection freeze decision provenance is invalid")
    implementation_files = provenance["implementation_files_at_head"]
    if implementation_files != list(ULTIMATE_JUDGEMENT_IMPLEMENTATION_PATHS):
        raise ValueError("selection freeze decision implementation file set changed")
    current_implementation_hash = implementation_sha256(
        *ULTIMATE_JUDGEMENT_IMPLEMENTATION_PATHS,
        root=root,
    )
    if current_implementation_hash != str(provenance["implementation_sha256"]):
        raise ValueError("selection freeze decision implementation bytes changed")
    primary_evidence = payload["primary_development_evidence"]
    if not isinstance(primary_evidence, Mapping):
        raise ValueError("selection freeze primary evidence must be an object")
    _require_exact_keys(
        primary_evidence,
        _PRIMARY_EVIDENCE_FIELDS,
        "selection freeze primary evidence",
    )
    if (
        primary_evidence.get("metric") != "pooled_five_fold_oof_macro_f1"
        or primary_evidence.get("seed") != 2753
        or primary_evidence.get("folds") != [0, 1, 2, 3, 4]
        or primary_evidence.get("valid_development_rows") != 32_753
        or type(primary_evidence.get("pooled_oof_macro_f1")) not in (int, float)
        or not 0 <= float(primary_evidence["pooled_oof_macro_f1"]) <= 1
        or not isinstance(primary_evidence.get("run_ids"), list)
        or len(primary_evidence["run_ids"]) != 5
        or len(set(primary_evidence["run_ids"])) != 5
        or any(
            len(str(primary_evidence[field])) != 64
            for field in ("config_semantic_sha256", "implementation_sha256")
        )
    ):
        raise ValueError("selection freeze changed the primary development boundary")
    selected_config_path = _verify_declaration(
        primary_evidence.get("config"),
        project_root=root,
        name="selection freeze selected config",
    )
    selected_config = _load_json_object(selected_config_path, "selection freeze selected config")
    selected_canonical_config = (
        load_i2_config(selected_config_path).to_dict()
        if selected_candidate == "I2"
        else load_experiment_config(selected_config_path).to_dict()
    )
    if (
        canonical_sha256(selected_canonical_config) != primary_evidence["config_semantic_sha256"]
        or selected_config.get("experiment_id") != selected_model["experiment_id"]
        or selected_config.get("model_family") != selected_model["model_family"]
        or selected_config.get("loss_id") != primary_evidence["loss_id"]
        or dict(selected_config.get("data", {})).get("image_size") != primary_evidence["image_size"]
        or dict(selected_config.get("data", {})).get("augmentation")
        != primary_evidence["augmentation"]
    ):
        raise ValueError("selection freeze selected config disagrees with model metadata")
    stability_evidence = payload["stability_evidence"]
    if not isinstance(stability_evidence, Mapping):
        raise ValueError("selection freeze stability evidence must be an object")
    _require_exact_keys(
        stability_evidence,
        _STABILITY_EVIDENCE_FIELDS,
        "selection freeze stability evidence",
    )
    if (
        stability_evidence.get("seed") != 2026
        or stability_evidence.get("experiment_id")
        != _CANDIDATE_STABILITY_EXPERIMENTS[selected_candidate]
        or type(stability_evidence.get("pooled_oof_macro_f1")) not in (int, float)
        or not 0 <= float(stability_evidence["pooled_oof_macro_f1"]) <= 1
        or stability_evidence.get("two_seeds_cover_all_randomness") is not False
    ):
        raise ValueError("selection freeze changed the stability boundary")
    return payload, resolved


def load_verified_ultimate_judgement_manifest(
    path: str | Path = DEFAULT_EVIDENCE_DIRECTORY / "manifest.json",
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    """Verify the complete G7 evidence boundary without rewriting frozen bytes."""
    root = Path(project_root)
    resolved = _resolve_evidence_path(path, project_root=root)
    manifest = _load_json_object(resolved, "ultimate judgement manifest")
    _require_exact_keys(manifest, _ULTIMATE_MANIFEST_FIELDS, "ultimate judgement manifest")
    identity = {
        "schema_version": "1.0.0",
        "gate": "G7-ULTIMATE-JUDGEMENT",
        "decision_status": "closed",
        "analysis_role": "development_evidence_model_selection_and_pre_holdout_freeze",
        "candidate_selection_affected": True,
        "ultimate_winner_frozen": True,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "git_dirty": False,
        "labels": list(SEASON_LABELS),
    }
    mismatches = [name for name, expected in identity.items() if manifest[name] != expected]
    if mismatches:
        raise ValueError(f"ultimate judgement manifest boundary changed: {mismatches}")
    selected_candidate = str(manifest["selected_candidate"])
    if (
        selected_candidate not in _CANDIDATE_PRIMARY_EXPERIMENTS
        or manifest["selected_experiment_id"] != _CANDIDATE_PRIMARY_EXPERIMENTS[selected_candidate]
    ):
        raise ValueError("ultimate judgement manifest changed the audited winner identity")
    for field in ("implementation_sha256", "runtime_sha256"):
        if len(str(manifest[field])) != 64:
            raise ValueError(f"ultimate judgement manifest has invalid {field}")
    if len(str(manifest["git_commit"])) != 40:
        raise ValueError("ultimate judgement manifest has invalid git_commit")
    artifacts = {
        str(name): _verify_declaration(
            declaration,
            project_root=root,
            name=f"ultimate judgement artifact {name}",
        )
        for name, declaration in dict(manifest["artifacts"]).items()
    }
    if set(artifacts) != _ULTIMATE_ARTIFACTS:
        raise ValueError("ultimate judgement manifest changed its artifact set")
    analysis_config = _verify_declaration(
        manifest["analysis_config"],
        project_root=root,
        name="ultimate judgement analysis config",
    )
    if analysis_config.name != ULTIMATE_JUDGEMENT_CONFIG_PATH.name:
        raise ValueError("ultimate judgement manifest changed its analysis config")
    for section in ("canonical_inputs", "upstream_manifests"):
        declarations = manifest[section]
        if not isinstance(declarations, Mapping) or not declarations:
            raise ValueError(f"ultimate judgement manifest lacks {section}")
        expected_fields = (
            _CANONICAL_INPUT_FIELDS if section == "canonical_inputs" else _UPSTREAM_MANIFEST_FIELDS
        )
        _require_exact_keys(
            declarations,
            expected_fields,
            f"ultimate judgement manifest {section}",
        )
        for name, declaration in declarations.items():
            _verify_declaration(
                declaration,
                project_root=root,
                name=f"ultimate judgement {section}.{name}",
            )
    runtime = _load_json_object(artifacts["runtime"], "ultimate judgement runtime")
    if canonical_sha256(runtime) != str(manifest["runtime_sha256"]):
        raise ValueError("ultimate judgement runtime semantic hash changed")
    freeze, resolved_freeze = load_verified_selection_freeze(
        artifacts["selection_freeze"],
        project_root=root,
    )
    if (
        resolved_freeze != artifacts["selection_freeze"]
        or freeze["decision_provenance"]["git_commit"] != manifest["git_commit"]
        or freeze["decision_provenance"]["implementation_sha256"]
        != manifest["implementation_sha256"]
        or freeze["selected_model"]["candidate"] != selected_candidate
        or freeze["selected_model"]["experiment_id"] != manifest["selected_experiment_id"]
        or freeze["decision_evidence"]["analysis_config"] != manifest["analysis_config"]
        or freeze["canonical_inputs"] != manifest["canonical_inputs"]
        or freeze["upstream_manifests"] != manifest["upstream_manifests"]
    ):
        raise ValueError("ultimate judgement manifest and selection freeze disagree")
    for name in ("scorecard", "rejected_alternatives", "decision"):
        if freeze["decision_evidence"][name] != manifest["artifacts"][name]:
            raise ValueError(f"ultimate judgement manifest and selection freeze disagree on {name}")
    return manifest, resolved, artifacts


def _build_ultimate_judgement_evidence_unlocked(
    *,
    project_root: str | Path = ROOT,
    analysis_config_path: str | Path = ULTIMATE_JUDGEMENT_CONFIG_PATH,
    gradcam_manifest_path: str | Path = DEFAULT_GRADCAM_MANIFEST,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    freeze_path: str | Path = TASK2_SELECTION_FREEZE_JSON,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build G7 while the caller holds the exclusive evidence lock."""
    root = Path(project_root)
    evidence_root = _resolve_evidence_path(evidence_directory, project_root=root)
    resolved_freeze = _resolve_evidence_path(freeze_path, project_root=root)
    config_path = _resolve_evidence_path(analysis_config_path, project_root=root)
    requested_gradcam = _resolve_evidence_path(gradcam_manifest_path, project_root=root)
    manifest_path = evidence_root / "manifest.json"
    if manifest_path.exists() and not resolved_freeze.exists():
        raise ValueError("ultimate judgement manifest exists without its immutable freeze")
    recovery_freeze: dict[str, Any] | None = None
    if resolved_freeze.exists():
        recovery_freeze, verified_freeze = load_verified_selection_freeze(
            resolved_freeze,
            project_root=root,
        )
        if verified_freeze != resolved_freeze:
            raise ValueError("ultimate judgement retry resolved a different freeze")
        requested_inputs = {
            "analysis_config": _portable_declaration(config_path, root=root),
            "gradcam": _portable_declaration(requested_gradcam, root=root),
        }
        frozen_inputs = {
            "analysis_config": recovery_freeze["decision_evidence"]["analysis_config"],
            "gradcam": recovery_freeze["upstream_manifests"]["gradcam"],
        }
        if requested_inputs != frozen_inputs:
            raise ValueError("immutable retry changed the G7 config or Grad-CAM evidence input")
        frozen_timestamp = str(recovery_freeze["decision_recorded_at_utc"])
        if recorded_at_utc is not None and _validate_timestamp(recorded_at_utc) != frozen_timestamp:
            raise ValueError("immutable retry changed the recorded decision timestamp")
        recorded_at_utc = frozen_timestamp
        if manifest_path.exists():
            existing_manifest, verified_manifest, artifacts = (
                load_verified_ultimate_judgement_manifest(
                    manifest_path,
                    project_root=root,
                )
            )
            if artifacts["selection_freeze"] != resolved_freeze:
                raise ValueError("immutable retry supplied a different freeze path")
            result = dict(existing_manifest)
            result["manifest_path"] = str(verified_manifest)
            result["manifest_sha256"] = compute_sha256(verified_manifest)
            result["selection_freeze_sha256"] = compute_sha256(resolved_freeze)
            return result
    spec = load_ultimate_judgement_spec(config_path)
    gradcam, resolved_gradcam, gradcam_sections = _load_verified_gradcam_boundary(
        requested_gradcam,
        project_root=root,
    )
    bootstrap, resolved_bootstrap, bootstrap_sections = load_verified_bootstrap_manifest(
        gradcam_sections["upstream"]["bootstrap"],
        project_root=root,
    )
    calibration_manifest, resolved_calibration, calibration_sections = (
        load_verified_calibration_manifest(
            bootstrap_sections["calibration_manifest"],
            project_root=root,
        )
    )
    robustness_manifest, resolved_robustness, robustness_sections = (
        load_verified_robustness_manifest(
            calibration_sections["robustness_manifest"],
            project_root=root,
        )
    )
    slice_manifest, resolved_slice, slice_sections = load_verified_slice_manifest(
        robustness_sections["slice_manifest"],
        project_root=root,
    )
    stability_manifest, resolved_stability, stability_sections = load_verified_stability_manifest(
        slice_sections["stability_manifest"],
        project_root=root,
    )
    declared_paths = {
        "bootstrap": resolved_bootstrap,
        "calibration": resolved_calibration,
        "robustness": resolved_robustness,
        "slice": resolved_slice,
        "stability": resolved_stability,
    }
    for name, resolved in declared_paths.items():
        if resolved != gradcam_sections["upstream"][name]:
            raise ValueError(f"Grad-CAM and verified {name} manifest paths disagree")
    canonical_hashes = {
        name: {
            compute_sha256(gradcam_sections["canonical_inputs"][name]),
            compute_sha256(bootstrap_sections["canonical_inputs"][name]),
            compute_sha256(calibration_sections["canonical_inputs"][name]),
            compute_sha256(robustness_sections["canonical_inputs"][name]),
            compute_sha256(slice_sections["canonical_inputs"][name]),
        }
        for name in ("splits", "label_maps")
    }
    if any(len(values) != 1 for values in canonical_hashes.values()):
        raise ValueError("G5-G7 evidence disagrees on canonical input bytes")

    stability = pd.read_csv(stability_sections["artifacts"]["seed_stability"])
    registry = pd.read_csv(stability_sections["artifacts"]["registry_snapshot"])
    slice_deltas = pd.read_csv(slice_sections["artifacts"]["candidate_slice_deltas"])
    robustness = pd.read_csv(robustness_sections["artifacts"]["candidate_comparison"])
    deployment_cost = pd.read_csv(robustness_sections["artifacts"]["deployment_cost"])
    calibration = pd.read_csv(calibration_sections["artifacts"]["calibration_summary"])
    deployment_temperatures = pd.read_csv(
        calibration_sections["artifacts"]["deployment_temperatures"]
    )
    interval_summary = pd.read_csv(bootstrap_sections["artifacts"]["interval_summary"])
    scorecard = build_candidate_scorecard(
        spec=spec,
        stability=stability,
        slice_deltas=slice_deltas,
        robustness=robustness,
        deployment_cost=deployment_cost,
        calibration=calibration,
    )
    scorecard, decision = apply_ultimate_judgement(
        scorecard,
        interval_summary=interval_summary,
        robustness=robustness,
        spec=spec,
    )
    selected_spec = spec.candidate_for_role(
        str(scorecard.loc[scorecard["selected"], "role"].item())
    )
    if selected_spec.candidate != decision["selected_candidate"]:
        raise ValueError("scorecard and final decision disagree")
    rejected = build_rejected_alternatives(project_root=root, scorecard=scorecard)
    registry_identity = _selected_registry_identity(registry, selected=selected_spec)
    selected_summary = _one_row(
        stability,
        stability["candidate"].eq(selected_spec.candidate)
        & stability["experiment_id"].eq(selected_spec.experiment_id_seed_2753)
        & pd.to_numeric(stability["seed"], errors="raise").eq(2753),
        "selected primary stability summary",
    )
    if registry_identity["median_best_epoch"] != int(selected_summary["median_best_epoch"]):
        raise ValueError("selected median best epoch differs between registry and summary")
    selected_config = stability_sections["input_configs"][selected_spec.experiment_id_seed_2753]
    selected_manifest_path = stability_sections["input_manifests"][
        selected_spec.experiment_id_seed_2753
    ]
    selected_manifest = _load_json_object(selected_manifest_path, "selected experiment manifest")
    if selected_manifest.get("run_ids") != registry_identity["run_ids"]:
        raise ValueError("selected experiment manifest and registry run IDs disagree")
    selected_config_payload = _load_json_object(selected_config, "selected experiment config")
    temperature = _one_row(
        deployment_temperatures,
        deployment_temperatures["candidate"].eq(selected_spec.candidate)
        & deployment_temperatures["experiment_id"].eq(selected_spec.experiment_id_seed_2753),
        "selected deployment temperature",
    )
    if (
        str(temperature["fit_scope"]) != spec.calibration_source
        or str(temperature["purpose"]) != spec.calibration_purpose
        or _strict_bool(
            temperature["evaluation_claim_allowed"],
            "deployment temperature evaluation_claim_allowed",
        )
    ):
        raise ValueError("selected deployment temperature crossed its claim boundary")
    label_maps = _load_json_object(gradcam_sections["canonical_inputs"]["label_maps"], "label maps")
    if tuple(label_maps["season"]["classes"]) != tuple(SEASON_LABELS):
        raise ValueError("selection freeze changed canonical Season class order")

    git_state = capture_git_state(root)
    if git_state.get("dirty") is not False or len(str(git_state.get("commit"))) != 40:
        raise ValueError("ultimate judgement evidence requires a clean Git commit")
    implementation_files = verify_implementation_at_head(
        *ULTIMATE_JUDGEMENT_IMPLEMENTATION_PATHS,
        root=root,
    )
    selection_implementation_hash = implementation_sha256(
        *ULTIMATE_JUDGEMENT_IMPLEMENTATION_PATHS,
        root=root,
    )
    runtime = capture_runtime()
    paths = {
        "scorecard": evidence_root / "scorecard.csv",
        "rejected_alternatives": evidence_root / "rejected_alternatives.csv",
        "runtime": evidence_root / "runtime.json",
        "decision": evidence_root / "decision.json",
    }
    atomic_write_csv(paths["scorecard"], scorecard)
    atomic_write_csv(paths["rejected_alternatives"], rejected)
    atomic_write_json(paths["runtime"], runtime)

    if recorded_at_utc is not None:
        timestamp = _validate_timestamp(recorded_at_utc)
    else:
        timestamp = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    decision["decision_recorded_at_utc"] = timestamp
    atomic_write_json(paths["decision"], decision)

    upstream_manifests = {
        "gradcam": _portable_declaration(resolved_gradcam, root=root),
        "bootstrap": _portable_declaration(resolved_bootstrap, root=root),
        "calibration": _portable_declaration(resolved_calibration, root=root),
        "robustness": _portable_declaration(resolved_robustness, root=root),
        "slices": _portable_declaration(resolved_slice, root=root),
        "stability": _portable_declaration(resolved_stability, root=root),
        "selected_experiment": _portable_declaration(selected_manifest_path, root=root),
    }
    current_provenance = {
        "git_commit": str(git_state["commit"]),
        "git_dirty": False,
        "implementation_sha256": selection_implementation_hash,
        "implementation_files_at_head": list(implementation_files),
    }
    frozen_provenance = (
        dict(recovery_freeze["decision_provenance"])
        if recovery_freeze is not None
        else current_provenance
    )
    freeze_payload = {
        "schema_version": "1.0.0",
        "freeze_id": f"task2-season-{selected_spec.candidate.lower()}-development-v1",
        "gate": "G7-ULTIMATE-JUDGEMENT",
        "status": "frozen",
        "analysis_role": "development_evidence_model_selection_and_pre_holdout_freeze",
        "decision_recorded_at_utc": timestamp,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "selected_model": {
            "candidate": selected_spec.candidate,
            "experiment_id": selected_spec.experiment_id_seed_2753,
            "model_family": registry_identity["model_family"],
            "scratch": True,
            "weights": None,
            "benchmark_only": False,
            "final_eligible": True,
            "inference_inputs": ["image"],
            "auxiliary_training_target": (
                "articleType" if selected_spec.candidate == "I2" else None
            ),
            "auxiliary_target_used_at_inference": False,
            "parameter_count": registry_identity["parameter_count"],
        },
        "primary_development_evidence": {
            "metric": spec.primary_metric,
            "pooled_oof_macro_f1": float(selected_summary["pooled_macro_f1"]),
            "seed": 2753,
            "folds": [0, 1, 2, 3, 4],
            "valid_development_rows": 32_753,
            "run_ids": registry_identity["run_ids"],
            "config": _portable_declaration(selected_config, root=root),
            "config_semantic_sha256": registry_identity["config_sha256"],
            "implementation_sha256": registry_identity["implementation_sha256"],
            "transform_id": registry_identity["transform_id"],
            "image_size": selected_config_payload["data"]["image_size"],
            "augmentation": selected_config_payload["data"]["augmentation"],
            "loss_id": registry_identity["loss_id"],
        },
        "stability_evidence": {
            "seed": 2026,
            "experiment_id": selected_spec.experiment_id_seed_2026,
            "pooled_oof_macro_f1": float(
                _one_row(
                    stability,
                    stability["candidate"].eq(selected_spec.candidate)
                    & pd.to_numeric(stability["seed"], errors="raise").eq(2026),
                    "selected second-seed summary",
                )["pooled_macro_f1"]
            ),
            "two_seeds_cover_all_randomness": False,
        },
        "selection_checks": decision["selection_checks"],
        "refit_rule": {
            "dataset": spec.refit_dataset,
            "seed": spec.refit_seed,
            "epochs": registry_identity["median_best_epoch"],
            "epoch_rule": spec.refit_epoch_rule,
            "source_best_epochs": registry_identity["best_epochs"],
            "validation_or_holdout_early_stopping": False,
            "normalisation_scope": spec.refit_normalisation_scope,
            "season_class_weights": None,
            "article_type_missing_labels": (
                "masked_not_dropped"
                if selected_spec.candidate == "I2"
                else "not_applicable_no_auxiliary_target"
            ),
            "checkpoint_rule": "save_the_declared_final_epoch_state",
        },
        "calibration": {
            "method": "scalar_temperature_scaling",
            "temperature": float(temperature["temperature"]),
            "fit_rows": int(temperature["fit_rows"]),
            "fit_scope": str(temperature["fit_scope"]),
            "purpose": str(temperature["purpose"]),
            "evaluation_claim_allowed": False,
            "app_review_threshold": None,
        },
        "canonical_inputs": {
            name: _portable_declaration(path, root=root)
            for name, path in gradcam_sections["canonical_inputs"].items()
        },
        "decision_evidence": {
            "analysis_config": _portable_declaration(config_path, root=root),
            "scorecard": _portable_declaration(paths["scorecard"], root=root),
            "rejected_alternatives": _portable_declaration(
                paths["rejected_alternatives"], root=root
            ),
            "decision": _portable_declaration(paths["decision"], root=root),
        },
        "decision_provenance": dict(frozen_provenance),
        "upstream_manifests": upstream_manifests,
        "limitations": [
            "Two fixed seeds do not cover all optimisation randomness.",
            "Brightness 0.85 severely damages macro-F1 and Spring recall.",
            "ArticleType conflict remains difficult despite an average I2 improvement.",
            "Grad-CAM tags are non-causal review hypotheses, not failure prevalence.",
            "No app review threshold is justified without a business error cost.",
        ],
        "immutability": {
            "different_payload_overwrite_allowed": False,
            "identical_retry_allowed": True,
            "model_change_after_holdout_allowed": False,
        },
    }
    _write_immutable_freeze(resolved_freeze, freeze_payload)

    artifacts = {
        **{name: _portable_declaration(path, root=root) for name, path in paths.items()},
        "selection_freeze": _portable_declaration(resolved_freeze, root=root),
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G7-ULTIMATE-JUDGEMENT",
        "decision_status": "closed",
        "analysis_role": "development_evidence_model_selection_and_pre_holdout_freeze",
        "selected_candidate": decision["selected_candidate"],
        "selected_experiment_id": decision["selected_experiment_id"],
        "candidate_selection_affected": True,
        "ultimate_winner_frozen": True,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "labels": list(SEASON_LABELS),
        "git_commit": str(frozen_provenance["git_commit"]),
        "git_dirty": False,
        "analysis_config": _portable_declaration(config_path, root=root),
        "implementation_sha256": selection_implementation_hash,
        "implementation_files_at_head": list(implementation_files),
        "runtime_sha256": canonical_sha256(runtime),
        "canonical_inputs": freeze_payload["canonical_inputs"],
        "upstream_manifests": upstream_manifests,
        "artifacts": artifacts,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    manifest["selection_freeze_sha256"] = compute_sha256(resolved_freeze)
    return manifest


def build_ultimate_judgement_evidence(
    *,
    project_root: str | Path = ROOT,
    analysis_config_path: str | Path = ULTIMATE_JUDGEMENT_CONFIG_PATH,
    gradcam_manifest_path: str | Path = DEFAULT_GRADCAM_MANIFEST,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    freeze_path: str | Path = TASK2_SELECTION_FREEZE_JSON,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the final development-only scorecard and immutable selection record."""
    root = Path(project_root)
    evidence_root = _resolve_evidence_path(evidence_directory, project_root=root)
    with _exclusive_build_lock(evidence_root / ".ultimate_judgement.lock"):
        return _build_ultimate_judgement_evidence_unlocked(
            project_root=root,
            analysis_config_path=analysis_config_path,
            gradcam_manifest_path=gradcam_manifest_path,
            evidence_directory=evidence_root,
            freeze_path=freeze_path,
            recorded_at_utc=recorded_at_utc,
        )


__all__ = [
    "build_candidate_scorecard",
    "build_rejected_alternatives",
    "build_ultimate_judgement_evidence",
    "apply_ultimate_judgement",
    "load_ultimate_judgement_spec",
    "load_verified_selection_freeze",
    "load_verified_ultimate_judgement_manifest",
    "ULTIMATE_JUDGEMENT_CONFIG_PATH",
    "ULTIMATE_JUDGEMENT_IMPLEMENTATION_PATHS",
]
