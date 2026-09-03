"""Hash-linked Grad-CAM contact sheets and real-ID failure evidence."""

from __future__ import annotations

import gc
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.images import transform_image_with_mask
from fashion.data.torch import build_image_transform
from fashion.task2.bootstrap_evidence import (
    BOOTSTRAP_IMPLEMENTATION_PATHS,
    load_verified_calibration_manifest,
)
from fashion.task2.calibration_evidence import load_verified_robustness_manifest
from fashion.task2.evidence import _portable_artifact_path, _resolve_evidence_path
from fashion.task2.gradcam import (
    GRADCAM_CONFIG_PATH,
    GradCamReviewSpec,
    audit_attention_location,
    build_failure_taxonomy,
    compute_gradcam,
    load_gradcam_review_spec,
    plot_gradcam_contact_sheet,
    select_gradcam_examples,
    summarise_failure_taxonomy,
)
from fashion.task2.robustness import (
    build_robustness_model,
    canonical_validation_frames,
    fold_stats_from_history,
    load_robustness_checkpoint,
)
from fashion.task2.robustness_evidence import load_verified_slice_manifest
from fashion.task2.slice_evidence import (
    load_candidate_oof_packs,
    load_declared_config_hashes,
    load_verified_stability_manifest,
)
from fashion.task2.slices import build_slice_assignments, load_slice_analysis_spec
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import implementation_sha256, verify_implementation_at_head
from fashion.train.metrics import SEASON_LABELS
from fashion.train.reproducibility import capture_git_state, capture_runtime

DEFAULT_BOOTSTRAP_MANIFEST = Path("results/evidence/task2/paired_bootstrap/manifest.json")
DEFAULT_EVIDENCE_DIRECTORY = TASK2_EVIDENCE_DIR / "gradcam_failure_review"
DEFAULT_TEMPORARY_DIRECTORY = Path("tmp/task2/gradcam")
GRADCAM_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *BOOTSTRAP_IMPLEMENTATION_PATHS,
            "src/fashion/data/images.py",
            "src/fashion/data/torch.py",
            "src/fashion/models/season.py",
            "src/fashion/task2/robustness.py",
            "src/fashion/task2/gradcam.py",
            "src/fashion/task2/gradcam_evidence.py",
        )
    )
)

_BOOTSTRAP_MANIFEST_FIELDS = {
    "analysis_config",
    "analysis_role",
    "artifacts",
    "calibration_manifest",
    "candidate_selection_affected",
    "canonical_inputs",
    "decision_status",
    "gate",
    "git_commit",
    "git_dirty",
    "holdout_opened",
    "implementation_files_at_head",
    "implementation_sha256",
    "input_predictions",
    "labels",
    "new_candidates_allowed",
    "random_seed_generalisability_claim_allowed",
    "runtime_sha256",
    "schema_version",
    "slice_assignment_sha256",
    "slice_manifest",
    "stability_coverage_sha256",
    "stability_manifest",
    "temporary_bootstrap_draws",
    "ultimate_winner_frozen",
}
_BOOTSTRAP_ARTIFACTS = {
    "decision",
    "group_audit",
    "interval_summary",
    "observed_metrics",
    "paired_bootstrap_figure",
    "registry_snapshot",
    "runtime",
}


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


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


def _verify_declaration_mapping(
    declarations: Any,
    *,
    project_root: Path,
    name: str,
) -> dict[str, Path]:
    if not isinstance(declarations, Mapping) or not declarations:
        raise ValueError(f"{name} declarations must be a non-empty object")
    return {
        str(key): _verify_declaration(
            value,
            project_root=project_root,
            name=f"{name}.{key}",
        )
        for key, value in declarations.items()
    }


def load_verified_bootstrap_manifest(
    path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST,
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Verify the paired-bootstrap boundary before Grad-CAM consumes it."""
    root = Path(project_root)
    resolved = _resolve_evidence_path(path, project_root=root)
    if not resolved.is_file():
        raise ValueError(f"paired-bootstrap evidence manifest does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("paired-bootstrap manifest must be an object")
    _require_exact_keys(manifest, _BOOTSTRAP_MANIFEST_FIELDS, "paired-bootstrap manifest")
    identity = {
        "schema_version": "1.0.0",
        "gate": "G6-PAIRED-BOOTSTRAP",
        "decision_status": "closed",
        "analysis_role": "development_oof_fitted_pair_uncertainty_only",
        "candidate_selection_affected": False,
        "new_candidates_allowed": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "random_seed_generalisability_claim_allowed": False,
        "git_dirty": False,
        "labels": list(SEASON_LABELS),
    }
    mismatches = [name for name, expected in identity.items() if manifest.get(name) != expected]
    if mismatches:
        raise ValueError(f"paired-bootstrap manifest boundary changed: {mismatches}")
    for field in (
        "implementation_sha256",
        "runtime_sha256",
        "slice_assignment_sha256",
        "stability_coverage_sha256",
    ):
        if len(str(manifest[field])) != 64:
            raise ValueError(f"paired-bootstrap manifest has invalid {field}")
    if len(str(manifest["git_commit"])) != 40:
        raise ValueError("paired-bootstrap manifest has invalid git_commit")
    implementation_files = manifest["implementation_files_at_head"]
    if not isinstance(implementation_files, list) or not implementation_files:
        raise ValueError("paired-bootstrap manifest lacks implementation files")

    artifacts = _verify_declaration_mapping(
        manifest["artifacts"],
        project_root=root,
        name="paired-bootstrap artifacts",
    )
    if set(artifacts) != _BOOTSTRAP_ARTIFACTS:
        raise ValueError("paired-bootstrap manifest changed its artifact set")
    canonical_inputs = _verify_declaration_mapping(
        manifest["canonical_inputs"],
        project_root=root,
        name="paired-bootstrap canonical inputs",
    )
    if set(canonical_inputs) != {"splits", "label_maps"}:
        raise ValueError("paired-bootstrap manifest changed its canonical inputs")
    predictions = _verify_declaration_mapping(
        manifest["input_predictions"],
        project_root=root,
        name="paired-bootstrap predictions",
    )
    if len(predictions) != 20:
        raise ValueError("paired-bootstrap manifest must retain twenty OOF files")
    analysis_config = _verify_declaration(
        manifest["analysis_config"],
        project_root=root,
        name="paired-bootstrap analysis config",
    )
    calibration_manifest = _verify_declaration(
        manifest["calibration_manifest"],
        project_root=root,
        name="paired-bootstrap calibration manifest",
    )
    slice_manifest = _verify_declaration(
        manifest["slice_manifest"],
        project_root=root,
        name="paired-bootstrap slice manifest",
    )
    stability_manifest = _verify_declaration(
        manifest["stability_manifest"],
        project_root=root,
        name="paired-bootstrap stability manifest",
    )

    temporary = manifest["temporary_bootstrap_draws"]
    if not isinstance(temporary, Mapping):
        raise ValueError("paired-bootstrap temporary draws declaration must be an object")
    _require_exact_keys(
        temporary,
        {"path", "row_count", "rows_per_comparison", "sha256"},
        "paired-bootstrap temporary draws",
    )
    temporary_path = _resolve_evidence_path(str(temporary["path"]), project_root=root)
    verify_artifact(temporary_path, str(temporary["sha256"]))
    if int(temporary["row_count"]) != 20_000 or set(temporary["rows_per_comparison"].values()) != {
        10_000
    }:
        raise ValueError("paired-bootstrap temporary draw counts changed")

    with artifacts["decision"].open(encoding="utf-8") as handle:
        decision = json.load(handle)
    decision_identity = {
        "gate": "G6-PAIRED-BOOTSTRAP",
        "decision_status": "closed",
        "analysis_role": "development_oof_fitted_pair_uncertainty_only",
        "current_candidate": "I2",
        "candidate_selection_affected": False,
        "new_candidates_allowed": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "random_seed_generalisability_claim_allowed": False,
    }
    decision_mismatches = [
        name for name, expected in decision_identity.items() if decision.get(name) != expected
    ]
    if decision_mismatches:
        raise ValueError(f"paired-bootstrap decision boundary changed: {decision_mismatches}")
    outcomes = decision.get("pair_outcomes", {})
    if set(outcomes) != {"primary_interval", "stability_sensitivity"} or any(
        value.get("interval_contains_zero") is not False for value in outcomes.values()
    ):
        raise ValueError("paired-bootstrap positive fitted-pair result changed")
    with artifacts["runtime"].open(encoding="utf-8") as handle:
        runtime = json.load(handle)
    if canonical_sha256(runtime) != str(manifest["runtime_sha256"]):
        raise ValueError("paired-bootstrap runtime semantic hash changed")
    return (
        dict(manifest),
        resolved,
        {
            "artifacts": artifacts,
            "canonical_inputs": canonical_inputs,
            "input_predictions": predictions,
            "analysis_config": analysis_config,
            "calibration_manifest": calibration_manifest,
            "slice_manifest": slice_manifest,
            "stability_manifest": stability_manifest,
            "temporary_bootstrap_draws": temporary_path,
            "decision": decision,
        },
    )


def _portable_declaration(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "path": _portable_artifact_path(path, fallback_root=root),
        "sha256": compute_sha256(path),
    }


def _validated_selected_images(selected: pd.DataFrame) -> pd.DataFrame:
    """Return one immutable image declaration per selected development ID."""
    required = {"id", "path", "image_sha256"}
    if not required <= set(selected.columns):
        raise ValueError("Grad-CAM selected rows lack image provenance")
    declarations = selected.loc[:, ["id", "path", "image_sha256"]].copy()
    versions = declarations.groupby("id", observed=True, dropna=False).agg(
        path_versions=("path", "nunique"),
        sha256_versions=("image_sha256", "nunique"),
    )
    if versions.ne(1).any(axis=None):
        conflicted = versions.loc[versions.ne(1).any(axis=1)].index.astype(str).tolist()
        raise ValueError(f"Grad-CAM selected image provenance conflicts for IDs: {conflicted}")
    return declarations.drop_duplicates("id").sort_values("id", kind="stable")


def _primary_packs(packs: list[Any], spec: GradCamReviewSpec) -> list[Any]:
    indexed = {(pack.candidate, pack.experiment_id, int(pack.seed)): pack for pack in packs}
    selected = []
    for candidate in spec.candidates:
        identity = (candidate.candidate, candidate.experiment_id, candidate.seed)
        if identity not in indexed:
            raise ValueError(f"Grad-CAM lacks frozen primary OOF pack: {identity}")
        selected.append(indexed[identity])
    return selected


def _build_review_context(
    packs: list[Any],
    development: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    metadata = development.loc[
        :,
        ["id", "path", "sha256", "articleType", "year", "productDisplayName"],
    ].rename(columns={"sha256": "image_sha256"})
    assignment_columns = [
        "id",
        "article_type_shortcut",
        "acquisition_year",
        "file_size_quartile",
        "product_family_size",
        "image_mode",
    ]
    context_base = metadata.merge(
        assignments.loc[:, assignment_columns],
        on="id",
        how="left",
        validate="one_to_one",
    )
    parts = []
    for pack in packs:
        if not {"id", "run_id"} <= set(pack.oof.columns):
            raise ValueError(f"{pack.candidate} OOF pack lacks run-level review context")
        part = pack.oof.loc[:, ["id", "run_id"]].merge(
            context_base,
            on="id",
            how="left",
            validate="one_to_one",
        )
        part.insert(0, "candidate", pack.candidate)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _sort_review_rows(frame: pd.DataFrame, spec: GradCamReviewSpec) -> pd.DataFrame:
    output = frame.copy()
    output["_candidate_order"] = output["candidate"].map(
        {candidate.candidate: index for index, candidate in enumerate(spec.candidates)}
    )
    output["_label_order"] = output["true_label"].map(
        {label: index for index, label in enumerate(spec.labels)}
    )
    output["_group_order"] = output["selection_group"].map(
        {group: index for index, group in enumerate(spec.correctness_groups)}
    )
    if output[["_candidate_order", "_label_order", "_group_order"]].isna().any().any():
        raise ValueError("Grad-CAM review rows contain an undeclared ordering value")
    return output.sort_values(
        ["_candidate_order", "_label_order", "_group_order", "selection_rank", "id"],
        kind="stable",
    ).drop(columns=["_candidate_order", "_label_order", "_group_order"])


def build_gradcam_decision(
    reviewed: pd.DataFrame,
    taxonomy: pd.DataFrame,
    summary: pd.DataFrame,
    spec: GradCamReviewSpec,
) -> dict[str, Any]:
    """Close the review gate without converting diagnostic attention into causality."""
    expected_reviewed = (
        len(spec.candidates)
        * len(spec.labels)
        * len(spec.correctness_groups)
        * spec.examples_per_group
    )
    expected_errors = len(spec.candidates) * len(spec.labels) * spec.examples_per_group
    if len(reviewed) != expected_reviewed or len(taxonomy) != expected_errors:
        raise ValueError("Grad-CAM decision received incomplete selected-example evidence")
    if reviewed["probability_max_absolute_delta"].max() > spec.probability_tolerance:
        raise ValueError("Grad-CAM decision detected OOF probability drift")
    if taxonomy["causal_claim_allowed"].astype(bool).any():
        raise ValueError("Grad-CAM decision cannot contain a causal failure claim")
    selected_counts = {
        f"{candidate}/{label}/{group}": int(count)
        for (candidate, label, group), count in reviewed.groupby(
            ["candidate", "true_label", "selection_group"], observed=True
        )
        .size()
        .items()
    }
    attention_flags = {
        str(candidate): int(group["attention_review_flag"].astype(bool).sum())
        for candidate, group in reviewed.groupby("candidate", observed=True)
    }
    taxonomy_counts = {
        str(candidate): {
            str(row["primary_failure_hypothesis"]): int(row["selected_error_count"])
            for row in group.to_dict(orient="records")
        }
        for candidate, group in summary.groupby("candidate", observed=True)
    }
    return {
        "schema_version": "1.0.0",
        "gate": "G6-GRADCAM-FAILURE-REVIEW",
        "decision_status": "closed",
        "analysis_role": "development_oof_explainability_and_failure_diagnosis_only",
        "current_candidate": "I2",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "causal_failure_claim_allowed": False,
        "human_causal_adjudication_completed": False,
        "selected_example_counts": selected_counts,
        "zero_heatmap_count": int(reviewed["zero_heatmap"].astype(bool).sum()),
        "attention_review_flag_count_by_candidate": attention_flags,
        "primary_failure_hypothesis_counts": taxonomy_counts,
        "maximum_probability_reconciliation_delta": float(
            reviewed["probability_max_absolute_delta"].max()
        ),
        "interpretation_rule": (
            "Grad-CAM shows where the fitted model was sensitive for one predicted class. "
            "It does not prove why the prediction occurred or that a highlighted region is causal."
        ),
        "selection_boundary": (
            "High-confidence correct and incorrect rows expose strong beliefs and severe mistakes; "
            "their taxonomy counts are not population prevalence estimates."
        ),
        "metadata_boundary": (
            "ArticleType, year, file size, family size, and image mode are review context only and "
            "never enter model inference."
        ),
        "limitations": [
            (
                "Grad-CAM is a coarse post-hoc diagnostic and can be unstable across "
                "layers or methods."
            ),
            "The non-white foreground mask is a proxy that can miss white products.",
            "Only fixed high-confidence primary-seed examples are reviewed.",
            "Diagnostic tags are hypotheses and require human interpretation.",
            "Holdout remains sealed and this gate does not freeze the winner.",
        ],
        "next_question": "Apply the frozen scorecard and record the ultimate Season judgement.",
    }


def _save_heatmaps(path: Path, heatmaps: np.ndarray) -> Path:
    buffer = io.BytesIO()
    np.save(buffer, heatmaps.astype(np.float32, copy=False), allow_pickle=False)
    return atomic_write_bytes(path, buffer.getvalue())


def build_gradcam_failure_evidence(
    *,
    project_root: str | Path = ROOT,
    analysis_config_path: str | Path = GRADCAM_CONFIG_PATH,
    bootstrap_manifest_path: str | Path = DEFAULT_BOOTSTRAP_MANIFEST,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
    temporary_directory: str | Path = DEFAULT_TEMPORARY_DIRECTORY,
) -> dict[str, Any]:
    """Build fixed Grad-CAM contact sheets from verified primary OOF checkpoints."""
    root = Path(project_root)
    config_path = _resolve_evidence_path(analysis_config_path, project_root=root)
    spec = load_gradcam_review_spec(config_path, project_root=root)
    bootstrap, resolved_bootstrap, bootstrap_sections = load_verified_bootstrap_manifest(
        bootstrap_manifest_path,
        project_root=root,
    )
    calibration, resolved_calibration, calibration_sections = load_verified_calibration_manifest(
        bootstrap_sections["calibration_manifest"],
        project_root=root,
    )
    _, resolved_robustness, _ = load_verified_robustness_manifest(
        calibration_sections["robustness_manifest"],
        project_root=root,
    )
    slice_manifest, resolved_slice, slice_sections = load_verified_slice_manifest(
        calibration_sections["slice_manifest"],
        project_root=root,
    )
    stability, resolved_stability, stability_sections = load_verified_stability_manifest(
        slice_sections["stability_manifest"],
        project_root=root,
    )

    coverage_hash = str(bootstrap["stability_coverage_sha256"])
    if {
        coverage_hash,
        str(calibration["stability_coverage_sha256"]),
        str(slice_manifest["coverage_sha256"]),
        str(stability["coverage_sha256"]),
    } != {coverage_hash}:
        raise ValueError("Grad-CAM upstream OOF coverage hashes disagree")
    if compute_sha256(resolved_slice) != compute_sha256(bootstrap_sections["slice_manifest"]):
        raise ValueError("Grad-CAM bootstrap and calibration slice manifests disagree")
    if compute_sha256(resolved_stability) != compute_sha256(
        bootstrap_sections["stability_manifest"]
    ):
        raise ValueError("Grad-CAM bootstrap and slice stability manifests disagree")
    if compute_sha256(resolved_robustness) != compute_sha256(
        calibration_sections["robustness_manifest"]
    ):
        raise ValueError("Grad-CAM robustness manifest declaration changed")
    for name in ("splits", "label_maps"):
        boundary_hashes = {
            compute_sha256(bootstrap_sections["canonical_inputs"][name]),
            compute_sha256(calibration_sections["canonical_inputs"][name]),
            compute_sha256(slice_sections["canonical_inputs"][name]),
        }
        if len(boundary_hashes) != 1:
            raise ValueError(f"Grad-CAM upstream boundaries disagree on {name}")

    split_path = bootstrap_sections["canonical_inputs"]["splits"]
    label_map_path = bootstrap_sections["canonical_inputs"]["label_maps"]
    splits = load_splits(split_path)
    with label_map_path.open(encoding="utf-8") as handle:
        label_maps = json.load(handle)
    season_map = label_maps.get("season", {})
    expected_label_map = {label: index for index, label in enumerate(SEASON_LABELS)}
    if (
        tuple(season_map.get("classes", ())) != tuple(SEASON_LABELS)
        or dict(season_map.get("label_to_index", {})) != expected_label_map
    ):
        raise ValueError("Grad-CAM changed the canonical Season label map")
    article_type_classes = int(label_maps.get("articleType", {}).get("num_classes", -1))
    if article_type_classes != 124:
        raise ValueError("I2 Grad-CAM requires the canonical ArticleType class count")
    development = get_samples(splits, partition="development", target="season").reset_index(
        drop=True
    )
    if len(development) != spec.expected_row_count:
        raise ValueError("Grad-CAM development Season row count changed")
    expected_ids = development["id"].astype(int).tolist()
    expected_targets = dict(
        zip(development["id"].astype(int), development["season"].astype(str), strict=True)
    )
    protected_ids = set(
        splits.loc[splits["partition"].isin(["holdout", "quarantine"]), "id"].astype(int)
    )

    registry = pd.read_csv(
        slice_sections["artifacts"]["registry_snapshot"],
        dtype=str,
        keep_default_na=False,
    )
    stability_summary = pd.read_csv(stability_sections["artifacts"]["seed_stability"])
    config_hashes = load_declared_config_hashes(stability_sections["input_configs"])
    slice_spec = load_slice_analysis_spec(slice_sections["analysis_config"])
    all_packs, coverage_by_experiment, prediction_artifacts = load_candidate_oof_packs(
        registry,
        slice_spec,
        project_root=root,
        expected_ids=expected_ids,
        expected_targets=expected_targets,
        protected_ids=protected_ids,
        split_sha256=compute_sha256(split_path),
        label_map_sha256=compute_sha256(label_map_path),
        config_sha256_by_experiment=config_hashes,
        stability_summary=stability_summary,
    )
    if {canonical_sha256(value) for value in coverage_by_experiment.values()} != {coverage_hash}:
        raise ValueError("Grad-CAM OOF coverage differs from the frozen boundary")
    if prediction_artifacts != slice_manifest["input_predictions"]:
        raise ValueError("Grad-CAM prediction bytes differ from the slice boundary")
    packs = _primary_packs(all_packs, spec)

    slice_bundle = build_slice_assignments(splits, slice_spec)
    if slice_bundle.assignment_sha256 != str(bootstrap["slice_assignment_sha256"]):
        raise ValueError("Grad-CAM slice assignments differ from the bootstrap boundary")
    review_context = _build_review_context(packs, development, slice_bundle.assignments)
    calibrated_oof = pd.read_csv(calibration_sections["temporary_calibrated_oof"])
    selected = select_gradcam_examples(
        calibrated_oof,
        review_context,
        spec,
        expected_ids=expected_ids,
        expected_targets=expected_targets,
        protected_ids=protected_ids,
    )
    selected = _sort_review_rows(selected, spec).reset_index(drop=True)

    git_state = capture_git_state(root)
    if git_state.get("commit") is None:
        raise ValueError("Grad-CAM evidence requires a Git commit")
    if git_state.get("dirty") is not False:
        raise ValueError("Grad-CAM evidence requires a clean tracked Git worktree")
    implementation_files = verify_implementation_at_head(
        *GRADCAM_IMPLEMENTATION_PATHS,
        root=root,
    )
    implementation_hash = implementation_sha256(
        *GRADCAM_IMPLEMENTATION_PATHS,
        root=root,
    )
    runtime = capture_runtime()

    frames = canonical_validation_frames(splits)
    pack_by_candidate = {pack.candidate: pack for pack in packs}
    raw_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for pack in packs:
        for row in pack.oof.to_dict(orient="records"):
            key = (pack.candidate, int(row["id"]))
            if key in raw_lookup:
                raise ValueError(f"Grad-CAM raw OOF duplicate: {key}")
            raw_lookup[key] = row

    reviewed_records: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    overlays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(spec.torch_threads)
    try:
        for candidate_spec in spec.candidates:
            pack = pack_by_candidate[candidate_spec.candidate]
            candidate_selected = selected.loc[selected["candidate"].eq(candidate_spec.candidate)]
            for fold, fold_selected in candidate_selected.groupby("fold", sort=True):
                fold = int(fold)
                registry_rows = pack.registry.loc[
                    pd.to_numeric(pack.registry["fold"], errors="raise").astype(int).eq(fold)
                ]
                if len(registry_rows) != 1:
                    raise ValueError(
                        f"Grad-CAM {candidate_spec.candidate}/fold {fold} registry drift"
                    )
                registry_row = registry_rows.iloc[0].to_dict()
                if set(fold_selected["run_id"].astype(str)) != {str(registry_row["run_id"])}:
                    raise ValueError("Grad-CAM selected OOF rows changed their fold run ID")
                training, validation = frames[fold]
                if not set(fold_selected["id"].astype(int)) <= set(validation["id"].astype(int)):
                    raise ValueError("Grad-CAM selected an ID outside its checkpoint fold")
                stats = fold_stats_from_history(
                    registry_row,
                    project_root=root,
                    expected_training_ids=training["id"].astype(int).tolist(),
                )
                if stats.image_size != (80, 60) or stats.validation_fold != fold:
                    raise ValueError("Grad-CAM fold preprocessing changed the frozen P0 geometry")
                transform = build_image_transform(stats, training=False)
                model = build_robustness_model(
                    candidate_spec.candidate,
                    article_type_classes=article_type_classes,
                )
                model, checkpoint_metadata, checkpoint_path = load_robustness_checkpoint(
                    model,
                    registry_row,
                    project_root=root,
                )
                model = model.to("cpu").float()
                checkpoint_rows.append(
                    {
                        "candidate": candidate_spec.candidate,
                        "experiment_id": candidate_spec.experiment_id,
                        "seed": candidate_spec.seed,
                        "fold": fold,
                        "run_id": str(registry_row["run_id"]),
                        "checkpoint_path": _portable_artifact_path(
                            checkpoint_path,
                            fallback_root=root,
                        ),
                        "checkpoint_sha256": str(checkpoint_metadata["checkpoint_sha256"]),
                        "checkpoint_bytes": int(checkpoint_metadata["checkpoint_bytes"]),
                        "history_sha256": str(registry_row["history_sha256"]),
                        "fold_stats_sha256": canonical_sha256(stats.to_dict()),
                        "selected_examples": len(fold_selected),
                    }
                )
                for selected_row in fold_selected.sort_values(
                    ["true_label", "selection_group", "selection_rank", "id"],
                    kind="stable",
                ).to_dict(orient="records"):
                    identifier = int(selected_row["id"])
                    relative_path = Path(str(selected_row["path"]))
                    if relative_path.is_absolute():
                        raise ValueError("Grad-CAM image paths must remain project-relative")
                    image_path = (root / relative_path).resolve()
                    try:
                        image_path.relative_to(root.resolve())
                    except ValueError as error:
                        raise ValueError("Grad-CAM image path escaped the project root") from error
                    verify_artifact(image_path, str(selected_row["image_sha256"]))
                    image_tensor = transform(image_path).unsqueeze(0).to("cpu", dtype=torch.float32)
                    target_index = expected_label_map[str(selected_row["predicted_label"])]
                    result = compute_gradcam(
                        model,
                        image_tensor,
                        candidate=candidate_spec.candidate,
                        target_index=target_index,
                    )
                    raw_row = raw_lookup[(candidate_spec.candidate, identifier)]
                    raw_probabilities = np.asarray(
                        [raw_row[f"prob_{label}"] for label in spec.labels],
                        dtype=float,
                    )
                    probability_delta = float(
                        np.max(np.abs(result.probabilities.astype(float) - raw_probabilities))
                    )
                    if probability_delta > spec.probability_tolerance:
                        raise ValueError(
                            "Grad-CAM probability drift for "
                            f"{candidate_spec.candidate}/{identifier}: "
                            f"{probability_delta}"
                        )
                    recomputed_label = spec.labels[int(result.probabilities.argmax())]
                    if (
                        recomputed_label != str(selected_row["predicted_label"])
                        or str(raw_row["y_pred"]) != recomputed_label
                    ):
                        raise ValueError("Grad-CAM checkpoint changed the frozen OOF prediction")
                    with Image.open(image_path) as source:
                        display_source = ImageOps.exif_transpose(source).convert("RGB")
                        rgb, content_mask = transform_image_with_mask(
                            display_source,
                            image_size=stats.image_size,
                            normalize_range=True,
                        )
                    attention = audit_attention_location(
                        result.heatmap,
                        rgb,
                        content_mask,
                        spec.attention,
                    )
                    key = (candidate_spec.candidate, identifier)
                    if key in overlays:
                        raise ValueError(f"Grad-CAM duplicate selected overlay: {key}")
                    overlays[key] = (rgb, result.heatmap)
                    true_index = expected_label_map[str(selected_row["true_label"])]
                    reviewed_records.append(
                        {
                            **selected_row,
                            "gradcam_target_label": recomputed_label,
                            "gradcam_target_index": target_index,
                            "raw_confidence": float(result.probabilities.max()),
                            "raw_true_probability": float(result.probabilities[true_index]),
                            "probability_max_absolute_delta": probability_delta,
                            "activation_channels": int(result.activation_shape[1]),
                            "activation_height": int(result.activation_shape[2]),
                            "activation_width": int(result.activation_shape[3]),
                            "checkpoint_sha256": str(checkpoint_metadata["checkpoint_sha256"]),
                            "fold_stats_sha256": canonical_sha256(stats.to_dict()),
                            **attention,
                        }
                    )
                del model
                gc.collect()
    finally:
        torch.set_num_threads(previous_threads)

    reviewed = _sort_review_rows(pd.DataFrame(reviewed_records), spec).reset_index(drop=True)
    expected_overlay_keys = set(zip(reviewed["candidate"], reviewed["id"].astype(int), strict=True))
    if len(reviewed) != len(selected) or set(overlays) != expected_overlay_keys:
        raise ValueError("Grad-CAM reviewed-example coverage differs from selection")
    taxonomy = build_failure_taxonomy(reviewed, spec)
    taxonomy_summary = summarise_failure_taxonomy(taxonomy)
    decision = build_gradcam_decision(reviewed, taxonomy, taxonomy_summary, spec)
    checkpoint_audit = pd.DataFrame(checkpoint_rows).sort_values(
        ["candidate", "fold"],
        kind="stable",
    )
    registry_snapshot = pd.concat(
        [pack.registry.assign(candidate=pack.candidate) for pack in packs],
        ignore_index=True,
    ).sort_values(["candidate", "fold", "run_id"], kind="stable")

    evidence_root = _resolve_evidence_path(evidence_directory, project_root=root)
    figure_root = _resolve_evidence_path(figure_directory, project_root=root)
    temporary_root = _resolve_evidence_path(temporary_directory, project_root=root)
    paths = {
        "selected_examples": evidence_root / "selected_examples.csv",
        "attention_metrics": evidence_root / "attention_metrics.csv",
        "failure_taxonomy": evidence_root / "failure_taxonomy.csv",
        "failure_taxonomy_summary": evidence_root / "failure_taxonomy_summary.csv",
        "heatmap_index": evidence_root / "heatmap_index.csv",
        "checkpoint_audit": evidence_root / "checkpoint_audit.csv",
        "registry_snapshot": evidence_root / "registry_snapshot.csv",
        "runtime": evidence_root / "runtime.json",
        "decision": evidence_root / "decision.json",
        "c2_contact_sheet": figure_root / "gradcam_c2_contact_sheet.png",
        "i2_contact_sheet": figure_root / "gradcam_i2_contact_sheet.png",
    }
    heatmap_rows = []
    heatmap_arrays = []
    for index, row in enumerate(reviewed.to_dict(orient="records")):
        key = (str(row["candidate"]), int(row["id"]))
        heatmap_arrays.append(overlays[key][1])
        heatmap_rows.append(
            {
                "array_index": index,
                "candidate": key[0],
                "id": key[1],
                "fold": int(row["fold"]),
                "gradcam_target_label": str(row["gradcam_target_label"]),
            }
        )
    heatmap_stack = np.stack(heatmap_arrays).astype(np.float32, copy=False)
    heatmap_index = pd.DataFrame(heatmap_rows)

    atomic_write_csv(paths["selected_examples"], selected)
    atomic_write_csv(paths["attention_metrics"], reviewed)
    atomic_write_csv(paths["failure_taxonomy"], taxonomy)
    atomic_write_csv(paths["failure_taxonomy_summary"], taxonomy_summary)
    atomic_write_csv(paths["heatmap_index"], heatmap_index)
    atomic_write_csv(paths["checkpoint_audit"], checkpoint_audit)
    atomic_write_csv(paths["registry_snapshot"], registry_snapshot)
    atomic_write_json(paths["runtime"], runtime)
    atomic_write_json(paths["decision"], decision)
    plot_gradcam_contact_sheet(
        reviewed,
        overlays,
        paths["c2_contact_sheet"],
        candidate="C2",
        spec=spec,
    )
    plot_gradcam_contact_sheet(
        reviewed,
        overlays,
        paths["i2_contact_sheet"],
        candidate="I2",
        spec=spec,
    )
    heatmap_path = _save_heatmaps(temporary_root / "heatmaps.npy", heatmap_stack)

    artifacts = {name: _portable_declaration(path, root=root) for name, path in paths.items()}
    used_checkpoint_inputs = {
        str(row["run_id"]): {
            "path": str(row["checkpoint_path"]),
            "sha256": str(row["checkpoint_sha256"]),
        }
        for row in checkpoint_rows
    }
    selected_images = _validated_selected_images(selected)
    input_images = {
        str(int(row.id)): {
            "path": str(row.path).replace("\\", "/"),
            "sha256": str(row.image_sha256),
        }
        for row in selected_images.itertuples(index=False)
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G6-GRADCAM-FAILURE-REVIEW",
        "decision_status": "closed",
        "analysis_role": "development_oof_explainability_and_failure_diagnosis_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "causal_failure_claim_allowed": False,
        "labels": list(SEASON_LABELS),
        "git_commit": str(git_state["commit"]),
        "git_dirty": False,
        "analysis_config": _portable_declaration(config_path, root=root),
        "bootstrap_manifest": _portable_declaration(resolved_bootstrap, root=root),
        "calibration_manifest": _portable_declaration(resolved_calibration, root=root),
        "robustness_manifest": _portable_declaration(resolved_robustness, root=root),
        "slice_manifest": _portable_declaration(resolved_slice, root=root),
        "stability_manifest": _portable_declaration(resolved_stability, root=root),
        "stability_coverage_sha256": coverage_hash,
        "slice_assignment_sha256": str(slice_bundle.assignment_sha256),
        "canonical_inputs": {
            "splits": _portable_declaration(split_path, root=root),
            "label_maps": _portable_declaration(label_map_path, root=root),
        },
        "implementation_sha256": implementation_hash,
        "implementation_files_at_head": list(implementation_files),
        "runtime_sha256": canonical_sha256(runtime),
        "input_predictions": {
            run_id: {
                "path": _portable_artifact_path(path, fallback_root=root),
                "sha256": compute_sha256(path),
            }
            for run_id, path in sorted(calibration_sections["input_predictions"].items())
        },
        "input_checkpoints": used_checkpoint_inputs,
        "input_images": input_images,
        "temporary_heatmaps": {
            "path": _portable_artifact_path(heatmap_path, fallback_root=root),
            "sha256": compute_sha256(heatmap_path),
            "shape": list(heatmap_stack.shape),
            "dtype": str(heatmap_stack.dtype),
            "index_sha256": compute_sha256(paths["heatmap_index"]),
        },
        "artifacts": artifacts,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


__all__ = [
    "DEFAULT_BOOTSTRAP_MANIFEST",
    "DEFAULT_EVIDENCE_DIRECTORY",
    "DEFAULT_TEMPORARY_DIRECTORY",
    "GRADCAM_IMPLEMENTATION_PATHS",
    "build_gradcam_decision",
    "build_gradcam_failure_evidence",
    "load_verified_bootstrap_manifest",
]
