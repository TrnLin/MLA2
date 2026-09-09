"""Deterministic Grad-CAM selection, attention audit, and failure diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from fashion.config import ROOT
from fashion.train.metrics import SEASON_LABELS

GRADCAM_CONFIG_PATH = Path("configs/task2/g6_gradcam_failure_review.json")
EXPECTED_CANDIDATES = (
    ("C2", "g3-c2-t0-resnet18", 2753),
    ("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
)
REVIEW_CONTEXT_COLUMNS = (
    "run_id",
    "path",
    "image_sha256",
    "articleType",
    "year",
    "productDisplayName",
    "article_type_shortcut",
    "acquisition_year",
    "file_size_quartile",
    "product_family_size",
    "image_mode",
)


@dataclass(frozen=True)
class GradCamCandidate:
    """One frozen primary-seed finalist reviewed by Grad-CAM."""

    candidate: str
    experiment_id: str
    seed: int


@dataclass(frozen=True)
class AttentionAuditSpec:
    """Heuristic spatial checks that remain non-causal review signals."""

    border_band_fraction: float
    foreground_white_threshold: int
    border_attention_lift_review_threshold: float
    foreground_attention_lift_review_threshold: float


@dataclass(frozen=True)
class FailureTaxonomySpec:
    """Allowed diagnostic tags and their deterministic primary-tag priority."""

    allowed_tags: tuple[str, ...]
    priority: tuple[str, ...]


@dataclass(frozen=True)
class GradCamReviewSpec:
    """Complete frozen G6 Grad-CAM and failure-review protocol."""

    candidates: tuple[GradCamCandidate, ...]
    expected_row_count: int
    labels: tuple[str, ...]
    correctness_groups: tuple[str, ...]
    examples_per_group: int
    probability_source: str
    ranking_rule: str
    device: str
    torch_threads: int
    probability_tolerance: float
    attention: AttentionAuditSpec
    taxonomy: FailureTaxonomySpec
    columns: int
    colormap: str
    overlay_alpha: float


@dataclass(frozen=True)
class GradCamComputation:
    """One predicted-class Grad-CAM result before rendering."""

    heatmap: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    target_index: int
    activation_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...]
    zero_heatmap: bool


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _require_mapping(value: Any, scope: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{scope} must be an object")
    return value


def load_gradcam_review_spec(
    path: str | Path = GRADCAM_CONFIG_PATH,
    *,
    project_root: str | Path = ROOT,
) -> GradCamReviewSpec:
    """Load the exact predeclared Grad-CAM protocol and reject silent drift."""
    candidate_path = Path(path)
    resolved = (
        candidate_path if candidate_path.is_absolute() else Path(project_root) / candidate_path
    )
    with resolved.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Grad-CAM config must be an object")
    _require_exact_keys(
        raw,
        {
            "analysis_id",
            "attention_audit",
            "candidate_experiments",
            "expected_row_count",
            "failure_taxonomy",
            "gradcam",
            "labels",
            "schema_version",
            "selection",
            "stage",
            "target",
            "visualisation",
            "warnings",
        },
        "Grad-CAM config",
    )
    identity = {
        "schema_version": "1.0.0",
        "analysis_id": "g6-deterministic-gradcam-failure-review",
        "stage": "g6_explainability_and_failure_analysis",
        "target": "season",
        "expected_row_count": 32_753,
    }
    mismatches = [name for name, expected in identity.items() if raw.get(name) != expected]
    if mismatches:
        raise ValueError(f"Grad-CAM config identity changed: {mismatches}")
    if tuple(raw["labels"]) != tuple(SEASON_LABELS):
        raise ValueError("Grad-CAM config changed the canonical Season label order")

    candidate_rows = raw["candidate_experiments"]
    if not isinstance(candidate_rows, list):
        raise ValueError("Grad-CAM candidate_experiments must be a list")
    candidates: list[GradCamCandidate] = []
    for index, value in enumerate(candidate_rows):
        row = _require_mapping(value, f"Grad-CAM candidate {index}")
        _require_exact_keys(row, {"candidate", "experiment_id", "seed"}, "Grad-CAM candidate")
        candidates.append(
            GradCamCandidate(
                candidate=str(row["candidate"]),
                experiment_id=str(row["experiment_id"]),
                seed=int(row["seed"]),
            )
        )
    observed_candidates = tuple(
        (candidate.candidate, candidate.experiment_id, candidate.seed) for candidate in candidates
    )
    if observed_candidates != EXPECTED_CANDIDATES:
        raise ValueError("Grad-CAM config changed the frozen candidate order or identity")

    selection = _require_mapping(raw["selection"], "Grad-CAM selection")
    _require_exact_keys(
        selection,
        {
            "correctness_groups",
            "examples_per_true_class_and_group",
            "probability_source",
            "ranking_rule",
            "selection_scope",
        },
        "Grad-CAM selection",
    )
    selection_identity = {
        "correctness_groups": ["correct", "incorrect"],
        "examples_per_true_class_and_group": 3,
        "probability_source": "five_fold_cross_fitted_calibrated_oof",
        "ranking_rule": "descending_top_label_confidence_then_ascending_id",
        "selection_scope": "primary_seed_oof_only",
    }
    if any(selection.get(name) != expected for name, expected in selection_identity.items()):
        raise ValueError("Grad-CAM selection protocol changed")

    protocol = _require_mapping(raw["gradcam"], "Grad-CAM protocol")
    _require_exact_keys(
        protocol,
        {
            "device",
            "heatmap_normalisation",
            "interpolation",
            "model_mode",
            "probability_reconciliation_max_absolute_delta",
            "target_class",
            "target_layer",
            "torch_threads",
            "zero_heatmap_policy",
        },
        "Grad-CAM protocol",
    )
    required_protocol = {
        "device": "cpu",
        "heatmap_normalisation": "relu_then_per_image_min_max",
        "interpolation": "bilinear_align_corners_false",
        "model_mode": "eval_float32",
        "target_class": "frozen_oof_predicted_label",
        "target_layer": "model_gradcam_target_layer",
        "torch_threads": 1,
        "zero_heatmap_policy": "retain_and_flag",
    }
    if any(protocol.get(name) != expected for name, expected in required_protocol.items()):
        raise ValueError("Grad-CAM computation protocol changed")
    probability_tolerance = float(protocol["probability_reconciliation_max_absolute_delta"])
    if probability_tolerance != 0.0001:
        raise ValueError("Grad-CAM probability reconciliation tolerance changed")

    attention = _require_mapping(raw["attention_audit"], "Grad-CAM attention audit")
    _require_exact_keys(
        attention,
        {
            "border_attention_lift_review_threshold",
            "border_band_fraction",
            "foreground_attention_lift_review_threshold",
            "foreground_proxy",
            "foreground_white_threshold",
            "interpretation",
        },
        "Grad-CAM attention audit",
    )
    if (
        attention.get("foreground_proxy")
        != ("any_rgb_channel_below_white_threshold_inside_resized_content")
        or attention.get("interpretation") != "attention_location_diagnostic_not_causal_explanation"
    ):
        raise ValueError("Grad-CAM attention interpretation changed")
    attention_spec = AttentionAuditSpec(
        border_band_fraction=float(attention["border_band_fraction"]),
        foreground_white_threshold=int(attention["foreground_white_threshold"]),
        border_attention_lift_review_threshold=float(
            attention["border_attention_lift_review_threshold"]
        ),
        foreground_attention_lift_review_threshold=float(
            attention["foreground_attention_lift_review_threshold"]
        ),
    )
    if attention_spec != AttentionAuditSpec(
        border_band_fraction=0.15,
        foreground_white_threshold=245,
        border_attention_lift_review_threshold=1.25,
        foreground_attention_lift_review_threshold=0.75,
    ):
        raise ValueError("Grad-CAM attention thresholds changed")

    taxonomy = _require_mapping(raw["failure_taxonomy"], "Grad-CAM failure taxonomy")
    _require_exact_keys(
        taxonomy,
        {
            "allowed_diagnostic_tags",
            "causal_claim_allowed",
            "priority_for_primary_hypothesis",
        },
        "Grad-CAM failure taxonomy",
    )
    allowed_tags = tuple(str(value) for value in taxonomy["allowed_diagnostic_tags"])
    priority = tuple(str(value) for value in taxonomy["priority_for_primary_hypothesis"])
    expected_allowed_tags = (
        "label_ambiguity_requires_human_review",
        "weak_data_proxy",
        "article_type_shortcut_conflict",
        "transform_or_background_attention",
        "spring_imbalance_association",
        "model_limitation_or_unmeasured_cause",
    )
    expected_priority = expected_allowed_tags[1:]
    if taxonomy["causal_claim_allowed"] is not False:
        raise ValueError("Grad-CAM failure taxonomy cannot allow causal claims")
    if allowed_tags != expected_allowed_tags or priority != expected_priority:
        raise ValueError("Grad-CAM failure taxonomy tags or priority changed")

    visual = _require_mapping(raw["visualisation"], "Grad-CAM visualisation")
    _require_exact_keys(
        visual,
        {
            "class_order",
            "columns",
            "heatmap_colormap",
            "overlay_alpha",
            "row_order",
        },
        "Grad-CAM visualisation",
    )
    if tuple(visual["class_order"]) != tuple(SEASON_LABELS):
        raise ValueError("Grad-CAM visualisation changed the Season class order")
    if visual["row_order"] != "true_class_then_correct_before_incorrect":
        raise ValueError("Grad-CAM visualisation row order changed")
    columns = int(visual["columns"])
    overlay_alpha = float(visual["overlay_alpha"])
    if columns != 3 or str(visual["heatmap_colormap"]) != "magma" or overlay_alpha != 0.45:
        raise ValueError("Grad-CAM visualisation geometry changed")

    warnings = _require_mapping(raw["warnings"], "Grad-CAM warnings")
    expected_warnings = {
        "analysis_cannot_reopen_g5_model_selection",
        "gradcam_is_not_causal_proof",
        "high_confidence_selection_is_not_population_prevalence",
        "holdout_is_forbidden",
        "image_metadata_is_review_context_not_an_inference_feature",
        "primary_seed_only_because_g5_already_tests_random_seed_stability",
        "ultimate_winner_remains_unfrozen",
    }
    _require_exact_keys(warnings, expected_warnings, "Grad-CAM warnings")
    if any(warnings[name] is not True for name in expected_warnings):
        raise ValueError("every Grad-CAM safety warning must remain enabled")

    return GradCamReviewSpec(
        candidates=tuple(candidates),
        expected_row_count=int(raw["expected_row_count"]),
        labels=tuple(str(value) for value in raw["labels"]),
        correctness_groups=tuple(str(value) for value in selection["correctness_groups"]),
        examples_per_group=int(selection["examples_per_true_class_and_group"]),
        probability_source=str(selection["probability_source"]),
        ranking_rule=str(selection["ranking_rule"]),
        device=str(protocol["device"]),
        torch_threads=int(protocol["torch_threads"]),
        probability_tolerance=probability_tolerance,
        attention=attention_spec,
        taxonomy=FailureTaxonomySpec(allowed_tags=allowed_tags, priority=priority),
        columns=columns,
        colormap=str(visual["heatmap_colormap"]),
        overlay_alpha=overlay_alpha,
    )


def _normalise_ids(values: pd.Series, *, scope: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.equal(numeric.to_numpy(), np.floor(numeric)).all():
        raise ValueError(f"{scope} IDs must be finite integers")
    return numeric.astype("int64")


def select_gradcam_examples(
    calibrated_oof: pd.DataFrame,
    review_context: pd.DataFrame,
    spec: GradCamReviewSpec,
    *,
    expected_ids: Sequence[int],
    expected_targets: Mapping[int, str],
    protected_ids: Collection[int] = (),
) -> pd.DataFrame:
    """Select fixed high-belief correct and incorrect examples for each true class."""
    probability_columns = [f"prob_{label}" for label in spec.labels]
    required = {
        "candidate",
        "experiment_id",
        "seed",
        "id",
        "fold",
        "y_true",
        "y_pred",
        *probability_columns,
    }
    missing = sorted(required - set(calibrated_oof.columns))
    if missing:
        raise ValueError(f"calibrated OOF is missing columns: {missing}")
    context_required = {"candidate", "id", *REVIEW_CONTEXT_COLUMNS}
    context_missing = sorted(context_required - set(review_context.columns))
    if context_missing:
        raise ValueError(f"Grad-CAM review context is missing columns: {context_missing}")

    frame = calibrated_oof.copy()
    frame["id"] = _normalise_ids(frame["id"], scope="calibrated OOF")
    context = review_context.copy()
    context["id"] = _normalise_ids(context["id"], scope="Grad-CAM review context")
    if context.duplicated(["candidate", "id"]).any():
        raise ValueError("Grad-CAM review context has duplicate candidate/ID rows")
    expected_id_set = {int(value) for value in expected_ids}
    if len(expected_id_set) != spec.expected_row_count:
        raise ValueError("Grad-CAM expected ID coverage changed")
    protected = {int(value) for value in protected_ids}

    selected_parts: list[pd.DataFrame] = []
    expected_identity = {
        candidate.candidate: (candidate.experiment_id, candidate.seed)
        for candidate in spec.candidates
    }
    if set(frame["candidate"].astype(str)) != set(expected_identity):
        raise ValueError("calibrated OOF changed the Grad-CAM candidate set")
    for candidate in spec.candidates:
        subset = frame.loc[frame["candidate"].astype(str).eq(candidate.candidate)].copy()
        if len(subset) != spec.expected_row_count or subset["id"].duplicated().any():
            raise ValueError(f"{candidate.candidate} calibrated OOF coverage changed")
        if set(subset["id"].astype(int)) != expected_id_set:
            raise ValueError(f"{candidate.candidate} calibrated OOF IDs changed")
        if set(subset["id"].astype(int)) & protected:
            raise ValueError("protected IDs entered Grad-CAM selection")
        if set(subset["experiment_id"].astype(str)) != {candidate.experiment_id} or set(
            pd.to_numeric(subset["seed"], errors="raise").astype(int)
        ) != {candidate.seed}:
            raise ValueError(f"{candidate.candidate} calibrated OOF identity changed")
        subset["fold"] = pd.to_numeric(subset["fold"], errors="raise").astype(int)
        if not set(subset["fold"]) <= set(range(5)):
            raise ValueError("Grad-CAM calibrated OOF contains an invalid fold")
        truth = subset["id"].map(expected_targets)
        if truth.isna().any() or not truth.astype(str).equals(subset["y_true"].astype(str)):
            raise ValueError(f"{candidate.candidate} calibrated OOF truth changed")
        probabilities = subset.loc[:, probability_columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
            raise ValueError("Grad-CAM calibrated probabilities must be finite and non-negative")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("Grad-CAM calibrated probabilities must sum to one")
        predicted = np.asarray(spec.labels, dtype=object)[probabilities.argmax(axis=1)]
        if not np.array_equal(predicted.astype(str), subset["y_pred"].astype(str).to_numpy()):
            raise ValueError("Grad-CAM calibrated probabilities changed predicted labels")
        subset["calibrated_confidence"] = probabilities.max(axis=1)
        true_indices = subset["y_true"].map(
            {label: index for index, label in enumerate(spec.labels)}
        )
        if true_indices.isna().any():
            raise ValueError("Grad-CAM calibrated OOF contains an unknown true label")
        subset["calibrated_true_probability"] = probabilities[
            np.arange(len(subset)), true_indices.astype(int).to_numpy()
        ]
        subset["selection_group"] = np.where(
            subset["y_true"].astype(str).eq(subset["y_pred"].astype(str)),
            "correct",
            "incorrect",
        )
        for label in spec.labels:
            for group in spec.correctness_groups:
                group_rows = subset.loc[
                    subset["y_true"].astype(str).eq(label) & subset["selection_group"].eq(group)
                ].sort_values(
                    ["calibrated_confidence", "id"],
                    ascending=[False, True],
                    kind="stable",
                )
                if len(group_rows) < spec.examples_per_group:
                    raise ValueError(
                        f"not enough {candidate.candidate}/{label}/{group} Grad-CAM examples"
                    )
                chosen = group_rows.head(spec.examples_per_group).copy()
                chosen["selection_rank"] = np.arange(1, spec.examples_per_group + 1)
                selected_parts.append(chosen)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.merge(
        context,
        on=["candidate", "id"],
        how="left",
        validate="one_to_one",
    )
    if selected.loc[:, REVIEW_CONTEXT_COLUMNS].isna().any().any():
        raise ValueError("Grad-CAM selection lacks complete review context")
    expected_rows = (
        len(spec.candidates)
        * len(spec.labels)
        * len(spec.correctness_groups)
        * spec.examples_per_group
    )
    if len(selected) != expected_rows:
        raise ValueError("Grad-CAM selected-example row count changed")
    selected = selected.rename(columns={"y_true": "true_label", "y_pred": "predicted_label"})
    columns = [
        "candidate",
        "experiment_id",
        "seed",
        "true_label",
        "selection_group",
        "selection_rank",
        "id",
        "fold",
        "predicted_label",
        "calibrated_confidence",
        "calibrated_true_probability",
        *REVIEW_CONTEXT_COLUMNS,
    ]
    return selected.loc[:, columns].reset_index(drop=True)


def _season_logits(model: nn.Module, candidate: str, images: torch.Tensor) -> torch.Tensor:
    if candidate == "I2":
        predictor = getattr(model, "predict_season_logits", None)
        if not callable(predictor):
            raise TypeError("I2 Grad-CAM model lacks predict_season_logits")
        logits = predictor(images)
    elif candidate == "C2":
        logits = model(images)
    else:
        raise ValueError(f"unknown Grad-CAM candidate: {candidate}")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise TypeError("Grad-CAM model must return [batch, class] Season logits")
    return logits


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    *,
    candidate: str,
    target_index: int,
) -> GradCamComputation:
    """Compute predicted-class Grad-CAM at the model's declared spatial target layer."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM requires one [1, channels, height, width] image")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("Grad-CAM input must be a finite floating tensor")
    if target_index not in range(len(SEASON_LABELS)):
        raise ValueError("Grad-CAM target index is outside the Season label range")
    target_layer = getattr(model, "gradcam_target_layer", None)
    if not isinstance(target_layer, nn.Module):
        raise TypeError("Grad-CAM model must expose an nn.Module gradcam_target_layer")

    captured: list[torch.Tensor] = []

    def capture_activation(_module: nn.Module, _inputs: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError("Grad-CAM target layer must return a 4D tensor")
        captured.append(output)

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    handle = target_layer.register_forward_hook(capture_activation)
    try:
        with torch.enable_grad():
            logits = _season_logits(model, candidate, image)
            if logits.shape != (1, len(SEASON_LABELS)):
                raise ValueError("Grad-CAM logits changed the four-class output contract")
            if len(captured) != 1:
                raise ValueError("Grad-CAM target layer must execute exactly once")
            activation = captured[0]
            gradient = torch.autograd.grad(
                logits[0, target_index],
                activation,
                retain_graph=False,
                create_graph=False,
            )[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            heatmap = F.interpolate(
                heatmap,
                size=tuple(int(value) for value in image.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            heatmap = heatmap - heatmap.min()
            maximum = float(heatmap.max().detach().cpu())
            zero_heatmap = maximum <= torch.finfo(heatmap.dtype).eps
            if not zero_heatmap:
                heatmap = heatmap / maximum
            probabilities = torch.softmax(logits, dim=1)
    finally:
        handle.remove()
        model.train(was_training)
    if not torch.isfinite(gradient).all() or not torch.isfinite(heatmap).all():
        raise ValueError("Grad-CAM produced non-finite gradients or heatmap values")
    return GradCamComputation(
        heatmap=heatmap.detach().cpu().numpy().astype(np.float32, copy=False),
        logits=logits.detach().cpu().numpy()[0].astype(np.float32, copy=False),
        probabilities=probabilities.detach().cpu().numpy()[0].astype(np.float32, copy=False),
        target_index=target_index,
        activation_shape=tuple(int(value) for value in activation.shape),
        gradient_shape=tuple(int(value) for value in gradient.shape),
        zero_heatmap=zero_heatmap,
    )


def audit_attention_location(
    heatmap: np.ndarray,
    rgb_image: np.ndarray,
    content_mask: np.ndarray,
    spec: AttentionAuditSpec,
) -> dict[str, Any]:
    """Measure heatmap mass on a white-background proxy and a fixed border band."""
    heat = np.asarray(heatmap, dtype=np.float64)
    rgb = np.asarray(rgb_image)
    content = np.asarray(content_mask, dtype=bool)
    if heat.ndim != 2 or rgb.shape != (*heat.shape, 3) or content.shape != heat.shape:
        raise ValueError("Grad-CAM heatmap, RGB image, and content mask shapes disagree")
    if not np.isfinite(heat).all() or (heat < 0.0).any():
        raise ValueError("Grad-CAM heatmap mass must be finite and non-negative")
    if rgb.dtype.kind == "f":
        if not np.isfinite(rgb).all() or rgb.min() < 0.0 or rgb.max() > 1.0:
            raise ValueError("floating Grad-CAM RGB images must be in [0, 1]")
        threshold = spec.foreground_white_threshold / 255.0
    else:
        if rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("integer Grad-CAM RGB images must be 8-bit compatible")
        threshold = spec.foreground_white_threshold
    if not content.any():
        raise ValueError("Grad-CAM content mask cannot be empty")

    height, width = heat.shape
    border_rows = max(1, math.ceil(height * spec.border_band_fraction))
    border_columns = max(1, math.ceil(width * spec.border_band_fraction))
    border = np.zeros_like(content)
    border[:border_rows, :] = True
    border[-border_rows:, :] = True
    border[:, :border_columns] = True
    border[:, -border_columns:] = True
    border &= content
    foreground = np.any(rgb < threshold, axis=2) & content
    background = content & ~foreground
    padding = ~content
    total_mass = float(heat.sum())
    zero_heatmap = total_mass <= np.finfo(np.float64).eps

    def share(mask: np.ndarray) -> float:
        return 0.0 if zero_heatmap else float(heat[mask].sum() / total_mass)

    def area(mask: np.ndarray) -> float:
        return float(mask.mean())

    def lift(mass_share: float, area_fraction: float) -> float:
        if area_fraction <= 0.0:
            return float("nan")
        return mass_share / area_fraction

    border_share = share(border)
    foreground_share = share(foreground)
    background_share = share(background)
    padding_share = share(padding)
    border_area = area(border)
    foreground_area = area(foreground)
    background_area = area(background)
    border_lift = lift(border_share, border_area)
    foreground_lift = lift(foreground_share, foreground_area)
    attention_review_flag = bool(
        zero_heatmap
        or foreground_area <= 0.0
        or border_lift >= spec.border_attention_lift_review_threshold
        or (
            np.isfinite(foreground_lift)
            and foreground_lift <= spec.foreground_attention_lift_review_threshold
        )
    )
    return {
        "zero_heatmap": zero_heatmap,
        "heatmap_mass": total_mass,
        "border_attention_share": border_share,
        "foreground_attention_share": foreground_share,
        "background_attention_share": background_share,
        "padding_attention_share": padding_share,
        "border_area_fraction": border_area,
        "foreground_proxy_area_fraction": foreground_area,
        "background_proxy_area_fraction": background_area,
        "border_attention_lift": border_lift,
        "foreground_attention_lift": foreground_lift,
        "attention_review_flag": attention_review_flag,
    }


def build_failure_taxonomy(
    reviewed_examples: pd.DataFrame,
    spec: GradCamReviewSpec,
) -> pd.DataFrame:
    """Attach non-causal diagnostic tags to the selected incorrect real IDs."""
    required = {
        "candidate",
        "id",
        "true_label",
        "predicted_label",
        "calibrated_confidence",
        "article_type_shortcut",
        "file_size_quartile",
        "image_mode",
        "border_attention_lift",
        "foreground_attention_lift",
        "attention_review_flag",
        "selection_group",
        "run_id",
    }
    missing = sorted(required - set(reviewed_examples.columns))
    if missing:
        raise ValueError(f"Grad-CAM failure taxonomy is missing columns: {missing}")
    errors = reviewed_examples.loc[reviewed_examples["selection_group"].eq("incorrect")].copy()
    expected_rows = len(spec.candidates) * len(spec.labels) * spec.examples_per_group
    if len(errors) != expected_rows or errors.duplicated(["candidate", "id"]).any():
        raise ValueError("Grad-CAM failure taxonomy error coverage changed")

    records: list[dict[str, Any]] = []
    for row in errors.to_dict(orient="records"):
        flags: set[str] = {"label_ambiguity_requires_human_review"}
        if str(row["image_mode"]) != "rgb" or str(row["file_size_quartile"]) == "q1_smallest":
            flags.add("weak_data_proxy")
        if str(row["article_type_shortcut"]) == "conflict":
            flags.add("article_type_shortcut_conflict")
        attention_review_flag = row["attention_review_flag"]
        if not isinstance(attention_review_flag, (bool, np.bool_)):
            raise ValueError("Grad-CAM attention review flags must be boolean")
        if bool(attention_review_flag):
            flags.add("transform_or_background_attention")
        if str(row["true_label"]) == "Spring":
            flags.add("spring_imbalance_association")
        if flags == {"label_ambiguity_requires_human_review"}:
            flags.add("model_limitation_or_unmeasured_cause")
        if not flags <= set(spec.taxonomy.allowed_tags):
            raise ValueError("Grad-CAM failure taxonomy produced an undeclared tag")
        primary = next(tag for tag in spec.taxonomy.priority if tag in flags)
        ordered = [tag for tag in spec.taxonomy.allowed_tags if tag in flags]
        note = (
            f"confidence={float(row['calibrated_confidence']):.3f}; "
            f"shortcut={row['article_type_shortcut']}; mode={row['image_mode']}; "
            f"file_size={row['file_size_quartile']}; "
            f"border_lift={float(row['border_attention_lift']):.3f}; "
            f"foreground_lift={float(row['foreground_attention_lift']):.3f}"
        )
        records.append(
            {
                **row,
                "diagnostic_tags": ";".join(ordered),
                "primary_failure_hypothesis": primary,
                "causal_claim_allowed": False,
                "human_label_ambiguity_review_required": True,
                "review_note": note,
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(
            ["candidate", "true_label", "selection_rank", "id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def summarise_failure_taxonomy(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Count selected severe errors by candidate and primary diagnostic hypothesis."""
    required = {"candidate", "primary_failure_hypothesis", "id"}
    missing = sorted(required - set(taxonomy.columns))
    if missing:
        raise ValueError(f"Grad-CAM taxonomy summary is missing columns: {missing}")
    return (
        taxonomy.groupby(["candidate", "primary_failure_hypothesis"], observed=True)
        .agg(selected_error_count=("id", "size"))
        .reset_index()
        .sort_values(
            ["candidate", "selected_error_count", "primary_failure_hypothesis"],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def plot_gradcam_contact_sheet(
    selected_examples: pd.DataFrame,
    overlays: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    *,
    candidate: str,
    spec: GradCamReviewSpec,
) -> Path:
    """Render one fixed 8-by-3 transparent-overlay contact sheet per candidate."""
    subset = selected_examples.loc[selected_examples["candidate"].eq(candidate)].copy()
    expected_rows = len(spec.labels) * len(spec.correctness_groups) * spec.examples_per_group
    if len(subset) != expected_rows:
        raise ValueError(f"{candidate} Grad-CAM contact-sheet coverage changed")
    figure, axes = plt.subplots(
        len(spec.labels) * len(spec.correctness_groups),
        spec.columns,
        figsize=(12, 24),
        constrained_layout=True,
    )
    row_index = 0
    for label in spec.labels:
        for group in spec.correctness_groups:
            rows = subset.loc[
                subset["true_label"].eq(label) & subset["selection_group"].eq(group)
            ].sort_values("selection_rank", kind="stable")
            if len(rows) != spec.examples_per_group:
                raise ValueError(f"{candidate}/{label}/{group} contact-sheet rows changed")
            for column_index, row in enumerate(rows.to_dict(orient="records")):
                axis = axes[row_index, column_index]
                key = (candidate, int(row["id"]))
                if key not in overlays:
                    raise ValueError(f"Grad-CAM contact sheet lacks overlay for {key}")
                rgb, heatmap = overlays[key]
                if np.asarray(rgb).shape[:2] != np.asarray(heatmap).shape:
                    raise ValueError(f"Grad-CAM overlay shapes disagree for {key}")
                axis.imshow(rgb)
                axis.imshow(
                    heatmap,
                    cmap=spec.colormap,
                    alpha=spec.overlay_alpha,
                    vmin=0.0,
                    vmax=1.0,
                )
                axis.set_title(
                    f"ID {int(row['id'])} | {row['true_label']} -> {row['predicted_label']}\n"
                    f"cross-fitted p={float(row['calibrated_confidence']):.3f}",
                    fontsize=8,
                )
                axis.axis("off")
            axes[row_index, 0].text(
                -0.08,
                0.5,
                f"{label}\n{group}",
                transform=axes[row_index, 0].transAxes,
                fontsize=10,
                fontweight="bold",
                ha="right",
                va="center",
            )
            row_index += 1
    figure.suptitle(
        f"{candidate}: predicted-class Grad-CAM on fixed OOF examples\n"
        "Three highest-confidence correct and incorrect rows per true class; "
        "attention is diagnostic, not causal proof",
        fontsize=14,
        fontweight="bold",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


__all__ = [
    "AttentionAuditSpec",
    "EXPECTED_CANDIDATES",
    "FailureTaxonomySpec",
    "GRADCAM_CONFIG_PATH",
    "GradCamCandidate",
    "GradCamComputation",
    "GradCamReviewSpec",
    "REVIEW_CONTEXT_COLUMNS",
    "audit_attention_location",
    "build_failure_taxonomy",
    "compute_gradcam",
    "load_gradcam_review_spec",
    "plot_gradcam_contact_sheet",
    "select_gradcam_examples",
    "summarise_failure_taxonomy",
]
