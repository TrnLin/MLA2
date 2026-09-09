"""Deterministic robustness and deployment-cost probes for frozen Task 2 models."""

from __future__ import annotations

import gc
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader

from fashion.config import ROOT, TASK2_CONFIG_DIR
from fashion.data.dataset import get_cv_split, get_samples
from fashion.data.hashing import compute_sha256
from fashion.data.images import transform_image_with_mask
from fashion.data.torch import EncodedClassificationDataset, FoldImageStats
from fashion.models.season import (
    SeasonModelSpec,
    assert_final_model,
    build_multitask_season_model,
    build_season_model,
)
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics, validate_oof
from fashion.train.reproducibility import make_torch_generator, seed_worker

ROBUSTNESS_CONFIG_PATH = TASK2_CONFIG_DIR / "g6_robustness_cost.json"
ROBUSTNESS_IMPLEMENTATION_PATHS = (
    "src/fashion/task2/experiments.py",
    "src/fashion/config.py",
    "src/fashion/data/hashing.py",
    "src/fashion/data/metadata.py",
    "src/fashion/data/splits.py",
    "src/fashion/train/artifacts.py",
    "src/fashion/train/cache.py",
    "src/fashion/train/metrics.py",
    "src/fashion/train/registry.py",
    "src/fashion/data/dataset.py",
    "src/fashion/data/images.py",
    "src/fashion/data/torch.py",
    "src/fashion/models/season.py",
    "src/fashion/train/engine.py",
    "src/fashion/train/reproducibility.py",
    "src/fashion/data/multitask.py",
    "src/fashion/train/multitask.py",
    "src/fashion/task2/multitask.py",
    "src/fashion/task2/evidence.py",
    "src/fashion/task2/slices.py",
    "src/fashion/task2/slice_evidence.py",
    "src/fashion/task2/robustness.py",
    "src/fashion/task2/robustness_evidence.py",
)
EXPECTED_CANDIDATES = (
    ("C2", "g3-c2-t0-resnet18", 2753),
    ("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
)
EXPECTED_CONDITIONS = (
    "clean",
    "jpeg_quality_85",
    "brightness_0_85",
    "brightness_1_15",
    "gaussian_blur_radius_1",
)
ExecutionMode = Literal["run", "load", "run_or_load"]


@dataclass(frozen=True)
class RobustnessCandidate:
    """One frozen model/seed identity used by the paired probe."""

    candidate: str
    experiment_id: str
    seed: int


@dataclass(frozen=True)
class RobustnessCondition:
    """One deterministic image condition applied before fold normalisation."""

    condition: str
    kind: str
    quality: int | None = None
    subsampling: int | None = None
    factor: float | None = None
    radius: float | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class RobustnessProtocol:
    """Frozen all-development perturbation execution settings."""

    batch_size: int
    num_workers: int
    pin_memory: bool
    folds: tuple[int, ...]
    cache_directory: str
    clean_reference: str
    clean_min_prediction_agreement: float
    clean_max_probability_delta: float
    amp_matches_training_evaluation: bool
    evaluation_scope: str
    perturbation_order: str
    prediction_artifacts_are_temporary: bool


@dataclass(frozen=True)
class CostProtocol:
    """Frozen single-image latency and memory settings."""

    batch_size: int
    checkpoint_fold: int
    cpu_threads: int
    devices: tuple[str, ...]
    fixed_input_rule: str
    model_only_warmups: int
    model_only_repeats: int
    end_to_end_warmups: int
    end_to_end_repeats: int
    synchronise_accelerator_each_repeat: bool
    memory_metrics: tuple[str, ...]


@dataclass(frozen=True)
class RobustnessCostSpec:
    """Strictly validated G6 robustness/cost declaration."""

    analysis_id: str
    expected_row_count: int
    candidates: tuple[RobustnessCandidate, ...]
    conditions: tuple[RobustnessCondition, ...]
    robustness: RobustnessProtocol
    cost: CostProtocol
    material_macro_f1_degradation: float


@dataclass(frozen=True)
class ProbeFoldResult:
    """One executed or hash-verified perturbation/fold prediction artifact."""

    predictions: pd.DataFrame
    record: dict[str, Any]


@dataclass(frozen=True)
class RobustnessTables:
    """Pooled, fold, and paired robustness summaries."""

    pooled_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    candidate_comparison: pd.DataFrame


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _parse_condition(payload: Mapping[str, Any]) -> RobustnessCondition:
    name = str(payload.get("condition", ""))
    kind = str(payload.get("kind", ""))
    expected_by_kind = {
        "none": {"condition", "kind", "source"},
        "jpeg_reencode": {"condition", "kind", "quality", "subsampling"},
        "brightness": {"condition", "kind", "factor"},
        "gaussian_blur": {"condition", "kind", "radius"},
    }
    if kind not in expected_by_kind:
        raise ValueError(f"unknown robustness condition kind: {kind}")
    _require_exact_keys(payload, expected_by_kind[kind], f"condition {name}")
    return RobustnessCondition(
        condition=name,
        kind=kind,
        quality=int(payload["quality"]) if "quality" in payload else None,
        subsampling=int(payload["subsampling"]) if "subsampling" in payload else None,
        factor=float(payload["factor"]) if "factor" in payload else None,
        radius=float(payload["radius"]) if "radius" in payload else None,
        source=str(payload["source"]) if "source" in payload else None,
    )


def load_robustness_cost_spec(
    path: str | Path = ROBUSTNESS_CONFIG_PATH,
) -> RobustnessCostSpec:
    """Load and fail closed if any declared G6 robustness choice changed."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    required_top = {
        "schema_version",
        "analysis_id",
        "stage",
        "target",
        "labels",
        "expected_row_count",
        "candidate_experiments",
        "robustness_conditions",
        "robustness_protocol",
        "cost_protocol",
        "material_macro_f1_degradation",
        "warnings",
    }
    _require_exact_keys(payload, required_top, "robustness config")
    required_identity = {
        "schema_version": "1.0.0",
        "analysis_id": "g6-robustness-cost",
        "stage": "g6_robustness_cost_analysis",
        "target": "season",
        "expected_row_count": 32_753,
    }
    mismatches = [
        name for name, expected in required_identity.items() if payload.get(name) != expected
    ]
    if mismatches:
        raise ValueError(f"robustness identity changed: {mismatches}")
    if tuple(payload["labels"]) != tuple(SEASON_LABELS):
        raise ValueError("robustness changed the canonical Season order")

    candidates = tuple(
        RobustnessCandidate(
            candidate=str(row.get("candidate", "")),
            experiment_id=str(row.get("experiment_id", "")),
            seed=int(row.get("seed", -1)),
        )
        for row in payload["candidate_experiments"]
    )
    if tuple((row.candidate, row.experiment_id, row.seed) for row in candidates) != (
        EXPECTED_CANDIDATES
    ):
        raise ValueError("robustness changed the primary-seed candidate pair")
    conditions = tuple(_parse_condition(row) for row in payload["robustness_conditions"])
    if tuple(row.condition for row in conditions) != EXPECTED_CONDITIONS:
        raise ValueError("robustness changed the condition order")
    expected_condition_values = {
        "clean": {
            "kind": "none",
            "source": "re_inferred_and_verified_against_frozen_oof",
        },
        "jpeg_quality_85": {"kind": "jpeg_reencode", "quality": 85, "subsampling": 2},
        "brightness_0_85": {"kind": "brightness", "factor": 0.85},
        "brightness_1_15": {"kind": "brightness", "factor": 1.15},
        "gaussian_blur_radius_1": {"kind": "gaussian_blur", "radius": 1.0},
    }
    for condition in conditions:
        observed = condition.to_dict()
        for field, expected in expected_condition_values[condition.condition].items():
            if observed.get(field) != expected:
                raise ValueError(f"robustness changed {condition.condition}.{field}")

    robustness_raw = dict(payload["robustness_protocol"])
    _require_exact_keys(
        robustness_raw,
        {
            "batch_size",
            "num_workers",
            "pin_memory",
            "folds",
            "cache_directory",
            "clean_reference",
            "clean_min_prediction_agreement",
            "clean_max_probability_delta",
            "amp_matches_training_evaluation",
            "evaluation_scope",
            "perturbation_order",
            "prediction_artifacts_are_temporary",
        },
        "robustness protocol",
    )
    robustness = RobustnessProtocol(
        batch_size=int(robustness_raw["batch_size"]),
        num_workers=int(robustness_raw["num_workers"]),
        pin_memory=bool(robustness_raw["pin_memory"]),
        folds=tuple(int(value) for value in robustness_raw["folds"]),
        cache_directory=str(robustness_raw["cache_directory"]),
        clean_reference=str(robustness_raw["clean_reference"]),
        clean_min_prediction_agreement=float(robustness_raw["clean_min_prediction_agreement"]),
        clean_max_probability_delta=float(robustness_raw["clean_max_probability_delta"]),
        amp_matches_training_evaluation=bool(robustness_raw["amp_matches_training_evaluation"]),
        evaluation_scope=str(robustness_raw["evaluation_scope"]),
        perturbation_order=str(robustness_raw["perturbation_order"]),
        prediction_artifacts_are_temporary=bool(
            robustness_raw["prediction_artifacts_are_temporary"]
        ),
    )
    if (
        robustness.batch_size != 128
        or robustness.num_workers != 4
        or robustness.folds != tuple(range(5))
        or robustness.cache_directory != "tmp/task2/robustness"
        or robustness.clean_reference != "verified_frozen_oof"
        or robustness.clean_min_prediction_agreement != 1.0
        or robustness.clean_max_probability_delta != 0.0001
        or not robustness.pin_memory
        or not robustness.amp_matches_training_evaluation
        or not robustness.prediction_artifacts_are_temporary
        or robustness.evaluation_scope
        != "all_valid_development_rows_once_through_their_validation_fold_checkpoint"
        or robustness.perturbation_order
        != "decode_exif_rgb_then_perturb_then_original_fold_normalisation"
    ):
        raise ValueError("robustness execution protocol changed")

    cost_raw = dict(payload["cost_protocol"])
    _require_exact_keys(
        cost_raw,
        {
            "batch_size",
            "checkpoint_fold",
            "cpu_threads",
            "devices",
            "fixed_input_rule",
            "model_only_warmups",
            "model_only_repeats",
            "end_to_end_warmups",
            "end_to_end_repeats",
            "synchronise_accelerator_each_repeat",
            "memory_metrics",
        },
        "cost protocol",
    )
    cost = CostProtocol(
        batch_size=int(cost_raw["batch_size"]),
        checkpoint_fold=int(cost_raw["checkpoint_fold"]),
        cpu_threads=int(cost_raw["cpu_threads"]),
        devices=tuple(str(value) for value in cost_raw["devices"]),
        fixed_input_rule=str(cost_raw["fixed_input_rule"]),
        model_only_warmups=int(cost_raw["model_only_warmups"]),
        model_only_repeats=int(cost_raw["model_only_repeats"]),
        end_to_end_warmups=int(cost_raw["end_to_end_warmups"]),
        end_to_end_repeats=int(cost_raw["end_to_end_repeats"]),
        synchronise_accelerator_each_repeat=bool(cost_raw["synchronise_accelerator_each_repeat"]),
        memory_metrics=tuple(str(value) for value in cost_raw["memory_metrics"]),
    )
    expected_memory = (
        "parameter_and_buffer_bytes",
        "training_checkpoint_bytes",
        "process_rss_delta_bytes",
        "peak_cuda_allocated_bytes",
    )
    if (
        cost.batch_size != 1
        or cost.checkpoint_fold != 0
        or cost.cpu_threads != 1
        or cost.devices != ("cpu", "cuda_if_available")
        or cost.fixed_input_rule != "lowest_id_in_checkpoint_validation_fold"
        or cost.model_only_warmups != 30
        or cost.model_only_repeats != 200
        or cost.end_to_end_warmups != 10
        or cost.end_to_end_repeats != 50
        or not cost.synchronise_accelerator_each_repeat
        or cost.memory_metrics != expected_memory
    ):
        raise ValueError("deployment-cost protocol changed")

    warnings = dict(payload["warnings"])
    required_warnings = {
        "analysis_does_not_change_the_g5_candidate",
        "confidence_is_uncalibrated",
        "cost_measurements_are_machine_specific",
        "holdout_is_forbidden",
        "primary_seed_only_because_g5_already_tests_random_seed_stability",
    }
    if set(warnings) != required_warnings or not all(warnings.values()):
        raise ValueError("robustness safety warnings changed")
    threshold = float(payload["material_macro_f1_degradation"])
    if threshold != 0.01:
        raise ValueError("material macro-F1 degradation threshold changed")
    return RobustnessCostSpec(
        analysis_id=str(payload["analysis_id"]),
        expected_row_count=int(payload["expected_row_count"]),
        candidates=candidates,
        conditions=conditions,
        robustness=robustness,
        cost=cost,
        material_macro_f1_degradation=threshold,
    )


def apply_robustness_condition(
    image: Image.Image,
    condition: RobustnessCondition,
) -> Image.Image:
    """Apply one deterministic perturbation to an already EXIF-corrected RGB image."""
    if image.mode != "RGB":
        raise ValueError("robustness perturbations require an RGB image")
    if condition.kind == "none":
        return image.copy()
    if condition.kind == "jpeg_reencode":
        if condition.quality != 85 or condition.subsampling != 2:
            raise ValueError("JPEG robustness requires quality 85 and subsampling 2")
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=condition.quality,
            subsampling=condition.subsampling,
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB").copy()
    if condition.kind == "brightness":
        if condition.factor not in {0.85, 1.15}:
            raise ValueError("brightness robustness factor must be 0.85 or 1.15")
        return ImageEnhance.Brightness(image).enhance(condition.factor)
    if condition.kind == "gaussian_blur":
        if condition.radius != 1.0:
            raise ValueError("Gaussian robustness radius must be 1.0")
        return image.filter(ImageFilter.GaussianBlur(radius=condition.radius))
    raise ValueError(f"unknown robustness condition kind: {condition.kind}")


class PerturbedTensorTransform:
    """Apply a frozen perturbation, then the original fold-fitted normalisation."""

    def __init__(self, *, stats: FoldImageStats, condition: RobustnessCondition) -> None:
        if any(value <= 0 for value in stats.std):
            raise ValueError("fold statistics contain non-positive standard deviations")
        self.stats = stats
        self.condition = condition

    def __call__(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            perturbed = apply_robustness_condition(image, self.condition)
            array, _ = transform_image_with_mask(
                perturbed,
                image_size=self.stats.image_size,
                normalize_range=True,
                mean=self.stats.mean,
                std=self.stats.std,
            )
        return torch.from_numpy(np.moveaxis(array, -1, 0).copy())


def fold_stats_from_history(
    registry_row: Mapping[str, Any],
    *,
    project_root: str | Path = ROOT,
    expected_training_ids: Sequence[int] | None = None,
) -> FoldImageStats:
    """Restore the exact fold statistics embedded in a hash-verified run history."""
    root = Path(project_root)
    history_path = Path(str(registry_row["history_path"]))
    if not history_path.is_absolute():
        history_path = root / history_path
    verify_artifact(history_path, str(registry_row["history_sha256"]))
    with history_path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    identity = {
        "run_id": str(registry_row["run_id"]),
        "experiment_id": str(registry_row["experiment_id"]),
        "fold": int(registry_row["fold"]),
        "seed": int(registry_row["seed"]),
    }
    for field, expected in identity.items():
        if history.get(field) != expected:
            raise ValueError(f"history {field} differs for {identity['run_id']}")
    raw_stats = dict(history.get("loader_audit", {}).get("stats", {}))
    for field in ("image_size", "mean", "std"):
        if field not in raw_stats:
            raise ValueError(f"history lacks loader stats field: {field}")
        raw_stats[field] = tuple(raw_stats[field])
    stats = FoldImageStats(**raw_stats)
    if stats.validation_fold != identity["fold"]:
        raise ValueError("history fold statistics belong to a different validation fold")
    if expected_training_ids is not None:
        expected_hash = canonical_sha256(sorted(int(value) for value in expected_training_ids))
        if stats.training_id_sha256 != expected_hash or stats.image_count != len(
            expected_training_ids
        ):
            raise ValueError("history fold statistics were fitted on different training IDs")
    return stats


def build_robustness_model(
    candidate: str,
    *,
    article_type_classes: int = 124,
) -> nn.Module:
    """Build the exact final-eligible scratch architecture for C2 or I2."""
    if candidate == "C2":
        model = build_season_model(
            SeasonModelSpec(family="resnet18_small_stem", num_classes=len(SEASON_LABELS))
        )
    elif candidate == "I2":
        model = build_multitask_season_model(
            SeasonModelSpec(family="smallcnn", num_classes=len(SEASON_LABELS)),
            article_type_classes=article_type_classes,
        )
    else:
        raise ValueError(f"unknown robustness candidate: {candidate}")
    assert_final_model(model)
    return model


def load_robustness_checkpoint(
    model: nn.Module,
    registry_row: Mapping[str, Any],
    *,
    project_root: str | Path = ROOT,
) -> tuple[nn.Module, dict[str, Any], Path]:
    """Verify one training checkpoint and strictly restore only its model state."""
    root = Path(project_root)
    checkpoint_path = Path(str(registry_row["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    verify_artifact(checkpoint_path, str(registry_row["checkpoint_sha256"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("format_version", -1)) != 1:
        raise ValueError("robustness checkpoint format is not supported")
    if tuple(checkpoint.get("labels", ())) != tuple(SEASON_LABELS):
        raise ValueError("robustness checkpoint changed the Season label order")
    if int(checkpoint.get("best_epoch", -1)) != int(registry_row["best_epoch"]):
        raise ValueError("robustness checkpoint best epoch differs from registry")
    if not math.isclose(
        float(checkpoint.get("best_macro_f1", np.nan)),
        float(registry_row["primary_metric_value"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("robustness checkpoint metric differs from registry")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": str(registry_row["checkpoint_sha256"]),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_macro_f1": float(checkpoint["best_macro_f1"]),
    }
    del checkpoint
    gc.collect()
    return model, metadata, checkpoint_path


def _season_logits(model: nn.Module, candidate: str, images: torch.Tensor) -> torch.Tensor:
    if candidate == "I2":
        predictor = getattr(model, "predict_season_logits", None)
        if not callable(predictor):
            raise TypeError("I2 deployment model lacks predict_season_logits")
        logits = predictor(images)
    else:
        logits = model(images)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise TypeError("robustness model must return [batch, class] Season logits")
    if logits.shape[1] != len(SEASON_LABELS):
        raise ValueError("robustness model returned the wrong Season class count")
    return logits


def _resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported robustness device: {requested}")
    return torch.device(requested)


def predict_robustness_fold(
    model: nn.Module,
    *,
    candidate: RobustnessCandidate,
    condition: RobustnessCondition,
    validation_frame: pd.DataFrame,
    stats: FoldImageStats,
    label_to_index: Mapping[str, int],
    project_root: str | Path = ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str = "auto",
    use_amp: bool = True,
    checkpoint_run_id: str,
    probe_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one perturbation through one matching validation-fold checkpoint."""
    if validation_frame.empty or validation_frame["id"].duplicated().any():
        raise ValueError("robustness validation frame must contain unique products")
    fold_values = pd.to_numeric(validation_frame["cv_fold"], errors="raise").astype(int)
    if set(fold_values) != {stats.validation_fold}:
        raise ValueError("robustness validation rows do not match the fold checkpoint")
    if not validation_frame["partition"].eq("development").all():
        raise ValueError("robustness may evaluate development rows only")
    labels = tuple(SEASON_LABELS)
    if dict(label_to_index) != {label: index for index, label in enumerate(labels)}:
        raise ValueError("robustness label indices changed")
    dataset = EncodedClassificationDataset(
        validation_frame.reset_index(drop=True),
        transform=PerturbedTensorTransform(stats=stats, condition=condition),
        target="season",
        label_to_index=dict(label_to_index),
        root=project_root,
    )
    resolved_device = _resolve_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and resolved_device.type == "cuda",
        worker_init_fn=seed_worker if num_workers else None,
        generator=make_torch_generator(candidate.seed + stats.validation_fold),
        persistent_workers=False,
    )
    amp_enabled = use_amp and resolved_device.type == "cuda"
    model = model.to(resolved_device).eval()
    identifiers: list[int] = []
    target_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(resolved_device, non_blocking=True)
            targets = batch["target"].to(resolved_device, dtype=torch.long, non_blocking=True)
            with torch.amp.autocast(
                device_type=resolved_device.type,
                enabled=amp_enabled,
            ):
                logits = _season_logits(model, candidate.candidate, images)
            identifiers.extend(int(value) for value in batch["id"].tolist())
            target_parts.append(targets.detach().cpu().numpy())
            probability_parts.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    runtime_seconds = time.perf_counter() - started
    targets_array = np.concatenate(target_parts).astype(np.int64, copy=False)
    probabilities = np.concatenate(probability_parts).astype(np.float64, copy=False)
    if len(identifiers) != len(validation_frame):
        raise ValueError("robustness loader did not return every validation product")
    predicted_indices = probabilities.argmax(axis=1)
    predictions = pd.DataFrame(
        {
            "probe_id": probe_id,
            "candidate": candidate.candidate,
            "experiment_id": candidate.experiment_id,
            "condition": condition.condition,
            "checkpoint_run_id": checkpoint_run_id,
            "id": identifiers,
            "fold": stats.validation_fold,
            "seed": candidate.seed,
            "y_true": [labels[index] for index in targets_array],
            "y_pred": [labels[index] for index in predicted_indices],
        }
    )
    for index, label in enumerate(labels):
        predictions[f"prob_{label}"] = probabilities[:, index]
    expected_targets = dict(
        zip(
            validation_frame["id"].astype(int),
            validation_frame["season"].astype(str),
            strict=True,
        )
    )
    validate_oof(
        predictions,
        expected_ids=validation_frame["id"],
        expected_targets=expected_targets,
        labels=labels,
    )
    peak_vram_bytes = None
    if resolved_device.type == "cuda":
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(resolved_device))
    metadata = {
        "runtime_seconds": runtime_seconds,
        "rows": len(predictions),
        "device": str(resolved_device),
        "amp_enabled": amp_enabled,
        "peak_vram_bytes": peak_vram_bytes,
    }
    model.to("cpu")
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, metadata


def verify_image_frame(
    frame: pd.DataFrame,
    *,
    project_root: str | Path = ROOT,
) -> str:
    """Verify current image bytes and hash the exact ordered image set."""
    required = {"id", "path", "sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"image frame lacks provenance columns: {missing}")
    if frame.empty or frame["id"].duplicated().any():
        raise ValueError("image frame must contain unique product IDs")
    root = Path(project_root).resolve()
    records: list[dict[str, Any]] = []
    for row in frame.sort_values("id", kind="stable").to_dict(orient="records"):
        path = Path(str(row["path"]))
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            portable_path = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"image path is outside project root: {resolved}") from error
        expected_sha256 = str(row["sha256"])
        if len(expected_sha256) != 64:
            raise ValueError(f"image {row['id']} has an invalid SHA-256 declaration")
        verify_artifact(resolved, expected_sha256)
        records.append(
            {
                "id": int(row["id"]),
                "path": portable_path,
                "sha256": expected_sha256,
            }
        )
    return canonical_sha256(records)


def robustness_cache_key(
    *,
    analysis_config_sha256: str,
    split_sha256: str,
    label_map_sha256: str,
    implementation_sha256_value: str,
    image_set_sha256: str,
    candidate: RobustnessCandidate,
    condition: RobustnessCondition,
    fold: int,
    checkpoint_sha256: str,
    history_sha256: str,
) -> str:
    """Hash every input that can change one perturbation/fold prediction file."""
    digests = (
        analysis_config_sha256,
        split_sha256,
        label_map_sha256,
        implementation_sha256_value,
        image_set_sha256,
        checkpoint_sha256,
        history_sha256,
    )
    if any(len(value) != 64 for value in digests):
        raise ValueError("robustness cache inputs require SHA-256 digests")
    if fold not in range(5):
        raise ValueError("robustness fold must be in range(5)")
    return canonical_sha256(
        {
            "schema_version": "1.0.0",
            "analysis_config_sha256": analysis_config_sha256,
            "split_sha256": split_sha256,
            "label_map_sha256": label_map_sha256,
            "implementation_sha256": implementation_sha256_value,
            "image_set_sha256": image_set_sha256,
            "candidate": asdict(candidate),
            "condition": condition.to_dict(),
            "fold": fold,
            "checkpoint_sha256": checkpoint_sha256,
            "history_sha256": history_sha256,
        }
    )


def _load_cached_probe(
    directory: Path,
    *,
    cache_key: str,
    expected_ids: Sequence[int],
    expected_targets: Mapping[int, str],
) -> ProbeFoldResult | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("schema_version") != "1.0.0"
            or manifest.get("cache_key") != cache_key
            or manifest.get("status") != "completed"
        ):
            return None
        prediction_path = directory / "predictions.csv"
        verify_artifact(prediction_path, str(manifest["prediction_sha256"]))
        predictions = pd.read_csv(prediction_path)
        coverage = validate_oof(
            predictions,
            expected_ids=expected_ids,
            expected_targets=expected_targets,
            labels=SEASON_LABELS,
        )
        if canonical_sha256(coverage) != str(manifest.get("coverage_sha256", "")):
            return None
        return ProbeFoldResult(
            predictions=predictions,
            record={**manifest, "source": "cache", "manifest_path": str(manifest_path)},
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_or_load_fold_probe(
    model: nn.Module,
    *,
    candidate: RobustnessCandidate,
    condition: RobustnessCondition,
    registry_row: Mapping[str, Any],
    validation_frame: pd.DataFrame,
    stats: FoldImageStats,
    label_to_index: Mapping[str, int],
    analysis_config_sha256: str,
    split_sha256: str,
    label_map_sha256: str,
    implementation_sha256_value: str,
    cache_directory: str | Path,
    mode: ExecutionMode = "run_or_load",
    project_root: str | Path = ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str = "auto",
    use_amp: bool = True,
    git_commit: str | None = None,
) -> ProbeFoldResult:
    """Reuse only an exact probe cache, otherwise execute and atomically replace it."""
    if mode not in {"run", "load", "run_or_load"}:
        raise ValueError(f"unknown robustness execution mode: {mode}")
    fold = int(registry_row["fold"])
    image_set_sha256 = verify_image_frame(
        validation_frame,
        project_root=project_root,
    )
    cache_key = robustness_cache_key(
        analysis_config_sha256=analysis_config_sha256,
        split_sha256=split_sha256,
        label_map_sha256=label_map_sha256,
        implementation_sha256_value=implementation_sha256_value,
        image_set_sha256=image_set_sha256,
        candidate=candidate,
        condition=condition,
        fold=fold,
        checkpoint_sha256=str(registry_row["checkpoint_sha256"]),
        history_sha256=str(registry_row["history_sha256"]),
    )
    output = Path(cache_directory) / cache_key
    expected_ids = validation_frame["id"].astype(int).tolist()
    expected_targets = dict(
        zip(
            validation_frame["id"].astype(int),
            validation_frame["season"].astype(str),
            strict=True,
        )
    )
    if mode != "run":
        cached = _load_cached_probe(
            output,
            cache_key=cache_key,
            expected_ids=expected_ids,
            expected_targets=expected_targets,
        )
        if cached is not None:
            return cached
        if mode == "load":
            raise FileNotFoundError(
                f"no hash-valid robustness cache for {candidate.candidate} fold {fold} "
                f"condition {condition.condition}"
            )

    probe_id = f"{candidate.candidate.lower()}-f{fold}-{condition.condition}-{cache_key[:12]}"
    predictions, runtime = predict_robustness_fold(
        model,
        candidate=candidate,
        condition=condition,
        validation_frame=validation_frame,
        stats=stats,
        label_to_index=label_to_index,
        project_root=project_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
        use_amp=use_amp,
        checkpoint_run_id=str(registry_row["run_id"]),
        probe_id=probe_id,
    )
    prediction_path = output / "predictions.csv"
    atomic_write_csv(prediction_path, predictions)
    predictions = pd.read_csv(prediction_path)
    coverage = validate_oof(
        predictions,
        expected_ids=expected_ids,
        expected_targets=expected_targets,
        labels=SEASON_LABELS,
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "completed",
        "cache_key": cache_key,
        "probe_id": probe_id,
        "candidate": candidate.candidate,
        "experiment_id": candidate.experiment_id,
        "seed": candidate.seed,
        "fold": fold,
        "condition": condition.condition,
        "condition_spec": condition.to_dict(),
        "checkpoint_run_id": str(registry_row["run_id"]),
        "checkpoint_sha256": str(registry_row["checkpoint_sha256"]),
        "history_sha256": str(registry_row["history_sha256"]),
        "analysis_config_sha256": analysis_config_sha256,
        "split_sha256": split_sha256,
        "label_map_sha256": label_map_sha256,
        "implementation_sha256": implementation_sha256_value,
        "image_set_sha256": image_set_sha256,
        "git_commit": git_commit,
        "prediction_path": str(prediction_path),
        "prediction_sha256": compute_sha256(prediction_path),
        "coverage_sha256": canonical_sha256(coverage),
        **runtime,
    }
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return ProbeFoldResult(
        predictions=predictions,
        record={**manifest, "source": "run", "manifest_path": str(manifest_path)},
    )


def reconcile_clean_probe(
    probed: pd.DataFrame,
    frozen_oof: pd.DataFrame,
    *,
    candidate: RobustnessCandidate,
    protocol: RobustnessProtocol,
) -> dict[str, Any]:
    """Prove that same-pipeline clean inference reproduces the frozen OOF reference."""
    probability_columns = [f"prob_{label}" for label in SEASON_LABELS]
    required = {"id", "y_true", "y_pred", *probability_columns}
    for name, frame in (("clean probe", probed), ("frozen OOF", frozen_oof)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks reconciliation columns: {missing}")
        if frame["id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs")
    if len(probed) != len(frozen_oof):
        raise ValueError("clean probe and frozen OOF row counts differ")
    merged = probed.loc[:, sorted(required)].merge(
        frozen_oof.loc[:, sorted(required)],
        on="id",
        how="inner",
        validate="one_to_one",
        suffixes=("_probe", "_frozen"),
    )
    if len(merged) != len(probed):
        raise ValueError("clean probe and frozen OOF IDs differ")
    if not merged["y_true_probe"].eq(merged["y_true_frozen"]).all():
        raise ValueError("clean probe and frozen OOF targets differ")
    agreement = float(merged["y_pred_probe"].eq(merged["y_pred_frozen"]).mean())
    probe_probabilities = merged.loc[
        :, [f"{column}_probe" for column in probability_columns]
    ].to_numpy(dtype=float)
    frozen_probabilities = merged.loc[
        :, [f"{column}_frozen" for column in probability_columns]
    ].to_numpy(dtype=float)
    absolute_delta = np.abs(probe_probabilities - frozen_probabilities)
    result = {
        "candidate": candidate.candidate,
        "experiment_id": candidate.experiment_id,
        "seed": candidate.seed,
        "support": len(merged),
        "clean_reference": protocol.clean_reference,
        "prediction_agreement": agreement,
        "minimum_prediction_agreement": protocol.clean_min_prediction_agreement,
        "maximum_probability_delta": float(absolute_delta.max()),
        "mean_probability_delta": float(absolute_delta.mean()),
        "maximum_probability_delta_tolerance": protocol.clean_max_probability_delta,
    }
    if agreement < protocol.clean_min_prediction_agreement:
        raise ValueError("clean probe predictions do not reproduce the frozen OOF reference")
    if result["maximum_probability_delta"] > protocol.clean_max_probability_delta:
        raise ValueError("clean probe probabilities drift beyond the frozen OOF tolerance")
    return result


def _metric_row(
    frame: pd.DataFrame,
    *,
    candidate: str,
    experiment_id: str,
    condition: str,
    fold: int,
) -> dict[str, Any]:
    probabilities = frame.loc[:, [f"prob_{label}" for label in SEASON_LABELS]].to_numpy(dtype=float)
    metrics = multiclass_metrics(
        frame["y_true"].astype(str),
        probabilities=probabilities,
        labels=SEASON_LABELS,
        y_pred=frame["y_pred"].astype(str),
    )
    return {
        "candidate": candidate,
        "experiment_id": experiment_id,
        "condition": condition,
        "fold": fold,
        "support": len(frame),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "nll": metrics["nll"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "spring_precision": metrics["per_class"]["Spring"]["precision"],
        "spring_recall": metrics["per_class"]["Spring"]["recall"],
        "spring_f1": metrics["per_class"]["Spring"]["f1"],
    }


def build_robustness_tables(
    predictions: Mapping[tuple[str, str], pd.DataFrame],
    spec: RobustnessCostSpec,
) -> RobustnessTables:
    """Calculate paired pooled/fold degradation from aligned condition predictions."""
    expected_keys = {
        (candidate.candidate, condition.condition)
        for candidate in spec.candidates
        for condition in spec.conditions
    }
    if set(predictions) != expected_keys:
        raise ValueError("robustness predictions do not match the frozen candidate/condition grid")
    pooled_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for candidate in spec.candidates:
        clean = predictions[(candidate.candidate, "clean")].copy()
        if len(clean) != spec.expected_row_count:
            raise ValueError(f"{candidate.candidate} clean OOF coverage changed")
        clean_view = clean.loc[:, ["id", "y_true", "y_pred"]].rename(
            columns={"y_pred": "clean_y_pred"}
        )
        for condition in spec.conditions:
            frame = predictions[(candidate.candidate, condition.condition)].copy()
            if len(frame) != spec.expected_row_count:
                raise ValueError(f"{candidate.candidate}/{condition.condition} coverage changed")
            joined = frame.merge(
                clean_view,
                on=["id", "y_true"],
                how="left",
                validate="one_to_one",
            )
            if joined["clean_y_pred"].isna().any():
                raise ValueError("robustness predictions do not align with clean OOF IDs")
            pooled = _metric_row(
                frame,
                candidate=candidate.candidate,
                experiment_id=candidate.experiment_id,
                condition=condition.condition,
                fold=-1,
            )
            pooled.update(
                {
                    "prediction_agreement_with_clean": float(
                        joined["y_pred"].eq(joined["clean_y_pred"]).mean()
                    ),
                    "clean_correct_to_condition_wrong": int(
                        (
                            joined["clean_y_pred"].eq(joined["y_true"])
                            & joined["y_pred"].ne(joined["y_true"])
                        ).sum()
                    ),
                    "clean_wrong_to_condition_correct": int(
                        (
                            joined["clean_y_pred"].ne(joined["y_true"])
                            & joined["y_pred"].eq(joined["y_true"])
                        ).sum()
                    ),
                }
            )
            pooled_rows.append(pooled)
            for fold in range(5):
                subset = frame.loc[
                    pd.to_numeric(frame["fold"], errors="raise").astype(int).eq(fold)
                ]
                fold_rows.append(
                    _metric_row(
                        subset,
                        candidate=candidate.candidate,
                        experiment_id=candidate.experiment_id,
                        condition=condition.condition,
                        fold=fold,
                    )
                )
    pooled_metrics = pd.DataFrame(pooled_rows)
    for candidate in ("C2", "I2"):
        candidate_mask = pooled_metrics["candidate"].eq(candidate)
        clean = pooled_metrics.loc[candidate_mask & pooled_metrics["condition"].eq("clean")]
        if len(clean) != 1:
            raise ValueError(f"{candidate} requires one clean robustness row")
        clean_row = clean.iloc[0]
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "spring_recall",
            "spring_f1",
            "nll",
            "brier",
            "ece",
        ):
            pooled_metrics.loc[candidate_mask, f"delta_{metric}_vs_clean"] = pooled_metrics.loc[
                candidate_mask, metric
            ] - float(clean_row[metric])
        pooled_metrics.loc[candidate_mask, "material_macro_f1_degradation"] = (
            pooled_metrics.loc[candidate_mask, "delta_macro_f1_vs_clean"]
            .lt(-spec.material_macro_f1_degradation)
            .to_numpy()
        )
    pooled_metrics = pooled_metrics.sort_values(
        ["candidate", "condition"], kind="stable"
    ).reset_index(drop=True)

    comparison_rows = []
    indexed = pooled_metrics.set_index(["candidate", "condition"])
    for condition in EXPECTED_CONDITIONS:
        c2 = indexed.loc[("C2", condition)]
        i2 = indexed.loc[("I2", condition)]
        row: dict[str, Any] = {"condition": condition}
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "spring_recall",
            "spring_f1",
            "prediction_agreement_with_clean",
        ):
            row[f"c2_{metric}"] = float(c2[metric])
            row[f"i2_{metric}"] = float(i2[metric])
            row[f"i2_minus_c2_{metric}"] = float(i2[metric] - c2[metric])
        comparison_rows.append(row)
    return RobustnessTables(
        pooled_metrics=pooled_metrics,
        fold_metrics=pd.DataFrame(fold_rows)
        .sort_values(["candidate", "condition", "fold"], kind="stable")
        .reset_index(drop=True),
        candidate_comparison=pd.DataFrame(comparison_rows),
    )


def model_tensor_bytes(model: nn.Module) -> int:
    """Return exact parameter-and-buffer storage bytes for an in-memory model."""
    tensors = list(model.parameters()) + list(model.buffers())
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


def _synchronise(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_repeats(
    operation: Any,
    *,
    warmups: int,
    repeats: int,
    device: torch.device,
    synchronise: bool,
) -> np.ndarray:
    if warmups < 0 or repeats < 1:
        raise ValueError("latency warmups must be non-negative and repeats positive")
    for _ in range(warmups):
        operation()
        _synchronise(device, synchronise)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        _synchronise(device, synchronise)
        started = time.perf_counter_ns()
        operation()
        _synchronise(device, synchronise)
        values[index] = (time.perf_counter_ns() - started) / 1_000_000
    return values


def _latency_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("latency values must be a finite one-dimensional array")
    return {
        f"{prefix}_mean_ms": float(values.mean()),
        f"{prefix}_median_ms": float(np.median(values)),
        f"{prefix}_p95_ms": float(np.quantile(values, 0.95)),
        f"{prefix}_throughput_per_second_at_median": float(1000 / np.median(values)),
    }


def measure_deployment_cost(
    model: nn.Module,
    *,
    candidate: RobustnessCandidate,
    checkpoint_run_id: str,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    transform: PerturbedTensorTransform,
    image_path: str | Path,
    input_id: int,
    protocol: CostProtocol,
    requested_device: str,
    rss_baseline_before_model_load: int | None = None,
) -> dict[str, Any]:
    """Measure model-only and warm-cache end-to-end single-image inference cost."""
    if requested_device == "cuda_if_available":
        if not torch.cuda.is_available():
            return {
                "candidate": candidate.candidate,
                "experiment_id": candidate.experiment_id,
                "checkpoint_run_id": checkpoint_run_id,
                "device": "cuda",
                "available": False,
                "reason": "cuda_unavailable",
            }
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"unsupported cost device request: {requested_device}")
    if protocol.batch_size != 1:
        raise ValueError("deployment-cost probe requires batch size one")

    process = psutil.Process()
    original_threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(protocol.cpu_threads)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = (
        int(rss_baseline_before_model_load)
        if rss_baseline_before_model_load is not None
        else int(process.memory_info().rss)
    )
    if rss_before < 0:
        raise ValueError("RSS baseline must be non-negative")
    model = model.to(device).eval()
    clean_tensor = transform(image_path).unsqueeze(0).to(device)
    rss_after_load = int(process.memory_info().rss)
    if device.type == "cuda":
        cuda_after_load = int(torch.cuda.memory_allocated(device))
    else:
        cuda_after_load = None

    def infer_tensor(tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = _season_logits(model, candidate.candidate, tensor)
            return torch.softmax(logits.float(), dim=1)

    def model_only() -> None:
        infer_tensor(clean_tensor)

    path = Path(image_path)

    def end_to_end() -> None:
        tensor = transform(path).unsqueeze(0).to(device)
        infer_tensor(tensor)

    try:
        model_values = _timed_repeats(
            model_only,
            warmups=protocol.model_only_warmups,
            repeats=protocol.model_only_repeats,
            device=device,
            synchronise=protocol.synchronise_accelerator_each_repeat,
        )
        end_to_end_values = _timed_repeats(
            end_to_end,
            warmups=protocol.end_to_end_warmups,
            repeats=protocol.end_to_end_repeats,
            device=device,
            synchronise=protocol.synchronise_accelerator_each_repeat,
        )
        rss_after_inference = int(process.memory_info().rss)
        cuda_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        return {
            "candidate": candidate.candidate,
            "experiment_id": candidate.experiment_id,
            "seed": candidate.seed,
            "checkpoint_run_id": checkpoint_run_id,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_fold": protocol.checkpoint_fold,
            "device": str(device),
            "available": True,
            "input_id": input_id,
            "batch_size": protocol.batch_size,
            "cpu_threads": protocol.cpu_threads if device.type == "cpu" else original_threads,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "parameter_and_buffer_bytes": model_tensor_bytes(model),
            "training_checkpoint_bytes": checkpoint_bytes,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_load_bytes": rss_after_load,
            "process_rss_after_inference_bytes": rss_after_inference,
            "process_rss_delta_bytes": max(
                0,
                max(rss_after_inference, rss_after_load) - rss_before,
            ),
            "cuda_allocated_after_load_bytes": cuda_after_load,
            "peak_cuda_allocated_bytes": cuda_peak,
            "model_only_warmups": protocol.model_only_warmups,
            "model_only_repeats": protocol.model_only_repeats,
            "end_to_end_warmups": protocol.end_to_end_warmups,
            "end_to_end_repeats": protocol.end_to_end_repeats,
            **_latency_summary(model_values, "model_only"),
            **_latency_summary(end_to_end_values, "end_to_end"),
        }
    finally:
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch.set_num_threads(original_threads)


def deployment_cost_cache_key(
    *,
    analysis_config_sha256: str,
    implementation_sha256_value: str,
    checkpoint_sha256: str,
    stats_sha256: str,
    runtime_sha256: str,
    input_image_sha256: str,
    candidate: RobustnessCandidate,
    requested_device: str,
    input_id: int,
    protocol: CostProtocol,
) -> str:
    """Hash the model, machine, input, and timing protocol for one cost probe."""
    digests = (
        analysis_config_sha256,
        implementation_sha256_value,
        checkpoint_sha256,
        stats_sha256,
        runtime_sha256,
        input_image_sha256,
    )
    if any(len(value) != 64 for value in digests):
        raise ValueError("deployment-cost cache inputs require SHA-256 digests")
    if requested_device not in protocol.devices:
        raise ValueError("deployment-cost device is absent from the frozen protocol")
    return canonical_sha256(
        {
            "schema_version": "1.0.0",
            "analysis_config_sha256": analysis_config_sha256,
            "implementation_sha256": implementation_sha256_value,
            "checkpoint_sha256": checkpoint_sha256,
            "stats_sha256": stats_sha256,
            "runtime_sha256": runtime_sha256,
            "input_image_sha256": input_image_sha256,
            "candidate": asdict(candidate),
            "requested_device": requested_device,
            "input_id": input_id,
            "protocol": asdict(protocol),
        }
    )


def run_or_load_deployment_cost(
    model: nn.Module,
    *,
    candidate: RobustnessCandidate,
    checkpoint_run_id: str,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    transform: PerturbedTensorTransform,
    image_path: str | Path,
    input_id: int,
    protocol: CostProtocol,
    requested_device: str,
    analysis_config_sha256: str,
    implementation_sha256_value: str,
    stats_sha256: str,
    runtime_sha256: str,
    input_image_sha256: str,
    cache_directory: str | Path,
    mode: ExecutionMode = "run_or_load",
    rss_baseline_before_model_load: int | None = None,
) -> dict[str, Any]:
    """Reuse only an exact machine-specific cost result, otherwise measure it once."""
    if mode not in {"run", "load", "run_or_load"}:
        raise ValueError(f"unknown deployment-cost execution mode: {mode}")
    verify_artifact(image_path, input_image_sha256)
    cache_key = deployment_cost_cache_key(
        analysis_config_sha256=analysis_config_sha256,
        implementation_sha256_value=implementation_sha256_value,
        checkpoint_sha256=checkpoint_sha256,
        stats_sha256=stats_sha256,
        runtime_sha256=runtime_sha256,
        input_image_sha256=input_image_sha256,
        candidate=candidate,
        requested_device=requested_device,
        input_id=input_id,
        protocol=protocol,
    )
    output = Path(cache_directory) / "cost" / cache_key
    result_path = output / "result.json"
    manifest_path = output / "manifest.json"
    if mode != "run" and manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("schema_version") == "1.0.0"
            and manifest.get("cache_key") == cache_key
            and manifest.get("status") == "completed"
        ):
            verify_artifact(result_path, str(manifest.get("result_sha256", "")))
            with result_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            return {
                **result,
                "source": "cache",
                "cache_key": cache_key,
                "result_path": str(result_path),
                "result_sha256": compute_sha256(result_path),
            }
    if mode == "load":
        raise FileNotFoundError(
            f"no hash-valid deployment-cost cache for {candidate.candidate}/{requested_device}"
        )
    result = measure_deployment_cost(
        model,
        candidate=candidate,
        checkpoint_run_id=checkpoint_run_id,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint_bytes,
        transform=transform,
        image_path=image_path,
        input_id=input_id,
        protocol=protocol,
        requested_device=requested_device,
        rss_baseline_before_model_load=rss_baseline_before_model_load,
    )
    atomic_write_json(result_path, result)
    manifest = {
        "schema_version": "1.0.0",
        "status": "completed",
        "cache_key": cache_key,
        "candidate": candidate.candidate,
        "experiment_id": candidate.experiment_id,
        "requested_device": requested_device,
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "analysis_config_sha256": analysis_config_sha256,
        "implementation_sha256": implementation_sha256_value,
        "stats_sha256": stats_sha256,
        "runtime_sha256": runtime_sha256,
        "input_image_sha256": input_image_sha256,
        "input_id": input_id,
        "result_path": str(result_path),
        "result_sha256": compute_sha256(result_path),
    }
    atomic_write_json(manifest_path, manifest)
    return {
        **result,
        "source": "run",
        "cache_key": cache_key,
        "result_path": str(result_path),
        "result_sha256": compute_sha256(result_path),
    }


def canonical_validation_frames(
    splits: pd.DataFrame,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return the five canonical training/validation Season frames for audit and probes."""
    frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in range(5):
        training, validation = get_cv_split(splits, fold)
        training = get_samples(training, target="season").reset_index(drop=True)
        validation = get_samples(validation, target="season").reset_index(drop=True)
        frames[fold] = (training, validation)
    return frames


__all__ = [
    "CostProtocol",
    "EXPECTED_CANDIDATES",
    "EXPECTED_CONDITIONS",
    "PerturbedTensorTransform",
    "ProbeFoldResult",
    "ROBUSTNESS_CONFIG_PATH",
    "ROBUSTNESS_IMPLEMENTATION_PATHS",
    "RobustnessCandidate",
    "RobustnessCondition",
    "RobustnessCostSpec",
    "RobustnessProtocol",
    "RobustnessTables",
    "apply_robustness_condition",
    "build_robustness_model",
    "build_robustness_tables",
    "canonical_validation_frames",
    "deployment_cost_cache_key",
    "fold_stats_from_history",
    "load_robustness_checkpoint",
    "load_robustness_cost_spec",
    "measure_deployment_cost",
    "model_tensor_bytes",
    "predict_robustness_fold",
    "reconcile_clean_probe",
    "robustness_cache_key",
    "run_or_load_fold_probe",
    "run_or_load_deployment_cost",
    "verify_image_frame",
]
