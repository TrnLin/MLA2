"""Verified image-only inference for the frozen Task 2 Season bundle."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import UnidentifiedImageError
from torch import nn

from fashion.config import ROOT, RUNS_CSV, TASK2_MODEL_MANIFEST_JSON
from fashion.data.hashing import compute_sha256
from fashion.data.torch import FoldImageStats, TensorImageTransform, build_image_transform
from fashion.models.season import (
    SeasonModelSpec,
    assert_final_model,
    build_multitask_season_model,
)
from fashion.task2.refit import load_verified_development_refit_manifest


class InvalidSeasonImageError(ValueError):
    """Raised when one requested inference image cannot be decoded safely."""


@dataclass(frozen=True)
class SeasonPrediction:
    """One calibrated image-only Season prediction and its audit identifiers."""

    image_path: str
    predicted_label: str
    probabilities: dict[str, float]
    confidence: float
    review_required: bool | None
    latency_ms: float
    run_id: str
    manifest_sha256: str
    bundle_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "predicted_label": self.predicted_label,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "review_required": self.review_required,
            "latency_ms": self.latency_ms,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "bundle_sha256": self.bundle_sha256,
        }


@dataclass(frozen=True)
class SeasonBundle:
    """Loaded scratch model plus the exact frozen preprocessing and calibration."""

    model: nn.Module
    transform: TensorImageTransform
    labels: tuple[str, ...]
    temperature: float
    review_threshold: float | None
    device: torch.device
    run_id: str
    manifest_path: Path
    manifest_sha256: str
    bundle_path: Path
    bundle_sha256: str


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("inference device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
    return torch.device(device)


def _require_probability_contract(
    probabilities: torch.Tensor,
    *,
    expected_classes: int,
) -> None:
    if probabilities.shape != (1, expected_classes):
        raise ValueError(
            "Season model returned an invalid probability shape: "
            f"expected (1, {expected_classes}), got {tuple(probabilities.shape)}"
        )
    if not bool(torch.isfinite(probabilities).all().item()):
        raise FloatingPointError("Season model returned non-finite probabilities")
    total = float(probabilities.sum().item())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise FloatingPointError(f"Season probabilities do not sum to one: {total}")


def load_season_bundle(
    manifest_path: str | Path = TASK2_MODEL_MANIFEST_JSON,
    *,
    registry_path: str | Path = RUNS_CSV,
    project_root: str | Path = ROOT,
    device: str = "auto",
) -> SeasonBundle:
    """Load only a fully verified, registry-bound, scratch Task 2 package."""
    root = Path(project_root).resolve()
    manifest, resolved_manifest, payload = load_verified_development_refit_manifest(
        manifest_path,
        project_root=root,
        registry_path=registry_path,
    )
    labels = tuple(str(value) for value in payload["labels"])
    preprocessing = payload["preprocessing"]
    loader_stats = manifest["loader_audit"]["stats"]
    stats = FoldImageStats(
        validation_fold=None,
        image_size=tuple(int(value) for value in preprocessing["image_size"]),
        image_count=int(manifest["valid_development_rows"]),
        content_pixel_count=int(preprocessing["content_pixel_count"]),
        mean=tuple(float(value) for value in preprocessing["mean"]),
        std=tuple(float(value) for value in preprocessing["std"]),
        training_id_sha256=str(loader_stats["training_id_sha256"]),
    )
    transform = build_image_transform(stats, training=False)
    model_spec = SeasonModelSpec(**payload["model_spec"])
    model = build_multitask_season_model(
        model_spec,
        article_type_classes=len(payload["auxiliary"]["labels"]),
    )
    assert_final_model(model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    selected_device = _resolve_device(device)
    model.to(selected_device)
    model.eval()
    temperature = float(payload["calibration"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("verified Season temperature must be finite and positive")
    threshold_value = payload["calibration"]["review_threshold"]
    review_threshold = None if threshold_value is None else float(threshold_value)
    if review_threshold is not None and not 0.0 <= review_threshold <= 1.0:
        raise ValueError("Season review threshold must be in [0, 1]")
    bundle_relative = Path(str(manifest["bundle"]["path"]))
    if bundle_relative.is_absolute():
        raise ValueError("verified Season bundle path must be project-relative")
    resolved_bundle = (root / bundle_relative).resolve()
    return SeasonBundle(
        model=model,
        transform=transform,
        labels=labels,
        temperature=temperature,
        review_threshold=review_threshold,
        device=selected_device,
        run_id=str(manifest["run_id"]),
        manifest_path=resolved_manifest,
        manifest_sha256=compute_sha256(resolved_manifest),
        bundle_path=resolved_bundle,
        bundle_sha256=str(manifest["bundle"]["sha256"]),
    )


def predict_season(bundle: SeasonBundle, image_path: str | Path) -> SeasonPrediction:
    """Predict calibrated Season probabilities from one image and no metadata."""
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Season inference image does not exist: {path}")
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
    started = time.perf_counter()
    try:
        image = bundle.transform(path).unsqueeze(0).to(bundle.device)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise InvalidSeasonImageError(
            f"Could not read a valid image for Season prediction: {path}"
        ) from error
    with torch.inference_mode():
        predictor = getattr(bundle.model, "predict_season_logits", None)
        if not callable(predictor):
            raise TypeError("verified Season model lacks the image-only prediction method")
        logits = predictor(image)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Season model logits must be a tensor")
        probabilities = torch.softmax(logits / bundle.temperature, dim=1)
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
    latency_ms = (time.perf_counter() - started) * 1_000
    probabilities = probabilities.detach().cpu()
    _require_probability_contract(probabilities, expected_classes=len(bundle.labels))
    values = probabilities[0].tolist()
    probability_map = {
        label: float(values[index]) for index, label in enumerate(bundle.labels)
    }
    predicted_index = int(probabilities.argmax(dim=1).item())
    confidence = probability_map[bundle.labels[predicted_index]]
    review_required = (
        None
        if bundle.review_threshold is None
        else confidence < bundle.review_threshold
    )
    return SeasonPrediction(
        image_path=path.as_posix(),
        predicted_label=bundle.labels[predicted_index],
        probabilities=probability_map,
        confidence=confidence,
        review_required=review_required,
        latency_ms=latency_ms,
        run_id=bundle.run_id,
        manifest_sha256=bundle.manifest_sha256,
        bundle_sha256=bundle.bundle_sha256,
    )


def predict_manifest(
    bundle: SeasonBundle,
    image_paths: Sequence[str | Path],
) -> tuple[SeasonPrediction, ...]:
    """Predict an ordered image list without reading labels or holdout metadata."""
    if not image_paths:
        raise ValueError("Season prediction manifest must contain at least one image")
    return tuple(predict_season(bundle, path) for path in image_paths)


__all__ = [
    "InvalidSeasonImageError",
    "SeasonBundle",
    "SeasonPrediction",
    "load_season_bundle",
    "predict_manifest",
    "predict_season",
]
