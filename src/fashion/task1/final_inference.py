"""Hash-verified, deterministic image-only inference for Task 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from fashion.config import ROOT, TASK1_MODEL_MANIFEST_JSON
from fashion.data.hashing import compute_sha256
from fashion.task1.models import Task1ModelConfig, Task1SmallCNN
from fashion.task1.preprocessing import (
    TASK1_CONTROL_PREPROCESSING,
    Task1ImageTransform,
    Task1Normalization,
    Task1PreprocessingConfig,
    build_task1_validation_transform,
)
from fashion.task1.refit import load_verified_task1_refit_manifest


class InvalidTask1ImageError(ValueError):
    """Raised when an input cannot be decoded as an image."""


@dataclass(frozen=True)
class Task1Prediction:
    image_path: str
    predicted_label: str
    predicted_index: int
    probabilities: dict[str, float]
    confidence: float
    run_id: str
    manifest_sha256: str
    bundle_sha256: str


@dataclass(frozen=True)
class Task1Bundle:
    model: nn.Module
    transform: Task1ImageTransform
    class_names: tuple[str, ...]
    device: torch.device
    run_id: str
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    bundle_path: Path
    bundle_sha256: str

    def predict_paths(
        self,
        paths: Sequence[str | Path],
        *,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Return one deterministic 124-class probability row per input path."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not paths:
            raise ValueError("predict_paths requires at least one image")
        rows: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            tensors: list[torch.Tensor] = []
            for raw_path in paths[start : start + batch_size]:
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Task 1 inference image does not exist: {path}")
                try:
                    tensors.append(torch.from_numpy(self.transform(path)))
                except (
                    Image.DecompressionBombError,
                    OSError,
                    UnidentifiedImageError,
                    ValueError,
                ) as error:
                    raise InvalidTask1ImageError(
                        f"Could not read a valid Task 1 image: {path}"
                    ) from error
            batch = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch)
                if logits.shape != (len(tensors), len(self.class_names)):
                    raise ValueError(
                        f"Task 1 model returned the wrong logit shape: {tuple(logits.shape)}"
                    )
                probabilities = torch.softmax(logits, dim=1)
            matrix = probabilities.detach().cpu().numpy().astype(np.float64, copy=False)
            if not np.isfinite(matrix).all():
                raise FloatingPointError("Task 1 model returned non-finite probabilities")
            if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
                raise FloatingPointError("Task 1 probabilities do not sum to one")
            rows.append(matrix)
        return np.concatenate(rows, axis=0)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
    return torch.device(device)


def _preprocessing_from_payload(payload: dict[str, Any]) -> Task1PreprocessingConfig:
    normalized = dict(payload)
    for name in (
        "image_size",
        "pad_color",
        "scale_range",
        "brightness_range",
        "contrast_range",
    ):
        normalized[name] = tuple(normalized[name])
    config = Task1PreprocessingConfig(**normalized)
    if config != TASK1_CONTROL_PREPROCESSING:
        raise RuntimeError("model bundle does not use the frozen no-augmentation preprocessing")
    return config


def load_task1_bundle(
    manifest_path: str | Path = TASK1_MODEL_MANIFEST_JSON,
    *,
    registry_path: str | Path | None = None,
    project_root: str | Path = ROOT,
    device: str = "auto",
) -> Task1Bundle:
    """Load only the completed scratch refit bound to its registry and file hashes."""
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = load_verified_task1_refit_manifest(
        manifest_file,
        project_root=root,
        registry_path=registry_path,
    )
    bundle_path = Path(str(manifest["_resolved_bundle_path"]))
    try:
        payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"could not load verified Task 1 bundle: {bundle_path}") from error
    required = {
        "format_version",
        "task",
        "target",
        "candidate_id",
        "model_family",
        "model_config",
        "model_state_dict",
        "class_names",
        "label_to_index",
        "preprocessing",
        "normalization",
        "training",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("Task 1 model bundle fields changed")
    classes = tuple(str(label) for label in payload["class_names"])
    expected_map = {label: index for index, label in enumerate(classes)}
    training = payload["training"]
    if (
        payload["format_version"] != 1
        or payload["task"] != "task1"
        or payload["target"] != "articleType"
        or payload["candidate_id"] != "task1_cnn_no_aug_unweighted_v1"
        or payload["model_family"] != "task1_small_cnn_v1"
        or len(classes) != 124
        or payload["label_to_index"] != expected_map
        or training
        != {
            "seed": 2753,
            "epochs": 20,
            "final_epoch": 20,
            "validation_used": False,
            "checkpoint_rule": "fixed_last_epoch",
        }
    ):
        raise RuntimeError("Task 1 model bundle is not deployment eligible")
    model_config = dict(payload["model_config"])
    if "adaptive_size" in model_config:
        model_config["adaptive_size"] = tuple(model_config["adaptive_size"])
    if Task1ModelConfig(**model_config) != Task1ModelConfig():
        raise RuntimeError("Task 1 model architecture differs from the frozen scratch CNN")
    preprocessing = _preprocessing_from_payload(dict(payload["preprocessing"]))
    normalization_payload = dict(payload["normalization"])
    normalization = Task1Normalization(
        mean=tuple(float(value) for value in normalization_payload["mean"]),
        std=tuple(float(value) for value in normalization_payload["std"]),
        fitted_products=int(normalization_payload["fitted_products"]),
        fitted_ids_sha256=str(normalization_payload["fitted_ids_sha256"]),
    )
    if normalization.fitted_products != int(manifest["development_rows"]):
        raise RuntimeError("normalization rows do not match the final refit development rows")
    model = Task1SmallCNN(124)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    selected_device = _resolve_device(device)
    model.to(selected_device)
    model.eval()
    public_manifest = {key: value for key, value in manifest.items() if not key.startswith("_")}
    return Task1Bundle(
        model=model,
        transform=build_task1_validation_transform(normalization, config=preprocessing),
        class_names=classes,
        device=selected_device,
        run_id=str(manifest["run_id"]),
        manifest=public_manifest,
        manifest_path=manifest_file,
        manifest_sha256=compute_sha256(manifest_file),
        bundle_path=bundle_path,
        bundle_sha256=str(manifest["bundle"]["sha256"]),
    )


def predict_article_type(
    bundle: Task1Bundle,
    image_path: str | Path,
) -> Task1Prediction:
    """Return one image-only article-type prediction with all class probabilities."""
    matrix = bundle.predict_paths([image_path], batch_size=1)
    values = matrix[0]
    index = int(np.argmax(values))
    probabilities = {
        label: float(values[position]) for position, label in enumerate(bundle.class_names)
    }
    return Task1Prediction(
        image_path=str(Path(image_path).expanduser().resolve()),
        predicted_label=bundle.class_names[index],
        predicted_index=index,
        probabilities=probabilities,
        confidence=float(values[index]),
        run_id=bundle.run_id,
        manifest_sha256=bundle.manifest_sha256,
        bundle_sha256=bundle.bundle_sha256,
    )
