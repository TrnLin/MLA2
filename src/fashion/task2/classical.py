"""HOG plus HSV histogram reference model with a linear support-vector classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fashion.config import RANDOM_SEED, ROOT
from fashion.data.images import resolve_image_size, transform_image_with_mask
from fashion.task2.baselines import _valid_target_rows
from fashion.train.artifacts import canonical_sha256
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics


@dataclass(frozen=True)
class HogHsvSpec:
    """Predeclared B1 feature and classifier settings."""

    image_size: tuple[int, int] = (80, 60)
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    hsv_bins: tuple[int, int, int] = (18, 8, 8)
    svm_c: float = 1.0
    max_iterations: int = 5_000

    def validate(self) -> None:
        resolve_image_size(self.image_size)
        if self.hog_orientations < 2:
            raise ValueError("hog_orientations must be at least 2")
        if any(value < 1 for value in (*self.hog_pixels_per_cell, *self.hog_cells_per_block)):
            raise ValueError("HOG cell dimensions must be positive")
        if any(value < 2 for value in self.hsv_bins):
            raise ValueError("each HSV histogram needs at least 2 bins")
        if self.svm_c <= 0 or self.max_iterations < 1:
            raise ValueError("svm_c and max_iterations must be positive")

    @property
    def feature_id(self) -> str:
        return f"hog-hsv-{canonical_sha256(asdict(self))[:16]}"


@dataclass(frozen=True)
class HogHsvSvmModel:
    """Fitted B1 pipeline and the exact training scope used to build it."""

    spec: HogHsvSpec
    labels: tuple[str, ...]
    pipeline: Pipeline
    training_product_count: int
    training_id_sha256: str
    feature_count: int
    probability_method: str = "uncalibrated_softmax_of_decision_scores"


@dataclass(frozen=True)
class HogHsvFoldResult:
    """One fold's B1 product-level predictions and metrics."""

    validation_fold: int
    model: HogHsvSvmModel
    oof: pd.DataFrame
    metrics: dict[str, Any]


def _resolve_image_path(root: Path, value: Any) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"manifest image path must be relative: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest image path escapes project root: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest image does not exist: {resolved}")
    return resolved


def extract_hog_hsv(path: str | Path, spec: HogHsvSpec = HogHsvSpec()) -> np.ndarray:
    """Extract deterministic shape and colour features while excluding padded HSV pixels."""
    spec.validate()
    with Image.open(path) as image:
        rgb, content_mask = transform_image_with_mask(
            image,
            image_size=spec.image_size,
            normalize_range=True,
        )
    shape = hog(
        rgb2gray(rgb),
        orientations=spec.hog_orientations,
        pixels_per_cell=spec.hog_pixels_per_cell,
        cells_per_block=spec.hog_cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32, copy=False)
    hsv_content = rgb2hsv(rgb)[content_mask]
    colour_parts: list[np.ndarray] = []
    for channel, bins in enumerate(spec.hsv_bins):
        histogram, _ = np.histogram(
            hsv_content[:, channel],
            bins=bins,
            range=(0.0, 1.0),
        )
        normalized = histogram.astype(np.float32)
        normalized /= max(float(normalized.sum()), 1.0)
        colour_parts.append(normalized)
    return np.concatenate([shape, *colour_parts]).astype(np.float32, copy=False)


@lru_cache(maxsize=None)
def _cached_hog_hsv(
    resolved_path: str,
    spec: HogHsvSpec,
    image_identity: str,
) -> np.ndarray:
    """Cache deterministic raw features; the identity prevents stale within-run reuse."""
    del image_identity
    values = extract_hog_hsv(resolved_path, spec)
    values.setflags(write=False)
    return values


def clear_hog_hsv_feature_cache() -> None:
    """Release cached raw features between independent matrices or tests when needed."""
    _cached_hog_hsv.cache_clear()


def _image_identity(row: Any, path: Path) -> str:
    declared_sha256 = str(getattr(row, "sha256", "")).strip().lower()
    if len(declared_sha256) == 64:
        return declared_sha256
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def extract_feature_matrix(
    frame: pd.DataFrame,
    *,
    spec: HogHsvSpec,
    root: str | Path = ROOT,
) -> np.ndarray:
    """Extract B1 features in stable frame order with path containment checks."""
    if "path" not in frame:
        raise ValueError("fold frame is missing the image path column")
    project_root = Path(root).resolve()
    features = []
    for row in frame.itertuples(index=False):
        path = _resolve_image_path(project_root, row.path)
        features.append(
            _cached_hog_hsv(
                str(path),
                spec,
                _image_identity(row, path),
            )
        )
    if not features:
        raise ValueError("cannot extract features from an empty frame")
    feature_count = len(features[0])
    if any(len(values) != feature_count for values in features):
        raise ValueError("HOG-HSV feature lengths differ across images")
    return np.stack(features).astype(np.float32, copy=False)


def fit_hog_hsv_svm(
    training_frame: pd.DataFrame,
    *,
    spec: HogHsvSpec = HogHsvSpec(),
    labels: tuple[str, ...] = SEASON_LABELS,
    target: str = "season",
    seed: int = RANDOM_SEED,
    root: str | Path = ROOT,
) -> HogHsvSvmModel:
    """Fit B1 only on valid rows from one training complement."""
    spec.validate()
    ordered_labels = tuple(str(label) for label in labels)
    if len(ordered_labels) < 2 or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("labels must contain at least two unique values")
    training = _valid_target_rows(training_frame, target)
    unknown = sorted(set(training[target].astype(str)) - set(ordered_labels))
    if unknown:
        raise ValueError(f"training fold contains unknown {target} labels: {unknown}")
    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    targets = training[target].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    if len(np.unique(targets)) < 2:
        raise ValueError("LinearSVC needs at least two observed training classes")
    features = extract_feature_matrix(training, spec=spec, root=root)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=spec.svm_c,
                    class_weight=None,
                    dual="auto",
                    max_iter=spec.max_iterations,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(features, targets)
    return HogHsvSvmModel(
        spec=spec,
        labels=ordered_labels,
        pipeline=pipeline,
        training_product_count=len(training),
        training_id_sha256=canonical_sha256(sorted(int(value) for value in training["id"])),
        feature_count=features.shape[1],
    )


def _full_decision_scores(model: HogHsvSvmModel, features: np.ndarray) -> np.ndarray:
    classifier: LinearSVC = model.pipeline.named_steps["svm"]
    raw_scores = np.asarray(model.pipeline.decision_function(features), dtype=np.float64)
    observed_classes = np.asarray(classifier.classes_, dtype=np.int64)
    if raw_scores.ndim == 1:
        raw_scores = np.column_stack([-raw_scores, raw_scores])
    if raw_scores.shape[1] != len(observed_classes):
        raise ValueError("LinearSVC decision columns do not match its fitted classes")
    scores = np.full((len(features), len(model.labels)), -1e9, dtype=np.float64)
    scores[:, observed_classes] = raw_scores
    return scores


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def evaluate_hog_hsv_svm_fold(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    validation_fold: int,
    spec: HogHsvSpec = HogHsvSpec(),
    labels: tuple[str, ...] = SEASON_LABELS,
    target: str = "season",
    seed: int = RANDOM_SEED,
    root: str | Path = ROOT,
) -> HogHsvFoldResult:
    """Fit B1 and emit one fold's fixed-label OOF evidence."""
    model = fit_hog_hsv_svm(
        training_frame,
        spec=spec,
        labels=labels,
        target=target,
        seed=seed,
        root=root,
    )
    validation = _valid_target_rows(validation_frame, target)
    if set(training_frame["id"].astype(int)) & set(validation["id"].astype(int)):
        raise ValueError("training and validation IDs overlap")
    if "cv_fold" in validation:
        folds = pd.to_numeric(validation["cv_fold"], errors="raise").astype(int)
        if not folds.eq(validation_fold).all():
            raise ValueError("validation rows do not match the requested canonical fold")
    features = extract_feature_matrix(validation, spec=spec, root=root)
    scores = _full_decision_scores(model, features)
    probabilities = _softmax(scores)
    predicted_indices = scores.argmax(axis=1)
    predictions = np.asarray(model.labels, dtype=object)[predicted_indices]
    truth = validation[target].astype(str).to_numpy()
    oof = pd.DataFrame(
        {
            "id": validation["id"].astype(int).to_numpy(),
            "fold": validation_fold,
            "y_true": truth,
            "y_pred": predictions,
        }
    )
    for index, label in enumerate(model.labels):
        oof[f"prob_{label}"] = probabilities[:, index]
    metrics = multiclass_metrics(
        truth,
        probabilities=probabilities,
        labels=model.labels,
        y_pred=predictions,
    )
    metrics["probability_method"] = model.probability_method
    return HogHsvFoldResult(
        validation_fold=validation_fold,
        model=model,
        oof=oof,
        metrics=metrics,
    )
