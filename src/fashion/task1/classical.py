"""Classic HOG feature and model candidates for Task 1."""

from __future__ import annotations

import io
import json
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from fashion.config import ROOT, TASK1_HOG_CACHE_DIR
from fashion.data.images import transform_image_with_mask
from fashion.task1.evaluation import TASK1_NUM_CLASSES
from fashion.train.artifacts import atomic_write_bytes, canonical_json_bytes, canonical_sha256

HOG_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Task1HogSpec:
    hog_id: str
    pixels_per_cell: tuple[int, int]
    expected_features: int
    orientations: int = 9
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True
    image_size: tuple[int, int] = (80, 60)
    pad_color: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        if not self.hog_id.strip():
            raise ValueError("hog_id must not be blank")
        height, width = self.image_size
        cell_y, cell_x = self.pixels_per_cell
        block_y, block_x = self.cells_per_block
        if min(height, width, cell_y, cell_x, block_y, block_x, self.orientations) <= 0:
            raise ValueError("HOG geometry values must be positive")
        blocks_y, blocks_x = height // cell_y - block_y + 1, width // cell_x - block_x + 1
        calculated = blocks_y * blocks_x * block_y * block_x * self.orientations
        if blocks_y <= 0 or blocks_x <= 0 or calculated != self.expected_features:
            raise ValueError("HOG geometry does not match expected feature count")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TASK1_HOG_COARSE = Task1HogSpec("task1_gray_hog_ppc16_v1", (16, 16), 288)
TASK1_HOG_FINE = Task1HogSpec("task1_gray_hog_ppc10_v1", (10, 10), 1260)
TASK1_HOG_SPECS = (TASK1_HOG_COARSE, TASK1_HOG_FINE)


@dataclass(frozen=True)
class Task1KNNConfig:
    n_neighbors: Literal[3, 5, 11]
    weights: Literal["uniform", "distance"]
    metric: Literal["euclidean"] = "euclidean"

    def __post_init__(self) -> None:
        if self.n_neighbors not in {3, 5, 11}:
            raise ValueError("KNN neighbours must be one of 3, 5, or 11")
        if self.weights not in {"uniform", "distance"} or self.metric != "euclidean":
            raise ValueError("KNN must use approved voting and Euclidean distance")

    @property
    def config_id(self) -> str:
        return f"knn-k{self.n_neighbors}-{self.weights}"


@dataclass(frozen=True)
class Task1LinearSVMConfig:
    C: Literal[0.1, 1.0, 10.0]
    class_weight: Literal["balanced"] | None
    random_state: int = 2753
    dual: Literal["auto"] = "auto"
    tol: float = 1e-4
    max_iter: int = 5000

    def __post_init__(self) -> None:
        if self.C not in {0.1, 1.0, 10.0}:
            raise ValueError("Linear SVM C must be one of 0.1, 1, or 10")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("Linear SVM class_weight must be normal or balanced")
        if self.random_state != 2753 or self.dual != "auto":
            raise ValueError("Linear SVM reproducibility settings are fixed")
        if self.tol != 1e-4 or self.max_iter != 5000:
            raise ValueError("Linear SVM convergence settings are fixed")

    @property
    def config_id(self) -> str:
        weight = "balanced" if self.class_weight else "normal"
        return f"linear-svm-c{self.C:g}-{weight}"


Task1ClassicalModelConfig: TypeAlias = Task1KNNConfig | Task1LinearSVMConfig

TASK1_KNN_GRID = tuple(
    Task1KNNConfig(k, weights) for k in (3, 5, 11) for weights in ("uniform", "distance")
)
TASK1_SVM_GRID = tuple(
    Task1LinearSVMConfig(C, class_weight)
    for C in (0.1, 1.0, 10.0)
    for class_weight in (None, "balanced")
)
TASK1_DEFAULT_KNN = Task1KNNConfig(5, "distance")
TASK1_DEFAULT_SVM = Task1LinearSVMConfig(1.0, None)


@dataclass(frozen=True)
class Task1HogFeatureSet:
    ids: np.ndarray
    features: np.ndarray
    spec: Task1HogSpec
    cache_path: Path


def _expand_observed_scores(
    observed_scores: np.ndarray, observed_classes: np.ndarray
) -> np.ndarray:
    scores = np.asarray(observed_scores, dtype=np.float64)
    classes = np.asarray(observed_classes)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("observed scores must be a finite matrix")
    if classes.ndim != 1 or len(classes) != scores.shape[1]:
        raise ValueError("observed classes must align with score columns")
    if not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("observed classes must be integers")
    classes = classes.astype(np.int64, copy=False)
    if np.any(classes < 0) or np.any(classes >= TASK1_NUM_CLASSES):
        raise ValueError("observed classes must be valid Task 1 indexes")
    if len(np.unique(classes)) != len(classes):
        raise ValueError("observed classes must be unique")
    if not np.allclose(scores.sum(axis=1), 1.0):
        raise ValueError("observed probability rows must sum to one")
    expanded = np.zeros((scores.shape[0], TASK1_NUM_CLASSES), dtype=np.float64)
    expanded[:, classes] = scores
    return expanded


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("SVM decision scores must be a finite matrix")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def query_task1_neighbors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    max_k: int,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Euclidean validation neighbours in bounded batches."""
    train = np.asarray(x_train)
    labels = np.asarray(y_train)
    validation = np.asarray(x_validation)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("training and validation features must be aligned matrices")
    if len(train) != len(labels) or not 0 < max_k <= len(train):
        raise ValueError("max_k must be positive and no larger than the training set")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = KNeighborsClassifier(
        n_neighbors=max_k,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    ).fit(train, labels)
    distance_batches: list[np.ndarray] = []
    index_batches: list[np.ndarray] = []
    for start in range(0, len(validation), batch_size):
        distances, indexes = model.kneighbors(validation[start : start + batch_size])
        distance_batches.append(distances)
        index_batches.append(indexes)
    return np.vstack(distance_batches), np.vstack(index_batches)


def knn_probabilities_from_neighbors(
    distances: np.ndarray,
    indexes: np.ndarray,
    y_train: np.ndarray,
    *,
    n_neighbors: int,
    weights: Literal["uniform", "distance"],
) -> np.ndarray:
    """Convert reusable nearest-neighbour results to fixed-class probabilities."""
    neighbour_distances = np.asarray(distances, dtype=np.float64)
    neighbour_indexes = np.asarray(indexes)
    labels = np.asarray(y_train)
    if neighbour_distances.ndim != 2 or neighbour_indexes.shape != neighbour_distances.shape:
        raise ValueError("neighbour distances and indexes must be aligned matrices")
    if not 0 < n_neighbors <= neighbour_distances.shape[1]:
        raise ValueError("n_neighbors must be available in the neighbour results")
    if weights not in {"uniform", "distance"}:
        raise ValueError("KNN weights must be uniform or distance")
    if not np.isfinite(neighbour_distances).all() or np.any(neighbour_distances < 0):
        raise ValueError("neighbour distances must be finite and non-negative")
    if not np.issubdtype(neighbour_indexes.dtype, np.integer):
        raise ValueError("neighbour indexes must be integers")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("training labels must be integers")
    if np.any(neighbour_indexes < 0) or np.any(neighbour_indexes >= len(labels)):
        raise ValueError("neighbour indexes must reference training labels")
    selected_distances = neighbour_distances[:, :n_neighbors]
    selected_labels = labels[neighbour_indexes[:, :n_neighbors]].astype(np.int64, copy=False)
    if np.any(selected_labels < 0) or np.any(selected_labels >= TASK1_NUM_CLASSES):
        raise ValueError("training labels must be valid Task 1 indexes")
    if weights == "uniform":
        vote_weights = np.ones_like(selected_distances)
    else:
        zero_distance = selected_distances == 0.0
        vote_weights = np.where(
            zero_distance.any(axis=1, keepdims=True), zero_distance, 1.0 / selected_distances
        )
    probabilities = np.zeros((len(selected_distances), TASK1_NUM_CLASSES), dtype=np.float64)
    np.add.at(
        probabilities,
        (np.arange(len(probabilities))[:, None], selected_labels),
        vote_weights,
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def fit_predict_task1_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    config: Task1KNNConfig,
) -> tuple[KNeighborsClassifier, np.ndarray]:
    model = KNeighborsClassifier(
        n_neighbors=config.n_neighbors,
        weights=config.weights,
        metric=config.metric,
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    distances, indexes = query_task1_neighbors(
        x_train, y_train, x_validation, max_k=config.n_neighbors, batch_size=512
    )
    probabilities = knn_probabilities_from_neighbors(
        distances, indexes, y_train, n_neighbors=config.n_neighbors, weights=config.weights
    )
    return model, probabilities


def fit_predict_task1_linear_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    config: Task1LinearSVMConfig,
) -> tuple[LinearSVC, np.ndarray]:
    model = LinearSVC(
        C=config.C,
        class_weight=config.class_weight,
        random_state=config.random_state,
        dual=config.dual,
        tol=config.tol,
        max_iter=config.max_iter,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
    decision = np.asarray(model.decision_function(x_validation), dtype=np.float64)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    observed = _stable_softmax(decision)
    return model, _expand_observed_scores(observed, model.classes_)


def extract_task1_hog(path: str | Path, spec: Task1HogSpec) -> np.ndarray:
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            rgb, _ = transform_image_with_mask(
                source, image_size=spec.image_size, pad_color=spec.pad_color, normalize_range=True
            )
    except Exception as error:
        raise ValueError(f"cannot extract HOG from {image_path}") from error
    features = hog(
        rgb2gray(np.asarray(rgb, dtype=np.float32)),
        orientations=spec.orientations,
        pixels_per_cell=spec.pixels_per_cell,
        cells_per_block=spec.cells_per_block,
        block_norm=spec.block_norm,
        transform_sqrt=spec.transform_sqrt,
        feature_vector=True,
        channel_axis=None,
    ).astype(np.float32, copy=False)
    if features.shape != (spec.expected_features,) or not np.isfinite(features).all():
        raise ValueError(f"invalid HOG feature vector for {image_path}")
    return features


def _normalized_development_hog_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "path", "sha256", "partition"}
    if required.difference(rows.columns):
        raise ValueError("HOG rows are missing required inventory columns")
    if rows.empty or not rows["partition"].eq("development").all():
        raise ValueError("HOG cache requires unique development rows")
    if rows["id"].isna().any():
        raise ValueError("HOG cache requires unique development rows")
    normalized = rows.copy()
    try:
        normalized["id"] = normalized["id"].to_numpy(dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("HOG cache requires integer development row IDs") from error
    if normalized["id"].duplicated().any():
        raise ValueError("HOG cache requires unique development rows")
    return normalized.sort_values("id", kind="stable")


def _hog_inventory(rows: pd.DataFrame) -> list[dict[str, object]]:
    ordered = _normalized_development_hog_rows(rows)
    return ordered.loc[:, ["id", "path", "sha256"]].to_dict(orient="records")


def load_or_build_task1_hog_features(
    rows: pd.DataFrame,
    spec: Task1HogSpec,
    *,
    root: str | Path = ROOT,
    cache_root: str | Path = TASK1_HOG_CACHE_DIR,
    split_sha256: str,
    extractor: Callable[[Path, Task1HogSpec], np.ndarray] = extract_task1_hog,
) -> Task1HogFeatureSet:
    ordered = _normalized_development_hog_rows(rows)
    inventory = _hog_inventory(ordered)
    identity = {
        "schema_version": HOG_CACHE_SCHEMA_VERSION,
        "split_sha256": split_sha256,
        "inventory_sha256": canonical_sha256(inventory),
        "hog": spec.to_dict(),
    }
    cache_path = Path(cache_root) / f"{spec.hog_id}-{canonical_sha256(identity)[:16]}.npz"
    expected_ids = np.asarray([int(row["id"]) for row in inventory], dtype=np.int64)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as stored:
                ids = stored["ids"]
                features = stored["features"]
                metadata = json.loads(bytes(stored["metadata"].tolist()).decode("utf-8"))
            valid = (
                ids.dtype == np.int64
                and features.dtype == np.float32
                and canonical_json_bytes(metadata) == canonical_json_bytes(identity)
                and np.array_equal(ids, expected_ids)
                and features.shape == (len(ids), spec.expected_features)
                and np.isfinite(features).all()
            )
            if valid:
                return Task1HogFeatureSet(ids, features, spec, cache_path)
        except Exception:
            pass

    ids = ordered["id"].to_numpy(dtype=np.int64)
    project_root = Path(root)
    vectors = [
        np.asarray(extractor(project_root / str(row.path), spec), dtype=np.float32)
        for row in ordered.itertuples()
    ]
    features = np.stack(vectors).astype(np.float32, copy=False)
    if features.shape != (len(ids), spec.expected_features) or not np.isfinite(features).all():
        raise ValueError("invalid Task 1 HOG cache features")
    metadata_bytes = canonical_json_bytes(identity)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        ids=ids,
        features=features,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
    )
    atomic_write_bytes(cache_path, buffer.getvalue())
    return Task1HogFeatureSet(ids, features, spec, cache_path)
