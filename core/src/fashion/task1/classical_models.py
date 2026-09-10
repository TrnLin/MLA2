"""Pure classical estimators and fixed-class scoring for Task 1."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from fashion.task1.evaluation import TASK1_NUM_CLASSES


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
        inverse_distances = np.zeros_like(selected_distances)
        np.divide(1.0, selected_distances, out=inverse_distances, where=~zero_distance)
        vote_weights = np.where(
            zero_distance.any(axis=1, keepdims=True), zero_distance, inverse_distances
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
    *,
    batch_size: int = 512,
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
        x_train, y_train, x_validation, max_k=config.n_neighbors, batch_size=batch_size
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
