import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import fashion.task1.classical_models as classical_models
from fashion.task1.classical_models import (
    Task1KNNConfig,
    Task1LinearSVMConfig,
    _stable_softmax,
    fit_predict_task1_knn,
    fit_predict_task1_linear_svm,
    knn_probabilities_from_neighbors,
    query_task1_neighbors,
)


def test_classical_model_module_has_no_project_io_dependencies() -> None:
    source = Path(classical_models.__file__).read_text(encoding="utf-8")
    assert "RunRegistry" not in source
    assert "atomic_write" not in source
    assert "load_splits" not in source


def _toy_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.array([[0, 0], [0, 1], [4, 4], [4, 5], [8, 8], [8, 9]], dtype=np.float32)
    y_train = np.array([2, 2, 7, 7, 19, 19], dtype=np.int64)
    x_validation = np.array([[0, 0.2], [4, 4.2], [8, 8.2]], dtype=np.float32)
    return x_train, y_train, x_validation


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_scores_expand_missing_classes_and_sum_to_one(weights: str) -> None:
    x_train, y_train, x_validation = _toy_features()
    config = Task1KNNConfig(n_neighbors=3, weights=weights)
    _, probabilities = fit_predict_task1_knn(x_train, y_train, x_validation, config)
    assert probabilities.shape == (3, 124)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, [0, 1, 3, 6, 8, 18, 20]] == 0.0)


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_linear_svm_scores_expand_missing_classes_and_are_finite(
    class_weight: str | None,
) -> None:
    x_train, y_train, x_validation = _toy_features()
    config = Task1LinearSVMConfig(C=1.0, class_weight=class_weight)
    _, probabilities = fit_predict_task1_linear_svm(x_train, y_train, x_validation, config)
    assert probabilities.shape == (3, 124)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, [0, 1, 3, 6, 8, 18, 20]] == 0.0)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("k", [3, 5])
def test_reused_neighbour_votes_match_sklearn(weights: str, k: int) -> None:
    x_train, y_train, x_validation = _toy_features()
    distances, indexes = query_task1_neighbors(x_train, y_train, x_validation, max_k=5)
    reused = knn_probabilities_from_neighbors(
        distances, indexes, y_train, n_neighbors=k, weights=weights
    )
    model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="euclidean")
    expected_local = model.fit(x_train, y_train).predict_proba(x_validation)
    expected = np.zeros((len(x_validation), 124))
    expected[:, model.classes_.astype(int)] = expected_local
    np.testing.assert_allclose(reused, expected)


def test_distance_votes_with_exact_matches_do_not_warn_or_divide_by_zero() -> None:
    distances = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    indexes = np.array([[0, 1, 2]], dtype=np.int64)
    labels = np.array([0, 1, 2], dtype=np.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probabilities = knn_probabilities_from_neighbors(
            distances, indexes, labels, n_neighbors=3, weights="distance"
        )

    np.testing.assert_allclose(probabilities[0, 0], 1.0)


def test_stable_softmax_handles_extreme_finite_logits() -> None:
    logits = np.array([[1_000.0, 0.0, -1_000.0], [-1_000.0, 0.0, 1_000.0]])

    probabilities = _stable_softmax(logits)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
