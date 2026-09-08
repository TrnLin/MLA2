"""Compatibility facade for Task 1 classical HOG models and fold training."""

from fashion.task1.classical_features import (
    HOG_CACHE_SCHEMA_VERSION,
    TASK1_HOG_COARSE,
    TASK1_HOG_FINE,
    TASK1_HOG_SPECS,
    Task1HogFeatureSet,
    Task1HogSpec,
    extract_task1_hog,
    load_or_build_task1_hog_features,
)
from fashion.task1.classical_models import (
    TASK1_DEFAULT_KNN,
    TASK1_DEFAULT_SVM,
    TASK1_KNN_GRID,
    TASK1_SVM_GRID,
    Task1ClassicalModelConfig,
    Task1KNNConfig,
    Task1LinearSVMConfig,
    fit_predict_task1_knn,
    fit_predict_task1_linear_svm,
    knn_probabilities_from_neighbors,
    query_task1_neighbors,
)
from fashion.task1.classical_training import (
    Task1ClassicalFoldResult,
    Task1ClassicalRunConfig,
    run_task1_classical_fold,
)

__all__ = [
    "HOG_CACHE_SCHEMA_VERSION",
    "TASK1_HOG_COARSE",
    "TASK1_HOG_FINE",
    "TASK1_HOG_SPECS",
    "Task1HogFeatureSet",
    "Task1HogSpec",
    "extract_task1_hog",
    "load_or_build_task1_hog_features",
    "TASK1_DEFAULT_KNN",
    "TASK1_DEFAULT_SVM",
    "TASK1_KNN_GRID",
    "TASK1_SVM_GRID",
    "Task1ClassicalModelConfig",
    "Task1KNNConfig",
    "Task1LinearSVMConfig",
    "fit_predict_task1_knn",
    "fit_predict_task1_linear_svm",
    "knn_probabilities_from_neighbors",
    "query_task1_neighbors",
    "Task1ClassicalFoldResult",
    "Task1ClassicalRunConfig",
    "run_task1_classical_fold",
]
