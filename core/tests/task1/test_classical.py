"""Public compatibility checks for the former classical module."""

import fashion.task1.classical as facade
import fashion.task1.classical_features as features
import fashion.task1.classical_models as models
import fashion.task1.classical_training as training


def test_classical_facade_reexports_all_public_symbols_from_owners() -> None:
    feature_names = (
        "HOG_CACHE_SCHEMA_VERSION",
        "TASK1_HOG_COARSE",
        "TASK1_HOG_FINE",
        "TASK1_HOG_SPECS",
        "Task1HogFeatureSet",
        "Task1HogSpec",
        "extract_task1_hog",
        "load_or_build_task1_hog_features",
    )
    model_names = (
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
    )
    training_names = (
        "Task1ClassicalFoldResult",
        "Task1ClassicalRunConfig",
        "run_task1_classical_fold",
    )

    for name in feature_names:
        assert getattr(facade, name) is getattr(features, name)
    for name in model_names:
        assert getattr(facade, name) is getattr(models, name)
    for name in training_names:
        assert getattr(facade, name) is getattr(training, name)
    assert "_stable_softmax" not in facade.__all__
    assert "_classical_implementation_sha256" not in facade.__all__
