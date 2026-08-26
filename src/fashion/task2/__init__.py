"""Season-classification baselines, experiments, evidence, and inference."""

from fashion.task2.baselines import (
    MajorityBaselineModel,
    MajorityFoldResult,
    evaluate_majority_fold,
    fit_training_fold_majority,
)
from fashion.task2.classical import (
    HogHsvFoldResult,
    HogHsvSpec,
    HogHsvSvmModel,
    evaluate_hog_hsv_svm_fold,
    extract_hog_hsv,
    fit_hog_hsv_svm,
)
from fashion.task2.experiments import (
    ExperimentConfig,
    ExperimentFoldOutput,
    load_experiment_config,
    run_matrix,
    run_or_load_experiment,
)

__all__ = [
    "MajorityBaselineModel",
    "MajorityFoldResult",
    "evaluate_majority_fold",
    "evaluate_hog_hsv_svm_fold",
    "extract_hog_hsv",
    "ExperimentConfig",
    "ExperimentFoldOutput",
    "fit_training_fold_majority",
    "fit_hog_hsv_svm",
    "HogHsvFoldResult",
    "HogHsvSpec",
    "HogHsvSvmModel",
    "load_experiment_config",
    "run_matrix",
    "run_or_load_experiment",
]
