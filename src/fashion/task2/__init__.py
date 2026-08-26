"""Season-classification baselines, experiments, evidence, and inference."""

from fashion.task2.baselines import (
    MajorityBaselineModel,
    MajorityFoldResult,
    evaluate_majority_fold,
    fit_training_fold_majority,
)

__all__ = [
    "MajorityBaselineModel",
    "MajorityFoldResult",
    "evaluate_majority_fold",
    "fit_training_fold_majority",
]
