"""Explicit identities for comparable Task 1 scratch-CNN experiments."""

from dataclasses import dataclass

from fashion.task1.losses import (
    TASK1_GENTLE_WEIGHTED_LOSS,
    TASK1_UNWEIGHTED_LOSS,
    Task1LossConfig,
)
from fashion.task1.preprocessing import (
    DEFAULT_TASK1_PREPROCESSING,
    TASK1_CONTROL_PREPROCESSING,
    Task1PreprocessingConfig,
)


@dataclass(frozen=True)
class Task1CnnCandidate:
    candidate_id: str
    preprocessing: Task1PreprocessingConfig
    loss: Task1LossConfig

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")


TASK1_NO_AUG_CANDIDATE = Task1CnnCandidate(
    "task1_cnn_no_aug_unweighted_v1", TASK1_CONTROL_PREPROCESSING, TASK1_UNWEIGHTED_LOSS
)
TASK1_MILD_AUG_CANDIDATE = Task1CnnCandidate(
    "task1_cnn_mild_aug_unweighted_v1", DEFAULT_TASK1_PREPROCESSING, TASK1_UNWEIGHTED_LOSS
)
TASK1_GENTLE_WEIGHTED_CANDIDATE = Task1CnnCandidate(
    "task1_cnn_no_aug_sqrt_weighted_v1", TASK1_CONTROL_PREPROCESSING, TASK1_GENTLE_WEIGHTED_LOSS
)
