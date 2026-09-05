"""Full-width G2 with only post-GeM dropout p=0.30 changed during training."""

from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_gender_narrow import (
    FOLDS,
    check_gender_narrow_sources,
    evaluate_gender_narrow_screen,
    run_gender_narrow_screen,
)

NAME = "gender_dropout_030"


def dropout_config(spec, *, fold, device_name):
    """Dropout lives in the frozen child spec; all G2 baseline controls stay intact."""
    if spec.to_dict() != dataset_v2_spec(NAME, spec.parent_run_ids).to_dict():
        raise ValueError("Gender dropout requires the frozen full-width G2 recipe")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("Gender dropout requires CUDA and only folds 0 and 4")
    return Task3BaselineConfig(target="gender")


def check_gender_dropout_sources(**kwargs):
    return check_gender_narrow_sources(**kwargs, experiment_name=NAME)


def evaluate_gender_dropout_screen(child, sources, classes, *, repetitions=10_000):
    return evaluate_gender_narrow_screen(
        child, sources, classes, repetitions=repetitions, experiment_name=NAME
    )


def run_gender_dropout_screen(**kwargs):
    """Reuse the same registered two-fold trainer, IEEE comparisons and frozen gates."""
    return run_gender_narrow_screen(**kwargs, experiment_name=NAME)
