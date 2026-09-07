"""One grayscale trial: dropout 0.30, unchanged darkening, grayscale probability 0.10."""

from fashion.train.task3_gender_dropout_darkening import (
    DARKENING_RUN_IDS as PARENT_RUN_IDS,
)
from fashion.train.task3_gender_dropout_darkening import (
    GRAYSCALE_NAME as NAME,
)
from fashion.train.task3_gender_dropout_darkening import (
    check_gender_dropout_darkening_sources,
    dropout_darkening_config,
    run_gender_dropout_darkening_screen,
)
from fashion.train.task3_gender_narrow import evaluate_gender_narrow_screen

__all__ = [
    "NAME",
    "PARENT_RUN_IDS",
    "grayscale_config",
    "check_gender_grayscale_sources",
    "evaluate_gender_grayscale_screen",
    "run_gender_grayscale_screen",
]


def grayscale_config(spec, *, fold, device_name):
    if spec.name != NAME:
        raise ValueError("The grayscale screen requires the p=0.10 recipe with dropout 0.30")
    return dropout_darkening_config(spec, fold=fold, device_name=device_name)


def check_gender_grayscale_sources(**kwargs):
    return check_gender_dropout_darkening_sources(**kwargs, experiment_name=NAME)


def evaluate_gender_grayscale_screen(child, sources, classes, *, repetitions=10_000):
    return evaluate_gender_narrow_screen(
        child, sources, classes, repetitions=repetitions, experiment_name=NAME
    )


def run_gender_grayscale_screen(**kwargs):
    """Use registered scratch training, exact 04aa parents and unchanged screen gates."""
    return run_gender_dropout_darkening_screen(**kwargs, experiment_name=NAME)
