"""One stronger-dropout trial: p=0.45 with the completed 04aa darkening policy."""

from fashion.train.task3_gender_dropout_darkening import (
    DARKENING_RUN_IDS as PARENT_RUN_IDS,
)
from fashion.train.task3_gender_dropout_darkening import (
    STRONGER_NAME as NAME,
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
    "stronger_dropout_config",
    "check_gender_stronger_dropout_sources",
    "evaluate_gender_stronger_dropout_screen",
    "run_gender_stronger_dropout_screen",
]


def stronger_dropout_config(spec, *, fold, device_name):
    if spec.name != NAME:
        raise ValueError("The stronger-dropout screen requires the p=0.45 darkening recipe")
    return dropout_darkening_config(spec, fold=fold, device_name=device_name)


def check_gender_stronger_dropout_sources(**kwargs):
    return check_gender_dropout_darkening_sources(**kwargs, experiment_name=NAME)


def evaluate_gender_stronger_dropout_screen(child, sources, classes, *, repetitions=10_000):
    return evaluate_gender_narrow_screen(
        child,
        sources,
        classes,
        repetitions=repetitions,
        experiment_name=NAME,
    )


def run_gender_stronger_dropout_screen(**kwargs):
    """Reuse registered scratch training, direct-parent checks and the unchanged gates."""
    return run_gender_dropout_darkening_screen(**kwargs, experiment_name=NAME)
