from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from fashion.task2 import class_balance, experiments
from fashion.task2.class_balance import (
    I1_DATA_CONFIG,
    I1_EXPERIMENT_ID,
    I1_OPTIMISATION_CONFIG,
    I1_STAGE,
)
from fashion.task2.experiments import ExperimentConfig
from fashion.train.losses import (
    EFFECTIVE_NUMBER_LOSS_ID,
    fit_effective_number_weights,
)
from fashion.train.metrics import SEASON_LABELS


def _legacy_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="seed-regression",
        method="deep",
        model_family="smallcnn",
        stage="test",
        folds=(0,),
        seeds=(2753,),
    )


def _i1_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=I1_EXPERIMENT_ID,
        method="deep",
        model_family="smallcnn",
        stage=I1_STAGE,
        folds=tuple(range(5)),
        seeds=(2753,),
        loss_id=EFFECTIVE_NUMBER_LOSS_ID,
        data=I1_DATA_CONFIG,
        optimisation=I1_OPTIMISATION_CONFIG,
    )


@pytest.mark.parametrize(
    ("module", "executor", "config"),
    [
        (experiments, experiments._execute_deep, _legacy_config()),
        (class_balance, class_balance._execute_i1, _i1_config()),
    ],
    ids=("legacy-deep", "i1-class-balanced"),
)
def test_declared_seed_controls_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module,
    executor,
    config: ExperimentConfig,
) -> None:
    loaders = SimpleNamespace(
        train=(),
        validation=(),
        labels=SEASON_LABELS,
        validation_ids=(1,),
        audit=lambda: {},
    )
    monkeypatch.setattr(module, "build_task_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(module, "_validate_output_oof", lambda *args, **kwargs: {})
    if module is class_balance:
        audit = fit_effective_number_weights(
            SEASON_LABELS,
            SEASON_LABELS,
            training_ids=(1, 2, 3, 4),
        )
        monkeypatch.setattr(module, "fit_i1_fold_balance", lambda *args, **kwargs: audit)

    initial_parameters: list[torch.Tensor] = []

    def capture_initial_parameters(model, *args, **kwargs):
        del args, kwargs
        initial_parameters.append(next(model.parameters()).detach().cpu().clone())
        return SimpleNamespace(
            to_oof_frame=lambda: pd.DataFrame(),
            history=[],
            metadata={},
        )

    monkeypatch.setattr(module, "train_fold", capture_initial_parameters)
    call = {
        "config": config,
        "fold": 0,
        "seed": 2753,
        "checkpoint_path": tmp_path / "model.pt",
        "data_root": tmp_path,
        "splits_path": tmp_path / "splits.csv",
        "label_map_path": tmp_path / "label_maps.json",
    }

    torch.manual_seed(1)
    executor(**call)
    torch.manual_seed(2)
    executor(**call)

    assert torch.equal(initial_parameters[0], initial_parameters[1])
