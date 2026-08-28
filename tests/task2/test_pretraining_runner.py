from __future__ import annotations

from dataclasses import replace

import pytest

from fashion.config import ROOT
from fashion.task2.experiments import load_experiment_config
from fashion.task2.pretraining import (
    P0S_EXPERIMENT_ID,
    PSTAR_EXPERIMENT_ID,
    run_pretraining_matrix,
    validate_pretraining_config,
    validate_pretraining_pair,
)

CONFIG_PATHS = (
    ROOT / "configs/task2/g4_p0s_resnet18_standard_scratch.json",
    ROOT / "configs/task2/g4_pstar_resnet18_standard_pretrained.json",
)


def _configs():
    return tuple(load_experiment_config(path) for path in CONFIG_PATHS)


def test_frozen_pretraining_pair_is_canonical_and_returned_scratch_first() -> None:
    scratch, pretrained = validate_pretraining_pair(tuple(reversed(_configs())))

    assert scratch.experiment_id == P0S_EXPERIMENT_ID
    assert scratch.model_family == "resnet18_standard_scratch"
    assert pretrained.experiment_id == PSTAR_EXPERIMENT_ID
    assert pretrained.model_family == "resnet18_standard_pretrained"


def test_pretraining_protocol_rejects_optimizer_drift() -> None:
    scratch, _ = _configs()
    drifted = replace(
        scratch,
        optimisation=replace(scratch.optimisation, learning_rate=1e-3),
    )

    with pytest.raises(ValueError, match="optimisation"):
        validate_pretraining_config(drifted)


def test_pretraining_pair_rejects_missing_control() -> None:
    scratch, _ = _configs()

    with pytest.raises(ValueError, match=r"exactly one P0S and one P\*"):
        validate_pretraining_pair((scratch, scratch))


def test_runner_validates_and_orders_pair_before_delegating(monkeypatch) -> None:
    observed = {}

    def fake_run_matrix(configs, **kwargs):
        observed["ids"] = [config.experiment_id for config in configs]
        observed["kwargs"] = kwargs
        return []

    monkeypatch.setattr("fashion.task2.pretraining.run_matrix", fake_run_matrix)

    outputs = run_pretraining_matrix(
        tuple(reversed(CONFIG_PATHS)),
        mode="load",
    )

    assert outputs == []
    assert observed == {
        "ids": [P0S_EXPERIMENT_ID, PSTAR_EXPERIMENT_ID],
        "kwargs": {"mode": "load"},
    }
