from __future__ import annotations

from dataclasses import replace

import pytest

from fashion.task2 import stability
from fashion.task2.stability import (
    C2_STABILITY_EXPERIMENT_ID,
    G5_SEED,
    G5_STAGE,
    I2_STABILITY_EXPERIMENT_ID,
    StabilityConfigPair,
    load_stability_pair,
    run_stability_matrix,
    validate_stability_pair,
)


def test_frozen_stability_pair_contains_only_eligible_seed_2026_candidates() -> None:
    pair = load_stability_pair()

    assert pair.c2.experiment_id == C2_STABILITY_EXPERIMENT_ID
    assert pair.i2.experiment_id == I2_STABILITY_EXPERIMENT_ID
    assert pair.c2.stage == pair.i2.stage == G5_STAGE
    assert pair.c2.seeds == pair.i2.seeds == (G5_SEED,)
    assert pair.c2.model_family == "resnet18_small_stem"
    assert pair.i2.model_family == "smallcnn"
    assert pair.i2.auxiliary.loss_weight == 0.3
    assert "pretrained" not in pair.c2.model_family
    assert "pretrained" not in pair.i2.model_family


def test_stability_pair_rejects_seed_drift() -> None:
    pair = load_stability_pair()
    drifted = StabilityConfigPair(
        c2=replace(pair.c2, seeds=(2753,)),
        i2=pair.i2,
    )

    with pytest.raises(ValueError, match="c2_seeds"):
        validate_stability_pair(drifted)


def test_stability_pair_rejects_auxiliary_weight_drift() -> None:
    pair = load_stability_pair()
    drifted = StabilityConfigPair(
        c2=pair.c2,
        i2=replace(
            pair.i2,
            auxiliary=replace(pair.i2.auxiliary, loss_weight=0.1),
        ),
    )

    with pytest.raises(ValueError, match="G5 I2 changes"):
        validate_stability_pair(drifted)


def test_stability_matrix_validates_pair_and_runs_c2_before_i2(monkeypatch) -> None:
    pair = load_stability_pair()
    observed: list[tuple[str, str, dict]] = []

    def fake_c2(config, **kwargs):
        observed.append(("c2", config.experiment_id, kwargs))
        return ["c2-output"]

    def fake_i2(config, **kwargs):
        observed.append(("i2", config.experiment_id, kwargs))
        return ["i2-output"]

    monkeypatch.setattr(stability, "run_or_load_experiment", fake_c2)
    monkeypatch.setattr(stability, "_run_stability_i2_experiment", fake_i2)

    outputs = run_stability_matrix(pair, mode="load")

    assert outputs == ["c2-output", "i2-output"]
    assert observed == [
        ("c2", C2_STABILITY_EXPERIMENT_ID, {"mode": "load"}),
        ("i2", I2_STABILITY_EXPERIMENT_ID, {"mode": "load"}),
    ]
