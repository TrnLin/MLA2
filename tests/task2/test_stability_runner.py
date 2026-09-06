from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
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
    validate_stability_implementation,
    validate_stability_pair,
)
from fashion.train.artifacts import canonical_sha256


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
    monkeypatch.setattr(
        stability,
        "validate_stability_implementation",
        lambda *args, **kwargs: {},
    )

    outputs = run_stability_matrix(pair, mode="load")

    assert outputs == ["c2-output", "i2-output"]
    assert observed == [
        ("c2", C2_STABILITY_EXPERIMENT_ID, {"mode": "load"}),
        ("i2", I2_STABILITY_EXPERIMENT_ID, {"mode": "load"}),
    ]


def test_stability_preflight_matches_five_primary_folds_per_candidate(
    monkeypatch,
) -> None:
    pair = load_stability_pair()
    primary_c2 = stability.load_experiment_config(stability.C2_PRIMARY_CONFIG_PATH)
    primary_i2 = stability.load_i2_config(stability.I2_PRIMARY_CONFIG_PATH)
    rows = []
    for candidate, config, implementation_hash in (
        ("C2", primary_c2, "c" * 64),
        ("I2", primary_i2, "d" * 64),
    ):
        for fold in range(5):
            rows.append(
                {
                    "run_id": f"{candidate}-f{fold}",
                    "experiment_id": config.experiment_id,
                    "status": "completed",
                    "seed": 2753,
                    "fold": fold,
                    "config_sha256": canonical_sha256(config.to_dict()),
                    "split_sha256": "a" * 64,
                    "label_map_sha256": "b" * 64,
                    "implementation_sha256": implementation_hash,
                    "git_dirty": False,
                }
            )
    registry = pd.DataFrame(rows)

    def fake_key(config, **kwargs):
        del kwargs
        implementation_hash = (
            "c" * 64 if config["model_family"] == "resnet18_small_stem" else "d" * 64
        )
        return SimpleNamespace(
            implementation_sha256=implementation_hash,
            split_sha256="a" * 64,
            label_map_sha256="b" * 64,
        )

    monkeypatch.setattr(
        stability.RunRegistry,
        "read",
        lambda self: registry,
    )
    monkeypatch.setattr(stability, "build_run_cache_key", fake_key)

    audit = validate_stability_implementation(pair)

    assert set(audit) == {"C2", "I2"}
    assert len(audit["C2"]["primary_run_ids"]) == 5
    assert len(audit["I2"]["primary_run_ids"]) == 5


def test_stability_matrix_blocks_before_training_when_primary_hashes_drift(
    monkeypatch,
) -> None:
    pair = load_stability_pair()
    runner_called = False

    def reject_drift(*args, **kwargs):
        del args, kwargs
        raise ValueError("stability implementation hash mismatch")

    def unexpected_runner(*args, **kwargs):
        nonlocal runner_called
        del args, kwargs
        runner_called = True
        return []

    monkeypatch.setattr(
        stability,
        "validate_stability_implementation",
        reject_drift,
        raising=False,
    )
    monkeypatch.setattr(stability, "run_or_load_experiment", unexpected_runner)
    monkeypatch.setattr(
        stability,
        "_run_stability_i2_experiment",
        unexpected_runner,
    )

    with pytest.raises(ValueError, match="implementation hash mismatch"):
        run_stability_matrix(pair, mode="run")
    assert runner_called is False
