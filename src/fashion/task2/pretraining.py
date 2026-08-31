"""Strict orchestration for the matched P0S/P* pretraining benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from fashion.task2.experiments import (
    DataRunConfig,
    ExperimentConfig,
    ExperimentFoldOutput,
    OptimisationRunConfig,
    load_experiment_config,
    run_matrix,
)

P0S_EXPERIMENT_ID = "g4-p0s-resnet18-standard-scratch"
PSTAR_EXPERIMENT_ID = "g4-pstar-resnet18-standard-pretrained"
PRETRAINING_STAGE = "g4_pretraining_benchmark"
PRETRAINING_FAMILIES = {
    P0S_EXPERIMENT_ID: "resnet18_standard_scratch",
    PSTAR_EXPERIMENT_ID: "resnet18_standard_pretrained",
}
PRETRAINING_DATA_CONFIG = DataRunConfig(
    image_size=(80, 60),
    augmentation="a0",
    batch_size=32,
    validation_batch_size=128,
    num_workers=4,
    pin_memory=True,
)
PRETRAINING_OPTIMISATION_CONFIG = OptimisationRunConfig(
    epochs=30,
    learning_rate=3e-4,
    weight_decay=1e-4,
    effective_batch_size=128,
    gradient_clip_norm=1.0,
    warmup_epochs=1.0,
    patience=5,
    min_delta=1e-4,
    use_amp=True,
    device="auto",
)


def validate_pretraining_config(config: ExperimentConfig) -> None:
    """Reject drift from the frozen matched benchmark protocol."""
    config.validate()
    expected_family = PRETRAINING_FAMILIES.get(config.experiment_id)
    if expected_family is None:
        raise ValueError(
            f"unknown pretraining benchmark experiment_id: {config.experiment_id}"
        )
    expected = {
        "method": "deep",
        "model_family": expected_family,
        "stage": PRETRAINING_STAGE,
        "target": "season",
        "folds": tuple(range(5)),
        "seeds": (2753,),
        "loss_id": "cross_entropy",
        "data": PRETRAINING_DATA_CONFIG,
        "optimisation": PRETRAINING_OPTIMISATION_CONFIG,
    }
    observed = {
        "method": config.method,
        "model_family": config.model_family,
        "stage": config.stage,
        "target": config.target,
        "folds": config.folds,
        "seeds": config.seeds,
        "loss_id": config.loss_id,
        "data": config.data,
        "optimisation": config.optimisation,
    }
    mismatches = [name for name, value in expected.items() if observed[name] != value]
    if mismatches:
        raise ValueError(
            f"pretraining benchmark config violates the frozen protocol: {mismatches}"
        )


def validate_pretraining_pair(
    configs: Sequence[ExperimentConfig],
) -> tuple[ExperimentConfig, ExperimentConfig]:
    """Require exactly one P0S and one P* config, returned in causal order."""
    if len(configs) != 2:
        raise ValueError("pretraining benchmark requires exactly the P0S and P* configs")
    by_id = {config.experiment_id: config for config in configs}
    if len(by_id) != 2 or set(by_id) != set(PRETRAINING_FAMILIES):
        raise ValueError("pretraining benchmark requires exactly one P0S and one P* config")
    for config in by_id.values():
        validate_pretraining_config(config)

    scratch = by_id[P0S_EXPERIMENT_ID]
    pretrained = by_id[PSTAR_EXPERIMENT_ID]
    scratch_payload = scratch.to_dict()
    pretrained_payload = pretrained.to_dict()
    for payload in (scratch_payload, pretrained_payload):
        payload.pop("experiment_id")
        payload.pop("model_family")
    if scratch_payload != pretrained_payload:
        raise ValueError("P0S and P* differ outside identity and initial weights")
    return scratch, pretrained


def run_pretraining_matrix(
    configs: Sequence[ExperimentConfig | str | Path],
    **kwargs: Any,
) -> list[ExperimentFoldOutput]:
    """Validate the complete pair before any fold is loaded or trained."""
    resolved = [
        load_experiment_config(config) if isinstance(config, (str, Path)) else config
        for config in configs
    ]
    matched = validate_pretraining_pair(resolved)
    return run_matrix(list(matched), **kwargs)


__all__ = [
    "P0S_EXPERIMENT_ID",
    "PSTAR_EXPERIMENT_ID",
    "PRETRAINING_STAGE",
    "run_pretraining_matrix",
    "validate_pretraining_config",
    "validate_pretraining_pair",
]
