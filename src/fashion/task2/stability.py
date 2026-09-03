"""Frozen second-seed execution for the two eligible Task 2 finalists."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import (
    LABEL_MAPS_JSON,
    ROOT,
    RUNS_CSV,
    SPLITS_CSV,
    TASK2_CHECKPOINT_DIR,
    TASK2_RUN_DIR,
)
from fashion.task2 import experiments as experiment_runner
from fashion.task2 import multitask as multitask_runner
from fashion.task2.experiments import (
    ExecutionMode,
    ExperimentConfig,
    ExperimentFoldOutput,
    load_experiment_config,
    run_or_load_experiment,
)
from fashion.task2.multitask import (
    I2_IMPLEMENTATION_PATHS,
    AuxiliaryRunConfig,
    I2ExperimentConfig,
    load_i2_config,
)
from fashion.train.artifacts import canonical_sha256
from fashion.train.cache import build_run_cache_key
from fashion.train.registry import RunRegistry

G5_STAGE = "g5_seed_stability"
G5_SEED = 2026
C2_PRIMARY_EXPERIMENT_ID = "g3-c2-t0-resnet18"
I2_PRIMARY_EXPERIMENT_ID = "g4-i2-article-type-lambda-0-3-c1"
C2_STABILITY_EXPERIMENT_ID = "g5-c2-t0-resnet18-s2026"
I2_STABILITY_EXPERIMENT_ID = "g5-i2-article-type-lambda-0-3-c1-s2026"
C2_PRIMARY_CONFIG_PATH = ROOT / "configs/task2/g3_c2_t0_resnet18.json"
I2_PRIMARY_CONFIG_PATH = ROOT / "configs/task2/g4_i2_article_type_lambda_0_3_c1.json"
C2_STABILITY_CONFIG_PATH = ROOT / "configs/task2/g5_c2_t0_resnet18_seed_2026.json"
I2_STABILITY_CONFIG_PATH = ROOT / "configs/task2/g5_i2_article_type_lambda_0_3_c1_seed_2026.json"


@dataclass(frozen=True)
class StabilityConfigPair:
    """The retained C2 comparator and selected I2 candidate at seed 2026."""

    c2: ExperimentConfig
    i2: I2ExperimentConfig


def _without_run_identity(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for field in ("experiment_id", "stage", "seeds"):
        normalized.pop(field)
    return normalized


def load_stability_i2_config(path: str | Path) -> I2ExperimentConfig:
    """Parse the seed-2026 I2 clone without weakening the primary I2 loader."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"G5 I2 config must be a JSON object: {source}")
    base_raw = dict(raw)
    if "auxiliary" not in base_raw:
        raise ValueError("G5 I2 config is missing the auxiliary block")
    auxiliary_raw = base_raw.pop("auxiliary")
    if not isinstance(auxiliary_raw, dict):
        raise ValueError("G5 I2 auxiliary block must be a JSON object")
    unknown = sorted(set(auxiliary_raw) - set(AuxiliaryRunConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown G5 I2 auxiliary fields: {unknown}")
    required = {"target", "loss_weight"}
    missing = sorted(required - set(auxiliary_raw))
    if missing:
        raise ValueError(f"G5 I2 auxiliary block is missing fields: {missing}")
    config = I2ExperimentConfig(
        base=ExperimentConfig.from_dict(base_raw),
        auxiliary=AuxiliaryRunConfig(**auxiliary_raw),
    )
    config.auxiliary.validate()
    return config


def validate_stability_pair(pair: StabilityConfigPair) -> StabilityConfigPair:
    """Reject every change except identity, stage, and seed 2753 to 2026."""
    pair.c2.validate()
    pair.i2.base.validate()
    pair.i2.auxiliary.validate()
    expected_identity = {
        "c2_experiment_id": C2_STABILITY_EXPERIMENT_ID,
        "i2_experiment_id": I2_STABILITY_EXPERIMENT_ID,
        "c2_stage": G5_STAGE,
        "i2_stage": G5_STAGE,
        "c2_seeds": (G5_SEED,),
        "i2_seeds": (G5_SEED,),
    }
    observed_identity = {
        "c2_experiment_id": pair.c2.experiment_id,
        "i2_experiment_id": pair.i2.experiment_id,
        "c2_stage": pair.c2.stage,
        "i2_stage": pair.i2.stage,
        "c2_seeds": pair.c2.seeds,
        "i2_seeds": pair.i2.seeds,
    }
    mismatches = [
        name for name, expected in expected_identity.items() if observed_identity[name] != expected
    ]
    if mismatches:
        raise ValueError(f"G5 stability identity mismatch: {mismatches}")

    primary_c2 = load_experiment_config(C2_PRIMARY_CONFIG_PATH)
    primary_i2 = load_i2_config(I2_PRIMARY_CONFIG_PATH)
    if _without_run_identity(pair.c2.to_dict()) != _without_run_identity(primary_c2.to_dict()):
        raise ValueError("G5 C2 changes a frozen training choice")
    if _without_run_identity(pair.i2.to_dict()) != _without_run_identity(primary_i2.to_dict()):
        raise ValueError("G5 I2 changes a frozen training choice")
    if pair.c2.folds != tuple(range(5)) or pair.i2.folds != tuple(range(5)):
        raise ValueError("G5 stability requires all five canonical folds")
    return pair


def load_stability_pair(
    c2_path: str | Path = C2_STABILITY_CONFIG_PATH,
    i2_path: str | Path = I2_STABILITY_CONFIG_PATH,
) -> StabilityConfigPair:
    """Load and validate the complete G5 pair before any fold can run."""
    return validate_stability_pair(
        StabilityConfigPair(
            c2=load_experiment_config(c2_path),
            i2=load_stability_i2_config(i2_path),
        )
    )


def validate_stability_implementation(
    pair: StabilityConfigPair,
    *,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
) -> dict[str, dict[str, Any]]:
    """Require current training code to match five valid primary-seed runs."""
    resolved = validate_stability_pair(pair)
    registry = RunRegistry(registry_path).read()
    if registry.empty:
        raise ValueError("stability preflight found no primary run registry")
    primary_c2 = load_experiment_config(C2_PRIMARY_CONFIG_PATH)
    primary_i2 = load_i2_config(I2_PRIMARY_CONFIG_PATH)
    specifications = (
        (
            "C2",
            primary_c2,
            resolved.c2,
            experiment_runner._implementation_paths("deep"),
        ),
        (
            "I2",
            primary_i2,
            resolved.i2,
            I2_IMPLEMENTATION_PATHS,
        ),
    )
    audit: dict[str, dict[str, Any]] = {}
    for candidate, primary, stability, implementation_paths in specifications:
        current_key = build_run_cache_key(
            stability.to_dict(),
            fold=0,
            seed=G5_SEED,
            implementation_paths=implementation_paths,
            split_path=splits_path,
            label_map_path=label_map_path,
            root=source_root,
        )
        primary_rows = registry.loc[
            registry["experiment_id"].eq(primary.experiment_id)
            & registry["status"].eq("completed")
            & pd.to_numeric(registry["seed"], errors="coerce").eq(2753)
            & registry["config_sha256"].eq(canonical_sha256(primary.to_dict()))
            & registry["split_sha256"].eq(current_key.split_sha256)
            & registry["label_map_sha256"].eq(current_key.label_map_sha256)
            & registry["implementation_sha256"].eq(current_key.implementation_sha256)
            & registry["git_dirty"].astype(str).str.lower().eq("false")
        ].copy()
        if (
            len(primary_rows) != 5
            or primary_rows["run_id"].nunique() != 5
            or set(
                pd.to_numeric(
                    primary_rows["fold"],
                    errors="raise",
                ).astype(int)
            )
            != set(range(5))
        ):
            raise ValueError(
                f"{candidate} stability implementation hash mismatch: "
                "current code does not match five clean primary-seed folds"
            )
        audit[candidate] = {
            "implementation_sha256": current_key.implementation_sha256,
            "split_sha256": current_key.split_sha256,
            "label_map_sha256": current_key.label_map_sha256,
            "primary_run_ids": sorted(primary_rows["run_id"].astype(str)),
        }
    return audit


def _run_stability_i2_experiment(
    config: I2ExperimentConfig,
    *,
    mode: ExecutionMode = "run_or_load",
    data_root: str | Path = ROOT,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
    checkpoint_directory: str | Path = TASK2_CHECKPOINT_DIR,
    run_directory: str | Path = TASK2_RUN_DIR,
) -> list[ExperimentFoldOutput]:
    """Reuse the unchanged I2 fold runner after the G5 clone audit passes."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    validate_stability_pair(
        StabilityConfigPair(
            c2=load_experiment_config(C2_STABILITY_CONFIG_PATH),
            i2=config,
        )
    )
    registry = RunRegistry(registry_path)
    return [
        multitask_runner._run_one(
            config,
            fold=fold,
            seed=seed,
            mode=mode,
            registry=registry,
            data_root=Path(data_root).resolve(),
            source_root=Path(source_root).resolve(),
            splits_path=Path(splits_path),
            label_map_path=Path(label_map_path),
            checkpoint_directory=Path(checkpoint_directory),
            run_directory=Path(run_directory),
        )
        for seed in config.seeds
        for fold in config.folds
    ]


def run_stability_matrix(
    pair: StabilityConfigPair | None = None,
    **kwargs: Any,
) -> list[ExperimentFoldOutput]:
    """Run C2 then I2 only after validating the complete eligible pair."""
    resolved = validate_stability_pair(pair or load_stability_pair())
    validate_stability_implementation(
        resolved,
        source_root=kwargs.get("source_root", ROOT),
        splits_path=kwargs.get("splits_path", SPLITS_CSV),
        label_map_path=kwargs.get("label_map_path", LABEL_MAPS_JSON),
        registry_path=kwargs.get("registry_path", RUNS_CSV),
    )
    outputs = run_or_load_experiment(resolved.c2, **kwargs)
    outputs.extend(_run_stability_i2_experiment(resolved.i2, **kwargs))
    return outputs


__all__ = [
    "C2_PRIMARY_EXPERIMENT_ID",
    "C2_STABILITY_EXPERIMENT_ID",
    "G5_SEED",
    "G5_STAGE",
    "I2_PRIMARY_EXPERIMENT_ID",
    "I2_STABILITY_EXPERIMENT_ID",
    "StabilityConfigPair",
    "load_stability_i2_config",
    "load_stability_pair",
    "run_stability_matrix",
    "validate_stability_implementation",
    "validate_stability_pair",
]
