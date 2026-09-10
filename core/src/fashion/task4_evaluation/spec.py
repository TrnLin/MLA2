"""The committed Task 4 holdout evaluation contract, frozen before the unlock."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from fashion.config import ROOT
from fashion.task4_evaluation.conditions import QUERY_CONDITIONS

_CONFIG_RELATIVE_PATH = "configs/task4/holdout_final_evaluation.json"
_SCHEMA_VERSION = "1.0.0"
_NDCG_AT_10_METRIC = "protocol_a_ndcg_at_10_query_mean"
_NDCG_AT_10_REQUIRED_K = 10

# Provenance pointers the config records for the report. They are expected keys, so they
# must not trip the unknown-field guard, but no caller reads them through the spec object
# and they deliberately become no attribute.
_RECORDED_ONLY_KEYS = frozenset(
    {
        "decision_record",
        "preprocessing_contract",
        "preprocessing_normalization",
        "schema_version",
    }
)


@dataclass(frozen=True, slots=True)
class HoldoutEvaluationSpec:
    """Every choice the one-shot holdout run is allowed to make, fixed in advance."""

    evaluation_id: str
    model_run_id: str
    model_checkpoint_sha256: str
    split_fingerprint: str
    expected_development_rows: int
    expected_holdout_rows: int
    expected_quarantine_rows: int
    expected_holdout_family_groups: int
    primary_metric: str
    k_values: tuple[int, ...]
    directions: tuple[str, ...]
    methods: tuple[str, ...]
    benchmark_only_methods: tuple[str, ...]
    conditions: tuple[str, ...]
    stressed_direction: str
    bootstrap_replicates: int
    bootstrap_seed: int
    random_floor_seed: int
    development_winner_score: float
    development_source_robustness_ratio: float
    gallery_directory: str
    model_package_directory: str


def _partition_expected_keys(
    exposed: frozenset[str], recorded: frozenset[str]
) -> frozenset[str]:
    if overlap := sorted(exposed.intersection(recorded)):
        raise RuntimeError(
            f"recorded-only config keys must not also be dataclass fields: {overlap}"
        )
    return exposed | recorded


_EXPOSED_KEYS = frozenset(field.name for field in fields(HoldoutEvaluationSpec))
_EXPECTED_KEYS = _partition_expected_keys(_EXPOSED_KEYS, _RECORDED_ONLY_KEYS)


def _str_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def load_holdout_evaluation_spec(
    *, project_root: str | Path = ROOT
) -> HoldoutEvaluationSpec:
    """Read the frozen holdout evaluation config and reject any drift from it."""
    path = Path(project_root) / _CONFIG_RELATIVE_PATH
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    present = set(payload)
    missing = _EXPECTED_KEYS.difference(present)
    unknown = present.difference(_EXPECTED_KEYS)
    if missing or unknown:
        raise ValueError(
            f"holdout evaluation config has missing fields: {sorted(missing)}; "
            f"unknown fields: {sorted(unknown)}"
        )
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(
            f"holdout evaluation config schema_version must be {_SCHEMA_VERSION}"
        )
    spec = HoldoutEvaluationSpec(
        evaluation_id=str(payload["evaluation_id"]),
        model_run_id=str(payload["model_run_id"]),
        model_checkpoint_sha256=str(payload["model_checkpoint_sha256"]),
        split_fingerprint=str(payload["split_fingerprint"]),
        expected_development_rows=int(payload["expected_development_rows"]),
        expected_holdout_rows=int(payload["expected_holdout_rows"]),
        expected_quarantine_rows=int(payload["expected_quarantine_rows"]),
        expected_holdout_family_groups=int(payload["expected_holdout_family_groups"]),
        primary_metric=str(payload["primary_metric"]),
        k_values=tuple(int(value) for value in payload["k_values"]),
        directions=_str_tuple(payload["directions"]),
        methods=_str_tuple(payload["methods"]),
        benchmark_only_methods=_str_tuple(payload["benchmark_only_methods"]),
        conditions=_str_tuple(payload["conditions"]),
        stressed_direction=str(payload["stressed_direction"]),
        bootstrap_replicates=int(payload["bootstrap_replicates"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
        random_floor_seed=int(payload["random_floor_seed"]),
        development_winner_score=float(payload["development_winner_score"]),
        development_source_robustness_ratio=float(
            payload["development_source_robustness_ratio"]
        ),
        gallery_directory=str(payload["gallery_directory"]),
        model_package_directory=str(payload["model_package_directory"]),
    )
    if spec.stressed_direction not in spec.directions:
        raise ValueError(
            f"stressed_direction {spec.stressed_direction!r} is not one of the declared "
            f"directions {list(spec.directions)}"
        )
    if extra := sorted(set(spec.benchmark_only_methods).difference(spec.methods)):
        raise ValueError(f"benchmark_only_methods are not declared methods: {extra}")
    if spec.conditions != tuple(QUERY_CONDITIONS):
        raise ValueError(
            "holdout evaluation config conditions must equal QUERY_CONDITIONS: "
            f"{list(QUERY_CONDITIONS)}"
        )
    if (
        spec.primary_metric == _NDCG_AT_10_METRIC
        and _NDCG_AT_10_REQUIRED_K not in spec.k_values
    ):
        raise ValueError(
            f"primary_metric {spec.primary_metric!r} requires k={_NDCG_AT_10_REQUIRED_K} "
            f"in k_values {list(spec.k_values)}"
        )
    return spec


__all__ = ["HoldoutEvaluationSpec", "load_holdout_evaluation_spec"]
