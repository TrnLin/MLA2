from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fashion.task4_evaluation.conditions import QUERY_CONDITIONS
from fashion.task4_evaluation.spec import (
    _EXPECTED_KEYS,
    _EXPOSED_KEYS,
    _RECORDED_ONLY_KEYS,
    HoldoutEvaluationSpec,
    _partition_expected_keys,
    load_holdout_evaluation_spec,
)
from fashion.task4_evaluation.views import (
    DEVELOPMENT_ROWS_EXPECTED,
    HOLDOUT_ROWS_EXPECTED,
    QUARANTINE_ROWS_EXPECTED,
)

_CONFIG_RELATIVE_PATH = "configs/task4/holdout_final_evaluation.json"


def _source() -> dict:
    with open(_CONFIG_RELATIVE_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _write_mutated_config(root: Path, payload: dict) -> None:
    target = root / _CONFIG_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_spec_is_frozen_to_the_selected_r5_bundle() -> None:
    spec = load_holdout_evaluation_spec()
    assert spec.evaluation_id == "m8-task4-holdout-final-evaluation"
    assert spec.model_run_id == "task4-candidate-r5-task9-preexec"
    assert spec.model_checkpoint_sha256 == (
        "521e96f3df9e28853309bd607030523f59ef8425326da56dfcf2b754e97631b3"
    )
    assert spec.primary_metric == "protocol_a_ndcg_at_10_query_mean"
    assert spec.k_values == (5, 10, 20)
    assert spec.expected_development_rows == 32_773
    assert spec.expected_holdout_rows == 5_778
    assert spec.expected_quarantine_rows == 61
    assert spec.expected_holdout_family_groups == 4_110
    assert spec.bootstrap_replicates == 10_000
    assert spec.bootstrap_seed == 2753


def test_spec_declares_two_directions_and_five_methods() -> None:
    spec = load_holdout_evaluation_spec()
    assert spec.directions == ("teacher", "v1")
    assert spec.methods == (
        "r5_scratch_autoencoder",
        "random_floor",
        "spatial_hsv_edge_probe",
        "hog_hsv_edge_fusion",
        "b1_pretrained_resnet18",
    )
    assert spec.benchmark_only_methods == ("b1_pretrained_resnet18",)
    assert spec.methods[0] not in spec.benchmark_only_methods


def test_spec_declares_seven_conditions_on_the_teacher_direction() -> None:
    spec = load_holdout_evaluation_spec()
    assert len(spec.conditions) == 7
    assert spec.conditions[0] == "clean"
    assert spec.stressed_direction == "teacher"
    assert spec.stressed_direction in spec.directions


def test_spec_conditions_match_the_frozen_condition_tuple() -> None:
    assert tuple(_source()["conditions"]) == QUERY_CONDITIONS


def test_spec_row_counts_match_the_view_constants() -> None:
    spec = load_holdout_evaluation_spec()
    assert spec.expected_development_rows == DEVELOPMENT_ROWS_EXPECTED
    assert spec.expected_holdout_rows == HOLDOUT_ROWS_EXPECTED
    assert spec.expected_quarantine_rows == QUARANTINE_ROWS_EXPECTED


def test_recorded_only_keys_are_disjoint_from_the_dataclass_fields() -> None:
    assert _RECORDED_ONLY_KEYS.isdisjoint(_EXPOSED_KEYS)
    assert _EXPECTED_KEYS == _EXPOSED_KEYS | _RECORDED_ONLY_KEYS


def test_a_recorded_only_key_that_is_also_a_field_is_rejected() -> None:
    overlapping = frozenset({"schema_version"})
    with pytest.raises(RuntimeError, match="must not also be dataclass fields"):
        _partition_expected_keys(overlapping, overlapping)


def test_spec_is_a_frozen_dataclass() -> None:
    spec = load_holdout_evaluation_spec()
    assert isinstance(spec, HoldoutEvaluationSpec)
    with pytest.raises(FrozenInstanceError):
        spec.bootstrap_seed = 1  # type: ignore[misc]


def test_spec_rejects_an_unknown_key(tmp_path: Path) -> None:
    payload = _source()
    payload["surprise"] = True
    _write_mutated_config(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown fields"):
        load_holdout_evaluation_spec(project_root=tmp_path)


def test_spec_rejects_a_missing_key(tmp_path: Path) -> None:
    payload = _source()
    del payload["bootstrap_seed"]
    _write_mutated_config(tmp_path, payload)
    with pytest.raises(ValueError, match="missing fields"):
        load_holdout_evaluation_spec(project_root=tmp_path)


def test_spec_rejects_a_primary_metric_whose_k_is_absent(tmp_path: Path) -> None:
    payload = _source()
    payload["k_values"] = [5, 20]
    _write_mutated_config(tmp_path, payload)
    with pytest.raises(ValueError, match="requires k=10"):
        load_holdout_evaluation_spec(project_root=tmp_path)


def test_spec_rejects_a_wrong_schema_version(tmp_path: Path) -> None:
    payload = _source()
    payload["schema_version"] = "2.0.0"
    _write_mutated_config(tmp_path, payload)
    with pytest.raises(ValueError, match="schema_version"):
        load_holdout_evaluation_spec(project_root=tmp_path)
