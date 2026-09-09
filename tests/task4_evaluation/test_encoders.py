from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task4.preprocessing import (
    PreprocessedImage,
    PreprocessingContract,
    load_preprocessed_image,
)
from fashion.task4_evaluation.encoders import (
    B1_EVIDENCE_MANIFEST_RELATIVE_PATH,
    B1_METHOD,
    HOG_FUSION_METHOD,
    PROBE_METHOD,
    R5_METHOD,
    RANDOM_FLOOR_METHOD,
    HogFusionEncoder,
    ProbeEncoder,
    _unit_norm,
    build_method_encoders,
    build_random_floor_rankings,
    load_b1_encoder,
    load_r5_encoder,
)
from fashion.task4_evaluation.spec import load_holdout_evaluation_spec

CONTRACT = PreprocessingContract(width=240, height=320)
SPEC = load_holdout_evaluation_spec()
ENCODER_METHODS = tuple(
    method for method in SPEC.methods if method != RANDOM_FLOOR_METHOD
)
STATISTICS_FREE_METHODS = (PROBE_METHOD, HOG_FUSION_METHOD)
STATISTICS_BOUND_METHODS = (R5_METHOD, B1_METHOD)


def _r5_manifest() -> dict:
    return json.loads(
        (ROOT / SPEC.model_package_directory / "manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _b1_evidence() -> dict:
    return json.loads(
        (ROOT / B1_EVIDENCE_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_b1_project(root: Path, evidence: dict, *, direction: str = "teacher") -> None:
    """Write a mutated evidence manifest plus the real statistics it points at."""
    _write_json(root / B1_EVIDENCE_MANIFEST_RELATIVE_PATH, evidence)
    relative = _b1_evidence()["source_artifacts"][direction]["statistics"]["path"]
    _write_json(root / relative, json.loads((ROOT / relative).read_text("utf-8")))


def _real_artifacts_present() -> bool:
    evidence = ROOT / B1_EVIDENCE_MANIFEST_RELATIVE_PATH
    package = ROOT / SPEC.model_package_directory
    if not evidence.is_file() or not (package / "weights.pt").is_file():
        return False
    manifest = json.loads(evidence.read_text(encoding="utf-8"))
    return (ROOT / manifest["checkpoint"]["path"]).is_file()


REAL_ENCODER_ARTIFACTS = pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real Task 4 R5 package or B1 benchmark checkpoint is absent",
)


@pytest.fixture(scope="module")
def development_image() -> PreprocessedImage:
    splits = pd.read_csv(
        ROOT / "data/processed/splits.csv",
        usecols=["id", "path", "partition"],
    )
    development = splits.loc[splits["partition"].eq("development")].sort_values("id")
    if development.empty:
        raise AssertionError("no development row is available for encoder tests")
    return load_preprocessed_image(ROOT / str(development.iloc[0]["path"]), CONTRACT)


@pytest.fixture(scope="module")
def teacher_encoders() -> dict[str, object]:
    return build_method_encoders(SPEC, project_root=ROOT, direction="teacher")


@pytest.fixture(scope="module")
def v1_encoders() -> dict[str, object]:
    return build_method_encoders(SPEC, project_root=ROOT, direction="v1")


@REAL_ENCODER_ARTIFACTS
def test_build_method_encoders_covers_every_method_but_the_random_floor(
    teacher_encoders: dict[str, object],
) -> None:
    assert set(teacher_encoders) == set(SPEC.methods) - {RANDOM_FLOOR_METHOD}
    assert RANDOM_FLOOR_METHOD not in teacher_encoders


def test_build_method_encoders_rejects_an_undeclared_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        build_method_encoders(SPEC, project_root=ROOT, direction="holdout")


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("method", ENCODER_METHODS)
def test_encoder_returns_a_finite_unit_norm_float32_vector(
    teacher_encoders: dict[str, object],
    development_image: PreprocessedImage,
    method: str,
) -> None:
    encoder = teacher_encoders[method]
    vector = encoder.encode(development_image)
    assert encoder.name == method
    assert vector.dtype == np.float32
    assert vector.ndim == 1
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    assert encoder.dimension == len(vector)


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("method", ENCODER_METHODS)
def test_encoder_repeats_bit_identical_vectors(
    teacher_encoders: dict[str, object],
    development_image: PreprocessedImage,
    method: str,
) -> None:
    encoder = teacher_encoders[method]
    assert np.array_equal(
        encoder.encode(development_image),
        encoder.encode(development_image),
    )


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("method", STATISTICS_BOUND_METHODS)
def test_learned_encoders_report_the_frozen_128_value_embedding(
    teacher_encoders: dict[str, object],
    development_image: PreprocessedImage,
    method: str,
) -> None:
    encoder = teacher_encoders[method]
    assert encoder.dimension == 128
    assert len(encoder.encode(development_image)) == 128


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("method", STATISTICS_FREE_METHODS)
def test_statistics_free_encoders_are_identical_across_directions(
    teacher_encoders: dict[str, object],
    v1_encoders: dict[str, object],
    development_image: PreprocessedImage,
    method: str,
) -> None:
    assert np.array_equal(
        teacher_encoders[method].encode(development_image),
        v1_encoders[method].encode(development_image),
    )


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("method", STATISTICS_BOUND_METHODS)
def test_statistics_bound_encoders_differ_across_directions(
    teacher_encoders: dict[str, object],
    v1_encoders: dict[str, object],
    development_image: PreprocessedImage,
    method: str,
) -> None:
    teacher_vector = teacher_encoders[method].encode(development_image)
    v1_vector = v1_encoders[method].encode(development_image)
    assert not np.array_equal(teacher_vector, v1_vector)
    assert np.linalg.norm(teacher_vector) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(v1_vector) == pytest.approx(1.0, abs=1e-5)


@REAL_ENCODER_ARTIFACTS
def test_b1_is_the_only_encoder_reporting_pretrained_weights(
    teacher_encoders: dict[str, object],
) -> None:
    pretrained = {
        method: encoder.pretrained for method, encoder in teacher_encoders.items()
    }
    assert pretrained[B1_METHOD] is True
    assert pretrained[R5_METHOD] is False
    assert [method for method, flag in pretrained.items() if flag] == [B1_METHOD]
    assert B1_METHOD in SPEC.benchmark_only_methods


def test_statistics_free_encoders_need_no_artifacts(
    development_image: PreprocessedImage,
) -> None:
    probe = ProbeEncoder()
    fusion = HogFusionEncoder()
    assert probe.dimension == len(probe.encode(development_image))
    assert fusion.dimension == len(fusion.encode(development_image))
    assert probe.pretrained is False
    assert fusion.pretrained is False


@REAL_ENCODER_ARTIFACTS
def test_r5_loader_rejects_a_package_claiming_pretrained_weights(
    tmp_path: Path,
) -> None:
    manifest = _r5_manifest()
    manifest["pretrained"] = True
    _write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest identity is invalid"):
        load_r5_encoder(tmp_path, direction="teacher")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_unit_norm_rejects_a_non_finite_embedding(value: float) -> None:
    vector = np.ones(4, dtype=np.float32)
    vector[2] = value
    with pytest.raises(ValueError, match="embedding must be finite"):
        _unit_norm(vector)


def test_unit_norm_rejects_an_all_zero_embedding() -> None:
    with pytest.raises(ValueError, match="embedding must have positive norm"):
        _unit_norm(np.zeros(4, dtype=np.float32))


@REAL_ENCODER_ARTIFACTS
def test_r5_loader_names_a_manifest_without_an_embedding_dimension(
    tmp_path: Path,
) -> None:
    manifest = _r5_manifest()
    del manifest["embedding_dim"]
    _write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="missing embedding_dim"):
        load_r5_encoder(tmp_path, direction="teacher")


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("value", [None, "128", [128]])
def test_r5_loader_rejects_a_non_integer_embedding_dimension(
    tmp_path: Path,
    value: object,
) -> None:
    manifest = _r5_manifest()
    manifest["embedding_dim"] = value
    _write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="embedding_dim must be an integer"):
        load_r5_encoder(tmp_path, direction="teacher")


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("key", ["checkpoint", "config_hash", "run_id", "run_kind"])
def test_b1_loader_names_a_missing_evidence_manifest_key(
    tmp_path: Path,
    key: str,
) -> None:
    evidence = _b1_evidence()
    del evidence[key]
    _write_b1_project(tmp_path, evidence)
    with pytest.raises(ValueError, match=f"missing {key}"):
        load_b1_encoder(
            tmp_path,
            direction="teacher",
            split_fingerprint=SPEC.split_fingerprint,
        )


@REAL_ENCODER_ARTIFACTS
@pytest.mark.parametrize("key", ["path", "sha256"])
def test_b1_loader_names_a_missing_checkpoint_record_key(
    tmp_path: Path,
    key: str,
) -> None:
    evidence = _b1_evidence()
    del evidence["checkpoint"][key]
    _write_b1_project(tmp_path, evidence)
    with pytest.raises(ValueError, match=f"missing {key}"):
        load_b1_encoder(
            tmp_path,
            direction="teacher",
            split_fingerprint=SPEC.split_fingerprint,
        )


@REAL_ENCODER_ARTIFACTS
def test_b1_loader_rejects_a_checkpoint_record_that_is_not_an_object(
    tmp_path: Path,
) -> None:
    evidence = _b1_evidence()
    evidence["checkpoint"] = "epoch-020.pt"
    _write_b1_project(tmp_path, evidence)
    with pytest.raises(ValueError, match="checkpoint must be one JSON object"):
        load_b1_encoder(
            tmp_path,
            direction="teacher",
            split_fingerprint=SPEC.split_fingerprint,
        )


@REAL_ENCODER_ARTIFACTS
def test_b1_loader_names_a_statistics_record_without_a_path(tmp_path: Path) -> None:
    evidence = _b1_evidence()
    del evidence["source_artifacts"]["teacher"]["statistics"]["path"]
    _write_b1_project(tmp_path, evidence)
    with pytest.raises(ValueError, match="missing path"):
        load_b1_encoder(
            tmp_path,
            direction="teacher",
            split_fingerprint=SPEC.split_fingerprint,
        )


def _floor_ids() -> tuple[list[int], list[int]]:
    queries = [11, 22, 33]
    gallery = [11, 22, 33, 44, 55, 66]
    return queries, gallery


def test_random_floor_rankings_are_deterministic_for_one_seed() -> None:
    queries, gallery = _floor_ids()
    first = build_random_floor_rankings(queries, gallery, seed=2753, max_k=3)
    second = build_random_floor_rankings(queries, gallery, seed=2753, max_k=3)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["query_id", "candidate_id", "distance", "rank"]


def test_random_floor_rankings_change_with_the_seed() -> None:
    queries, gallery = _floor_ids()
    first = build_random_floor_rankings(queries, gallery, seed=2753, max_k=3)
    other = build_random_floor_rankings(queries, gallery, seed=1, max_k=3)
    assert not first["candidate_id"].equals(other["candidate_id"])


def test_random_floor_rankings_give_every_query_consecutive_ranks() -> None:
    queries, gallery = _floor_ids()
    rankings = build_random_floor_rankings(queries, gallery, seed=2753, max_k=4)
    assert len(rankings) == len(queries) * 4
    for query_id in queries:
        rows = rankings.loc[rankings["query_id"].eq(query_id)]
        assert list(rows["rank"]) == [1, 2, 3, 4]
        assert rows["candidate_id"].nunique() == 4
        assert list(rows["distance"]) == [1.0, 2.0, 3.0, 4.0]


def test_random_floor_rankings_never_rank_a_query_against_itself() -> None:
    queries, gallery = _floor_ids()
    rankings = build_random_floor_rankings(queries, gallery, seed=2753, max_k=5)
    assert not rankings["query_id"].eq(rankings["candidate_id"]).any()


def test_random_floor_rankings_reject_an_impossible_depth() -> None:
    queries, gallery = _floor_ids()
    with pytest.raises(ValueError, match="max_k"):
        build_random_floor_rankings(queries, gallery, seed=2753, max_k=6)
