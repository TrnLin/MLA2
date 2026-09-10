from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import fashion.task4_evaluation.blind as blind_module
from fashion.data.hashing import compute_sha256
from fashion.data.splits import cv_assignment_digest
from fashion.task4.gallery_artifact import (
    ALL_DEVELOPMENT_FOLD,
    SOURCE_IDENTITY_FIELDS,
    export_teacher_gallery_artifact,
)
from fashion.task4.preprocessing import PreprocessingContract, preprocess_image
from fashion.task4_evaluation.blind import build_blind_holdout_evidence
from fashion.task4_evaluation.conditions import QUERY_CONDITIONS, apply_query_condition
from fashion.task4_evaluation.encoders import RANDOM_FLOOR_METHOD
from fashion.train.artifacts import ArtifactVerificationError, canonical_sha256

DEVELOPMENT_ROWS = 25
HOLDOUT_ROWS = 12
QUARANTINE_ROWS = 1
METHODS = (
    "r5_scratch_autoencoder",
    "random_floor",
    "spatial_hsv_edge_probe",
    "hog_hsv_edge_fusion",
    "b1_pretrained_resnet18",
)
CHECKPOINT_SHA256 = "c" * 64
RUN_ID = "task4-r5-test"
CONTRACT = PreprocessingContract(width=240, height=320)
BANNED_COLUMNS = {"articleType", "y_true", "relevance", "grade", "ndcg_at_10"}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_image(path: Path, product_id: int, *, external: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    offset = 37 if external else 0
    pixels[:, :] = (
        (product_id * 17 + offset) % 220,
        (product_id * 31 + offset) % 220,
        (product_id * 47 + offset) % 220,
    )
    pixels[10:70, 15:45] = (
        (product_id * 13 + 30 + offset) % 255,
        (product_id * 19 + 50 + offset) % 255,
        (product_id * 23 + 70 + offset) % 255,
    )
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG", quality=95)


def _split_row(
    root: Path,
    product_id: int,
    partition: str,
    *,
    fold: int | str,
    family: str,
) -> dict[str, object]:
    relative = Path(f"data/raw/teacher/train/images/{product_id}.jpg")
    image_path = root / relative
    _write_image(image_path, product_id, external=False)
    protected = partition != "development"
    return {
        "id": product_id,
        "gender": "" if protected else "Unisex",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "" if protected else "Tshirts",
        "baseColour": "Blue" if product_id % 2 else "Red",
        "season": "" if protected else "Summer",
        "usage": "" if protected else "Casual",
        "year": 2011,
        "productDisplayName": f"Product {product_id}",
        "product_name_repaired": f"Product {product_id}",
        "has_articleType_label": not protected,
        "has_season_label": not protected,
        "has_gender_label": not protected,
        "has_usage_label": not protected,
        "path": relative.as_posix(),
        "width": 60,
        "height": 80,
        "aspect_ratio": 0.75,
        "mode": "RGB",
        "format": "JPEG",
        "file_size_bytes": image_path.stat().st_size,
        "sha256": compute_sha256(image_path),
        "product_name_key": f"name-{product_id}",
        "product_family_group": family,
        "family_group_basis": "product_name_key",
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "pre_quarantine_reason": "",
        "duplicate_group": f"duplicate-{product_id}",
        "is_cross_role_duplicate": False,
        "quarantine_reason": (
            "approved test quarantine"
            if protected and partition == "quarantine"
            else ""
        ),
        "partition": partition,
        "cv_fold": fold,
    }


def _write_source_cache(root: Path, splits: pd.DataFrame) -> str:
    directory = root / "source-cache"
    directory.mkdir()
    development = splits.loc[splits["partition"].eq("development")].sort_values("id")
    ids = development["id"].to_numpy(dtype=np.int64)
    features = np.zeros((len(ids), 128), dtype=np.float32)
    features[np.arange(len(ids)), np.arange(len(ids)) % 128] = 1.0
    np.save(directory / "ids.npy", ids, allow_pickle=False)
    np.save(directory / "features.npy", features, allow_pickle=False)
    identity = {
        "schema_version": "1.0.0",
        "scope": "development",
        "run_id": RUN_ID,
        "run_kind": "candidate",
        "method": "R5",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_hash": "a" * 64,
        "split_fingerprint": cv_assignment_digest(splits),
        "source": "teacher",
        "image_cache_manifest_sha256": "d" * 64,
        "source_fingerprint": "e" * 64,
        "source_statistics_sha256": "f" * 64,
        "fold": 1,
        "contract": CONTRACT.to_dict(),
        "rows": len(ids),
        "dimension": 128,
    }
    source_identity = hashlib.sha256(_canonical_json(identity)).hexdigest()
    manifest = {
        **identity,
        "feature_cache_identity_sha256": source_identity,
        "feature_method": "R5",
        "transform_seconds": 0.1,
        "source_bytes": len(ids) * 240 * 320 * 3,
        "ids_sha256": compute_sha256(directory / "ids.npy"),
        "features_sha256": compute_sha256(directory / "features.npy"),
    }
    assert set(manifest) == set(SOURCE_IDENTITY_FIELDS) | {
        "feature_cache_identity_sha256",
        "feature_method",
        "transform_seconds",
        "source_bytes",
        "ids_sha256",
        "features_sha256",
    }
    (directory / "manifest.json").write_bytes(_canonical_json(manifest))
    return source_identity


def _write_spec(root: Path, *, split_fingerprint: str) -> None:
    payload = {
        "benchmark_only_methods": ["b1_pretrained_resnet18"],
        "bootstrap_replicates": 100,
        "bootstrap_seed": 2753,
        "conditions": list(QUERY_CONDITIONS),
        "decision_record": "docs/decisions/test.md",
        "development_source_robustness_ratio": 1.0,
        "development_winner_score": 0.5,
        "directions": ["teacher", "v1"],
        "evaluation_id": "task4-blind-test",
        "expected_development_rows": DEVELOPMENT_ROWS,
        "expected_holdout_family_groups": HOLDOUT_ROWS // 2,
        "expected_holdout_rows": HOLDOUT_ROWS,
        "expected_quarantine_rows": QUARANTINE_ROWS,
        "gallery_directory": "models/task4_holdout_gallery",
        "k_values": [5, 10, 20],
        "methods": list(METHODS),
        "model_checkpoint_sha256": CHECKPOINT_SHA256,
        "model_package_directory": "models/task4_r5",
        "model_run_id": RUN_ID,
        "preprocessing_contract": "results/evidence/task4/preprocessing_contract.json",
        "preprocessing_normalization": (
            "results/evidence/task4/preprocessing_normalization_fold1.json"
        ),
        "primary_metric": "protocol_a_ndcg_at_10_query_mean",
        "random_floor_seed": 2753,
        "schema_version": "1.0.0",
        "split_fingerprint": split_fingerprint,
        "stressed_direction": "teacher",
    }
    path = root / "configs/task4/holdout_final_evaluation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_project(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(DEVELOPMENT_ROWS):
        product_id = index + 1
        rows.append(
            _split_row(
                root,
                product_id,
                "development",
                fold=index % 5,
                family=f"development-{product_id}",
            )
        )
    for index in range(HOLDOUT_ROWS):
        product_id = 101 + index
        rows.append(
            _split_row(
                root,
                product_id,
                "holdout",
                fold="",
                family=f"holdout-{index // 2}",
            )
        )
        external_path = root / f"data/external/images/{product_id}.jpg"
        _write_image(external_path, product_id, external=True)
    rows.append(
        _split_row(
            root,
            999,
            "quarantine",
            fold="",
            family="quarantine-only",
        )
    )
    splits = pd.DataFrame(rows)
    split_path = root / "data/processed/splits.csv"
    split_path.parent.mkdir(parents=True)
    splits.to_csv(split_path, index=False, lineterminator="\n")

    variants = []
    for row in splits.loc[splits["partition"].eq("holdout")].itertuples(index=False):
        external_path = Path(f"data/external/images/{row.id}.jpg")
        variants.append(
            {
                "id": row.id,
                "teacher_path": row.path,
                "teacher_width": row.width,
                "teacher_height": row.height,
                "teacher_file_size_bytes": row.file_size_bytes,
                "partition": row.partition,
                "cv_fold": "",
                "product_family_group": row.product_family_group,
                "duplicate_group": row.duplicate_group,
                "external_path": external_path.as_posix(),
                "external_width": 60,
                "external_height": 80,
                "external_aspect_ratio": 0.75,
                "external_mode": "RGB",
            }
        )
    variant_path = root / "data/processed/task4/external_variant_index.csv.gz"
    variant_path.parent.mkdir(parents=True)
    pd.DataFrame(variants).to_csv(variant_path, index=False, compression="gzip")

    split_fingerprint = cv_assignment_digest(splits)
    source_identity = _write_source_cache(root, splits)
    export_teacher_gallery_artifact(
        root / "source-cache",
        root / "models/task4_holdout_gallery",
        splits_path=split_path,
        expected_source_identity_sha256=source_identity,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_run_id=RUN_ID,
        fold=ALL_DEVELOPMENT_FOLD,
    )
    model_manifest = {
        "source_checkpoint": {
            "run_id": RUN_ID,
            "score": 0.5,
            "sha256": CHECKPOINT_SHA256,
        }
    }
    model_path = root / "models/task4_r5/manifest.json"
    model_path.parent.mkdir(parents=True)
    model_path.write_text(json.dumps(model_manifest), encoding="utf-8")
    (root / "results").mkdir()
    pd.DataFrame(
        [{"run_id": RUN_ID, "split_fingerprint": split_fingerprint}]
    ).to_csv(root / "results/runs.csv", index=False)
    _write_spec(root, split_fingerprint=split_fingerprint)
    (root / "README.md").write_text("clean\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task 4 Test",
            "-c",
            "user.email=task4@example.test",
            "commit",
            "-q",
            "-m",
            "test fixture",
        ],
        cwd=root,
        check=True,
    )
    return splits


class _RecordingEncoder:
    dimension = 128
    pretrained = False

    def __init__(
        self,
        name: str,
        direction: str,
        recordings: dict[tuple[str, str], list[np.ndarray]],
    ) -> None:
        self.name = name
        self.direction = direction
        self.recordings = recordings

    def encode(self, image: Any) -> np.ndarray:
        self.recordings.setdefault((self.direction, self.name), []).append(
            image.pixels.copy()
        )
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[0] = 1.0
        vector[1] = np.float32(image.pixels.mean() / 255.0)
        return vector / np.linalg.norm(vector)


def _encoder_factory(
    recordings: dict[tuple[str, str], list[np.ndarray]],
    calls: list[str],
) -> Callable[..., dict[str, _RecordingEncoder]]:
    def factory(
        spec: Any,
        *,
        project_root: Path,
        direction: str,
        device: str,
    ) -> dict[str, _RecordingEncoder]:
        assert project_root.is_absolute()
        assert device == "cpu"
        calls.append(direction)
        return {
            method: _RecordingEncoder(method, direction, recordings)
            for method in spec.methods
            if method != RANDOM_FLOOR_METHOD
        }

    return factory


def _commit_fixture_change(root: Path, path: Path, message: str) -> None:
    subprocess.run(["git", "add", path], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task 4 Test",
            "-c",
            "user.email=task4@example.test",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=root,
        check=True,
    )


@pytest.fixture(scope="module")
def completed_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("task4-blind-complete")
    splits = _write_project(root)
    recordings: dict[tuple[str, str], list[np.ndarray]] = {}
    calls: list[str] = []
    receipt = build_blind_holdout_evidence(
        project_root=root,
        encoder_factory=_encoder_factory(recordings, calls),
    )
    return {
        "root": root,
        "splits": splits,
        "receipt": receipt,
        "recordings": recordings,
        "factory_calls": calls,
    }


def test_visible_holdout_article_types_stop_before_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _write_project(tmp_path)
    splits.loc[splits["partition"].eq("holdout"), "articleType"] = "Tshirts"
    monkeypatch.setattr(blind_module, "load_splits", lambda _path: splits)
    with pytest.raises(
        RuntimeError,
        match="^blind prediction phase received visible holdout article types$",
    ):
        build_blind_holdout_evidence(
            project_root=tmp_path,
            encoder_factory=_encoder_factory({}, []),
        )


def test_partition_count_drift_is_rejected(tmp_path: Path) -> None:
    _write_project(tmp_path)
    config_path = tmp_path / "configs/task4/holdout_final_evaluation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["expected_holdout_rows"] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _commit_fixture_change(tmp_path, config_path, "mutate expected holdout rows")
    with pytest.raises(ValueError, match="partition counts"):
        build_blind_holdout_evidence(
            project_root=tmp_path,
            encoder_factory=_encoder_factory({}, []),
        )


def test_receipt_hashes_label_free_complete_rankings(
    completed_project: dict[str, Any],
) -> None:
    root = completed_project["root"]
    receipt = completed_project["receipt"]
    assert receipt["labels_opened"] is False
    assert receipt["teacher_test_scored"] is False
    assert len(receipt["git"]["commit"]) == 40
    for record in receipt["artifacts"].values():
        path = root / record["path"]
        assert path.is_file()
        assert path.parent == (
            root / "results/evidence/task4/final_evaluation"
        )
        assert path.stat().st_size == record["bytes"]
        assert compute_sha256(path) == record["sha256"]
        if path.suffix == ".csv":
            assert not BANNED_COLUMNS.intersection(pd.read_csv(path, nrows=1).columns)
    output_dir = root / "results/evidence/task4/final_evaluation"
    assert {path.name for path in output_dir.iterdir()} == {
        "blind_runtime.json",
        "gallery_manifest.csv",
        "holdout_family_rankings.csv",
        "holdout_primary_rankings.csv",
        "holdout_query_manifest.csv",
        "prediction_receipt.json",
    }

    query_manifest = pd.read_csv(
        root / receipt["artifacts"]["holdout_query_manifest"]["path"],
        keep_default_na=False,
    )
    assert query_manifest["articleType_redacted"].eq("").all()

    primary = pd.read_csv(
        root / receipt["artifacts"]["holdout_primary_rankings"]["path"]
    )
    counts = primary.groupby(
        ["method", "direction", "condition", "query_id"]
    ).size()
    assert counts.eq(20).all()
    teacher = primary.loc[primary["direction"].eq("teacher")]
    v1 = primary.loc[primary["direction"].eq("v1")]
    assert set(teacher["condition"]) == set(QUERY_CONDITIONS)
    assert set(v1["condition"]) == {"clean"}


def test_receipt_binds_the_exact_gallery_image_manifest(
    completed_project: dict[str, Any],
) -> None:
    root = completed_project["root"]
    receipt = completed_project["receipt"]
    gallery = pd.read_csv(
        root / receipt["artifacts"]["gallery_manifest"]["path"],
        keep_default_na=False,
    )
    records = [
        {"id": int(row.id), "path": str(row.path), "sha256": str(row.sha256)}
        for row in gallery.loc[:, ["id", "path", "sha256"]].itertuples(index=False)
    ]
    assert receipt["inputs"]["gallery_image_set_sha256"] == canonical_sha256(records)


def test_conditions_are_query_only_and_run_before_preprocessing(
    completed_project: dict[str, Any],
) -> None:
    root = completed_project["root"]
    recordings = completed_project["recordings"]
    assert completed_project["factory_calls"] == ["teacher", "v1"]

    r5_teacher = recordings[("teacher", "r5_scratch_autoencoder")]
    r5_v1 = recordings[("v1", "r5_scratch_autoencoder")]
    assert len(r5_teacher) == HOLDOUT_ROWS * len(QUERY_CONDITIONS)
    assert len(r5_v1) == HOLDOUT_ROWS
    for method in set(METHODS) - {RANDOM_FLOOR_METHOD, "r5_scratch_autoencoder"}:
        assert len(recordings[("teacher", method)]) == (
            DEVELOPMENT_ROWS + HOLDOUT_ROWS * len(QUERY_CONDITIONS)
        )
        assert len(recordings[("v1", method)]) == HOLDOUT_ROWS

    query_path = root / "data/raw/teacher/train/images/101.jpg"
    for condition_index, condition in enumerate(QUERY_CONDITIONS):
        with Image.open(query_path) as source:
            expected = preprocess_image(apply_query_condition(source, condition), CONTRACT)
        observed = r5_teacher[condition_index * HOLDOUT_ROWS]
        assert observed.shape == (320, 240, 3)
        assert np.array_equal(observed, expected.pixels)


def test_changed_gallery_image_bytes_abort_before_ranking(tmp_path: Path) -> None:
    _write_project(tmp_path)
    image_path = tmp_path / "data/raw/teacher/train/images/1.jpg"
    image_path.write_bytes(image_path.read_bytes() + b"changed after gallery export")
    _commit_fixture_change(tmp_path, image_path, "mutate gallery bytes")
    with pytest.raises(ArtifactVerificationError, match=r"1\.jpg"):
        build_blind_holdout_evidence(
            project_root=tmp_path,
            encoder_factory=_encoder_factory({}, []),
        )
    final_dir = tmp_path / "results/evidence/task4/final_evaluation"
    assert not final_dir.exists()


def test_query_bytes_changed_after_early_verification_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(tmp_path)
    original = blind_module._validate_encoder_registry
    changed = False

    def mutate_after_query_verification(encoders: Any, spec: Any) -> None:
        nonlocal changed
        original(encoders, spec)
        if not changed:
            path = tmp_path / "data/raw/teacher/train/images/101.jpg"
            path.write_bytes(path.read_bytes() + b"changed before encoding")
            changed = True

    monkeypatch.setattr(
        blind_module,
        "_validate_encoder_registry",
        mutate_after_query_verification,
    )
    with pytest.raises(ArtifactVerificationError, match=r"101\.jpg"):
        build_blind_holdout_evidence(
            project_root=tmp_path,
            encoder_factory=_encoder_factory({}, []),
        )
    assert changed is True


@pytest.mark.parametrize(
    ("input_name", "relative_path"),
    [
        ("config", "configs/task4/holdout_final_evaluation.json"),
        ("splits", "data/processed/splits.csv"),
        ("variant_index", "data/processed/task4/external_variant_index.csv.gz"),
        ("gallery_manifest", "models/task4_holdout_gallery/manifest.json"),
        ("model_manifest", "models/task4_r5/manifest.json"),
        ("runs", "results/runs.csv"),
    ],
)
def test_blind_run_uses_captured_non_image_input_after_live_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    relative_path: str,
) -> None:
    _write_project(tmp_path)
    input_path = tmp_path / relative_path
    original = input_path.read_bytes()
    captured_labels: list[str] = []

    def mutate_after_capture(label: str, path: Path) -> None:
        captured_labels.append(label)
        if label == input_name:
            assert path == input_path
            path.write_bytes(b"invalid live replacement")

    clean_state = blind_module._git_state(tmp_path)
    monkeypatch.setattr(
        blind_module,
        "_after_blind_input_bytes_captured",
        mutate_after_capture,
        raising=False,
    )
    monkeypatch.setattr(blind_module, "_git_state", lambda _root: clean_state)

    receipt = build_blind_holdout_evidence(
        project_root=tmp_path,
        encoder_factory=_encoder_factory({}, []),
    )

    assert captured_labels == [
        "config",
        "splits",
        "variant_index",
        "gallery_manifest",
        "model_manifest",
        "runs",
    ]
    record = receipt["inputs"][input_name]
    assert record["path"] == relative_path
    assert record["sha256"] == hashlib.sha256(original).hexdigest()
    assert record["bytes"] == len(original)


def test_published_blind_evidence_cannot_be_rebuilt(
    completed_project: dict[str, Any],
) -> None:
    root = completed_project["root"]
    output_dir = root / "results/evidence/task4/final_evaluation"
    before = {
        path.name: compute_sha256(path)
        for path in output_dir.iterdir()
        if path.is_file()
    }
    factory_calls: list[str] = []
    with pytest.raises(RuntimeError, match="publish-once.*fresh destination"):
        build_blind_holdout_evidence(
            project_root=root,
            encoder_factory=_encoder_factory({}, factory_calls),
        )
    assert factory_calls == []
    after = {
        path.name: compute_sha256(path)
        for path in output_dir.iterdir()
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "marker",
    ["unlock_receipt.json", "evaluation_manifest.json"],
)
def test_later_phase_state_blocks_blind_phase_before_split_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    output_dir = tmp_path / "results/evidence/task4/final_evaluation"
    output_dir.mkdir(parents=True)
    (output_dir / marker).write_text("{}\n", encoding="utf-8")

    def unexpected_split_read(_path: Path) -> pd.DataFrame:
        raise AssertionError("split must not be read after publication")

    monkeypatch.setattr(blind_module, "load_splits", unexpected_split_read)
    with pytest.raises(RuntimeError, match="publish-once.*fresh destination"):
        build_blind_holdout_evidence(
            project_root=tmp_path,
            encoder_factory=_encoder_factory({}, []),
        )


def test_dirty_tracked_tree_stops_before_receipt(tmp_path: Path) -> None:
    _write_project(tmp_path)
    (tmp_path / "README.md").write_text("dirty\n", encoding="utf-8")
    output_dir = tmp_path / "results/evidence/task4/final_evaluation"
    with pytest.raises(RuntimeError, match="receipt was not written"):
        build_blind_holdout_evidence(
            project_root=tmp_path,
            encoder_factory=_encoder_factory({}, []),
        )
    assert not output_dir.exists()
    assert list(output_dir.parent.glob(".final_evaluation.*")) == []
