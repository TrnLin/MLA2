from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import fashion.task4 as task4
from fashion.data.splits import cv_assignment_digest
from fashion.task4.gallery_artifact import (
    GALLERY_ARTIFACT_SCHEMA_VERSION,
    METADATA_COLUMNS,
    SOURCE_IDENTITY_FIELDS,
    export_teacher_gallery_artifact,
    load_teacher_gallery_artifact,
)
from scripts.task4 import export_teacher_gallery as launcher

SOURCE_EXTRA_FIELDS = (
    "feature_method",
    "transform_seconds",
    "source_bytes",
    "ids_sha256",
    "features_sha256",
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_row(product_id: int, fold: int) -> dict[str, object]:
    return {
        "id": product_id,
        "path": f"data/raw/teacher/train/images/{product_id}.jpg",
        "sha256": f"{product_id:064x}",
        "articleType": "Tshirts",
        "baseColour": "Blue",
        "productDisplayName": f"Product {product_id}",
        "duplicate_group": f"duplicate-{product_id}",
        "product_name_key": f"product-{product_id}",
        "product_family_group": f"family-{product_id}",
        "partition": "development",
        "cv_fold": fold,
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "quarantine_reason": "",
        "season": "Summer",
        "gender": "Unisex",
        "usage": "Casual",
        "has_articleType_label": True,
        "has_season_label": True,
        "has_gender_label": True,
        "has_usage_label": True,
    }


def _split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _split_row(1, 1),
            _split_row(2, 0),
            _split_row(3, 2),
            _split_row(4, 3),
            _split_row(5, 4),
        ]
    )


def _write_splits(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _unit_features(rows: int, dimension: int = 128) -> np.ndarray:
    features = np.zeros((rows, dimension), dtype=np.float32)
    if dimension:
        features[:, 0] = 1.0
    return features


def _write_source_cache(
    directory: Path,
    splits: pd.DataFrame,
    *,
    ids: np.ndarray | None = None,
    features: np.ndarray | None = None,
) -> str:
    directory.mkdir()
    source_ids = np.arange(1, 6, dtype=np.int64) if ids is None else ids
    source_features = (
        _unit_features(len(source_ids)) if features is None else features
    )
    np.save(directory / "ids.npy", source_ids, allow_pickle=False)
    np.save(directory / "features.npy", source_features, allow_pickle=False)
    identity = {
        "schema_version": "1.0.0",
        "scope": "development",
        "run_id": "task4-r5-test",
        "run_kind": "candidate",
        "method": "R5",
        "checkpoint_sha256": "c" * 64,
        "config_hash": "a" * 64,
        "split_fingerprint": cv_assignment_digest(splits),
        "source": "teacher",
        "image_cache_manifest_sha256": "d" * 64,
        "source_fingerprint": "e" * 64,
        "source_statistics_sha256": "f" * 64,
        "fold": 1,
        "contract": {
            "width": 240,
            "height": 320,
            "pad_color": [255, 255, 255],
            "colour_mode": "RGB",
            "resize": "aspect_preserving_letterbox",
            "resample": "LANCZOS",
        },
        "rows": len(source_ids),
        "dimension": 128,
    }
    source_identity = hashlib.sha256(_canonical_json(identity)).hexdigest()
    manifest = {
        **identity,
        "feature_cache_identity_sha256": source_identity,
        "feature_method": "R5",
        "transform_seconds": 0.25,
        "source_bytes": len(source_ids) * 240 * 320 * 3,
        "ids_sha256": _sha256(directory / "ids.npy"),
        "features_sha256": _sha256(directory / "features.npy"),
    }
    assert set(manifest) == set(SOURCE_IDENTITY_FIELDS) | set(SOURCE_EXTRA_FIELDS) | {
        "feature_cache_identity_sha256"
    }
    (directory / "manifest.json").write_bytes(_canonical_json(manifest))
    return source_identity


@pytest.fixture
def synthetic_source(tmp_path: Path) -> dict[str, Any]:
    splits = _split_frame()
    splits_path = _write_splits(tmp_path / "splits.csv", splits)
    source_cache = tmp_path / "source-cache"
    source_identity = _write_source_cache(source_cache, splits)
    return {
        "splits": splits,
        "splits_path": splits_path,
        "source_cache": source_cache,
        "source_identity": source_identity,
    }


def _export(source: dict[str, Any], destination: Path) -> Path:
    return export_teacher_gallery_artifact(
        source["source_cache"],
        destination,
        splits_path=source["splits_path"],
        expected_source_identity_sha256=source["source_identity"],
        expected_checkpoint_sha256="c" * 64,
        expected_run_id="task4-r5-test",
    )


def _rewrite_manifest(
    directory: Path,
    change: Callable[[dict[str, Any]], None],
) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest.pop("artifact_identity_sha256")
    change(manifest)
    manifest["artifact_identity_sha256"] = hashlib.sha256(
        _canonical_json(manifest)
    ).hexdigest()
    manifest_path.write_bytes(_canonical_json(manifest))


def _reseal_file(directory: Path, filename: str) -> None:
    def update(manifest: dict[str, Any]) -> None:
        manifest["files"][filename]["sha256"] = _sha256(directory / filename)
        manifest["files"][filename]["bytes"] = (directory / filename).stat().st_size

    _rewrite_manifest(directory, update)


def test_gallery_artifact_round_trip_is_fold_safe_and_read_only(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    gallery = load_teacher_gallery_artifact(exported)

    assert GALLERY_ARTIFACT_SCHEMA_VERSION == "1.0.0"
    assert gallery.directory == exported
    assert gallery.ids.tolist() == [2, 3, 4, 5]
    assert gallery.features.shape == (4, 128)
    assert gallery.metadata["id"].tolist() == [2, 3, 4, 5]
    assert tuple(gallery.metadata.columns) == METADATA_COLUMNS
    assert gallery.manifest["safety"] == {
        "development_only": True,
        "holdout_opened": False,
        "official_teacher_test_opened": False,
        "quarantine_opened": False,
    }
    assert gallery.identity_sha256 == gallery.manifest["artifact_identity_sha256"]
    assert not gallery.ids.flags.writeable
    assert not gallery.features.flags.writeable


def test_export_is_byte_deterministic(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    first = _export(synthetic_source, tmp_path / "gallery-a")
    second = _export(synthetic_source, tmp_path / "gallery-b")

    for name in ("README.md", "ids.npy", "features.npy", "metadata.csv", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_export_rejects_wrong_pinned_source_identity(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    synthetic_source["source_identity"] = "0" * 64

    with pytest.raises(ValueError, match="source.*identity"):
        _export(synthetic_source, tmp_path / "gallery")


@pytest.mark.parametrize("filename", ["ids.npy", "features.npy"])
def test_export_rejects_changed_source_array_bytes(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
    filename: str,
) -> None:
    with (synthetic_source["source_cache"] / filename).open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="source.*SHA-256|source.*hash"):
        _export(synthetic_source, tmp_path / "gallery")


def test_export_rejects_source_id_set_different_from_development(
    tmp_path: Path,
) -> None:
    splits = _split_frame()
    source_cache = tmp_path / "source-cache"
    source_identity = _write_source_cache(
        source_cache,
        splits,
        ids=np.array([1, 2, 3, 4, 6], dtype=np.int64),
    )
    source = {
        "splits_path": _write_splits(tmp_path / "splits.csv", splits),
        "source_cache": source_cache,
        "source_identity": source_identity,
    }

    with pytest.raises(ValueError, match="development IDs"):
        _export(source, tmp_path / "gallery")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("id", 1),
        ("sha256", f"{1:064x}"),
        ("duplicate_group", "duplicate-1"),
        ("product_family_group", "family-1"),
    ],
)
def test_export_rejects_gallery_query_overlap(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    splits = _split_frame()
    splits.loc[splits["id"].eq(2), column] = value
    source_cache = tmp_path / "source-cache"
    source_identity = _write_source_cache(source_cache, splits)
    source = {
        "splits_path": _write_splits(tmp_path / "splits.csv", splits),
        "source_cache": source_cache,
        "source_identity": source_identity,
    }

    with pytest.raises(
        ValueError,
        match="unique|duplicate|crosses cv folds|query and gallery",
    ):
        _export(source, tmp_path / "gallery")


@pytest.mark.parametrize(
    "bad_path",
    ["/tmp/2.jpg", "../teacher/2.jpg"],
)
def test_export_rejects_unsafe_metadata_paths(
    tmp_path: Path,
    bad_path: str,
) -> None:
    splits = _split_frame()
    splits.loc[splits["id"].eq(2), "path"] = bad_path
    source_cache = tmp_path / "source-cache"
    source_identity = _write_source_cache(source_cache, splits)
    source = {
        "splits_path": _write_splits(tmp_path / "splits.csv", splits),
        "source_cache": source_cache,
        "source_identity": source_identity,
    }

    with pytest.raises(ValueError, match="metadata path"):
        _export(source, tmp_path / "gallery")


def test_export_rejects_existing_destination(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "gallery"
    destination.mkdir()

    with pytest.raises(ValueError, match="destination already exists"):
        _export(synthetic_source, destination)


@pytest.mark.parametrize(
    ("ids", "features", "message"),
    [
        (
            np.array([2, 1, 3, 4, 5], dtype=np.int64),
            _unit_features(5),
            "source.*arrays",
        ),
        (
            np.arange(1, 6, dtype=np.int32),
            _unit_features(5),
            "source.*arrays",
        ),
        (
            np.arange(1, 6, dtype=np.int64),
            _unit_features(5).astype(np.float64),
            "source.*arrays",
        ),
        (
            np.arange(1, 6, dtype=np.int64),
            np.full((5, 128), np.nan, dtype=np.float32),
            "source.*arrays",
        ),
        (
            np.arange(1, 6, dtype=np.int64),
            _unit_features(5, dimension=127),
            "source.*arrays",
        ),
        (
            np.arange(1, 6, dtype=np.int64),
            np.zeros((5, 128), dtype=np.float32),
            "source.*arrays",
        ),
    ],
    ids=["unsorted-ids", "wrong-id-dtype", "wrong-feature-dtype", "nonfinite", "dimension", "norm"],
)
def test_export_rejects_invalid_source_arrays(
    tmp_path: Path,
    ids: np.ndarray,
    features: np.ndarray,
    message: str,
) -> None:
    splits = _split_frame()
    source_cache = tmp_path / "source-cache"
    source_identity = _write_source_cache(
        source_cache,
        splits,
        ids=ids,
        features=features,
    )
    source = {
        "splits_path": _write_splits(tmp_path / "splits.csv", splits),
        "source_cache": source_cache,
        "source_identity": source_identity,
    }

    with pytest.raises(ValueError, match=message):
        _export(source, tmp_path / "gallery")


@pytest.mark.parametrize("filename", ["features.npy", "metadata.csv"])
def test_loader_rejects_changed_exported_bytes(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
    filename: str,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    with (exported / filename).open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="SHA-256|byte count"):
        load_teacher_gallery_artifact(exported)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "9.9.9"),
        ("artifact_type", "wrong-artifact"),
    ],
)
def test_loader_rejects_wrong_manifest_identity(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    _rewrite_manifest(exported, lambda manifest: manifest.__setitem__(field, value))

    with pytest.raises(ValueError, match="manifest identity"):
        load_teacher_gallery_artifact(exported)


@pytest.mark.parametrize("bad_path", ["/tmp/ids.npy", "../ids.npy", "nested/ids.npy"])
def test_loader_rejects_nonlocal_manifest_file_paths(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
    bad_path: str,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    _rewrite_manifest(
        exported,
        lambda manifest: manifest["files"]["ids.npy"].__setitem__("path", bad_path),
    )

    with pytest.raises(ValueError, match="file path"):
        load_teacher_gallery_artifact(exported)


@pytest.mark.parametrize(
    "features",
    [
        _unit_features(4).astype(np.float64),
        np.full((4, 128), np.nan, dtype=np.float32),
        _unit_features(4, dimension=127),
        np.zeros((4, 128), dtype=np.float32),
    ],
    ids=["wrong-dtype", "nonfinite", "dimension", "norm"],
)
def test_loader_rejects_invalid_exported_features(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
    features: np.ndarray,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    np.save(exported / "features.npy", features, allow_pickle=False)
    _reseal_file(exported, "features.npy")

    with pytest.raises(ValueError, match="gallery.*arrays"):
        load_teacher_gallery_artifact(exported)


@pytest.mark.parametrize(
    "ids",
    [
        np.array([3, 2, 4, 5], dtype=np.int64),
        np.array([2, 3, 4, 5], dtype=np.int32),
    ],
    ids=["unsorted", "wrong-dtype"],
)
def test_loader_rejects_invalid_exported_ids(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
    ids: np.ndarray,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    np.save(exported / "ids.npy", ids, allow_pickle=False)
    _reseal_file(exported, "ids.npy")

    with pytest.raises(ValueError, match="gallery.*arrays"):
        load_teacher_gallery_artifact(exported)


def test_loader_rejects_metadata_id_order_mismatch(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    metadata = pd.read_csv(exported / "metadata.csv", keep_default_na=False)
    metadata = metadata.iloc[::-1]
    metadata.to_csv(exported / "metadata.csv", index=False, lineterminator="\n")
    _reseal_file(exported, "metadata.csv")

    with pytest.raises(ValueError, match="metadata.*ID order"):
        load_teacher_gallery_artifact(exported)


@pytest.mark.parametrize("cv_fold", [1, 99])
def test_loader_rejects_non_gallery_metadata_row(
    synthetic_source: dict[str, Any],
    tmp_path: Path,
    cv_fold: int,
) -> None:
    exported = _export(synthetic_source, tmp_path / "gallery")
    metadata = pd.read_csv(exported / "metadata.csv", keep_default_na=False)
    metadata.loc[0, "cv_fold"] = cv_fold
    metadata.to_csv(exported / "metadata.csv", index=False, lineterminator="\n")
    _reseal_file(exported, "metadata.csv")

    with pytest.raises(ValueError, match="fold-1 gallery"):
        load_teacher_gallery_artifact(exported)


def test_gallery_artifact_api_is_public() -> None:
    assert task4.GALLERY_ARTIFACT_SCHEMA_VERSION == "1.0.0"
    assert task4.TeacherGallery is not None
    assert task4.export_teacher_gallery_artifact is export_teacher_gallery_artifact
    assert task4.load_teacher_gallery_artifact is load_teacher_gallery_artifact


def test_pinned_exporter_forwards_only_export_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_export(
        source_cache: Path,
        destination: Path,
        **kwargs: object,
    ) -> Path:
        captured.update(
            source_cache=source_cache,
            destination=destination,
            **kwargs,
        )
        return destination

    source_cache = tmp_path / "source"
    destination = tmp_path / "gallery"
    splits = tmp_path / "splits.csv"
    monkeypatch.setattr(launcher, "export_teacher_gallery_artifact", fake_export, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_teacher_gallery.py",
            "--source-cache",
            str(source_cache),
            "--destination",
            str(destination),
            "--splits",
            str(splits),
        ],
    )

    launcher.main()

    assert captured == {
        "source_cache": source_cache,
        "destination": destination,
        "splits_path": splits,
        "expected_source_identity_sha256": (
            "00ea23461b4d5ef35a59d2078be8e7f6e9a24f5089e42a4057b1bfdd057747bc"
        ),
        "expected_checkpoint_sha256": (
            "521e96f3df9e28853309bd607030523f59ef8425326da56dfcf2b754e97631b3"
        ),
        "expected_run_id": "task4-candidate-r5-task9-preexec",
    }
    assert capsys.readouterr().out == f"Teacher gallery: {destination}\n"
