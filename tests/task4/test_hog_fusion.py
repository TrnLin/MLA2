from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import FeatureIndex

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PreprocessingContract(width=240, height=320)
HOG_FINGERPRINT = "c4604537474ff5ed8889a9e480053b225064fb20027e147582960bb3afd3a848"


def _load_fusion():
    path = ROOT / "src/fashion/task4/hog_fusion.py"
    assert path.is_file(), "Task 4 HOG fusion production module is missing"
    spec = importlib.util.spec_from_file_location("fashion.task4.hog_fusion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unit(dimension: int, position: int = 0) -> np.ndarray:
    values = np.zeros(dimension, dtype=np.float32)
    values[position] = np.float32(1.0)
    return values


def _index(
    *,
    component: str,
    source: str = "teacher",
    ids: np.ndarray | None = None,
    fold: int = 1,
    contract: PreprocessingContract = CONTRACT,
    method: str | None = None,
    descriptor_fingerprint: str | None = None,
    checkpoint_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
) -> FeatureIndex:
    dimension = 1_728 if component == "hog" else 400
    ids = np.array([1, 2], dtype=np.int64) if ids is None else ids
    features = np.stack([_unit(dimension, 0), _unit(dimension, 1)])
    return FeatureIndex(
        source=source,
        contract=contract,
        ids=ids,
        features=features,
        transform_seconds=1.0,
        source_bytes=100,
        method=method
        or (
            "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1"
            if component == "hog"
            else "spatial-hsv-edge-4x4-v2"
        ),
        fold=fold,
        checkpoint_fingerprint=checkpoint_fingerprint,
        config_fingerprint=config_fingerprint,
        descriptor_fingerprint=(
            descriptor_fingerprint
            if descriptor_fingerprint is not None
            else (HOG_FINGERPRINT if component == "hog" else None)
        ),
    )


def _manifests(source: str = "teacher") -> tuple[dict[str, object], dict[str, object]]:
    contract = CONTRACT.to_dict()
    hog = {
        "schema_version": "1.0.0",
        "scope": "development",
        "fold": 1,
        "source": source,
        "rows": 2,
        "dimension": 1_728,
        "contract": contract,
        "method": "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1",
        "descriptor_fingerprint": HOG_FINGERPRINT,
        "checkpoint_fingerprint": None,
        "config_fingerprint": None,
        "id_sha256": "a" * 64,
        "ids_file_sha256": "b" * 64,
        "features_file_sha256": "c" * 64,
        "image_cache_manifest_sha256": "d" * 64,
        "source_fingerprint": "e" * 64,
    }
    spatial = {
        "schema_version": "1.0.0",
        "scope": "development",
        "source": source,
        "rows": 2,
        "feature_shape": [2, 400],
        "feature_dtype": "float32",
        "contract": contract,
        "probe": "spatial-hsv-edge-4x4-v2",
        "source_fingerprint": "e" * 64,
    }
    return hog, spatial


def test_equal_fusion_has_exact_shape_dtype_norm_and_block_contribution() -> None:
    fusion = _load_fusion()

    descriptor = fusion.fuse_descriptors(_unit(1_728, 7), _unit(400, 11))

    assert descriptor.shape == (2_128,)
    assert descriptor.dtype == np.float32
    assert np.isfinite(descriptor).all()
    assert np.linalg.norm(descriptor) == pytest.approx(1.0, abs=1e-6)
    expected_block_norm = 1.0 / np.sqrt(2.0)
    assert np.linalg.norm(descriptor[:1_728]) == pytest.approx(expected_block_norm, abs=1e-6)
    assert np.linalg.norm(descriptor[1_728:]) == pytest.approx(expected_block_norm, abs=1e-6)


@pytest.mark.parametrize(
    ("hog", "spatial", "message"),
    [
        (_unit(1_727), _unit(400), "HOG.*1,728"),
        (_unit(1_728).astype(np.float64), _unit(400), "HOG.*float32"),
        (np.full(1_728, np.nan, dtype=np.float32), _unit(400), "HOG.*finite"),
        (np.zeros(1_728, dtype=np.float32), _unit(400), "HOG.*unit"),
        (_unit(1_728), _unit(399), "HSV-edge.*400"),
        (_unit(1_728), np.full(400, np.inf, dtype=np.float32), "HSV-edge.*finite"),
        (_unit(1_728), _unit(400) * np.float32(2.0), "HSV-edge.*unit"),
    ],
)
def test_equal_fusion_rejects_malformed_component_vectors(
    hog: np.ndarray,
    spatial: np.ndarray,
    message: str,
) -> None:
    fusion = _load_fusion()

    with pytest.raises(ValueError, match=message):
        fusion.fuse_descriptors(hog, spatial)


def test_fusion_config_has_stable_complete_identity_and_fixed_production_weight() -> None:
    fusion = _load_fusion()

    assert fusion.HOG_FUSION_DESCRIPTOR_DIM == 2_128
    assert fusion.HOG_FUSION_METHOD == "hog-plus-spatial-hsv-edge-equal-v1"
    assert fusion.HOG_FUSION_CONFIG.to_dict() == {
        "schema_version": "1.0.0",
        "method": "hog-plus-spatial-hsv-edge-equal-v1",
        "components": [
            {
                "name": "hog",
                "method": "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1",
                "descriptor_fingerprint": HOG_FINGERPRINT,
                "dimension": 1_728,
                "weight": 1.0 / np.sqrt(2.0),
            },
            {
                "name": "spatial_hsv_edge",
                "method": "spatial-hsv-edge-4x4-v2",
                "descriptor_fingerprint": None,
                "dimension": 400,
                "weight": 1.0 / np.sqrt(2.0),
            },
        ],
        "preprocessing_contract": CONTRACT.to_dict(),
        "concatenation_rule": "hog_then_spatial_hsv_edge",
        "normalization_rule": "independent_unit_components_then_equal_weight_concat",
        "descriptor_dimension": 2_128,
    }
    assert len(fusion.HOG_FUSION_CONFIG_FINGERPRINT) == 64
    assert fusion.HOG_FUSION_CONFIG.fingerprint() == fusion.HOG_FUSION_CONFIG_FINGERPRINT
    changed = replace(
        fusion.HOG_FUSION_CONFIG,
        hog_method="declared-different-hog-v2",
    )
    assert changed.fingerprint() != fusion.HOG_FUSION_CONFIG_FINGERPRINT
    with pytest.raises(ValueError, match="equal"):
        replace(fusion.HOG_FUSION_CONFIG, hog_weight=0.7)


@pytest.mark.parametrize(
    ("mutate_hog", "mutate_spatial", "message"),
    [
        ({"ids": np.array([2, 1], dtype=np.int64)}, {}, "IDs.*order"),
        ({"source": "teacher"}, {"source": "v1"}, "source"),
        ({"fold": 0}, {}, "fold"),
        (
            {"contract": PreprocessingContract(width=96, height=128)},
            {},
            "contract",
        ),
        ({"method": "wrong-hog"}, {}, "HOG.*method"),
        ({"descriptor_fingerprint": "f" * 64}, {}, "HOG.*fingerprint"),
        ({"checkpoint_fingerprint": "checkpoint"}, {}, "untrained"),
        ({}, {"method": "wrong-spatial"}, "HSV-edge.*method"),
    ],
)
def test_fusion_rejects_index_identity_and_provenance_mismatches(
    mutate_hog: dict[str, object],
    mutate_spatial: dict[str, object],
    message: str,
) -> None:
    fusion = _load_fusion()
    hog = _index(component="hog")
    spatial = _index(component="spatial")
    for key, value in mutate_hog.items():
        object.__setattr__(hog, key, value)
    for key, value in mutate_spatial.items():
        object.__setattr__(spatial, key, value)
    hog_manifest, spatial_manifest = _manifests()

    with pytest.raises(ValueError, match=message):
        fusion.fuse_feature_indexes(
            hog,
            spatial,
            hog_manifest=hog_manifest,
            spatial_manifest=spatial_manifest,
        )


def test_fused_index_has_only_untrained_descriptor_identity() -> None:
    fusion = _load_fusion()
    hog = _index(component="hog")
    spatial = _index(component="spatial")
    hog_manifest, spatial_manifest = _manifests()

    fused = fusion.fuse_feature_indexes(
        hog,
        spatial,
        hog_manifest=hog_manifest,
        spatial_manifest=spatial_manifest,
    )

    assert fused.source == "teacher"
    assert np.array_equal(fused.ids, np.array([1, 2], dtype=np.int64))
    assert fused.features.shape == (2, 2_128)
    assert fused.features.dtype == np.float32
    assert fused.method == fusion.HOG_FUSION_METHOD
    assert fused.fold == 1
    assert fused.checkpoint_fingerprint is None
    assert fused.config_fingerprint is None
    assert fused.descriptor_fingerprint == fusion.HOG_FUSION_CONFIG_FINGERPRINT
    assert fused.provenance["hog"]["features_file_sha256"] == "c" * 64
    assert fused.provenance["spatial_hsv_edge"]["source_fingerprint"] == "e" * 64


def test_fusion_rejects_manifest_provenance_mismatch() -> None:
    fusion = _load_fusion()
    hog = _index(component="hog")
    spatial = _index(component="spatial")
    hog_manifest, spatial_manifest = _manifests()
    spatial_manifest["source_fingerprint"] = "different"

    with pytest.raises(ValueError, match="source fingerprint"):
        fusion.fuse_feature_indexes(
            hog,
            spatial,
            hog_manifest=hog_manifest,
            spatial_manifest=spatial_manifest,
        )
