"""Strict equal-block HOG and spatial HSV-edge fusion for Task 4."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from fashion.task4.hog import (
    HOG_CONFIG_FINGERPRINT,
    HOG_DESCRIPTOR_DIM,
    HOG_METHOD,
    extract_hog,
)
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import CachedFeatureIndex, FeatureIndex
from fashion.task4.probe import PROBE_VERSION, extract_spatial_probe

HOG_FUSION_DESCRIPTOR_DIM = 2_128
HOG_FUSION_METHOD = "hog-plus-spatial-hsv-edge-equal-v1"
SPATIAL_HSV_EDGE_DESCRIPTOR_DIM = 400
_CONTRACT = PreprocessingContract(width=240, height=320)
_EQUAL_WEIGHT = float(1.0 / np.sqrt(2.0))
_UNIT_ATOL = 1e-5


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HOGFusionConfig:
    """Complete, versioned identity of the approved equal-block fusion."""

    method: str = HOG_FUSION_METHOD
    hog_method: str = HOG_METHOD
    hog_descriptor_fingerprint: str = HOG_CONFIG_FINGERPRINT
    hog_dimension: int = HOG_DESCRIPTOR_DIM
    hog_weight: float = _EQUAL_WEIGHT
    spatial_method: str = PROBE_VERSION
    spatial_descriptor_fingerprint: None = None
    spatial_dimension: int = SPATIAL_HSV_EDGE_DESCRIPTOR_DIM
    spatial_weight: float = _EQUAL_WEIGHT
    input_width: int = 240
    input_height: int = 320

    def __post_init__(self) -> None:
        if not self.method.strip() or not self.hog_method.strip():
            raise ValueError("fusion methods must be non-empty")
        if not self.spatial_method.strip():
            raise ValueError("spatial HSV-edge method must be non-empty")
        if not self.hog_descriptor_fingerprint.strip():
            raise ValueError("HOG descriptor fingerprint must be non-empty")
        if self.hog_dimension != HOG_DESCRIPTOR_DIM:
            raise ValueError("fusion HOG dimension must be 1,728")
        if self.spatial_dimension != SPATIAL_HSV_EDGE_DESCRIPTOR_DIM:
            raise ValueError("fusion HSV-edge dimension must be 400")
        if (self.input_width, self.input_height) != (240, 320):
            raise ValueError("fusion requires the frozen 240x320 contract")
        if not np.isclose(self.hog_weight, _EQUAL_WEIGHT, atol=0.0, rtol=0.0):
            raise ValueError("fusion requires equal HOG and HSV-edge contribution")
        if not np.isclose(self.spatial_weight, _EQUAL_WEIGHT, atol=0.0, rtol=0.0):
            raise ValueError("fusion requires equal HOG and HSV-edge contribution")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "method": self.method,
            "components": [
                {
                    "name": "hog",
                    "method": self.hog_method,
                    "descriptor_fingerprint": self.hog_descriptor_fingerprint,
                    "dimension": self.hog_dimension,
                    "weight": self.hog_weight,
                },
                {
                    "name": "spatial_hsv_edge",
                    "method": self.spatial_method,
                    "descriptor_fingerprint": self.spatial_descriptor_fingerprint,
                    "dimension": self.spatial_dimension,
                    "weight": self.spatial_weight,
                },
            ],
            "preprocessing_contract": _CONTRACT.to_dict(),
            "concatenation_rule": "hog_then_spatial_hsv_edge",
            "normalization_rule": ("independent_unit_components_then_equal_weight_concat"),
            "descriptor_dimension": HOG_FUSION_DESCRIPTOR_DIM,
        }

    def fingerprint(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()))


HOG_FUSION_CONFIG = HOGFusionConfig()
HOG_FUSION_CONFIG_FINGERPRINT = HOG_FUSION_CONFIG.fingerprint()


@dataclass(frozen=True)
class ParentFeatureCache:
    """One reopened and fully validated parent descriptor cache."""

    component: Literal["hog", "spatial_hsv_edge"]
    cache_dir: Path
    index: FeatureIndex
    manifest: dict[str, object]
    manifest_sha256: str
    ids_file_sha256: str
    features_file_sha256: str

    def identity(self) -> dict[str, object]:
        return {
            "component": self.component,
            "manifest_sha256": self.manifest_sha256,
            "ids_file_sha256": self.ids_file_sha256,
            "features_file_sha256": self.features_file_sha256,
            "manifest": self.manifest,
        }


__all__ = (
    "HOG_FUSION_CONFIG",
    "HOG_FUSION_CONFIG_FINGERPRINT",
    "HOG_FUSION_DESCRIPTOR_DIM",
    "HOG_FUSION_METHOD",
    "HOGFusionConfig",
    "ParentFeatureCache",
    "ensure_fused_feature_index",
    "extract_hog_fusion",
    "fuse_descriptors",
    "fuse_feature_indexes",
    "load_parent_feature_cache",
)


def _validated_vector(
    values: np.ndarray,
    *,
    dimension: int,
    label: str,
) -> np.ndarray:
    vector = np.asarray(values)
    if vector.shape != (dimension,):
        raise ValueError(f"{label} descriptor must contain exactly {dimension:,} values")
    if vector.dtype != np.float32:
        raise ValueError(f"{label} descriptor must have float32 dtype")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} descriptor must be finite")
    if not np.isclose(
        np.linalg.norm(vector),
        1.0,
        atol=_UNIT_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{label} descriptor must have unit norm")
    return vector


def fuse_descriptors(
    hog: np.ndarray,
    spatial_hsv_edge: np.ndarray,
    config: HOGFusionConfig = HOG_FUSION_CONFIG,
) -> np.ndarray:
    """Fuse two valid unit descriptors with one fixed equal-block weight."""

    hog_vector = _validated_vector(
        hog,
        dimension=config.hog_dimension,
        label="HOG",
    )
    spatial_vector = _validated_vector(
        spatial_hsv_edge,
        dimension=config.spatial_dimension,
        label="HSV-edge",
    )
    fused = np.concatenate(
        [
            hog_vector * np.float32(config.hog_weight),
            spatial_vector * np.float32(config.spatial_weight),
        ]
    ).astype(np.float32, copy=False)
    if (
        fused.shape != (HOG_FUSION_DESCRIPTOR_DIM,)
        or not np.isfinite(fused).all()
        or not np.isclose(np.linalg.norm(fused), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("fused descriptor is malformed, non-finite, or non-unit")
    return fused


def extract_hog_fusion(
    pixels: np.ndarray,
    content_mask: np.ndarray,
) -> np.ndarray:
    """Compute both approved parent descriptors and fuse them."""

    return fuse_descriptors(
        extract_hog(pixels, content_mask),
        extract_spatial_probe(pixels, content_mask),
    )


def _validate_feature_matrix(
    index: FeatureIndex,
    *,
    dimension: int,
    label: str,
) -> None:
    ids = np.asarray(index.ids)
    features = np.asarray(index.features)
    if ids.dtype != np.int64 or ids.ndim != 1:
        raise ValueError(f"{label} IDs must be a one-dimensional int64 array")
    if not np.array_equal(ids, np.sort(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label} IDs and order must be sorted and unique")
    if features.dtype != np.float32 or features.shape != (len(ids), dimension):
        raise ValueError(f"{label} feature matrix has malformed shape or dtype")
    for start in range(0, len(features), 1024):
        batch = np.asarray(features[start : start + 1024])
        if not np.isfinite(batch).all():
            raise ValueError(f"{label} features must be finite")
        if not np.allclose(
            np.linalg.norm(batch, axis=1),
            1.0,
            atol=_UNIT_ATOL,
            rtol=0.0,
        ):
            raise ValueError(f"{label} features must have unit norm")


def _validate_parent_manifests(
    hog_index: FeatureIndex,
    spatial_index: FeatureIndex,
    *,
    hog_manifest: dict[str, object],
    spatial_manifest: dict[str, object],
    config: HOGFusionConfig,
) -> None:
    source = hog_index.source
    hog_expected = {
        "scope": "development",
        "fold": 1,
        "source": source,
        "rows": len(hog_index.ids),
        "dimension": config.hog_dimension,
        "contract": _CONTRACT.to_dict(),
        "method": config.hog_method,
        "descriptor_fingerprint": config.hog_descriptor_fingerprint,
        "checkpoint_fingerprint": None,
        "config_fingerprint": None,
    }
    if any(hog_manifest.get(key) != value for key, value in hog_expected.items()):
        raise ValueError("HOG parent manifest provenance does not match its index")
    spatial_expected = {
        "scope": "development",
        "source": source,
        "rows": len(spatial_index.ids),
        "feature_shape": [len(spatial_index.ids), config.spatial_dimension],
        "feature_dtype": "float32",
        "contract": _CONTRACT.to_dict(),
        "probe": config.spatial_method,
    }
    if any(spatial_manifest.get(key) != value for key, value in spatial_expected.items()):
        raise ValueError("HSV-edge parent manifest provenance does not match its index")
    if hog_manifest.get("source_fingerprint") != spatial_manifest.get("source_fingerprint"):
        raise ValueError("parent source fingerprints do not match")


def _validate_component_indexes(
    hog_index: FeatureIndex,
    spatial_index: FeatureIndex,
    *,
    hog_manifest: dict[str, object],
    spatial_manifest: dict[str, object],
    config: HOGFusionConfig,
) -> None:
    if hog_index.source != spatial_index.source:
        raise ValueError("component source identities must match")
    if hog_index.fold != 1 or spatial_index.fold != 1:
        raise ValueError("component fold provenance must be development fold 1")
    if hog_index.contract != _CONTRACT or spatial_index.contract != _CONTRACT:
        raise ValueError("component indexes must use the frozen preprocessing contract")
    if not np.array_equal(hog_index.ids, spatial_index.ids):
        raise ValueError("component IDs and order must match exactly")
    if hog_index.method != config.hog_method:
        raise ValueError("HOG method provenance does not match")
    if hog_index.descriptor_fingerprint != config.hog_descriptor_fingerprint:
        raise ValueError("HOG descriptor fingerprint provenance does not match")
    if hog_index.checkpoint_fingerprint is not None or hog_index.config_fingerprint is not None:
        raise ValueError("HOG component must carry only untrained provenance")
    if spatial_index.method != config.spatial_method:
        raise ValueError("HSV-edge method provenance does not match")
    if (
        spatial_index.checkpoint_fingerprint is not None
        or spatial_index.config_fingerprint is not None
        or spatial_index.descriptor_fingerprint is not None
    ):
        raise ValueError("HSV-edge component must carry legacy untrained provenance")
    _validate_feature_matrix(
        hog_index,
        dimension=config.hog_dimension,
        label="HOG",
    )
    _validate_feature_matrix(
        spatial_index,
        dimension=config.spatial_dimension,
        label="HSV-edge",
    )
    _validate_parent_manifests(
        hog_index,
        spatial_index,
        hog_manifest=hog_manifest,
        spatial_manifest=spatial_manifest,
        config=config,
    )


def fuse_feature_indexes(
    hog_index: FeatureIndex,
    spatial_index: FeatureIndex,
    *,
    hog_manifest: dict[str, object],
    spatial_manifest: dict[str, object],
    config: HOGFusionConfig = HOG_FUSION_CONFIG,
) -> FeatureIndex:
    """Fuse two complete, aligned component indexes after strict validation."""

    _validate_component_indexes(
        hog_index,
        spatial_index,
        hog_manifest=hog_manifest,
        spatial_manifest=spatial_manifest,
        config=config,
    )
    features = np.concatenate(
        [
            np.asarray(hog_index.features) * np.float32(config.hog_weight),
            np.asarray(spatial_index.features) * np.float32(config.spatial_weight),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    _validate_feature_matrix(
        FeatureIndex(
            source=hog_index.source,
            contract=_CONTRACT,
            ids=hog_index.ids,
            features=features,
            transform_seconds=0.0,
            source_bytes=0,
            method=config.method,
            fold=1,
            descriptor_fingerprint=config.fingerprint(),
        ),
        dimension=HOG_FUSION_DESCRIPTOR_DIM,
        label="fused",
    )
    return FeatureIndex(
        source=hog_index.source,
        contract=_CONTRACT,
        ids=np.asarray(hog_index.ids),
        features=features,
        transform_seconds=(
            float(hog_index.transform_seconds) + float(spatial_index.transform_seconds)
        ),
        source_bytes=max(int(hog_index.source_bytes), int(spatial_index.source_bytes)),
        method=config.method,
        fold=1,
        checkpoint_fingerprint=None,
        config_fingerprint=None,
        descriptor_fingerprint=config.fingerprint(),
        provenance={
            "hog": dict(hog_manifest),
            "spatial_hsv_edge": dict(spatial_manifest),
        },
    )


def _load_parent_arrays(
    cache_dir: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, str, str, str]:
    manifest_path = cache_dir / "manifest.json"
    ids_path = cache_dir / "ids.npy"
    features_path = cache_dir / "features.npy"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    ids_hash = _sha256_file(ids_path)
    features_hash = _sha256_file(features_path)
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    return (
        manifest,
        ids,
        features,
        _sha256_bytes(manifest_bytes),
        ids_hash,
        features_hash,
    )


def load_parent_feature_cache(
    cache_dir: str | Path,
    *,
    component: Literal["hog", "spatial_hsv_edge"],
) -> ParentFeatureCache:
    """Reopen and validate one immutable parent evidence cache."""

    directory = Path(cache_dir)
    manifest, ids, features, manifest_hash, ids_hash, features_hash = _load_parent_arrays(directory)
    if component == "hog":
        if (
            manifest.get("ids_file_sha256") != ids_hash
            or manifest.get("features_file_sha256") != features_hash
        ):
            raise ValueError("HOG parent cache files fail their declared hashes")
        descriptor_fingerprint = HOG_CONFIG_FINGERPRINT
        method = HOG_METHOD
        fold = 1
        dimension = HOG_DESCRIPTOR_DIM
    elif component == "spatial_hsv_edge":
        descriptor_fingerprint = None
        method = PROBE_VERSION
        fold = 1
        dimension = SPATIAL_HSV_EDGE_DESCRIPTOR_DIM
    else:
        raise ValueError("unknown fusion parent component")
    index = FeatureIndex(
        source=str(manifest.get("source")),
        contract=_CONTRACT,
        ids=ids,
        features=features,
        transform_seconds=float(manifest.get("transform_seconds", 0.0)),
        source_bytes=int(manifest.get("source_bytes", 0)),
        method=method,
        fold=fold,
        checkpoint_fingerprint=None,
        config_fingerprint=None,
        descriptor_fingerprint=descriptor_fingerprint,
    )
    _validate_feature_matrix(index, dimension=dimension, label=component)
    return ParentFeatureCache(
        component=component,
        cache_dir=directory,
        index=index,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        ids_file_sha256=ids_hash,
        features_file_sha256=features_hash,
    )


@contextmanager
def _cache_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fused_cache_expected(
    hog_cache: ParentFeatureCache,
    spatial_cache: ParentFeatureCache,
    config: HOGFusionConfig,
) -> dict[str, object]:
    _validate_component_indexes(
        hog_cache.index,
        spatial_cache.index,
        hog_manifest=hog_cache.manifest,
        spatial_manifest=spatial_cache.manifest,
        config=config,
    )
    return {
        "schema_version": "1.0.0",
        "scope": "development",
        "fold": 1,
        "source": hog_cache.index.source,
        "rows": len(hog_cache.index.ids),
        "dimension": HOG_FUSION_DESCRIPTOR_DIM,
        "feature_dtype": "float32",
        "contract": _CONTRACT.to_dict(),
        "method": config.method,
        "descriptor_config": config.to_dict(),
        "descriptor_fingerprint": config.fingerprint(),
        "checkpoint_fingerprint": None,
        "config_fingerprint": None,
        "source_fingerprint": hog_cache.manifest.get("source_fingerprint"),
        "parents": {
            "hog": hog_cache.identity(),
            "spatial_hsv_edge": spatial_cache.identity(),
        },
    }


def _load_fused_cache(
    directory: Path,
    expected: dict[str, object],
) -> CachedFeatureIndex:
    manifest, ids, features, _, ids_hash, features_hash = _load_parent_arrays(directory)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("fused cache identity does not match")
    if (
        manifest.get("ids_file_sha256") != ids_hash
        or manifest.get("features_file_sha256") != features_hash
    ):
        raise ValueError("fused cache files fail their declared hashes")
    index = FeatureIndex(
        source=str(expected["source"]),
        contract=_CONTRACT,
        ids=ids,
        features=features,
        transform_seconds=float(manifest["transform_seconds"]),
        source_bytes=int(manifest["source_bytes"]),
        method=str(expected["method"]),
        fold=1,
        checkpoint_fingerprint=None,
        config_fingerprint=None,
        descriptor_fingerprint=str(expected["descriptor_fingerprint"]),
        provenance=expected["parents"],
    )
    _validate_feature_matrix(
        index,
        dimension=HOG_FUSION_DESCRIPTOR_DIM,
        label="fused",
    )
    return CachedFeatureIndex(cache_dir=directory, index=index, manifest=manifest)


def _build_fused_cache(
    directory: Path,
    hog_cache: ParentFeatureCache,
    spatial_cache: ParentFeatureCache,
    expected: dict[str, object],
    config: HOGFusionConfig,
) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        np.save(
            temporary / "ids.npy",
            np.asarray(hog_cache.index.ids),
            allow_pickle=False,
        )
        fused = np.lib.format.open_memmap(
            temporary / "features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(hog_cache.index.ids), HOG_FUSION_DESCRIPTOR_DIM),
        )
        for start in range(0, len(fused), 1024):
            stop = min(start + 1024, len(fused))
            fused[start:stop, : config.hog_dimension] = np.asarray(
                hog_cache.index.features[start:stop]
            ) * np.float32(config.hog_weight)
            fused[start:stop, config.hog_dimension :] = np.asarray(
                spatial_cache.index.features[start:stop]
            ) * np.float32(config.spatial_weight)
        fused.flush()
        manifest = {
            **expected,
            "transform_seconds": (
                float(hog_cache.index.transform_seconds)
                + float(spatial_cache.index.transform_seconds)
            ),
            "source_bytes": max(
                int(hog_cache.index.source_bytes),
                int(spatial_cache.index.source_bytes),
            ),
            "ids_file_sha256": _sha256_file(temporary / "ids.npy"),
            "features_file_sha256": _sha256_file(temporary / "features.npy"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def ensure_fused_feature_index(
    hog_cache: ParentFeatureCache,
    spatial_cache: ParentFeatureCache,
    *,
    cache_root: str | Path,
    config: HOGFusionConfig = HOG_FUSION_CONFIG,
) -> CachedFeatureIndex:
    """Reuse or build one fused cache without opening source images."""

    if hog_cache.component != "hog" or spatial_cache.component != "spatial_hsv_edge":
        raise ValueError("fusion parents are in the wrong component order")
    expected = _fused_cache_expected(hog_cache, spatial_cache, config)
    identity = _sha256_bytes(_canonical_json(expected))[:20]
    directory = Path(cache_root) / str(expected["source"]) / identity
    with _cache_lock(directory.parent / f".{identity}.lock"):
        if directory.is_dir():
            try:
                return _load_fused_cache(directory, expected)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        _build_fused_cache(
            directory,
            hog_cache,
            spatial_cache,
            expected,
            config,
        )
        return _load_fused_cache(directory, expected)
