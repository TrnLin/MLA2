"""Development-only evidence for Task 4 learned retrieval models."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from threadpoolctl import threadpool_info, threadpool_limits
from torch import nn

from fashion.config import ROOT
from fashion.data.splits import cv_assignment_digest, validate_split_structure
from fashion.task4.analysis import (
    CanvasStressEvaluation,
    build_query_support,
    evaluate_canvas_stress,
    mark_failure_slices,
    select_example_ids,
    summarize_failure_slices,
)
from fashion.task4.baseline import Direction, build_query_metrics
from fashion.task4.benchmark import (
    IndexCost,
    TimingPolicy,
    benchmark_source_direction,
    build_cost_record,
    build_protocol_a_search,
    summarize_timings,
)
from fashion.task4.cache import DevelopmentImageCache, load_development_image_cache
from fashion.task4.models import EMBEDDING_DIM, EMBEDDING_NORM_ATOL
from fashion.task4.preprocessing import (
    PreprocessingContract,
    load_preprocessed_image,
    normalize_for_model,
    preprocess_image,
)
from fashion.task4.preprocessing_experiment import (
    CachedFeatureIndex,
    FeatureIndex,
    PairEvaluation,
    build_odd_aspect_canvas,
    evaluate_source_pair,
    source_directions,
)
from fashion.task4.protocol import (
    RetrievalViews,
    build_development_views,
    compute_relevance_coverage,
    evaluate_family_rankings,
    evaluate_primary_rankings,
    prepare_rankings,
)
from fashion.task4.training import (
    CheckpointRecord,
    TrainingResult,
    TrainingSessionConfig,
    load_checkpoint,
    select_best_checkpoint,
)
from fashion.train.registry import Task4RunRegistry as RunRegistry

LEARNED_EVIDENCE_SCHEMA_VERSION = "1.0.0"
TEACHER_SLICE_CAVEAT = (
    "Grayscale and unusual-geometry slice membership comes from the teacher file; "
    "for V1 queries it describes the paired teacher view."
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_CONTRACT = PreprocessingContract(width=240, height=320)
_INDEX_LIMIT_BYTES = 2**30
_MEASUREMENT_ROUTE = "spawned_batch_one_checkpoint_extraction_v1"
GALLERY_POLICY_TIMING_ROUTE = "spawned_batch_one_gallery_policy_search_v1"


def _utc_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_FRAME_ARTIFACTS = {
    "quality_summary": "quality_summary.csv",
    "query_metrics": "query_metrics.csv",
    "rankings": "rankings.csv",
    "failure_slices": "failure_slices.csv",
    "canvas_summary": "canvas_summary.csv",
    "canvas_per_query": "canvas_per_query.csv",
    "canvas_rankings": "canvas_rankings.csv",
    "examples": "examples.csv",
    "timing_samples": "timing_samples.csv",
    "timing_summary": "timing_summary.csv",
    "gallery_comparison": "gallery_comparison.csv",
    "gallery_rankings": "gallery_rankings.csv",
}
LEARNED_ARTIFACT_NAMES = (
    *_FRAME_ARTIFACTS.values(),
    "cost.json",
    "manifest.json",
)
_PROVENANCE_COLUMNS = (
    "schema_version",
    "scope",
    "run_id",
    "run_kind",
    "method",
    "fold",
    "checkpoint_sha256",
    "config_hash",
    "split_fingerprint",
)
_ARTIFACT_PAYLOAD_COLUMNS = {
    "quality_summary": (
        "size",
        "width",
        "height",
        "query_source",
        "gallery_source",
        "protocol",
        "query_transform_seconds",
        "gallery_transform_seconds",
        "query_source_bytes",
        "gallery_source_bytes",
        "feature_bytes_per_image",
        "uint8_tensor_bytes_per_image",
        "float32_tensor_bytes_per_image",
        "metric",
        "k",
        "aggregation",
        "value",
        "query_count",
        "class_count",
    ),
    "query_metrics": (
        "size",
        "query_source",
        "gallery_source",
        "protocol",
        "query_id",
        "articleType",
        "ndcg_at_5",
        "precision_any_at_5",
        "precision_strict_at_5",
        "tie_rate_at_5",
        "ndcg_at_10",
        "precision_any_at_10",
        "precision_strict_at_10",
        "tie_rate_at_10",
        "ndcg_at_20",
        "precision_any_at_20",
        "precision_strict_at_20",
        "tie_rate_at_20",
        "recall_at_10",
        "hit_rate_at_10",
        "precision_at_10",
    ),
    "rankings": (
        "query_source",
        "gallery_source",
        "protocol",
        "query_id",
        "candidate_id",
        "distance",
        "rank",
    ),
    "failure_slices": (
        "size",
        "query_source",
        "gallery_source",
        "protocol",
        "slice",
        "metric",
        "k",
        "aggregation",
        "value",
        "total_queries",
        "scored_queries",
        "excluded_queries",
        "coverage",
        "caveat",
    ),
    "canvas_summary": (
        "size",
        "query_source",
        "gallery_source",
        "query_variant",
        "queries",
        "ndcg_at_10",
        "ndcg_change_from_clean",
        "mean_top10_overlap",
        "caveat",
    ),
    "canvas_per_query": (
        "size",
        "query_source",
        "gallery_source",
        "query_variant",
        "query_id",
        "clean_ndcg_at_10",
        "canvas_ndcg_at_10",
        "ndcg_change_from_clean",
        "top10_overlap",
        "caveat",
    ),
    "canvas_rankings": (
        "query_source",
        "gallery_source",
        "query_variant",
        "query_id",
        "candidate_id",
        "distance",
        "rank",
    ),
    "examples": (
        "size",
        "query_source",
        "gallery_source",
        "slice",
        "query_variant",
        "query_id",
        "metric",
        "value",
        "rank",
        "candidate_id",
        "distance",
    ),
    "timing_samples": (
        "query_id",
        "query_source",
        "gallery_source",
        "encoding_seconds",
        "search_seconds",
        "end_to_end_seconds",
    ),
    "timing_summary": (
        "query_source",
        "gallery_source",
        "metric",
        "percentile",
        "value_seconds",
        "timed_queries",
    ),
    "gallery_comparison": (
        "query_source",
        "gallery_policy",
        "metric",
        "k",
        "aggregation",
        "value",
    ),
    "gallery_rankings": (
        "query_source",
        "gallery_policy",
        "query_id",
        "candidate_id",
        "distance",
        "rank",
    ),
}
LEARNED_ARTIFACT_SCHEMAS = {
    name: (*_PROVENANCE_COLUMNS, *payload) for name, payload in _ARTIFACT_PAYLOAD_COLUMNS.items()
}
_SORT_COLUMNS = (
    "query_source",
    "gallery_source",
    "gallery_policy",
    "protocol",
    "query_variant",
    "slice",
    "metric",
    "k",
    "aggregation",
    "query_id",
    "rank",
    "candidate_id",
)

__all__ = (
    "LEARNED_ARTIFACT_NAMES",
    "LEARNED_ARTIFACT_SCHEMAS",
    "LEARNED_EVIDENCE_SCHEMA_VERSION",
    "GALLERY_POLICY_TIMING_ROUTE",
    "TEACHER_SLICE_CAVEAT",
    "THREAD_VARIABLES",
    "GallerySourceEvaluation",
    "LazyCPUQueryExtractor",
    "LearnedAnalysis",
    "LearnedQualityEvaluation",
    "LearnedEvidenceResult",
    "LearnedProvenance",
    "LearnedSourceArtifacts",
    "LearnedTimingEncoder",
    "StabilityEvidenceResult",
    "attach_teacher_slice_caveat",
    "assemble_learned_examples",
    "assemble_canvas_rankings",
    "assemble_gallery_rankings",
    "assemble_learned_rankings",
    "benchmark_learned_directions",
    "build_learned_cost_record",
    "build_learned_canvas_indexes",
    "build_learned_evidence",
    "build_stability_evidence",
    "complete_learned_evidence",
    "reconstruct_training_result",
    "encode_development_cache",
    "ensure_learned_feature_index",
    "evaluate_gallery_sources",
    "evaluate_learned_analysis",
    "evaluate_learned_quality",
    "make_milestone_scorer",
    "measure_learned_index_build",
    "rank_two_view_gallery",
    "record_evidence_failure",
    "summarize_learned_scores",
    "validate_learned_manifest",
    "validate_stability_evidence_artifact",
    "validate_gallery_policy_timing_artifact",
    "write_gallery_policy_timing_artifact",
    "write_learned_artifacts",
)


@dataclass(frozen=True)
class LearnedProvenance:
    """Exact model, split, source, normalization, and feature-cache identity."""

    schema_version: str
    run_id: str
    run_kind: str
    method: str
    fold: int
    checkpoint_sha256: str
    config_hash: str
    split_fingerprint: str
    source: str
    image_cache_manifest_sha256: str
    source_fingerprint: str
    statistics_sha256: str
    feature_cache_identity_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != LEARNED_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("learned provenance schema version is invalid")
        if self.source not in {"teacher", "v1"}:
            raise ValueError("learned provenance source must be teacher or v1")
        _validated_fold(self.fold)
        for label, value in (
            ("checkpoint SHA-256", self.checkpoint_sha256),
            ("config hash", self.config_hash),
            ("split fingerprint", self.split_fingerprint),
            ("image-cache manifest SHA-256", self.image_cache_manifest_sha256),
            ("source fingerprint", self.source_fingerprint),
            ("statistics SHA-256", self.statistics_sha256),
            ("feature-cache identity SHA-256", self.feature_cache_identity_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"learned provenance {label} is malformed")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.run_id, self.run_kind, self.method)
        ):
            raise ValueError("learned provenance run and method identity must not be blank")

    @property
    def model_identity(self) -> tuple[object, ...]:
        """Identity shared by teacher and V1 features from one selected model."""

        return (
            self.schema_version,
            self.run_id,
            self.run_kind,
            self.method,
            self.fold,
            self.checkpoint_sha256,
            self.config_hash,
            self.split_fingerprint,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "method": self.method,
            "fold": self.fold,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_hash": self.config_hash,
            "split_fingerprint": self.split_fingerprint,
            "source": self.source,
            "image_cache_manifest_sha256": self.image_cache_manifest_sha256,
            "source_fingerprint": self.source_fingerprint,
            "statistics_sha256": self.statistics_sha256,
            "feature_cache_identity_sha256": self.feature_cache_identity_sha256,
        }


@dataclass(frozen=True)
class LearnedQualityEvaluation:
    """Complete four-direction Protocol A/B learned quality evidence."""

    summary: pd.DataFrame
    query_metrics: pd.DataFrame
    pair_evaluations: dict[Direction, PairEvaluation]
    selected_metrics: dict[str, float]
    provenance: dict[str, LearnedProvenance] = field(default_factory=dict)


@dataclass(frozen=True)
class GallerySourceEvaluation:
    """Teacher-only, V1-only, and product-collapsed two-view evidence."""

    comparison: pd.DataFrame
    evaluations: dict[tuple[str, str], PairEvaluation]
    provenance: dict[str, LearnedProvenance] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnedAnalysis:
    """Frozen failure membership, slices, canvas stress, and example IDs."""

    membership: pd.DataFrame
    failure_slices: pd.DataFrame
    canvas_summary: pd.DataFrame
    canvas_per_query: pd.DataFrame
    example_ids: dict[str, int]
    canvas_rankings: dict[str, pd.DataFrame] = field(default_factory=dict)
    caveat: str = TEACHER_SLICE_CAVEAT
    provenance: dict[str, LearnedProvenance] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnedEvidenceResult:
    """One fully built, reopened, and registry-linked learned evidence package."""

    manifest_path: Path
    registry_row: dict[str, str]
    quality: LearnedQualityEvaluation
    analysis: LearnedAnalysis
    gallery: GallerySourceEvaluation


@dataclass(frozen=True)
class StabilityEvidenceResult:
    """One lightweight stability package and completed registry row."""

    manifest_path: Path
    registry_row: dict[str, str]
    development_winner_score: float
    total_query_count: int
    scorable_query_count: int
    primary_coverage: float


@dataclass(frozen=True)
class LearnedSourceArtifacts:
    """Paths reopened to prove one source's cache/statistics/feature lineage."""

    image_cache_manifest: Path
    statistics: Path
    feature_cache_manifest: Path


@dataclass(frozen=True)
class ReopenedSourceArtifactChain:
    """Canonical source data reopened from durable artifact paths."""

    cache: DevelopmentImageCache
    statistics: dict[str, object]
    index: FeatureIndex
    feature_manifest: dict[str, object]
    provenance: LearnedProvenance


@dataclass(frozen=True)
class LearnedTimingEncoder:
    """The actual checkpoint extractor bound to its feature-cache manifest."""

    extractor: LazyCPUQueryExtractor
    feature_manifest: Mapping[str, object]
    provenance: LearnedProvenance = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.extractor, LazyCPUQueryExtractor):
            raise TypeError("timing encoder requires the actual lazy CPU query extractor")
        manifest = dict(self.feature_manifest)
        try:
            provenance = _provenance_from_identity(manifest)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("timing feature manifest has malformed provenance") from error
        expected = {
            "source": self.extractor.source,
            "run_id": self.extractor.session.run_id,
            "run_kind": self.extractor.session.run_kind,
            "method": self.extractor.session.expected_registry_identity.method,
            "fold": self.extractor.session.validation_fold,
            "checkpoint_sha256": self.extractor.checkpoint.sha256,
            "config_hash": self.extractor.session.config_hash,
            "split_fingerprint": self.extractor.session.split_fingerprint,
            "source_statistics_sha256": _sha256_bytes(
                _canonical_json(dict(self.extractor.statistics))
            ),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("timing extractor does not match its feature-cache provenance")
        object.__setattr__(self, "feature_manifest", manifest)
        object.__setattr__(self, "provenance", provenance)

    @property
    def source(self) -> str:
        return self.provenance.source

    def __call__(self, row: pd.Series) -> np.ndarray:
        return self.extractor(row)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
    ).encode("utf-8")


@contextmanager
def _identity_lock(parent: Path, identity: str):
    """Serialize publication for one deterministic identity."""

    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{identity}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace_directory(temporary: Path, destination: Path) -> None:
    """Publish one complete directory with a Linux atomic exchange on rerun."""

    if not destination.exists():
        os.replace(temporary, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic directory exchange requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(temporary),
        -100,
        os.fsencode(destination),
        2,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    shutil.rmtree(temporary)


@contextmanager
def _single_threaded_cpu():
    """Set and prove one live native/PyTorch CPU thread for measurement."""

    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        pools = [
            {
                "user_api": str(pool.get("user_api")),
                "internal_api": str(pool.get("internal_api")),
                "num_threads": int(pool.get("num_threads", 0)),
                "prefix": str(pool.get("prefix")),
            }
            for pool in threadpool_info()
            if pool.get("user_api") in {"blas", "openmp"}
        ]
        if (
            torch.get_num_threads() != 1
            or any(os.environ.get(variable) != "1" for variable in THREAD_VARIABLES)
            or not pools
            or any(pool["num_threads"] != 1 for pool in pools)
        ):
            raise ValueError("timing requires one live CPU thread in PyTorch, BLAS, and OpenMP")
        yield pools


def _validated_fold(fold: object) -> int:
    if isinstance(fold, bool) or not isinstance(fold, Integral) or int(fold) not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    return int(fold)


def _validate_checkpoint_identity(
    checkpoint: CheckpointRecord,
    session: TrainingSessionConfig,
) -> None:
    if not isinstance(checkpoint, CheckpointRecord):
        raise ValueError("checkpoint must be a CheckpointRecord")
    expected = {
        "run ID": (checkpoint.run_id, session.run_id),
        "run kind": (checkpoint.run_kind, session.run_kind),
        "config hash": (checkpoint.config_hash, session.config_hash),
        "split fingerprint": (
            checkpoint.split_fingerprint,
            session.split_fingerprint,
        ),
        "weight origin": (
            checkpoint.weight_origin,
            session.model_metadata.weight_origin,
        ),
        "parent run ID": (checkpoint.parent_run_id, session.parent_run_id),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"checkpoint {label} does not match the training session")
    if checkpoint.epoch not in session.hyperparameters.checkpoint_epochs:
        raise ValueError("checkpoint epoch is not an approved milestone")


def _cache_contract(cache: DevelopmentImageCache) -> PreprocessingContract:
    manifest_contract = cache.manifest.get("contract")
    if manifest_contract != _CONTRACT.to_dict():
        raise ValueError("image cache must use the frozen 240x320 contract")
    return _CONTRACT


def _validate_source_statistics(
    cache: DevelopmentImageCache,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(cache, DevelopmentImageCache):
        raise ValueError("cache must be a DevelopmentImageCache")
    if cache.manifest.get("scope") != "development":
        raise ValueError("learned evidence accepts development image caches only")
    source = cache.manifest.get("source")
    if source not in {"teacher", "v1"}:
        raise ValueError("image cache source must be teacher or v1")
    _cache_contract(cache)
    if statistics.get("source") != source:
        raise ValueError("RGB statistics source does not match the image cache source")
    if statistics.get("source_fingerprint") != cache.manifest.get("source_fingerprint"):
        raise ValueError("RGB statistics source fingerprint does not match the image cache")
    if statistics.get("contract") != cache.manifest.get("contract"):
        raise ValueError("RGB statistics contract does not match the image cache")
    if statistics.get("validation_fold") != session.validation_fold:
        raise ValueError("RGB statistics validation fold does not match the session")
    if statistics.get("split_fingerprint") != session.split_fingerprint:
        raise ValueError("RGB statistics split fingerprint does not match the session")
    mean = np.asarray(statistics.get("mean"), dtype=np.float32)
    std = np.asarray(statistics.get("std"), dtype=np.float32)
    if (
        mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError(
            "RGB mean and standard deviation must contain three finite values "
            "with positive standard deviation"
        )
    if (
        cache.ids.dtype != np.int64
        or cache.images.dtype != np.uint8
        or cache.content_bounds.dtype != np.int32
        or cache.ids.shape != (len(cache.images),)
        or cache.content_bounds.shape != (len(cache.images), 4)
        or list(cache.images.shape) != cache.manifest.get("array_shape")
        or len(np.unique(cache.ids)) != len(cache.ids)
        or not np.array_equal(cache.ids, np.sort(cache.ids))
    ):
        raise ValueError("image cache arrays do not match their deterministic manifest")
    return mean, std


def _normalized_cache_batch(
    images: np.ndarray,
    bounds: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    values = (images.astype(np.float32) / np.float32(255.0) - mean) / std
    for row, raw_bounds in enumerate(bounds):
        top, left, bottom, right = (int(value) for value in raw_bounds)
        if not (0 <= top < bottom <= images.shape[1] and 0 <= left < right <= images.shape[2]):
            raise ValueError("image cache content bounds are invalid")
        mask = np.zeros(images.shape[1:3], dtype=bool)
        mask[top:bottom, left:right] = True
        values[row, ~mask] = 0.0
    return torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2)))


def _validated_embeddings(values: torch.Tensor, rows: int) -> np.ndarray:
    embeddings = values.detach().to(device="cpu", dtype=torch.float32).numpy()
    if embeddings.shape != (rows, EMBEDDING_DIM):
        raise ValueError(f"learned embeddings must have shape ({rows}, {EMBEDDING_DIM})")
    if not np.isfinite(embeddings).all():
        raise ValueError("learned embeddings must be finite")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=EMBEDDING_NORM_ATOL, rtol=0.0):
        raise ValueError("learned embeddings must have unit norm")
    return embeddings.astype(np.float32, copy=False)


def encode_development_cache(
    cache: DevelopmentImageCache,
    *,
    model: nn.Module,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> FeatureIndex:
    """Encode one validated development cache with exact saved source statistics."""

    _validate_checkpoint_identity(checkpoint, session)
    mean, std = _validate_source_statistics(cache, statistics, session)
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or int(batch_size) <= 0:
        raise ValueError("batch size must be a positive integer")
    if not hasattr(model, "encode"):
        raise ValueError("learned model must provide encode")
    parsed_device = torch.device(device)
    model.to(parsed_device)
    model.eval()
    started = time.perf_counter()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(cache.ids), int(batch_size)):
            stop = min(start + int(batch_size), len(cache.ids))
            batch = _normalized_cache_batch(
                np.asarray(cache.images[start:stop]),
                np.asarray(cache.content_bounds[start:stop]),
                mean,
                std,
            ).to(parsed_device)
            encoded = model.encode(batch)
            if not isinstance(encoded, torch.Tensor):
                raise ValueError("model encode must return a tensor")
            outputs.append(_validated_embeddings(encoded, stop - start))
    features = np.concatenate(outputs) if outputs else np.empty((0, EMBEDDING_DIM), np.float32)
    if len(features) == 0:
        raise ValueError("development image cache must not be empty")
    return FeatureIndex(
        source=str(cache.manifest["source"]),
        contract=_CONTRACT,
        ids=np.asarray(cache.ids, dtype=np.int64).copy(),
        features=features,
        transform_seconds=time.perf_counter() - started,
        source_bytes=int(cache.images.nbytes),
        method=session.expected_registry_identity.method,
        fold=int(session.validation_fold),
        checkpoint_fingerprint=checkpoint.sha256,
        config_fingerprint=session.config_hash,
        provenance=_build_learned_provenance(cache, statistics, session, checkpoint),
    )


def _feature_cache_identity(
    cache: DevelopmentImageCache,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> dict[str, object]:
    identity = {
        "schema_version": LEARNED_EVIDENCE_SCHEMA_VERSION,
        "scope": "development",
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "method": session.expected_registry_identity.method,
        "checkpoint_sha256": checkpoint.sha256,
        "config_hash": session.config_hash,
        "split_fingerprint": session.split_fingerprint,
        "source": cache.manifest["source"],
        "image_cache_manifest_sha256": _sha256_bytes(_canonical_json(cache.manifest)),
        "source_fingerprint": cache.manifest["source_fingerprint"],
        "source_statistics_sha256": _sha256_bytes(_canonical_json(dict(statistics))),
        "fold": int(session.validation_fold),
        "contract": _CONTRACT.to_dict(),
        "rows": len(cache.ids),
        "dimension": EMBEDDING_DIM,
    }
    identity["feature_cache_identity_sha256"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _provenance_from_identity(identity: Mapping[str, object]) -> LearnedProvenance:
    return LearnedProvenance(
        schema_version=str(identity["schema_version"]),
        run_id=str(identity["run_id"]),
        run_kind=str(identity["run_kind"]),
        method=str(identity["method"]),
        fold=int(identity["fold"]),
        checkpoint_sha256=str(identity["checkpoint_sha256"]),
        config_hash=str(identity["config_hash"]),
        split_fingerprint=str(identity["split_fingerprint"]),
        source=str(identity["source"]),
        image_cache_manifest_sha256=str(identity["image_cache_manifest_sha256"]),
        source_fingerprint=str(identity["source_fingerprint"]),
        statistics_sha256=str(identity["source_statistics_sha256"]),
        feature_cache_identity_sha256=str(identity["feature_cache_identity_sha256"]),
    )


def _build_learned_provenance(
    cache: DevelopmentImageCache,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> LearnedProvenance:
    return _provenance_from_identity(
        _feature_cache_identity(cache, statistics, session, checkpoint)
    )


def _validate_index_provenance(
    index: FeatureIndex,
    *,
    expected_source: str | None = None,
) -> LearnedProvenance:
    provenance = index.provenance
    if not isinstance(provenance, LearnedProvenance):
        raise ValueError("learned feature index requires exact learned provenance")
    if expected_source is not None and provenance.source != expected_source:
        raise ValueError("learned feature index provenance source does not match")
    expected = {
        "source": provenance.source,
        "method": provenance.method,
        "fold": provenance.fold,
        "checkpoint_fingerprint": provenance.checkpoint_sha256,
        "config_fingerprint": provenance.config_hash,
    }
    if any(getattr(index, field) != value for field, value in expected.items()):
        raise ValueError("learned feature index fields disagree with exact provenance")
    return provenance


def _validate_provenance_set(
    indexes: Mapping[str, FeatureIndex],
) -> dict[str, LearnedProvenance]:
    if set(indexes) != {"teacher", "v1"}:
        raise ValueError("learned indexes require teacher and v1 sources")
    provenance = {
        source: _validate_index_provenance(indexes[source], expected_source=source)
        for source in ("teacher", "v1")
    }
    if provenance["teacher"].model_identity != provenance["v1"].model_identity:
        raise ValueError("learned indexes must share one exact model provenance")
    return provenance


def _load_learned_cache(
    directory: Path,
    expected: Mapping[str, object],
) -> CachedFeatureIndex:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("learned feature cache identity does not match")
    ids_path = directory / "ids.npy"
    features_path = directory / "features.npy"
    if _sha256_file(ids_path) != manifest.get("ids_sha256") or _sha256_file(
        features_path
    ) != manifest.get("features_sha256"):
        raise ValueError("learned feature cache files fail their manifest hashes")
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if (
        ids.dtype != np.int64
        or ids.shape != (int(expected["rows"]),)
        or features.dtype != np.float32
        or features.shape != (int(expected["rows"]), EMBEDDING_DIM)
        or not np.array_equal(ids, np.sort(ids))
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(features).all()
        or not np.allclose(
            np.linalg.norm(features, axis=1),
            1.0,
            atol=EMBEDDING_NORM_ATOL,
            rtol=0.0,
        )
    ):
        raise ValueError("learned feature cache arrays are corrupt")
    index = FeatureIndex(
        source=str(expected["source"]),
        contract=_CONTRACT,
        ids=ids,
        features=features,
        transform_seconds=float(manifest["transform_seconds"]),
        source_bytes=int(manifest["source_bytes"]),
        method=str(manifest["feature_method"]),
        fold=int(expected["fold"]),
        checkpoint_fingerprint=str(expected["checkpoint_sha256"]),
        config_fingerprint=str(expected["config_hash"]),
        provenance=_provenance_from_identity(expected),
    )
    return CachedFeatureIndex(cache_dir=directory, index=index, manifest=manifest)


def _reopen_source_artifacts(
    artifacts: Mapping[str, LearnedSourceArtifacts],
    *,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    expected_development_ids: Sequence[int] | None = None,
) -> dict[str, ReopenedSourceArtifactChain]:
    """Reopen and cross-check every manifest in the exact provenance chain."""

    if set(artifacts) != {"teacher", "v1"}:
        raise ValueError("source artifacts require teacher and v1 paths")
    _validate_checkpoint_identity(checkpoint, session)
    reopened_sources: dict[str, ReopenedSourceArtifactChain] = {}
    expected_ids = (
        sorted(int(value) for value in expected_development_ids)
        if expected_development_ids is not None
        else None
    )
    for source in ("teacher", "v1"):
        paths = artifacts[source]
        if not isinstance(paths, LearnedSourceArtifacts):
            raise ValueError("source artifact paths are malformed")
        try:
            image_manifest = json.loads(paths.image_cache_manifest.read_text(encoding="utf-8"))
            statistics = json.loads(paths.statistics.read_text(encoding="utf-8"))
            feature_manifest = json.loads(paths.feature_cache_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("source provenance manifests cannot be reopened") from error
        if not all(
            isinstance(value, dict)
            for value in (
                image_manifest,
                statistics,
                feature_manifest,
            )
        ):
            raise ValueError("source provenance manifests must be JSON objects")
        try:
            image_cache = load_development_image_cache(paths.image_cache_manifest.parent)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError("image cache manifest and arrays cannot be reopened") from error
        if image_cache.manifest != image_manifest:
            raise ValueError("image cache manifest does not match the reopened cache")
        _validate_source_statistics(image_cache, statistics, session)
        expected_feature_fields = {
            "schema_version",
            "scope",
            "run_id",
            "run_kind",
            "method",
            "checkpoint_sha256",
            "config_hash",
            "split_fingerprint",
            "source",
            "image_cache_manifest_sha256",
            "source_fingerprint",
            "source_statistics_sha256",
            "fold",
            "contract",
            "rows",
            "dimension",
            "feature_cache_identity_sha256",
            "feature_method",
            "transform_seconds",
            "source_bytes",
            "ids_sha256",
            "features_sha256",
        }
        if set(feature_manifest) != expected_feature_fields:
            raise ValueError("feature-cache manifest schema is malformed")
        identity = {
            key: value
            for key, value in feature_manifest.items()
            if key
            not in {
                "feature_method",
                "transform_seconds",
                "source_bytes",
                "ids_sha256",
                "features_sha256",
            }
        }
        identity_without_digest = dict(identity)
        stored_identity_digest = identity_without_digest.pop(
            "feature_cache_identity_sha256",
            None,
        )
        if stored_identity_digest != _sha256_bytes(_canonical_json(identity_without_digest)):
            raise ValueError("feature-cache identity digest does not match its manifest fields")
        try:
            reopened = _load_learned_cache(paths.feature_cache_manifest.parent, identity)
            source_provenance = _provenance_from_identity(identity)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError("feature-cache manifest or arrays cannot be reopened") from error
        if expected_ids is not None and reopened.index.ids.astype(int).tolist() != expected_ids:
            raise ValueError("feature-cache IDs do not match canonical development IDs")
        image_hash = _sha256_bytes(_canonical_json(image_manifest))
        statistics_hash = _sha256_bytes(_canonical_json(statistics))
        if (
            image_cache.manifest.get("scope") != "development"
            or image_manifest.get("source") != source
            or statistics.get("source") != source
            or statistics.get("source_fingerprint") != image_manifest.get("source_fingerprint")
            or source_provenance.source != source
            or source_provenance.image_cache_manifest_sha256 != image_hash
            or source_provenance.statistics_sha256 != statistics_hash
        ):
            raise ValueError(
                "cache/statistics provenance disagrees with the feature-cache manifest"
            )
        expected_model = (
            LEARNED_EVIDENCE_SCHEMA_VERSION,
            session.run_id,
            session.run_kind,
            session.expected_registry_identity.method,
            session.validation_fold,
            checkpoint.sha256,
            session.config_hash,
            session.split_fingerprint,
        )
        if source_provenance.model_identity != expected_model:
            raise ValueError("reopened source provenance does not match the evidence session")
        reopened_sources[source] = ReopenedSourceArtifactChain(
            cache=image_cache,
            statistics=dict(statistics),
            index=reopened.index,
            feature_manifest=dict(feature_manifest),
            provenance=source_provenance,
        )
    if (
        reopened_sources["teacher"].provenance.model_identity
        != reopened_sources["v1"].provenance.model_identity
    ):
        raise ValueError("reopened sources do not share one exact model identity")
    return reopened_sources


def _load_evidence_model(
    checkpoint: CheckpointRecord,
    session: TrainingSessionConfig,
    device: torch.device | str,
) -> nn.Module:
    parsed_device = torch.device(device)
    loaded = load_checkpoint(
        checkpoint.path,
        expected_sha256=checkpoint.sha256,
        expected_config_hash=session.config_hash,
        expected_split_fingerprint=session.split_fingerprint,
        expected_weight_origin=session.model_metadata.weight_origin,
        expected_parent_run_id=session.parent_run_id,
        expected_run_id=session.run_id,
        expected_run_kind=session.run_kind,
        map_location=parsed_device,
    )
    loaded.model.to(parsed_device)
    loaded.model.eval()
    return loaded.model


def ensure_learned_feature_index(
    *,
    cache: DevelopmentImageCache,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    cache_root: str | Path,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> CachedFeatureIndex:
    """Open or atomically build an exact checkpoint-keyed learned feature index."""

    _validate_checkpoint_identity(checkpoint, session)
    _validate_source_statistics(cache, statistics, session)
    expected = _feature_cache_identity(cache, statistics, session, checkpoint)
    identity_hash = _sha256_bytes(_canonical_json(expected))[:20]
    directory = (
        Path(cache_root)
        / session.run_kind
        / session.run_id
        / checkpoint.sha256[:16]
        / session.config_hash[:16]
        / f"fold-{session.validation_fold}"
        / str(cache.manifest["source"])
        / identity_hash
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with _identity_lock(directory.parent, directory.name):
        if directory.is_dir():
            try:
                return _load_learned_cache(directory, expected)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass

        model = _load_evidence_model(checkpoint, session, device)
        index = encode_development_cache(
            cache,
            model=model,
            statistics=statistics,
            session=session,
            checkpoint=checkpoint,
            batch_size=batch_size,
            device=device,
        )
        temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
        try:
            np.save(temporary / "ids.npy", index.ids, allow_pickle=False)
            np.save(temporary / "features.npy", index.features, allow_pickle=False)
            manifest = {
                **expected,
                "feature_method": index.method,
                "transform_seconds": index.transform_seconds,
                "source_bytes": index.source_bytes,
                "ids_sha256": _sha256_file(temporary / "ids.npy"),
                "features_sha256": _sha256_file(temporary / "features.npy"),
            }
            (temporary / "manifest.json").write_bytes(_canonical_json(manifest))
            _replace_directory(temporary, directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return _load_learned_cache(directory, expected)


def _assert_query_coverage(
    frame: pd.DataFrame,
    expected_query_ids: Sequence[int],
    *,
    group_columns: Sequence[str],
    label: str,
) -> None:
    expected = sorted(int(value) for value in expected_query_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected query IDs must be unique and non-empty")
    required = {"query_id", *group_columns}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    for _, rows in frame.groupby(list(group_columns), sort=False, dropna=False):
        observed = pd.to_numeric(rows["query_id"], errors="coerce")
        if (
            observed.isna().any()
            or not observed.mod(1).eq(0).all()
            or sorted(observed.astype(int).tolist()) != expected
        ):
            raise ValueError(f"{label} must contain every expected query exactly once")


def _selected_ndcg(summary: pd.DataFrame, direction: Direction) -> float:
    required = {
        "protocol",
        "metric",
        "k",
        "aggregation",
        "query_source",
        "gallery_source",
        "value",
    }
    if missing := required.difference(summary.columns):
        raise ValueError(f"quality summary is missing columns: {sorted(missing)}")
    selected = summary.loc[
        summary["protocol"].eq("primary")
        & summary["metric"].eq("ndcg")
        & pd.to_numeric(summary["k"], errors="coerce").eq(10)
        & summary["aggregation"].eq("query_mean")
        & summary["query_source"].eq(direction[0])
        & summary["gallery_source"].eq(direction[1])
    ]
    if len(selected) != 1:
        raise ValueError(f"quality summary needs one nDCG@10 row for {direction}")
    value = float(selected["value"].iloc[0])
    if not math.isfinite(value):
        raise ValueError("quality summary selected nDCG values must be finite")
    return value


def summarize_learned_scores(summary: pd.DataFrame) -> dict[str, float]:
    """Select frozen nDCG rows and calculate the three learned headline values."""

    teacher = _selected_ndcg(summary, ("teacher", "teacher"))
    v1 = _selected_ndcg(summary, ("v1", "v1"))
    teacher_v1 = _selected_ndcg(summary, ("teacher", "v1"))
    v1_teacher = _selected_ndcg(summary, ("v1", "teacher"))
    same_source = (teacher + v1) / 2.0
    cross_source = (teacher_v1 + v1_teacher) / 2.0
    ratio = cross_source / same_source if same_source != 0.0 else 0.0
    if ratio < 0 or not math.isfinite(ratio):
        raise ValueError("source robustness ratio must be finite and non-negative")
    return {
        "development_winner_score": float(same_source),
        "cross_source_score": float(cross_source),
        "source_robustness_ratio": float(ratio),
    }


def evaluate_learned_quality(
    splits: pd.DataFrame,
    indexes: Mapping[str, FeatureIndex],
    *,
    fold: int,
    k_values: tuple[int, ...] = (5, 10, 20),
    family_k: int = 10,
    chunk_size: int = 256,
) -> LearnedQualityEvaluation:
    """Reuse the frozen source-pair evaluator for one learned validation fold."""

    selected_fold = _validated_fold(fold)
    if set(indexes) != {"teacher", "v1"}:
        raise ValueError("learned quality requires exactly teacher and v1 indexes")
    provenance = _validate_provenance_set(indexes)
    for source in ("teacher", "v1"):
        index = indexes[source]
        if index.source != source:
            raise ValueError("learned feature index source does not match its key")
        if index.contract != _CONTRACT:
            raise ValueError("learned quality requires the frozen 240x320 contract")
        if index.fold != selected_fold:
            raise ValueError("learned feature index fold does not match evaluation fold")
        _validate_unit_matrix(index.features, label=f"{source} learned features")
    primary, family = build_development_views(splits, validation_fold=selected_fold)
    pairs = {
        direction: evaluate_source_pair(
            indexes[direction[0]],
            indexes[direction[1]],
            primary_views=primary,
            family_views=family,
            fold=selected_fold,
            k_values=k_values,
            family_k=family_k,
            chunk_size=chunk_size,
        )
        for direction in source_directions()
    }
    summary = (
        pd.concat(
            [pairs[direction].summary for direction in source_directions()],
            ignore_index=True,
        )
        .sort_values(
            ["query_source", "gallery_source", "protocol", "metric", "k", "aggregation"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    query_metrics = (
        build_query_metrics(pairs)
        .sort_values(
            ["query_source", "gallery_source", "protocol", "query_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    _assert_query_coverage(
        query_metrics,
        primary.queries["id"].astype(int).tolist(),
        group_columns=("query_source", "gallery_source", "protocol"),
        label="learned query metrics",
    )
    return LearnedQualityEvaluation(
        summary=summary,
        query_metrics=query_metrics,
        pair_evaluations=pairs,
        selected_metrics=summarize_learned_scores(summary),
        provenance=provenance,
    )


@dataclass(frozen=True)
class _MilestoneScorer:
    evaluate: Callable[[nn.Module], pd.DataFrame | LearnedQualityEvaluation]
    fold: int

    def __call__(self, model: nn.Module, epoch: int) -> float:
        del epoch
        result = self.evaluate(model)
        summary = result.summary if isinstance(result, LearnedQualityEvaluation) else result
        if "fold" not in summary:
            raise ValueError("milestone summary is missing its validation fold")
        observed_folds = pd.to_numeric(summary["fold"], errors="coerce")
        if observed_folds.isna().any() or set(observed_folds.astype(int)) != {self.fold}:
            raise ValueError("milestone summary fold does not match the selected validation fold")
        return summarize_learned_scores(summary)["development_winner_score"]


def make_milestone_scorer(
    evaluate: Callable[[nn.Module], pd.DataFrame | LearnedQualityEvaluation],
    *,
    fold: int,
) -> Callable[[nn.Module, int], float]:
    """Build an injected checkpoint scorer from frozen evaluator output."""

    if not callable(evaluate):
        raise ValueError("milestone evaluator must be callable")
    return _MilestoneScorer(evaluate, _validated_fold(fold))


def attach_teacher_slice_caveat(
    failure_slices: pd.DataFrame,
    canvas_summary: pd.DataFrame,
    canvas_per_query: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Persist the teacher-derived membership caveat on all robustness evidence."""

    frames = []
    for frame in (failure_slices, canvas_summary, canvas_per_query):
        labelled = frame.copy()
        labelled["caveat"] = TEACHER_SLICE_CAVEAT
        frames.append(labelled)
    return frames[0], frames[1], frames[2]


def evaluate_learned_analysis(
    quality: LearnedQualityEvaluation,
    *,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
    canvas_indexes: Mapping[str, FeatureIndex],
    gallery_index: FeatureIndex,
    fold: int,
) -> LearnedAnalysis:
    """Reuse frozen failure, canvas, and deterministic example selection."""

    selected_fold = _validated_fold(fold)
    if set(quality.provenance) != {"teacher", "v1"}:
        raise ValueError("analysis quality is missing exact learned provenance")
    expected_v1 = quality.provenance["v1"]
    observed = [
        _validate_index_provenance(index, expected_source="v1")
        for index in (*canvas_indexes.values(), gallery_index)
    ]
    if any(value != expected_v1 for value in observed):
        raise ValueError("analysis indexes must share quality learned provenance")
    for label, index in (
        *((orientation, index) for orientation, index in canvas_indexes.items()),
        ("gallery", gallery_index),
    ):
        _validate_unit_matrix(index.features, label=f"{label} learned features")
    membership = mark_failure_slices(build_query_support(primary_views, family_views))
    slices = summarize_failure_slices(quality.query_metrics, membership)
    canvas: CanvasStressEvaluation = evaluate_canvas_stress(
        quality.pair_evaluations[("v1", "v1")],
        canvas_indexes,
        gallery_index,
        primary_views,
        fold=selected_fold,
    )
    slices, canvas_summary, canvas_per_query = attach_teacher_slice_caveat(
        slices,
        canvas.summary,
        canvas.per_query,
    )
    examples = select_example_ids(quality.query_metrics, membership, canvas_per_query)
    return LearnedAnalysis(
        membership=membership.sort_values("query_id").reset_index(drop=True),
        failure_slices=slices,
        canvas_summary=canvas_summary,
        canvas_per_query=canvas_per_query,
        example_ids=examples,
        canvas_rankings=canvas.rankings,
        provenance=dict(quality.provenance),
    )


def build_learned_canvas_indexes(
    cache: DevelopmentImageCache,
    *,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    query_ids: Sequence[int],
    device: torch.device | str = "cpu",
) -> dict[str, FeatureIndex]:
    """Encode deterministic wide/tall validation canvases with saved source statistics."""

    _validate_checkpoint_identity(checkpoint, session)
    mean, std = _validate_source_statistics(cache, statistics, session)
    if cache.manifest.get("source") != "v1":
        raise ValueError("canvas evidence requires the V1 development cache")
    requested = sorted(int(value) for value in query_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("canvas query IDs must be unique and non-empty")
    positions = {int(product_id): position for position, product_id in enumerate(cache.ids)}
    if missing := sorted(set(requested).difference(positions)):
        raise ValueError(f"canvas cache is missing query IDs: {missing[:10]}")
    parsed_device = torch.device(device)
    model = _load_evidence_model(checkpoint, session, parsed_device)
    model.eval()
    indexes: dict[str, FeatureIndex] = {}
    for orientation in ("wide", "tall"):
        started = time.perf_counter()
        features: list[np.ndarray] = []
        with torch.inference_mode():
            for product_id in requested:
                position = positions[product_id]
                top, left, bottom, right = (int(value) for value in cache.content_bounds[position])
                content = np.asarray(cache.images[position])[top:bottom, left:right]
                canvas = build_odd_aspect_canvas(Image.fromarray(content), orientation)
                transformed = preprocess_image(canvas, _CONTRACT)
                normalized = normalize_for_model(transformed, mean=mean, std=std)
                batch = (
                    torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
                    .unsqueeze(0)
                    .to(parsed_device)
                )
                features.append(_validated_embeddings(model.encode(batch), 1)[0])
        indexes[orientation] = FeatureIndex(
            source="v1",
            contract=_CONTRACT,
            ids=np.asarray(requested, dtype=np.int64),
            features=np.stack(features).astype(np.float32, copy=False),
            transform_seconds=time.perf_counter() - started,
            source_bytes=int(cache.images.nbytes),
            method=session.expected_registry_identity.method,
            fold=session.validation_fold,
            checkpoint_fingerprint=checkpoint.sha256,
            config_fingerprint=session.config_hash,
            provenance=_build_learned_provenance(cache, statistics, session, checkpoint),
        )
    return indexes


def assemble_learned_rankings(quality: LearnedQualityEvaluation) -> pd.DataFrame:
    """Label and combine all four direction/Protocol A/B ranking frames."""

    records: list[pd.DataFrame] = []
    if set(quality.provenance) != {"teacher", "v1"}:
        raise ValueError("ranking assembly requires exact learned provenance")
    if set(quality.pair_evaluations) != set(source_directions()):
        raise ValueError("ranking assembly requires all four source directions")
    for query_source, gallery_source in source_directions():
        evaluation = quality.pair_evaluations[(query_source, gallery_source)]
        for protocol, rankings in (
            ("primary", evaluation.primary_rankings),
            ("family", evaluation.family_rankings),
        ):
            required = {"query_id", "candidate_id", "distance", "rank"}
            if set(rankings.columns) != required:
                raise ValueError("ranking input columns are malformed")
            records.append(
                rankings.assign(
                    query_source=query_source,
                    gallery_source=gallery_source,
                    protocol=protocol,
                ).loc[
                    :,
                    [
                        "query_source",
                        "gallery_source",
                        "protocol",
                        "query_id",
                        "candidate_id",
                        "distance",
                        "rank",
                    ],
                ]
            )
    return _ordered_frame(pd.concat(records, ignore_index=True))


def assemble_canvas_rankings(analysis: LearnedAnalysis) -> pd.DataFrame:
    """Persist the exact clean/wide/tall V1 ranking rows used by examples."""

    if set(analysis.canvas_rankings) != {"clean", "wide", "tall"}:
        raise ValueError("canvas ranking assembly requires clean, wide, and tall rankings")
    frames = []
    for variant in ("clean", "wide", "tall"):
        rankings = analysis.canvas_rankings[variant]
        required = {"query_id", "candidate_id", "distance", "rank"}
        if set(rankings.columns) != required:
            raise ValueError("canvas ranking input columns are malformed")
        frames.append(
            rankings.loc[rankings["rank"].le(10)]
            .assign(
                query_source="v1",
                gallery_source="v1",
                query_variant=variant,
            )
            .loc[
                :,
                [
                    "query_source",
                    "gallery_source",
                    "query_variant",
                    "query_id",
                    "candidate_id",
                    "distance",
                    "rank",
                ],
            ]
        )
    return _ordered_frame(pd.concat(frames, ignore_index=True))


def assemble_learned_examples(
    analysis: LearnedAnalysis,
    quality: LearnedQualityEvaluation,
) -> pd.DataFrame:
    """Turn deterministic example IDs into labelled V1-to-V1 Top-5 ranking rows."""
    if set(quality.provenance) != {"teacher", "v1"} or analysis.provenance != quality.provenance:
        raise ValueError("example assembly requires one exact learned provenance")

    clean = quality.pair_evaluations[("v1", "v1")]
    rows: list[dict[str, object]] = []
    family_slices = {"family_unavailable", "weak_family"}
    for slice_name, query_id in sorted(analysis.example_ids.items()):
        if slice_name == "canvas_failure":
            canvas_rows = analysis.canvas_per_query.loc[
                analysis.canvas_per_query["query_id"].eq(query_id)
                & analysis.canvas_per_query["query_variant"].isin(("wide", "tall"))
            ].sort_values(
                ["ndcg_change_from_clean", "query_variant"],
                kind="mergesort",
            )
            if canvas_rows.empty:
                raise ValueError("canvas example has no wide/tall query evidence")
            variant = str(canvas_rows.iloc[0]["query_variant"])
        else:
            variant = "clean"
        protocol = "family" if slice_name in family_slices else "primary"
        metric = "recall_at_10" if protocol == "family" else "ndcg_at_10"
        metric_rows = quality.query_metrics.loc[
            quality.query_metrics["query_source"].eq("v1")
            & quality.query_metrics["gallery_source"].eq("v1")
            & quality.query_metrics["protocol"].eq(protocol)
            & quality.query_metrics["query_id"].eq(query_id)
        ]
        if len(metric_rows) != 1:
            raise ValueError("selected example has no unique query metric")
        ranking = (
            analysis.canvas_rankings[variant]
            if slice_name == "canvas_failure"
            else (clean.family_rankings if protocol == "family" else clean.primary_rankings)
        )
        top = ranking.loc[ranking["query_id"].eq(query_id) & ranking["rank"].le(5)]
        if top.empty:
            raise ValueError("selected example has no ranking rows")
        metric_value = (
            analysis.canvas_per_query.loc[
                analysis.canvas_per_query["query_id"].eq(query_id)
                & analysis.canvas_per_query["query_variant"].eq(variant),
                "canvas_ndcg_at_10",
            ].iloc[0]
            if slice_name == "canvas_failure"
            else metric_rows.iloc[0][metric]
        )
        for result in top.itertuples(index=False):
            rows.append(
                {
                    "size": "240x320",
                    "query_source": "v1",
                    "gallery_source": "v1",
                    "slice": slice_name,
                    "query_variant": variant,
                    "query_id": int(query_id),
                    "metric": metric,
                    "value": float(metric_value),
                    "rank": int(result.rank),
                    "candidate_id": int(result.candidate_id),
                    "distance": float(result.distance),
                }
            )
    if not rows:
        raise ValueError("learned examples must not be empty")
    return _ordered_frame(pd.DataFrame.from_records(rows))


def _validate_unit_matrix(values: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.dtype != np.float32:
        raise ValueError(f"{label} must use float32 features")
    if (
        matrix.ndim != 2
        or matrix.shape[1] != EMBEDDING_DIM
        or not np.isfinite(matrix).all()
        or not np.allclose(
            np.linalg.norm(matrix, axis=1),
            1.0,
            atol=EMBEDDING_NORM_ATOL,
            rtol=0.0,
        )
    ):
        raise ValueError(f"{label} must be a finite unit-normalized 128-value feature matrix")
    return matrix


def rank_two_view_gallery(
    *,
    query_index: FeatureIndex,
    gallery_indexes: Mapping[str, FeatureIndex],
    views: RetrievalViews,
    protocol: Literal["primary", "family"],
    max_k: int,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Rank products after collapsing each teacher/V1 pair to minimum distance."""

    if set(gallery_indexes) != {"teacher", "v1"}:
        raise ValueError("two-view gallery requires teacher and v1 indexes")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or int(chunk_size) <= 0:
        raise ValueError("chunk size must be a positive integer")
    gallery_provenance = _validate_provenance_set(gallery_indexes)
    query_provenance = _validate_index_provenance(query_index)
    if query_provenance != gallery_provenance.get(query_provenance.source):
        raise ValueError("two-view query provenance does not match its gallery source")
    view_query_ids = pd.to_numeric(views.queries["id"], errors="coerce")
    if view_query_ids.isna().any() or not view_query_ids.mod(1).eq(0).all():
        raise ValueError("two-view query IDs must be integer-compatible")
    numeric_query_ids = view_query_ids.to_numpy(dtype=np.int64)
    query_positions = {
        int(product_id): position for position, product_id in enumerate(query_index.ids)
    }
    if missing := sorted(set(numeric_query_ids).difference(query_positions)):
        raise ValueError(f"two-view query index is missing IDs: {missing[:10]}")
    queries = _validate_unit_matrix(
        query_index.features[[query_positions[int(value)] for value in numeric_query_ids]],
        label="query features",
    )
    view_gallery_ids = pd.to_numeric(views.gallery["id"], errors="coerce")
    if view_gallery_ids.isna().any() or not view_gallery_ids.mod(1).eq(0).all():
        raise ValueError("gallery view IDs must be integer-compatible")
    ordered_ids = np.sort(view_gallery_ids.to_numpy(dtype=np.int64))
    if len(np.unique(ordered_ids)) != len(ordered_ids):
        raise ValueError("gallery view IDs must be unique")

    matrices: list[np.ndarray] = []
    for source in ("teacher", "v1"):
        index = gallery_indexes[source]
        if index.source != source or index.contract != _CONTRACT:
            raise ValueError("two-view gallery index source or contract is invalid")
        positions = {int(product_id): position for position, product_id in enumerate(index.ids)}
        if missing := sorted(set(ordered_ids).difference(positions)):
            raise ValueError(f"two-view gallery index is missing IDs: {missing[:10]}")
        matrices.append(
            _validate_unit_matrix(
                index.features[[positions[int(value)] for value in ordered_ids]],
                label=f"{source} gallery features",
            )
        )
    if any(matrix.shape[1] != queries.shape[1] for matrix in matrices):
        raise ValueError("query and gallery feature dimensions must match")

    records: list[dict[str, object]] = []
    for start in range(0, len(numeric_query_ids), int(chunk_size)):
        stop = min(start + int(chunk_size), len(numeric_query_ids))
        teacher_distance = np.clip(
            1.0 - queries[start:stop] @ matrices[0].T,
            0.0,
            2.0,
        )
        v1_distance = np.clip(
            1.0 - queries[start:stop] @ matrices[1].T,
            0.0,
            2.0,
        )
        distances = np.minimum(teacher_distance, v1_distance)
        for row, query_id in enumerate(numeric_query_ids[start:stop]):
            records.extend(
                {
                    "query_id": int(query_id),
                    "candidate_id": int(candidate_id),
                    "distance": float(distance),
                }
                for candidate_id, distance in zip(
                    ordered_ids,
                    distances[row],
                    strict=True,
                )
            )
    return prepare_rankings(
        pd.DataFrame.from_records(records),
        views,
        protocol=protocol,
        max_k=max_k,
    )


def _two_view_pair(
    query_index: FeatureIndex,
    gallery_indexes: Mapping[str, FeatureIndex],
    *,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
    fold: int,
    k_values: tuple[int, ...],
    family_k: int,
) -> PairEvaluation:
    primary_rankings = rank_two_view_gallery(
        query_index=query_index,
        gallery_indexes=gallery_indexes,
        views=primary_views,
        protocol="primary",
        max_k=max(k_values),
    )
    primary_per_query, primary_summary = evaluate_primary_rankings(
        primary_rankings,
        primary_views,
        k_values=k_values,
    )
    family_rankings = rank_two_view_gallery(
        query_index=query_index,
        gallery_indexes=gallery_indexes,
        views=family_views,
        protocol="family",
        max_k=family_k,
    )
    family_per_query, family_summary = evaluate_family_rankings(
        family_rankings,
        family_views,
        k=family_k,
    )
    summary_frames = []
    for protocol, frame in (
        ("primary", primary_summary),
        ("family", family_summary),
    ):
        labelled = frame.copy()
        for position, (column, value) in enumerate(
            {
                "method": query_index.method,
                "fold": fold,
                "checkpoint_fingerprint": query_index.checkpoint_fingerprint,
                "config_fingerprint": query_index.config_fingerprint,
                "size": "240x320",
                "width": 240,
                "height": 320,
                "query_source": query_index.source,
                "gallery_source": "two_view",
                "protocol": protocol,
                "scope": "development",
                "query_transform_seconds": query_index.transform_seconds,
                "gallery_transform_seconds": sum(
                    index.transform_seconds for index in gallery_indexes.values()
                ),
                "query_source_bytes": query_index.source_bytes,
                "gallery_source_bytes": sum(
                    index.source_bytes for index in gallery_indexes.values()
                ),
                "feature_bytes_per_image": EMBEDDING_DIM * 4,
                "uint8_tensor_bytes_per_image": _CONTRACT.pixel_count * 3,
                "float32_tensor_bytes_per_image": _CONTRACT.pixel_count * 3 * 4,
            }.items()
        ):
            labelled.insert(position, column, value)
        summary_frames.append(labelled)
    return PairEvaluation(
        summary=pd.concat(summary_frames, ignore_index=True),
        primary_rankings=primary_rankings,
        family_rankings=family_rankings,
        primary_per_query=primary_per_query,
        family_per_query=family_per_query,
        method=query_index.method,
        fold=fold,
        checkpoint_fingerprint=query_index.checkpoint_fingerprint,
        config_fingerprint=query_index.config_fingerprint,
    )


def evaluate_gallery_sources(
    splits: pd.DataFrame,
    indexes: Mapping[str, FeatureIndex],
    *,
    fold: int,
    k_values: tuple[int, ...] = (5, 10, 20),
    family_k: int = 10,
) -> GallerySourceEvaluation:
    """Evaluate teacher-only, V1-only, and product-collapsed two-view galleries."""

    selected_fold = _validated_fold(fold)
    if set(indexes) != {"teacher", "v1"}:
        raise ValueError("gallery comparison requires teacher and v1 indexes")
    provenance = _validate_provenance_set(indexes)
    for source in ("teacher", "v1"):
        index = indexes[source]
        if index.source != source or index.fold != selected_fold or index.contract != _CONTRACT:
            raise ValueError("gallery comparison index identity is invalid")
        _validate_unit_matrix(index.features, label=f"{source} gallery features")
    primary, family = build_development_views(splits, validation_fold=selected_fold)
    evaluations: dict[tuple[str, str], PairEvaluation] = {}
    records: list[dict[str, object]] = []
    for gallery_policy in ("teacher", "v1", "two_view"):
        query_scores: list[float] = []
        for query_source in ("teacher", "v1"):
            if gallery_policy == "two_view":
                result = _two_view_pair(
                    indexes[query_source],
                    indexes,
                    primary_views=primary,
                    family_views=family,
                    fold=selected_fold,
                    k_values=k_values,
                    family_k=family_k,
                )
            else:
                result = evaluate_source_pair(
                    indexes[query_source],
                    indexes[gallery_policy],
                    primary_views=primary,
                    family_views=family,
                    fold=selected_fold,
                    k_values=k_values,
                    family_k=family_k,
                )
            evaluations[(query_source, gallery_policy)] = result
            score = _selected_ndcg(result.summary, (query_source, gallery_policy))
            query_scores.append(score)
            records.append(
                {
                    "scope": "development",
                    "fold": selected_fold,
                    "query_source": query_source,
                    "gallery_policy": gallery_policy,
                    "metric": "ndcg",
                    "k": 10,
                    "aggregation": "query_mean",
                    "value": score,
                }
            )
        records.append(
            {
                "scope": "development",
                "fold": selected_fold,
                "query_source": "equal_teacher_v1_mean",
                "gallery_policy": gallery_policy,
                "metric": "gallery_quality",
                "k": 10,
                "aggregation": "equal_source_mean",
                "value": float(np.mean(query_scores)),
            }
        )
    comparison = (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["gallery_policy", "query_source", "metric"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return GallerySourceEvaluation(
        comparison=comparison,
        evaluations=evaluations,
        provenance=provenance,
    )


def assemble_gallery_rankings(gallery: GallerySourceEvaluation) -> pd.DataFrame:
    """Persist primary rankings behind every gallery-policy comparison row."""

    expected = {
        (query_source, policy)
        for policy in ("teacher", "v1", "two_view")
        for query_source in ("teacher", "v1")
    }
    if set(gallery.evaluations) != expected:
        raise ValueError("gallery ranking assembly requires all source and policy pairs")
    frames = []
    for query_source, policy in sorted(expected):
        rankings = gallery.evaluations[(query_source, policy)].primary_rankings
        frames.append(
            rankings.assign(
                query_source=query_source,
                gallery_policy=policy,
            ).loc[
                :,
                [
                    "query_source",
                    "gallery_policy",
                    "query_id",
                    "candidate_id",
                    "distance",
                    "rank",
                ],
            ]
        )
    return _ordered_frame(pd.concat(frames, ignore_index=True))


@dataclass
class LazyCPUQueryExtractor:
    """Spawn-picklable, lazy CPU batch-one checkpoint query extractor."""

    source: str
    checkpoint: CheckpointRecord
    session: TrainingSessionConfig
    statistics: Mapping[str, object]
    path_column: str
    root: str | Path = ROOT
    _model: nn.Module | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_checkpoint_identity(self.checkpoint, self.session)
        if self.source not in {"teacher", "v1"}:
            raise ValueError("query source must be teacher or v1")
        if not isinstance(self.path_column, str) or not self.path_column.strip():
            raise ValueError("query path column must not be blank")
        mean = np.asarray(self.statistics.get("mean"), dtype=np.float32)
        std = np.asarray(self.statistics.get("std"), dtype=np.float32)
        if (
            mean.shape != (3,)
            or std.shape != (3,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError("query source statistics are invalid")
        if self.statistics.get("validation_fold") != self.session.validation_fold:
            raise ValueError("query statistics fold does not match the session")
        if self.statistics.get("split_fingerprint") != self.session.split_fingerprint:
            raise ValueError("query statistics split does not match the session")
        if self.statistics.get("source") != self.source:
            raise ValueError("query statistics source does not match the extractor source")

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def _load_model(self) -> nn.Module:
        if self._model is None:
            self._model = _load_evidence_model(self.checkpoint, self.session, "cpu")
        return self._model

    def __call__(self, row: pd.Series) -> np.ndarray:
        if row.get("partition") != "development":
            raise ValueError("query timing accepts development rows only")
        if self.path_column not in row:
            raise ValueError(f"query row is missing {self.path_column}")
        path = Path(str(row[self.path_column]))
        resolved = path if path.is_absolute() else Path(self.root) / path
        transformed = load_preprocessed_image(resolved, _CONTRACT)
        normalized = normalize_for_model(
            transformed,
            mean=self.statistics["mean"],  # type: ignore[arg-type]
            std=self.statistics["std"],  # type: ignore[arg-type]
        )
        batch = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
        model = self._load_model()
        model.eval()
        with torch.inference_mode():
            values = model.encode(batch)
        return _validated_embeddings(values, 1)[0]


def benchmark_learned_directions(
    query_rows: pd.DataFrame,
    *,
    views: RetrievalViews,
    indexes: Mapping[str, FeatureIndex],
    encoders: Mapping[str, LearnedTimingEncoder],
    policy: TimingPolicy = TimingPolicy(),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    fold: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full timing route while live native and PyTorch pools are one thread."""

    with _single_threaded_cpu():
        return _benchmark_learned_directions(
            query_rows,
            views=views,
            indexes=indexes,
            encoders=encoders,
            policy=policy,
            clock_ns=clock_ns,
            fold=fold,
        )


def _benchmark_learned_directions(
    query_rows: pd.DataFrame,
    *,
    views: RetrievalViews,
    indexes: Mapping[str, FeatureIndex],
    encoders: Mapping[str, LearnedTimingEncoder],
    policy: TimingPolicy = TimingPolicy(),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    fold: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the full Protocol A Top-20 route for every query and direction."""

    selected_fold = _validated_fold(fold)
    if "partition" not in query_rows or set(query_rows["partition"].astype(str)) != {"development"}:
        raise ValueError("timing query rows must be development only")
    if set(indexes) != {"teacher", "v1"} or set(encoders) != {"teacher", "v1"}:
        raise ValueError("timing requires teacher and v1 indexes, encoders, and manifests")
    query_ids = pd.to_numeric(query_rows["id"], errors="coerce")
    canonical_ids = pd.to_numeric(views.queries["id"], errors="coerce")
    if (
        query_ids.isna().any()
        or canonical_ids.isna().any()
        or not query_ids.mod(1).eq(0).all()
        or not canonical_ids.mod(1).eq(0).all()
        or sorted(query_ids.astype(int).tolist()) != sorted(canonical_ids.astype(int).tolist())
    ):
        raise ValueError("timing query rows must equal the canonical fold query IDs")
    for source in ("teacher", "v1"):
        index = indexes[source]
        encoder = encoders[source]
        provenance = _validate_index_provenance(index, expected_source=source)
        if (
            not isinstance(encoder, LearnedTimingEncoder)
            or encoder.source != source
            or encoder.provenance != provenance
        ):
            raise ValueError("timing encoders must carry their exact source provenance")
        manifest = encoder.feature_manifest
        expected_manifest = {
            **provenance.to_dict(),
            "source_statistics_sha256": provenance.statistics_sha256,
            "feature_method": index.method,
        }
        expected_manifest.pop("statistics_sha256")
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            if manifest.get("source_statistics_sha256") != provenance.statistics_sha256:
                raise ValueError(
                    "timing encoder statistics provenance does not match its feature index"
                )
            raise ValueError("timing encoder and feature-cache manifest provenance do not match")
        _validate_unit_matrix(index.features, label=f"{source} timing index features")
    samples: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for query_source, gallery_source in source_directions():
        direction = benchmark_source_direction(
            query_rows,
            query_source=query_source,
            gallery_source=gallery_source,
            encode=encoders[query_source],
            search=build_protocol_a_search(
                views=views,
                gallery_index=indexes[gallery_source],
            ),
            policy=policy,
            clock_ns=clock_ns,
            fold=selected_fold,
        )
        samples.append(direction)
        summaries.append(summarize_timings(direction))
    combined_samples = (
        pd.concat(samples, ignore_index=True)
        .sort_values(
            ["query_source", "gallery_source", "query_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    combined_summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(
            ["query_source", "gallery_source", "metric", "percentile"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    _assert_query_coverage(
        combined_samples,
        query_rows["id"].astype(int).tolist(),
        group_columns=("query_source", "gallery_source"),
        label="learned timing samples",
    )
    return combined_samples, combined_summary


def _learned_index_build_worker(
    connection: Connection,
    cache: DevelopmentImageCache,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> None:
    try:
        with _single_threaded_cpu():
            model = _load_evidence_model(checkpoint, session, "cpu")
            started = time.perf_counter()
            index = encode_development_cache(
                cache,
                model=model,
                statistics=statistics,
                session=session,
                checkpoint=checkpoint,
                batch_size=1,
                device="cpu",
            )
        connection.send(
            (
                "ok",
                IndexCost(
                    source=index.source,  # type: ignore[arg-type]
                    contract=index.contract,
                    rows=len(index.ids),
                    dimension=index.features.shape[1],
                    payload_bytes=int(index.features.nbytes),
                    index_bytes=int(index.features.nbytes + index.ids.nbytes),
                    build_seconds=float(time.perf_counter() - started),
                    peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                ),
            )
        )
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def measure_learned_index_build(
    cache: DevelopmentImageCache,
    *,
    statistics: Mapping[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> IndexCost:
    """Measure batch-one learned extraction, storage, and RSS in a spawned child."""

    _validate_checkpoint_identity(checkpoint, session)
    _validate_source_statistics(cache, statistics, session)
    from multiprocessing import get_context

    context = get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_learned_index_build_worker,
        args=(sending, cache, dict(statistics), session, checkpoint),
    )
    process.start()
    sending.close()
    process.join()
    if process.exitcode != 0:
        receiving.close()
        raise RuntimeError(f"learned index build child exited with code {process.exitcode}")
    if not receiving.poll():
        receiving.close()
        raise RuntimeError("learned index build child returned no result")
    status, payload = receiving.recv()
    receiving.close()
    if status == "error":
        raise RuntimeError(f"learned index build child failed:\n{payload}")
    if not isinstance(payload, IndexCost):
        raise RuntimeError("learned index build child returned an invalid result")
    return payload


def build_learned_cost_record(
    timing_summary: pd.DataFrame,
    *,
    caches: Mapping[str, DevelopmentImageCache],
    statistics: Mapping[str, Mapping[str, object]],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    policy: TimingPolicy,
    fold: int,
    selected_gallery_policy: Literal["teacher", "v1", "two_view"],
) -> dict[str, object]:
    """Measure costs while proving the live CPU thread-pool policy."""

    with _single_threaded_cpu() as native_thread_pools:
        return _build_learned_cost_record(
            timing_summary,
            caches=caches,
            statistics=statistics,
            session=session,
            checkpoint=checkpoint,
            policy=policy,
            fold=fold,
            selected_gallery_policy=selected_gallery_policy,
            native_thread_pools=native_thread_pools,
        )


def _build_learned_cost_record(
    timing_summary: pd.DataFrame,
    *,
    caches: Mapping[str, DevelopmentImageCache],
    statistics: Mapping[str, Mapping[str, object]],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    policy: TimingPolicy,
    fold: int,
    selected_gallery_policy: Literal["teacher", "v1", "two_view"],
    native_thread_pools: list[dict[str, object]],
) -> dict[str, object]:
    """Measure and build selected-checkpoint cost evidence without caller-authored values."""

    selected_fold = _validated_fold(fold)
    if selected_fold != session.validation_fold:
        raise ValueError("cost fold does not match the training session")
    if set(caches) != {"teacher", "v1"} or set(statistics) != {"teacher", "v1"}:
        raise ValueError("cost measurement requires teacher and v1 caches and statistics")
    if selected_gallery_policy not in {"teacher", "v1", "two_view"}:
        raise ValueError("selected gallery policy is invalid")
    _validate_checkpoint_identity(checkpoint, session)
    model = _load_evidence_model(checkpoint, session, "cpu")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_bytes = checkpoint.path.stat().st_size
    index_costs = {
        source: measure_learned_index_build(
            caches[source],
            statistics=statistics[source],
            session=session,
            checkpoint=checkpoint,
        )
        for source in ("teacher", "v1")
    }
    record = build_cost_record(
        timing_summary,
        index_costs,  # type: ignore[arg-type]
        policy=policy,
    )
    record["hardware"]["native_thread_pools"] = native_thread_pools
    record["fold"] = selected_fold
    record["method"] = session.expected_registry_identity.method
    record["run_id"] = session.run_id
    record["run_kind"] = session.run_kind
    record["checkpoint_sha256"] = checkpoint.sha256
    record["config_hash"] = session.config_hash
    record["split_fingerprint"] = session.split_fingerprint
    record["source_provenance"] = {
        source: _build_learned_provenance(
            caches[source],
            statistics[source],
            session,
            checkpoint,
        ).to_dict()
        for source in ("teacher", "v1")
    }
    record.pop("probe_version", None)
    record["parameters"] = int(parameter_count)
    record["checkpoint_bytes"] = int(checkpoint_bytes)
    record["feature_bytes"] = {
        source: int(cost.payload_bytes) for source, cost in index_costs.items()
    }
    record["index_bytes"] = {source: int(cost.index_bytes) for source, cost in index_costs.items()}
    selected_sources = (
        ("teacher", "v1") if selected_gallery_policy == "two_view" else (selected_gallery_policy,)
    )
    selected_total = sum(index_costs[source].index_bytes for source in selected_sources)
    record["selected_gallery_policy"] = selected_gallery_policy
    record["selected_policy_total_index_bytes"] = int(selected_total)
    record["p95_end_to_end_under_one_second"] = bool(
        timing_summary.loc[
            timing_summary["metric"].eq("end_to_end") & timing_summary["percentile"].eq("p95"),
            "value_seconds",
        ]
        .lt(1.0)
        .all()
    )
    record["index_under_one_gibibyte"] = bool(selected_total < _INDEX_LIMIT_BYTES)
    record["measurement_route"] = _MEASUREMENT_ROUTE
    record["measurement_sha256"] = _sha256_bytes(_canonical_json(record))
    return record


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in _SORT_COLUMNS if column in frame]
    if not columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(
        columns,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _ranking_candidate_sets(rankings: pd.DataFrame, *, k: int) -> pd.Series:
    top = rankings.loc[pd.to_numeric(rankings["rank"], errors="raise").le(k)].copy()
    top["query_id"] = pd.to_numeric(top["query_id"], errors="raise").astype(int)
    top["candidate_id"] = pd.to_numeric(top["candidate_id"], errors="raise").astype(int)
    return top.groupby("query_id", sort=True)["candidate_id"].agg(
        lambda values: frozenset(int(value) for value in values)
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")


def _validate_learned_cost(
    cost: Mapping[str, object],
    *,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    timing_summary: pd.DataFrame,
    canonical_development_rows: int,
    provenance: Mapping[str, LearnedProvenance] | None = None,
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "scope",
        "fold",
        "contract",
        "hardware",
        "warmup_queries",
        "timed_queries",
        "timing_summary",
        "per_source_index_cost",
        "p95_end_to_end_under_one_second",
        "index_under_one_gibibyte",
        "method",
        "run_id",
        "run_kind",
        "checkpoint_sha256",
        "config_hash",
        "split_fingerprint",
        "source_provenance",
        "parameters",
        "checkpoint_bytes",
        "feature_bytes",
        "index_bytes",
        "selected_gallery_policy",
        "selected_policy_total_index_bytes",
        "measurement_route",
        "measurement_sha256",
    }
    if set(cost) != expected_keys:
        raise ValueError(
            f"learned cost fields must be {sorted(expected_keys)}, found {sorted(cost)}"
        )
    if cost.get("measurement_route") != _MEASUREMENT_ROUTE:
        raise ValueError("learned cost measurement route is malformed")
    expected_identity = {
        "schema_version": 1,
        "scope": "development",
        "fold": session.validation_fold,
        "contract": _CONTRACT.to_dict(),
        "method": session.expected_registry_identity.method,
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "checkpoint_sha256": checkpoint.sha256,
        "config_hash": session.config_hash,
        "split_fingerprint": session.split_fingerprint,
    }
    for key, expected in expected_identity.items():
        if cost.get(key) != expected:
            raise ValueError(f"learned cost {key} does not match evidence provenance")
    for key in ("parameters", "checkpoint_bytes", "timed_queries", "warmup_queries"):
        value = cost.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"learned cost {key} must be a non-negative integer")
    if _sha256_file(checkpoint.path) != checkpoint.sha256:
        raise ValueError("learned cost checkpoint SHA-256 does not match the selected checkpoint")
    if cost["checkpoint_bytes"] != checkpoint.path.stat().st_size:
        raise ValueError("learned cost checkpoint bytes do not match the selected checkpoint")
    hardware = cost.get("hardware")
    expected_hardware_fields = {
        "cpu",
        "logical_cores",
        "operating_system",
        "python_version",
        "numpy_version",
        "thread_count",
        "thread_environment",
        "native_thread_pools",
    }
    if not isinstance(hardware, dict) or set(hardware) != expected_hardware_fields:
        raise ValueError("learned cost hardware fields are malformed")
    native_pools = hardware.get("native_thread_pools")
    if (
        hardware.get("thread_count") != 1
        or hardware.get("thread_environment") != {name: "1" for name in THREAD_VARIABLES}
        or not isinstance(native_pools, list)
        or not native_pools
        or any(
            not isinstance(pool, dict)
            or set(pool) != {"user_api", "internal_api", "num_threads", "prefix"}
            or pool.get("user_api") not in {"blas", "openmp"}
            or pool.get("num_threads") != 1
            for pool in native_pools
        )
    ):
        raise ValueError("learned cost does not prove one live native CPU thread")
    if (
        type(cost.get("p95_end_to_end_under_one_second")) is not bool
        or type(cost.get("index_under_one_gibibyte")) is not bool
    ):
        raise ValueError("learned cost gates must be booleans")
    timing_records = cost.get("timing_summary")
    if not isinstance(timing_records, list):
        raise ValueError("learned cost timing summary must be records")
    timing_columns = _ARTIFACT_PAYLOAD_COLUMNS["timing_summary"]
    stored_timing = _ordered_frame(pd.DataFrame.from_records(timing_records).loc[:, timing_columns])
    expected_timing = _ordered_frame(
        timing_summary.drop(columns=list(_PROVENANCE_COLUMNS), errors="ignore")
    ).loc[:, timing_columns]
    try:
        pd.testing.assert_frame_equal(
            stored_timing.reset_index(drop=True),
            expected_timing.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-8,
            rtol=0.0,
        )
    except AssertionError as error:
        raise ValueError("learned cost timing summary disagrees with timing artifact") from error
    per_source = cost.get("per_source_index_cost")
    feature_bytes = cost.get("feature_bytes")
    index_bytes = cost.get("index_bytes")
    if (
        not isinstance(per_source, dict)
        or set(per_source) != {"teacher", "v1"}
        or not isinstance(feature_bytes, dict)
        or set(feature_bytes) != {"teacher", "v1"}
        or not isinstance(index_bytes, dict)
        or set(index_bytes) != {"teacher", "v1"}
    ):
        raise ValueError("learned cost needs exact teacher and v1 index records")
    index_schema = {
        "source",
        "contract",
        "rows",
        "dimension",
        "payload_bytes",
        "index_bytes",
        "build_seconds",
        "peak_rss_bytes",
    }
    for source in ("teacher", "v1"):
        record = per_source[source]
        if not isinstance(record, dict) or set(record) != index_schema:
            raise ValueError("learned per-source index cost schema is malformed")
        numeric_integers = ("rows", "dimension", "payload_bytes", "index_bytes", "peak_rss_bytes")
        if any(
            isinstance(record.get(field), bool)
            or not isinstance(record.get(field), Integral)
            or int(record[field]) < 0
            for field in numeric_integers
        ):
            raise ValueError("learned per-source index cost numeric fields are malformed")
        build_seconds = record.get("build_seconds")
        if (
            isinstance(build_seconds, bool)
            or not isinstance(build_seconds, Real)
            or not math.isfinite(float(build_seconds))
            or float(build_seconds) < 0.0
        ):
            raise ValueError("learned per-source index cost build seconds are malformed")
        if (
            record.get("source") != source
            or record.get("contract") != _CONTRACT.to_dict()
            or record.get("dimension") != EMBEDDING_DIM
            or record.get("rows") != canonical_development_rows
            or record.get("payload_bytes")
            != canonical_development_rows * EMBEDDING_DIM * np.dtype(np.float32).itemsize
            or record.get("index_bytes")
            != record.get("payload_bytes")
            + canonical_development_rows * np.dtype(np.int64).itemsize
            or feature_bytes[source] != record.get("payload_bytes")
            or index_bytes[source] != record.get("index_bytes")
        ):
            raise ValueError(
                "learned per-source index cost values or canonical development rows "
                "are inconsistent"
            )
    stored_provenance = cost.get("source_provenance")
    if not isinstance(stored_provenance, dict) or set(stored_provenance) != {"teacher", "v1"}:
        raise ValueError("learned cost source provenance is malformed")
    parsed_provenance = {
        source: LearnedProvenance(**stored_provenance[source]) for source in ("teacher", "v1")
    }
    if parsed_provenance["teacher"].model_identity != parsed_provenance["v1"].model_identity:
        raise ValueError("learned cost source provenance model identity disagrees")
    if provenance is not None and parsed_provenance != dict(provenance):
        raise ValueError("learned cost source provenance disagrees with feature indexes")
    policy = cost.get("selected_gallery_policy")
    if policy not in {"teacher", "v1", "two_view"}:
        raise ValueError("learned cost selected gallery policy is invalid")
    selected_sources = ("teacher", "v1") if policy == "two_view" else (policy,)
    total = sum(int(index_bytes[source]) for source in selected_sources)
    if cost.get("selected_policy_total_index_bytes") != total:
        raise ValueError("learned cost selected-policy storage total is inconsistent")
    if cost.get("index_under_one_gibibyte") is not (total < _INDEX_LIMIT_BYTES):
        raise ValueError("learned cost storage gate is inconsistent")
    p95 = expected_timing.loc[
        expected_timing["metric"].eq("end_to_end") & expected_timing["percentile"].eq("p95"),
        "value_seconds",
    ]
    if cost.get("p95_end_to_end_under_one_second") is not bool(p95.lt(1.0).all()):
        raise ValueError("learned cost timing gate is inconsistent")
    try:
        model = _load_evidence_model(checkpoint, session, "cpu")
    except Exception:
        model = None
    if model is not None:
        measured_parameters = sum(parameter.numel() for parameter in model.parameters())
        if int(cost["parameters"]) != int(measured_parameters):
            raise ValueError(
                "learned cost parameter count does not match the selected checkpoint model"
            )
    try:
        json.dumps(cost, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("learned cost artifact must be finite and JSON-safe") from error
    digest_payload = dict(cost)
    stored_digest = digest_payload.pop("measurement_sha256", None)
    if stored_digest != _sha256_bytes(_canonical_json(digest_payload)):
        raise ValueError("learned cost measurement digest does not match its values")
    return dict(cost)


def _validate_artifact_inputs(
    frames: Mapping[str, pd.DataFrame],
    *,
    expected_query_ids: Sequence[int],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
) -> dict[str, pd.DataFrame]:
    if set(frames) != set(_FRAME_ARTIFACTS):
        missing = sorted(set(_FRAME_ARTIFACTS).difference(frames))
        extra = sorted(set(frames).difference(_FRAME_ARTIFACTS))
        raise ValueError(f"learned artifact frames mismatch; missing={missing}, extra={extra}")
    expected_provenance: dict[str, object] = {
        "schema_version": LEARNED_EVIDENCE_SCHEMA_VERSION,
        "scope": "development",
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "method": session.expected_registry_identity.method,
        "fold": session.validation_fold,
        "checkpoint_sha256": checkpoint.sha256,
        "config_hash": session.config_hash,
        "split_fingerprint": session.split_fingerprint,
    }
    prepared: dict[str, pd.DataFrame] = {}
    for name, source_frame in frames.items():
        frame = source_frame.copy()
        aliases = {
            "checkpoint_fingerprint": "checkpoint_sha256",
            "config_fingerprint": "config_hash",
        }
        for old, new in aliases.items():
            if old in frame:
                if new in frame and not frame[old].equals(frame[new]):
                    raise ValueError(f"{name} artifact has conflicting {new} provenance")
                frame[new] = frame[old]
                frame = frame.drop(columns=old)
        if "descriptor_fingerprint" in frame:
            observed = frame["descriptor_fingerprint"]
            if (observed.notna() & observed.astype(str).str.strip().ne("")).any():
                raise ValueError(
                    f"{name} artifact carries untrained descriptor provenance; learned "
                    "evidence must come from a trained checkpoint"
                )
            frame = frame.drop(columns="descriptor_fingerprint")
        for column, value in expected_provenance.items():
            if column in frame:
                observed = frame[column]
                if observed.isna().any() or set(observed.astype(str)) != {str(value)}:
                    raise ValueError(f"{name} artifact {column} provenance does not match")
            else:
                frame.insert(min(len(frame.columns), len(prepared)), column, value)
        if name in {"failure_slices", "canvas_summary", "canvas_per_query"}:
            if "caveat" in frame and set(frame["caveat"].astype(str)) != {TEACHER_SLICE_CAVEAT}:
                raise ValueError(f"{name} artifact caveat does not match")
            frame["caveat"] = TEACHER_SLICE_CAVEAT
        expected_columns = LEARNED_ARTIFACT_SCHEMAS[name]
        if set(frame.columns) != set(expected_columns):
            raise ValueError(
                f"{name} artifact columns must be {expected_columns}, found {tuple(frame.columns)}"
            )
        prepared[name] = _ordered_frame(frame.loc[:, expected_columns])

    for name, frame in prepared.items():
        if frame.empty:
            raise ValueError(f"{name} artifact must not be empty")
    _assert_query_coverage(
        prepared["query_metrics"],
        expected_query_ids,
        group_columns=("query_source", "gallery_source", "protocol"),
        label="query metrics",
    )
    _assert_query_coverage(
        prepared["rankings"].drop_duplicates(
            ["query_source", "gallery_source", "protocol", "query_id"]
        ),
        expected_query_ids,
        group_columns=("query_source", "gallery_source", "protocol"),
        label="rankings",
    )
    ranking_keys = [
        "query_source",
        "gallery_source",
        "protocol",
        "query_id",
        "candidate_id",
    ]
    if prepared["rankings"].duplicated(ranking_keys).any():
        raise ValueError("rankings contain duplicate product IDs within a query")
    ranking_ids = (
        prepared["rankings"]
        .loc[:, ["query_id", "candidate_id", "rank"]]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )
    distances = pd.to_numeric(prepared["rankings"]["distance"], errors="coerce")
    if (
        ranking_ids.isna().any().any()
        or not np.isfinite(ranking_ids.to_numpy()).all()
        or not ranking_ids.mod(1).eq(0).all().all()
        or ranking_ids[["query_id", "candidate_id", "rank"]].le(0).any().any()
        or distances.isna().any()
        or not np.isfinite(distances).all()
        or not distances.between(0.0, 2.0).all()
    ):
        raise ValueError("rankings require integer IDs/ranks and finite [0, 2] distances")
    for keys, rows in prepared["rankings"].groupby(ranking_keys[:4], sort=False):
        ranks = pd.to_numeric(rows["rank"], errors="coerce")
        expected_count = 20 if keys[2] == "primary" else 10
        if len(rows) != expected_count:
            raise ValueError("rankings require frozen Top-20 Protocol A and Top-10 Protocol B")
        if not np.array_equal(
            np.sort(ranks.to_numpy()),
            np.arange(1, len(rows) + 1),
        ):
            raise ValueError("rankings require consecutive one-based ranks")
        ranked = rows.sort_values("rank", kind="mergesort")
        expected_order = rows.assign(
            _numeric_candidate=pd.to_numeric(rows["candidate_id"], errors="raise")
        ).sort_values(["distance", "_numeric_candidate"], kind="mergesort")
        if (
            ranked["candidate_id"].astype(int).tolist()
            != expected_order["candidate_id"].astype(int).tolist()
        ):
            raise ValueError("ranking rank must follow distance then numeric candidate ID order")
    _assert_query_coverage(
        prepared["timing_samples"],
        expected_query_ids,
        group_columns=("query_source", "gallery_source"),
        label="timing samples",
    )
    expected_directions = set(source_directions())
    for name in ("query_metrics", "rankings", "timing_samples"):
        observed = set(
            prepared[name][["query_source", "gallery_source"]].itertuples(
                index=False,
                name=None,
            )
        )
        if observed != expected_directions:
            raise ValueError(f"{name} artifact requires all four source directions")
    expected_protocol_groups = {
        (query_source, gallery_source, protocol)
        for query_source, gallery_source in source_directions()
        for protocol in ("primary", "family")
    }
    for name in ("query_metrics", "rankings"):
        observed_groups = set(
            prepared[name][["query_source", "gallery_source", "protocol"]].itertuples(
                index=False, name=None
            )
        )
        if observed_groups != expected_protocol_groups:
            raise ValueError(f"{name} artifact requires complete Protocol A/B groups")

    quality = prepared["quality_summary"]
    quality_keys = ["query_source", "gallery_source", "protocol", "metric", "k", "aggregation"]
    if quality.duplicated(quality_keys).any():
        raise ValueError("quality summary requires exactly one row per metric combination")
    expected_primary = {
        (metric, k, aggregation)
        for metric in ("ndcg", "precision_any", "precision_strict", "tie_rate")
        for k in (5, 10, 20)
        for aggregation in ("query_mean", "article_type_macro")
    }
    expected_family = {
        (metric, 10, "query_mean")
        for metric in ("recall", "hit_rate", "precision", "coverage", "tie_rate")
    }
    for direction in source_directions():
        rows = quality.loc[
            quality["query_source"].eq(direction[0]) & quality["gallery_source"].eq(direction[1])
        ]
        for protocol, expected_grid in (
            ("primary", expected_primary),
            ("family", expected_family),
        ):
            observed_grid = set(
                rows.loc[rows["protocol"].eq(protocol), ["metric", "k", "aggregation"]].itertuples(
                    index=False, name=None
                )
            )
            if observed_grid != expected_grid:
                raise ValueError("quality summary has an incomplete metric grid")
    quality_values = pd.to_numeric(quality["value"], errors="coerce")
    query_counts = pd.to_numeric(quality["query_count"], errors="coerce")
    class_counts = pd.to_numeric(quality["class_count"], errors="coerce")
    class_count_rows = quality["protocol"].eq("primary")
    present_quality = quality_values.notna()
    if (
        (present_quality & ~np.isfinite(quality_values)).any()
        or (present_quality & ~quality_values.between(0.0, 1.0)).any()
        or (quality_values.isna() & query_counts.ne(0)).any()
        or query_counts.isna().any()
        or not query_counts.mod(1).eq(0).all()
        or query_counts.lt(0).any()
        or query_counts.gt(len(expected_query_ids)).any()
        or class_counts[class_count_rows].isna().any()
        or not class_counts[class_count_rows].mod(1).eq(0).all()
        or class_counts[class_count_rows].lt(0).any()
        or class_counts[class_count_rows].gt(query_counts[class_count_rows]).any()
        or class_counts[~class_count_rows].notna().any()
    ):
        raise ValueError("quality summary numeric counts or values are malformed")

    query_metric_columns = _ARTIFACT_PAYLOAD_COLUMNS["query_metrics"][6:]
    query_values = (
        prepared["query_metrics"]
        .loc[:, query_metric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )
    present = query_values.notna()
    if (
        (present & ~np.isfinite(query_values)).any().any()
        or (query_values[present] < 0.0).any().any()
        or (query_values[present] > 1.0).any().any()
    ):
        raise ValueError("query metric values must be null or finite in [0, 1]")

    recomputed_pairs: dict[Direction, PairEvaluation] = {}
    recomputed_summaries: list[pd.DataFrame] = []
    for direction in source_directions():
        selected = prepared["rankings"].loc[
            prepared["rankings"]["query_source"].eq(direction[0])
            & prepared["rankings"]["gallery_source"].eq(direction[1])
        ]
        primary_rankings = selected.loc[
            selected["protocol"].eq("primary"),
            ["query_id", "candidate_id", "distance", "rank"],
        ]
        family_rankings = selected.loc[
            selected["protocol"].eq("family"),
            ["query_id", "candidate_id", "distance", "rank"],
        ]
        primary_per_query, primary_summary = evaluate_primary_rankings(
            primary_rankings,
            primary_views,
            k_values=(5, 10, 20),
        )
        family_per_query, family_summary = evaluate_family_rankings(
            family_rankings,
            family_views,
            k=10,
        )
        recomputed_pairs[direction] = PairEvaluation(
            summary=quality.loc[
                quality["query_source"].eq(direction[0])
                & quality["gallery_source"].eq(direction[1])
            ]
            .rename(
                columns={
                    "checkpoint_sha256": "checkpoint_fingerprint",
                    "config_hash": "config_fingerprint",
                }
            )
            .assign(descriptor_fingerprint=None)
            .copy(),
            primary_rankings=primary_rankings,
            family_rankings=family_rankings,
            primary_per_query=primary_per_query,
            family_per_query=family_per_query,
            method=session.expected_registry_identity.method,
            fold=session.validation_fold,
            checkpoint_fingerprint=checkpoint.sha256,
            config_fingerprint=session.config_hash,
            descriptor_fingerprint=None,
        )
        for protocol, values in (("primary", primary_summary), ("family", family_summary)):
            recomputed_summaries.append(
                values.assign(
                    query_source=direction[0],
                    gallery_source=direction[1],
                    protocol=protocol,
                )
            )
    expected_query_metrics = _ordered_frame(build_query_metrics(recomputed_pairs))
    observed_query_metrics = _ordered_frame(
        prepared["query_metrics"].drop(columns=list(_PROVENANCE_COLUMNS))
    )
    compare_query_columns = list(_ARTIFACT_PAYLOAD_COLUMNS["query_metrics"])
    try:
        pd.testing.assert_frame_equal(
            observed_query_metrics.loc[:, compare_query_columns].reset_index(drop=True),
            expected_query_metrics.loc[:, compare_query_columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-8,
            rtol=0.0,
        )
    except AssertionError as error:
        raise ValueError("query metrics disagree with persisted rankings") from error

    recomputed_quality = _ordered_frame(pd.concat(recomputed_summaries, ignore_index=True))
    observed_quality = _ordered_frame(quality)
    quality_compare = [
        "query_source",
        "gallery_source",
        "protocol",
        "metric",
        "k",
        "aggregation",
        "value",
        "query_count",
        "class_count",
    ]
    observed_quality_compare = observed_quality.loc[:, quality_compare].reset_index(drop=True)
    recomputed_quality_compare = recomputed_quality.loc[:, quality_compare].reset_index(drop=True)
    for column in ("value", "query_count", "class_count"):
        observed_quality_compare[column] = pd.to_numeric(
            observed_quality_compare[column], errors="coerce"
        ).astype(float)
        recomputed_quality_compare[column] = pd.to_numeric(
            recomputed_quality_compare[column], errors="coerce"
        ).astype(float)
    try:
        pd.testing.assert_frame_equal(
            observed_quality_compare,
            recomputed_quality_compare,
            check_dtype=False,
            check_exact=False,
            atol=1e-8,
            rtol=0.0,
        )
    except AssertionError as error:
        raise ValueError("quality summary disagrees with query metrics and rankings") from error

    timing_values = (
        prepared["timing_samples"]
        .loc[:, ["encoding_seconds", "search_seconds", "end_to_end_seconds"]]
        .apply(pd.to_numeric, errors="coerce")
    )
    if (
        timing_values.isna().any().any()
        or not np.isfinite(timing_values.to_numpy()).all()
        or timing_values.lt(0).any().any()
    ):
        raise ValueError("timing samples must contain finite non-negative values")
    summary = prepared["timing_summary"]
    expected_timing_grid = {
        (*direction, metric, percentile)
        for direction in source_directions()
        for metric in ("encoding", "search", "end_to_end")
        for percentile in ("p50", "p95")
    }
    observed_timing_grid = set(
        summary[["query_source", "gallery_source", "metric", "percentile"]].itertuples(
            index=False, name=None
        )
    )
    if observed_timing_grid != expected_timing_grid:
        raise ValueError("timing summary has an incomplete metric grid")
    if set(pd.to_numeric(summary["timed_queries"], errors="coerce")) != {
        len(set(int(value) for value in expected_query_ids))
    }:
        raise ValueError("timing summary query count does not match canonical queries")
    recomputed_timing = _ordered_frame(
        pd.concat(
            [
                summarize_timings(rows)
                for _, rows in prepared["timing_samples"].groupby(
                    ["query_source", "gallery_source"],
                    sort=False,
                )
            ],
            ignore_index=True,
        )
    )
    timing_compare = list(_ARTIFACT_PAYLOAD_COLUMNS["timing_summary"])
    try:
        pd.testing.assert_frame_equal(
            _ordered_frame(summary).loc[:, timing_compare].reset_index(drop=True),
            recomputed_timing.loc[:, timing_compare].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-8,
            rtol=0.0,
        )
    except AssertionError as error:
        raise ValueError("timing summary disagrees with timing samples") from error

    gallery = prepared["gallery_comparison"]
    expected_gallery_rows = {
        (
            query_source,
            policy,
            "gallery_quality" if query_source == "equal_teacher_v1_mean" else "ndcg",
            10,
            "equal_source_mean" if query_source == "equal_teacher_v1_mean" else "query_mean",
        )
        for policy in ("teacher", "v1", "two_view")
        for query_source in ("teacher", "v1", "equal_teacher_v1_mean")
    }
    gallery_keys = ["query_source", "gallery_policy", "metric", "k", "aggregation"]
    if set(gallery[gallery_keys].itertuples(index=False, name=None)) != expected_gallery_rows:
        raise ValueError("gallery comparison requires the exact metric and source/policy rows")
    gallery_rankings = prepared["gallery_rankings"]
    expected_gallery_groups = {
        (query_source, policy)
        for query_source in ("teacher", "v1")
        for policy in ("teacher", "v1", "two_view")
    }
    if (
        set(gallery_rankings[["query_source", "gallery_policy"]].itertuples(index=False, name=None))
        != expected_gallery_groups
    ):
        raise ValueError("gallery rankings require every source and policy pair")
    _assert_query_coverage(
        gallery_rankings.drop_duplicates(["query_source", "gallery_policy", "query_id"]),
        expected_query_ids,
        group_columns=("query_source", "gallery_policy"),
        label="gallery rankings",
    )
    recomputed_gallery_values: dict[tuple[str, str], float] = {}
    for keys, rows in gallery_rankings.groupby(
        ["query_source", "gallery_policy", "query_id"], sort=False
    ):
        if len(rows) != 20 or sorted(rows["rank"].astype(int).tolist()) != list(range(1, 21)):
            raise ValueError("gallery rankings require frozen consecutive Top-20 rows")
        ranked = rows.sort_values("rank", kind="mergesort")
        expected_order = rows.assign(
            _candidate=pd.to_numeric(rows["candidate_id"], errors="raise")
        ).sort_values(["distance", "_candidate"], kind="mergesort")
        if (
            ranked["candidate_id"].astype(int).tolist()
            != expected_order["candidate_id"].astype(int).tolist()
        ):
            raise ValueError("gallery ranking rank must follow distance and numeric candidate ID")
    for query_source, policy in expected_gallery_groups:
        rankings = gallery_rankings.loc[
            gallery_rankings["query_source"].eq(query_source)
            & gallery_rankings["gallery_policy"].eq(policy),
            ["query_id", "candidate_id", "distance", "rank"],
        ]
        _, summary_rows = evaluate_primary_rankings(rankings, primary_views, k_values=(10,))
        recomputed_gallery_values[(query_source, policy)] = float(
            summary_rows.loc[
                summary_rows["metric"].eq("ndcg") & summary_rows["aggregation"].eq("query_mean"),
                "value",
            ].iloc[0]
        )
    for policy in ("teacher", "v1", "two_view"):
        source_rows = gallery.loc[
            gallery["gallery_policy"].eq(policy) & gallery["query_source"].isin(("teacher", "v1"))
        ]
        observed = {
            str(row.query_source): float(row.value) for row in source_rows.itertuples(index=False)
        }
        wanted = {
            source: recomputed_gallery_values[(source, policy)] for source in ("teacher", "v1")
        }
        mean_value = float(
            gallery.loc[
                gallery["gallery_policy"].eq(policy)
                & gallery["query_source"].eq("equal_teacher_v1_mean"),
                "value",
            ].iloc[0]
        )
        if any(
            not np.isclose(observed[source], wanted[source], atol=1e-8, rtol=0.0)
            for source in wanted
        ) or not np.isclose(mean_value, np.mean(list(wanted.values())), atol=1e-8, rtol=0.0):
            raise ValueError("gallery comparison disagrees with its source rows and rankings")

    failure = prepared["failure_slices"]
    slice_rules = {
        "grayscale": ("primary", "ndcg"),
        "rare_article_type": ("primary", "ndcg"),
        "rare_type_colour": ("primary", "ndcg"),
        "unusual_geometry": ("primary", "ndcg"),
        "family_unavailable": ("family", "recall"),
        "weak_family": ("family", "recall"),
    }
    expected_failure_grid = {
        (*direction, protocol, slice_name, metric, 10, "query_mean")
        for direction in source_directions()
        for slice_name, (protocol, metric) in slice_rules.items()
    }
    observed_failure_grid = set(
        failure[
            [
                "query_source",
                "gallery_source",
                "protocol",
                "slice",
                "metric",
                "k",
                "aggregation",
            ]
        ].itertuples(index=False, name=None)
    )
    if observed_failure_grid != expected_failure_grid:
        raise ValueError("failure slice artifact has an incomplete grid")
    failure_counts = failure[["total_queries", "scored_queries", "excluded_queries"]].apply(
        pd.to_numeric, errors="coerce"
    )
    coverage_values = pd.to_numeric(failure["coverage"], errors="coerce")
    failure_values = pd.to_numeric(failure["value"], errors="coerce")
    if (
        failure_counts.isna().any().any()
        or not failure_counts.mod(1).eq(0).all().all()
        or failure_counts.lt(0).any().any()
        or not (failure_counts["scored_queries"] + failure_counts["excluded_queries"])
        .eq(failure_counts["total_queries"])
        .all()
        or coverage_values.isna().any()
        or not coverage_values.between(0.0, 1.0).all()
        or (
            failure_values.notna()
            & (
                ~np.isfinite(failure_values)
                | ~failure_values.between(0.0, 1.0)
                | failure_counts["scored_queries"].eq(0)
            )
        ).any()
        or (failure_values.isna() & failure_counts["scored_queries"].gt(0)).any()
        or not np.allclose(
            coverage_values,
            failure_counts["scored_queries"]
            .div(failure_counts["total_queries"].replace(0, np.nan))
            .fillna(0.0),
            atol=1e-8,
            rtol=0.0,
        )
    ):
        raise ValueError("failure slice numeric values, counts, or coverage are inconsistent")
    membership = mark_failure_slices(build_query_support(primary_views, family_views))
    expected_failure = summarize_failure_slices(prepared["query_metrics"], membership)
    expected_failure["caveat"] = TEACHER_SLICE_CAVEAT
    failure_compare = list(_ARTIFACT_PAYLOAD_COLUMNS["failure_slices"])
    try:
        pd.testing.assert_frame_equal(
            _ordered_frame(failure).loc[:, failure_compare].reset_index(drop=True),
            _ordered_frame(expected_failure).loc[:, failure_compare].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-8,
            rtol=0.0,
        )
    except AssertionError as error:
        raise ValueError("failure slice values disagree with query metrics") from error

    canvas_summary = prepared["canvas_summary"]
    if (
        set(canvas_summary["query_variant"]) != {"clean", "wide", "tall"}
        or len(canvas_summary) != 3
        or not pd.to_numeric(canvas_summary["queries"], errors="coerce")
        .eq(len(expected_query_ids))
        .all()
    ):
        raise ValueError("canvas summary requires clean, wide, and tall rows")
    canvas_per_query = prepared["canvas_per_query"]
    expected_canvas_grid = {
        (query_id, variant)
        for query_id in expected_query_ids
        for variant in ("clean", "wide", "tall")
    }
    if (
        set(canvas_per_query[["query_id", "query_variant"]].itertuples(index=False, name=None))
        != expected_canvas_grid
    ):
        raise ValueError("canvas per-query artifact has an incomplete query/variant grid")
    canvas_rankings = prepared["canvas_rankings"]
    _assert_query_coverage(
        canvas_rankings.drop_duplicates(["query_variant", "query_id"]),
        expected_query_ids,
        group_columns=("query_variant",),
        label="canvas rankings",
    )
    if (
        set(canvas_rankings["query_variant"]) != {"clean", "wide", "tall"}
        or set(canvas_rankings["query_source"]) != {"v1"}
        or set(canvas_rankings["gallery_source"]) != {"v1"}
    ):
        raise ValueError("canvas rankings require exact V1 clean, wide, and tall groups")
    for _, rows in canvas_rankings.groupby(["query_variant", "query_id"], sort=False):
        if len(rows) != 10 or sorted(rows["rank"].astype(int)) != list(range(1, 11)):
            raise ValueError("canvas rankings require consecutive Top-10 rows")
        ranked = rows.sort_values("rank", kind="mergesort")
        wanted = rows.assign(
            _candidate=pd.to_numeric(rows["candidate_id"], errors="raise")
        ).sort_values(["distance", "_candidate"], kind="mergesort")
        if (
            ranked["candidate_id"].astype(int).tolist()
            != wanted["candidate_id"].astype(int).tolist()
        ):
            raise ValueError("canvas rankings must follow distance and numeric candidate ID")
    canvas_rankings_by_variant = {
        variant: canvas_rankings.loc[
            canvas_rankings["query_variant"].eq(variant),
            ["query_id", "candidate_id", "distance", "rank"],
        ].copy()
        for variant in ("clean", "wide", "tall")
    }
    clean_metrics, _ = evaluate_primary_rankings(
        canvas_rankings_by_variant["clean"],
        primary_views,
        k_values=(10,),
    )
    clean_values = clean_metrics.loc[:, ["query_id", "ndcg_at_10"]].rename(
        columns={"ndcg_at_10": "clean_ndcg_at_10"}
    )
    clean_sets = _ranking_candidate_sets(canvas_rankings_by_variant["clean"], k=10)
    expected_canvas_rows: list[pd.DataFrame] = []
    for variant in ("clean", "wide", "tall"):
        canvas_metrics, _ = evaluate_primary_rankings(
            canvas_rankings_by_variant[variant],
            primary_views,
            k_values=(10,),
        )
        canvas_values = canvas_metrics.loc[:, ["query_id", "ndcg_at_10"]].rename(
            columns={"ndcg_at_10": "canvas_ndcg_at_10"}
        )
        if variant == "clean":
            overlaps = pd.DataFrame(
                {
                    "query_id": sorted(clean_sets.index.astype(int).tolist()),
                    "top10_overlap": 1.0,
                }
            )
        else:
            variant_sets = _ranking_candidate_sets(canvas_rankings_by_variant[variant], k=10)
            overlaps = pd.DataFrame(
                {
                    "query_id": sorted(clean_sets.index.astype(int).tolist()),
                    "top10_overlap": [
                        len(clean_sets.loc[query_id] & variant_sets.loc[query_id]) / 10
                        for query_id in sorted(clean_sets.index.astype(int).tolist())
                    ],
                }
            )
        labelled = (
            clean_values.merge(canvas_values, on="query_id", validate="one_to_one")
            .merge(overlaps, on="query_id", validate="one_to_one")
            .assign(
                size="240x320",
                query_source="v1",
                gallery_source="v1",
                query_variant=variant,
                caveat=TEACHER_SLICE_CAVEAT,
            )
        )
        labelled["ndcg_change_from_clean"] = (
            0.0
            if variant == "clean"
            else labelled["canvas_ndcg_at_10"] - labelled["clean_ndcg_at_10"]
        )
        expected_canvas_rows.append(labelled)
    expected_canvas_per_query = _ordered_frame(
        pd.concat(expected_canvas_rows, ignore_index=True).loc[
            :,
            list(_ARTIFACT_PAYLOAD_COLUMNS["canvas_per_query"]),
        ]
    )
    observed_canvas_per_query = _ordered_frame(
        canvas_per_query.drop(columns=list(_PROVENANCE_COLUMNS))
    ).loc[:, list(_ARTIFACT_PAYLOAD_COLUMNS["canvas_per_query"])]
    try:
        pd.testing.assert_frame_equal(
            observed_canvas_per_query.reset_index(drop=True),
            expected_canvas_per_query.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-8,
            rtol=0.0,
        )
    except AssertionError as error:
        raise ValueError(
            "canvas per-query metrics and overlap disagree with canvas rankings"
        ) from error
    for frame, columns, label in (
        (canvas_summary, ("mean_top10_overlap",), "canvas summary"),
        (canvas_per_query, ("top10_overlap",), "canvas per-query"),
        (gallery, ("value",), "gallery comparison"),
    ):
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        if (
            values.isna().any().any()
            or not np.isfinite(values.to_numpy()).all()
            or values.lt(0.0).any().any()
            or values.gt(1.0).any().any()
        ):
            raise ValueError(f"{label} bounded values must be finite and in [0, 1]")
    summary_ndcg = pd.to_numeric(canvas_summary["ndcg_at_10"], errors="coerce")
    clean_ndcg = pd.to_numeric(canvas_per_query["clean_ndcg_at_10"], errors="coerce")
    canvas_ndcg = pd.to_numeric(canvas_per_query["canvas_ndcg_at_10"], errors="coerce")
    if (
        (summary_ndcg.notna() & (~np.isfinite(summary_ndcg) | ~summary_ndcg.between(0, 1))).any()
        or (clean_ndcg.notna() & (~np.isfinite(clean_ndcg) | ~clean_ndcg.between(0, 1))).any()
        or (canvas_ndcg.notna() & (~np.isfinite(canvas_ndcg) | ~canvas_ndcg.between(0, 1))).any()
        or not clean_ndcg.isna().equals(canvas_ndcg.isna())
    ):
        raise ValueError("canvas nDCG values must share valid scorable/null semantics")
    summary_changes = pd.to_numeric(canvas_summary["ndcg_change_from_clean"], errors="coerce")
    per_query_changes = pd.to_numeric(canvas_per_query["ndcg_change_from_clean"], errors="coerce")
    clean_variants = canvas_per_query["query_variant"].eq("clean")
    expected_change_null = clean_ndcg.isna() & ~clean_variants
    clean_summary = float(
        canvas_summary.loc[canvas_summary["query_variant"].eq("clean"), "ndcg_at_10"].iloc[0]
    )
    if (
        not summary_changes.isna().equals(summary_ndcg.isna())
        or (
            summary_changes.notna()
            & (~np.isfinite(summary_changes) | ~summary_changes.between(-1.0, 1.0))
        ).any()
        or not np.allclose(
            summary_changes.dropna(),
            (summary_ndcg - clean_summary).dropna(),
            atol=1e-8,
            rtol=0.0,
        )
        or not per_query_changes.isna().equals(expected_change_null)
        or (
            per_query_changes.notna()
            & (~np.isfinite(per_query_changes) | ~per_query_changes.between(-1.0, 1.0))
        ).any()
        or not np.allclose(
            per_query_changes.loc[~clean_variants].dropna(),
            (canvas_ndcg - clean_ndcg).loc[~clean_variants].dropna(),
            atol=1e-8,
            rtol=0.0,
        )
        or not np.allclose(
            per_query_changes.loc[clean_variants],
            0.0,
            atol=1e-8,
            rtol=0.0,
        )
    ):
        raise ValueError("canvas numeric change values are inconsistent")
    expected_canvas_summary = (
        canvas_per_query.groupby("query_variant", sort=False)
        .agg(
            queries=("query_id", "size"),
            ndcg_at_10=("canvas_ndcg_at_10", "mean"),
            ndcg_change_from_clean=("ndcg_change_from_clean", "mean"),
            mean_top10_overlap=("top10_overlap", "mean"),
        )
        .reset_index()
    )
    canvas_compare = [
        "query_variant",
        "queries",
        "ndcg_at_10",
        "ndcg_change_from_clean",
        "mean_top10_overlap",
    ]
    try:
        pd.testing.assert_frame_equal(
            _ordered_frame(canvas_summary).loc[:, canvas_compare].reset_index(drop=True),
            _ordered_frame(expected_canvas_summary).loc[:, canvas_compare].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-8,
            rtol=0.0,
        )
    except AssertionError as error:
        raise ValueError("canvas summary disagrees with canvas per-query rows") from error

    timing_summary_values = pd.to_numeric(
        prepared["timing_summary"]["value_seconds"],
        errors="coerce",
    )
    if (
        timing_summary_values.isna().any()
        or not np.isfinite(timing_summary_values).all()
        or timing_summary_values.lt(0).any()
    ):
        raise ValueError("timing summary values must be finite and non-negative")

    examples = prepared["examples"]
    example_numbers = examples[["query_id", "candidate_id", "rank", "distance"]].apply(
        pd.to_numeric, errors="coerce"
    )
    example_values = pd.to_numeric(examples["value"], errors="coerce")
    if (
        example_numbers.isna().any().any()
        or not np.isfinite(example_numbers.to_numpy()).all()
        or not example_numbers[["query_id", "candidate_id", "rank"]].mod(1).eq(0).all().all()
        or not example_numbers["query_id"].astype(int).isin(expected_query_ids).all()
        or not example_numbers["rank"].between(1, 5).all()
        or not example_numbers["distance"].between(0.0, 2.0).all()
        or (
            example_values.notna()
            & (~np.isfinite(example_values) | ~example_values.between(0.0, 1.0))
        ).any()
    ):
        raise ValueError("example numeric values, IDs, ranks, or distances are malformed")
    expected_example_ids = select_example_ids(
        prepared["query_metrics"],
        membership,
        canvas_per_query,
    )
    if set(examples["slice"]) != set(expected_example_ids):
        raise ValueError("examples do not match deterministic selected slices")
    family_slices = {"family_unavailable", "weak_family"}
    for _, rows in examples.groupby("slice", sort=False):
        ranks = sorted(rows["rank"].astype(int).tolist())
        slice_name = str(rows["slice"].iloc[0])
        query_id = expected_example_ids[slice_name]
        if ranks != list(range(1, 6)) or set(rows["query_id"].astype(int)) != {query_id}:
            raise ValueError("examples do not match deterministic IDs or complete Top-5 rows")
        protocol = "family" if slice_name in family_slices else "primary"
        metric = "recall_at_10" if protocol == "family" else "ndcg_at_10"
        if slice_name == "canvas_failure":
            canvas_rows = canvas_per_query.loc[
                canvas_per_query["query_id"].eq(query_id)
                & canvas_per_query["query_variant"].isin(("wide", "tall"))
            ].sort_values(["ndcg_change_from_clean", "query_variant"], kind="mergesort")
            variant = str(canvas_rows.iloc[0]["query_variant"])
            wanted_value = canvas_rows.iloc[0]["canvas_ndcg_at_10"]
            ranking_rows = prepared["canvas_rankings"].loc[
                prepared["canvas_rankings"]["query_variant"].eq(variant)
                & prepared["canvas_rankings"]["query_id"].eq(query_id)
                & prepared["canvas_rankings"]["rank"].le(5)
            ]
        else:
            variant = "clean"
            metric_row = prepared["query_metrics"].loc[
                prepared["query_metrics"]["query_source"].eq("v1")
                & prepared["query_metrics"]["gallery_source"].eq("v1")
                & prepared["query_metrics"]["protocol"].eq(protocol)
                & prepared["query_metrics"]["query_id"].eq(query_id)
            ]
            wanted_value = metric_row.iloc[0][metric]
            ranking_rows = prepared["rankings"].loc[
                prepared["rankings"]["query_source"].eq("v1")
                & prepared["rankings"]["gallery_source"].eq("v1")
                & prepared["rankings"]["protocol"].eq(protocol)
                & prepared["rankings"]["query_id"].eq(query_id)
                & prepared["rankings"]["rank"].le(5)
            ]
        observed_ranking = rows[["query_id", "candidate_id", "distance", "rank"]].sort_values(
            "rank"
        )
        wanted_ranking = ranking_rows[["query_id", "candidate_id", "distance", "rank"]].sort_values(
            "rank"
        )
        observed_values = pd.to_numeric(rows["value"], errors="coerce")
        values_match = (
            observed_values.isna().all()
            if pd.isna(wanted_value)
            else np.allclose(observed_values, float(wanted_value), atol=1e-8, rtol=0.0)
        )
        if (
            set(rows["query_variant"]) != {variant}
            or set(rows["metric"]) != {metric}
            or not values_match
            or not observed_ranking.reset_index(drop=True).equals(
                wanted_ranking.reset_index(drop=True)
            )
        ):
            raise ValueError("examples disagree with deterministic IDs, metrics, or rankings")
    return prepared


def write_learned_artifacts(
    *,
    evidence_dir: str | Path,
    run_id: str,
    run_kind: str,
    checkpoint: CheckpointRecord,
    session: TrainingSessionConfig,
    canonical_splits: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    source_artifacts: Mapping[str, LearnedSourceArtifacts],
    selected_gallery_policy: Literal["teacher", "v1", "two_view"],
    timing_policy: TimingPolicy = TimingPolicy(),
    caveat: str = TEACHER_SLICE_CAVEAT,
) -> Path:
    """Measure, validate, atomically write, reopen, and manifest one learned evidence run."""

    _validate_checkpoint_identity(checkpoint, session)
    if run_id != session.run_id or run_kind != session.run_kind:
        raise ValueError("artifact run identity does not match the training session")
    if caveat != TEACHER_SLICE_CAVEAT:
        raise ValueError("learned evidence must retain the frozen teacher-slice caveat")
    validate_split_structure(canonical_splits)
    if cv_assignment_digest(canonical_splits) != session.split_fingerprint:
        raise ValueError("artifact canonical split fingerprint does not match the session")
    canonical_views, family_views = build_development_views(
        canonical_splits,
        validation_fold=session.validation_fold,
    )
    expected_query_ids = canonical_views.queries["id"].astype(int).tolist()
    prepared = _validate_artifact_inputs(
        frames,
        expected_query_ids=expected_query_ids,
        session=session,
        checkpoint=checkpoint,
        primary_views=canonical_views,
        family_views=family_views,
    )
    canonical_development_rows = int(canonical_splits["partition"].eq("development").sum())
    development_ids = sorted(
        canonical_splits.loc[canonical_splits["partition"].eq("development"), "id"].astype(int)
    )
    reopened_sources = _reopen_source_artifacts(
        source_artifacts,
        session=session,
        checkpoint=checkpoint,
        expected_development_ids=development_ids,
    )
    validated_provenance = {
        source: reopened_sources[source].provenance for source in ("teacher", "v1")
    }
    measured_cost = build_learned_cost_record(
        prepared["timing_summary"],
        caches={source: reopened_sources[source].cache for source in ("teacher", "v1")},
        statistics={source: reopened_sources[source].statistics for source in ("teacher", "v1")},
        session=session,
        checkpoint=checkpoint,
        policy=timing_policy,
        fold=session.validation_fold,
        selected_gallery_policy=selected_gallery_policy,
    )
    validated_cost = _validate_learned_cost(
        measured_cost,
        session=session,
        checkpoint=checkpoint,
        timing_summary=prepared["timing_summary"],
        canonical_development_rows=canonical_development_rows,
        provenance=validated_provenance,
    )

    destination = Path(evidence_dir)
    expected_destination = ROOT / "results/evidence/task4/learned" / run_id
    if destination.resolve() != expected_destination.resolve():
        raise ValueError(
            "learned evidence destination must be results/evidence/task4/learned/<run_id>"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-staging-", dir=destination.parent))
    artifacts: list[dict[str, object]] = []
    try:
        for key, filename in _FRAME_ARTIFACTS.items():
            content = _csv_bytes(prepared[key])
            path = staging / filename
            _atomic_write_bytes(path, content)
            artifacts.append(
                {
                    "name": key,
                    "path": filename,
                    "sha256": _sha256_bytes(content),
                    "rows": len(prepared[key]),
                    "columns": list(prepared[key].columns),
                }
            )
        cost_content = _canonical_json(validated_cost)
        _atomic_write_bytes(staging / "cost.json", cost_content)
        artifacts.append(
            {
                "name": "cost",
                "path": "cost.json",
                "sha256": _sha256_bytes(cost_content),
                "rows": None,
            }
        )
        selected_metrics = summarize_learned_scores(
            pd.read_csv(staging / _FRAME_ARTIFACTS["quality_summary"])
        )
        manifest = {
            "schema_version": LEARNED_EVIDENCE_SCHEMA_VERSION,
            "scope": "development",
            "run_id": run_id,
            "run_kind": run_kind,
            "method": session.expected_registry_identity.method,
            "fold": session.validation_fold,
            "checkpoint": {
                "epoch": checkpoint.epoch,
                "path": str(checkpoint.path),
                "sha256": checkpoint.sha256,
            },
            "config_hash": session.config_hash,
            "split_fingerprint": session.split_fingerprint,
            "source_provenance": {
                source: value.to_dict() for source, value in validated_provenance.items()
            },
            "source_artifacts": {
                source: {
                    name: {
                        "path": str(getattr(source_artifacts[source], name)),
                        "sha256": _sha256_file(getattr(source_artifacts[source], name)),
                    }
                    for name in (
                        "image_cache_manifest",
                        "statistics",
                        "feature_cache_manifest",
                    )
                }
                for source in ("teacher", "v1")
            },
            "contract": _CONTRACT.to_dict(),
            "query_ids": sorted(set(int(value) for value in expected_query_ids)),
            "query_count": len(set(int(value) for value in expected_query_ids)),
            "coverage": {
                "query_metrics_complete": True,
                "rankings_complete": True,
                "timing_complete": True,
            },
            "caveat": caveat,
            "gates": {
                "p95_end_to_end_under_one_second": validated_cost[
                    "p95_end_to_end_under_one_second"
                ],
                "index_under_one_gibibyte": validated_cost["index_under_one_gibibyte"],
            },
            "selected_metrics": selected_metrics,
            "artifacts": artifacts,
        }
        staged_manifest = staging / "manifest.json"
        _atomic_write_bytes(staged_manifest, _canonical_json(manifest))
        validate_learned_manifest(
            staged_manifest,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=canonical_splits,
            _allow_staging=True,
        )
        with _identity_lock(destination.parent, run_id):
            _replace_directory(staging, destination)
        return destination / "manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _gallery_policy_hardware_record(
    native_thread_pools: list[dict[str, object]],
    policy: TimingPolicy,
) -> dict[str, object]:
    return {
        "cpu": platform.processor() or platform.machine() or "unknown",
        "logical_cores": os.cpu_count(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thread_count": policy.thread_count,
        "thread_environment": {name: os.environ.get(name, "") for name in THREAD_VARIABLES},
        "native_thread_pools": native_thread_pools,
    }


def _canonical_policy_query_rows(
    canonical_splits: pd.DataFrame,
    views: RetrievalViews,
    *,
    fold: int,
) -> pd.DataFrame:
    required = {"id", "partition", "cv_fold"}
    if missing := required.difference(canonical_splits.columns):
        raise ValueError(f"gallery policy timing split rows are missing columns: {sorted(missing)}")
    query_ids = pd.to_numeric(views.queries["id"], errors="coerce")
    if query_ids.isna().any() or not query_ids.mod(1).eq(0).all():
        raise ValueError("gallery policy timing query IDs must be integer-compatible")
    wanted = sorted(query_ids.astype(int).tolist())
    rows = canonical_splits.loc[canonical_splits["id"].astype(int).isin(wanted)].copy()
    row_ids = pd.to_numeric(rows["id"], errors="coerce")
    if (
        row_ids.isna().any()
        or not row_ids.mod(1).eq(0).all()
        or sorted(row_ids.astype(int).tolist()) != wanted
        or not rows["partition"].eq("development").all()
        or not rows["cv_fold"].astype(int).eq(fold).all()
    ):
        raise ValueError(
            "gallery policy timing rows must be the canonical development fold queries"
        )
    rows["id"] = row_ids.astype(np.int64)
    return rows.sort_values("id", kind="mergesort").reset_index(drop=True)


def _build_cached_query_encoder(
    *,
    source: str,
    chain: ReopenedSourceArtifactChain,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> Callable[[pd.Series], np.ndarray]:
    _validate_checkpoint_identity(checkpoint, session)
    if source not in {"teacher", "v1"} or chain.cache.manifest.get("source") != source:
        raise ValueError("gallery policy timing query source is invalid")
    mean, std = _validate_source_statistics(chain.cache, chain.statistics, session)
    positions = {int(product_id): position for position, product_id in enumerate(chain.cache.ids)}
    model: nn.Module | None = None

    def encode(row: pd.Series) -> np.ndarray:
        nonlocal model
        if row.get("partition") != "development":
            raise ValueError("gallery policy timing accepts development rows only")
        query_id = pd.to_numeric(pd.Series([row["id"]]), errors="coerce").iloc[0]
        if pd.isna(query_id) or not float(query_id).is_integer():
            raise ValueError("gallery policy timing query ID is malformed")
        product_id = int(query_id)
        if product_id not in positions:
            raise ValueError(f"{source} query cache is missing ID {product_id}")
        if model is None:
            model = _load_evidence_model(checkpoint, session, "cpu")
        position = positions[product_id]
        batch = _normalized_cache_batch(
            np.asarray(chain.cache.images[position : position + 1]),
            np.asarray(chain.cache.content_bounds[position : position + 1]),
            mean,
            std,
        )
        model.eval()
        with torch.inference_mode():
            encoded = model.encode(batch)
        return _validated_embeddings(encoded, 1)[0]

    return encode


def _build_two_view_policy_search(
    *,
    query_source: str,
    views: RetrievalViews,
    indexes: Mapping[str, FeatureIndex],
) -> Callable[[int, np.ndarray], pd.DataFrame]:
    if query_source not in {"teacher", "v1"} or set(indexes) != {"teacher", "v1"}:
        raise ValueError("two-view policy timing requires exact teacher and v1 indexes")
    ordered_queries = _canonical_policy_query_rows(
        views.queries.assign(partition="development", cv_fold=0),
        views,
        fold=0,
    ).set_index("id", drop=False)
    query_template = indexes[query_source]

    def search(query_id: int, feature: np.ndarray) -> pd.DataFrame:
        if query_id not in ordered_queries.index:
            raise ValueError(f"query ID {query_id} is outside the retrieval view")
        one_query = ordered_queries.loc[[query_id]].drop(columns=["partition", "cv_fold"])
        query_index = FeatureIndex(
            source=query_source,
            contract=query_template.contract,
            ids=np.asarray([query_id], dtype=np.int64),
            features=np.asarray(feature, dtype=np.float32).reshape(1, -1),
            transform_seconds=0.0,
            source_bytes=0,
            method=query_template.method,
            fold=query_template.fold,
            checkpoint_fingerprint=query_template.checkpoint_fingerprint,
            config_fingerprint=query_template.config_fingerprint,
            provenance=query_template.provenance,
        )
        return rank_two_view_gallery(
            query_index=query_index,
            gallery_indexes=indexes,
            views=RetrievalViews(queries=one_query, gallery=views.gallery),
            protocol="primary",
            max_k=20,
            chunk_size=1,
        )

    return search


def _benchmark_gallery_policy(
    query_rows: pd.DataFrame,
    *,
    query_source: str,
    gallery_policy: str,
    views: RetrievalViews,
    indexes: Mapping[str, FeatureIndex],
    encode: Callable[[pd.Series], np.ndarray],
    policy: TimingPolicy,
    clock_ns: Callable[[], int],
    fold: int,
) -> pd.DataFrame:
    if query_source not in {"teacher", "v1"} or gallery_policy not in {
        "teacher",
        "v1",
        "two_view",
    }:
        raise ValueError("gallery policy timing source or policy is invalid")
    if gallery_policy == "two_view":
        search = _build_two_view_policy_search(
            query_source=query_source,
            views=views,
            indexes=indexes,
        )
    else:
        search = build_protocol_a_search(
            views=views,
            gallery_index=indexes[gallery_policy],
        )
    ordered = query_rows.sort_values("id", kind="mergesort").reset_index(drop=True)
    for _, row in ordered.iloc[: policy.warmup_queries].iterrows():
        search(int(row["id"]), encode(row))

    records: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        started = clock_ns()
        feature = encode(row)
        encoded = clock_ns()
        search(int(row["id"]), feature)
        searched = clock_ns()
        records.append(
            {
                "scope": "development",
                "fold": fold,
                "query_id": int(row["id"]),
                "query_source": query_source,
                "gallery_policy": gallery_policy,
                "encoding_seconds": (encoded - started) / 1e9,
                "search_seconds": (searched - encoded) / 1e9,
                "end_to_end_seconds": (searched - started) / 1e9,
            }
        )
    return pd.DataFrame.from_records(records)


def _summarize_gallery_policy_timings(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        raise ValueError("gallery policy timing samples must not be empty")
    required = {
        "scope",
        "fold",
        "query_id",
        "query_source",
        "gallery_policy",
        "encoding_seconds",
        "search_seconds",
        "end_to_end_seconds",
    }
    if missing := required.difference(samples.columns):
        raise ValueError(f"gallery policy timing samples are missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for policy_name, rows in samples.groupby("gallery_policy", sort=True):
        for metric, column in (
            ("encoding", "encoding_seconds"),
            ("search", "search_seconds"),
            ("end_to_end", "end_to_end_seconds"),
        ):
            values = pd.to_numeric(rows[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any():
                raise ValueError("gallery policy timing samples must be finite and non-negative")
            for percentile, quantile in (("p50", 0.50), ("p95", 0.95)):
                records.append(
                    {
                        "gallery_policy": str(policy_name),
                        "metric": metric,
                        "percentile": percentile,
                        "value_seconds": float(np.quantile(values, quantile)),
                        "timed_queries": len(rows),
                        "measurement_kind": "measured",
                    }
                )
    return pd.DataFrame.from_records(records)


def write_gallery_policy_timing_artifact(
    destination: str | Path,
    *,
    manifest_path: str | Path,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    canonical_splits: pd.DataFrame,
    policy: TimingPolicy = TimingPolicy(),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> Path:
    """Measure and atomically write trusted gallery-policy timing evidence."""

    _validate_checkpoint_identity(checkpoint, session)
    validate_split_structure(canonical_splits)
    if cv_assignment_digest(canonical_splits) != session.split_fingerprint:
        raise ValueError("gallery policy timing split fingerprint does not match the session")
    manifest = validate_learned_manifest(
        manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=canonical_splits,
    )
    source_artifact_records = manifest["source_artifacts"]
    artifacts = {
        source: LearnedSourceArtifacts(
            image_cache_manifest=Path(
                source_artifact_records[source]["image_cache_manifest"]["path"]
            ),
            statistics=Path(source_artifact_records[source]["statistics"]["path"]),
            feature_cache_manifest=Path(
                source_artifact_records[source]["feature_cache_manifest"]["path"]
            ),
        )
        for source in ("teacher", "v1")
    }
    development_ids = sorted(
        canonical_splits.loc[canonical_splits["partition"].eq("development"), "id"].astype(int)
    )
    reopened = _reopen_source_artifacts(
        artifacts,
        session=session,
        checkpoint=checkpoint,
        expected_development_ids=development_ids,
    )
    indexes = {source: reopened[source].index for source in ("teacher", "v1")}
    primary_views, _ = build_development_views(
        canonical_splits,
        validation_fold=session.validation_fold,
    )
    query_rows = _canonical_policy_query_rows(
        canonical_splits,
        primary_views,
        fold=session.validation_fold,
    )
    encoders = {
        source: _build_cached_query_encoder(
            source=source,
            chain=reopened[source],
            session=session,
            checkpoint=checkpoint,
        )
        for source in ("teacher", "v1")
    }

    with _single_threaded_cpu() as native_thread_pools:
        sample_frames = [
            _benchmark_gallery_policy(
                query_rows,
                query_source=query_source,
                gallery_policy=gallery_policy,
                views=primary_views,
                indexes=indexes,
                encode=encoders[query_source],
                policy=policy,
                clock_ns=clock_ns,
                fold=session.validation_fold,
            )
            for gallery_policy in ("teacher", "v1", "two_view")
            for query_source in ("teacher", "v1")
        ]
        timing_samples = (
            pd.concat(sample_frames, ignore_index=True)
            .sort_values(
                ["gallery_policy", "query_source", "query_id"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        timing_summary = _summarize_gallery_policy_timings(timing_samples)
        timed_queries = int(timing_samples.groupby("gallery_policy").size().iloc[0])
        record = {
            "schema_version": 1,
            "scope": "development",
            "run_id": session.run_id,
            "run_kind": session.run_kind,
            "method": session.expected_registry_identity.method,
            "fold": session.validation_fold,
            "checkpoint_sha256": checkpoint.sha256,
            "config_hash": session.config_hash,
            "split_fingerprint": session.split_fingerprint,
            "measurement_route": GALLERY_POLICY_TIMING_ROUTE,
            "measurement_kind": "measured",
            "hardware": _gallery_policy_hardware_record(native_thread_pools, policy),
            "warmup_queries": policy.warmup_queries,
            "timed_queries": timed_queries,
            "timing_samples": timing_samples.to_dict("records"),
            "timing_summary": timing_summary.to_dict("records"),
        }
        record["measurement_sha256"] = _sha256_bytes(_canonical_json(record))
    path = Path(destination)
    content = _canonical_json(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".staging",
            delete=False,
        ) as handle:
            staging_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        validate_gallery_policy_timing_artifact(
            staging_path,
            session=session,
            checkpoint=checkpoint,
        )
        os.replace(staging_path, path)
        staging_path = None
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
    return path


def validate_gallery_policy_timing_artifact(
    artifact_path: str | Path,
    *,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> dict[str, float]:
    """Validate measured teacher, V1, and two-view gallery-policy CPU p95 values."""

    _validate_checkpoint_identity(checkpoint, session)
    path = Path(artifact_path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"gallery policy timing artifact cannot be read: {error}") from error
    expected_fields = {
        "schema_version",
        "scope",
        "run_id",
        "run_kind",
        "method",
        "fold",
        "checkpoint_sha256",
        "config_hash",
        "split_fingerprint",
        "measurement_route",
        "measurement_kind",
        "hardware",
        "warmup_queries",
        "timed_queries",
        "timing_samples",
        "timing_summary",
        "measurement_sha256",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise ValueError("gallery policy timing artifact schema is invalid")
    expected_identity = {
        "schema_version": 1,
        "scope": "development",
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "method": session.expected_registry_identity.method,
        "fold": session.validation_fold,
        "checkpoint_sha256": checkpoint.sha256,
        "config_hash": session.config_hash,
        "split_fingerprint": session.split_fingerprint,
        "measurement_route": GALLERY_POLICY_TIMING_ROUTE,
        "measurement_kind": "measured",
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("gallery policy timing identity or measured route is invalid")
    measured_sha256 = record["measurement_sha256"]
    unsigned = dict(record)
    del unsigned["measurement_sha256"]
    if not isinstance(measured_sha256, str) or measured_sha256 != _sha256_bytes(
        _canonical_json(unsigned)
    ):
        raise ValueError("gallery policy timing measurement SHA-256 is invalid")
    hardware = record["hardware"]
    expected_hardware_fields = {
        "cpu",
        "logical_cores",
        "operating_system",
        "python_version",
        "numpy_version",
        "thread_count",
        "thread_environment",
        "native_thread_pools",
    }
    native_pools = hardware.get("native_thread_pools") if isinstance(hardware, dict) else None
    if (
        not isinstance(hardware, dict)
        or set(hardware) != expected_hardware_fields
        or not isinstance(hardware["cpu"], str)
        or not hardware["cpu"].strip()
        or not isinstance(hardware["operating_system"], str)
        or not hardware["operating_system"].strip()
        or hardware["thread_count"] != 1
        or hardware["thread_environment"] != {name: "1" for name in THREAD_VARIABLES}
        or not isinstance(native_pools, list)
        or not native_pools
        or any(
            not isinstance(pool, dict) or int(pool.get("num_threads", 0)) != 1
            for pool in native_pools
        )
    ):
        raise ValueError("gallery policy timing CPU identity is invalid")
    warmup_queries = record["warmup_queries"]
    if (
        isinstance(warmup_queries, bool)
        or not isinstance(warmup_queries, Integral)
        or warmup_queries < 0
    ):
        raise ValueError("gallery policy timing warmup_queries must be non-negative")
    timed_queries = record["timed_queries"]
    if (
        isinstance(timed_queries, bool)
        or not isinstance(timed_queries, Integral)
        or timed_queries <= 0
    ):
        raise ValueError("gallery policy timing timed_queries must be positive")
    samples = record["timing_samples"]
    sample_fields = {
        "scope",
        "fold",
        "query_id",
        "query_source",
        "gallery_policy",
        "encoding_seconds",
        "search_seconds",
        "end_to_end_seconds",
    }
    if (
        not isinstance(samples, list)
        or not samples
        or any(not isinstance(row, dict) or set(row) != sample_fields for row in samples)
    ):
        raise ValueError("gallery policy timing samples are invalid")
    sample_frame = pd.DataFrame.from_records(samples)
    if (
        set(sample_frame["scope"].astype(str)) != {"development"}
        or set(sample_frame["fold"].astype(int)) != {session.validation_fold}
        or set(sample_frame["query_source"].astype(str)) != {"teacher", "v1"}
        or set(sample_frame["gallery_policy"].astype(str)) != {"teacher", "v1", "two_view"}
    ):
        raise ValueError("gallery policy timing samples do not match the measured route")
    sample_counts = sample_frame.groupby("gallery_policy").size()
    if len(sample_counts) != 3 or not sample_counts.eq(int(timed_queries)).all():
        raise ValueError("gallery policy timing samples must share one timed-query count")
    per_query_counts = sample_frame.groupby(["gallery_policy", "query_source", "query_id"]).size()
    if not per_query_counts.eq(1).all():
        raise ValueError("gallery policy timing must measure each query once per policy/source")
    numeric_sample_columns = ["encoding_seconds", "search_seconds", "end_to_end_seconds"]
    numeric_samples = sample_frame[numeric_sample_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric_samples.isna().any().any()
        or not np.isfinite(numeric_samples.to_numpy()).all()
        or numeric_samples.lt(0).any().any()
    ):
        raise ValueError("gallery policy timing samples must be finite and non-negative")
    for column in numeric_sample_columns:
        sample_frame[column] = numeric_samples[column].astype(float)
    timing_summary = record["timing_summary"]
    if not isinstance(timing_summary, list):
        raise ValueError("gallery policy timing summary must be records")
    expected_row_fields = {
        "gallery_policy",
        "metric",
        "percentile",
        "value_seconds",
        "timed_queries",
        "measurement_kind",
    }
    expected_summary_keys = {
        (policy, metric, percentile)
        for policy in ("teacher", "v1", "two_view")
        for metric in ("encoding", "search", "end_to_end")
        for percentile in ("p50", "p95")
    }
    observed_summary_keys: set[tuple[str, str, str]] = set()
    values: dict[str, float] = {}
    for row in timing_summary:
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise ValueError("gallery policy timing row schema is invalid")
        policy = str(row["gallery_policy"])
        metric = str(row["metric"])
        percentile = str(row["percentile"])
        key = (policy, metric, percentile)
        value = row["value_seconds"]
        if (
            policy not in {"teacher", "v1", "two_view"}
            or key in observed_summary_keys
            or key not in expected_summary_keys
            or row["measurement_kind"] != "measured"
            or row["timed_queries"] != timed_queries
            or isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("gallery policy timing rows are not exact measured p95 values")
        observed_summary_keys.add(key)
        sample_values = sample_frame.loc[sample_frame["gallery_policy"].eq(policy)]
        column = {
            "encoding": "encoding_seconds",
            "search": "search_seconds",
            "end_to_end": "end_to_end_seconds",
        }[metric]
        quantile = 0.50 if percentile == "p50" else 0.95
        expected_value = float(np.quantile(sample_values[column], quantile))
        if not math.isclose(float(value), expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("gallery policy timing summary disagrees with measured samples")
        if metric == "end_to_end" and percentile == "p95":
            values[policy] = float(value)
    if observed_summary_keys != expected_summary_keys or set(values) != {
        "teacher",
        "v1",
        "two_view",
    }:
        raise ValueError("gallery policy timing requires exact measured policies")
    return values


def validate_learned_manifest(
    manifest_path: str | Path,
    *,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    canonical_splits: pd.DataFrame,
    _allow_staging: bool = False,
) -> dict[str, object]:
    """Reopen one final manifest and validate its identity, coverage, and artifacts."""

    _validate_checkpoint_identity(checkpoint, session)
    validate_split_structure(canonical_splits)
    if cv_assignment_digest(canonical_splits) != session.split_fingerprint:
        raise ValueError("manifest canonical split fingerprint does not match the session")
    path = Path(manifest_path)
    expected_path = ROOT / "results/evidence/task4/learned" / session.run_id / "manifest.json"
    if not _allow_staging and path.resolve() != expected_path.resolve():
        raise ValueError(
            "learned manifest must be results/evidence/task4/learned/<run_id>/manifest.json"
        )
    if not path.is_file():
        raise ValueError("evidence manifest does not exist")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"evidence manifest cannot be read: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("evidence manifest must be a JSON object")
    expected_manifest_fields = {
        "schema_version",
        "scope",
        "run_id",
        "run_kind",
        "method",
        "fold",
        "checkpoint",
        "config_hash",
        "split_fingerprint",
        "source_provenance",
        "source_artifacts",
        "contract",
        "query_ids",
        "query_count",
        "coverage",
        "caveat",
        "gates",
        "selected_metrics",
        "artifacts",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("evidence manifest fields do not match the exact schema")
    expected = {
        "schema_version": LEARNED_EVIDENCE_SCHEMA_VERSION,
        "scope": "development",
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "method": session.expected_registry_identity.method,
        "fold": session.validation_fold,
        "config_hash": session.config_hash,
        "split_fingerprint": session.split_fingerprint,
        "caveat": TEACHER_SLICE_CAVEAT,
        "contract": _CONTRACT.to_dict(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"evidence manifest {key} does not match")
    source_provenance = manifest.get("source_provenance")
    if not isinstance(source_provenance, dict) or set(source_provenance) != {
        "teacher",
        "v1",
    }:
        raise ValueError("evidence manifest source provenance is malformed")
    try:
        parsed_provenance = {
            source: LearnedProvenance(**source_provenance[source]) for source in ("teacher", "v1")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evidence manifest source provenance fields are malformed") from error
    source_artifact_records = manifest.get("source_artifacts")
    artifact_names = {
        "image_cache_manifest",
        "statistics",
        "feature_cache_manifest",
    }
    if (
        not isinstance(source_artifact_records, dict)
        or set(source_artifact_records) != {"teacher", "v1"}
        or any(
            not isinstance(source_artifact_records[source], dict)
            or set(source_artifact_records[source]) != artifact_names
            or any(
                not isinstance(source_artifact_records[source][name], dict)
                or set(source_artifact_records[source][name]) != {"path", "sha256"}
                for name in artifact_names
            )
            for source in ("teacher", "v1")
        )
    ):
        raise ValueError("evidence manifest source artifact provenance is malformed")
    reopened_paths = {
        source: LearnedSourceArtifacts(
            **{name: Path(source_artifact_records[source][name]["path"]) for name in artifact_names}
        )
        for source in ("teacher", "v1")
    }
    for source in ("teacher", "v1"):
        for name in artifact_names:
            record = source_artifact_records[source][name]
            artifact_path = getattr(reopened_paths[source], name)
            if not artifact_path.is_file() or record["sha256"] != _sha256_file(artifact_path):
                raise ValueError("evidence manifest source artifact provenance hash disagrees")
    development_ids = sorted(
        canonical_splits.loc[canonical_splits["partition"].eq("development"), "id"].astype(int)
    )
    reopened_sources = _reopen_source_artifacts(
        reopened_paths,
        session=session,
        checkpoint=checkpoint,
        expected_development_ids=development_ids,
    )
    reopened_provenance = {
        source: reopened_sources[source].provenance for source in ("teacher", "v1")
    }
    if reopened_provenance != parsed_provenance:
        raise ValueError("evidence manifest reopened source provenance disagrees")
    if parsed_provenance["teacher"].model_identity != parsed_provenance[
        "v1"
    ].model_identity or parsed_provenance["teacher"].model_identity != (
        LEARNED_EVIDENCE_SCHEMA_VERSION,
        session.run_id,
        session.run_kind,
        session.expected_registry_identity.method,
        session.validation_fold,
        checkpoint.sha256,
        session.config_hash,
        session.split_fingerprint,
    ):
        raise ValueError("evidence manifest source provenance model identity disagrees")
    stored_checkpoint = manifest.get("checkpoint")
    expected_checkpoint = {
        "epoch": checkpoint.epoch,
        "path": str(checkpoint.path),
        "sha256": checkpoint.sha256,
    }
    if not isinstance(stored_checkpoint, dict) or stored_checkpoint != expected_checkpoint:
        raise ValueError("evidence manifest checkpoint identity does not match")
    query_count = manifest.get("query_count")
    if (
        isinstance(query_count, bool)
        or not isinstance(query_count, Integral)
        or int(query_count) <= 0
    ):
        raise ValueError("evidence manifest query count must be positive")
    query_ids = manifest.get("query_ids")
    if (
        not isinstance(query_ids, list)
        or len(query_ids) != int(query_count)
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in query_ids)
        or query_ids != sorted(set(int(value) for value in query_ids))
    ):
        raise ValueError("evidence manifest query IDs are malformed")
    canonical_views, _ = build_development_views(
        canonical_splits,
        validation_fold=session.validation_fold,
    )
    canonical_query_ids = sorted(canonical_views.queries["id"].astype(int).tolist())
    if query_ids != canonical_query_ids:
        raise ValueError("evidence manifest query IDs do not match the canonical fold")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict) or coverage != {
        "query_metrics_complete": True,
        "rankings_complete": True,
        "timing_complete": True,
    }:
        raise ValueError("evidence manifest coverage must be complete")
    gates = manifest.get("gates")
    if (
        not isinstance(gates, dict)
        or type(gates.get("p95_end_to_end_under_one_second")) is not bool
        or type(gates.get("index_under_one_gibibyte")) is not bool
    ):
        raise ValueError("evidence manifest gates are malformed")
    selected = manifest.get("selected_metrics")
    if not isinstance(selected, dict) or set(selected) != {
        "development_winner_score",
        "cross_source_score",
        "source_robustness_ratio",
    }:
        raise ValueError("evidence manifest selected metrics are malformed")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
        for value in selected.values()
    ):
        raise ValueError("evidence manifest selected metrics must be finite")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("evidence manifest artifacts must be non-empty")
    names: set[str] = set()
    relative_paths: set[str] = set()
    reopened_frames: dict[str, pd.DataFrame] = {}
    reopened_cost: dict[str, object] | None = None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("evidence manifest artifact records must be objects")
        name = artifact.get("name")
        relative = artifact.get("path")
        digest = artifact.get("sha256")
        expected_artifact_fields = (
            {"name", "path", "sha256", "rows"}
            if name == "cost"
            else {"name", "path", "sha256", "rows", "columns"}
        )
        if set(artifact) != expected_artifact_fields:
            raise ValueError("evidence manifest artifact record fields are malformed")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(relative, str)
            or not relative
            or relative in relative_paths
        ):
            raise ValueError("evidence manifest artifact names and paths must be unique")
        artifact_path = Path(relative)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ValueError("evidence manifest artifact paths must be relative")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("evidence manifest artifact SHA-256 is malformed")
        resolved = path.parent / artifact_path
        if not resolved.is_file() or _sha256_file(resolved) != digest:
            raise ValueError(f"evidence artifact hash validation failed: {relative}")
        rows = artifact.get("rows")
        if rows is not None:
            if (
                isinstance(rows, bool)
                or not isinstance(rows, Integral)
                or int(rows) < 0
                or resolved.suffix != ".csv"
                or len(pd.read_csv(resolved)) != int(rows)
            ):
                raise ValueError(f"evidence artifact row count is invalid: {relative}")
            frame = pd.read_csv(resolved)
            expected_columns = LEARNED_ARTIFACT_SCHEMAS.get(str(name))
            if expected_columns is None or tuple(frame.columns) != expected_columns:
                raise ValueError(f"evidence artifact columns are invalid: {relative}")
            if artifact.get("columns") != list(expected_columns):
                raise ValueError(f"evidence manifest columns disagree: {relative}")
            reopened_frames[str(name)] = frame
        elif name == "cost":
            try:
                raw_cost = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("learned cost artifact cannot be read") from error
            if not isinstance(raw_cost, dict):
                raise ValueError("learned cost artifact must be an object")
            reopened_cost = raw_cost
        names.add(name)
        relative_paths.add(relative)
    if names != {*_FRAME_ARTIFACTS, "cost"}:
        raise ValueError("evidence manifest is missing required artifact names")
    canonical_views, family_views = build_development_views(
        canonical_splits,
        validation_fold=session.validation_fold,
    )
    validated_frames = _validate_artifact_inputs(
        reopened_frames,
        expected_query_ids=query_ids,
        session=session,
        checkpoint=checkpoint,
        primary_views=canonical_views,
        family_views=family_views,
    )
    if reopened_cost is None:
        raise ValueError("evidence manifest is missing learned cost")
    validated_cost = _validate_learned_cost(
        reopened_cost,
        session=session,
        checkpoint=checkpoint,
        timing_summary=validated_frames["timing_summary"],
        canonical_development_rows=int(canonical_splits["partition"].eq("development").sum()),
        provenance=parsed_provenance,
    )
    derived_metrics = summarize_learned_scores(validated_frames["quality_summary"])
    if any(
        not math.isclose(
            float(selected[key]),
            float(derived_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        for key in derived_metrics
    ):
        raise ValueError("manifest selected metrics disagree with quality evidence")
    expected_gates = {
        "p95_end_to_end_under_one_second": validated_cost["p95_end_to_end_under_one_second"],
        "index_under_one_gibibyte": validated_cost["index_under_one_gibibyte"],
    }
    if gates != expected_gates:
        raise ValueError("manifest gates disagree with learned cost evidence")
    return manifest


def _validate_registry_identity(
    session: TrainingSessionConfig,
    row: Mapping[str, object],
) -> None:
    for field_name, expected in session.expected_registry_identity.as_dict().items():
        actual = row.get(field_name, "")
        normalized_expected = str(expected).lower() if isinstance(expected, bool) else str(expected)
        if str(actual) != normalized_expected:
            raise ValueError(
                f"registry {field_name.replace('_', ' ')} does not match training session"
            )


def _load_checkpoint_record(
    path: Path,
    session: TrainingSessionConfig,
) -> CheckpointRecord:
    digest = _sha256_file(path)
    loaded = load_checkpoint(
        path,
        expected_sha256=digest,
        expected_config_hash=session.config_hash,
        expected_split_fingerprint=session.split_fingerprint,
        expected_weight_origin=session.model_metadata.weight_origin,
        expected_parent_run_id=session.parent_run_id,
        expected_run_id=session.run_id,
        expected_run_kind=session.run_kind,
        map_location="cpu",
    )
    return CheckpointRecord(
        epoch=int(loaded.epoch),
        path=path,
        sha256=str(loaded.sha256),
        config_hash=session.config_hash,
        score=float(loaded.score),
        split_fingerprint=session.split_fingerprint,
        weight_origin=session.model_metadata.weight_origin,
        parent_run_id=session.parent_run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
    )


def reconstruct_training_result(
    checkpoint_paths: Sequence[str | Path],
    *,
    session: TrainingSessionConfig,
    registry_row: Mapping[str, object],
    allow_failed_evidence: bool = False,
    recoverable_error_type: str = "ValueError",
    recoverable_error_message: str = "query metric values must be null or finite in [0, 1]",
) -> TrainingResult:
    """Reconstruct a validated handoff from a known complete milestone path list."""

    _validate_registry_identity(session, registry_row)
    allowed_statuses = {"running", "failed"} if allow_failed_evidence else {"running"}
    if registry_row.get("status") not in allowed_statuses:
        raise ValueError("evidence reconstruction requires a running registry row")
    if allow_failed_evidence and registry_row.get("status") == "failed":
        if (
            registry_row.get("error_type") != recoverable_error_type
            or registry_row.get("error_message") != recoverable_error_message
        ):
            raise ValueError("evidence recovery requires the exact failed artifact-write row")
    paths = [Path(path) for path in checkpoint_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("checkpoint reconstruction paths must be unique and non-empty")
    records = tuple(
        sorted(
            (_load_checkpoint_record(path, session) for path in paths),
            key=lambda record: record.epoch,
        )
    )
    if tuple(record.epoch for record in records) != session.hyperparameters.checkpoint_epochs:
        raise ValueError("checkpoint reconstruction requires the complete milestone set")
    return TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=records,
        best_checkpoint=select_best_checkpoint(records),
    )


def _validate_training_result(
    result: TrainingResult,
    session: TrainingSessionConfig,
    registry_row: Mapping[str, object],
) -> None:
    _validate_registry_identity(session, registry_row)
    if result.run_id != session.run_id:
        raise ValueError("training result run ID does not match the session")
    if result.run_kind != session.run_kind:
        raise ValueError("training result run kind does not match the session")
    if tuple(record.epoch for record in result.checkpoints) != (
        session.hyperparameters.checkpoint_epochs
    ):
        raise ValueError("training result must contain the complete milestone set")
    for record in result.checkpoints:
        _validate_checkpoint_identity(record, session)
        loaded = load_checkpoint(
            record.path,
            expected_sha256=record.sha256,
            expected_config_hash=session.config_hash,
            expected_split_fingerprint=session.split_fingerprint,
            expected_weight_origin=session.model_metadata.weight_origin,
            expected_parent_run_id=session.parent_run_id,
            expected_run_id=session.run_id,
            expected_run_kind=session.run_kind,
            map_location="cpu",
        )
        if int(loaded.epoch) != record.epoch or float(loaded.score) != float(record.score):
            raise ValueError("checkpoint payload epoch or score does not match its record")
        del loaded
    if result.best_checkpoint != select_best_checkpoint(result.checkpoints):
        raise ValueError("training result selected checkpoint is not the best milestone")


def _progress(message: str, **fields: object) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"task4-progress stage={message} {details}".strip(), flush=True)


def _recomputed_primary_scorable_queries(
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
    *,
    total_queries: int,
) -> int:
    """Recompute the frozen evaluator's scorable Protocol A query count at K=10."""

    coverage = compute_relevance_coverage(primary_views, family_views, k_values=(10,))
    rows = coverage.loc[coverage["protocol"].eq("primary") & coverage["k"].eq(10)]
    if len(rows) != 1:
        raise ValueError("stability evidence requires one recomputed primary coverage row")
    recomputed_total = int(rows["total_queries"].iloc[0])
    scorable = int(rows["scored_queries"].iloc[0])
    if recomputed_total != total_queries or scorable <= 0 or scorable > total_queries:
        raise ValueError("stability evidence recomputed primary coverage is malformed")
    return scorable


def _stability_primary_coverage(
    quality: LearnedQualityEvaluation,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
) -> dict[str, float]:
    """Measure primary nDCG@10 coverage against the recomputed scorable query count.

    ``query_count`` counts the queries with a defined metric, so a fold with legitimately
    undefined queries reports fewer than the total canonical queries. Coverage is complete
    when every direction agrees with the frozen evaluator's independently recomputed
    scorable count for the same fold.
    """

    primary = quality.summary.loc[
        quality.summary["protocol"].eq("primary")
        & quality.summary["metric"].eq("ndcg")
        & pd.to_numeric(quality.summary["k"], errors="coerce").eq(10)
        & quality.summary["aggregation"].eq("query_mean")
    ].copy()
    observed_directions = set(
        primary[["query_source", "gallery_source"]].itertuples(index=False, name=None)
    )
    if len(primary) != 4 or observed_directions != set(source_directions()):
        raise ValueError("stability evidence requires four primary nDCG@10 rows")
    total_queries = len(primary_views.queries)
    scorable_queries = _recomputed_primary_scorable_queries(
        primary_views,
        family_views,
        total_queries=total_queries,
    )
    counts = pd.to_numeric(primary["query_count"], errors="coerce")
    if (
        counts.isna().any()
        or not counts.mod(1).eq(0).all()
        or not counts.eq(scorable_queries).all()
    ):
        raise ValueError("stability evidence primary coverage is incomplete")
    return {
        "total_query_count": total_queries,
        "scorable_query_count": scorable_queries,
        "primary_coverage": scorable_queries / total_queries,
    }


def validate_stability_evidence_artifact(
    artifact_path: str | Path,
    *,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    canonical_splits: pd.DataFrame,
) -> dict[str, object]:
    """Validate the lightweight stability-only evidence artifact."""

    path = Path(artifact_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("stability evidence artifact cannot be reopened") from error
    if payload.get("schema_version") != LEARNED_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("stability evidence schema version is invalid")
    if payload.get("artifact_type") != "task4_stability_evidence":
        raise ValueError("stability evidence artifact type is invalid")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("stability evidence identity is malformed")
    expected_identity = {
        "run_id": session.run_id,
        "run_kind": "stability",
        "method": session.candidate.candidate,
        "fold": session.validation_fold,
        "config_hash": session.config_hash,
        "split_fingerprint": cv_assignment_digest(canonical_splits),
        "checkpoint_sha256": checkpoint.sha256,
        "parent_run_id": session.parent_run_id,
    }
    if dict(identity) != expected_identity:
        raise ValueError("stability evidence identity does not match session")
    selected = payload.get("selected_metrics")
    coverage = payload.get("coverage")
    if not isinstance(selected, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("stability evidence selected metrics or coverage is malformed")
    score = selected.get("development_winner_score")
    if not isinstance(score, Real) or not math.isfinite(float(score)):
        raise ValueError("stability evidence primary score is invalid")
    if set(coverage) != {"total_query_count", "scorable_query_count", "primary_coverage"}:
        raise ValueError("stability evidence coverage fields do not match the exact schema")
    total_queries = coverage["total_query_count"]
    scorable_queries = coverage["scorable_query_count"]
    primary_coverage = coverage["primary_coverage"]
    if (
        isinstance(total_queries, bool)
        or isinstance(scorable_queries, bool)
        or not isinstance(total_queries, Integral)
        or not isinstance(scorable_queries, Integral)
        or not isinstance(primary_coverage, Real)
        or not math.isfinite(float(primary_coverage))
    ):
        raise ValueError("stability evidence coverage counts are malformed")
    primary_views, family_views = build_development_views(
        canonical_splits,
        validation_fold=session.validation_fold,
    )
    expected_total = len(primary_views.queries)
    expected_scorable = _recomputed_primary_scorable_queries(
        primary_views,
        family_views,
        total_queries=expected_total,
    )
    if int(total_queries) != expected_total or int(scorable_queries) != expected_scorable:
        raise ValueError("stability evidence coverage counts do not match the canonical fold")
    if float(primary_coverage) != expected_scorable / expected_total:
        raise ValueError("stability evidence coverage is inconsistent with its counts")
    return payload


def build_stability_evidence(
    registry: RunRegistry,
    *,
    result: TrainingResult,
    session: TrainingSessionConfig,
    splits: pd.DataFrame,
    caches: Mapping[str, DevelopmentImageCache],
    statistics: Mapping[str, Mapping[str, object]],
    statistics_paths: Mapping[str, str | Path],
    feature_cache_root: str | Path,
    evidence_root: str | Path,
    completed_at: str | None = None,
    device: torch.device | str = "cpu",
    recover_failed_evidence: bool = False,
    recovery_error_type: str = "",
    recovery_error_message: str = "",
) -> StabilityEvidenceResult:
    """Build only primary-score and coverage evidence for fresh stability runs."""

    matching = [row for row in registry.read() if row["run_id"] == session.run_id]
    allowed_statuses = {"running", "failed"} if recover_failed_evidence else {"running"}
    if len(matching) != 1 or matching[0]["status"] not in allowed_statuses:
        raise ValueError("stability evidence requires one running registry row")
    if recover_failed_evidence and matching[0]["status"] == "failed":
        if (
            matching[0].get("error_type") != recovery_error_type
            or matching[0].get("error_message") != recovery_error_message
        ):
            raise ValueError("stability recovery requires the exact failed evidence row")
        if str(matching[0].get("evidence_manifest_path") or "").strip():
            raise ValueError("stability recovery requires a row with no completed evidence")
    if session.run_kind != "stability":
        raise ValueError("lightweight stability evidence requires a stability session")
    _validate_training_result(result, session, matching[0])
    validate_split_structure(splits)
    if cv_assignment_digest(splits) != session.split_fingerprint:
        raise ValueError("canonical split fingerprint does not match the training session")
    if set(caches) != {"teacher", "v1"} or set(statistics) != {"teacher", "v1"}:
        raise ValueError("stability evidence requires teacher and v1 caches/statistics")
    if set(statistics_paths) != {"teacher", "v1"}:
        raise ValueError("stability evidence requires teacher and v1 statistics paths")
    if Path(evidence_root).resolve() != (ROOT / "results/evidence/task4").resolve():
        raise ValueError("evidence root must be the repository results/evidence/task4 directory")

    primary_views, family_views = build_development_views(
        splits,
        validation_fold=session.validation_fold,
    )
    checkpoint = result.best_checkpoint
    _progress("stability-evidence", run_id=session.run_id, event="feature-cache", device=device)
    cached_indexes = {
        source: ensure_learned_feature_index(
            cache=caches[source],
            statistics=statistics[source],
            session=session,
            checkpoint=checkpoint,
            cache_root=feature_cache_root,
            device=device,
        )
        for source in ("teacher", "v1")
    }
    indexes = {source: cached_indexes[source].index for source in ("teacher", "v1")}
    _progress("stability-evidence", run_id=session.run_id, event="primary-score")
    quality = evaluate_learned_quality(
        splits,
        indexes,
        fold=session.validation_fold,
    )
    selected = summarize_learned_scores(quality.summary)
    coverage = _stability_primary_coverage(quality, primary_views, family_views)
    evidence_dir = Path(evidence_root) / "stability" / session.run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence_dir / "stability_evidence.json"
    payload = {
        "schema_version": LEARNED_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": "task4_stability_evidence",
        "identity": {
            "run_id": session.run_id,
            "run_kind": session.run_kind,
            "method": session.candidate.candidate,
            "fold": session.validation_fold,
            "config_hash": session.config_hash,
            "split_fingerprint": session.split_fingerprint,
            "checkpoint_sha256": checkpoint.sha256,
            "parent_run_id": session.parent_run_id,
        },
        "checkpoint": {
            "epoch": checkpoint.epoch,
            "path": str(checkpoint.path),
            "score": checkpoint.score,
        },
        "selected_metrics": {
            "development_winner_score": selected["development_winner_score"],
        },
        "coverage": {
            "total_query_count": coverage["total_query_count"],
            "scorable_query_count": coverage["scorable_query_count"],
            "primary_coverage": coverage["primary_coverage"],
        },
        "source_feature_manifests": {
            source: str(cached_indexes[source].cache_dir / "manifest.json")
            for source in ("teacher", "v1")
        },
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_bytes(_canonical_json(payload))
    os.replace(temporary, manifest_path)
    validate_stability_evidence_artifact(
        manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )
    updates = {
        "selected_epoch": checkpoint.epoch,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.sha256,
        "development_winner_score": selected["development_winner_score"],
        "evidence_manifest_path": str(manifest_path),
        "completed_at_utc": completed_at,
        "status": "completed",
    }
    if recover_failed_evidence and matching[0]["status"] == "failed":
        row = registry.recover_failed_evidence(
            session.run_id,
            updates,
            expected_error_type=recovery_error_type,
            expected_error_message=recovery_error_message,
        )
    else:
        row = registry.update(session.run_id, updates)
    return StabilityEvidenceResult(
        manifest_path=manifest_path,
        registry_row=row,
        development_winner_score=float(selected["development_winner_score"]),
        total_query_count=int(coverage["total_query_count"]),
        scorable_query_count=int(coverage["scorable_query_count"]),
        primary_coverage=float(coverage["primary_coverage"]),
    )


def build_learned_evidence(
    registry: RunRegistry,
    *,
    result: TrainingResult,
    session: TrainingSessionConfig,
    splits: pd.DataFrame,
    caches: Mapping[str, DevelopmentImageCache],
    statistics: Mapping[str, Mapping[str, object]],
    statistics_paths: Mapping[str, str | Path],
    query_rows: pd.DataFrame,
    path_columns: Mapping[str, str],
    feature_cache_root: str | Path,
    evidence_root: str | Path,
    selected_gallery_policy: Literal["teacher", "v1", "two_view"],
    completed_at: str,
    timing_policy: TimingPolicy = TimingPolicy(),
    device: torch.device | str = "cpu",
    recover_failed_evidence: bool = False,
    recovery_error_type: str = "",
    recovery_error_message: str = "",
) -> LearnedEvidenceResult:
    """Build the full selected-checkpoint package and complete the registry last."""

    matching = [row for row in registry.read() if row["run_id"] == session.run_id]
    allowed_statuses = {"running", "failed"} if recover_failed_evidence else {"running"}
    if len(matching) != 1 or matching[0]["status"] not in allowed_statuses:
        raise ValueError("evidence builder requires one running registry row")
    if recover_failed_evidence and matching[0]["status"] == "failed":
        if (
            matching[0].get("error_type") != recovery_error_type
            or matching[0].get("error_message") != recovery_error_message
        ):
            raise ValueError("evidence recovery requires the exact failed artifact-write row")
    _validate_training_result(result, session, matching[0])
    validate_split_structure(splits)
    if cv_assignment_digest(splits) != session.split_fingerprint:
        raise ValueError("canonical split fingerprint does not match the training session")
    if (
        set(caches) != {"teacher", "v1"}
        or set(statistics) != {"teacher", "v1"}
        or set(statistics_paths) != {"teacher", "v1"}
    ):
        raise ValueError("evidence builder requires teacher and v1 caches/statistics")
    if set(path_columns) != {"teacher", "v1"}:
        raise ValueError("evidence builder requires teacher and v1 path columns")
    if Path(evidence_root).resolve() != (ROOT / "results/evidence/task4").resolve():
        raise ValueError("evidence root must be the repository results/evidence/task4 directory")

    development = splits.loc[splits["partition"].eq("development")]
    development_ids = sorted(development["id"].astype(int).tolist())
    for source in ("teacher", "v1"):
        if sorted(int(value) for value in caches[source].ids) != development_ids:
            raise ValueError(f"{source} image cache must cover canonical development IDs")
        _validate_source_statistics(caches[source], statistics[source], session)
    primary_views, family_views = build_development_views(
        splits,
        validation_fold=session.validation_fold,
    )
    canonical_query_ids = sorted(primary_views.queries["id"].astype(int).tolist())
    if "partition" not in query_rows or set(query_rows["partition"].astype(str)) != {"development"}:
        raise ValueError("evidence query rows must be development only")
    query_ids = pd.to_numeric(query_rows["id"], errors="coerce")
    if (
        query_ids.isna().any()
        or not query_ids.mod(1).eq(0).all()
        or sorted(query_ids.astype(int).tolist()) != canonical_query_ids
    ):
        raise ValueError("evidence query rows must equal the canonical fold queries")
    for source, column in path_columns.items():
        if column not in query_rows or query_rows[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{source} query path column is missing or blank")

    checkpoint = result.best_checkpoint
    _progress("full-evidence", run_id=session.run_id, event="feature-cache", device=device)
    cached_indexes = {
        source: ensure_learned_feature_index(
            cache=caches[source],
            statistics=statistics[source],
            session=session,
            checkpoint=checkpoint,
            cache_root=feature_cache_root,
            device=device,
        )
        for source in ("teacher", "v1")
    }
    indexes = {source: cached_indexes[source].index for source in ("teacher", "v1")}
    _progress("full-evidence", run_id=session.run_id, event="quality")
    quality = evaluate_learned_quality(
        splits,
        indexes,
        fold=session.validation_fold,
    )
    _progress("full-evidence", run_id=session.run_id, event="canvas")
    canvas_indexes = build_learned_canvas_indexes(
        caches["v1"],
        statistics=statistics["v1"],
        session=session,
        checkpoint=checkpoint,
        query_ids=canonical_query_ids,
        device=device,
    )
    analysis = evaluate_learned_analysis(
        quality,
        primary_views=primary_views,
        family_views=family_views,
        canvas_indexes=canvas_indexes,
        gallery_index=indexes["v1"],
        fold=session.validation_fold,
    )
    _progress("full-evidence", run_id=session.run_id, event="gallery")
    gallery = evaluate_gallery_sources(
        splits,
        indexes,
        fold=session.validation_fold,
    )
    encoders = {
        source: LearnedTimingEncoder(
            extractor=LazyCPUQueryExtractor(
                source=source,
                checkpoint=checkpoint,
                session=session,
                statistics=statistics[source],
                path_column=path_columns[source],
            ),
            feature_manifest=cached_indexes[source].manifest,
        )
        for source in ("teacher", "v1")
    }
    _progress("full-evidence", run_id=session.run_id, event="cpu-timing")
    timing_samples, timing_summary = benchmark_learned_directions(
        query_rows,
        views=primary_views,
        indexes=indexes,
        encoders=encoders,
        policy=timing_policy,
        fold=session.validation_fold,
    )
    frames = {
        "quality_summary": quality.summary,
        "query_metrics": quality.query_metrics,
        "rankings": assemble_learned_rankings(quality),
        "failure_slices": analysis.failure_slices,
        "canvas_summary": analysis.canvas_summary,
        "canvas_per_query": analysis.canvas_per_query,
        "canvas_rankings": assemble_canvas_rankings(analysis),
        "examples": assemble_learned_examples(analysis, quality),
        "timing_samples": timing_samples,
        "timing_summary": timing_summary,
        "gallery_comparison": gallery.comparison,
        "gallery_rankings": assemble_gallery_rankings(gallery),
    }
    source_artifacts = {
        source: LearnedSourceArtifacts(
            image_cache_manifest=caches[source].cache_dir / "manifest.json",
            statistics=Path(statistics_paths[source]),
            feature_cache_manifest=cached_indexes[source].cache_dir / "manifest.json",
        )
        for source in ("teacher", "v1")
    }
    _progress("full-evidence", run_id=session.run_id, event="write")
    manifest_path = write_learned_artifacts(
        evidence_dir=Path(evidence_root) / "learned" / session.run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=frames,
        source_artifacts=source_artifacts,
        selected_gallery_policy=selected_gallery_policy,
        timing_policy=timing_policy,
    )
    _progress("full-evidence", run_id=session.run_id, event="complete-registry")
    registry_row = complete_learned_evidence(
        registry,
        result=result,
        session=session,
        manifest_path=manifest_path,
        canonical_splits=splits,
        completed_at=_utc_z() if completed_at is None else completed_at,
        recover_failed_evidence=recover_failed_evidence,
        recovery_error_type=recovery_error_type,
        recovery_error_message=recovery_error_message,
    )
    return LearnedEvidenceResult(
        manifest_path=manifest_path,
        registry_row=registry_row,
        quality=quality,
        analysis=analysis,
        gallery=gallery,
    )


def complete_learned_evidence(
    registry: RunRegistry,
    *,
    result: TrainingResult,
    session: TrainingSessionConfig,
    manifest_path: str | Path,
    canonical_splits: pd.DataFrame,
    completed_at: str,
    recover_failed_evidence: bool = False,
    recovery_error_type: str = "",
    recovery_error_message: str = "",
) -> dict[str, str]:
    """Validate all checkpoints and complete one existing running row exactly once."""

    matching = [row for row in registry.read() if row["run_id"] == session.run_id]
    if len(matching) != 1:
        raise ValueError("registry must contain exactly one row for evidence completion")
    row = matching[0]
    allowed_statuses = {"running", "failed"} if recover_failed_evidence else {"running"}
    if row["status"] not in allowed_statuses:
        raise ValueError("evidence completion requires a running registry row")
    if recover_failed_evidence and row["status"] == "failed":
        if (
            row.get("error_type") != recovery_error_type
            or row.get("error_message") != recovery_error_message
        ):
            raise ValueError("evidence recovery requires the exact failed artifact-write row")
    _validate_training_result(result, session, row)
    manifest = Path(manifest_path)
    validated_manifest = validate_learned_manifest(
        manifest,
        session=session,
        checkpoint=result.best_checkpoint,
        canonical_splits=canonical_splits,
    )
    artifact_paths = {
        artifact["name"]: manifest.parent / artifact["path"]
        for artifact in validated_manifest["artifacts"]  # type: ignore[index]
    }
    quality = pd.read_csv(artifact_paths["quality_summary"])
    timing = pd.read_csv(artifact_paths["timing_summary"])
    cost = json.loads(artifact_paths["cost"].read_text(encoding="utf-8"))
    selected = summarize_learned_scores(quality)
    protocol_b = quality.loc[
        quality["protocol"].eq("family")
        & quality["metric"].eq("recall")
        & quality["k"].eq(10)
        & quality["aggregation"].eq("query_mean"),
        "value",
    ]
    if len(protocol_b) != 4:
        raise ValueError("completion requires four Protocol B Recall@10 rows")
    p95 = timing.loc[
        timing["metric"].eq("end_to_end") & timing["percentile"].eq("p95"),
        "value_seconds",
    ]
    if len(p95) != 4:
        raise ValueError("completion requires four p95 end-to-end rows")
    best = result.best_checkpoint
    updates = {
        "selected_epoch": best.epoch,
        "checkpoint_path": str(best.path),
        "checkpoint_sha256": best.sha256,
        "parameter_count": int(cost["parameters"]),
        "development_winner_score": selected["development_winner_score"],
        "cross_source_score": selected["cross_source_score"],
        "source_robustness_ratio": selected["source_robustness_ratio"],
        "protocol_b_recall_at_10": float(protocol_b.mean()),
        "p95_end_to_end_seconds": float(p95.max()),
        "index_bytes": int(cost["selected_policy_total_index_bytes"]),
        "evidence_manifest_path": str(manifest),
        "completed_at_utc": completed_at,
        "status": "completed",
    }
    if recover_failed_evidence:
        return registry.recover_failed_evidence(
            session.run_id,
            updates,
            expected_error_type=recovery_error_type,
            expected_error_message=recovery_error_message,
        )
    return registry.update(session.run_id, updates)


def record_evidence_failure(
    registry: RunRegistry,
    run_id: str,
    error: BaseException,
    *,
    completed_at: str,
) -> None:
    """Mark evidence failure without ever replacing or re-raising the original error."""

    message = str(error).strip() or error.__class__.__name__
    try:
        registry.update(
            run_id,
            {
                "status": "failed",
                "completed_at_utc": completed_at,
                "error_type": error.__class__.__name__,
                "error_message": message[:500],
            },
        )
    except BaseException as registry_error:
        error.add_note(
            "failed to record evidence failure: "
            f"{registry_error.__class__.__name__}: {registry_error}"
        )
