"""Reusable training, evaluation, artifact, and experiment contracts."""

from fashion.train.artifacts import (
    ArtifactVerificationError,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import (
    CachedRun,
    RunCacheKey,
    build_run_cache_key,
    find_cached_run,
    implementation_sha256,
)
from fashion.train.engine import FoldResult, TrainConfig, train_fold
from fashion.train.metrics import (
    OOFValidationError,
    cross_fit_temperature,
    fit_temperature,
    multiclass_metrics,
    paired_group_bootstrap,
    temperature_scale_probabilities,
    validate_oof,
)
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import (
    capture_git_state,
    capture_runtime,
    make_torch_generator,
    seed_everything,
    seed_worker,
)

__all__ = [
    "ArtifactVerificationError",
    "atomic_write_bytes",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_text",
    "build_run_cache_key",
    "CachedRun",
    "canonical_json_bytes",
    "canonical_sha256",
    "capture_git_state",
    "capture_runtime",
    "cross_fit_temperature",
    "FoldResult",
    "find_cached_run",
    "fit_temperature",
    "implementation_sha256",
    "make_torch_generator",
    "multiclass_metrics",
    "new_run_id",
    "OOFValidationError",
    "paired_group_bootstrap",
    "RunRecord",
    "RunRegistry",
    "RunCacheKey",
    "seed_everything",
    "seed_worker",
    "tracked_run",
    "temperature_scale_probabilities",
    "TrainConfig",
    "train_fold",
    "validate_oof",
    "verify_artifact",
]
