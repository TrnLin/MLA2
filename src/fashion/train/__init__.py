"""Reusable training contracts exposed without importing heavy ML libraries."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "ArtifactVerificationError": "artifacts",
    "atomic_write_bytes": "artifacts",
    "atomic_write_csv": "artifacts",
    "atomic_write_json": "artifacts",
    "atomic_write_text": "artifacts",
    "canonical_json_bytes": "artifacts",
    "canonical_sha256": "artifacts",
    "verify_artifact": "artifacts",
    "CachedRun": "cache",
    "RunCacheKey": "cache",
    "build_run_cache_key": "cache",
    "find_cached_run": "cache",
    "implementation_sha256": "cache",
    "FoldResult": "engine",
    "TrainConfig": "engine",
    "train_fold": "engine",
    "OOFValidationError": "metrics",
    "cross_fit_temperature": "metrics",
    "fit_temperature": "metrics",
    "multiclass_metrics": "metrics",
    "paired_group_bootstrap": "metrics",
    "temperature_scale_probabilities": "metrics",
    "validate_oof": "metrics",
    "RefitResult": "multitask",
    "RefitTrainConfig": "multitask",
    "train_masked_multitask_refit": "multitask",
    "DuplicateRunError": "registry",
    "ImmutableRunError": "registry",
    "RegistryError": "registry",
    "RegistrySchemaError": "registry",
    "RUN_COLUMNS": "registry",
    "RUN_KINDS": "registry",
    "RunRecord": "registry",
    "RunRegistry": "registry",
    "RunRegistryError": "registry",
    "TASK2_RUN_COLUMNS": "registry",
    "TASK4_RUN_COLUMNS": "registry",
    "Task4RunRegistry": "registry",
    "new_run_id": "registry",
    "tracked_run": "registry",
    "capture_git_state": "reproducibility",
    "capture_runtime": "reproducibility",
    "make_torch_generator": "reproducibility",
    "seed_everything": "reproducibility",
    "seed_worker": "reproducibility",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load an exported contract only when a caller requests it."""

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
