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
    "canonical_json_bytes",
    "canonical_sha256",
    "capture_git_state",
    "capture_runtime",
    "make_torch_generator",
    "seed_everything",
    "seed_worker",
    "verify_artifact",
]
