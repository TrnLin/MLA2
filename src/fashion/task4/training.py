"""Deterministic training, checkpoint, and run-lifecycle tools for Task 4."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import StrEnum
from functools import partial
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fashion.task4.learned_data import CrossSourcePairDataset, FamilyBatchSampler
from fashion.task4.losses import content_mask_mse_loss, r4_loss, vicreg_loss
from fashion.task4.models import (
    B1_WEIGHT_ORIGIN,
    SCRATCH_WEIGHT_ORIGIN,
    ModelMetadata,
    _build_b1_checkpoint_encoder,
    build_autoencoder,
    build_b1_encoder,
    build_retrieval_encoder,
)
from fashion.train.registry import RUN_KINDS, RunRegistry

CandidateName: TypeAlias = Literal["R1", "R2", "R3", "R4", "R5", "B1"]
PairObjective: TypeAlias = Literal["R1", "R2", "R3", "R4", "R5", "B1"]

CHECKPOINT_SCHEMA_VERSION = 1
AMP_INITIAL_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 40901
_SHA256_LENGTH = 64

__all__ = (
    "AMP_GROWTH_INTERVAL",
    "AMP_INITIAL_SCALE",
    "CHECKPOINT_SCHEMA_VERSION",
    "BatchLoss",
    "AugmentationPolicy",
    "CandidateConfig",
    "CheckpointRecord",
    "CheckpointValidationError",
    "LoadedCheckpoint",
    "NonFiniteTrainingError",
    "RegistryIdentity",
    "SourcePolicy",
    "TrainingHyperparameters",
    "TrainingResult",
    "TrainingSessionConfig",
    "WarmupCosineScheduler",
    "amp_is_enabled",
    "apply_optimization_step",
    "build_optimizer",
    "canonical_config_json",
    "compute_batch_loss",
    "configure_determinism",
    "derive_worker_seed",
    "configuration_sha256",
    "learning_rate_at_step",
    "load_checkpoint",
    "make_grad_scaler",
    "make_data_generator",
    "make_worker_init_fn",
    "recover_stale_running",
    "run_training_attempt",
    "save_checkpoint",
    "select_best_checkpoint",
    "train_epochs",
    "validate_checkpoint_registry_binding",
    "validate_training_loader",
    "validate_training_session_binding",
)


class NonFiniteTrainingError(RuntimeError):
    """Raised when a loss, gradient, or parameter becomes non-finite."""


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint fails identity or integrity validation."""


class SourcePolicy(StrEnum):
    """Approved source pairing used by every learned comparison."""

    TEACHER_V1_PAIRS = "teacher_v1_pairs"


class AugmentationPolicy(StrEnum):
    """Approved geometry state for one training session."""

    NONE = "none"
    GEOMETRY = "geometry"


def _positive_finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _positive_integer(name: str, value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_sha256(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class TrainingHyperparameters:
    """Immutable validated values for the shared learned-model recipe."""

    seed: int = 2753
    product_batch_size: int = 64
    images_per_product: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    minimum_learning_rate: float = 1e-6
    gradient_clip_norm: float = 5.0
    amp_initial_scale: float = AMP_INITIAL_SCALE
    amp_growth_interval: int = AMP_GROWTH_INTERVAL
    planned_epochs: int = 100
    checkpoint_epochs: tuple[int, ...] = (20, 40, 60, 80, 100)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        _positive_integer("product batch size", self.product_batch_size)
        if self.images_per_product != 2:
            raise ValueError("images per product must be exactly two")
        learning_rate = _positive_finite("learning rate", self.learning_rate)
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, Real)
            or not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
        ):
            raise ValueError("weight decay must be non-negative and finite")
        warmup_epochs = _positive_integer("warm-up epochs", self.warmup_epochs)
        planned_epochs = _positive_integer("planned epochs", self.planned_epochs)
        if warmup_epochs >= planned_epochs:
            raise ValueError("warm-up epochs must be less than planned epochs")
        minimum = _positive_finite("minimum learning rate", self.minimum_learning_rate)
        if minimum >= learning_rate:
            raise ValueError("minimum learning rate must be below the learning rate")
        _positive_finite("gradient clip norm", self.gradient_clip_norm)
        _positive_finite("AMP initial scale", self.amp_initial_scale)
        growth_interval = _positive_integer("AMP growth interval", self.amp_growth_interval)
        if growth_interval <= 40900:
            raise ValueError("AMP growth interval must exceed the full-run step count")
        if not isinstance(self.checkpoint_epochs, tuple):
            raise ValueError("checkpoint epochs must be an immutable tuple")
        checkpoints = tuple(self.checkpoint_epochs)
        if (
            not checkpoints
            or any(
                isinstance(epoch, bool)
                or not isinstance(epoch, Integral)
                or int(epoch) <= 0
                or int(epoch) > planned_epochs
                for epoch in checkpoints
            )
            or tuple(sorted(set(int(epoch) for epoch in checkpoints))) != checkpoints
            or checkpoints[-1] != planned_epochs
        ):
            raise ValueError(
                "checkpoint epochs must be unique, increasing, in range, "
                "and include the final epoch"
            )


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """Approved model-factory route for one comparison candidate."""

    candidate: CandidateName
    architecture: Literal["resnet18", "resnet34"]

    def __post_init__(self) -> None:
        if self.candidate not in {"R1", "R2", "R3", "R4", "R5", "B1"}:
            raise ValueError("candidate must be one of R1-R5 or B1")
        if self.architecture not in {"resnet18", "resnet34"}:
            raise ValueError("architecture must be resnet18 or resnet34")
        if self.candidate == "R1" and self.architecture != "resnet18":
            raise ValueError("R1 must use the approved resnet18 encoder")
        if self.candidate == "R2" and self.architecture != "resnet34":
            raise ValueError("R2 must use the approved resnet34 encoder")
        if self.candidate == "R5" and self.architecture != "resnet18":
            raise ValueError("R5 must use the approved resnet18 autoencoder")
        if self.candidate == "B1" and self.architecture != "resnet18":
            raise ValueError("B1 must use the approved resnet18 pretrained encoder")

    @property
    def pretrained(self) -> bool:
        return self.candidate == "B1"

    @property
    def weight_origin(self) -> str:
        return B1_WEIGHT_ORIGIN if self.pretrained else SCRATCH_WEIGHT_ORIGIN


@dataclass(frozen=True, slots=True)
class RegistryIdentity:
    """Immutable registry fields derived from one training session."""

    run_id: str
    parent_run_id: str
    task: str
    run_kind: str
    fold: int
    method: str
    architecture: str
    objective: str
    source_policy: str
    pretrained: bool
    weight_origin: str
    deployment_eligibility: str
    seed: int
    embedding_dim: int
    planned_epochs: int
    config_hash: str
    split_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        """Return values using the registry's public field names."""

        return asdict(self)


_CANDIDATE_OBJECTIVES = {
    "R1": "vicreg",
    "R2": "vicreg",
    "R3": "vicreg",
    "R4": "vicreg_triplet",
    "R5": "content_mask_mse",
    "B1": "vicreg",
}


@dataclass(frozen=True, slots=True)
class TrainingSessionConfig:
    """One complete immutable binding shared by registry, training, and checkpoints."""

    run_id: str
    run_kind: str
    candidate: CandidateConfig
    hyperparameters: TrainingHyperparameters
    objective: str
    source_policy: SourcePolicy
    augmentation_policy: AugmentationPolicy
    validation_fold: int
    split_fingerprint: str
    parent_run_id: str | None = None
    model_metadata: ModelMetadata = field(init=False)
    config_json: str = field(init=False)
    config_hash: str = field(init=False)
    expected_registry_identity: RegistryIdentity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run ID must not be blank")
        if self.run_kind not in RUN_KINDS:
            raise ValueError("run kind is invalid")
        if not isinstance(self.candidate, CandidateConfig):
            raise ValueError("candidate must be a CandidateConfig")
        if not isinstance(self.hyperparameters, TrainingHyperparameters):
            raise ValueError("hyperparameters must be TrainingHyperparameters")
        expected_objective = _CANDIDATE_OBJECTIVES[self.candidate.candidate]
        if self.objective != expected_objective:
            raise ValueError(
                f"objective for {self.candidate.candidate} must be {expected_objective}"
            )
        if self.source_policy is not SourcePolicy.TEACHER_V1_PAIRS:
            raise ValueError("source policy must be SourcePolicy.TEACHER_V1_PAIRS")
        if not isinstance(self.augmentation_policy, AugmentationPolicy):
            raise ValueError("augmentation policy must be an AugmentationPolicy")
        allowed_augmentation = {
            "R1": {AugmentationPolicy.NONE},
            "R2": {AugmentationPolicy.NONE},
            "R3": {AugmentationPolicy.GEOMETRY},
            "R4": {AugmentationPolicy.NONE, AugmentationPolicy.GEOMETRY},
            "R5": {AugmentationPolicy.NONE},
            "B1": {AugmentationPolicy.NONE},
        }[self.candidate.candidate]
        if self.augmentation_policy not in allowed_augmentation:
            raise ValueError(
                f"{self.candidate.candidate} augmentation policy is not approved"
            )
        if self.candidate.candidate == "B1":
            if self.run_kind not in {"benchmark", "smoke"}:
                raise ValueError("B1 run kind must be benchmark or smoke")
        elif self.run_kind == "benchmark":
            raise ValueError("scratch candidates cannot use benchmark run kind")
        if (
            isinstance(self.validation_fold, bool)
            or not isinstance(self.validation_fold, Integral)
            or int(self.validation_fold) not in range(5)
        ):
            raise ValueError("validation fold must be an integer in range(5)")
        _validate_sha256("split fingerprint", self.split_fingerprint)
        if self.parent_run_id is not None and (
            not isinstance(self.parent_run_id, str) or not self.parent_run_id.strip()
        ):
            raise ValueError("parent run ID must be nonblank or None")
        metadata = _expected_metadata(self.candidate)
        object.__setattr__(self, "model_metadata", metadata)
        config_json = canonical_config_json(_session_config_values(self, metadata))
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        object.__setattr__(self, "config_json", config_json)
        object.__setattr__(self, "config_hash", config_hash)
        object.__setattr__(
            self,
            "expected_registry_identity",
            RegistryIdentity(
                run_id=self.run_id,
                parent_run_id=self.parent_run_id or "",
                task="task4",
                run_kind=self.run_kind,
                fold=int(self.validation_fold),
                method=self.candidate.candidate,
                architecture=self.candidate.architecture,
                objective=self.objective,
                source_policy=self.source_policy,
                pretrained=self.candidate.pretrained,
                weight_origin=metadata.weight_origin,
                deployment_eligibility=(
                    "eligible" if metadata.deployment_eligible else "comparison_only"
                ),
                seed=int(self.hyperparameters.seed),
                embedding_dim=128,
                planned_epochs=int(self.hyperparameters.planned_epochs),
                config_hash=config_hash,
                split_fingerprint=self.split_fingerprint,
            ),
        )

    def build_cpu_model(self) -> nn.Module:
        """Construct this session's approved model after deterministic setup."""

        return _build_training_candidate(self.candidate)


@dataclass(frozen=True, slots=True)
class BatchLoss:
    """One differentiable objective and optional route diagnostics."""

    total: torch.Tensor
    product_embeddings: torch.Tensor | None = None
    included_values: int | None = None


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Immutable identity and score for one milestone checkpoint."""

    epoch: int
    path: Path
    sha256: str
    config_hash: str
    score: float
    split_fingerprint: str
    weight_origin: str
    parent_run_id: str | None
    run_id: str
    run_kind: str

    def __post_init__(self) -> None:
        _positive_integer("checkpoint epoch", self.epoch)
        _validate_sha256("checkpoint SHA-256", self.sha256)
        _validate_sha256("checkpoint config hash", self.config_hash)
        _validate_sha256("checkpoint split fingerprint", self.split_fingerprint)
        if not isinstance(self.score, Real) or not math.isfinite(float(self.score)):
            raise ValueError("checkpoint score must be finite")
        if not self.weight_origin:
            raise ValueError("checkpoint weight origin must not be blank")
        if self.parent_run_id is not None and not self.parent_run_id.strip():
            raise ValueError("checkpoint parent run ID must be nonblank or None")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("checkpoint run ID must not be blank")
        if self.run_kind not in RUN_KINDS:
            raise ValueError("checkpoint run kind is invalid")


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated model and resumable optimizer/schedule state."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: WarmupCosineScheduler
    raw_scaler_state: dict[str, Any]
    epoch: int
    score: float
    session: TrainingSessionConfig
    candidate: CandidateConfig
    hyperparameters: TrainingHyperparameters
    config_hash: str
    split_fingerprint: str
    weight_origin: str
    parent_run_id: str | None
    sha256: str

    @property
    def optimizer_state(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    @property
    def scheduler_state(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    @property
    def scaler_state(self) -> dict[str, Any]:
        return self.raw_scaler_state

    def make_resume_scaler(self, device: torch.device | str) -> torch.amp.GradScaler:
        """Restore scaler state only on the explicit resume device."""

        if self.raw_scaler_state and not amp_is_enabled(device):
            raise ValueError("non-empty scaler state requires an available CUDA resume device")
        scaler = make_grad_scaler(
            device,
            initial_scale=self.hyperparameters.amp_initial_scale,
            growth_interval=self.hyperparameters.amp_growth_interval,
        )
        if self.raw_scaler_state:
            scaler.load_state_dict(self.raw_scaler_state)
        return scaler


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Successful training handoff which remains running pending Task 6 evidence."""

    run_id: str
    run_kind: str
    checkpoints: tuple[CheckpointRecord, ...]
    best_checkpoint: CheckpointRecord
    status: Literal["running"] = "running"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be blank")
        if self.run_kind not in RUN_KINDS:
            raise ValueError("run kind is invalid")
        if not self.checkpoints or self.best_checkpoint not in self.checkpoints:
            raise ValueError("best checkpoint must belong to the milestone set")
        if self.status != "running":
            raise ValueError("successful training must remain running for evidence handoff")


def derive_worker_seed(base_seed: int, worker_id: int) -> int:
    """Derive a stable uint32 seed for one deterministic data-loader worker."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral) or base_seed < 0:
        raise ValueError("base seed must be a non-negative integer")
    if isinstance(worker_id, bool) or not isinstance(worker_id, Integral) or worker_id < 0:
        raise ValueError("worker ID must be a non-negative integer")
    return (int(base_seed) + int(worker_id)) % (2**32)


def configure_determinism(seed: int = 2753) -> None:
    """Seed all random sources and require deterministic torch algorithms."""

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA is already initialized; deterministic setup must run first")
    derived_seed = derive_worker_seed(seed, 0)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(derived_seed)
    np.random.seed(derived_seed)
    torch.manual_seed(derived_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _initialize_worker(worker_id: int, *, base_seed: int) -> None:
    seed = derive_worker_seed(base_seed, worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_worker_init_fn(base_seed: int = 2753) -> Callable[[int], None]:
    """Return a spawn-picklable worker initializer with the frozen derivation."""

    derive_worker_seed(base_seed, 0)
    return partial(_initialize_worker, base_seed=int(base_seed))


def make_data_generator(seed: int = 2753) -> torch.Generator:
    """Return an independently seeded generator for deterministic data ordering."""

    generator = torch.Generator()
    generator.manual_seed(derive_worker_seed(seed, 0))
    return generator


def amp_is_enabled(device: torch.device | str) -> bool:
    """Return whether AMP is safe and active on the requested device."""

    parsed = torch.device(device)
    return parsed.type == "cuda" and torch.cuda.is_available()


def make_grad_scaler(
    device: torch.device | str,
    *,
    initial_scale: Real = AMP_INITIAL_SCALE,
    growth_interval: int = AMP_GROWTH_INTERVAL,
) -> torch.amp.GradScaler:
    """Build a CUDA scaler which is safely disabled for CPU training."""

    scale = _positive_finite("AMP initial scale", initial_scale)
    growth = _positive_integer("AMP growth interval", growth_interval)
    if growth <= 40900:
        raise ValueError("AMP growth interval must exceed the full-run step count")
    enabled = amp_is_enabled(device)
    return torch.amp.GradScaler(
        "cuda",
        enabled=enabled,
        init_scale=scale,
        growth_interval=growth,
    )


def build_optimizer(
    model: nn.Module,
    config: TrainingHyperparameters,
) -> torch.optim.AdamW:
    """Build AdamW with the shared learning rate and weight decay."""

    parameters = _model_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def learning_rate_at_step(
    step: int,
    *,
    steps_per_epoch: int,
    config: TrainingHyperparameters,
) -> float:
    """Return the exact per-update warm-up/cosine learning rate."""

    if isinstance(step, bool) or not isinstance(step, Integral):
        raise ValueError("step must be an integer")
    steps = _positive_integer("steps per epoch", steps_per_epoch)
    total_steps = config.planned_epochs * steps
    step_index = int(step)
    if not 0 <= step_index < total_steps:
        raise ValueError("step must lie inside the planned training updates")
    warmup_steps = config.warmup_epochs * steps
    if step_index < warmup_steps:
        return config.learning_rate * (step_index + 1) / warmup_steps
    decay_updates = total_steps - warmup_steps
    if decay_updates == 1:
        return config.minimum_learning_rate
    progress = (step_index - warmup_steps) / (decay_updates - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


class WarmupCosineScheduler:
    """Minimal serializable per-step scheduler for the frozen recipe."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        steps_per_epoch: int,
        config: TrainingHyperparameters,
    ) -> None:
        self.optimizer = optimizer
        self.steps_per_epoch = _positive_integer("steps per epoch", steps_per_epoch)
        self.config = config
        self.step_index = 0
        self._set_rate()

    def _set_rate(self) -> None:
        rate = learning_rate_at_step(
            self.step_index,
            steps_per_epoch=self.steps_per_epoch,
            config=self.config,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def step(self) -> None:
        total_steps = self.config.planned_epochs * self.steps_per_epoch
        if self.step_index >= total_steps - 1:
            self.step_index = total_steps - 1
        else:
            self.step_index += 1
        self._set_rate()

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "steps_per_epoch": self.steps_per_epoch,
            "config_hash": configuration_sha256(self.config),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_keys = {"step_index", "steps_per_epoch", "config_hash"}
        if set(state) != expected_keys:
            raise ValueError("scheduler state keys do not match")
        if int(state["steps_per_epoch"]) != self.steps_per_epoch:
            raise ValueError("scheduler steps per epoch do not match")
        if state["config_hash"] != configuration_sha256(self.config):
            raise ValueError("scheduler config hash does not match")
        step_index = int(state["step_index"])
        total_steps = self.config.planned_epochs * self.steps_per_epoch
        if not 0 <= step_index < total_steps:
            raise ValueError("scheduler step index is out of range")
        self.step_index = step_index
        self._set_rate()


def _required_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch {key!r} must be a tensor")
    return value


def compute_batch_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    objective: PairObjective,
) -> BatchLoss:
    """Compute one approved objective without using retrieval-only ``encode``."""

    if objective == "R5":
        teacher = _required_tensor(batch, "teacher")
        v1 = _required_tensor(batch, "v1")
        teacher_mask = _required_tensor(batch, "teacher_content_mask")
        v1_mask = _required_tensor(batch, "v1_content_mask")
        images = torch.cat((teacher, v1), dim=0)
        masks = torch.cat((teacher_mask, v1_mask), dim=0)
        output = model(images)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("R5 model must return reconstruction and bottleneck")
        reconstruction, _ = output
        reconstruction_loss = content_mask_mse_loss(reconstruction, images, masks)
        return BatchLoss(
            total=reconstruction_loss.total,
            included_values=reconstruction_loss.included_values,
        )

    if objective not in {"R1", "R2", "R3", "R4", "B1"}:
        raise ValueError("objective must be one of R1-R5 or B1")
    teacher_projection = model(_required_tensor(batch, "teacher"))
    v1_projection = model(_required_tensor(batch, "v1"))
    if objective != "R4":
        return BatchLoss(total=vicreg_loss(teacher_projection, v1_projection).total)

    product_projection = (teacher_projection + v1_projection) / 2.0
    product_norms = torch.linalg.vector_norm(product_projection, dim=1)
    if not torch.isfinite(product_norms).all() or torch.any(product_norms <= 0):
        raise ValueError("R4 product projections must have finite non-zero norms")
    product_embeddings = F.normalize(product_projection, p=2, dim=1)
    combined = r4_loss(
        teacher_projection,
        v1_projection,
        triplet_embeddings=product_embeddings,
        product_ids=batch["id"],
        family_groups=batch["product_family_group"],
        sha256=batch["sha256"],
        duplicate_groups=batch["duplicate_group"],
    )
    return BatchLoss(total=combined.total, product_embeddings=product_embeddings)


def _model_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _require_finite_parameters(parameters: Sequence[nn.Parameter]) -> None:
    if any(not torch.isfinite(parameter.detach()).all() for parameter in parameters):
        raise NonFiniteTrainingError("non-finite parameter")


def apply_optimization_step(
    loss: torch.Tensor,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float,
) -> float:
    """Validate, backpropagate, clip, and update exactly once."""

    parameters = _model_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    _require_finite_parameters(parameters)
    if loss.ndim != 0 or not torch.isfinite(loss.detach()):
        raise NonFiniteTrainingError("non-finite loss")
    clip_norm = _positive_finite("gradient clip norm", gradient_clip_norm)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise NonFiniteTrainingError("non-finite gradient")
    unclipped_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=clip_norm,
        error_if_nonfinite=True,
    )
    if not torch.isfinite(unclipped_norm):
        raise NonFiniteTrainingError("non-finite gradient norm")
    scaler.step(optimizer)
    scaler.update()
    _require_finite_parameters(parameters)
    return float(unclipped_norm.detach().cpu())


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_config_json(config: Any) -> str:
    """Serialize a configuration deterministically for stable identity."""

    return json.dumps(
        _json_value(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def configuration_sha256(config: Any) -> str:
    """Hash the canonical UTF-8 configuration."""

    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def _session_config_values(
    session: TrainingSessionConfig,
    metadata: ModelMetadata,
) -> dict[str, Any]:
    return {
        "candidate": {
            "candidate": session.candidate.candidate,
            "architecture": session.candidate.architecture,
            "pretrained": session.candidate.pretrained,
        },
        "hyperparameters": asdict(session.hyperparameters),
        "objective": session.objective,
        "source_policy": session.source_policy,
        "augmentation_policy": session.augmentation_policy,
        "validation_fold": int(session.validation_fold),
        "split_fingerprint": session.split_fingerprint,
        "parent_run_id": session.parent_run_id,
        "model_metadata": _metadata_dict(metadata),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_dict(metadata: ModelMetadata) -> dict[str, Any]:
    return asdict(metadata)


def _expected_metadata(candidate: CandidateConfig) -> ModelMetadata:
    return ModelMetadata(
        architecture=candidate.architecture,
        pretrained=candidate.pretrained,
        weight_origin=candidate.weight_origin,
        deployment_eligible=not candidate.pretrained,
    )


def _serialized_registry_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _validate_registry_identity(
    session: TrainingSessionConfig,
    registry_row: Mapping[str, Any],
) -> None:
    for field_name, expected in session.expected_registry_identity.as_dict().items():
        actual = registry_row.get(field_name)
        if _serialized_registry_value(actual) != _serialized_registry_value(expected):
            label = field_name.replace("_", " ")
            raise ValueError(f"registry {label} does not match training session")


def validate_training_session_binding(
    session: TrainingSessionConfig,
    *,
    model: nn.Module,
    registry_row: Mapping[str, Any],
    batches: Iterable[Mapping[str, Any]],
) -> None:
    """Validate model, data loader, and appended row against one session."""

    if not isinstance(session, TrainingSessionConfig):
        raise ValueError("session must be a TrainingSessionConfig")
    _validate_registry_identity(session, registry_row)
    if getattr(model, "metadata", None) != session.model_metadata:
        raise ValueError("model metadata does not match training session")
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("session model factory must return a CPU model")
    validate_training_loader(session, batches)


_FAMILY_PAIR_COLUMNS = ("id", "product_family_group", "sha256", "duplicate_group")


def _canonical_family_frame(pairs: object, label: str) -> pd.DataFrame:
    if not isinstance(pairs, pd.DataFrame):
        raise ValueError(f"{label} must retain its canonical pair frame")
    if missing := set(_FAMILY_PAIR_COLUMNS).difference(pairs.columns):
        raise ValueError(f"{label} pair frame is missing columns: {sorted(missing)}")
    frame = pairs.loc[:, list(_FAMILY_PAIR_COLUMNS)].copy()
    ids = pd.to_numeric(frame["id"], errors="coerce")
    if ids.isna().any() or not ids.mod(1).eq(0).all():
        raise ValueError(f"{label} pair IDs must be integer-compatible")
    frame["id"] = ids.astype(np.int64)
    for column in _FAMILY_PAIR_COLUMNS[1:]:
        frame[column] = frame[column].astype(str)
    return frame.sort_values("id").reset_index(drop=True)


def _validate_family_sampler_rows(
    sampler: FamilyBatchSampler,
    dataset: CrossSourcePairDataset,
) -> None:
    expected = _canonical_family_frame(
        getattr(dataset, "pairs", None),
        "R4 loader dataset",
    )
    actual = _canonical_family_frame(
        getattr(sampler, "pairs", None),
        "R4 family sampler",
    )
    if actual["id"].tolist() != expected["id"].tolist():
        raise ValueError("family sampler product IDs do not match the loader dataset")
    if not actual.equals(expected):
        raise ValueError("family sampler pair metadata does not match the loader dataset")


def validate_training_loader(
    session: TrainingSessionConfig,
    batches: Iterable[Mapping[str, Any]],
) -> None:
    """Validate the concrete pair dataset, geometry, sampler, and batch size."""

    if not isinstance(batches, DataLoader):
        raise ValueError("training batches must be a concrete DataLoader")
    dataset = getattr(batches, "dataset", None)
    if not isinstance(dataset, CrossSourcePairDataset):
        raise ValueError("training loader dataset must be CrossSourcePairDataset")
    if dataset.validation_fold != session.validation_fold:
        raise ValueError("loader dataset validation fold does not match session")
    if dataset.split_fingerprint != session.split_fingerprint:
        raise ValueError("loader dataset split fingerprint does not match session")
    has_geometry = dataset.geometry_policy is not None
    expects_geometry = session.augmentation_policy is AugmentationPolicy.GEOMETRY
    if has_geometry != expects_geometry:
        raise ValueError("loader geometry state does not match augmentation policy")

    sampler = getattr(batches, "sampler", None)
    batch_sampler = getattr(batches, "batch_sampler", None)
    if isinstance(sampler, FamilyBatchSampler):
        raise ValueError(
            "FamilyBatchSampler must be the loader batch_sampler, never its sampler"
        )
    has_family_sampler = isinstance(batch_sampler, FamilyBatchSampler)
    if session.candidate.candidate == "R4" and not has_family_sampler:
        raise ValueError("R4 loader must use FamilyBatchSampler as its batch_sampler")
    if session.candidate.candidate != "R4" and has_family_sampler:
        raise ValueError("non-R4 loader must not use FamilyBatchSampler")
    if has_family_sampler:
        _validate_family_sampler_rows(batch_sampler, dataset)
        return

    batch_size = getattr(batches, "batch_size", None)
    drop_last = getattr(batches, "drop_last", None)
    if batch_size is None and batch_sampler is not None:
        batch_size = getattr(batch_sampler, "batch_size", None)
        drop_last = getattr(batch_sampler, "drop_last", None)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or int(batch_size) != session.hyperparameters.product_batch_size
    ):
        raise ValueError("training loader must prove batch size 64")
    if drop_last is not True:
        raise ValueError("ordinary training DataLoader must set drop_last=True")


def validate_checkpoint_registry_binding(
    result: TrainingResult,
    *,
    session: TrainingSessionConfig,
    registry_row: Mapping[str, Any],
) -> None:
    """Validate the full training result and selected checkpoint payload for Task 6."""

    _validate_registry_identity(session, registry_row)
    if not isinstance(result, TrainingResult):
        raise ValueError("completion input must be an immutable TrainingResult")
    if result.run_id != session.run_id:
        raise ValueError("training result run ID does not match session")
    if result.run_kind != session.run_kind:
        raise ValueError("training result run kind does not match session")
    expected_epochs = session.hyperparameters.checkpoint_epochs
    if tuple(record.epoch for record in result.checkpoints) != expected_epochs:
        raise ValueError("training result must contain the full milestone set")
    for record in result.checkpoints:
        _validate_checkpoint_session(record, session)
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
        if loaded.epoch != record.epoch:
            raise ValueError("checkpoint payload epoch does not match milestone record")
        if loaded.score != record.score:
            raise ValueError("checkpoint payload score does not match milestone record")
        del loaded

    recomputed = select_best_checkpoint(result.checkpoints)
    if result.best_checkpoint != recomputed:
        raise ValueError("selected checkpoint is not the recomputed best checkpoint")

    checkpoint = result.best_checkpoint
    completion_binding = {
        "selected_epoch": checkpoint.epoch,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.sha256,
    }
    for field_name, expected in completion_binding.items():
        if _serialized_registry_value(registry_row.get(field_name)) != str(expected):
            label = field_name.replace("_", " ")
            raise ValueError(f"registry {label} does not match checkpoint")


def _validate_checkpoint_session(
    checkpoint: CheckpointRecord,
    session: TrainingSessionConfig,
) -> None:
    bound_values = {
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
        "parent run id": (checkpoint.parent_run_id, session.parent_run_id),
    }
    for label, (actual, expected) in bound_values.items():
        if actual != expected:
            raise ValueError(f"checkpoint {label} does not match training session")
    if checkpoint.epoch not in session.hyperparameters.checkpoint_epochs:
        raise ValueError("checkpoint epoch is not an approved session milestone")


def _build_candidate(candidate: CandidateConfig) -> nn.Module:
    if candidate.candidate == "R5":
        return build_autoencoder()
    if candidate.candidate == "B1":
        return _build_b1_checkpoint_encoder()
    return build_retrieval_encoder(candidate.architecture)


def _build_training_candidate(candidate: CandidateConfig) -> nn.Module:
    if candidate.candidate == "R5":
        return build_autoencoder()
    if candidate.candidate == "B1":
        return build_b1_encoder()
    return build_retrieval_encoder(candidate.architecture)


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    session: TrainingSessionConfig,
    score: float = 0.0,
) -> CheckpointRecord:
    """Atomically save complete resumable state and return its file identity."""

    checkpoint_epoch = _positive_integer("checkpoint epoch", epoch)
    if checkpoint_epoch not in session.hyperparameters.checkpoint_epochs:
        raise ValueError("checkpoint epoch is not an approved milestone")
    if checkpoint_epoch > session.hyperparameters.planned_epochs:
        raise ValueError("checkpoint epoch exceeds planned epochs")
    if not isinstance(score, Real) or not math.isfinite(float(score)):
        raise ValueError("checkpoint score must be finite")
    metadata = getattr(model, "metadata", None)
    if metadata != session.model_metadata:
        raise ValueError("model metadata does not match training session")
    if scheduler.config != session.hyperparameters:
        raise ValueError("scheduler hyperparameters do not match training session")
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "epoch": checkpoint_epoch,
        "score": float(score),
        "model_metadata": _metadata_dict(metadata),
        "canonical_config_json": session.config_json,
        "config_hash": session.config_hash,
        "amp_growth_interval": session.hyperparameters.amp_growth_interval,
        "split_fingerprint": session.split_fingerprint,
        "weight_origin": metadata.weight_origin,
        "parent_run_id": session.parent_run_id,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return CheckpointRecord(
        epoch=checkpoint_epoch,
        path=target,
        sha256=_sha256_file(target),
        config_hash=session.config_hash,
        score=float(score),
        split_fingerprint=session.split_fingerprint,
        weight_origin=session.model_metadata.weight_origin,
        parent_run_id=session.parent_run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
    )


def _decode_checkpoint_config(
    payload: Mapping[str, Any],
) -> TrainingSessionConfig:
    config_json = payload.get("canonical_config_json")
    config_hash = payload.get("config_hash")
    if not isinstance(config_json, str) or not isinstance(config_hash, str):
        raise CheckpointValidationError("checkpoint config identity is missing")
    actual_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if actual_hash != config_hash:
        raise CheckpointValidationError("checkpoint config hash does not match canonical config")
    try:
        decoded = json.loads(config_json)
        candidate_values = decoded["candidate"]
        hyperparameter_values = dict(decoded["hyperparameters"])
        candidate = CandidateConfig(
            candidate=candidate_values["candidate"],
            architecture=candidate_values["architecture"],
        )
        if candidate_values["pretrained"] is not candidate.pretrained:
            raise CheckpointValidationError(
                "checkpoint pretrained provenance contradicts its factory route"
            )
        hyperparameter_values["checkpoint_epochs"] = tuple(
            hyperparameter_values["checkpoint_epochs"]
        )
        hyperparameters = TrainingHyperparameters(**hyperparameter_values)
        session = TrainingSessionConfig(
            run_id=payload["run_id"],
            run_kind=payload["run_kind"],
            candidate=candidate,
            hyperparameters=hyperparameters,
            objective=decoded["objective"],
            source_policy=SourcePolicy(decoded["source_policy"]),
            augmentation_policy=AugmentationPolicy(decoded["augmentation_policy"]),
            validation_fold=decoded["validation_fold"],
            split_fingerprint=decoded["split_fingerprint"],
            parent_run_id=decoded["parent_run_id"],
        )
    except CheckpointValidationError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointValidationError(f"checkpoint config is invalid: {error}") from error
    if session.config_json != config_json:
        raise CheckpointValidationError("checkpoint canonical config is not normalized")
    if session.config_hash != config_hash:
        raise CheckpointValidationError("checkpoint config hash does not match decoded session")
    if decoded["model_metadata"] != _metadata_dict(session.model_metadata):
        raise CheckpointValidationError("checkpoint config model metadata does not match")
    return session


def load_checkpoint(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_config_hash: str,
    expected_split_fingerprint: str,
    expected_weight_origin: str,
    expected_parent_run_id: str | None,
    expected_run_id: str,
    expected_run_kind: str,
    map_location: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    """Validate a checkpoint and reconstruct only its approved model factory."""

    target = Path(path)
    try:
        _validate_sha256("expected checkpoint SHA-256", expected_sha256)
    except ValueError as error:
        raise CheckpointValidationError(str(error)) from error
    actual_sha256 = _sha256_file(target)
    if actual_sha256 != expected_sha256:
        raise CheckpointValidationError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = torch.load(target, map_location=map_location, weights_only=True)
    except Exception as error:
        raise CheckpointValidationError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(payload, Mapping):
        raise CheckpointValidationError("checkpoint payload must be a mapping")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointValidationError("checkpoint schema version does not match")
    session = _decode_checkpoint_config(payload)
    candidate = session.candidate
    hyperparameters = session.hyperparameters
    config_hash = session.config_hash
    if payload.get("run_id") != expected_run_id:
        raise CheckpointValidationError("checkpoint run ID does not match expected run")
    if payload.get("run_kind") != expected_run_kind:
        raise CheckpointValidationError("checkpoint run kind does not match expected kind")
    if config_hash != expected_config_hash:
        raise CheckpointValidationError("checkpoint config hash does not match expected config")
    if payload.get("amp_growth_interval") != hyperparameters.amp_growth_interval:
        raise CheckpointValidationError("checkpoint AMP growth interval does not match config")
    split_fingerprint = payload.get("split_fingerprint")
    try:
        _validate_sha256("split fingerprint", split_fingerprint)
    except ValueError as error:
        raise CheckpointValidationError(str(error)) from error
    if split_fingerprint != session.split_fingerprint:
        raise CheckpointValidationError("checkpoint split fingerprint contradicts config")
    if (
        split_fingerprint != expected_split_fingerprint
    ):
        raise CheckpointValidationError("checkpoint split fingerprint does not match")
    stored_weight_origin = payload.get("weight_origin")
    if stored_weight_origin != candidate.weight_origin:
        raise CheckpointValidationError(
            "checkpoint weight origin contradicts its factory provenance"
        )
    if (
        stored_weight_origin != expected_weight_origin
    ):
        raise CheckpointValidationError("checkpoint weight origin does not match expected origin")

    model = _build_candidate(candidate)
    expected_metadata = _expected_metadata(candidate)
    if payload.get("model_metadata") != _metadata_dict(expected_metadata):
        raise CheckpointValidationError("checkpoint model metadata does not match factory metadata")
    stored_state = payload.get("model_state_dict")
    if not isinstance(stored_state, Mapping):
        raise CheckpointValidationError("checkpoint model state dict is missing")
    if set(stored_state) != set(model.state_dict()):
        raise CheckpointValidationError("checkpoint state-dict keys do not match factory model")
    try:
        model.load_state_dict(stored_state, strict=True)
    except RuntimeError as error:
        raise CheckpointValidationError(f"checkpoint model state is invalid: {error}") from error
    epoch = payload.get("epoch")
    try:
        parsed_epoch = _positive_integer("checkpoint epoch", epoch)
    except ValueError as error:
        raise CheckpointValidationError(str(error)) from error
    if parsed_epoch > hyperparameters.planned_epochs:
        raise CheckpointValidationError("checkpoint epoch exceeds planned epochs")
    if parsed_epoch not in hyperparameters.checkpoint_epochs:
        raise CheckpointValidationError("checkpoint epoch is not an approved milestone")
    score = payload.get("score")
    if not isinstance(score, Real) or not math.isfinite(float(score)):
        raise CheckpointValidationError("checkpoint score must be finite")
    optimizer_state = payload.get("optimizer_state_dict")
    scheduler_state = payload.get("scheduler_state_dict")
    scaler_state = payload.get("scaler_state_dict")
    resumable_states = (optimizer_state, scheduler_state, scaler_state)
    if not all(isinstance(state, dict) for state in resumable_states):
        raise CheckpointValidationError("checkpoint resumable training state is incomplete")
    parent_run_id = payload.get("parent_run_id")
    if parent_run_id is not None and (
        not isinstance(parent_run_id, str) or not parent_run_id.strip()
    ):
        raise CheckpointValidationError("checkpoint parent run ID is invalid")
    if parent_run_id != session.parent_run_id:
        raise CheckpointValidationError("checkpoint parent identity contradicts config")
    if parent_run_id != expected_parent_run_id:
        raise CheckpointValidationError("checkpoint parent identity does not match expected parent")
    try:
        optimizer = build_optimizer(model, hyperparameters)
        optimizer.load_state_dict(optimizer_state)
        steps_per_epoch = int(scheduler_state["steps_per_epoch"])
        scheduler = WarmupCosineScheduler(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            config=hyperparameters,
        )
        scheduler.load_state_dict(scheduler_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointValidationError(
            f"checkpoint resumable training state is invalid: {error}"
        ) from error
    return LoadedCheckpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        raw_scaler_state=dict(scaler_state),
        epoch=parsed_epoch,
        score=float(score),
        session=session,
        candidate=candidate,
        hyperparameters=hyperparameters,
        config_hash=config_hash,
        split_fingerprint=split_fingerprint,
        weight_origin=stored_weight_origin,
        parent_run_id=parent_run_id,
        sha256=actual_sha256,
    )


def select_best_checkpoint(records: Sequence[CheckpointRecord]) -> CheckpointRecord:
    """Select highest score, breaking exact ties toward the earlier epoch."""

    if not records:
        raise ValueError("at least one checkpoint is required")
    epochs = [record.epoch for record in records]
    if len(set(epochs)) != len(epochs):
        raise ValueError("checkpoint epochs must be unique")
    return min(records, key=lambda record: (-float(record.score), record.epoch))


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _log_progress(stage: str, **fields: object) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"task4-progress stage={stage} {details}".strip(), flush=True)


def train_epochs(
    *,
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device | str,
    session: TrainingSessionConfig,
    score_callback: Callable[[nn.Module, int], float],
    checkpoint_callback: Callable[[int, float], CheckpointRecord],
) -> TrainingResult:
    """Train all planned epochs and delegate milestone scoring and persistence."""

    parsed_device = torch.device(device)
    config = session.hyperparameters
    if scheduler.config != config:
        raise ValueError("scheduler hyperparameters do not match training session")
    model.to(parsed_device)
    model.train()
    checkpoints: list[CheckpointRecord] = []
    for epoch in range(1, config.planned_epochs + 1):
        _log_progress(
            "training",
            run_id=session.run_id,
            epoch=f"{epoch}/{config.planned_epochs}",
            event="start",
        )
        epoch_targets: list[Any] = []
        seen_targets: set[int] = set()
        for attribute in ("dataset", "sampler", "batch_sampler"):
            target = getattr(batches, attribute, None)
            if (
                target is not None
                and hasattr(target, "set_epoch")
                and id(target) not in seen_targets
            ):
                epoch_targets.append(target)
                seen_targets.add(id(target))
        for target in epoch_targets:
            target.set_epoch(epoch - 1)
        observed_batch = False
        for batch in batches:
            observed_batch = True
            device_batch = _batch_to_device(batch, parsed_device)
            with torch.amp.autocast(
                device_type=parsed_device.type,
                enabled=amp_is_enabled(parsed_device),
            ):
                batch_loss = compute_batch_loss(
                    model,
                    device_batch,
                    objective=session.candidate.candidate,
                )
            apply_optimization_step(
                batch_loss.total,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            scheduler.step()
        if not observed_batch:
            raise ValueError("training batches must not be empty")
        if epoch in config.checkpoint_epochs:
            was_training = model.training
            model.eval()
            try:
                with torch.inference_mode():
                    _log_progress(
                        "checkpoint",
                        run_id=session.run_id,
                        epoch=epoch,
                        event="score",
                    )
                    score = float(score_callback(model, epoch))
            finally:
                model.train(was_training)
            if not math.isfinite(score):
                raise NonFiniteTrainingError("non-finite checkpoint score")
            record = checkpoint_callback(epoch, score)
            _log_progress(
                "checkpoint",
                run_id=session.run_id,
                epoch=epoch,
                event="saved",
                checkpoint_sha256=record.sha256,
            )
            if record.epoch != epoch or float(record.score) != score:
                raise ValueError("checkpoint callback returned mismatched epoch or score")
            _validate_checkpoint_session(record, session)
            checkpoints.append(record)
        _log_progress(
            "training",
            run_id=session.run_id,
            epoch=f"{epoch}/{config.planned_epochs}",
            event="complete",
        )
    best = select_best_checkpoint(checkpoints)
    return TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=tuple(checkpoints),
        best_checkpoint=best,
    )


def _utc_z(now: datetime | None = None) -> str:
    instant = datetime.now(timezone.utc) if now is None else now
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("time must be timezone-aware")
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def run_training_attempt(
    *,
    registry: RunRegistry,
    row: Mapping[str, Any],
    session: TrainingSessionConfig,
    batches: Iterable[Mapping[str, Any]],
    device_factory: Callable[[], torch.device],
    train: Callable[
        [
            torch.device,
            nn.Module,
            TrainingSessionConfig,
            Iterable[Mapping[str, Any]],
        ],
        Any,
    ],
) -> Any:
    """Create the model only after append, validation, and deterministic setup."""

    written = registry.append(row)
    run_id = written["run_id"]
    try:
        _validate_registry_identity(session, written)
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA is already initialized; deterministic setup must run first"
            )
        configure_determinism(session.hyperparameters.seed)
        model = session.build_cpu_model()
        validate_training_session_binding(
            session,
            model=model,
            registry_row=written,
            batches=batches,
        )
        device = device_factory()
        return train(device, model, session, batches)
    except BaseException as error:
        message = str(error).strip() or error.__class__.__name__
        try:
            registry.update(
                run_id,
                {
                    "status": "failed",
                    "completed_at_utc": _utc_z(),
                    "error_type": error.__class__.__name__,
                    "error_message": message[:500],
                },
            )
        except BaseException as registry_error:
            error.add_note(
                "failed to mark registry row failed: "
                f"{registry_error.__class__.__name__}: {registry_error}"
            )
            raise error from registry_error
        raise


def recover_stale_running(
    registry: RunRegistry,
    *,
    run_ids: Iterable[str],
    stale_before: datetime,
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Explicitly fail rows whose running process has been abandoned."""

    if stale_before.tzinfo is None or stale_before.utcoffset() is None:
        raise ValueError("stale-before time must be timezone-aware")
    completion_time = _utc_z(now)
    threshold = stale_before.astimezone(timezone.utc)
    selected_ids = set(run_ids)
    if any(not isinstance(run_id, str) or not run_id for run_id in selected_ids):
        raise ValueError("recovery run IDs must be nonblank strings")
    recovered: list[str] = []
    for row in registry.read():
        if row["run_id"] not in selected_ids or row["status"] != "running":
            continue
        started = datetime.fromisoformat(row["started_at_utc"].replace("Z", "+00:00"))
        if started >= threshold:
            continue
        registry.update(
            row["run_id"],
            {
                "status": "failed",
                "completed_at_utc": completion_time,
                "error_type": "StaleRunningAttempt",
                "error_message": "running attempt was explicitly recovered as abandoned",
            },
        )
        recovered.append(row["run_id"])
    return tuple(recovered)
