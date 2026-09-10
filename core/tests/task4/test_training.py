from __future__ import annotations

import hashlib
import importlib
import json
import math
import pickle
import random
import weakref
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import fashion.task4 as task4
import fashion.task4.models as model_module
from fashion.task4.learned_data import (
    CrossSourcePairDataset,
    FamilyBatchSampler,
    TrainingPairsProvenance,
)
from fashion.task4.models import (
    B1_WEIGHT_ORIGIN,
    SCRATCH_WEIGHT_ORIGIN,
    build_retrieval_encoder,
)
from fashion.train.registry import (
    TASK4_RUN_COLUMNS as RUN_COLUMNS,
)
from fashion.train.registry import (
    Task4RunRegistry as RunRegistry,
)

training = importlib.import_module("fashion.task4.training")


def _write_malicious_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousCheckpointValue:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return _write_malicious_marker, (str(self.marker),)


def test_task4_package_exports_stable_training_interfaces() -> None:
    assert task4.TrainingHyperparameters is training.TrainingHyperparameters
    assert task4.TrainingSessionConfig is training.TrainingSessionConfig
    assert task4.SourcePolicy is training.SourcePolicy
    assert task4.AugmentationPolicy is training.AugmentationPolicy
    assert task4.TrainingPairsProvenance is TrainingPairsProvenance
    assert task4.CandidateConfig is training.CandidateConfig
    assert task4.TrainingResult is training.TrainingResult
    assert task4.compute_batch_loss is training.compute_batch_loss
    assert task4.load_checkpoint is training.load_checkpoint
    assert task4.run_training_attempt is training.run_training_attempt
    assert task4.validate_training_loader is training.validate_training_loader
    assert (
        task4.validate_checkpoint_registry_binding
        is training.validate_checkpoint_registry_binding
    )


def _running_row(run_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        **{column: "" for column in RUN_COLUMNS},
        "schema_version": "1",
        "run_id": run_id,
        "started_at_utc": "2025-08-29T01:02:03Z",
        "task": "task4",
        "run_kind": "candidate",
        "status": "running",
        "fold": 1,
        "method": "R1",
        "architecture": "resnet18",
        "objective": "vicreg",
        "source_policy": "teacher_v1_pairs",
        "pretrained": False,
        "weight_origin": SCRATCH_WEIGHT_ORIGIN,
        "deployment_eligibility": "eligible",
        "seed": 2753,
        "embedding_dim": 128,
        "planned_epochs": 100,
        "config_hash": "a" * 64,
        "split_fingerprint": "b" * 64,
        "git_commit": "c" * 40,
        "dirty_tree": False,
    }
    row.update(overrides)
    return row


def _session(
    *,
    run_id: str = "run-1",
    run_kind: str | None = None,
    candidate: str = "R1",
    architecture: str = "resnet18",
    objective: str = "vicreg",
    source_policy: Any = None,
    augmentation_policy: Any = None,
    parent_run_id: str | None = None,
    hyperparameters: Any = None,
) -> Any:
    return training.TrainingSessionConfig(
        run_id=run_id,
        run_kind=run_kind or ("benchmark" if candidate == "B1" else "candidate"),
        candidate=training.CandidateConfig(candidate, architecture),
        hyperparameters=hyperparameters or training.TrainingHyperparameters(),
        objective=objective,
        source_policy=source_policy or training.SourcePolicy.TEACHER_V1_PAIRS,
        augmentation_policy=augmentation_policy
        or (
            training.AugmentationPolicy.GEOMETRY
            if candidate == "R3"
            else training.AugmentationPolicy.NONE
        ),
        validation_fold=1,
        split_fingerprint="b" * 64,
        parent_run_id=parent_run_id,
    )


def _bound_row(session: Any, **overrides: object) -> dict[str, object]:
    row = _running_row(session.run_id)
    row.update(session.expected_registry_identity.as_dict())
    row.update(overrides)
    return row


def test_frozen_training_config_has_the_approved_recipe() -> None:
    config = training.TrainingHyperparameters()

    assert config.seed == 2753
    assert config.product_batch_size == 64
    assert config.images_per_product == 2
    assert config.learning_rate == 3e-4
    assert config.weight_decay == 1e-4
    assert config.warmup_epochs == 5
    assert config.minimum_learning_rate == 1e-6
    assert config.gradient_clip_norm == 5.0
    assert config.amp_initial_scale == 1024.0
    assert config.amp_growth_interval == 40901
    assert config.planned_epochs == 100
    assert config.checkpoint_epochs == (20, 40, 60, 80, 100)
    with pytest.raises(FrozenInstanceError):
        config.seed = 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"product_batch_size": 0}, "batch"),
        ({"images_per_product": 1}, "two"),
        ({"learning_rate": math.nan}, "learning rate"),
        ({"weight_decay": -1.0}, "weight decay"),
        ({"warmup_epochs": 0}, "warm-up"),
        ({"planned_epochs": 5}, "warm-up"),
        ({"minimum_learning_rate": 4e-4}, "minimum"),
        ({"gradient_clip_norm": 0.0}, "clip"),
        ({"amp_initial_scale": 0.0}, "AMP"),
        ({"amp_growth_interval": 40900}, "growth"),
        ({"checkpoint_epochs": [20, 40, 60, 80, 100]}, "tuple"),
        ({"checkpoint_epochs": (20, 20, 100)}, "checkpoint"),
    ],
)
def test_training_config_rejects_invalid_overrides(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        training.TrainingHyperparameters(**override)


def test_step_schedule_has_exact_warmup_cosine_boundaries() -> None:
    config = training.TrainingHyperparameters(
        warmup_epochs=2,
        planned_epochs=4,
        checkpoint_epochs=(4,),
    )

    observed = [
        training.learning_rate_at_step(step, steps_per_epoch=2, config=config)
        for step in (0, 3, 4, 7)
    ]

    assert observed == pytest.approx([7.5e-5, 3e-4, 3e-4, 1e-6], abs=1e-12)


def test_scheduler_state_round_trip_preserves_the_next_rate() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    config = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=3,
        checkpoint_epochs=(3,),
    )
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = training.WarmupCosineScheduler(optimizer, steps_per_epoch=2, config=config)
    scheduler.step()
    state = scheduler.state_dict()

    other_optimizer = torch.optim.AdamW([nn.Parameter(torch.tensor(1.0))], lr=9e-4)
    restored = training.WarmupCosineScheduler(
        other_optimizer,
        steps_per_epoch=2,
        config=config,
    )
    restored.load_state_dict(state)

    assert restored.step_index == scheduler.step_index
    assert other_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def _random_snapshot() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


def test_deterministic_setup_repeats_python_numpy_and_torch() -> None:
    training.configure_determinism(2753)
    first = _random_snapshot()
    training.configure_determinism(2753)
    second = _random_snapshot()

    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_worker_seed_derivation_is_stable_and_distinct() -> None:
    assert training.derive_worker_seed(2753, 0) == 2753
    assert training.derive_worker_seed(2753, 1) == 2754
    assert training.derive_worker_seed(2**32 - 1, 1) == 0
    with pytest.raises(ValueError, match="worker"):
        training.derive_worker_seed(2753, -1)


def test_worker_initializer_is_spawn_picklable() -> None:
    initializer = training.make_worker_init_fn(2753)

    restored = pickle.loads(pickle.dumps(initializer))
    restored(3)
    first = _random_snapshot()
    restored(3)
    second = _random_snapshot()

    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_seeded_data_generators_repeat_order() -> None:
    first = torch.randperm(20, generator=training.make_data_generator(2753))
    second = torch.randperm(20, generator=training.make_data_generator(2753))

    assert torch.equal(first, second)


def test_amp_is_disabled_on_cpu_and_enabled_on_available_cuda() -> None:
    assert not training.amp_is_enabled(torch.device("cpu"))
    scaler = training.make_grad_scaler(torch.device("cpu"))
    assert not scaler.is_enabled()
    assert scaler.get_scale() == 1.0
    with pytest.raises(ValueError, match="AMP"):
        training.make_grad_scaler(torch.device("cpu"), initial_scale=0.0)
    if torch.cuda.is_available():
        assert training.amp_is_enabled(torch.device("cuda"))
        cuda_scaler = training.make_grad_scaler(torch.device("cuda"))
        assert cuda_scaler.is_enabled()
        assert cuda_scaler.get_scale() == pytest.approx(1024.0)


def test_grad_scaler_uses_the_frozen_growth_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class CapturingScaler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(training.torch.amp, "GradScaler", CapturingScaler)

    training.make_grad_scaler(torch.device("cpu"))

    assert calls == [
        {
            "args": ("cuda",),
            "kwargs": {
                "enabled": False,
                "init_scale": 1024.0,
                "growth_interval": 40901,
            },
        }
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_grad_scaler_uses_the_declared_safe_initial_scale() -> None:
    config = training.TrainingHyperparameters()

    scaler = training.make_grad_scaler(
        torch.device("cuda"),
        initial_scale=config.amp_initial_scale,
        growth_interval=config.amp_growth_interval,
    )

    assert scaler.is_enabled()
    assert scaler.get_scale() == pytest.approx(config.amp_initial_scale)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        del images
        raise AssertionError("encode must not run during training")


def _pair_batch() -> dict[str, Any]:
    return {
        "id": torch.tensor([1, 2, 3]),
        "teacher": torch.tensor(
            [
                [[[1.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[0.0]], [[0.0]], [[1.0]]],
            ]
        ),
        "v1": torch.tensor(
            [
                [[[3.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[0.0]], [[0.0]], [[2.0]]],
            ]
        ),
        "product_family_group": ["same", "same", "other"],
        "sha256": ["sha-1", "sha-2", "sha-3"],
        "duplicate_group": ["dup-1", "dup-2", "dup-3"],
    }


@pytest.mark.parametrize("objective", ["R1", "R2", "R3", "B1"])
def test_vicreg_routes_use_forward_projections_without_encode(objective: str) -> None:
    result = training.compute_batch_loss(_TinyEncoder(), _pair_batch(), objective=objective)

    assert result.total.ndim == 0
    assert torch.isfinite(result.total)
    assert result.product_embeddings is None


def test_r4_averages_unnormalized_views_then_normalizes_one_product_embedding() -> None:
    result = training.compute_batch_loss(_TinyEncoder(), _pair_batch(), objective="R4")

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(result.product_embeddings, expected)
    assert torch.isfinite(result.total)


def test_r4_rejects_a_zero_product_embedding() -> None:
    model = _TinyEncoder()
    with torch.no_grad():
        model.projection.weight.zero_()

    with pytest.raises(ValueError, match="non-zero"):
        training.compute_batch_loss(model, _pair_batch(), objective="R4")


class _PixelAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixels = nn.Parameter(torch.zeros(2, 3, 2, 2))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert images.shape == (2, 3, 2, 2)
        return self.pixels, images.mean(dim=(2, 3))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        del images
        raise AssertionError("encode must not run during training")


def test_r5_concatenates_views_and_masks_and_excludes_padding_gradients() -> None:
    model = _PixelAutoencoder()
    batch = {
        "teacher": torch.ones(1, 3, 2, 2),
        "v1": torch.full((1, 3, 2, 2), 2.0),
        "teacher_content_mask": torch.tensor([[[True, False], [False, False]]]),
        "v1_content_mask": torch.tensor([[[False, False], [False, True]]]),
    }

    result = training.compute_batch_loss(model, batch, objective="R5")
    result.total.backward()

    assert result.included_values == 6
    assert torch.count_nonzero(model.pixels.grad[0, :, 0, 0]) == 3
    assert torch.count_nonzero(model.pixels.grad[1, :, 1, 1]) == 3
    assert torch.count_nonzero(model.pixels.grad[:, :, 0, 1]) == 0
    assert torch.count_nonzero(model.pixels.grad[:, :, 1, 0]) == 0


@pytest.mark.parametrize("failure", ["loss", "gradient", "parameter"])
def test_optimization_step_rejects_non_finite_training_state(failure: str) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = training.make_grad_scaler(torch.device("cpu"))
    if failure == "loss":
        loss = parameter * torch.tensor(float("nan"))
    elif failure == "gradient":
        loss = parameter.square()
        parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    else:
        loss = parameter.square()
        with torch.no_grad():
            parameter.fill_(float("inf"))

    with pytest.raises(training.NonFiniteTrainingError, match=failure):
        training.apply_optimization_step(
            loss,
            model=nn.ParameterList([parameter]),
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=5.0,
        )


class _NormCheckingSGD(torch.optim.SGD):
    def __init__(self, parameters: Any, *, maximum_norm: float) -> None:
        super().__init__(parameters, lr=0.0)
        self.maximum_norm = maximum_norm
        self.observed_norm = math.inf

    def step(self, closure: Any = None) -> Any:
        parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        self.observed_norm = float(
            torch.linalg.vector_norm(
                torch.stack([parameter.grad.detach().norm() for parameter in parameters])
            )
        )
        assert self.observed_norm <= self.maximum_norm + 1e-6
        return super().step(closure)


def test_gradient_clipping_happens_before_optimizer_step() -> None:
    parameter = nn.Parameter(torch.tensor([100.0, -100.0]))
    model = nn.ParameterList([parameter])
    optimizer = _NormCheckingSGD(model.parameters(), maximum_norm=0.1)

    observed = training.apply_optimization_step(
        parameter.square().sum(),
        model=model,
        optimizer=optimizer,
        scaler=training.make_grad_scaler(torch.device("cpu")),
        gradient_clip_norm=0.1,
    )

    assert observed > 0.1
    assert optimizer.observed_norm == pytest.approx(0.1, abs=1e-6)


class _CorruptingSGD(torch.optim.SGD):
    def step(self, closure: Any = None) -> Any:
        result = super().step(closure)
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("nan"))
        return result


def test_optimization_step_detects_post_optimizer_parameter_corruption() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    model = nn.ParameterList([parameter])
    optimizer = _CorruptingSGD(model.parameters(), lr=0.1)

    with pytest.raises(training.NonFiniteTrainingError, match="parameter"):
        training.apply_optimization_step(
            parameter.square(),
            model=model,
            optimizer=optimizer,
            scaler=training.make_grad_scaler("cpu"),
            gradient_clip_norm=5.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_amp_orders_scale_backward_unscale_clip_step_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    parameter = nn.Parameter(torch.tensor(100.0, device="cuda"))
    model = nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = training.make_grad_scaler("cuda")
    parameter.register_hook(lambda gradient: events.append("backward") or gradient)
    real_scale = scaler.scale
    real_unscale = scaler.unscale_
    real_step = scaler.step
    real_update = scaler.update
    real_clip = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(scaler, "scale", lambda loss: events.append("scale") or real_scale(loss))
    monkeypatch.setattr(
        scaler,
        "unscale_",
        lambda selected: events.append("unscale") or real_unscale(selected),
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: events.append("clip") or real_clip(*args, **kwargs),
    )
    monkeypatch.setattr(
        scaler,
        "step",
        lambda selected: events.append("step") or real_step(selected),
    )
    monkeypatch.setattr(scaler, "update", lambda: events.append("update") or real_update())

    training.apply_optimization_step(
        parameter.square(),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=0.1,
    )

    assert events == ["scale", "backward", "unscale", "clip", "step", "update"]


def test_candidate_config_enforces_approved_factory_routes() -> None:
    assert training.CandidateConfig("R2", "resnet34").architecture == "resnet34"
    assert training.CandidateConfig("B1", "resnet18").pretrained
    with pytest.raises(ValueError, match="R1"):
        training.CandidateConfig("R1", "resnet34")
    with pytest.raises(ValueError, match="R2"):
        training.CandidateConfig("R2", "resnet18")
    with pytest.raises(ValueError, match="R5"):
        training.CandidateConfig("R5", "resnet34")
    with pytest.raises(ValueError, match="B1"):
        training.CandidateConfig("B1", "resnet34")


def test_training_session_is_immutable_and_owns_complete_identity() -> None:
    session = _session(parent_run_id="parent-1")

    assert session.model_metadata.architecture == "resnet18"
    assert session.model_metadata.weight_origin == SCRATCH_WEIGHT_ORIGIN
    assert session.expected_registry_identity.config_hash == session.config_hash
    assert session.expected_registry_identity.split_fingerprint == "b" * 64
    assert json.loads(session.config_json) == {
        "augmentation_policy": "none",
        "candidate": {
            "architecture": "resnet18",
            "candidate": "R1",
            "pretrained": False,
        },
        "hyperparameters": {
            "checkpoint_epochs": [20, 40, 60, 80, 100],
            "amp_initial_scale": 1024.0,
            "amp_growth_interval": 40901,
            "gradient_clip_norm": 5.0,
            "images_per_product": 2,
            "learning_rate": 0.0003,
            "minimum_learning_rate": 0.000001,
            "planned_epochs": 100,
            "product_batch_size": 64,
            "seed": 2753,
            "warmup_epochs": 5,
            "weight_decay": 0.0001,
        },
        "model_metadata": {
            "architecture": "resnet18",
            "deployment_eligible": True,
            "pretrained": False,
            "weight_origin": SCRATCH_WEIGHT_ORIGIN,
        },
        "objective": "vicreg",
        "parent_run_id": "parent-1",
        "source_policy": "teacher_v1_pairs",
        "split_fingerprint": "b" * 64,
        "validation_fold": 1,
    }
    with pytest.raises(FrozenInstanceError):
        session.objective = "changed"


def test_same_architecture_r4_parent_and_policy_change_config_identity() -> None:
    first = _session(
        candidate="R4",
        objective="vicreg_triplet",
        augmentation_policy=training.AugmentationPolicy.NONE,
        parent_run_id="r3-a",
    )
    second = _session(
        candidate="R4",
        objective="vicreg_triplet",
        augmentation_policy=training.AugmentationPolicy.GEOMETRY,
        parent_run_id="r3-b",
    )

    assert first.model_metadata == second.model_metadata
    assert first.config_hash != second.config_hash
    assert first.config_json != second.config_json


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"objective": "wrong"}, "objective"),
        ({"source_policy": "teacher_v1_pairs"}, "source"),
        ({"augmentation_policy": "none"}, "augmentation"),
        ({"validation_fold": 5}, "fold"),
        ({"split_fingerprint": "bad"}, "split"),
        ({"parent_run_id": ""}, "parent"),
    ],
)
def test_training_session_rejects_invalid_bound_parts(
    override: dict[str, object],
    message: str,
) -> None:
    values = {
        "run_id": "run-1",
        "run_kind": "candidate",
        "candidate": training.CandidateConfig("R1", "resnet18"),
        "hyperparameters": training.TrainingHyperparameters(),
        "objective": "vicreg",
        "source_policy": training.SourcePolicy.TEACHER_V1_PAIRS,
        "augmentation_policy": training.AugmentationPolicy.NONE,
        "validation_fold": 1,
        "split_fingerprint": "b" * 64,
        "parent_run_id": None,
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        training.TrainingSessionConfig(**values)


@pytest.mark.parametrize(
    ("candidate", "run_kind", "augmentation", "message"),
    [
        ("R1", "candidate", "geometry", "R1"),
        ("R2", "candidate", "geometry", "R2"),
        ("R3", "candidate", "none", "R3"),
        ("R5", "candidate", "geometry", "R5"),
        ("B1", "benchmark", "geometry", "B1"),
        ("B1", "candidate", "none", "run kind"),
        ("R1", "benchmark", "none", "scratch"),
    ],
)
def test_training_session_enforces_candidate_policy_and_run_kind(
    candidate: str,
    run_kind: str,
    augmentation: str,
    message: str,
) -> None:
    objective = {
        "R1": "vicreg",
        "R2": "vicreg",
        "R3": "vicreg",
        "R5": "content_mask_mse",
        "B1": "vicreg",
    }[candidate]
    architecture = "resnet34" if candidate == "R2" else "resnet18"

    with pytest.raises(ValueError, match=message):
        _session(
            candidate=candidate,
            architecture=architecture,
            objective=objective,
            run_kind=run_kind,
            augmentation_policy=training.AugmentationPolicy(augmentation),
        )


def test_b1_smoke_and_r4_inherited_geometry_are_valid() -> None:
    assert _session(candidate="B1", run_kind="smoke").run_kind == "smoke"
    r4 = _session(
        candidate="R4",
        objective="vicreg_triplet",
        augmentation_policy=training.AugmentationPolicy.GEOMETRY,
    )
    assert r4.augmentation_policy is training.AugmentationPolicy.GEOMETRY


def test_optimizer_builder_uses_frozen_adamw_values() -> None:
    model = nn.Linear(2, 1)
    config = training.TrainingHyperparameters()

    optimizer = training.build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_canonical_config_json_and_hash_ignore_mapping_order() -> None:
    left = {"z": 2, "a": {"b": 1}}
    right = {"a": {"b": 1}, "z": 2}

    assert training.canonical_config_json(left) == '{"a":{"b":1},"z":2}'
    assert training.configuration_sha256(left) == training.configuration_sha256(right)


def _checkpoint(
    tmp_path: Path,
) -> tuple[Any, Any, nn.Module, torch.optim.Optimizer, Any, Any]:
    model = build_retrieval_encoder("resnet18")
    hyperparameters = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=2,
        checkpoint_epochs=(1, 2),
    )
    session = _session(hyperparameters=hyperparameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    scheduler = training.WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=1,
        config=hyperparameters,
    )
    scaler = training.make_grad_scaler(torch.device("cpu"))
    record = training.save_checkpoint(
        tmp_path / "model-1.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        session=session,
        score=0.42,
    )
    return record, session, model, optimizer, scheduler, scaler


def _load_checkpoint(record: Any, session: Any) -> Any:
    return training.load_checkpoint(
        record.path,
        expected_sha256=record.sha256,
        expected_config_hash=session.config_hash,
        expected_split_fingerprint=session.split_fingerprint,
        expected_weight_origin=session.model_metadata.weight_origin,
        expected_parent_run_id=session.parent_run_id,
        expected_run_id=session.run_id,
        expected_run_kind=session.run_kind,
    )


def test_checkpoint_round_trip_reconstructs_factory_and_training_state(tmp_path: Path) -> None:
    record, session, model, optimizer, scheduler, scaler = _checkpoint(tmp_path)

    loaded = _load_checkpoint(record, session)
    payload = torch.load(record.path, map_location="cpu", weights_only=True)

    assert loaded.epoch == 1
    assert loaded.score == pytest.approx(0.42)
    assert record.run_id == session.run_id
    assert record.run_kind == session.run_kind
    assert loaded.session.run_id == session.run_id
    assert loaded.session.run_kind == session.run_kind
    assert type(loaded.model) is type(model)
    assert loaded.model.metadata == model.metadata
    assert loaded.optimizer_state == optimizer.state_dict()
    assert loaded.scheduler_state == scheduler.state_dict()
    assert loaded.scaler_state == scaler.state_dict()
    assert payload["amp_growth_interval"] == session.hyperparameters.amp_growth_interval
    assert isinstance(loaded.optimizer, torch.optim.AdamW)
    assert isinstance(loaded.scheduler, training.WarmupCosineScheduler)
    assert loaded.hyperparameters.amp_growth_interval == 40901
    assert isinstance(loaded.make_resume_scaler("cpu"), torch.amp.GradScaler)
    assert loaded.parent_run_id is None


def test_resume_scaler_uses_checkpoint_growth_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    loaded = _load_checkpoint(record, session)
    calls: list[dict[str, object]] = []

    class DummyScaler:
        def load_state_dict(self, state: dict[str, object]) -> None:
            calls.append({"loaded_state": state})

    def fake_scaler(device: torch.device | str, **kwargs: object) -> DummyScaler:
        calls.append({"device": device, **kwargs})
        return DummyScaler()

    monkeypatch.setattr(training, "make_grad_scaler", fake_scaler)

    loaded.make_resume_scaler("cpu")

    assert calls[0]["growth_interval"] == loaded.hyperparameters.amp_growth_interval


def test_checkpoint_rejects_corrupt_file_digest(tmp_path: Path) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    with record.path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(training.CheckpointValidationError, match="SHA-256"):
        _load_checkpoint(record, session)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("expected_config_hash", "c" * 64, "config"),
        ("expected_split_fingerprint", "c" * 64, "split"),
        ("expected_weight_origin", "wrong", "weight origin"),
        ("expected_parent_run_id", "wrong-parent", "parent"),
        ("expected_run_id", "wrong-run", "run ID"),
        ("expected_run_kind", "stability", "run kind"),
    ],
)
def test_checkpoint_rejects_expected_identity_mismatches(
    tmp_path: Path,
    keyword: str,
    value: str,
    message: str,
) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    expected = {
        "expected_config_hash": session.config_hash,
        "expected_split_fingerprint": session.split_fingerprint,
        "expected_weight_origin": session.model_metadata.weight_origin,
        "expected_parent_run_id": session.parent_run_id,
        "expected_run_id": session.run_id,
        "expected_run_kind": session.run_kind,
    }
    expected[keyword] = value

    with pytest.raises(training.CheckpointValidationError, match=message):
        training.load_checkpoint(
            record.path,
            expected_sha256=record.sha256,
            **expected,
        )


def test_checkpoint_rejects_state_key_mismatch_even_with_valid_file_digest(
    tmp_path: Path,
) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    payload = torch.load(record.path, map_location="cpu", weights_only=True)
    payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
    torch.save(payload, record.path)
    changed_digest = hashlib.sha256(record.path.read_bytes()).hexdigest()

    with pytest.raises(training.CheckpointValidationError, match="state-dict keys"):
        record = training.CheckpointRecord(
            epoch=record.epoch,
            path=record.path,
            sha256=changed_digest,
            config_hash=record.config_hash,
            score=record.score,
            split_fingerprint=record.split_fingerprint,
            weight_origin=record.weight_origin,
            parent_run_id=record.parent_run_id,
            run_id=record.run_id,
            run_kind=record.run_kind,
        )
        _load_checkpoint(record, session)


def test_checkpoint_rejects_internal_canonical_config_tampering(tmp_path: Path) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    payload = torch.load(record.path, map_location="cpu", weights_only=True)
    decoded = json.loads(payload["canonical_config_json"])
    decoded["hyperparameters"]["seed"] = 99
    payload["canonical_config_json"] = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    torch.save(payload, record.path)
    changed_digest = hashlib.sha256(record.path.read_bytes()).hexdigest()

    with pytest.raises(training.CheckpointValidationError, match="config hash"):
        training.load_checkpoint(
            record.path,
            expected_sha256=changed_digest,
            expected_config_hash=session.config_hash,
            expected_split_fingerprint=session.split_fingerprint,
            expected_weight_origin=session.model_metadata.weight_origin,
            expected_parent_run_id=session.parent_run_id,
            expected_run_id=session.run_id,
            expected_run_kind=session.run_kind,
        )


def test_checkpoint_load_rejects_epoch_outside_session_milestones(
    tmp_path: Path,
) -> None:
    hyperparameters = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=3,
        checkpoint_epochs=(1, 3),
    )
    session = _session(hyperparameters=hyperparameters)
    model = build_retrieval_encoder("resnet18")
    optimizer = training.build_optimizer(model, hyperparameters)
    scheduler = training.WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=1,
        config=hyperparameters,
    )
    record = training.save_checkpoint(
        tmp_path / "outside-milestone.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=training.make_grad_scaler("cpu"),
        epoch=1,
        session=session,
    )
    payload = torch.load(record.path, map_location="cpu", weights_only=True)
    payload["epoch"] = 2
    torch.save(payload, record.path)
    digest = hashlib.sha256(record.path.read_bytes()).hexdigest()

    with pytest.raises(training.CheckpointValidationError, match="milestone"):
        training.load_checkpoint(
            record.path,
            expected_sha256=digest,
            expected_config_hash=session.config_hash,
            expected_split_fingerprint=session.split_fingerprint,
            expected_weight_origin=session.model_metadata.weight_origin,
            expected_parent_run_id=session.parent_run_id,
            expected_run_id=session.run_id,
            expected_run_kind=session.run_kind,
        )


@pytest.mark.parametrize(
    "missing",
    [
        "expected_config_hash",
        "expected_split_fingerprint",
        "expected_weight_origin",
        "expected_parent_run_id",
        "expected_run_id",
        "expected_run_kind",
    ],
)
def test_checkpoint_load_requires_every_expected_identity(
    tmp_path: Path,
    missing: str,
) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    expected = {
        "expected_config_hash": session.config_hash,
        "expected_split_fingerprint": session.split_fingerprint,
        "expected_weight_origin": session.model_metadata.weight_origin,
        "expected_parent_run_id": session.parent_run_id,
        "expected_run_id": session.run_id,
        "expected_run_kind": session.run_kind,
    }
    expected.pop(missing)

    with pytest.raises(TypeError, match=missing):
        training.load_checkpoint(
            record.path,
            expected_sha256=record.sha256,
            **expected,
        )


def test_checkpoint_weights_only_rejects_custom_pickle_without_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed.txt"
    checkpoint = tmp_path / "malicious.pt"
    torch.save({"payload": _MaliciousCheckpointValue(marker)}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(training.CheckpointValidationError, match="cannot be loaded"):
        training.load_checkpoint(
            checkpoint,
            expected_sha256=digest,
            expected_config_hash="a" * 64,
            expected_split_fingerprint="b" * 64,
            expected_weight_origin=SCRATCH_WEIGHT_ORIGIN,
            expected_parent_run_id=None,
            expected_run_id="malicious",
            expected_run_kind="candidate",
        )

    assert not marker.exists()


def test_b1_checkpoint_load_rebuilds_offline_then_loads_full_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_module._build_b1_checkpoint_encoder()
    hyperparameters = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=2,
        checkpoint_epochs=(1, 2),
    )
    session = _session(
        candidate="B1",
        architecture="resnet18",
        hyperparameters=hyperparameters,
    )
    optimizer = training.build_optimizer(model, hyperparameters)
    scheduler = training.WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=1,
        config=hyperparameters,
    )
    record = training.save_checkpoint(
        tmp_path / "b1.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=training.make_grad_scaler("cpu"),
        epoch=1,
        session=session,
        score=0.3,
    )
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network access")),
    )

    loaded = _load_checkpoint(record, session)

    assert loaded.model.metadata.pretrained
    assert loaded.model.metadata.weight_origin == B1_WEIGHT_ORIGIN
    assert all(
        torch.equal(loaded.model.state_dict()[key], value)
        for key, value in model.state_dict().items()
    )


def test_cpu_map_location_retains_nonempty_cuda_scaler_state(tmp_path: Path) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    payload = torch.load(record.path, map_location="cpu", weights_only=True)
    raw_scaler_state = {
        "scale": 8192.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 7,
    }
    payload["scaler_state_dict"] = raw_scaler_state
    torch.save(payload, record.path)
    changed_digest = hashlib.sha256(record.path.read_bytes()).hexdigest()

    loaded = training.load_checkpoint(
        record.path,
        expected_sha256=changed_digest,
        expected_config_hash=session.config_hash,
        expected_split_fingerprint=session.split_fingerprint,
        expected_weight_origin=session.model_metadata.weight_origin,
        expected_parent_run_id=session.parent_run_id,
        expected_run_id=session.run_id,
        expected_run_kind=session.run_kind,
        map_location="cpu",
    )

    assert loaded.scaler_state == raw_scaler_state
    with pytest.raises(ValueError, match="CUDA resume device"):
        loaded.make_resume_scaler("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_empty_cpu_scaler_checkpoint_makes_fresh_enabled_cuda_scaler(
    tmp_path: Path,
) -> None:
    record, session, *_ = _checkpoint(tmp_path)
    loaded = _load_checkpoint(record, session)
    assert loaded.raw_scaler_state == {}

    scaler = loaded.make_resume_scaler("cuda")

    assert scaler.is_enabled()


def test_atomic_checkpoint_save_interruption_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, session, model, optimizer, scheduler, scaler = _checkpoint(tmp_path)
    original = record.path.read_bytes()

    def interrupted_save(payload: object, path: Path) -> None:
        del payload
        Path(path).write_bytes(b"partial")
        raise OSError("injected save interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)

    with pytest.raises(OSError, match="interruption"):
        training.save_checkpoint(
            record.path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=2,
            session=session,
            score=0.4,
        )

    assert record.path.read_bytes() == original
    assert list(tmp_path.glob(".model-1.pt.*.tmp")) == []


def test_checkpoint_to_registry_binding_validates_task6_completion_input(
    tmp_path: Path,
) -> None:
    record, session, model, optimizer, scheduler, scaler = _checkpoint(tmp_path)
    best = training.save_checkpoint(
        tmp_path / "model-2.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        session=session,
        score=0.7,
    )
    result = training.TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=(record, best),
        best_checkpoint=best,
    )
    row = _bound_row(
        session,
        selected_epoch=best.epoch,
        checkpoint_path=str(best.path),
        checkpoint_sha256=best.sha256,
    )

    training.validate_checkpoint_registry_binding(
        result,
        session=session,
        registry_row=row,
    )

    for field, value in (
        ("config_hash", "c" * 64),
        ("split_fingerprint", "c" * 64),
        ("weight_origin", "wrong"),
        ("parent_run_id", "wrong"),
        ("method", "R4"),
        ("objective", "wrong"),
        ("source_policy", "wrong"),
        ("fold", 2),
        ("selected_epoch", record.epoch),
        ("checkpoint_path", str(tmp_path / "wrong.pt")),
        ("checkpoint_sha256", "d" * 64),
    ):
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            training.validate_checkpoint_registry_binding(
                result,
                session=session,
                registry_row={**row, field: value},
            )


def test_completion_binding_rejects_cross_run_wrong_kind_and_incomplete_results(
    tmp_path: Path,
) -> None:
    first, session, model, optimizer, scheduler, scaler = _checkpoint(tmp_path)
    second = training.save_checkpoint(
        tmp_path / "model-2.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        session=session,
        score=0.7,
    )
    row = _bound_row(
        session,
        selected_epoch=second.epoch,
        checkpoint_path=str(second.path),
        checkpoint_sha256=second.sha256,
    )

    for records, message in (
        ((replace(first, run_id="other"), second), "run ID"),
        ((replace(first, run_kind="stability"), second), "run kind"),
        ((second,), "milestone"),
    ):
        result = training.TrainingResult(
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoints=records,
            best_checkpoint=second,
        )
        with pytest.raises(ValueError, match=message):
            training.validate_checkpoint_registry_binding(
                result,
                session=session,
                registry_row=row,
            )

    wrong_best = training.TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=(first, second),
        best_checkpoint=first,
    )
    with pytest.raises(ValueError, match="recomputed best"):
        training.validate_checkpoint_registry_binding(
            wrong_best,
            session=session,
            registry_row=row,
        )


def test_completion_binding_reads_payload_instead_of_trusting_record_fields(
    tmp_path: Path,
) -> None:
    first, session, model, optimizer, scheduler, scaler = _checkpoint(tmp_path)
    second = training.save_checkpoint(
        tmp_path / "model-2.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        session=session,
        score=0.7,
    )
    payload = torch.load(second.path, map_location="cpu", weights_only=True)
    payload["run_id"] = "payload-other-run"
    torch.save(payload, second.path)
    forged = replace(
        second,
        sha256=hashlib.sha256(second.path.read_bytes()).hexdigest(),
        run_id=session.run_id,
    )
    result = training.TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=(first, forged),
        best_checkpoint=forged,
    )
    row = _bound_row(
        session,
        selected_epoch=forged.epoch,
        checkpoint_path=str(forged.path),
        checkpoint_sha256=forged.sha256,
    )

    with pytest.raises(ValueError, match="run ID"):
        training.validate_checkpoint_registry_binding(
            result,
            session=session,
            registry_row=row,
        )


def _completion_bundle(
    tmp_path: Path,
) -> tuple[Any, Any, Any, dict[str, object]]:
    first, session, model, optimizer, scheduler, scaler = _checkpoint(tmp_path)
    second = training.save_checkpoint(
        tmp_path / "model-2.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        session=session,
        score=0.7,
    )
    result = training.TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=(first, second),
        best_checkpoint=second,
    )
    row = _bound_row(
        session,
        selected_epoch=second.epoch,
        checkpoint_path=str(second.path),
        checkpoint_sha256=second.sha256,
    )
    return result, session, second, row


def test_completion_rejects_header_only_selected_checkpoint(tmp_path: Path) -> None:
    result, session, selected, row = _completion_bundle(tmp_path)
    payload = torch.load(selected.path, map_location="cpu", weights_only=True)
    header_keys = {
        "checkpoint_schema_version",
        "run_id",
        "run_kind",
        "epoch",
        "score",
        "model_metadata",
        "canonical_config_json",
        "config_hash",
        "amp_growth_interval",
        "split_fingerprint",
        "weight_origin",
        "parent_run_id",
    }
    torch.save({key: payload[key] for key in header_keys}, selected.path)
    forged = replace(
        selected,
        sha256=hashlib.sha256(selected.path.read_bytes()).hexdigest(),
    )
    forged_result = replace(
        result,
        checkpoints=(result.checkpoints[0], forged),
        best_checkpoint=forged,
    )
    forged_row = {
        **row,
        "checkpoint_sha256": forged.sha256,
    }

    with pytest.raises(training.CheckpointValidationError, match="model state"):
        training.validate_checkpoint_registry_binding(
            forged_result,
            session=session,
            registry_row=forged_row,
        )


def test_completion_rejects_corrupt_non_selected_checkpoint(tmp_path: Path) -> None:
    result, session, _, row = _completion_bundle(tmp_path)
    first = result.checkpoints[0]
    with first.path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(training.CheckpointValidationError, match="SHA-256"):
        training.validate_checkpoint_registry_binding(
            result,
            session=session,
            registry_row=row,
        )


def test_completion_rejects_missing_non_selected_checkpoint(tmp_path: Path) -> None:
    result, session, _, row = _completion_bundle(tmp_path)
    result.checkpoints[0].path.unlink()

    with pytest.raises(FileNotFoundError):
        training.validate_checkpoint_registry_binding(
            result,
            session=session,
            registry_row=row,
        )


def test_completion_rejects_cross_run_non_selected_payload(tmp_path: Path) -> None:
    result, session, _, row = _completion_bundle(tmp_path)
    first = result.checkpoints[0]
    payload = torch.load(first.path, map_location="cpu", weights_only=True)
    payload["run_id"] = "other-payload-run"
    torch.save(payload, first.path)
    forged_first = replace(
        first,
        sha256=hashlib.sha256(first.path.read_bytes()).hexdigest(),
    )
    forged_result = replace(
        result,
        checkpoints=(forged_first, result.checkpoints[1]),
    )

    with pytest.raises(training.CheckpointValidationError, match="run ID"):
        training.validate_checkpoint_registry_binding(
            forged_result,
            session=session,
            registry_row=row,
        )


def test_completion_rejects_caller_forged_non_selected_score(tmp_path: Path) -> None:
    result, session, _, row = _completion_bundle(tmp_path)
    forged_first = replace(result.checkpoints[0], score=0.1)
    forged_result = replace(
        result,
        checkpoints=(forged_first, result.checkpoints[1]),
    )

    with pytest.raises(ValueError, match="score"):
        training.validate_checkpoint_registry_binding(
            forged_result,
            session=session,
            registry_row=row,
        )


def test_completion_validates_milestone_models_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, session, _, row = _completion_bundle(tmp_path)
    live_models: list[weakref.ReferenceType[nn.Module]] = []
    real_factory = training._build_candidate

    def observed_factory(candidate: Any) -> nn.Module:
        assert not any(reference() is not None for reference in live_models)
        model = real_factory(candidate)
        live_models.append(weakref.ref(model))
        return model

    monkeypatch.setattr(training, "_build_candidate", observed_factory)

    training.validate_checkpoint_registry_binding(
        result,
        session=session,
        registry_row=row,
    )

    assert len(live_models) == 2


def test_best_checkpoint_uses_highest_score_and_earlier_exact_tie(tmp_path: Path) -> None:
    session = _session()
    records = (
        training.CheckpointRecord(
            epoch=20,
            path=tmp_path / "20.pt",
            sha256="a" * 64,
            config_hash=session.config_hash,
            score=0.4,
            split_fingerprint=session.split_fingerprint,
            weight_origin=session.model_metadata.weight_origin,
            parent_run_id=session.parent_run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
        ),
        training.CheckpointRecord(
            epoch=40,
            path=tmp_path / "40.pt",
            sha256="b" * 64,
            config_hash=session.config_hash,
            score=0.7,
            split_fingerprint=session.split_fingerprint,
            weight_origin=session.model_metadata.weight_origin,
            parent_run_id=session.parent_run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
        ),
        training.CheckpointRecord(
            epoch=60,
            path=tmp_path / "60.pt",
            sha256="c" * 64,
            config_hash=session.config_hash,
            score=0.7,
            split_fingerprint=session.split_fingerprint,
            weight_origin=session.model_metadata.weight_origin,
            parent_run_id=session.parent_run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
        ),
    )

    assert training.select_best_checkpoint(records).epoch == 40


def test_training_loop_scores_exact_milestones_and_returns_running_result(
    tmp_path: Path,
) -> None:
    config = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=3,
        checkpoint_epochs=(1, 2, 3),
    )
    session = _session(hyperparameters=config)
    model = _TinyEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = training.WarmupCosineScheduler(optimizer, steps_per_epoch=1, config=config)
    epochs_scored: list[int] = []

    def score_callback(candidate_model: nn.Module, epoch: int) -> float:
        assert candidate_model is model
        assert not candidate_model.training
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        epochs_scored.append(epoch)
        return {1: 0.2, 2: 0.5, 3: 0.5}[epoch]

    def checkpoint_callback(epoch: int, score: float) -> Any:
        path = tmp_path / f"{epoch}.pt"
        path.write_bytes(str(epoch).encode("ascii"))
        return training.CheckpointRecord(
            epoch=epoch,
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            config_hash=session.config_hash,
            score=score,
            split_fingerprint=session.split_fingerprint,
            weight_origin=session.model_metadata.weight_origin,
            parent_run_id=session.parent_run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
        )

    result = training.train_epochs(
        model=model,
        batches=[_pair_batch()],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=training.make_grad_scaler(torch.device("cpu")),
        device=torch.device("cpu"),
        session=session,
        score_callback=score_callback,
        checkpoint_callback=checkpoint_callback,
    )

    assert epochs_scored == [1, 2, 3]
    assert [record.epoch for record in result.checkpoints] == [1, 2, 3]
    assert result.best_checkpoint.epoch == 2
    assert result.status == "running"
    assert model.training


def test_scoring_callback_error_restores_prior_training_mode() -> None:
    config = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=2,
        checkpoint_epochs=(1, 2),
    )
    session = _session(hyperparameters=config)
    model = _TinyEncoder()
    optimizer = training.build_optimizer(model, config)
    scheduler = training.WarmupCosineScheduler(optimizer, steps_per_epoch=1, config=config)

    def fail_scoring(candidate_model: nn.Module, epoch: int) -> float:
        del epoch
        assert not candidate_model.training
        assert torch.is_inference_mode_enabled()
        raise RuntimeError("score failed")

    with pytest.raises(RuntimeError, match="score failed"):
        training.train_epochs(
            model=model,
            batches=[_pair_batch()],
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=training.make_grad_scaler("cpu"),
            device="cpu",
            session=session,
            score_callback=fail_scoring,
            checkpoint_callback=lambda epoch, score: (_ for _ in ()).throw(
                AssertionError((epoch, score))
            ),
        )

    assert model.training


class _EpochTarget:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def _family_pair_frame(rows: int = 64) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": index + 1,
                "product_family_group": (
                    f"family-{index // 2}" if index < 32 else f"single-{index}"
                ),
                "sha256": f"sha-{index}",
                "duplicate_group": f"duplicate-{index}",
            }
            for index in range(rows)
        ]
    )


class _PolicyDataset(CrossSourcePairDataset):
    def __init__(
        self,
        *,
        geometry: bool,
        validation_fold: int = 1,
        split_fingerprint: str = "b" * 64,
        rows: int = 64,
    ) -> None:
        self.geometry_policy = object() if geometry else None
        self.validation_fold = validation_fold
        self.split_fingerprint = split_fingerprint
        self.rows = rows
        self.pairs = _family_pair_frame(rows)
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        teacher = torch.zeros(3, 32, 32)
        v1 = torch.zeros(3, 32, 32)
        teacher[index % 3].fill_(1.0)
        v1[(index + 1) % 3].fill_(1.0)
        row = self.pairs.iloc[index]
        return {
            "id": int(row["id"]),
            "teacher": teacher,
            "v1": v1,
            "product_family_group": str(row["product_family_group"]),
            "sha256": str(row["sha256"]),
            "duplicate_group": str(row["duplicate_group"]),
        }


class _DistinctEpochLoader:
    def __init__(self) -> None:
        self.dataset = _PolicyDataset(geometry=False)
        self.sampler = _EpochTarget()
        self.batch_sampler = _EpochTarget()
        self.batch_size = 64

    def __iter__(self) -> Iterator[dict[str, Any]]:
        batch = _pair_batch()
        batch["teacher"] = batch["teacher"].expand(-1, -1, 32, 32).clone()
        batch["v1"] = batch["v1"].expand(-1, -1, 32, 32).clone()
        yield batch


class _LoaderShape(DataLoader[dict[str, Any]]):
    def __init__(
        self,
        *,
        geometry: bool = False,
        family: bool = False,
        family_pairs: pd.DataFrame | None = None,
        family_via_sampler: bool = False,
        batch_size: int | None = 64,
        drop_last: bool = True,
        validation_fold: int = 1,
        split_fingerprint: str = "b" * 64,
        rows: int = 64,
    ) -> None:
        dataset = _PolicyDataset(
            geometry=geometry,
            validation_fold=validation_fold,
            split_fingerprint=split_fingerprint,
            rows=rows,
        )
        if family or family_via_sampler:
            sampler = FamilyBatchSampler(
                dataset.pairs if family_pairs is None else family_pairs
            )
            if family_via_sampler:
                DataLoader.__init__(
                    self,
                    dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            else:
                DataLoader.__init__(self, dataset, batch_sampler=sampler)
        else:
            DataLoader.__init__(
                self,
                dataset,
                batch_size=batch_size,
                drop_last=drop_last,
            )


def test_loader_validation_accepts_candidate_specific_data_contracts() -> None:
    training.validate_training_loader(_session(), _LoaderShape())
    training.validate_training_loader(
        _session(candidate="R3", augmentation_policy=training.AugmentationPolicy.GEOMETRY),
        _LoaderShape(geometry=True),
    )
    training.validate_training_loader(
        _session(candidate="R4", objective="vicreg_triplet"),
        _LoaderShape(family=True),
    )


@pytest.mark.parametrize(
    ("loader_values", "message"),
    [
        ({"validation_fold": 2}, "validation fold"),
        ({"split_fingerprint": "c" * 64}, "split fingerprint"),
    ],
)
def test_real_dataloader_provenance_must_match_session(
    loader_values: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        training.validate_training_loader(_session(), _LoaderShape(**loader_values))


def test_real_dataloader_accepts_matching_fold_split_and_complete_batches() -> None:
    training.validate_training_loader(
        _session(),
        _LoaderShape(
            validation_fold=1,
            split_fingerprint="b" * 64,
            batch_size=64,
            drop_last=True,
        ),
    )


@pytest.mark.parametrize(
    "loader",
    [
        _LoaderShape(batch_size=None, drop_last=False),
        _LoaderShape(batch_size=64, drop_last=False, rows=65),
    ],
)
def test_ordinary_dataloader_rejects_unknown_or_incomplete_batch_contract(
    loader: DataLoader[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="batch size|drop_last"):
        training.validate_training_loader(_session(), loader)


@pytest.mark.parametrize(
    ("candidate", "augmentation", "loader_values", "message"),
    [
        ("R1", "none", {"geometry": True}, "geometry"),
        ("R3", "geometry", {"geometry": False}, "geometry"),
        ("R4", "none", {"family": False}, "FamilyBatchSampler"),
        ("R1", "none", {"family": True}, "non-R4"),
        ("R1", "none", {"batch_size": 32}, "64"),
        ("R1", "none", None, "DataLoader"),
    ],
)
def test_loader_validation_rejects_wrong_policy_sampler_and_batch(
    candidate: str,
    augmentation: str,
    loader_values: dict[str, Any] | None,
    message: str,
) -> None:
    session = _session(
        candidate=candidate,
        objective="vicreg_triplet" if candidate == "R4" else "vicreg",
        augmentation_policy=training.AugmentationPolicy(augmentation),
    )
    loader = _LoaderShape(**loader_values) if loader_values is not None else object()
    with pytest.raises(ValueError, match=message):
        training.validate_training_loader(session, loader)


def test_r4_accepts_a_family_batch_sampler_bound_to_the_loader_dataset_rows() -> None:
    session = _session(candidate="R4", objective="vicreg_triplet")
    loader = _LoaderShape(family=True)

    training.validate_training_loader(session, loader)

    assert isinstance(loader.batch_sampler, FamilyBatchSampler)
    assert loader.batch_sampler.pairs["id"].tolist() == (
        loader.dataset.pairs["id"].tolist()
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("id", 1000, "product IDs"),
        ("product_family_group", "changed-family", "metadata"),
        ("sha256", "changed-sha", "metadata"),
        ("duplicate_group", "changed-duplicate", "metadata"),
    ],
)
def test_r4_rejects_a_family_sampler_built_from_different_rows(
    column: str,
    value: object,
    message: str,
) -> None:
    session = _session(candidate="R4", objective="vicreg_triplet")
    family_pairs = _family_pair_frame()
    if column == "id":
        family_pairs["id"] = family_pairs["id"] + int(value)
    else:
        family_pairs.loc[40, column] = value

    with pytest.raises(ValueError, match=message):
        training.validate_training_loader(
            session,
            _LoaderShape(family=True, family_pairs=family_pairs),
        )


def test_r4_rejects_a_family_sampler_supplied_through_the_ordinary_sampler() -> None:
    session = _session(candidate="R4", objective="vicreg_triplet")
    loader = _LoaderShape(family_via_sampler=True)

    assert isinstance(loader.sampler, FamilyBatchSampler)
    with pytest.raises(ValueError, match="batch_sampler"):
        training.validate_training_loader(session, loader)


def test_r4_rejects_a_dataset_without_a_canonical_pair_frame() -> None:
    session = _session(candidate="R4", objective="vicreg_triplet")
    loader = _LoaderShape(family=True)
    loader.dataset.pairs = None

    with pytest.raises(ValueError, match="canonical pair frame"):
        training.validate_training_loader(session, loader)


def test_lifecycle_propagates_each_epoch_to_real_dataloader_dataset(
    tmp_path: Path,
) -> None:
    config = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=3,
        checkpoint_epochs=(3,),
    )
    session = _session(run_id="epoch-lifecycle", hyperparameters=config)
    loader = _LoaderShape()

    def train_from_lifecycle(
        device: torch.device,
        model: nn.Module,
        bound_session: Any,
        batches: Any,
    ) -> Any:
        optimizer = training.build_optimizer(model, config)
        return training.train_epochs(
            model=model,
            batches=batches,
            optimizer=optimizer,
            scheduler=training.WarmupCosineScheduler(
                optimizer,
                steps_per_epoch=1,
                config=config,
            ),
            scaler=training.make_grad_scaler(device),
            device=device,
            session=bound_session,
            score_callback=lambda candidate_model, epoch: 0.1,
            checkpoint_callback=lambda epoch, score: training.CheckpointRecord(
                epoch=epoch,
                path=tmp_path / "model.pt",
                sha256="a" * 64,
                config_hash=session.config_hash,
                score=score,
                split_fingerprint=session.split_fingerprint,
                weight_origin=session.model_metadata.weight_origin,
                parent_run_id=session.parent_run_id,
                run_id=session.run_id,
                run_kind=session.run_kind,
            ),
        )

    training.run_training_attempt(
        registry=RunRegistry(tmp_path / "runs.csv"),
        row=_bound_row(session),
        session=session,
        batches=loader,
        device_factory=lambda: torch.device("cpu"),
        train=train_from_lifecycle,
    )

    assert loader.dataset.epochs == [0, 1, 2]


def test_training_propagates_each_epoch_once_to_all_distinct_loader_targets(
    tmp_path: Path,
) -> None:
    config = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=2,
        checkpoint_epochs=(2,),
    )
    session = _session(run_id="epoch-targets", hyperparameters=config)
    model = _TinyEncoder()
    optimizer = training.build_optimizer(model, config)
    loader = _DistinctEpochLoader()

    training.train_epochs(
        model=model,
        batches=loader,
        optimizer=optimizer,
        scheduler=training.WarmupCosineScheduler(
            optimizer,
            steps_per_epoch=1,
            config=config,
        ),
        scaler=training.make_grad_scaler("cpu"),
        device="cpu",
        session=session,
        score_callback=lambda candidate_model, epoch: 0.1,
        checkpoint_callback=lambda epoch, score: training.CheckpointRecord(
            epoch=epoch,
            path=tmp_path / "model.pt",
            sha256="a" * 64,
            config_hash=session.config_hash,
            score=score,
            split_fingerprint=session.split_fingerprint,
            weight_origin=session.model_metadata.weight_origin,
            parent_run_id=session.parent_run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
        ),
    )

    assert loader.dataset.epochs == [0, 1]
    assert loader.sampler.epochs == [0, 1]
    assert loader.batch_sampler.epochs == [0, 1]


def test_training_rejects_checkpoint_from_a_different_session(tmp_path: Path) -> None:
    config = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=2,
        checkpoint_epochs=(1, 2),
    )
    session = _session(hyperparameters=config)
    other = _session(
        candidate="R4",
        objective="vicreg_triplet",
        parent_run_id="other-parent",
        hyperparameters=config,
    )
    model = _TinyEncoder()
    optimizer = training.build_optimizer(model, config)

    with pytest.raises(ValueError, match="config hash"):
        training.train_epochs(
            model=model,
            batches=[_pair_batch()],
            optimizer=optimizer,
            scheduler=training.WarmupCosineScheduler(
                optimizer,
                steps_per_epoch=1,
                config=config,
            ),
            scaler=training.make_grad_scaler("cpu"),
            device="cpu",
            session=session,
            score_callback=lambda model, epoch: 0.1,
            checkpoint_callback=lambda epoch, score: training.CheckpointRecord(
                epoch=epoch,
                path=tmp_path / "wrong.pt",
                sha256="a" * 64,
                config_hash=other.config_hash,
                score=score,
                split_fingerprint=other.split_fingerprint,
                weight_origin=other.model_metadata.weight_origin,
                parent_run_id=other.parent_run_id,
                run_id=other.run_id,
                run_kind=other.run_kind,
            ),
        )


def test_run_lifecycle_appends_before_device_work_and_leaves_success_running(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    session = _session(run_id="success")
    loader = _LoaderShape()

    def device_factory() -> torch.device:
        assert registry.read()[0]["status"] == "running"
        return torch.device("cpu")

    def inspect_factory_model(
        device: torch.device,
        bound_model: nn.Module,
        bound_session: Any,
        bound_batches: Any,
    ) -> tuple[Any, ...]:
        assert all(tensor.device.type == "cpu" for tensor in bound_model.parameters())
        return device.type, bound_model.metadata, bound_session, bound_batches

    result = training.run_training_attempt(
        registry=registry,
        row=_bound_row(session),
        session=session,
        batches=loader,
        device_factory=device_factory,
        train=inspect_factory_model,
    )

    assert result == ("cpu", session.model_metadata, session, loader)
    assert registry.read()[0]["status"] == "running"


def test_run_lifecycle_marks_every_ordinary_exception_failed(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    session = _session(run_id="failed")

    with pytest.raises(RuntimeError, match="device failed"):
        training.run_training_attempt(
            registry=registry,
            row=_bound_row(session),
            session=session,
            batches=_LoaderShape(),
            device_factory=lambda: (_ for _ in ()).throw(RuntimeError("device failed")),
            train=lambda device, model, bound_session, batches: (
                device,
                model,
                bound_session,
                batches,
            ),
        )

    row = registry.read()[0]
    assert row["status"] == "failed"
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "device failed"
    assert row["completed_at_utc"].endswith("Z")


class _RecordingRegistry(RunRegistry):
    def __init__(self, path: Path, events: list[str]) -> None:
        super().__init__(path)
        self.events = events

    def append(self, row: Any) -> dict[str, str]:
        self.events.append("append")
        return super().append(row)


def test_run_attempt_order_is_append_determinism_factory_then_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    session = _session(run_id="ordered")
    registry = _RecordingRegistry(tmp_path / "runs.csv", events)
    real_factory = training.build_retrieval_encoder
    monkeypatch.setattr(
        training,
        "configure_determinism",
        lambda seed: events.append(f"determinism:{seed}"),
    )
    monkeypatch.setattr(
        training,
        "build_retrieval_encoder",
        lambda architecture: events.append("factory") or real_factory(architecture),
    )

    training.run_training_attempt(
        registry=registry,
        row=_bound_row(session),
        session=session,
        batches=_LoaderShape(),
        device_factory=lambda: events.append("device") or torch.device("cpu"),
        train=lambda device, bound_model, bound_session, batches: None,
    )

    assert events == ["append", "determinism:2753", "factory", "device"]


def test_lifecycle_builds_identical_scratch_initial_state_for_repeated_attempts(
    tmp_path: Path,
) -> None:
    states: list[dict[str, torch.Tensor]] = []
    sessions = [_session(run_id="repeat-a"), _session(run_id="repeat-b")]
    assert sessions[0].config_hash == sessions[1].config_hash

    for session in sessions:
        training.run_training_attempt(
            registry=RunRegistry(tmp_path / f"{session.run_id}.csv"),
            row=_bound_row(session),
            session=session,
            batches=_LoaderShape(),
            device_factory=lambda: torch.device("cpu"),
            train=lambda device, model, bound_session, batches: states.append(
                {key: value.detach().clone() for key, value in model.state_dict().items()}
            ),
        )

    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


def test_model_factory_failure_marks_appended_row_failed_before_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(run_id="factory-failed")
    device_called = False
    monkeypatch.setattr(
        training,
        "build_retrieval_encoder",
        lambda architecture: (_ for _ in ()).throw(RuntimeError("factory failed")),
    )

    def device_factory() -> torch.device:
        nonlocal device_called
        device_called = True
        return torch.device("cpu")

    registry = RunRegistry(tmp_path / "runs.csv")
    with pytest.raises(RuntimeError, match="factory failed"):
        training.run_training_attempt(
            registry=registry,
            row=_bound_row(session),
            session=session,
            batches=_LoaderShape(),
            device_factory=device_factory,
            train=lambda device, model, bound_session, batches: None,
        )

    assert not device_called
    assert registry.read()[0]["status"] == "failed"
    assert registry.read()[0]["error_message"] == "factory failed"


@pytest.mark.parametrize("failure", ["metadata", "device"])
def test_lifecycle_rejects_invalid_factory_model_before_device_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    session = _session(run_id=f"invalid-model-{failure}")
    if failure == "metadata":
        model = build_retrieval_encoder("resnet34")
    else:
        model = build_retrieval_encoder("resnet18")
        model.to("meta")
    monkeypatch.setattr(training, "build_retrieval_encoder", lambda architecture: model)
    device_called = False

    def device_factory() -> torch.device:
        nonlocal device_called
        device_called = True
        return torch.device("cpu")

    registry = RunRegistry(tmp_path / f"{failure}.csv")
    with pytest.raises(ValueError, match="metadata|CPU"):
        training.run_training_attempt(
            registry=registry,
            row=_bound_row(session),
            session=session,
            batches=_LoaderShape(),
            device_factory=device_factory,
            train=lambda device, built, bound_session, batches: None,
        )

    assert not device_called
    assert registry.read()[0]["status"] == "failed"


def test_lifecycle_rejects_wrong_loader_before_device_creation(tmp_path: Path) -> None:
    session = _session(run_id="wrong-loader")
    device_called = False

    def device_factory() -> torch.device:
        nonlocal device_called
        device_called = True
        return torch.device("cpu")

    registry = RunRegistry(tmp_path / "runs.csv")
    with pytest.raises(ValueError, match="geometry"):
        training.run_training_attempt(
            registry=registry,
            row=_bound_row(session),
            session=session,
            batches=_LoaderShape(geometry=True),
            device_factory=device_factory,
            train=lambda device, model, bound_session, batches: None,
        )

    assert not device_called
    assert registry.read()[0]["status"] == "failed"


def test_b1_scratch_eligible_row_fails_before_device_creation(tmp_path: Path) -> None:
    session = _session(run_id="b1", candidate="B1")
    row = _bound_row(
        session,
        pretrained=False,
        weight_origin=SCRATCH_WEIGHT_ORIGIN,
        deployment_eligibility="eligible",
    )
    device_called = False

    def device_factory() -> torch.device:
        nonlocal device_called
        device_called = True
        return torch.device("cpu")

    with pytest.raises(ValueError, match="pretrained"):
        training.run_training_attempt(
            registry=RunRegistry(tmp_path / "runs.csv"),
            row=row,
            session=session,
            batches=_LoaderShape(),
            device_factory=device_factory,
            train=lambda device, model, bound_session, batches: None,
        )

    assert not device_called
    assert RunRegistry(tmp_path / "runs.csv").read()[0]["status"] == "failed"


def test_run_attempt_rejects_already_initialized_cuda_before_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(run_id="cuda-initialized")
    device_called = False
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def device_factory() -> torch.device:
        nonlocal device_called
        device_called = True
        return torch.device("cpu")

    with pytest.raises(RuntimeError, match="CUDA.*initialized"):
        training.run_training_attempt(
            registry=RunRegistry(tmp_path / "runs.csv"),
            row=_bound_row(session),
            session=session,
            batches=_LoaderShape(),
            device_factory=device_factory,
            train=lambda device, model, bound_session, batches: None,
        )

    assert not device_called


def test_run_attempt_validates_appended_row_before_cuda_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(run_id="row-before-cuda")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(ValueError, match="registry method"):
        training.run_training_attempt(
            registry=RunRegistry(tmp_path / "runs.csv"),
            row=_bound_row(session, method="R2"),
            session=session,
            batches=_LoaderShape(),
            device_factory=lambda: (_ for _ in ()).throw(
                AssertionError("device must not run")
            ),
            train=lambda device, model, bound_session, batches: None,
        )


def test_registry_failure_does_not_replace_original_training_exception(tmp_path: Path) -> None:
    session = _session(run_id="primary-error")
    primary = RuntimeError("primary training error")
    registry = RunRegistry(tmp_path / "runs.csv")

    def fail_update(run_id: str, updates: Any) -> dict[str, str]:
        del run_id, updates
        raise OSError("registry update failed")

    registry.update = fail_update  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as captured:
        training.run_training_attempt(
            registry=registry,
            row=_bound_row(session),
            session=session,
            batches=_LoaderShape(),
            device_factory=lambda: torch.device("cpu"),
            train=lambda device, model, bound_session, batches: (_ for _ in ()).throw(
                primary
            ),
        )

    assert captured.value is primary
    assert isinstance(captured.value.__cause__, OSError)
    assert any("registry update failed" in note for note in captured.value.__notes__)


def test_stale_recovery_requires_explicit_ids_and_leaves_unselected_old_rows(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("selected", started_at_utc="2026-08-29T01:00:00Z"))
    registry.append(_running_row("unselected", started_at_utc="2026-08-29T01:00:00Z"))
    registry.append(_running_row("fresh", started_at_utc="2026-08-29T03:30:00Z"))
    registry.append(_running_row("terminal", started_at_utc="2026-08-29T01:00:00Z"))
    registry.update(
        "terminal",
        {
            "status": "failed",
            "completed_at_utc": "2026-08-29T02:00:00Z",
            "error_type": "RuntimeError",
            "error_message": "already failed",
        },
    )

    recovered = training.recover_stale_running(
        registry,
        run_ids={"selected", "fresh", "terminal", "unknown"},
        stale_before=datetime(2026, 8, 29, 3, 0, tzinfo=timezone.utc),
        now=datetime(2026, 8, 29, 4, 0, tzinfo=timezone.utc),
    )

    assert recovered == ("selected",)
    rows = {row["run_id"]: row for row in registry.read()}
    assert rows["selected"]["status"] == "failed"
    assert rows["selected"]["error_type"] == "StaleRunningAttempt"
    assert rows["unselected"]["status"] == "running"
    assert rows["fresh"]["status"] == "running"
    assert rows["terminal"]["error_message"] == "already failed"
