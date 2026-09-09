"""One fixed-budget Task 1 refit on labelled development rows only."""

from __future__ import annotations

import io
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import pandas as pd
import torch
import torch.nn.functional as functional
from filelock import FileLock
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from fashion.config import (
    LABEL_MAPS_JSON,
    RANDOM_SEED,
    ROOT,
    RUNS_CSV,
    SPLITS_CSV,
    TASK1_FINAL_EVALUATION_CONFIG_JSON,
    TASK1_FINAL_REFIT_EVIDENCE_DIR,
    TASK1_MODEL_MANIFEST_JSON,
    TASK1_MODEL_PATH,
)
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.task1.dataset import Task1TorchDataset
from fashion.task1.evaluation import validate_task1_label_map
from fashion.task1.models import Task1SmallCNN, count_trainable_parameters
from fashion.task1.preprocessing import (
    TASK1_CONTROL_PREPROCESSING,
    Task1Normalization,
    build_task1_validation_transform,
    fit_task1_development_normalization,
)
from fashion.task1.registry import Task1RunRegistry
from fashion.train.artifacts import atomic_write_bytes, atomic_write_csv, atomic_write_json
from fashion.train.registry import RunRecord, new_run_id, tracked_run
from fashion.train.reproducibility import make_torch_generator, seed_everything, seed_worker

ExecutionMode = Literal["run", "load", "run_or_load"]
FINAL_HISTORY_PATH = TASK1_FINAL_REFIT_EVIDENCE_DIR / "training_history.csv"
EXPERIMENT_ID = "task1-cnn-task1_cnn_no_aug_unweighted_v1-final-refit"


@dataclass(frozen=True)
class Task1FixedEpochConfig:
    epochs: int = 20
    batch_size: int = 128
    seed: int = RANDOM_SEED
    max_lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    num_workers: int = 0

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.max_lr <= 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("optimization values are invalid")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class Task1FixedEpochResult:
    model: nn.Module
    history: pd.DataFrame
    normalization: Task1Normalization
    final_epoch: int
    validation_used: bool = False
    checkpoint_rule: str = "fixed_last_epoch"


@dataclass(frozen=True)
class Task1RefitOutcome:
    source: str
    run_id: str
    manifest_path: str
    manifest_sha256: str
    bundle_path: str
    bundle_sha256: str
    final_epoch: int
    development_rows: int


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_task1_fixed_epochs(
    development_rows: pd.DataFrame,
    label_to_index: Mapping[str, int],
    *,
    config: Task1FixedEpochConfig = Task1FixedEpochConfig(),
    root: str | Path = ROOT,
    device: torch.device | None = None,
    model_factory: Callable[[int], nn.Module] = Task1SmallCNN,
) -> Task1FixedEpochResult:
    """Train from random weights for every requested epoch, with no validation pass."""
    config.validate()
    if development_rows.empty or not development_rows["partition"].eq("development").all():
        raise ValueError("final refit may use non-empty development rows only")
    if len(label_to_index) != 124:
        raise ValueError("final refit requires the fixed 124-class label map")
    normalization = fit_task1_development_normalization(
        development_rows,
        root=root,
        config=TASK1_CONTROL_PREPROCESSING,
    )
    transform = build_task1_validation_transform(
        normalization,
        config=TASK1_CONTROL_PREPROCESSING,
    )
    dataset = Task1TorchDataset(
        development_rows,
        transform,
        label_to_index,
        root=root,
    )
    selected_device = device or _select_device()
    model = model_factory(124).to(selected_device)
    optimizer = Adam(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(dataset) / config.batch_size)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            generator=make_torch_generator(config.seed + epoch),
            worker_init_fn=seed_worker if config.num_workers else None,
        )
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in loader:
            images = batch["image"].to(selected_device)
            targets = batch["label"].to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(images), targets)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("training loss must be finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            count = int(targets.numel())
            loss_sum += float(loss.item()) * count
            sample_count += count
        if sample_count != len(dataset):
            raise RuntimeError("training loader did not consume every development row")
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "train_samples": sample_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    model.eval()
    return Task1FixedEpochResult(
        model=model,
        history=pd.DataFrame(history),
        normalization=normalization,
        final_epoch=config.epochs,
    )


_CONFIG_FIELDS = {
    "schema_version",
    "target",
    "candidate_id",
    "preprocessing_id",
    "loss_id",
    "model_family",
    "scratch",
    "seed",
    "epochs",
    "batch_size",
    "max_lr",
    "weight_decay",
    "grad_clip_norm",
    "checkpoint_rule",
    "validation_used",
    "augmentation_used",
    "normalization_scope",
}


def _load_frozen_config(path: Path) -> tuple[dict[str, Any], Task1FixedEpochConfig]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Task 1 final evaluation config must be an object")
    refit_payload = payload.get("refit", payload)
    if not isinstance(refit_payload, dict) or set(refit_payload) != _CONFIG_FIELDS:
        raise ValueError("Task 1 final evaluation config fields changed")
    expected = {
        "schema_version": 1,
        "target": "articleType",
        "candidate_id": "task1_cnn_no_aug_unweighted_v1",
        "preprocessing_id": "task1_rgb_60x80_no_aug_v1",
        "loss_id": "cross_entropy_unweighted_v1",
        "model_family": "task1_small_cnn_v1",
        "scratch": True,
        "seed": 2753,
        "epochs": 20,
        "batch_size": 128,
        "max_lr": 0.001,
        "weight_decay": 0.00001,
        "grad_clip_norm": 1.0,
        "checkpoint_rule": "fixed_last_epoch",
        "validation_used": False,
        "augmentation_used": False,
        "normalization_scope": "development_only",
    }
    if refit_payload != expected:
        raise ValueError("Task 1 final refit contract differs from the frozen recipe")
    config = Task1FixedEpochConfig(
        epochs=int(refit_payload["epochs"]),
        batch_size=int(refit_payload["batch_size"]),
        seed=int(refit_payload["seed"]),
        max_lr=float(refit_payload["max_lr"]),
        weight_decay=float(refit_payload["weight_decay"]),
        grad_clip_norm=float(refit_payload["grad_clip_norm"]),
    )
    return refit_payload, config


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside project root: {path}") from error


def _declaration(path: Path, root: Path, *, digest_path: Path | None = None) -> dict[str, str]:
    return {"path": _relative(path, root), "sha256": compute_sha256(digest_path or path)}


def _serialize_bundle(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def _load_article_type_map(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        maps = json.load(handle)
    if "articleType" not in maps:
        raise ValueError("label map file has no articleType map")
    return dict(maps["articleType"])


def _development_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {"id", "path", "partition", "articleType", "has_articleType_label"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"split file is missing columns: {sorted(missing)}")
    valid = rows["has_articleType_label"].astype(str).str.lower().isin({"true", "1"})
    result = rows.loc[rows["partition"].eq("development") & valid].copy()
    if result.empty or result["id"].duplicated().any():
        raise ValueError("final refit requires unique labelled development rows")
    return result


def _preflight_outputs(paths: tuple[Path, ...], mode: ExecutionMode) -> Literal["run", "load"]:
    present = [path.is_file() for path in paths]
    if mode == "run":
        if any(present):
            raise FileExistsError("final refit outputs already exist; use load to verify them")
        return "run"
    if mode == "load":
        if not all(present):
            raise FileNotFoundError("final refit outputs are incomplete")
        return "load"
    if all(present):
        return "load"
    if any(present):
        raise RuntimeError("partial final refit outputs require manual inspection")
    return "run"


def _guard_refit(function):
    @wraps(function)
    def guarded(**kwargs):
        root = Path(kwargs.get("project_root", ROOT)).resolve()
        for key, relative in {
            "splits_path": "data/processed/splits.csv",
            "label_map_path": "data/processed/label_maps.json",
            "config_path": "configs/task1/final_evaluation.json",
            "model_path": "models/task1_article_type.pt",
            "manifest_path": "models/task1_article_type.manifest.json",
            "history_path": "results/evidence/task1/development_refit/training_history.csv",
            "registry_path": "results/runs.csv",
        }.items():
            kwargs.setdefault(key, root / relative)
        folder = (
            Path(kwargs["history_path"]).parent
            if "history_path" in kwargs
            else root / "results/evidence/task1/development_refit"
        )
        folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder / ".refit.lock"), timeout=0):
            manifest = Path(
                kwargs.get("manifest_path", root / "models/task1_article_type.manifest.json")
            )
            marker = folder / "refit_attempt.json"
            if kwargs.get("mode", "run_or_load") != "load" and not manifest.exists():
                if marker.exists():
                    raise RuntimeError(
                        "failed or partial final refit attempt requires manual inspection"
                    )
                atomic_write_json(marker, {"status": "started; inspect before retry"})
            return function(**kwargs)

    return guarded


@_guard_refit
def run_task1_refit(
    *,
    mode: ExecutionMode = "run_or_load",
    project_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    config_path: str | Path = TASK1_FINAL_EVALUATION_CONFIG_JSON,
    model_path: str | Path = TASK1_MODEL_PATH,
    manifest_path: str | Path = TASK1_MODEL_MANIFEST_JSON,
    history_path: str | Path = FINAL_HISTORY_PATH,
    registry_path: str | Path = RUNS_CSV,
    trainer: Callable[..., Task1FixedEpochResult] = train_task1_fixed_epochs,
    allow_test_trainer: bool = False,
) -> Task1RefitOutcome:
    """Train once or verify the already published, immutable final refit."""
    if mode not in {"run", "load", "run_or_load"}:
        raise ValueError("mode must be run, load, or run_or_load")
    root = Path(project_root).resolve()
    split = Path(splits_path).resolve()
    labels_file = Path(label_map_path).resolve()
    frozen_config = Path(config_path).resolve()
    bundle = Path(model_path).resolve()
    manifest = Path(manifest_path).resolve()
    history = Path(history_path).resolve()
    action = _preflight_outputs((bundle, manifest, history), mode)
    if action == "load":
        payload = load_verified_task1_refit_manifest(
            manifest,
            project_root=root,
            registry_path=registry_path,
        )
        return Task1RefitOutcome(
            source="loaded",
            run_id=str(payload["run_id"]),
            manifest_path=str(manifest),
            manifest_sha256=compute_sha256(manifest),
            bundle_path=str(bundle),
            bundle_sha256=str(payload["bundle"]["sha256"]),
            final_epoch=int(payload["final_epoch"]),
            development_rows=int(payload["development_rows"]),
        )

    frozen, train_config = _load_frozen_config(frozen_config)
    if trainer is not train_task1_fixed_epochs and not allow_test_trainer:
        raise ValueError("custom final refit trainers are test-only")
    label_map = _load_article_type_map(labels_file)
    label_to_index, class_names = validate_task1_label_map(label_map)
    if allow_test_trainer:
        rows = _development_rows(split)
    else:
        if split != root / "data/processed/splits.csv":
            raise ValueError("final refit requires the canonical splits.csv")
        canonical = load_splits(split)
        rows = canonical.loc[canonical.partition.eq("development")].copy()
        if len(rows) != 32773 or not rows.has_articleType_label.all():
            raise ValueError("final refit requires all 32773 labelled development rows")
        for row in rows.itertuples():
            image_path = (root / row.path).resolve()
            if not image_path.is_relative_to(root) or compute_sha256(image_path) != row.sha256:
                raise ValueError("development image differs from its canonical digest")
    config_digest = compute_sha256(frozen_config)
    record = RunRecord(
        run_id=new_run_id(EXPERIMENT_ID, None, train_config.seed),
        task="task1",
        stage="final_refit",
        experiment_id=EXPERIMENT_ID,
        model_family="task1_small_cnn_v1",
        benchmark_only=False,
        final_eligible=not allow_test_trainer,
        scratch=True,
        fold=None,
        seed=train_config.seed,
        transform_id="task1_rgb_60x80_no_aug_v1",
        loss_id="cross_entropy_unweighted_v1",
        epochs_requested=train_config.epochs,
        primary_metric_name="development_training_loss",
        config_sha256=config_digest,
        split_sha256=compute_sha256(split),
        label_map_sha256=compute_sha256(labels_file),
        implementation_sha256=compute_sha256(Path(__file__)),
    )
    registry = Task1RunRegistry(registry_path)
    with tracked_run(registry, record) as active:
        seed_everything(train_config.seed)
        trained = trainer(
            rows,
            label_to_index,
            config=train_config,
            root=root,
            device=_select_device(),
        )
        if trained.final_epoch != train_config.epochs and not allow_test_trainer:
            raise RuntimeError("trainer did not complete the frozen final epoch")
        model_config = (
            asdict(trained.model.config)
            if hasattr(trained.model, "config")
            else {"num_classes": 124}
        )
        bundle_payload = {
            "format_version": 1,
            "task": "task1",
            "target": "articleType",
            "candidate_id": frozen["candidate_id"],
            "model_family": frozen["model_family"],
            "model_config": model_config,
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in trained.model.state_dict().items()
            },
            "class_names": class_names,
            "label_to_index": label_to_index,
            "preprocessing": TASK1_CONTROL_PREPROCESSING.to_dict(),
            "normalization": trained.normalization.to_dict(),
            "training": {
                "seed": train_config.seed,
                "epochs": train_config.epochs,
                "final_epoch": trained.final_epoch,
                "validation_used": False,
                "checkpoint_rule": "fixed_last_epoch",
            },
        }
        with TemporaryDirectory(prefix="task1-refit-", dir=root) as temporary_dir:
            staging = Path(temporary_dir)
            staged_bundle = staging / bundle.name
            staged_history = staging / history.name
            staged_manifest = staging / manifest.name
            atomic_write_bytes(staged_bundle, _serialize_bundle(bundle_payload))
            atomic_write_csv(staged_history, trained.history)
            manifest_payload = {
                "schema_version": 1,
                "status": "test_only" if allow_test_trainer else "completed",
                "run_id": active.run_id,
                "task": "task1",
                "target": "articleType",
                "candidate_id": frozen["candidate_id"],
                "model_family": frozen["model_family"],
                "scratch": True,
                "final_epoch": trained.final_epoch,
                "development_rows": len(rows),
                "canonical_inputs": {
                    "splits": _declaration(split, root),
                    "label_map": _declaration(labels_file, root),
                    "config": _declaration(frozen_config, root),
                },
                "bundle": _declaration(bundle, root, digest_path=staged_bundle),
                "history": _declaration(history, root, digest_path=staged_history),
                "training": {
                    "validation_used": False,
                    "checkpoint_rule": "fixed_last_epoch",
                    "normalization_scope": "development_only",
                },
            }
            atomic_write_json(staged_manifest, manifest_payload)
            for source, destination in (
                (staged_bundle, bundle),
                (staged_history, history),
                (staged_manifest, manifest),
            ):
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)

        active.epochs_completed = trained.final_epoch
        active.best_epoch = trained.final_epoch
        active.primary_metric_value = float(trained.history.iloc[-1]["train_loss"])
        active.metrics = {
            "final_train_loss": active.primary_metric_value,
            "validation_used": False,
            "checkpoint_rule": "fixed_last_epoch",
        }
        active.parameter_count = count_trainable_parameters(trained.model)
        active.checkpoint_path = _relative(bundle, root)
        active.checkpoint_sha256 = compute_sha256(bundle)
        active.history_path = _relative(history, root)
        active.history_sha256 = compute_sha256(history)

    return Task1RefitOutcome(
        source="trained",
        run_id=record.run_id,
        manifest_path=str(manifest),
        manifest_sha256=compute_sha256(manifest),
        bundle_path=str(bundle),
        bundle_sha256=compute_sha256(bundle),
        final_epoch=train_config.epochs,
        development_rows=len(rows),
    )


def _verify_declaration(value: Any, root: Path, scope: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise RuntimeError(f"{scope} declaration is invalid")
    relative = Path(str(value["path"]))
    if relative.is_absolute():
        raise RuntimeError(f"{scope} path must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{scope} path escapes the project") from error
    if not resolved.is_file():
        raise RuntimeError(f"{scope} does not exist: {resolved}")
    actual = compute_sha256(resolved)
    if actual != str(value["sha256"]):
        raise RuntimeError(
            f"SHA-256 mismatch for {scope}: expected {value['sha256']}, got {actual}"
        )
    return resolved


def load_verified_task1_refit_manifest(
    manifest_path: str | Path = TASK1_MODEL_MANIFEST_JSON,
    *,
    project_root: str | Path = ROOT,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify all final-refit files and the matching completed registry row."""
    root = Path(project_root).resolve()
    manifest = Path(manifest_path).resolve()
    with manifest.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {
        "schema_version",
        "status",
        "run_id",
        "task",
        "target",
        "candidate_id",
        "model_family",
        "scratch",
        "final_epoch",
        "development_rows",
        "canonical_inputs",
        "bundle",
        "history",
        "training",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("Task 1 final refit manifest fields changed")
    if (
        payload["schema_version"] != 1
        or payload["status"] != "completed"
        or payload["task"] != "task1"
        or payload["target"] != "articleType"
        or payload["candidate_id"] != "task1_cnn_no_aug_unweighted_v1"
        or payload["model_family"] != "task1_small_cnn_v1"
        or payload["scratch"] is not True
        or payload["final_epoch"] != 20
        or payload["development_rows"] != 32773
        or payload["training"]
        != {
            "validation_used": False,
            "checkpoint_rule": "fixed_last_epoch",
            "normalization_scope": "development_only",
        }
    ):
        raise RuntimeError("Task 1 final refit is not deployment eligible")
    for name, declaration in payload["canonical_inputs"].items():
        _verify_declaration(declaration, root, f"canonical {name}")
    bundle_path = _verify_declaration(payload["bundle"], root, "model bundle")
    history_path = _verify_declaration(payload["history"], root, "training history")
    resolved_registry = root / "results/runs.csv" if registry_path is None else Path(registry_path)
    rows = Task1RunRegistry(resolved_registry).read()
    match = rows.loc[rows["run_id"].eq(str(payload["run_id"]))]
    if len(match) != 1:
        raise RuntimeError("final refit registry row is missing or duplicated")
    row = match.iloc[0]
    if (
        row["status"] != "completed"
        or row["stage"] != "final_refit"
        or row["scratch"] != "true"
        or row["final_eligible"] != "true"
        or row["best_epoch"] != "20"
        or row["checkpoint_sha256"] != payload["bundle"]["sha256"]
        or row["history_sha256"] != payload["history"]["sha256"]
    ):
        raise RuntimeError("final refit registry row does not match the manifest")
    payload["_resolved_bundle_path"] = str(bundle_path)
    payload["_resolved_history_path"] = str(history_path)
    return payload
