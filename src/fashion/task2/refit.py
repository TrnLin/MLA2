"""Development-only fixed-epoch refit and hash-verified Task 2 model bundle."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd
import torch

from fashion.config import (
    LABEL_MAPS_JSON,
    ROOT,
    RUNS_CSV,
    SPLITS_CSV,
    TASK2_EVIDENCE_DIR,
    TASK2_MODEL_MANIFEST_JSON,
    TASK2_MODEL_PATH,
    TASK2_SELECTION_FREEZE_JSON,
)
from fashion.data.dataset import load_label_maps
from fashion.data.hashing import compute_sha256
from fashion.data.multitask import build_development_multitask_loader
from fashion.data.torch import ImageTransformSpec
from fashion.models.season import (
    SeasonModelSpec,
    assert_final_model,
    build_multitask_season_model,
)
from fashion.task2.multitask import I2ExperimentConfig, load_i2_config
from fashion.task2.ultimate_judgement import load_verified_selection_freeze
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import implementation_sha256, verify_implementation_at_head
from fashion.train.multitask import (
    RefitResult,
    RefitTrainConfig,
    train_masked_multitask_refit,
)
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import capture_git_state, capture_runtime, seed_everything

ExecutionMode = Literal["run", "load", "run_or_load"]

REFIT_GATE = "G8-DEVELOPMENT-REFIT"
REFIT_STAGE = "development_refit"
REFIT_ANALYSIS_ROLE = "post_selection_development_refit_without_holdout_evaluation"
REFIT_EVIDENCE_DIRECTORY = TASK2_EVIDENCE_DIR / "development_refit"
REFIT_HISTORY_CSV = REFIT_EVIDENCE_DIRECTORY / "training_history.csv"
REFIT_RUNTIME_JSON = REFIT_EVIDENCE_DIRECTORY / "runtime.json"
REFIT_IMPLEMENTATION_PATHS = (
    "src/fashion/config.py",
    "src/fashion/data/dataset.py",
    "src/fashion/data/hashing.py",
    "src/fashion/data/images.py",
    "src/fashion/data/multitask.py",
    "src/fashion/data/torch.py",
    "src/fashion/models/season.py",
    "src/fashion/task2/multitask.py",
    "src/fashion/task2/refit.py",
    "src/fashion/task2/ultimate_judgement.py",
    "src/fashion/train/artifacts.py",
    "src/fashion/train/cache.py",
    "src/fashion/train/engine.py",
    "src/fashion/train/multitask.py",
    "src/fashion/train/registry.py",
    "src/fashion/train/reproducibility.py",
)
HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "train_season_loss",
    "train_auxiliary_loss",
    "train_accuracy",
    "train_samples",
    "train_auxiliary_labeled_samples",
    "learning_rate",
)
MANIFEST_FIELDS = {
    "schema_version",
    "gate",
    "status",
    "analysis_role",
    "refit_id",
    "run_id",
    "selected_candidate",
    "selected_experiment_id",
    "model_family",
    "scratch",
    "weights",
    "benchmark_only",
    "final_eligible",
    "holdout_opened",
    "holdout_metrics_present",
    "validation_used",
    "early_stopping_used",
    "primary_metric_name",
    "evaluation_claim_allowed",
    "seed",
    "final_epoch",
    "valid_development_rows",
    "parameter_count",
    "temperature",
    "loader_audit",
    "training_metadata",
    "git_commit",
    "git_dirty",
    "implementation_files_at_head",
    "implementation_sha256",
    "canonical_inputs",
    "selection_freeze",
    "selected_config",
    "bundle",
    "artifacts",
    "model_change_after_holdout_allowed",
}
BUNDLE_FIELDS = {
    "format_version",
    "task",
    "target",
    "candidate",
    "experiment_id",
    "model_spec",
    "model_boundary",
    "model_state_dict",
    "labels",
    "label_to_index",
    "auxiliary",
    "preprocessing",
    "calibration",
    "training",
    "provenance",
    "inference",
}
LOADER_AUDIT_FIELDS = {
    "partition",
    "training_products",
    "validation_products",
    "protected_products",
    "labels",
    "auxiliary_target",
    "auxiliary_labels",
    "auxiliary_training_products",
    "auxiliary_training_id_sha256",
    "training_id_sha256",
    "train_transform_id",
    "normalisation_scope",
    "stats",
}
TRAINING_METADATA_FIELDS = {
    "amp_enabled",
    "accumulation_steps",
    "updates_completed",
    "selection_metric",
    "validation_used",
    "early_stopping_used",
    "checkpoint_rule",
    "auxiliary_weight",
}
RUNTIME_FIELDS = {
    "schema_version",
    "gate",
    "run_id",
    "runtime_seconds",
    "peak_vram_mb",
    "device",
    "parameter_count",
    "environment",
    "training_metadata",
    "holdout_opened",
    "validation_used",
    "primary_metric_name",
}


@dataclass(frozen=True)
class RefitOutcome:
    """One verified final-refit result, whether trained now or loaded."""

    source: str
    manifest_path: str
    manifest_sha256: str
    bundle_path: str
    bundle_sha256: str
    run_id: str
    final_epoch: int
    valid_development_rows: int


@dataclass(frozen=True)
class _RefitContract:
    freeze: dict[str, Any]
    freeze_path: Path
    config: I2ExperimentConfig
    config_path: Path
    train_config: RefitTrainConfig
    implementation_files: tuple[str, ...]
    implementation_sha256: str
    git_commit: str

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(
            {
                "selected_config": self.config.to_dict(),
                "refit_rule": self.freeze["refit_rule"],
                "selection_freeze_sha256": compute_sha256(self.freeze_path),
            }
        )


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _load_json_object(path: Path, scope: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{scope} must be a JSON object")
    return payload


def _relative_path(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside project root: {resolved}") from error


def _declaration(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "path": _relative_path(path, root=root),
        "sha256": compute_sha256(path),
    }


def _resolve_declaration(
    declaration: Any,
    *,
    root: Path,
    scope: str,
) -> Path:
    if not isinstance(declaration, Mapping) or set(declaration) != {"path", "sha256"}:
        raise ValueError(f"{scope} must declare only path and sha256")
    raw_path = Path(str(declaration["path"]))
    if raw_path.is_absolute():
        raise ValueError(f"{scope} path must be project-relative")
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{scope} path escapes project root") from error
    verify_artifact(resolved, str(declaration["sha256"]))
    return resolved


def _build_train_config(
    freeze: Mapping[str, Any],
    selected_config: I2ExperimentConfig,
) -> RefitTrainConfig:
    rule = dict(freeze["refit_rule"])
    optimisation = selected_config.optimisation
    return RefitTrainConfig(
        seed=int(rule["seed"]),
        epochs=int(rule["epochs"]),
        batch_size=selected_config.data.batch_size,
        effective_batch_size=optimisation.effective_batch_size,
        learning_rate=optimisation.learning_rate,
        weight_decay=optimisation.weight_decay,
        gradient_clip_norm=optimisation.gradient_clip_norm,
        warmup_epochs=optimisation.warmup_epochs,
        use_amp=optimisation.use_amp,
        device=optimisation.device,
    )


def _load_refit_contract(
    *,
    project_root: Path,
    freeze_path: Path,
    splits_path: Path,
    label_map_path: Path,
) -> _RefitContract:
    freeze, resolved_freeze = load_verified_selection_freeze(
        freeze_path,
        project_root=project_root,
    )
    selected = freeze["selected_model"]
    if selected["candidate"] != "I2":
        raise ValueError("this frozen refit implementation supports the selected I2 bundle only")
    expected_splits = _resolve_declaration(
        freeze["canonical_inputs"]["splits"],
        root=project_root,
        scope="selection freeze splits",
    )
    expected_labels = _resolve_declaration(
        freeze["canonical_inputs"]["label_maps"],
        root=project_root,
        scope="selection freeze label maps",
    )
    if splits_path.resolve() != expected_splits or label_map_path.resolve() != expected_labels:
        raise ValueError("refit inputs must be the exact canonical files frozen at G7")
    config_path = _resolve_declaration(
        freeze["primary_development_evidence"]["config"],
        root=project_root,
        scope="selection freeze selected config",
    )
    config = load_i2_config(config_path)
    if config.experiment_id != selected["experiment_id"]:
        raise ValueError("selected config no longer matches the frozen experiment")
    train_config = _build_train_config(freeze, config)
    train_config.validate()

    git_state = capture_git_state(project_root)
    commit = str(git_state.get("commit") or "")
    dirty = git_state.get("dirty")
    if len(commit) != 40 or dirty is not False:
        raise ValueError("development refit requires a clean Git commit for provenance")
    implementation_files = verify_implementation_at_head(
        *REFIT_IMPLEMENTATION_PATHS,
        root=project_root,
    )
    implementation_digest = implementation_sha256(
        *REFIT_IMPLEMENTATION_PATHS,
        root=project_root,
    )
    return _RefitContract(
        freeze=freeze,
        freeze_path=resolved_freeze,
        config=config,
        config_path=config_path,
        train_config=train_config,
        implementation_files=implementation_files,
        implementation_sha256=implementation_digest,
        git_commit=commit,
    )


def _save_bundle(path: Path, payload: Mapping[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    atomic_write_bytes(path, buffer.getvalue())
    return compute_sha256(path)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _history_frame(result: RefitResult) -> pd.DataFrame:
    history = pd.DataFrame(result.history)
    if tuple(history.columns) != HISTORY_COLUMNS:
        raise ValueError("refit history schema changed")
    if len(history) != result.final_epoch:
        raise ValueError("refit history does not contain every declared epoch")
    return history


def _outcome(manifest: Mapping[str, Any], manifest_path: Path, *, source: str) -> RefitOutcome:
    return RefitOutcome(
        source=source,
        manifest_path=manifest_path.as_posix(),
        manifest_sha256=compute_sha256(manifest_path),
        bundle_path=str(manifest["bundle"]["path"]),
        bundle_sha256=str(manifest["bundle"]["sha256"]),
        run_id=str(manifest["run_id"]),
        final_epoch=int(manifest["final_epoch"]),
        valid_development_rows=int(manifest["valid_development_rows"]),
    )


def load_verified_development_refit_manifest(
    path: str | Path = TASK2_MODEL_MANIFEST_JSON,
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Verify the final model, evidence, freeze, and scratch boundary before use."""
    root = Path(project_root).resolve()
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = _load_json_object(manifest_path, "development refit manifest")
    _require_exact_keys(manifest, MANIFEST_FIELDS, "development refit manifest")
    identity = {
        "schema_version": "1.0.0",
        "gate": REFIT_GATE,
        "status": "complete",
        "analysis_role": REFIT_ANALYSIS_ROLE,
        "selected_candidate": "I2",
        "scratch": True,
        "weights": None,
        "benchmark_only": False,
        "final_eligible": True,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "validation_used": False,
        "early_stopping_used": False,
        "primary_metric_name": None,
        "evaluation_claim_allowed": False,
        "git_dirty": False,
        "model_change_after_holdout_allowed": False,
    }
    changed = [name for name, expected in identity.items() if manifest[name] != expected]
    if changed:
        raise ValueError(f"development refit boundary changed: {changed}")
    if len(str(manifest["git_commit"])) != 40:
        raise ValueError("development refit Git commit is invalid")
    if manifest["implementation_files_at_head"] != list(REFIT_IMPLEMENTATION_PATHS):
        raise ValueError("development refit implementation file set changed")
    current_implementation = implementation_sha256(
        *REFIT_IMPLEMENTATION_PATHS,
        root=root,
    )
    if current_implementation != manifest["implementation_sha256"]:
        raise ValueError("development refit implementation bytes changed")

    canonical_inputs = manifest["canonical_inputs"]
    _require_exact_keys(canonical_inputs, {"splits", "label_maps"}, "canonical inputs")
    splits_path = _resolve_declaration(canonical_inputs["splits"], root=root, scope="refit splits")
    label_map_path = _resolve_declaration(
        canonical_inputs["label_maps"], root=root, scope="refit label maps"
    )
    freeze_path = _resolve_declaration(
        manifest["selection_freeze"], root=root, scope="selection freeze"
    )
    freeze, verified_freeze = load_verified_selection_freeze(
        freeze_path,
        project_root=root,
    )
    if verified_freeze != freeze_path:
        raise ValueError("selection freeze resolved to a different path")
    selected_config_path = _resolve_declaration(
        manifest["selected_config"], root=root, scope="selected config"
    )
    selected_config = load_i2_config(selected_config_path)
    selected = freeze["selected_model"]
    rule = freeze["refit_rule"]
    loader_audit = manifest["loader_audit"]
    training_metadata = manifest["training_metadata"]
    if not isinstance(loader_audit, Mapping) or not isinstance(training_metadata, Mapping):
        raise ValueError("refit audits must be objects")
    _require_exact_keys(loader_audit, LOADER_AUDIT_FIELDS, "refit loader audit")
    _require_exact_keys(
        training_metadata,
        TRAINING_METADATA_FIELDS,
        "refit training metadata",
    )
    stats = loader_audit["stats"]
    if not isinstance(stats, Mapping):
        raise ValueError("refit stats audit must be an object")
    _require_exact_keys(
        stats,
        {
            "validation_fold",
            "image_size",
            "image_count",
            "content_pixel_count",
            "mean",
            "std",
            "training_id_sha256",
        },
        "refit stats audit",
    )
    mappings = load_label_maps(label_map_path)
    canonical_labels = list(mappings["season"]["classes"])
    canonical_label_to_index = dict(mappings["season"]["label_to_index"])
    canonical_auxiliary_labels = list(mappings["articleType"]["classes"])
    canonical_auxiliary_to_index = dict(mappings["articleType"]["label_to_index"])
    expected_train_config = _build_train_config(freeze, selected_config)
    expected_transform_id = ImageTransformSpec(
        image_size=selected_config.data.image_size,
        augmentation=selected_config.data.augmentation,
    ).transform_id
    if (
        manifest["refit_id"] != freeze["freeze_id"]
        or not str(manifest["run_id"]).strip()
        or selected["candidate"] != manifest["selected_candidate"]
        or selected["experiment_id"] != manifest["selected_experiment_id"]
        or selected["model_family"] != manifest["model_family"]
        or selected_config.experiment_id != manifest["selected_experiment_id"]
        or rule["seed"] != manifest["seed"]
        or rule["epochs"] != manifest["final_epoch"]
        or manifest["valid_development_rows"]
        != freeze["primary_development_evidence"]["valid_development_rows"]
        or manifest["parameter_count"] != selected["parameter_count"]
        or float(manifest["temperature"]) != float(freeze["calibration"]["temperature"])
        or loader_audit.get("partition") != "development"
        or loader_audit.get("training_products") != manifest["valid_development_rows"]
        or loader_audit.get("validation_products") != 0
        or loader_audit.get("protected_products") != 0
        or loader_audit.get("normalisation_scope") != rule["normalisation_scope"]
        or loader_audit.get("labels") != canonical_labels
        or loader_audit.get("auxiliary_target") != "articleType"
        or loader_audit.get("auxiliary_labels") != canonical_auxiliary_labels
        or loader_audit.get("train_transform_id") != expected_transform_id
        or stats.get("validation_fold") is not None
        or stats.get("image_size") != list(selected_config.data.image_size)
        or stats.get("image_count") != manifest["valid_development_rows"]
        or stats.get("training_id_sha256") != loader_audit.get("training_id_sha256")
        or type(stats.get("content_pixel_count")) is not int
        or stats["content_pixel_count"] <= 0
        or not isinstance(stats.get("mean"), list)
        or len(stats["mean"]) != 3
        or not isinstance(stats.get("std"), list)
        or len(stats["std"]) != 3
        or any(float(value) <= 0 for value in stats["std"])
        or training_metadata.get("validation_used") is not False
        or training_metadata.get("early_stopping_used") is not False
        or training_metadata.get("selection_metric") is not None
        or training_metadata.get("checkpoint_rule") != rule["checkpoint_rule"]
        or training_metadata.get("auxiliary_weight") != selected_config.auxiliary.loss_weight
    ):
        raise ValueError("development refit disagrees with the immutable G7 contract")
    if compute_sha256(splits_path) != freeze["canonical_inputs"]["splits"]["sha256"]:
        raise ValueError("refit splits disagree with the selection freeze")
    if compute_sha256(label_map_path) != freeze["canonical_inputs"]["label_maps"]["sha256"]:
        raise ValueError("refit label maps disagree with the selection freeze")

    artifacts = manifest["artifacts"]
    _require_exact_keys(artifacts, {"history", "runtime"}, "refit artifacts")
    history_path = _resolve_declaration(artifacts["history"], root=root, scope="history")
    runtime_path = _resolve_declaration(artifacts["runtime"], root=root, scope="runtime")
    history = pd.read_csv(history_path)
    if (
        tuple(history.columns) != HISTORY_COLUMNS
        or len(history) != manifest["final_epoch"]
        or history["epoch"].tolist() != list(range(1, manifest["final_epoch"] + 1))
        or not history["train_samples"].eq(manifest["valid_development_rows"]).all()
        or any("validation" in column.lower() for column in history.columns)
    ):
        raise ValueError("development refit history is incomplete or contains validation")
    runtime = _load_json_object(runtime_path, "development refit runtime")
    _require_exact_keys(runtime, RUNTIME_FIELDS, "development refit runtime")
    if (
        runtime.get("schema_version") != "1.0.0"
        or runtime.get("gate") != REFIT_GATE
        or runtime.get("run_id") != manifest["run_id"]
        or runtime.get("holdout_opened") is not False
        or runtime.get("validation_used") is not False
        or runtime.get("primary_metric_name") is not None
        or runtime.get("parameter_count") != manifest["parameter_count"]
        or runtime.get("training_metadata") != dict(training_metadata)
    ):
        raise ValueError("development refit runtime crossed an evaluation boundary")

    bundle_path = _resolve_declaration(manifest["bundle"], root=root, scope="model bundle")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict):
        raise ValueError("model bundle must be a mapping")
    _require_exact_keys(bundle, BUNDLE_FIELDS, "model bundle")
    bundle_section_fields = {
        "model_spec": set(SeasonModelSpec.__dataclass_fields__),
        "model_boundary": {
            "class",
            "training_origin",
            "benchmark_only",
            "final_eligible",
            "weights",
        },
        "auxiliary": {
            "target",
            "labels",
            "label_to_index",
            "loss_weight",
            "missing_label_rule",
            "used_at_inference",
        },
        "preprocessing": {
            "image_size",
            "mean",
            "std",
            "content_pixel_count",
            "normalisation_scope",
            "training_augmentation",
            "inference_augmentation",
            "geometry",
        },
        "calibration": {
            "method",
            "temperature",
            "fit_scope",
            "purpose",
            "review_threshold",
        },
        "training": set(RefitTrainConfig.__dataclass_fields__)
        | {
            "final_epoch",
            "dataset",
            "validation_used",
            "early_stopping_used",
            "season_class_weights",
            "checkpoint_rule",
        },
        "provenance": {
            "run_id",
            "git_commit",
            "git_dirty",
            "selection_freeze_sha256",
            "selected_config_sha256",
            "split_sha256",
            "label_map_sha256",
            "implementation_sha256",
            "training_id_sha256",
        },
        "inference": {"inputs", "output", "class_order", "auxiliary_target_used"},
    }
    for name, expected_fields in bundle_section_fields.items():
        section = bundle[name]
        if not isinstance(section, Mapping):
            raise ValueError(f"model bundle {name} must be an object")
        _require_exact_keys(section, expected_fields, f"model bundle {name}")
    model_spec = SeasonModelSpec(**bundle["model_spec"])
    model = build_multitask_season_model(
        model_spec,
        article_type_classes=len(bundle["auxiliary"]["labels"]),
    )
    boundary = assert_final_model(model)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    if (
        bundle["format_version"] != 1
        or bundle["task"] != "task2"
        or bundle["target"] != "season"
        or bundle["candidate"] != manifest["selected_candidate"]
        or bundle["experiment_id"] != manifest["selected_experiment_id"]
        or bundle["model_boundary"] != boundary
        or bundle["labels"] != canonical_labels
        or bundle["label_to_index"] != canonical_label_to_index
        or bundle["auxiliary"]["target"] != "articleType"
        or bundle["auxiliary"]["labels"] != canonical_auxiliary_labels
        or bundle["auxiliary"]["label_to_index"] != canonical_auxiliary_to_index
        or bundle["auxiliary"]["loss_weight"] != selected_config.auxiliary.loss_weight
        or bundle["auxiliary"]["missing_label_rule"] != rule["article_type_missing_labels"]
        or bundle["auxiliary"]["used_at_inference"] is not False
        or bundle["preprocessing"]["image_size"] != stats["image_size"]
        or bundle["preprocessing"]["mean"] != stats["mean"]
        or bundle["preprocessing"]["std"] != stats["std"]
        or bundle["preprocessing"]["content_pixel_count"] != stats["content_pixel_count"]
        or bundle["preprocessing"]["normalisation_scope"] != rule["normalisation_scope"]
        or bundle["preprocessing"]["training_augmentation"] != selected_config.data.augmentation
        or bundle["preprocessing"]["inference_augmentation"] != "none"
        or bundle["preprocessing"]["geometry"] != "preserve_aspect_ratio_then_pad"
        or bundle["training"]
        != {
            **asdict(expected_train_config),
            "final_epoch": manifest["final_epoch"],
            "dataset": rule["dataset"],
            "validation_used": False,
            "early_stopping_used": False,
            "season_class_weights": None,
            "checkpoint_rule": rule["checkpoint_rule"],
        }
        or bundle["calibration"]
        != {
            "method": freeze["calibration"]["method"],
            "temperature": manifest["temperature"],
            "fit_scope": freeze["calibration"]["fit_scope"],
            "purpose": freeze["calibration"]["purpose"],
            "review_threshold": None,
        }
        or bundle["provenance"]["run_id"] != manifest["run_id"]
        or bundle["provenance"]["selection_freeze_sha256"] != manifest["selection_freeze"]["sha256"]
        or bundle["provenance"]["selected_config_sha256"] != manifest["selected_config"]["sha256"]
        or bundle["provenance"]["split_sha256"] != manifest["canonical_inputs"]["splits"]["sha256"]
        or bundle["provenance"]["label_map_sha256"]
        != manifest["canonical_inputs"]["label_maps"]["sha256"]
        or bundle["provenance"]["implementation_sha256"] != manifest["implementation_sha256"]
        or bundle["provenance"]["training_id_sha256"] != loader_audit["training_id_sha256"]
        or bundle["provenance"]["git_commit"] != manifest["git_commit"]
        or bundle["provenance"]["git_dirty"] is not False
        or bundle["inference"]["inputs"] != ["image"]
        or bundle["inference"]["output"] != "temperature_scaled_season_probabilities"
        or bundle["inference"]["class_order"] != canonical_labels
        or bundle["inference"]["auxiliary_target_used"] is not False
        or sum(parameter.numel() for parameter in model.parameters()) != manifest["parameter_count"]
    ):
        raise ValueError("model bundle metadata disagrees with its verified manifest")
    return manifest, manifest_path, bundle


def run_or_load_development_refit(
    *,
    mode: ExecutionMode = "run_or_load",
    project_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    freeze_path: str | Path = TASK2_SELECTION_FREEZE_JSON,
    registry_path: str | Path = RUNS_CSV,
    bundle_path: str | Path = TASK2_MODEL_PATH,
    manifest_path: str | Path = TASK2_MODEL_MANIFEST_JSON,
    history_path: str | Path = REFIT_HISTORY_CSV,
    runtime_path: str | Path = REFIT_RUNTIME_JSON,
) -> RefitOutcome:
    """Run once or load the frozen I2 refit while keeping holdout sealed."""
    if mode not in {"run", "load", "run_or_load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    root = Path(project_root).resolve()

    def resolved(path: str | Path) -> Path:
        candidate = Path(path)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    splits = resolved(splits_path)
    labels = resolved(label_map_path)
    freeze = resolved(freeze_path)
    registry_file = resolved(registry_path)
    bundle_file = resolved(bundle_path)
    manifest_file = resolved(manifest_path)
    history_file = resolved(history_path)
    runtime_file = resolved(runtime_path)
    for output in (bundle_file, manifest_file, history_file, runtime_file):
        try:
            output.relative_to(root)
        except ValueError as error:
            raise ValueError(f"refit output is outside project root: {output}") from error

    if manifest_file.is_file():
        manifest, verified_manifest, _ = load_verified_development_refit_manifest(
            manifest_file,
            project_root=root,
        )
        if mode == "run":
            raise FileExistsError("a verified development refit already exists; use load")
        return _outcome(manifest, verified_manifest, source="load")
    if mode == "load":
        raise FileNotFoundError(f"development refit manifest does not exist: {manifest_file}")
    if bundle_file.exists():
        raise FileExistsError(
            "an unmanifested Task 2 model bundle exists; audit it before a new refit"
        )

    contract = _load_refit_contract(
        project_root=root,
        freeze_path=freeze,
        splits_path=splits,
        label_map_path=labels,
    )
    seed_everything(contract.train_config.seed)
    experiment_id = f"task2-season-{contract.freeze['selected_model']['candidate'].lower()}-refit"
    run_id = new_run_id(experiment_id, None, contract.train_config.seed)
    record = RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        fold=None,
        seed=contract.train_config.seed,
        config_sha256=contract.config_sha256,
        split_sha256=compute_sha256(splits),
        label_map_sha256=compute_sha256(labels),
        implementation_sha256=contract.implementation_sha256,
        stage=REFIT_STAGE,
        model_family=contract.config.model_family,
        benchmark_only=False,
        final_eligible=True,
        scratch=True,
        git_commit=contract.git_commit,
        git_dirty=False,
        transform_id=str(contract.freeze["primary_development_evidence"]["transform_id"]),
        loss_id=contract.config.loss_id,
        epochs_requested=contract.train_config.epochs,
        primary_metric_name="",
    )
    registry = RunRegistry(registry_file)
    runtime_environment = capture_runtime()
    with tracked_run(registry, record) as run:
        loaders = build_development_multitask_loader(
            image_size=contract.config.data.image_size,
            batch_size=contract.config.data.batch_size,
            main_target=contract.config.target,
            auxiliary_target=contract.config.auxiliary.target,
            augmentation=contract.config.data.augmentation,
            seed=contract.train_config.seed,
            num_workers=contract.config.data.num_workers,
            pin_memory=contract.config.data.pin_memory,
            root=root,
            splits_path=splits,
            label_map_path=labels,
        )
        loader_audit = loaders.audit()
        expected_rows = int(
            contract.freeze["primary_development_evidence"]["valid_development_rows"]
        )
        if (
            len(loaders.training_ids) != expected_rows
            or loaders.stats.image_count != expected_rows
            or loader_audit["protected_products"] != 0
            or loader_audit["validation_products"] != 0
        ):
            raise ValueError("development refit loader does not match the frozen row boundary")
        model_spec = SeasonModelSpec(
            family=contract.config.model_family,
            num_classes=len(loaders.labels),
        )
        model = build_multitask_season_model(
            model_spec,
            article_type_classes=len(loaders.auxiliary_labels),
        )
        boundary = assert_final_model(model)
        result = train_masked_multitask_refit(
            model,
            loaders.train,
            config=contract.train_config,
            auxiliary_weight=contract.config.auxiliary.loss_weight,
        )
        expected_parameters = int(contract.freeze["selected_model"]["parameter_count"])
        if (
            result.epochs_completed != contract.train_config.epochs
            or result.final_epoch != contract.train_config.epochs
            or result.parameter_count != expected_parameters
            or result.metadata["validation_used"] is not False
            or result.metadata["early_stopping_used"] is not False
            or result.metadata["selection_metric"] is not None
        ):
            raise ValueError("refit trainer did not preserve the frozen fixed-epoch rule")

        temperature = float(contract.freeze["calibration"]["temperature"])
        bundle_payload = {
            "format_version": 1,
            "task": "task2",
            "target": contract.config.target,
            "candidate": contract.freeze["selected_model"]["candidate"],
            "experiment_id": contract.config.experiment_id,
            "model_spec": asdict(model_spec),
            "model_boundary": boundary,
            "model_state_dict": _cpu_state_dict(model),
            "labels": list(loaders.labels),
            "label_to_index": loaders.label_to_index,
            "auxiliary": {
                "target": contract.config.auxiliary.target,
                "labels": list(loaders.auxiliary_labels),
                "label_to_index": loaders.auxiliary_label_to_index,
                "loss_weight": contract.config.auxiliary.loss_weight,
                "missing_label_rule": contract.freeze["refit_rule"]["article_type_missing_labels"],
                "used_at_inference": False,
            },
            "preprocessing": {
                "image_size": list(loaders.stats.image_size),
                "mean": list(loaders.stats.mean),
                "std": list(loaders.stats.std),
                "content_pixel_count": loaders.stats.content_pixel_count,
                "normalisation_scope": loader_audit["normalisation_scope"],
                "training_augmentation": contract.config.data.augmentation,
                "inference_augmentation": "none",
                "geometry": "preserve_aspect_ratio_then_pad",
            },
            "calibration": {
                "method": contract.freeze["calibration"]["method"],
                "temperature": temperature,
                "fit_scope": contract.freeze["calibration"]["fit_scope"],
                "purpose": contract.freeze["calibration"]["purpose"],
                "review_threshold": None,
            },
            "training": {
                **asdict(contract.train_config),
                "final_epoch": result.final_epoch,
                "dataset": contract.freeze["refit_rule"]["dataset"],
                "validation_used": False,
                "early_stopping_used": False,
                "season_class_weights": None,
                "checkpoint_rule": contract.freeze["refit_rule"]["checkpoint_rule"],
            },
            "provenance": {
                "run_id": run_id,
                "git_commit": contract.git_commit,
                "git_dirty": False,
                "selection_freeze_sha256": compute_sha256(contract.freeze_path),
                "selected_config_sha256": compute_sha256(contract.config_path),
                "split_sha256": compute_sha256(splits),
                "label_map_sha256": compute_sha256(labels),
                "implementation_sha256": contract.implementation_sha256,
                "training_id_sha256": loader_audit["training_id_sha256"],
            },
            "inference": {
                "inputs": ["image"],
                "output": "temperature_scaled_season_probabilities",
                "class_order": list(loaders.labels),
                "auxiliary_target_used": False,
            },
        }
        bundle_sha256 = _save_bundle(bundle_file, bundle_payload)
        history = _history_frame(result)
        atomic_write_csv(history_file, history)
        runtime_payload = {
            "schema_version": "1.0.0",
            "gate": REFIT_GATE,
            "run_id": run_id,
            "runtime_seconds": result.runtime_seconds,
            "peak_vram_mb": result.peak_vram_mb,
            "device": result.device,
            "parameter_count": result.parameter_count,
            "environment": runtime_environment,
            "training_metadata": result.metadata,
            "holdout_opened": False,
            "validation_used": False,
            "primary_metric_name": None,
        }
        atomic_write_json(runtime_file, runtime_payload)
        run.epochs_completed = result.epochs_completed
        run.best_epoch = None
        run.parameter_count = result.parameter_count
        run.runtime_seconds = result.runtime_seconds
        run.peak_vram_mb = result.peak_vram_mb
        run.checkpoint_path = _relative_path(bundle_file, root=root)
        run.checkpoint_sha256 = bundle_sha256
        run.history_path = _relative_path(history_file, root=root)
        run.history_sha256 = compute_sha256(history_file)
        run.metrics = {}

    manifest = {
        "schema_version": "1.0.0",
        "gate": REFIT_GATE,
        "status": "complete",
        "analysis_role": REFIT_ANALYSIS_ROLE,
        "refit_id": contract.freeze["freeze_id"],
        "run_id": run_id,
        "selected_candidate": contract.freeze["selected_model"]["candidate"],
        "selected_experiment_id": contract.config.experiment_id,
        "model_family": contract.config.model_family,
        "scratch": True,
        "weights": None,
        "benchmark_only": False,
        "final_eligible": True,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "validation_used": False,
        "early_stopping_used": False,
        "primary_metric_name": None,
        "evaluation_claim_allowed": False,
        "seed": result.seed,
        "final_epoch": result.final_epoch,
        "valid_development_rows": len(loaders.training_ids),
        "parameter_count": result.parameter_count,
        "temperature": temperature,
        "loader_audit": loader_audit,
        "training_metadata": result.metadata,
        "git_commit": contract.git_commit,
        "git_dirty": False,
        "implementation_files_at_head": list(contract.implementation_files),
        "implementation_sha256": contract.implementation_sha256,
        "canonical_inputs": {
            "splits": _declaration(splits, root=root),
            "label_maps": _declaration(labels, root=root),
        },
        "selection_freeze": _declaration(contract.freeze_path, root=root),
        "selected_config": _declaration(contract.config_path, root=root),
        "bundle": _declaration(bundle_file, root=root),
        "artifacts": {
            "history": _declaration(history_file, root=root),
            "runtime": _declaration(runtime_file, root=root),
        },
        "model_change_after_holdout_allowed": False,
    }
    atomic_write_json(manifest_file, manifest)
    verified, verified_path, _ = load_verified_development_refit_manifest(
        manifest_file,
        project_root=root,
    )
    return _outcome(verified, verified_path, source="run")


__all__ = [
    "REFIT_EVIDENCE_DIRECTORY",
    "REFIT_GATE",
    "REFIT_HISTORY_CSV",
    "REFIT_IMPLEMENTATION_PATHS",
    "REFIT_RUNTIME_JSON",
    "RefitOutcome",
    "load_verified_development_refit_manifest",
    "run_or_load_development_refit",
]
