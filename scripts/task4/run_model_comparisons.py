import os
import sys


def _bootstrap_cuda_visibility() -> None:
    """Narrow direct GPU invocations before importing torch."""

    argv = sys.argv
    if "--dry-run" in argv or "--budget-gate-combine" in argv:
        return
    gpu_flag = "--gpu"
    if gpu_flag not in argv:
        return
    gpu_index = argv.index(gpu_flag) + 1
    if gpu_index >= len(argv):
        return
    requested = argv[gpu_index]
    if requested == "0" and os.environ.get("TASK4_CUDA_ISOLATED") == "1":
        return
    if any(os.environ.get(name) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK")):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = requested
    os.environ["TASK4_PHYSICAL_CUDA_DEVICE"] = requested
    os.environ["TASK4_CUDA_ISOLATED"] = "1"
    argv[gpu_index] = "0"


_bootstrap_cuda_visibility()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mla2-task4-cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mla2-task4-cache/xdg")

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import time
import uuid
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fashion.config import ROOT, SPLITS_CSV
from fashion.data.dataset import load_splits
from fashion.data.splits import cv_assignment_digest, validate_split_structure
from fashion.task4.augmentation import DEFAULT_GEOMETRY_POLICY
from fashion.task4.cache import (
    ensure_development_image_cache,
    fit_cached_fold_rgb_statistics,
)
from fashion.task4.experiments import (
    B1_CONFIG,
    R1_CONFIG,
    R2_CONFIG,
    R5_CONFIG,
    DeploymentDecisionInput,
    EvidenceIdentity,
    ExperimentConfig,
    ExperimentConfigArtifact,
    StabilityEvidenceInput,
    StabilityFinalist,
    StabilityRunPlan,
    Task6ManifestInput,
    build_experiment_matrix,
    build_stability_plan,
    derive_r3_config,
    derive_r4_config,
    select_deployment_candidate,
    select_gallery_source,
    select_stability_finalists,
    summarize_stability,
)
from fashion.task4.learned_data import (
    CrossSourcePairDataset,
    FamilyBatchSampler,
    build_training_pairs,
)
from fashion.task4.learned_evidence import (
    build_learned_evidence,
    build_stability_evidence,
    encode_development_cache,
    evaluate_learned_quality,
    make_milestone_scorer,
    reconstruct_training_result,
    record_evidence_failure,
)
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.training import (
    AugmentationPolicy,
    CandidateConfig,
    CheckpointRecord,
    SourcePolicy,
    TrainingHyperparameters,
    TrainingSessionConfig,
    WarmupCosineScheduler,
    build_optimizer,
    compute_batch_loss,
    configure_determinism,
    load_checkpoint,
    make_data_generator,
    make_grad_scaler,
    make_worker_init_fn,
    run_training_attempt,
    save_checkpoint,
    train_epochs,
)
from fashion.train.registry import (
    RUN_COLUMNS,
    TASK4_RUN_COLUMNS,
)
from fashion.train.registry import (
    Task4RunRegistry as RunRegistry,
)

PHASES = (
    "smoke",
    "candidate",
    "stability",
    "gallery",
    "evidence",
    "evidence-resume",
    "evidence-recovery",
    "stability-evidence-recovery",
)
SMOKE_FAMILIES = {
    "incremental_encoder": R1_CONFIG,
    "autoencoder": R5_CONFIG,
    "pretrained_benchmark": B1_CONFIG,
}
THROUGHPUT_CONFIGS = {
    "incremental_encoder_r1": R1_CONFIG,
    "incremental_encoder_resnet34": R2_CONFIG,
    "geometry_encoder_r3": ExperimentConfig(
        candidate=CandidateConfig("R3", "resnet34"),
        hyperparameters=TrainingHyperparameters(),
        objective="vicreg",
        source_policy=SourcePolicy.TEACHER_V1_PAIRS,
        augmentation_policy=AugmentationPolicy.GEOMETRY,
        run_kind="candidate",
        parent_run_id="budget-parent-r2",
    ),
    "triplet_encoder_r4": ExperimentConfig(
        candidate=CandidateConfig("R4", "resnet34"),
        hyperparameters=TrainingHyperparameters(),
        objective="vicreg_triplet",
        source_policy=SourcePolicy.TEACHER_V1_PAIRS,
        augmentation_policy=AugmentationPolicy.GEOMETRY,
        run_kind="candidate",
        parent_run_id="budget-parent-r3",
    ),
    "autoencoder": R5_CONFIG,
    "pretrained_benchmark": B1_CONFIG,
}
BUDGET_GPU_HOURS = 98.0
BUDGET_WARMUP_STEPS = 2
BUDGET_MEASURED_STEPS = 5
BUDGET_NON_TRAINING_OVERHEAD_GPU_HOURS = 0.25
CONSERVATIVE_STABILITY_RUNS = 10
SCRATCH_THROUGHPUT_FAMILIES = tuple(
    family for family, config in THROUGHPUT_CONFIGS.items() if not config.pretrained
)
PHASE_REQUEST_SCHEMA_VERSION = 1
PHASE_REQUEST_ARTIFACT_TYPE = "task4_phase_request"
GALLERY_PHASE_REQUEST_ARTIFACT_TYPE = "task4_gallery_phase_request"
EVIDENCE_RESUME_ARTIFACT_TYPE = "task4_evidence_resume_request"
EVIDENCE_RECOVERY_ARTIFACT_TYPE = "task4_evidence_recovery_request"
R5_RECOVERABLE_RUN_ID = "task4-candidate-r5-task9-preexec"
R5_RECOVERABLE_ERROR_TYPE = "ValueError"
R5_RECOVERABLE_ERROR_MESSAGE = "query metric values must be null or finite in [0, 1]"
STABILITY_EVIDENCE_RECOVERY_ARTIFACT_TYPE = "task4_stability_evidence_recovery_request"
STABILITY_RECOVERABLE_ERROR_TYPE = "ValueError"
STABILITY_RECOVERABLE_ERROR_MESSAGE = "stability evidence primary coverage is incomplete"
STABILITY_EVIDENCE_SCOPE = "lightweight_primary_score_and_coverage"
PHASE_REQUEST_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_type",
    "phase",
    "candidate",
    "run_id",
    "paths",
    "budget_gate",
    "hyperparameters",
    "selected_gallery_policy",
}
STABILITY_PHASE_REQUEST_FIELDS = {
    "mode",
    "fold",
    "attempt_token",
    "parent_run_id",
    "lineage",
    "evidence_scope",
}
GALLERY_PHASE_REQUEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "phase",
    "deployment_inputs",
    "selected_fold1_manifest",
}
GALLERY_DEPLOYMENT_INPUT_FIELDS = {
    "registry_row",
    "candidate_score",
    "finalist_config_artifact_path",
    "stability_rows",
    "stability_manifests",
    "candidate_manifest",
}
GALLERY_MANIFEST_SPEC_FIELDS = {
    "run_id",
    "config_artifact_path",
    "checkpoint_path",
    "manifest_path",
}
PHASE_REQUEST_PATH_FIELDS = {
    "variant_index_path",
    "cache_root",
    "checkpoint_root",
    "evidence_root",
    "feature_cache_root",
}
PHASE_REQUEST_HYPERPARAMETER_FIELDS = {
    "seed",
    "product_batch_size",
    "images_per_product",
    "learning_rate",
    "weight_decay",
    "warmup_epochs",
    "minimum_learning_rate",
    "gradient_clip_norm",
    "amp_initial_scale",
    "amp_growth_interval",
    "planned_epochs",
    "checkpoint_epochs",
}
PROTECTED_PARTITIONS = {"holdout", "quarantine"}
TEACHER_TEST_MARKERS = (
    "data/raw/teacher/test",
    "teacher/test",
    "images_test",
    "styles_prediction.csv",
)


class SmokeResult(NamedTuple):
    run_id: str
    family: str
    method: str
    status: str


def _utc_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _hyperparameters_from_json(values: dict[str, object]) -> TrainingHyperparameters:
    converted = dict(values)
    if "checkpoint_epochs" in converted:
        converted["checkpoint_epochs"] = tuple(converted["checkpoint_epochs"])  # type: ignore[arg-type]
    return TrainingHyperparameters(**converted)


def _experiment_config_artifact_payload(
    session: TrainingSessionConfig,
    *,
    checkpoint_sha256: str,
) -> dict[str, object]:
    config_values = json.loads(session.config_json)
    return {
        "schema_version": 1,
        "artifact_type": "task4_experiment_config",
        "identity": {
            "run_id": session.run_id,
            "method": session.candidate.candidate,
            "fold": session.validation_fold,
            "config_hash": session.config_hash,
            "split_fingerprint": session.split_fingerprint,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "candidate": config_values["candidate"],
        "hyperparameters": config_values["hyperparameters"],
        "objective": session.objective,
        "source_policy": session.source_policy.value,
        "augmentation_policy": session.augmentation_policy.value,
        "run_kind": session.run_kind,
        "parent_run_id": session.parent_run_id,
        "canonical_config_json": session.config_json,
        "candidate_config_hash": session.config_hash,
    }


def write_experiment_config_artifact(
    path: Path,
    *,
    session: TrainingSessionConfig,
    checkpoint_sha256: str,
) -> Path:
    payload = _experiment_config_artifact_payload(session, checkpoint_sha256=checkpoint_sha256)
    write_json_atomic(Path(path), payload)
    return Path(path)


def _load_experiment_config_artifact(path: Path) -> ExperimentConfigArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("experiment config artifact schema version must be 1")
    if payload.get("artifact_type") != "task4_experiment_config":
        raise ValueError("experiment config artifact type is invalid")
    identity_values = payload["identity"]
    candidate_values = payload["candidate"]
    if not isinstance(identity_values, dict) or not isinstance(candidate_values, dict):
        raise ValueError("experiment config artifact identity is malformed")
    artifact = ExperimentConfigArtifact(
        identity=EvidenceIdentity(
            run_id=str(identity_values["run_id"]),
            method=str(identity_values["method"]),  # type: ignore[arg-type]
            fold=int(identity_values["fold"]),
            config_hash=str(identity_values["config_hash"]),
            split_fingerprint=str(identity_values["split_fingerprint"]),
            checkpoint_sha256=str(identity_values["checkpoint_sha256"]),
        ),
        candidate=CandidateConfig(
            str(candidate_values["candidate"]),  # type: ignore[arg-type]
            str(candidate_values["architecture"]),  # type: ignore[arg-type]
        ),
        hyperparameters=_hyperparameters_from_json(payload["hyperparameters"]),  # type: ignore[arg-type]
        objective=str(payload["objective"]),
        source_policy=SourcePolicy(str(payload["source_policy"])),
        augmentation_policy=AugmentationPolicy(str(payload["augmentation_policy"])),
        run_kind=str(payload["run_kind"]),
        parent_run_id=payload["parent_run_id"] if payload["parent_run_id"] else None,  # type: ignore[arg-type]
        canonical_config_json=str(payload["canonical_config_json"]),
        candidate_config_hash=str(payload["candidate_config_hash"]),
    )
    artifact.validate()
    return artifact


def current_git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return commit, bool(status.strip())


def load_canonical_splits() -> pd.DataFrame:
    splits = load_splits(SPLITS_CSV)
    validate_split_structure(splits)
    return splits


def reject_sealed_image_rows(frame: pd.DataFrame) -> None:
    if "partition" not in frame:
        raise ValueError("image rows must carry canonical partition values")
    protected = frame["partition"].isin(PROTECTED_PARTITIONS)
    if protected.any():
        ids = frame.loc[protected, "id"].astype(str).head(5).tolist()
        raise ValueError(f"sealed holdout/quarantine rows reached image access: {ids}")
    path_columns = [
        column for column in frame.columns if column.endswith("_path") or column == "path"
    ]
    for column in path_columns:
        lowered = frame[column].astype(str).str.replace("\\", "/", regex=False).str.lower()
        if lowered.map(lambda value: any(marker in value for marker in TEACHER_TEST_MARKERS)).any():
            raise ValueError("official teacher-test path reached image access")


def _reject_distributed_environment() -> None:
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        if os.environ.get(name):
            raise RuntimeError("distributed training is not allowed for Task 4")


def isolated_child_env(physical_gpu: int) -> dict[str, str]:
    _reject_distributed_environment()
    if isinstance(physical_gpu, bool) or physical_gpu < 0:
        raise ValueError("physical GPU index must be a non-negative integer")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["TASK4_PHYSICAL_CUDA_DEVICE"] = str(physical_gpu)
    env["TASK4_CUDA_ISOLATED"] = "1"
    return env


def requested_physical_gpu(gpu: int) -> int:
    return int(os.environ.get("TASK4_PHYSICAL_CUDA_DEVICE", str(gpu)))


def physical_gpu_identity(device: torch.device) -> dict[str, object]:
    ordinal = requested_physical_gpu(0)
    try:
        properties = torch.cuda.get_device_properties(device)
        uuid_value = str(getattr(properties, "uuid", ""))
    except Exception:
        uuid_value = ""
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:
        name = ""
    return {
        "ordinal": ordinal,
        "name": name,
        "uuid": uuid_value,
    }


def pin_single_gpu(index: int) -> torch.device:
    _reject_distributed_environment()
    if isinstance(index, bool) or index != 0:
        raise ValueError("isolated CUDA processes must address visible GPU 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run a registered GPU attempt")
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, found {visible}")
    try:
        torch.cuda.set_device(0)
    except RuntimeError as error:
        if "No CUDA GPUs are available" in str(error) and visible != 1:
            raise RuntimeError(
                f"expected exactly one visible CUDA device, found {visible}"
            ) from error
        raise
    return torch.device("cuda:0")


def load_config_artifacts(root: Path) -> dict[str, Any]:
    artifacts: dict[str, ExperimentConfigArtifact] = {}
    for path in sorted(Path(root).rglob("experiment_config.json")):
        artifact = _load_experiment_config_artifact(path)
        run_id = artifact.identity.run_id
        if run_id in artifacts:
            raise ValueError(f"duplicate experiment config artifact for run {run_id}")
        artifacts[run_id] = artifact
    return artifacts


def _session_for(config: Any, *, run_id: str, split_fingerprint: str) -> TrainingSessionConfig:
    return TrainingSessionConfig(
        run_id=run_id,
        run_kind="smoke",
        candidate=config.candidate,
        hyperparameters=config.hyperparameters,
        objective=config.objective,
        source_policy=config.source_policy,
        augmentation_policy=config.augmentation_policy,
        validation_fold=1,
        split_fingerprint=split_fingerprint,
        parent_run_id=config.parent_run_id,
    )


def _base_registry_row(session: TrainingSessionConfig) -> dict[str, object]:
    commit, dirty = current_git_identity()
    row: dict[str, object] = {column: "" for column in TASK4_RUN_COLUMNS}
    row.update(
        {
            "schema_version": "1",
            "started_at_utc": _utc_z(),
            "completed_at_utc": "",
            "status": "running",
            "git_commit": commit,
            "dirty_tree": dirty,
        }
    )
    row.update(session.expected_registry_identity.as_dict())
    return row


def _synthetic_batch(config: TrainingHyperparameters, device: torch.device) -> dict[str, Any]:
    size = int(config.product_batch_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    teacher = torch.rand(size, 3, 320, 240, generator=generator).to(device)
    v1 = torch.rand(size, 3, 320, 240, generator=generator).to(device)
    masks = torch.ones(size, 320, 240, dtype=torch.bool, device=device)
    return {
        "id": torch.arange(size, dtype=torch.int64, device=device),
        "teacher": teacher,
        "v1": v1,
        "teacher_content_mask": masks,
        "v1_content_mask": masks,
        "product_family_group": [f"family-{index // 2}" for index in range(size)],
        "sha256": [f"sha-{index}" for index in range(size)],
        "duplicate_group": [f"duplicate-{index}" for index in range(size)],
    }


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _execute_synthetic_smoke(
    session: TrainingSessionConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    model = session.build_cpu_model().to(device)
    model.train()
    optimizer = build_optimizer(model, session.hyperparameters)
    scheduler = WarmupCosineScheduler(optimizer, steps_per_epoch=1, config=session.hyperparameters)
    scaler = make_grad_scaler(
        device,
        initial_scale=session.hyperparameters.amp_initial_scale,
        growth_interval=session.hyperparameters.amp_growth_interval,
    )
    gpu_identity = physical_gpu_identity(device) if device.type == "cuda" else {}
    batch = _synthetic_batch(session.hyperparameters, device)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        loss = compute_batch_loss(model, batch, objective=session.candidate.candidate)
    from fashion.task4.training import apply_optimization_step, save_checkpoint

    apply_optimization_step(
        loss.total,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=session.hyperparameters.gradient_clip_norm,
    )
    scheduler.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_checkpoint(
        output_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=session.hyperparameters.planned_epochs,
        session=session,
        score=0.0,
    )
    manifest = output_dir / "smoke_manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": 1,
            "run_id": session.run_id,
            "run_kind": session.run_kind,
            "method": session.candidate.candidate,
            "fold": session.validation_fold,
            "config_hash": session.config_hash,
            "split_fingerprint": session.split_fingerprint,
            "checkpoint_sha256": checkpoint.sha256,
            "result": "synthetic_batch64_forward_backward_passed",
            "visible_device": str(device),
            "physical_gpu": gpu_identity,
        },
    )
    return {
        "parameter_count": _parameter_count(model),
        "selected_epoch": checkpoint.epoch,
        "checkpoint_path": checkpoint.path,
        "checkpoint_sha256": checkpoint.sha256,
        "evidence_manifest_path": manifest,
        "development_winner_score": 0.0,
        "cross_source_score": 0.0,
        "source_robustness_ratio": 0.0,
        "protocol_b_recall_at_10": 0.0,
        "p95_end_to_end_seconds": 0.0,
        "index_bytes": 0,
    }


def run_registered_smoke_attempt(
    *,
    registry: RunRegistry,
    config: Any,
    split_fingerprint: str,
    gpu: int,
    output_root: Path,
    device_factory: Any = pin_single_gpu,
) -> str:
    family = next(name for name, candidate in SMOKE_FAMILIES.items() if candidate is config)
    run_id = f"task4-smoke-{family}-{uuid.uuid4().hex[:12]}"
    session = _session_for(config, run_id=run_id, split_fingerprint=split_fingerprint)
    registry.append(_base_registry_row(session))
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA is already initialized; deterministic setup must run first")
        configure_determinism(session.hyperparameters.seed)
        device = device_factory(gpu)
        run_output = output_root / run_id
        run_output.mkdir(parents=True, exist_ok=True)
        outputs = {
            "selected_epoch": session.hyperparameters.planned_epochs,
            "development_winner_score": 0.0,
            "cross_source_score": 0.0,
            "source_robustness_ratio": 0.0,
            "protocol_b_recall_at_10": 0.0,
            "p95_end_to_end_seconds": 0.0,
            "index_bytes": 0,
            **_execute_synthetic_smoke(session, device, run_output),
        }
        registry.update(
            run_id,
            {
                "status": "completed",
                "completed_at_utc": _utc_z(),
                **outputs,
            },
        )
    except BaseException as error:
        registry.update(
            run_id,
            {
                "status": "failed",
                "completed_at_utc": _utc_z(),
                "error_type": error.__class__.__name__,
                "error_message": (str(error).strip() or error.__class__.__name__)[:500],
            },
        )
        raise
    return run_id


def _dry_run_registry_rows(registry: RunRegistry) -> list[dict[str, str]]:
    if not registry.path.exists():
        return []
    with registry.path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) not in {TASK4_RUN_COLUMNS, RUN_COLUMNS}:
            raise ValueError("registry header does not match the run schema")
        return [
            {column: row.get(column, "") for column in TASK4_RUN_COLUMNS}
            for row in reader
            if row.get("task") == "task4"
        ]


def _candidate_matrix_status(registry: RunRegistry) -> str:
    rows = _dry_run_registry_rows(registry)
    artifacts = load_config_artifacts(ROOT / "results/evidence/task4")
    try:
        matrix = build_experiment_matrix(rows, config_artifacts=artifacts)
    except ValueError as error:
        return f"candidate matrix: pending parents ({error})"
    return "candidate matrix: " + ", ".join(
        f"{method}(parent={config.parent_run_id or 'none'})" for method, config in matrix.items()
    )


def print_dry_run(*, phase: str, registry: RunRegistry, split_fingerprint: str) -> None:
    print(f"phase: {phase}")
    print(f"split_fingerprint: {split_fingerprint}")
    print("smoke matrix:")
    for family, config in SMOKE_FAMILIES.items():
        print(f"  {family} -> {config.method}")
    print(_candidate_matrix_status(registry))


def development_image_rows(splits: pd.DataFrame) -> pd.DataFrame:
    rows = splits.loc[splits["partition"].eq("development")].copy()
    reject_sealed_image_rows(rows)
    return rows


def _path_id_series(frame: pd.DataFrame, column: str) -> pd.Series:
    stems = frame[column].astype(str).map(lambda value: Path(value).stem)
    if not stems.str.fullmatch(r"\d+").all():
        raise ValueError(f"{column} must end in the product ID")
    return stems.astype("int64")


def _assert_optional_canonical_agreement(
    variant: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    column: str,
) -> None:
    if column not in variant:
        return
    left = variant.set_index("id", drop=False).sort_index()[column].astype(str)
    right = canonical.set_index("id", drop=False).sort_index()[column].astype(str)
    if not left.equals(right):
        raise ValueError(f"variant {column} disagrees with canonical splits")


def join_candidate_variant_index(
    variant_index: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    validate_split_structure(splits)
    required_variant_columns = {"id", "teacher_path", "external_path", "external_sha256"}
    if missing := required_variant_columns.difference(variant_index.columns):
        raise ValueError(f"variant index is missing columns: {sorted(missing)}")
    variant = variant_index.copy()
    variant_ids = pd.to_numeric(variant["id"], errors="coerce")
    if variant_ids.isna().any() or not variant_ids.mod(1).eq(0).all():
        raise ValueError("variant IDs must be integer-compatible")
    variant["id"] = variant_ids.astype("int64")
    if variant["id"].duplicated().any():
        raise ValueError("variant IDs must be unique")
    if variant["external_sha256"].astype(str).str.strip().eq("").any():
        raise ValueError("variant external source coverage is incomplete")
    for path_column in ("teacher_path", "external_path"):
        if not _path_id_series(variant, path_column).equals(variant["id"]):
            raise ValueError("variant paths must end in their product IDs")

    canonical = splits.loc[
        :,
        [
            "id",
            "path",
            "sha256",
            "partition",
            "cv_fold",
            "duplicate_group",
            "product_family_group",
        ],
    ].rename(columns={"path": "teacher_path"})
    canonical["id"] = pd.to_numeric(canonical["id"], errors="raise").astype("int64")
    if set(variant["id"].astype(int)) != set(canonical["id"].astype(int)):
        raise ValueError("variant ID coverage does not match canonical splits")
    _assert_optional_canonical_agreement(variant, canonical, column="teacher_path")
    for structural_column in ("partition", "duplicate_group", "product_family_group"):
        _assert_optional_canonical_agreement(variant, canonical, column=structural_column)
    if "cv_fold" in variant:
        left = pd.to_numeric(
            variant.set_index("id", drop=False).sort_index()["cv_fold"].replace(
                r"^\s*$",
                pd.NA,
                regex=True,
            ),
            errors="coerce",
        )
        right = pd.to_numeric(
            canonical.set_index("id", drop=False).sort_index()["cv_fold"].replace(
                r"^\s*$",
                pd.NA,
                regex=True,
            ),
            errors="coerce",
        )
        if not left.fillna(-1).astype(int).equals(right.fillna(-1).astype(int)):
            raise ValueError("variant cv_fold disagrees with canonical splits")

    canonical_columns = {
        "teacher_path",
        "sha256",
        "teacher_sha256",
        "partition",
        "cv_fold",
        "duplicate_group",
        "product_family_group",
    }
    base = variant.drop(columns=[column for column in canonical_columns if column in variant])
    joined = base.merge(canonical, on="id", how="left", validate="one_to_one")
    if joined["sha256"].isna().any():
        raise ValueError("canonical splits do not cover every variant ID")
    joined["teacher_sha256"] = joined["sha256"].astype(str)
    return joined.sort_values("id", kind="mergesort").reset_index(drop=True)


def load_joined_candidate_variant_index(
    variant_index_path: str | Path,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    path = Path(variant_index_path)
    resolved = path if path.is_absolute() else ROOT / path
    variant = pd.read_csv(resolved, keep_default_na=False)
    return join_candidate_variant_index(variant, splits)


def _steps_per_epoch(splits: pd.DataFrame) -> int:
    development = splits.loc[splits["partition"].eq("development")].copy()
    folds = pd.to_numeric(development["cv_fold"], errors="raise").astype(int)
    training_rows = int(folds.ne(1).sum())
    return training_rows // 64


def measure_budget_family(
    *,
    family: str,
    split_fingerprint: str,
    splits: pd.DataFrame,
    gpu: int,
    output_path: Path,
) -> dict[str, object]:
    if family not in THROUGHPUT_CONFIGS:
        raise ValueError("unknown throughput family")
    steps = _steps_per_epoch(splits)
    config = THROUGHPUT_CONFIGS[family]
    run_id = f"task4-budget-{family}-{uuid.uuid4().hex[:12]}"
    session = _session_for(config, run_id=run_id, split_fingerprint=split_fingerprint)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA is already initialized; budget estimate needs a fresh process")
    configure_determinism(session.hyperparameters.seed)
    device = pin_single_gpu(gpu)
    gpu_identity = physical_gpu_identity(device)
    model = session.build_cpu_model().to(device)
    model.train()
    optimizer = build_optimizer(model, session.hyperparameters)
    scaler = make_grad_scaler(
        device,
        initial_scale=session.hyperparameters.amp_initial_scale,
        growth_interval=session.hyperparameters.amp_growth_interval,
    )
    from fashion.task4.training import apply_optimization_step

    def timed_step() -> float:
        batch = _synthetic_batch(session.hyperparameters, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = compute_batch_loss(model, batch, objective=session.candidate.candidate)
        apply_optimization_step(
            loss.total,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=session.hyperparameters.gradient_clip_norm,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - started

    for _ in range(BUDGET_WARMUP_STEPS):
        timed_step()
    samples = [timed_step() for _ in range(BUDGET_MEASURED_STEPS)]
    step_seconds = max(samples)
    run_gpu_hours = step_seconds * steps * session.hyperparameters.planned_epochs / 3600.0
    record = {
        "family": family,
        "method": session.candidate.candidate,
        "architecture": session.candidate.architecture,
        "objective": session.objective,
        "augmentation_policy": session.augmentation_policy.value,
        "config_hash": session.config_hash,
        "amp_growth_interval": session.hyperparameters.amp_growth_interval,
        "step_seconds": step_seconds,
        "conservative_step_seconds": step_seconds,
        "sample_step_seconds": samples,
        "warmup_steps": BUDGET_WARMUP_STEPS,
        "measured_steps": BUDGET_MEASURED_STEPS,
        "steps_per_epoch": steps,
        "run_gpu_hours": run_gpu_hours,
        "non_training_overhead_gpu_hours": BUDGET_NON_TRAINING_OVERHEAD_GPU_HOURS,
        "non_training_overhead_basis": (
            "declared 0.25 GPU-hours per run for checkpointing, evidence scoring, "
            "and package writes"
        ),
        "parameter_count": _parameter_count(model),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": str(device),
        "physical_gpu": gpu_identity,
    }
    write_json_atomic(output_path, record)
    return record


def _aggregate_budget_records(
    records: list[dict[str, object]],
    *,
    split_fingerprint: str,
    output_path: Path,
) -> dict[str, object]:
    by_family = {str(record["family"]): record for record in records}
    required = set(THROUGHPUT_CONFIGS)
    if set(by_family) != required:
        raise ValueError("budget records must cover every throughput family")
    training_total = sum(float(record["run_gpu_hours"]) for record in records)
    overhead_total = sum(
        float(record.get("non_training_overhead_gpu_hours", 0.0)) for record in records
    )
    stability_training, stability_overhead, stability_total = (
        _conservative_stability_budget(by_family)
    )
    total = training_total + overhead_total + stability_total
    physical_gpu = records[0].get("physical_gpu", {}) if records else {}
    record = {
        "schema_version": 1,
        "split_fingerprint": split_fingerprint,
        "batch_size": 64,
        "planned_epochs": 100,
        "budget_gpu_hours": BUDGET_GPU_HOURS,
        "estimate_rule": (
            "Sum six fold-1 runs plus ten conservative stability runs; stability "
            "uses the slowest eligible scratch run plus declared per-run overhead"
        ),
        "warmup_steps": BUDGET_WARMUP_STEPS,
        "measured_steps": BUDGET_MEASURED_STEPS,
        "fold1_candidate_runs": len(records),
        "training_gpu_hours": training_total,
        "non_training_overhead_gpu_hours": overhead_total,
        "conservative_stability_runs": CONSERVATIVE_STABILITY_RUNS,
        "conservative_stability_training_gpu_hours": stability_training,
        "conservative_stability_overhead_gpu_hours": stability_overhead,
        "conservative_stability_gpu_hours": stability_total,
        "physical_gpu": physical_gpu,
        "results": records,
        "estimated_full_matrix_gpu_hours": total,
        "fits_budget": total <= BUDGET_GPU_HOURS,
    }
    write_json_atomic(output_path, record)
    return record


def _conservative_stability_budget(
    by_family: Mapping[str, Mapping[str, object]],
) -> tuple[float, float, float]:
    scratch_records = [by_family[family] for family in SCRATCH_THROUGHPUT_FAMILIES]
    if len(scratch_records) != len(SCRATCH_THROUGHPUT_FAMILIES):
        raise ValueError("budget records must include every eligible scratch family")
    slowest_training = max(float(record["run_gpu_hours"]) for record in scratch_records)
    slowest_overhead = max(
        float(record.get("non_training_overhead_gpu_hours", 0.0))
        for record in scratch_records
    )
    training = CONSERVATIVE_STABILITY_RUNS * slowest_training
    overhead = CONSERVATIVE_STABILITY_RUNS * slowest_overhead
    return training, overhead, training + overhead


def _corrected_budget_totals(
    artifact: Mapping[str, object],
) -> dict[str, float]:
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ValueError("budget artifact results must be a list")
    by_family = {
        str(record.get("family")): record
        for record in results
        if isinstance(record, dict)
    }
    training = sum(float(record["run_gpu_hours"]) for record in by_family.values())
    overhead = sum(
        float(record.get("non_training_overhead_gpu_hours", 0.0))
        for record in by_family.values()
    )
    stability_training, stability_overhead, stability = _conservative_stability_budget(by_family)
    return {
        "fold1_gpu_hours": training + overhead,
        "training_gpu_hours": training,
        "non_training_overhead_gpu_hours": overhead,
        "conservative_stability_training_gpu_hours": stability_training,
        "conservative_stability_overhead_gpu_hours": stability_overhead,
        "conservative_stability_gpu_hours": stability,
        "estimated_full_matrix_gpu_hours": training + overhead + stability,
    }


def _relative_artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validated_budget_artifact(path: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1:
        raise ValueError("budget artifact schema version must be 1")
    if artifact.get("batch_size") != 64 or artifact.get("planned_epochs") != 100:
        raise ValueError("budget artifact must use the approved batch size and epoch count")
    if float(artifact.get("budget_gpu_hours", 0.0)) != BUDGET_GPU_HOURS:
        raise ValueError("budget artifact uses the wrong GPU-hour budget")
    _validate_sha = str(artifact.get("split_fingerprint", ""))
    if len(_validate_sha) != 64 or any(
        character not in "0123456789abcdef" for character in _validate_sha
    ):
        raise ValueError("budget artifact split fingerprint is invalid")
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ValueError("budget artifact results must be a list")
    by_family = {
        str(record.get("family")): record
        for record in results
        if isinstance(record, dict)
    }
    if set(by_family) != set(THROUGHPUT_CONFIGS):
        raise ValueError("budget artifact must cover every throughput family")
    physical_gpu = artifact.get("physical_gpu")
    if not isinstance(physical_gpu, dict) or "ordinal" not in physical_gpu:
        raise ValueError("budget artifact must record physical GPU identity")
    for family, record in by_family.items():
        if record.get("method") != THROUGHPUT_CONFIGS[family].method:
            raise ValueError("budget artifact method does not match the approved family")
        if record.get("deterministic_algorithms") is not True:
            raise ValueError("budget artifact must be deterministic")
        if not str(record.get("device", "")).startswith("cuda:"):
            raise ValueError("budget artifact must come from CUDA")
        if record.get("physical_gpu") != physical_gpu:
            raise ValueError("budget family record physical GPU does not match artifact")
        samples = record.get("sample_step_seconds")
        if not isinstance(samples, list) or len(samples) != int(record.get("measured_steps", 0)):
            raise ValueError("budget artifact must include measured step samples")
        if int(record.get("warmup_steps", 0)) < BUDGET_WARMUP_STEPS:
            raise ValueError("budget artifact warmup count is too small")
        if int(record.get("measured_steps", 0)) < BUDGET_MEASURED_STEPS:
            raise ValueError("budget artifact measured sample count is too small")
        if float(record.get("conservative_step_seconds", 0.0)) != max(
            float(value) for value in samples
        ):
            raise ValueError("budget artifact conservative timing must be the slowest sample")
    totals = _corrected_budget_totals(artifact)
    expected_training = totals["training_gpu_hours"]
    expected_overhead = totals["non_training_overhead_gpu_hours"]
    expected_total = totals["estimated_full_matrix_gpu_hours"]
    expected_fold1_total = totals["fold1_gpu_hours"]
    observed_total = float(artifact.get("estimated_full_matrix_gpu_hours", float("nan")))
    has_task9_fields = "conservative_stability_gpu_hours" in artifact
    compatible_total = expected_total if has_task9_fields else expected_fold1_total
    if abs(observed_total - compatible_total) > 1e-9:
        raise ValueError("budget artifact total does not match the estimate rule")
    if abs(float(artifact.get("training_gpu_hours", 0.0)) - expected_training) > 1e-9:
        raise ValueError("budget artifact training subtotal is inconsistent")
    if abs(float(artifact.get("non_training_overhead_gpu_hours", 0.0)) - expected_overhead) > 1e-9:
        raise ValueError("budget artifact overhead subtotal is inconsistent")
    if has_task9_fields:
        if int(artifact.get("fold1_candidate_runs", 0)) != len(THROUGHPUT_CONFIGS):
            raise ValueError("budget artifact must count all fold-1 candidate runs")
        if int(artifact.get("conservative_stability_runs", 0)) != CONSERVATIVE_STABILITY_RUNS:
            raise ValueError("budget artifact must count ten conservative stability runs")
        for field_name in (
            "conservative_stability_training_gpu_hours",
            "conservative_stability_overhead_gpu_hours",
            "conservative_stability_gpu_hours",
        ):
            if abs(float(artifact.get(field_name, 0.0)) - totals[field_name]) > 1e-9:
                raise ValueError("budget artifact stability subtotal is inconsistent")
    if bool(artifact.get("fits_budget")) != (observed_total <= BUDGET_GPU_HOURS):
        raise ValueError("budget artifact budget verdict is inconsistent")
    return artifact


def write_budget_gate_artifact(
    *,
    source_paths: tuple[Path, Path],
    output_path: Path,
) -> dict[str, object]:
    """Write the canonical Task 8 gate artifact from two validated GPU budget files."""

    if len(source_paths) != 2:
        raise ValueError("budget gate requires exactly two source artifacts")
    sources: list[dict[str, object]] = []
    split_fingerprints: set[str] = set()
    for source_path in source_paths:
        source = Path(source_path)
        artifact = _validated_budget_artifact(source)
        split_fingerprints.add(str(artifact["split_fingerprint"]))
        first_result = artifact["results"][0]
        if not isinstance(first_result, dict):
            raise ValueError("budget artifact result is malformed")
        totals = _corrected_budget_totals(artifact)
        sources.append(
            {
                "path": _relative_artifact_path(source),
                "sha256": sha256_file(source),
                "device": str(first_result["device"]),
                "physical_gpu": artifact["physical_gpu"],
                "estimated_fold1_candidate_gpu_hours": totals["fold1_gpu_hours"],
                "conservative_stability_training_gpu_hours": totals[
                    "conservative_stability_training_gpu_hours"
                ],
                "conservative_stability_overhead_gpu_hours": totals[
                    "conservative_stability_overhead_gpu_hours"
                ],
                "conservative_stability_gpu_hours": totals["conservative_stability_gpu_hours"],
                "estimated_full_matrix_gpu_hours": totals["estimated_full_matrix_gpu_hours"],
                "fits_budget": totals["estimated_full_matrix_gpu_hours"] <= BUDGET_GPU_HOURS,
                "per_family": artifact["results"],
            }
        )
    if len(split_fingerprints) != 1:
        raise ValueError("budget artifacts must use the same split fingerprint")
    selected = max(sources, key=lambda source: float(source["estimated_full_matrix_gpu_hours"]))
    selected_hours = float(selected["estimated_full_matrix_gpu_hours"])
    gate = {
        "schema_version": 1,
        "gate": "task4_model_comparison_budget",
        "split_fingerprint": split_fingerprints.pop(),
        "budget_gpu_hours": BUDGET_GPU_HOURS,
        "selection_rule": (
            "Use the slower A6000 estimate after adding ten conservative "
            "scratch stability runs"
        ),
        "source_artifacts": sources,
        "selected_source_path": selected["path"],
        "selected_source_sha256": selected["sha256"],
        "selected_device": selected["device"],
        "selected_physical_gpu": selected["physical_gpu"],
        "fold1_candidate_runs": len(THROUGHPUT_CONFIGS),
        "conservative_stability_runs": CONSERVATIVE_STABILITY_RUNS,
        "selected_estimated_fold1_candidate_gpu_hours": selected[
            "estimated_fold1_candidate_gpu_hours"
        ],
        "selected_conservative_stability_training_gpu_hours": selected[
            "conservative_stability_training_gpu_hours"
        ],
        "selected_conservative_stability_overhead_gpu_hours": selected[
            "conservative_stability_overhead_gpu_hours"
        ],
        "selected_conservative_stability_gpu_hours": selected[
            "conservative_stability_gpu_hours"
        ],
        "selected_estimated_full_matrix_gpu_hours": selected_hours,
        "fits_budget": selected_hours <= BUDGET_GPU_HOURS,
        "decision": "passed" if selected_hours <= BUDGET_GPU_HOURS else "failed",
    }
    write_json_atomic(output_path, gate)
    return gate


def run_budget_estimate_in_children(
    *,
    split_fingerprint: str,
    gpu: int,
    output_path: Path,
) -> int:
    part_dir = output_path.parent / f".{output_path.stem}-{uuid.uuid4().hex[:8]}"
    part_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures = 0
    for family in THROUGHPUT_CONFIGS:
        part_path = part_dir / f"{family}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "smoke",
            "--budget-family",
            family,
            "--budget-output",
            str(part_path),
            "--gpu",
            str(gpu),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=isolated_child_env(requested_physical_gpu(gpu)),
        )
        if completed.returncode != 0:
            failures += 1
            continue
        records.append(json.loads(part_path.read_text(encoding="utf-8")))
    if failures:
        return 1
    budget = _aggregate_budget_records(
        records,
        split_fingerprint=split_fingerprint,
        output_path=output_path,
    )
    print(
        "budget: "
        f"{budget['estimated_full_matrix_gpu_hours']:.2f}/"
        f"{budget['budget_gpu_hours']:.2f} GPU-hours; "
        f"fits={budget['fits_budget']}"
    )
    return 0


def validate_budget_gate(
    path: Path,
    *,
    split_fingerprint: str,
    require_passed: bool = True,
) -> dict[str, object]:
    gate_path = Path(path)
    if not gate_path.is_file():
        raise ValueError("passed smoke/budget gate artifact is required")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != 1 or gate.get("gate") != "task4_model_comparison_budget":
        raise ValueError("budget gate artifact has the wrong schema")
    if gate.get("split_fingerprint") != split_fingerprint:
        raise ValueError("budget gate split fingerprint does not match canonical splits")
    selected_hours = float(gate.get("selected_estimated_full_matrix_gpu_hours", float("inf")))
    budget_hours = float(gate.get("budget_gpu_hours", BUDGET_GPU_HOURS))
    decision = str(gate.get("decision") or "")
    fits_budget = gate.get("fits_budget") is True
    if require_passed and (decision != "passed" or not fits_budget):
        raise ValueError("passed smoke/budget gate artifact is required")
    if selected_hours > budget_hours:
        if require_passed:
            raise ValueError("budget gate exceeds the GPU-hour budget")
        decision = "failed"
        fits_budget = False
    return {
        "path": _relative_artifact_path(gate_path),
        "sha256": sha256_file(gate_path),
        "decision": decision,
        "selected_device": gate.get("selected_device", ""),
        "selected_estimated_full_matrix_gpu_hours": selected_hours,
        "budget_gpu_hours": budget_hours,
        "fits_budget": fits_budget,
    }


def _checkpoint_sha256_from_result(result: object) -> str:
    if isinstance(result, dict) and "checkpoint_sha256" in result:
        return str(result["checkpoint_sha256"])
    best = getattr(result, "best_checkpoint", None)
    sha = getattr(best, "sha256", None)
    if isinstance(sha, str):
        return sha
    return "0" * 64


def _read_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("phase artifact must be a JSON object")
    return payload


def _candidate_config(candidate: str) -> ExperimentConfig:
    if candidate == "R1":
        return R1_CONFIG
    if candidate == "R2":
        return R2_CONFIG
    if candidate == "R5":
        return R5_CONFIG
    if candidate == "B1":
        return B1_CONFIG
    if candidate in {"R3", "R4"}:
        return THROUGHPUT_CONFIGS[
            "geometry_encoder_r3" if candidate == "R3" else "triplet_encoder_r4"
        ]
    raise ValueError("unknown candidate")


def _repo_relative_path_value(path: str | Path) -> str:
    candidate = Path(path)
    if not str(candidate).strip():
        raise ValueError("phase request paths must not be blank")
    resolved = candidate if candidate.is_absolute() else ROOT / candidate
    try:
        relative = resolved.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("phase request paths must stay inside the repository") from error
    if any(part == ".." for part in relative.parts):
        raise ValueError("phase request paths must be normalized repository paths")
    return relative.as_posix()


def _hyperparameters_payload(config: TrainingHyperparameters) -> dict[str, object]:
    return {
        name: list(value) if name == "checkpoint_epochs" else value
        for name, value in (
            ("seed", config.seed),
            ("product_batch_size", config.product_batch_size),
            ("images_per_product", config.images_per_product),
            ("learning_rate", config.learning_rate),
            ("weight_decay", config.weight_decay),
            ("warmup_epochs", config.warmup_epochs),
            ("minimum_learning_rate", config.minimum_learning_rate),
            ("gradient_clip_norm", config.gradient_clip_norm),
            ("amp_initial_scale", config.amp_initial_scale),
            ("amp_growth_interval", config.amp_growth_interval),
            ("planned_epochs", config.planned_epochs),
            ("checkpoint_epochs", config.checkpoint_epochs),
        )
    }


def validate_phase_request_payload(
    payload: Mapping[str, object],
    *,
    phase: str,
) -> dict[str, object]:
    allowed_fields = set(PHASE_REQUEST_TOP_LEVEL_FIELDS)
    if phase == "stability":
        allowed_fields.update(STABILITY_PHASE_REQUEST_FIELDS)
    if set(payload).difference(allowed_fields):
        raise ValueError("phase request schema contains unknown fields")
    required = PHASE_REQUEST_TOP_LEVEL_FIELDS - {"selected_gallery_policy"}
    if missing := required.difference(payload):
        raise ValueError(f"phase request schema is missing fields: {sorted(missing)}")
    if payload.get("schema_version") != PHASE_REQUEST_SCHEMA_VERSION:
        raise ValueError("phase request schema version must be 1")
    if payload.get("artifact_type") != PHASE_REQUEST_ARTIFACT_TYPE:
        raise ValueError("phase request artifact type is invalid")
    if payload.get("phase") != phase:
        raise ValueError("phase request does not match the CLI phase")
    candidate = str(payload.get("candidate") or "")
    if phase == "candidate" and candidate not in {"R1", "R2", "R3", "R4", "R5", "B1"}:
        raise ValueError("candidate phase request has an invalid candidate")
    if phase == "stability" and candidate not in {"R1", "R2", "R3", "R4", "R5"}:
        raise ValueError("stability phase request has an invalid scratch candidate")
    run_id = str(payload.get("run_id") or "")
    if not run_id.strip():
        raise ValueError("phase request run_id must not be blank")
    paths = payload.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != PHASE_REQUEST_PATH_FIELDS:
        raise ValueError("phase request paths schema is invalid")
    budget_gate = payload.get("budget_gate")
    if (
        not isinstance(budget_gate, Mapping)
        or set(budget_gate) != {"gate", "budget_gpu_hours"}
        or budget_gate.get("gate") != "task4_model_comparison_budget"
        or float(budget_gate.get("budget_gpu_hours", 0.0)) != BUDGET_GPU_HOURS
    ):
        raise ValueError("phase request budget gate identity is invalid")
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("phase request hyperparameters must be an object")
    if set(hyperparameters).difference(PHASE_REQUEST_HYPERPARAMETER_FIELDS):
        raise ValueError("phase request hyperparameters schema is invalid")
    config = _candidate_config(candidate)
    merged = _hyperparameters_payload(config.hyperparameters)
    merged.update(dict(hyperparameters))
    if not isinstance(merged["checkpoint_epochs"], tuple):
        merged["checkpoint_epochs"] = tuple(merged["checkpoint_epochs"])  # type: ignore[arg-type]
    parsed_hyperparameters = TrainingHyperparameters(**merged)
    selected_gallery_policy = str(payload.get("selected_gallery_policy") or "v1")
    if selected_gallery_policy not in {"teacher", "v1", "two_view"}:
        raise ValueError("phase request selected gallery policy is invalid")
    validated: dict[str, object] = {
        "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
        "artifact_type": PHASE_REQUEST_ARTIFACT_TYPE,
        "phase": phase,
        "candidate": candidate,
        "run_id": run_id,
        "selected_gallery_policy": selected_gallery_policy,
        "hyperparameters": _hyperparameters_payload(parsed_hyperparameters),
        "paths": {
            key: _repo_relative_path_value(paths[key])  # type: ignore[index]
            for key in sorted(PHASE_REQUEST_PATH_FIELDS)
        },
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": BUDGET_GPU_HOURS,
        },
    }
    for key, value in validated["paths"].items():  # type: ignore[union-attr]
        validated[key] = Path(str(value))
    if phase == "stability":
        mode = str(payload.get("mode") or "run")
        if mode not in {"run", "plan"}:
            raise ValueError("stability phase request mode is invalid")
        fold = int(payload.get("fold", -1))
        if fold not in range(5):
            raise ValueError("stability phase request fold must be in range(5)")
        attempt_token = str(payload.get("attempt_token") or "")
        parent_run_id = str(payload.get("parent_run_id") or "")
        evidence_scope = str(payload.get("evidence_scope") or "")
        if (
            not attempt_token.strip()
            or not parent_run_id.strip()
            or evidence_scope != "lightweight_primary_score_and_coverage"
        ):
            raise ValueError("stability phase request lineage or evidence scope is invalid")
        lineage = payload.get("lineage")
        if (
            not isinstance(lineage, Mapping)
            or lineage.get("parent_run_id") != parent_run_id
            or lineage.get("copied_recipe_from_run_id") != parent_run_id
            or lineage.get("fresh_retrain_from_scratch") is not True
        ):
            raise ValueError("stability phase request lineage is invalid")
        validated.update(
            {
                "mode": mode,
                "fold": fold,
                "attempt_token": attempt_token,
                "parent_run_id": parent_run_id,
                "lineage": dict(lineage),
                "evidence_scope": evidence_scope,
            }
        )
    return validated


def load_phase_request(path: str | Path, *, phase: str) -> dict[str, object]:
    return validate_phase_request_payload(_read_json(path), phase=phase)


def _validated_gallery_manifest_spec(spec: object) -> dict[str, object]:
    if not isinstance(spec, Mapping) or set(spec) != GALLERY_MANIFEST_SPEC_FIELDS:
        raise ValueError("gallery manifest spec schema is invalid")
    run_id = str(spec.get("run_id") or "")
    if not run_id.strip():
        raise ValueError("gallery manifest spec run_id must not be blank")
    return {
        "run_id": run_id,
        "config_artifact_path": _repo_relative_path_value(
            str(spec["config_artifact_path"])
        ),
        "checkpoint_path": _repo_relative_path_value(str(spec["checkpoint_path"])),
        "manifest_path": _repo_relative_path_value(str(spec["manifest_path"])),
    }


def validate_gallery_phase_request_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if set(payload) != GALLERY_PHASE_REQUEST_FIELDS:
        raise ValueError("gallery phase request schema is invalid")
    if payload.get("schema_version") != PHASE_REQUEST_SCHEMA_VERSION:
        raise ValueError("gallery phase request schema version must be 1")
    if payload.get("artifact_type") != GALLERY_PHASE_REQUEST_ARTIFACT_TYPE:
        raise ValueError("gallery phase request artifact type is invalid")
    if payload.get("phase") != "gallery":
        raise ValueError("gallery phase request phase is invalid")
    raw_inputs = payload.get("deployment_inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 2:
        raise ValueError("gallery phase request requires two deployment inputs")
    validated_inputs: list[dict[str, object]] = []
    candidate_run_ids: set[str] = set()
    methods: set[str] = set()
    candidate_specs: dict[str, dict[str, object]] = {}
    for raw_input in raw_inputs:
        if (
            not isinstance(raw_input, Mapping)
            or set(raw_input) != GALLERY_DEPLOYMENT_INPUT_FIELDS
        ):
            raise ValueError("gallery deployment input schema is invalid")
        registry_row = raw_input.get("registry_row")
        if not isinstance(registry_row, Mapping):
            raise ValueError("gallery deployment registry row is malformed")
        run_id = str(registry_row.get("run_id") or "")
        method = str(registry_row.get("method") or "")
        if (
            not run_id.strip()
            or method not in {"R1", "R2", "R3", "R4", "R5"}
            or registry_row.get("run_kind") != "candidate"
            or registry_row.get("status") != "completed"
            or int(registry_row.get("fold", -1)) != 1
        ):
            raise ValueError("gallery deployment candidate row is invalid")
        candidate_run_ids.add(run_id)
        methods.add(method)
        candidate_manifest = _validated_gallery_manifest_spec(
            raw_input.get("candidate_manifest")
        )
        if candidate_manifest["run_id"] != run_id:
            raise ValueError("gallery candidate manifest run_id does not match registry")
        stability_rows = raw_input.get("stability_rows")
        stability_manifests = raw_input.get("stability_manifests")
        if (
            not isinstance(stability_rows, list)
            or not isinstance(stability_manifests, list)
            or len(stability_rows) != 5
            or len(stability_manifests) != 5
        ):
            raise ValueError("gallery deployment input requires five stability rows and manifests")
        fold_rows: dict[int, Mapping[str, object]] = {}
        for row in stability_rows:
            if not isinstance(row, Mapping):
                raise ValueError("gallery stability registry row is malformed")
            fold = int(row.get("fold", -1))
            if (
                fold in fold_rows
                or fold not in range(5)
                or row.get("method") != method
                or row.get("run_kind") != "stability"
                or row.get("status") != "completed"
                or row.get("parent_run_id") not in (None, "", run_id)
            ):
                raise ValueError("gallery stability registry rows are invalid")
            fold_rows[fold] = dict(row)
        validated_manifests = [
            _validated_gallery_manifest_spec(spec) for spec in stability_manifests
        ]
        if {
            str(row.get("run_id") or "") for row in fold_rows.values()
        } != {str(spec["run_id"]) for spec in validated_manifests}:
            raise ValueError("gallery stability rows and manifests do not match")
        try:
            candidate_score = float(raw_input["candidate_score"])
        except (TypeError, ValueError) as error:
            raise ValueError("gallery candidate score is invalid") from error
        if not math.isfinite(candidate_score):
            raise ValueError("gallery candidate score is invalid")
        validated_input = {
            "registry_row": dict(registry_row),
            "candidate_score": candidate_score,
            "finalist_config_artifact_path": _repo_relative_path_value(
                str(raw_input["finalist_config_artifact_path"])
            ),
            "stability_rows": [dict(fold_rows[fold]) for fold in range(5)],
            "stability_manifests": validated_manifests,
            "candidate_manifest": candidate_manifest,
        }
        validated_inputs.append(validated_input)
        candidate_specs[run_id] = candidate_manifest
    if len(candidate_run_ids) != 2 or len(methods) != 2:
        raise ValueError("gallery deployment inputs require two distinct finalists")
    selected = _validated_gallery_manifest_spec(payload.get("selected_fold1_manifest"))
    if selected != candidate_specs.get(str(selected["run_id"])):
        raise ValueError("selected fold-1 manifest must be one deployment finalist")
    return {
        "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
        "artifact_type": GALLERY_PHASE_REQUEST_ARTIFACT_TYPE,
        "phase": "gallery",
        "deployment_inputs": validated_inputs,
        "selected_fold1_manifest": selected,
    }


def load_gallery_phase_request(path: str | Path) -> dict[str, object]:
    return validate_gallery_phase_request_payload(_read_json(path))


def write_canonical_gallery_phase_request(
    path: Path,
    *,
    deployment_inputs: Sequence[Mapping[str, object]],
    selected_fold1_manifest: Mapping[str, object],
) -> Path:
    payload = validate_gallery_phase_request_payload(
        {
            "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
            "artifact_type": GALLERY_PHASE_REQUEST_ARTIFACT_TYPE,
            "phase": "gallery",
            "deployment_inputs": list(deployment_inputs),
            "selected_fold1_manifest": selected_fold1_manifest,
        }
    )
    write_json_atomic(path, payload)
    return path


def write_canonical_candidate_phase_request(
    path: Path,
    *,
    candidate: str,
    run_id: str,
    variant_index_path: Path,
    cache_root: Path,
    checkpoint_root: Path,
    evidence_root: Path,
    feature_cache_root: Path,
    selected_gallery_policy: str = "v1",
) -> Path:
    config = _candidate_config(candidate)
    payload: dict[str, object] = {
        "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
        "artifact_type": PHASE_REQUEST_ARTIFACT_TYPE,
        "phase": "candidate",
        "candidate": candidate,
        "run_id": run_id,
        "selected_gallery_policy": selected_gallery_policy,
        "paths": {
            "variant_index_path": _repo_relative_path_value(variant_index_path),
            "cache_root": _repo_relative_path_value(cache_root),
            "checkpoint_root": _repo_relative_path_value(checkpoint_root),
            "evidence_root": _repo_relative_path_value(evidence_root),
            "feature_cache_root": _repo_relative_path_value(feature_cache_root),
        },
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": BUDGET_GPU_HOURS,
        },
        "hyperparameters": _hyperparameters_payload(config.hyperparameters),
    }
    validate_phase_request_payload(payload, phase="candidate")
    write_json_atomic(path, payload)
    return path


def write_canonical_stability_phase_request(
    path: Path,
    *,
    finalist: StabilityFinalist,
    plan: StabilityRunPlan,
    variant_index_path: Path,
    cache_root: Path,
    checkpoint_root: Path,
    evidence_root: Path,
    feature_cache_root: Path,
    selected_gallery_policy: str = "v1",
) -> Path:
    if plan.candidate_run_id != finalist.candidate_run_id:
        raise ValueError("stability request plan does not match finalist")
    session = plan.training_session()
    payload: dict[str, object] = {
        "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
        "artifact_type": PHASE_REQUEST_ARTIFACT_TYPE,
        "phase": "stability",
        "candidate": plan.method,
        "run_id": plan.run_id,
        "selected_gallery_policy": selected_gallery_policy,
        "paths": {
            "variant_index_path": _repo_relative_path_value(variant_index_path),
            "cache_root": _repo_relative_path_value(cache_root),
            "checkpoint_root": _repo_relative_path_value(checkpoint_root),
            "evidence_root": _repo_relative_path_value(evidence_root),
            "feature_cache_root": _repo_relative_path_value(feature_cache_root),
        },
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": BUDGET_GPU_HOURS,
        },
        "hyperparameters": _hyperparameters_payload(session.hyperparameters),
        "mode": "run",
        "fold": plan.fold,
        "attempt_token": plan.attempt_token,
        "parent_run_id": plan.parent_run_id,
        "lineage": {
            "parent_run_id": plan.parent_run_id,
            "copied_recipe_from_run_id": finalist.candidate_run_id,
            "fresh_retrain_from_scratch": True,
        },
        "evidence_scope": "lightweight_primary_score_and_coverage",
    }
    validate_phase_request_payload(payload, phase="stability")
    write_json_atomic(path, payload)
    return path


def _resume_request_paths(payload: Mapping[str, object]) -> dict[str, Path]:
    paths = payload.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != PHASE_REQUEST_PATH_FIELDS:
        raise ValueError("evidence resume request paths schema is invalid")
    return {key: Path(str(value)) for key, value in paths.items()}


def validate_evidence_resume_request_payload(payload: Mapping[str, object]) -> dict[str, object]:
    expected_fields = {
        "schema_version",
        "artifact_type",
        "phase",
        "candidate",
        "run_id",
        "paths",
        "checkpoint_paths",
        "budget_gate",
        "selected_gallery_policy",
    }
    if set(payload) != expected_fields:
        raise ValueError("evidence resume request schema is invalid")
    if payload.get("schema_version") != PHASE_REQUEST_SCHEMA_VERSION:
        raise ValueError("evidence resume request schema version must be 1")
    if payload.get("artifact_type") != EVIDENCE_RESUME_ARTIFACT_TYPE:
        raise ValueError("evidence resume request artifact type is invalid")
    if payload.get("phase") != "evidence-resume":
        raise ValueError("evidence resume request phase is invalid")
    candidate = str(payload.get("candidate") or "")
    if candidate not in {"R1", "R2", "R3", "R4", "R5", "B1"}:
        raise ValueError("evidence resume request has an invalid candidate")
    run_id = str(payload.get("run_id") or "")
    if not run_id.strip():
        raise ValueError("evidence resume request run_id must not be blank")
    budget_gate = payload.get("budget_gate")
    if (
        not isinstance(budget_gate, Mapping)
        or set(budget_gate) != {"gate", "budget_gpu_hours"}
        or budget_gate.get("gate") != "task4_model_comparison_budget"
        or float(budget_gate.get("budget_gpu_hours", 0.0)) != BUDGET_GPU_HOURS
    ):
        raise ValueError("evidence resume request budget gate identity is invalid")
    raw_checkpoints = payload.get("checkpoint_paths")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 5:
        raise ValueError("evidence resume request requires exact five checkpoint paths")
    checkpoint_paths = [Path(str(item)) for item in raw_checkpoints]
    if len(set(checkpoint_paths)) != 5:
        raise ValueError("evidence resume checkpoint paths must be unique")
    validated: dict[str, object] = {
        "schema_version": payload["schema_version"],
        "artifact_type": payload["artifact_type"],
        "phase": payload["phase"],
        "candidate": candidate,
        "run_id": run_id,
        "checkpoint_paths": checkpoint_paths,
        "selected_gallery_policy": str(payload["selected_gallery_policy"]),
        "budget_gate": dict(budget_gate),
    }
    paths = _resume_request_paths(payload)
    for key, value in paths.items():
        validated[key] = value
    return validated


def load_evidence_resume_request(path: str | Path) -> dict[str, object]:
    return validate_evidence_resume_request_payload(_read_json(path))


def validate_evidence_recovery_request_payload(payload: Mapping[str, object]) -> dict[str, object]:
    expected_fields = {
        "schema_version",
        "artifact_type",
        "phase",
        "candidate",
        "run_id",
        "paths",
        "checkpoint_paths",
        "budget_gate",
        "selected_gallery_policy",
        "recoverable_error",
    }
    if set(payload) != expected_fields:
        raise ValueError("evidence recovery request schema is invalid")
    if payload.get("schema_version") != PHASE_REQUEST_SCHEMA_VERSION:
        raise ValueError("evidence recovery request schema version must be 1")
    if payload.get("artifact_type") != EVIDENCE_RECOVERY_ARTIFACT_TYPE:
        raise ValueError("evidence recovery request artifact type is invalid")
    if payload.get("phase") != "evidence-recovery":
        raise ValueError("evidence recovery request phase is invalid")
    if payload.get("candidate") != "R5" or payload.get("run_id") != R5_RECOVERABLE_RUN_ID:
        raise ValueError("evidence recovery is limited to the exact failed R5 row")
    error_identity = payload.get("recoverable_error")
    if (
        not isinstance(error_identity, Mapping)
        or error_identity.get("type") != R5_RECOVERABLE_ERROR_TYPE
        or error_identity.get("message") != R5_RECOVERABLE_ERROR_MESSAGE
    ):
        raise ValueError("evidence recovery request has the wrong failed-attempt identity")
    budget_gate = payload.get("budget_gate")
    if (
        not isinstance(budget_gate, Mapping)
        or set(budget_gate) != {"gate", "budget_gpu_hours"}
        or budget_gate.get("gate") != "task4_model_comparison_budget"
        or float(budget_gate.get("budget_gpu_hours", 0.0)) != BUDGET_GPU_HOURS
    ):
        raise ValueError("evidence recovery request budget gate identity is invalid")
    raw_checkpoints = payload.get("checkpoint_paths")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 5:
        raise ValueError("evidence recovery request requires exact five checkpoint paths")
    checkpoint_paths = [Path(str(item)) for item in raw_checkpoints]
    if len(set(checkpoint_paths)) != 5:
        raise ValueError("evidence recovery checkpoint paths must be unique")
    selected_gallery_policy = str(payload.get("selected_gallery_policy") or "")
    if selected_gallery_policy not in {"teacher", "v1", "two_view"}:
        raise ValueError("evidence recovery selected gallery policy is invalid")
    validated: dict[str, object] = {
        "schema_version": payload["schema_version"],
        "artifact_type": payload["artifact_type"],
        "phase": payload["phase"],
        "candidate": "R5",
        "run_id": R5_RECOVERABLE_RUN_ID,
        "checkpoint_paths": checkpoint_paths,
        "selected_gallery_policy": selected_gallery_policy,
        "budget_gate": dict(budget_gate),
        "recoverable_error": dict(error_identity),
    }
    paths = _resume_request_paths(payload)
    for key, value in paths.items():
        validated[key] = value
    return validated


def load_evidence_recovery_request(path: str | Path) -> dict[str, object]:
    return validate_evidence_recovery_request_payload(_read_json(path))


def validate_stability_evidence_recovery_request_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate a request to complete one failed stability run's evidence step."""

    expected_fields = {
        "schema_version",
        "artifact_type",
        "phase",
        "candidate",
        "run_id",
        "fold",
        "attempt_token",
        "parent_run_id",
        "paths",
        "checkpoint_paths",
        "budget_gate",
        "recoverable_error",
        "evidence_scope",
    }
    if set(payload) != expected_fields:
        raise ValueError("stability evidence recovery request schema is invalid")
    if payload.get("schema_version") != PHASE_REQUEST_SCHEMA_VERSION:
        raise ValueError("stability evidence recovery request schema version must be 1")
    if payload.get("artifact_type") != STABILITY_EVIDENCE_RECOVERY_ARTIFACT_TYPE:
        raise ValueError("stability evidence recovery request artifact type is invalid")
    if payload.get("phase") != "stability-evidence-recovery":
        raise ValueError("stability evidence recovery request phase is invalid")
    if payload.get("evidence_scope") != STABILITY_EVIDENCE_SCOPE:
        raise ValueError("stability recovery evidence scope must stay lightweight")
    candidate = str(payload.get("candidate") or "")
    if candidate not in {"R1", "R2", "R3", "R4", "R5"}:
        raise ValueError("stability evidence recovery requires a scratch candidate")
    for field in ("run_id", "attempt_token", "parent_run_id"):
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"stability evidence recovery request {field} must not be blank")
    fold = payload.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, Integral) or int(fold) not in range(5):
        raise ValueError("stability evidence recovery fold must be an integer in range(5)")
    error_identity = payload.get("recoverable_error")
    if (
        not isinstance(error_identity, Mapping)
        or set(error_identity) != {"type", "message"}
        or error_identity.get("type") != STABILITY_RECOVERABLE_ERROR_TYPE
        or error_identity.get("message") != STABILITY_RECOVERABLE_ERROR_MESSAGE
    ):
        raise ValueError(
            "stability evidence recovery request has the wrong failed-attempt identity"
        )
    budget_gate = payload.get("budget_gate")
    if (
        not isinstance(budget_gate, Mapping)
        or set(budget_gate) != {"gate", "budget_gpu_hours"}
        or budget_gate.get("gate") != "task4_model_comparison_budget"
        or float(budget_gate.get("budget_gpu_hours", 0.0)) != BUDGET_GPU_HOURS
    ):
        raise ValueError("stability evidence recovery request budget gate identity is invalid")
    raw_checkpoints = payload.get("checkpoint_paths")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 5:
        raise ValueError("stability evidence recovery requires exact five checkpoint paths")
    checkpoint_paths = [Path(str(item)) for item in raw_checkpoints]
    if len(set(checkpoint_paths)) != 5:
        raise ValueError("stability evidence recovery checkpoint paths must be unique")
    validated: dict[str, object] = {
        "schema_version": payload["schema_version"],
        "artifact_type": payload["artifact_type"],
        "phase": payload["phase"],
        "candidate": candidate,
        "run_id": str(payload["run_id"]),
        "fold": int(fold),
        "attempt_token": str(payload["attempt_token"]),
        "parent_run_id": str(payload["parent_run_id"]),
        "checkpoint_paths": checkpoint_paths,
        "budget_gate": dict(budget_gate),
        "recoverable_error": dict(error_identity),
        "evidence_scope": STABILITY_EVIDENCE_SCOPE,
    }
    for key, value in _resume_request_paths(payload).items():
        validated[key] = value
    return validated


def load_stability_evidence_recovery_request(path: str | Path) -> dict[str, object]:
    return validate_stability_evidence_recovery_request_payload(_read_json(path))


def write_canonical_stability_evidence_recovery_request(
    path: Path,
    *,
    candidate: str,
    run_id: str,
    fold: int,
    attempt_token: str,
    parent_run_id: str,
    checkpoint_paths: Sequence[Path],
    variant_index_path: Path,
    cache_root: Path,
    checkpoint_root: Path,
    evidence_root: Path,
    feature_cache_root: Path,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
        "artifact_type": STABILITY_EVIDENCE_RECOVERY_ARTIFACT_TYPE,
        "phase": "stability-evidence-recovery",
        "candidate": candidate,
        "run_id": run_id,
        "fold": int(fold),
        "attempt_token": attempt_token,
        "parent_run_id": parent_run_id,
        "paths": {
            "variant_index_path": _repo_relative_path_value(variant_index_path),
            "cache_root": _repo_relative_path_value(cache_root),
            "checkpoint_root": _repo_relative_path_value(checkpoint_root),
            "evidence_root": _repo_relative_path_value(evidence_root),
            "feature_cache_root": _repo_relative_path_value(feature_cache_root),
        },
        "checkpoint_paths": [
            _repo_relative_path_value(item) for item in checkpoint_paths
        ],
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": BUDGET_GPU_HOURS,
        },
        "recoverable_error": {
            "type": STABILITY_RECOVERABLE_ERROR_TYPE,
            "message": STABILITY_RECOVERABLE_ERROR_MESSAGE,
        },
        "evidence_scope": STABILITY_EVIDENCE_SCOPE,
    }
    validate_stability_evidence_recovery_request_payload(payload)
    write_json_atomic(path, payload)
    return path


def write_canonical_evidence_resume_request(
    path: Path,
    *,
    candidate: str,
    run_id: str,
    checkpoint_paths: Sequence[Path],
    variant_index_path: Path,
    cache_root: Path,
    checkpoint_root: Path,
    evidence_root: Path,
    feature_cache_root: Path,
    selected_gallery_policy: str = "v1",
) -> Path:
    payload: dict[str, object] = {
        "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
        "artifact_type": EVIDENCE_RESUME_ARTIFACT_TYPE,
        "phase": "evidence-resume",
        "candidate": candidate,
        "run_id": run_id,
        "selected_gallery_policy": selected_gallery_policy,
        "paths": {
            "variant_index_path": _repo_relative_path_value(variant_index_path),
            "cache_root": _repo_relative_path_value(cache_root),
            "checkpoint_root": _repo_relative_path_value(checkpoint_root),
            "evidence_root": _repo_relative_path_value(evidence_root),
            "feature_cache_root": _repo_relative_path_value(feature_cache_root),
        },
        "checkpoint_paths": [_repo_relative_path_value(path) for path in checkpoint_paths],
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": BUDGET_GPU_HOURS,
        },
    }
    validate_evidence_resume_request_payload(payload)
    write_json_atomic(path, payload)
    return path


def write_canonical_evidence_recovery_request(
    path: Path,
    *,
    candidate: str,
    run_id: str,
    checkpoint_paths: Sequence[Path],
    variant_index_path: Path,
    cache_root: Path,
    checkpoint_root: Path,
    evidence_root: Path,
    feature_cache_root: Path,
    selected_gallery_policy: str = "v1",
) -> Path:
    payload: dict[str, object] = {
        "schema_version": PHASE_REQUEST_SCHEMA_VERSION,
        "artifact_type": EVIDENCE_RECOVERY_ARTIFACT_TYPE,
        "phase": "evidence-recovery",
        "candidate": candidate,
        "run_id": run_id,
        "selected_gallery_policy": selected_gallery_policy,
        "paths": {
            "variant_index_path": _repo_relative_path_value(variant_index_path),
            "cache_root": _repo_relative_path_value(cache_root),
            "checkpoint_root": _repo_relative_path_value(checkpoint_root),
            "evidence_root": _repo_relative_path_value(evidence_root),
            "feature_cache_root": _repo_relative_path_value(feature_cache_root),
        },
        "checkpoint_paths": [_repo_relative_path_value(path) for path in checkpoint_paths],
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": BUDGET_GPU_HOURS,
        },
        "recoverable_error": {
            "type": R5_RECOVERABLE_ERROR_TYPE,
            "message": R5_RECOVERABLE_ERROR_MESSAGE,
        },
    }
    validate_evidence_recovery_request_payload(payload)
    write_json_atomic(path, payload)
    return path


def _phase_request(args: argparse.Namespace) -> dict[str, object]:
    if args.phase_request is None:
        return {}
    if args.phase == "evidence-resume":
        return load_evidence_resume_request(args.phase_request)
    if args.phase == "evidence-recovery":
        return load_evidence_recovery_request(args.phase_request)
    if args.phase == "stability-evidence-recovery":
        return load_stability_evidence_recovery_request(args.phase_request)
    if args.phase == "gallery":
        return load_gallery_phase_request(args.phase_request)
    return load_phase_request(args.phase_request, phase=args.phase)


def _verify_candidate_dry_run_request(
    request: Mapping[str, object],
    *,
    splits: pd.DataFrame,
) -> dict[str, object]:
    if not request:
        return {"status": "not_requested"}
    variant_path = _request_path(request, "variant_index_path")
    variant = load_joined_candidate_variant_index(variant_path, splits)
    development = development_image_rows(variant)
    expected_ids = set(splits.loc[splits["partition"].eq("development"), "id"].astype(int))
    observed_ids = set(development["id"].astype(int))
    if observed_ids != expected_ids:
        raise ValueError("phase request variant index does not match canonical development IDs")
    return {
        "status": "passed",
        "variant_index_path": _relative_artifact_path(variant_path),
        "development_rows": len(development),
        "teacher_sha256_joined_from_splits": True,
        "structural_fields_joined_from_splits": True,
        "sealed_rows_rejected_before_pixels": True,
        "opened_pixels": 0,
    }


def _request_path(request: Mapping[str, object], key: str, *, required: bool = True) -> Path | None:
    value = request.get(key)
    if value in (None, ""):
        if required:
            raise ValueError(f"phase request requires {key}")
        return None
    return Path(str(value))


def _request_mapping_paths(request: Mapping[str, object], key: str) -> dict[str, Path]:
    raw = request.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"phase request requires {key}")
    return {str(source): Path(str(path)) for source, path in raw.items()}


def _hyperparameters_from_request(
    config: ExperimentConfig,
    request: Mapping[str, object],
) -> TrainingHyperparameters:
    values = {
        name: getattr(config.hyperparameters, name)
        for name in (
            "seed",
            "product_batch_size",
            "images_per_product",
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
            "minimum_learning_rate",
            "gradient_clip_norm",
            "amp_initial_scale",
            "amp_growth_interval",
            "planned_epochs",
            "checkpoint_epochs",
        )
    }
    overrides = request.get("hyperparameters")
    if isinstance(overrides, Mapping):
        values.update(overrides)
    if not isinstance(values["checkpoint_epochs"], tuple):
        values["checkpoint_epochs"] = tuple(values["checkpoint_epochs"])  # type: ignore[arg-type]
    return TrainingHyperparameters(**values)


def _config_with_request_overrides(
    config: ExperimentConfig,
    request: Mapping[str, object],
) -> ExperimentConfig:
    return ExperimentConfig(
        candidate=config.candidate,
        hyperparameters=_hyperparameters_from_request(config, request),
        objective=config.objective,
        source_policy=config.source_policy,
        augmentation_policy=config.augmentation_policy,
        run_kind=config.run_kind,
        parent_run_id=config.parent_run_id,
    )


def _load_or_fit_statistics(
    *,
    request: Mapping[str, object],
    source: str,
    cache: Any,
    development_rows: pd.DataFrame,
    splits: pd.DataFrame,
    session: TrainingSessionConfig,
    default_root: Path,
    statistics_paths: dict[str, Path],
) -> tuple[dict[str, object], Path]:
    if source in statistics_paths:
        path = statistics_paths[source]
        return _read_json(path), path
    path = default_root / f"{source}-fold{session.validation_fold}-statistics.json"
    statistics = fit_cached_fold_rgb_statistics(
        cache,
        development_rows,
        validation_fold=session.validation_fold,
        canonical_splits=splits,
    )
    write_json_atomic(path, statistics)
    return statistics, path


def _candidate_data_context(
    *,
    request: Mapping[str, object],
    session: TrainingSessionConfig,
    splits: pd.DataFrame,
    phase_output: Path,
) -> dict[str, object]:
    variant_index = load_joined_candidate_variant_index(
        _request_path(request, "variant_index_path"),
        splits,
    )
    development_rows = development_image_rows(variant_index)
    cache_root = _request_path(request, "cache_root")
    statistics_paths = (
        _request_mapping_paths(request, "statistics_paths")
        if "statistics_paths" in request
        else {}
    )
    contract = PreprocessingContract(width=240, height=320)
    caches = {
        "teacher": ensure_development_image_cache(
            development_rows,
            path_column="teacher_path",
            source="teacher",
            contract=contract,
            cache_root=cache_root,
            sha_column="sha256",
        ),
        "v1": ensure_development_image_cache(
            development_rows,
            path_column="external_path",
            source="v1",
            contract=contract,
            cache_root=cache_root,
            sha_column="external_sha256",
        ),
    }
    statistics: dict[str, dict[str, object]] = {}
    resolved_statistics_paths: dict[str, Path] = {}
    stats_root = phase_output.parent / "statistics" / session.run_id
    for source in ("teacher", "v1"):
        statistics[source], resolved_statistics_paths[source] = _load_or_fit_statistics(
            request=request,
            source=source,
            cache=caches[source],
            development_rows=development_rows,
            splits=splits,
            session=session,
            default_root=stats_root,
            statistics_paths=statistics_paths,
        )
    pairs = build_training_pairs(
        variant_index,
        validation_fold=session.validation_fold,
        canonical_splits=splits,
    )
    dataset = CrossSourcePairDataset(
        pairs,
        teacher_cache=caches["teacher"],
        v1_cache=caches["v1"],
        teacher_statistics=statistics["teacher"],
        v1_statistics=statistics["v1"],
        validation_fold=session.validation_fold,
        split_fingerprint=session.split_fingerprint,
        geometry_policy=DEFAULT_GEOMETRY_POLICY
        if session.augmentation_policy is AugmentationPolicy.GEOMETRY
        else None,
    )
    if session.candidate.candidate == "R4":
        loader = DataLoader(
            dataset,
            batch_sampler=FamilyBatchSampler(pairs, seed=session.hyperparameters.seed),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=session.hyperparameters.product_batch_size,
            shuffle=True,
            drop_last=True,
            generator=make_data_generator(session.hyperparameters.seed),
            worker_init_fn=make_worker_init_fn(session.hyperparameters.seed),
            num_workers=0,
        )
    query_rows = development_rows.loc[
        pd.to_numeric(development_rows["cv_fold"], errors="coerce").eq(session.validation_fold)
    ].copy()
    return {
        "loader": loader,
        "caches": caches,
        "statistics": statistics,
        "statistics_paths": resolved_statistics_paths,
        "query_rows": query_rows,
        "path_columns": {"teacher": "teacher_path", "v1": "external_path"},
    }


def _model_state_sha256(
    model: torch.nn.Module,
    *,
    session: TrainingSessionConfig,
    epoch: int,
) -> str:
    buffer = io.BytesIO()
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "run_id": session.run_id,
            "run_kind": session.run_kind,
            "epoch": epoch,
            "config_hash": session.config_hash,
            "split_fingerprint": session.split_fingerprint,
            "model_state_dict": state,
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _log_progress(stage: str, **fields: object) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"task4-progress stage={stage} {details}".strip(), flush=True)


def _evaluate_protocol_a_milestone(
    model: torch.nn.Module,
    *,
    epoch: int,
    session: TrainingSessionConfig,
    splits: pd.DataFrame,
    caches: Mapping[str, object],
    statistics: Mapping[str, Mapping[str, object]],
    checkpoint_root: Path,
    device: torch.device,
) -> pd.DataFrame:
    checkpoint = CheckpointRecord(
        epoch=epoch,
        path=checkpoint_root / session.run_id / f"epoch-{epoch:03d}.pt",
        sha256=_model_state_sha256(model, session=session, epoch=epoch),
        config_hash=session.config_hash,
        score=0.0,
        split_fingerprint=session.split_fingerprint,
        weight_origin=session.model_metadata.weight_origin,
        parent_run_id=session.parent_run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
    )
    indexes = {
        source: encode_development_cache(
            caches[source],  # type: ignore[arg-type]
            model=model,
            statistics=statistics[source],
            session=session,
            checkpoint=checkpoint,
            batch_size=session.hyperparameters.product_batch_size,
            device=device,
        )
        for source in ("teacher", "v1")
    }
    return evaluate_learned_quality(
        splits,
        indexes,
        fold=session.validation_fold,
    ).summary


def _run_training_and_evidence(
    *,
    registry: RunRegistry,
    request: Mapping[str, object],
    session: TrainingSessionConfig,
    splits: pd.DataFrame,
    phase_output: Path,
    gpu: int,
) -> dict[str, object]:
    _log_progress("candidate-data", run_id=session.run_id, event="start")
    context = _candidate_data_context(
        request=request,
        session=session,
        splits=splits,
        phase_output=phase_output,
    )
    loader = context["loader"]
    checkpoint_root = Path(
        request.get("checkpoint_root") or phase_output.parent / "checkpoints"
    )
    evidence_root = Path(request.get("evidence_root") or ROOT / "results/evidence/task4")
    feature_cache_root = Path(
        request.get("feature_cache_root") or ROOT / "results/cache/task4/features"
    )
    selected_gallery_policy = str(request.get("selected_gallery_policy") or "v1")
    evidence_device: dict[str, torch.device] = {}

    def train(
        device: torch.device,
        model: torch.nn.Module,
        bound_session: TrainingSessionConfig,
        batches: Any,
    ) -> Any:
        _log_progress("training", run_id=bound_session.run_id, event="start", device=device)
        evidence_device["device"] = device
        optimizer = build_optimizer(model, bound_session.hyperparameters)
        scheduler = WarmupCosineScheduler(
            optimizer,
            steps_per_epoch=len(batches),
            config=bound_session.hyperparameters,
        )
        scaler = make_grad_scaler(
            device,
            initial_scale=bound_session.hyperparameters.amp_initial_scale,
            growth_interval=bound_session.hyperparameters.amp_growth_interval,
        )

        def score_callback(scored_model: torch.nn.Module, epoch: int) -> float:
            scorer = make_milestone_scorer(
                lambda model_to_score: _evaluate_protocol_a_milestone(
                    model_to_score,
                    epoch=epoch,
                    session=bound_session,
                    splits=splits,
                    caches=context["caches"],
                    statistics=context["statistics"],
                    checkpoint_root=checkpoint_root,
                    device=device,
                ),
                fold=bound_session.validation_fold,
            )
            return float(scorer(scored_model, epoch))

        def checkpoint_callback(epoch: int, score: float) -> Any:
            return save_checkpoint(
                checkpoint_root / bound_session.run_id / f"epoch-{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                session=bound_session,
                score=score,
            )

        training_result = train_epochs(
            model=model,
            batches=batches,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            session=bound_session,
            score_callback=score_callback,
            checkpoint_callback=checkpoint_callback,
        )
        _log_progress("training", run_id=bound_session.run_id, event="complete")
        return training_result

    result = run_training_attempt(
        registry=registry,
        row=_base_registry_row(session),
        session=session,
        batches=loader,
        device_factory=lambda: pin_single_gpu(gpu),
        train=train,
    )
    try:
        selected_device = evidence_device.get("device", torch.device("cpu"))
        _log_progress("evidence", run_id=session.run_id, event="start", device=selected_device)
        if session.run_kind == "stability":
            evidence = build_stability_evidence(
                registry,
                result=result,
                session=session,
                splits=splits,
                caches=context["caches"],
                statistics=context["statistics"],
                statistics_paths=context["statistics_paths"],
                feature_cache_root=feature_cache_root,
                evidence_root=evidence_root,
                completed_at=_utc_z(),
                device=selected_device,
            )
        else:
            evidence = build_learned_evidence(
                registry,
                result=result,
                session=session,
                splits=splits,
                caches=context["caches"],
                statistics=context["statistics"],
                statistics_paths=context["statistics_paths"],
                query_rows=context["query_rows"],
                path_columns=context["path_columns"],
                feature_cache_root=feature_cache_root,
                evidence_root=evidence_root,
                selected_gallery_policy=selected_gallery_policy,  # type: ignore[arg-type]
                completed_at=None,
                device=selected_device,
            )
        _log_progress("evidence", run_id=session.run_id, event="complete")
    except BaseException as error:
        record_evidence_failure(registry, session.run_id, error, completed_at=_utc_z())
        raise
    config_path = write_experiment_config_artifact(
        Path(evidence.manifest_path).with_name("experiment_config.json"),
        session=session,
        checkpoint_sha256=result.best_checkpoint.sha256,
    )
    return {
        "run_id": session.run_id,
        "manifest": _relative_artifact_path(evidence.manifest_path),
        "config_artifact": _relative_artifact_path(config_path),
        "checkpoint_sha256": result.best_checkpoint.sha256,
    }


def _session_from_registry_artifact(
    artifact: ExperimentConfigArtifact,
    *,
    run_id: str | None = None,
    run_kind: str | None = None,
    fold: int | None = None,
) -> TrainingSessionConfig:
    return TrainingSessionConfig(
        run_id=run_id or artifact.identity.run_id,
        run_kind=run_kind or artifact.run_kind,
        candidate=artifact.candidate,
        hyperparameters=artifact.hyperparameters,
        objective=artifact.objective,
        source_policy=artifact.source_policy,
        augmentation_policy=artifact.augmentation_policy,
        validation_fold=artifact.identity.fold if fold is None else fold,
        split_fingerprint=artifact.identity.split_fingerprint,
        parent_run_id=artifact.parent_run_id,
    )


def _checkpoint_record_from_loaded(
    loaded: object,
    *,
    path: Path,
    session: TrainingSessionConfig,
) -> CheckpointRecord:
    return CheckpointRecord(
        epoch=int(loaded.epoch),  # type: ignore[attr-defined]
        path=path,
        sha256=str(loaded.sha256),  # type: ignore[attr-defined]
        config_hash=session.config_hash,
        score=float(loaded.score),  # type: ignore[attr-defined]
        split_fingerprint=session.split_fingerprint,
        weight_origin=session.model_metadata.weight_origin,
        parent_run_id=session.parent_run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
    )


def _registry_checkpoint_path(
    request_path: Path,
    *,
    registry_row: Mapping[str, object],
    run_id: str,
) -> Path:
    if registry_row.get("run_id") != run_id:
        raise ValueError("registry checkpoint path belongs to another run")
    registry_value = str(registry_row.get("checkpoint_path") or "")
    if not registry_value.strip():
        raise ValueError("registry checkpoint path is missing")
    registry_path = Path(registry_value)
    root = ROOT.resolve()

    def resolved_inside_repo(path: Path) -> Path:
        resolved = (path if path.is_absolute() else ROOT / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("checkpoint path must stay inside the repository") from error
        return resolved

    if resolved_inside_repo(request_path) != resolved_inside_repo(registry_path):
        raise ValueError("request and registry checkpoint paths do not match")
    return registry_path


def _task6_input_from_request(
    spec: Mapping[str, object],
    *,
    artifacts: Mapping[str, ExperimentConfigArtifact],
    splits: pd.DataFrame,
    registry_row: Mapping[str, object] | None = None,
) -> Task6ManifestInput:
    run_id = str(spec.get("run_id") or "")
    artifact = artifacts.get(run_id)
    if artifact is None:
        artifact_path = _request_path(spec, "config_artifact_path")
        artifact = _load_experiment_config_artifact(artifact_path)
    session = _session_from_registry_artifact(artifact)
    checkpoint_path = _request_path(spec, "checkpoint_path")
    loaded = load_checkpoint(
        checkpoint_path,
        expected_sha256=artifact.identity.checkpoint_sha256,
        expected_config_hash=session.config_hash,
        expected_split_fingerprint=session.split_fingerprint,
        expected_weight_origin=session.model_metadata.weight_origin,
        expected_parent_run_id=session.parent_run_id,
        expected_run_id=session.run_id,
        expected_run_kind=session.run_kind,
        map_location="cpu",
    )
    checkpoint = _checkpoint_record_from_loaded(
        loaded,
        path=(
            _registry_checkpoint_path(
                checkpoint_path,
                registry_row=registry_row,
                run_id=session.run_id,
            )
            if registry_row is not None
            else checkpoint_path
        ),
        session=session,
    )
    return Task6ManifestInput(
        manifest_path=_request_path(spec, "manifest_path"),
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )


def _stability_input_from_request(
    spec: Mapping[str, object],
    *,
    artifacts: Mapping[str, ExperimentConfigArtifact],
    splits: pd.DataFrame,
    registry_row: Mapping[str, object] | None = None,
) -> StabilityEvidenceInput:
    run_id = str(spec.get("run_id") or "")
    artifact = artifacts.get(run_id)
    if artifact is None:
        artifact_path = _request_path(spec, "config_artifact_path")
        artifact = _load_experiment_config_artifact(artifact_path)
    session = _session_from_registry_artifact(artifact)
    checkpoint_path = _request_path(spec, "checkpoint_path")
    loaded = load_checkpoint(
        checkpoint_path,
        expected_sha256=artifact.identity.checkpoint_sha256,
        expected_config_hash=session.config_hash,
        expected_split_fingerprint=session.split_fingerprint,
        expected_weight_origin=session.model_metadata.weight_origin,
        expected_parent_run_id=session.parent_run_id,
        expected_run_id=session.run_id,
        expected_run_kind="stability",
        map_location="cpu",
    )
    checkpoint = _checkpoint_record_from_loaded(
        loaded,
        path=(
            _registry_checkpoint_path(
                checkpoint_path,
                registry_row=registry_row,
                run_id=session.run_id,
            )
            if registry_row is not None
            else checkpoint_path
        ),
        session=session,
    )
    return StabilityEvidenceInput(
        artifact_path=_request_path(spec, "manifest_path"),
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )


def _development_summary(
    specs: list[Mapping[str, object]],
    *,
    artifacts: Mapping[str, ExperimentConfigArtifact],
    splits: pd.DataFrame,
) -> dict[str, object]:
    validated = [
        _task6_input_from_request(spec, artifacts=artifacts, splits=splits).validated()
        for spec in specs
    ]
    return {
        "status": "validated",
        "manifest_count": len(validated),
        "run_ids": sorted(evidence.identity.run_id for evidence in validated),
        "mean_development_winner_score": sum(
            evidence.development_winner_score for evidence in validated
        )
        / len(validated),
    }


def _candidate_phase_config(
    rows: list[dict[str, str]],
    *,
    artifacts: dict[str, ExperimentConfigArtifact],
    candidate: str,
) -> ExperimentConfig:
    if candidate in {"R1", "R2", "R5", "B1"}:
        return _candidate_config(candidate)
    if candidate == "R3":
        return derive_r3_config(rows, config_artifacts=artifacts)
    if candidate == "R4":
        return derive_r4_config(rows, config_artifacts=artifacts)
    matrix = build_experiment_matrix(rows, config_artifacts=artifacts)
    try:
        return matrix[candidate]  # type: ignore[index]
    except KeyError as error:
        raise ValueError("candidate phase requires a resolved --candidate") from error


def _candidate_phase_payload(
    args: argparse.Namespace,
    *,
    registry: RunRegistry,
    rows: list[dict[str, str]],
    artifacts: dict[str, ExperimentConfigArtifact],
    split_fingerprint: str,
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    requested_candidate = args.candidate or str(request.get("candidate") or "")
    if not requested_candidate:
        raise ValueError("candidate phase requires a resolved --candidate")
    config = _config_with_request_overrides(
        _candidate_phase_config(rows, artifacts=artifacts, candidate=requested_candidate),
        request,
    )
    run_id = (
        args.run_id
        or str(request.get("run_id") or "")
        or f"task4-candidate-{requested_candidate.lower()}-{uuid.uuid4().hex[:12]}"
    )
    if any(row.get("run_id") == run_id for row in rows):
        raise ValueError("retry attempts require a new run ID")
    session = config.training_session(run_id=run_id, split_fingerprint=split_fingerprint)
    if args.dry_run:
        return {
            "status": "ready",
            "candidate": requested_candidate,
            "run_id": run_id,
            "config_hash": session.config_hash,
            "safety": _verify_candidate_dry_run_request(request, splits=splits),
        }
    return _run_training_and_evidence(
        registry=registry,
        request=request,
        session=session,
        splits=splits,
        phase_output=args.phase_output,
        gpu=args.gpu,
    )


def _stability_phase_payload(
    args: argparse.Namespace,
    *,
    registry: RunRegistry,
    rows: list[dict[str, str]],
    artifacts: dict[str, ExperimentConfigArtifact],
    split_fingerprint: str,
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    finalists = select_stability_finalists(rows, config_artifacts=artifacts)
    attempt_token = args.run_id or request.get("attempt_token") or uuid.uuid4().hex
    attempt_token = str(attempt_token)
    first_method = getattr(finalists[0], "method", str(finalists[0]))
    method = args.candidate or request.get("candidate") or first_method
    finalist = next(
        (
            candidate
            for candidate in finalists
            if getattr(candidate, "method", str(candidate)) == method
        ),
        None,
    )
    if finalist is None:
        raise ValueError("stability phase requires one of the two finalist candidates")
    plan = build_stability_plan(finalist, attempt_token=attempt_token)
    requested_fold = request.get("fold")
    if args.dry_run or request.get("mode") == "plan":
        selected = None
        if requested_fold is not None:
            selected = next((run for run in plan if run.fold == int(requested_fold)), None)
            if selected is None:
                raise ValueError("stability fold must be in range(5)")
            if selected.run_id != request.get("run_id"):
                raise ValueError("stability request run_id does not match its fold plan")
            safety = _verify_candidate_dry_run_request(request, splits=splits)
            return {
                "status": "ready",
                "mode": "stability_run",
                "finalists": [
                    getattr(candidate, "candidate_run_id", str(candidate))
                    for candidate in finalists
                ],
                "selected_finalist": getattr(finalist, "candidate_run_id", str(finalist)),
                "selected_run": selected.run_id,
                "fold": selected.fold,
                "parent_run_id": selected.parent_run_id,
                "attempt_token": selected.attempt_token,
                "evidence_scope": "lightweight_primary_score_and_coverage",
                "full_evidence_artifacts": "forbidden",
                "safety": safety,
                "opened_pixels": 0,
            }
        return {
            "finalists": [
                getattr(candidate, "candidate_run_id", str(candidate)) for candidate in finalists
            ],
            "selected_finalist": getattr(finalist, "candidate_run_id", str(finalist)),
            "planned_runs": [getattr(run, "run_id", str(run)) for run in plan],
        }
    if requested_fold is None:
        if "evidence_manifests" not in request:
            raise ValueError("stability summary requires exact five evidence manifests")
        evidence_specs = request["evidence_manifests"]
        if not isinstance(evidence_specs, list) or len(evidence_specs) != 5:
            raise ValueError("stability summary requires exact five evidence manifests")
        evidence_inputs = {
            str(spec["run_id"]): _task6_input_from_request(spec, artifacts=artifacts, splits=splits)
            for spec in evidence_specs
            if isinstance(spec, Mapping)
        }
        summary = summarize_stability(
            rows,
            finalist=finalist,
            config_artifacts=artifacts,
            evidence_manifests=evidence_inputs,
        )
        return {
            "finalist": summary.finalist.candidate_run_id,
            "run_ids": list(summary.run_ids),
            "mean": summary.mean,
            "standard_deviation": summary.standard_deviation,
        }
    selected = next((run for run in plan if run.fold == int(requested_fold)), None)
    if selected is None:
        raise ValueError("stability fold must be in range(5)")
    if selected.run_id != request.get("run_id"):
        raise ValueError("stability request run_id does not match its fold plan")
    session = selected.training_session()
    if session.split_fingerprint != split_fingerprint:
        raise ValueError("stability session split fingerprint does not match canonical splits")
    return _run_training_and_evidence(
        registry=registry,
        request=request,
        session=session,
        splits=splits,
        phase_output=args.phase_output,
        gpu=args.gpu,
    )


def _deployment_inputs_from_request(
    request: Mapping[str, object],
    *,
    artifacts: Mapping[str, ExperimentConfigArtifact],
    splits: pd.DataFrame,
) -> tuple[DeploymentDecisionInput, DeploymentDecisionInput]:
    specs = request.get("deployment_inputs")
    if not isinstance(specs, list) or len(specs) != 2:
        raise ValueError("gallery phase requires two deployment decision inputs")
    inputs = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise ValueError("deployment decision input is malformed")
        registry_row = spec.get("registry_row")
        if not isinstance(registry_row, Mapping):
            raise ValueError("deployment decision registry row is malformed")
        finalist_artifact = _load_experiment_config_artifact(
            _request_path(spec, "finalist_config_artifact_path")
        )
        finalist = StabilityFinalist(
            finalist_artifact.identity,
            float(spec.get("candidate_score", 0.0)),
            finalist_artifact,
        )
        stability_specs = spec.get("stability_manifests")
        if not isinstance(stability_specs, list) or len(stability_specs) != 5:
            raise ValueError("deployment input requires five stability manifests")
        stability_rows = spec.get("stability_rows")
        if not isinstance(stability_rows, list) or len(stability_rows) != 5:
            raise ValueError("deployment input requires five stability registry rows")
        stability_rows_by_run = {
            str(row.get("run_id") or ""): row
            for row in stability_rows
            if isinstance(row, Mapping)
        }
        stability_inputs = {
            str(item["run_id"]): _stability_input_from_request(
                item,
                artifacts=artifacts,
                splits=splits,
                registry_row=stability_rows_by_run.get(str(item["run_id"])),
            )
            for item in stability_specs
            if isinstance(item, Mapping)
        }
        inputs.append(
            DeploymentDecisionInput(
                registry_row=registry_row,
                finalist=finalist,
                stability_rows=tuple(stability_rows),
                stability_config_artifacts=artifacts,
                stability_evidence_manifests=stability_inputs,
                candidate_evidence_manifest=_task6_input_from_request(
                    spec["candidate_manifest"],  # type: ignore[arg-type]
                    artifacts=artifacts,
                    splits=splits,
                    registry_row=registry_row,
                ),
            )
        )
    return inputs[0], inputs[1]


def _gallery_phase_payload(
    args: argparse.Namespace,
    *,
    artifacts: Mapping[str, ExperimentConfigArtifact],
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    winner = select_deployment_candidate(
        _deployment_inputs_from_request(request, artifacts=artifacts, splits=splits)
    )
    selected_spec = request.get("selected_fold1_manifest")
    if not isinstance(selected_spec, Mapping):
        raise ValueError("gallery phase requires selected_fold1_manifest")
    selected_run_id = str(selected_spec.get("run_id") or "")
    if selected_run_id != winner.identity.run_id:
        raise ValueError("gallery selected fold-1 manifest does not match deployment winner")
    if args.dry_run:
        return {
            "status": "ready",
            "deployment_winner": winner.identity.run_id,
            "selected_fold1_manifest": selected_run_id,
            "gallery_timing": "forbidden",
            "opened_pixels": 0,
        }
    selected_registry_row = next(
        (
            deployment_spec.get("registry_row")
            for deployment_spec in request.get("deployment_inputs", [])
            if isinstance(deployment_spec, Mapping)
            and isinstance(deployment_spec.get("registry_row"), Mapping)
            and deployment_spec["registry_row"].get("run_id") == selected_run_id
        ),
        None,
    )
    if not isinstance(selected_registry_row, Mapping):
        raise ValueError("gallery selected registry row is missing")
    selected = select_gallery_source(
        _task6_input_from_request(
            selected_spec,
            artifacts=artifacts,
            splits=splits,
            registry_row=selected_registry_row,
        ),
        policy_timing_output_path=Path(args.phase_output).with_name("gallery_policy_timing.json"),
    )
    return {
        "deployment_winner": winner.identity.run_id,
        "selected_gallery": selected.policy,
        "selected_gallery_p95_seconds": selected.p95_end_to_end_seconds,
        "selected_gallery_index_bytes": selected.index_bytes,
    }


def _evidence_phase_payload(
    args: argparse.Namespace,
    *,
    artifacts: Mapping[str, ExperimentConfigArtifact],
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    specs = request.get("manifests")
    if not isinstance(specs, list) or not specs:
        raise ValueError("evidence phase requires manifest specs")
    return _development_summary(specs, artifacts=artifacts, splits=splits)


def _running_registry_row(
    rows: Sequence[Mapping[str, object]],
    run_id: str,
) -> Mapping[str, object]:
    matching = [row for row in rows if row.get("run_id") == run_id]
    if len(matching) != 1:
        raise ValueError("evidence resume requires exactly one existing registry row")
    row = matching[0]
    if row.get("status") != "running":
        raise ValueError("evidence resume requires an existing running registry row")
    return row


def _recoverable_failed_registry_row(
    rows: Sequence[Mapping[str, object]],
    run_id: str,
) -> Mapping[str, object]:
    matching = [row for row in rows if row.get("run_id") == run_id]
    if len(matching) != 1:
        raise ValueError("evidence recovery requires exactly one existing registry row")
    row = matching[0]
    if (
        row.get("status") != "failed"
        or row.get("run_id") != R5_RECOVERABLE_RUN_ID
        or row.get("error_type") != R5_RECOVERABLE_ERROR_TYPE
        or row.get("error_message") != R5_RECOVERABLE_ERROR_MESSAGE
    ):
        raise ValueError("evidence recovery requires the exact failed R5 artifact-write row")
    return row


def _recoverable_failed_stability_registry_row(
    rows: Sequence[Mapping[str, object]],
    run_id: str,
) -> Mapping[str, object]:
    """Accept only a stability row that failed with the exact recoverable evidence error."""

    matching = [row for row in rows if row.get("run_id") == run_id]
    if len(matching) != 1:
        raise ValueError("stability recovery requires exactly one existing registry row")
    row = matching[0]
    if (
        row.get("status") != "failed"
        or row.get("run_kind") != "stability"
        or row.get("error_type") != STABILITY_RECOVERABLE_ERROR_TYPE
        or row.get("error_message") != STABILITY_RECOVERABLE_ERROR_MESSAGE
    ):
        raise ValueError("stability recovery requires the exact failed stability evidence row")
    if str(row.get("evidence_manifest_path") or "").strip():
        raise ValueError("stability recovery requires a row with no completed stability evidence")
    if not str(row.get("completed_at_utc") or "").strip():
        raise ValueError("stability recovery requires a row whose failure was recorded")
    return row


def _write_stability_recovery_audit(
    path: Path,
    *,
    request: Mapping[str, object],
    registry_row: Mapping[str, object],
    result: object,
) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_type": "task4_failed_stability_evidence_attempt_audit",
        "run_id": request["run_id"],
        "candidate": request["candidate"],
        "fold": request["fold"],
        "attempt_token": request["attempt_token"],
        "parent_run_id": request["parent_run_id"],
        "evidence_scope": request["evidence_scope"],
        "recoverable_error": request["recoverable_error"],
        "failed_registry_row": dict(registry_row),
        "checkpoint_epochs": [record.epoch for record in result.checkpoints],
        "best_checkpoint_sha256": result.best_checkpoint.sha256,
        "checkpoint_paths": [
            _repo_relative_path_value(record.path) for record in result.checkpoints
        ],
    }
    destination = Path(path)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if destination.is_file():
        if json.loads(destination.read_text(encoding="utf-8")) != normalized:
            raise ValueError("failed stability evidence attempt audit is immutable")
        return destination
    write_json_atomic(destination, payload)
    return destination


def _evidence_resume_session(
    request: Mapping[str, object],
    *,
    split_fingerprint: str,
) -> TrainingSessionConfig:
    candidate = str(request.get("candidate") or "")
    run_id = str(request.get("run_id") or "")
    if candidate not in {"R1", "R2", "R5", "B1"}:
        raise ValueError("evidence resume currently requires a base candidate request")
    return _candidate_config(candidate).training_session(
        run_id=run_id,
        split_fingerprint=split_fingerprint,
    )


def _checkpoint_paths_from_resume_request(request: Mapping[str, object]) -> list[Path]:
    raw_paths = request.get("checkpoint_paths")
    if not isinstance(raw_paths, list) or len(raw_paths) != 5:
        raise ValueError("evidence resume requires exact five checkpoint paths")
    paths = [Path(path) if Path(path).is_absolute() else ROOT / Path(path) for path in raw_paths]
    if len(set(paths)) != 5:
        raise ValueError("evidence resume checkpoint paths must be unique")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"evidence resume checkpoint is missing: {missing[0]}")
    return paths


def _write_evidence_recovery_audit(
    path: Path,
    *,
    request: Mapping[str, object],
    registry_row: Mapping[str, object],
    result: object,
) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_type": "task4_failed_evidence_attempt_audit",
        "run_id": request["run_id"],
        "candidate": request["candidate"],
        "recoverable_error": request["recoverable_error"],
        "failed_registry_row": dict(registry_row),
        "checkpoint_epochs": [record.epoch for record in result.checkpoints],
        "best_checkpoint_sha256": result.best_checkpoint.sha256,
        "checkpoint_paths": [
            _repo_relative_path_value(record.path) for record in result.checkpoints
        ],
    }
    write_json_atomic(path, payload)
    return path


def _evidence_resume_phase_payload(
    args: argparse.Namespace,
    *,
    registry: RunRegistry,
    split_fingerprint: str,
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    session = _evidence_resume_session(request, split_fingerprint=split_fingerprint)
    rows = registry.read()
    row = _running_registry_row(rows, session.run_id)
    checkpoint_paths = _checkpoint_paths_from_resume_request(request)
    result = reconstruct_training_result(
        checkpoint_paths,
        session=session,
        registry_row=row,
    )
    if args.dry_run:
        safety = _verify_candidate_dry_run_request(request, splits=splits)
        return {
            "status": "ready",
            "mode": "evidence_resume",
            "run_id": session.run_id,
            "candidate": session.candidate.candidate,
            "config_hash": session.config_hash,
            "checkpoint_epochs": [record.epoch for record in result.checkpoints],
            "best_checkpoint_sha256": result.best_checkpoint.sha256,
            "safety": safety,
            "opened_pixels": 0,
            "requires_gpu": True,
            "training_call": "forbidden",
        }
    context = _candidate_data_context(
        request=request,
        session=session,
        splits=splits,
        phase_output=args.phase_output,
    )
    device = pin_single_gpu(args.gpu)
    evidence_root = Path(request.get("evidence_root") or ROOT / "results/evidence/task4")
    feature_cache_root = Path(
        request.get("feature_cache_root") or ROOT / "results/cache/task4/features"
    )
    selected_gallery_policy = str(request.get("selected_gallery_policy") or "v1")
    evidence = build_learned_evidence(
        registry,
        result=result,
        session=session,
        splits=splits,
        caches=context["caches"],
        statistics=context["statistics"],
        statistics_paths=context["statistics_paths"],
        query_rows=context["query_rows"],
        path_columns=context["path_columns"],
        feature_cache_root=feature_cache_root,
        evidence_root=evidence_root,
        selected_gallery_policy=selected_gallery_policy,  # type: ignore[arg-type]
        completed_at=None,
        device=device,
    )
    config_path = write_experiment_config_artifact(
        Path(evidence.manifest_path).with_name("experiment_config.json"),
        session=session,
        checkpoint_sha256=result.best_checkpoint.sha256,
    )
    return {
        "run_id": session.run_id,
        "mode": "evidence_resume",
        "manifest": _relative_artifact_path(evidence.manifest_path),
        "config_artifact": _relative_artifact_path(config_path),
        "checkpoint_sha256": result.best_checkpoint.sha256,
    }


def _evidence_recovery_phase_payload(
    args: argparse.Namespace,
    *,
    registry: RunRegistry,
    split_fingerprint: str,
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    session = _evidence_resume_session(request, split_fingerprint=split_fingerprint)
    rows = registry.read()
    row = _recoverable_failed_registry_row(rows, session.run_id)
    checkpoint_paths = _checkpoint_paths_from_resume_request(request)
    result = reconstruct_training_result(
        checkpoint_paths,
        session=session,
        registry_row=row,
        allow_failed_evidence=True,
    )
    audit_path = Path(args.phase_output).with_name("task9-r5-recovery-failure-attempt.json")
    if args.dry_run:
        safety = _verify_candidate_dry_run_request(request, splits=splits)
        return {
            "status": "ready",
            "mode": "evidence_recovery",
            "run_id": session.run_id,
            "candidate": session.candidate.candidate,
            "config_hash": session.config_hash,
            "checkpoint_epochs": [record.epoch for record in result.checkpoints],
            "best_checkpoint_sha256": result.best_checkpoint.sha256,
            "failed_row_status": row["status"],
            "recoverable_error": request["recoverable_error"],
            "planned_failure_audit": _relative_artifact_path(audit_path),
            "safety": safety,
            "opened_pixels": 0,
            "requires_gpu": True,
            "training_call": "forbidden",
        }
    audit_path = _write_evidence_recovery_audit(
        audit_path,
        request=request,
        registry_row=row,
        result=result,
    )
    context = _candidate_data_context(
        request=request,
        session=session,
        splits=splits,
        phase_output=args.phase_output,
    )
    device = pin_single_gpu(args.gpu)
    evidence_root = Path(request.get("evidence_root") or ROOT / "results/evidence/task4")
    feature_cache_root = Path(
        request.get("feature_cache_root") or ROOT / "results/cache/task4/features"
    )
    selected_gallery_policy = str(request.get("selected_gallery_policy") or "v1")
    evidence = build_learned_evidence(
        registry,
        result=result,
        session=session,
        splits=splits,
        caches=context["caches"],
        statistics=context["statistics"],
        statistics_paths=context["statistics_paths"],
        query_rows=context["query_rows"],
        path_columns=context["path_columns"],
        feature_cache_root=feature_cache_root,
        evidence_root=evidence_root,
        selected_gallery_policy=selected_gallery_policy,  # type: ignore[arg-type]
        completed_at=None,
        device=device,
        recover_failed_evidence=True,
        recovery_error_type=R5_RECOVERABLE_ERROR_TYPE,
        recovery_error_message=R5_RECOVERABLE_ERROR_MESSAGE,
    )
    config_path = write_experiment_config_artifact(
        Path(evidence.manifest_path).with_name("experiment_config.json"),
        session=session,
        checkpoint_sha256=result.best_checkpoint.sha256,
    )
    return {
        "run_id": session.run_id,
        "mode": "evidence_recovery",
        "manifest": _relative_artifact_path(evidence.manifest_path),
        "config_artifact": _relative_artifact_path(config_path),
        "failure_audit": _relative_artifact_path(audit_path),
        "checkpoint_sha256": result.best_checkpoint.sha256,
    }


def _stability_recovery_plan(
    request: Mapping[str, object],
    *,
    rows: Sequence[Mapping[str, object]],
    artifacts: Mapping[str, ExperimentConfigArtifact],
) -> StabilityRunPlan:
    """Bind the requested run ID to the authoritative finalist/fold stability plan."""

    method = str(request["candidate"])
    finalists = select_stability_finalists(rows, config_artifacts=artifacts)
    finalist = next(
        (
            candidate
            for candidate in finalists
            if getattr(candidate, "method", str(candidate)) == method
        ),
        None,
    )
    if finalist is None:
        raise ValueError("stability recovery requires one of the two finalist candidates")
    plan = build_stability_plan(finalist, attempt_token=str(request["attempt_token"]))
    selected = next((run for run in plan if run.fold == int(request["fold"])), None)
    if selected is None:
        raise ValueError("stability fold must be in range(5)")
    if selected.run_id != request["run_id"]:
        raise ValueError("stability recovery request run_id does not match its fold plan")
    if selected.parent_run_id != request["parent_run_id"]:
        raise ValueError("stability recovery request parent lineage does not match its plan")
    return selected


def _stability_evidence_recovery_phase_payload(
    args: argparse.Namespace,
    *,
    registry: RunRegistry,
    rows: list[dict[str, str]],
    artifacts: dict[str, ExperimentConfigArtifact],
    split_fingerprint: str,
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    plan = _stability_recovery_plan(request, rows=rows, artifacts=artifacts)
    session = plan.training_session()
    if session.split_fingerprint != split_fingerprint:
        raise ValueError("stability session split fingerprint does not match canonical splits")
    if session.run_kind != "stability":
        raise ValueError("stability recovery requires a stability session")
    row = _recoverable_failed_stability_registry_row(registry.read(), session.run_id)
    checkpoint_paths = _checkpoint_paths_from_resume_request(request)
    result = reconstruct_training_result(
        checkpoint_paths,
        session=session,
        registry_row=row,
        allow_failed_evidence=True,
        recoverable_error_type=STABILITY_RECOVERABLE_ERROR_TYPE,
        recoverable_error_message=STABILITY_RECOVERABLE_ERROR_MESSAGE,
    )
    audit_path = Path(args.phase_output).with_name(
        f"{session.run_id}-stability-recovery-failure-attempt.json"
    )
    if args.dry_run:
        safety = _verify_candidate_dry_run_request(request, splits=splits)
        return {
            "status": "ready",
            "mode": "stability_evidence_recovery",
            "run_id": session.run_id,
            "candidate": session.candidate.candidate,
            "fold": session.validation_fold,
            "attempt_token": plan.attempt_token,
            "parent_run_id": plan.parent_run_id,
            "config_hash": session.config_hash,
            "checkpoint_epochs": [record.epoch for record in result.checkpoints],
            "best_checkpoint_sha256": result.best_checkpoint.sha256,
            "failed_row_status": row["status"],
            "recoverable_error": request["recoverable_error"],
            "evidence_scope": STABILITY_EVIDENCE_SCOPE,
            "planned_failure_audit": _relative_artifact_path(audit_path),
            "safety": safety,
            "opened_pixels": 0,
            "requires_gpu": True,
            "training_call": "forbidden",
            "full_evidence_artifacts": "forbidden",
        }
    audit_path = _write_stability_recovery_audit(
        audit_path,
        request=request,
        registry_row=row,
        result=result,
    )
    context = _candidate_data_context(
        request=request,
        session=session,
        splits=splits,
        phase_output=args.phase_output,
    )
    device = pin_single_gpu(args.gpu)
    evidence_root = Path(request.get("evidence_root") or ROOT / "results/evidence/task4")
    feature_cache_root = Path(
        request.get("feature_cache_root") or ROOT / "results/cache/task4/features"
    )
    evidence = build_stability_evidence(
        registry,
        result=result,
        session=session,
        splits=splits,
        caches=context["caches"],
        statistics=context["statistics"],
        statistics_paths=context["statistics_paths"],
        feature_cache_root=feature_cache_root,
        evidence_root=evidence_root,
        completed_at=_utc_z(),
        device=device,
        recover_failed_evidence=True,
        recovery_error_type=STABILITY_RECOVERABLE_ERROR_TYPE,
        recovery_error_message=STABILITY_RECOVERABLE_ERROR_MESSAGE,
    )
    config_path = write_experiment_config_artifact(
        Path(evidence.manifest_path).with_name("experiment_config.json"),
        session=session,
        checkpoint_sha256=result.best_checkpoint.sha256,
    )
    return {
        "run_id": session.run_id,
        "mode": "stability_evidence_recovery",
        "fold": session.validation_fold,
        "manifest": _relative_artifact_path(evidence.manifest_path),
        "config_artifact": _relative_artifact_path(config_path),
        "failure_audit": _relative_artifact_path(audit_path),
        "checkpoint_sha256": result.best_checkpoint.sha256,
        "development_winner_score": evidence.development_winner_score,
        "total_query_count": evidence.total_query_count,
        "scorable_query_count": evidence.scorable_query_count,
        "primary_coverage": evidence.primary_coverage,
    }


def _phase_specific_payload(
    args: argparse.Namespace,
    *,
    registry: RunRegistry,
    rows: list[dict[str, str]],
    artifacts: dict[str, ExperimentConfigArtifact],
    split_fingerprint: str,
    splits: pd.DataFrame,
    request: Mapping[str, object],
) -> dict[str, object]:
    if args.phase == "candidate":
        return _candidate_phase_payload(
            args,
            registry=registry,
            rows=rows,
            artifacts=artifacts,
            split_fingerprint=split_fingerprint,
            splits=splits,
            request=request,
        )
    if args.phase == "stability":
        return _stability_phase_payload(
            args,
            registry=registry,
            rows=rows,
            artifacts=artifacts,
            split_fingerprint=split_fingerprint,
            splits=splits,
            request=request,
        )
    if args.phase == "gallery":
        return _gallery_phase_payload(args, artifacts=artifacts, splits=splits, request=request)
    if args.phase == "evidence":
        return _evidence_phase_payload(args, artifacts=artifacts, splits=splits, request=request)
    if args.phase == "evidence-resume":
        return _evidence_resume_phase_payload(
            args,
            registry=registry,
            split_fingerprint=split_fingerprint,
            splits=splits,
            request=request,
        )
    if args.phase == "evidence-recovery":
        return _evidence_recovery_phase_payload(
            args,
            registry=registry,
            split_fingerprint=split_fingerprint,
            splits=splits,
            request=request,
        )
    if args.phase == "stability-evidence-recovery":
        return _stability_evidence_recovery_phase_payload(
            args,
            registry=registry,
            rows=rows,
            artifacts=artifacts,
            split_fingerprint=split_fingerprint,
            splits=splits,
            request=request,
        )
    raise ValueError("unknown downstream phase")


def run_downstream_phase(
    args: argparse.Namespace,
    *,
    registry: RunRegistry,
    split_fingerprint: str,
) -> int:
    gate = validate_budget_gate(
        args.budget_gate,
        split_fingerprint=split_fingerprint,
        require_passed=not args.dry_run,
    )
    if args.phase_output is None:
        raise ValueError("non-smoke phases require --phase-output for restartable orchestration")
    request = _phase_request(args)
    rows = _dry_run_registry_rows(registry)
    artifacts = load_config_artifacts(ROOT / "results/evidence/task4")
    try:
        matrix = build_experiment_matrix(rows, config_artifacts=artifacts)
        matrix_status = "resolved"
    except ValueError as error:
        matrix = {}
        matrix_status = f"pending: {error}"
    package = {
        "schema_version": 1,
        "phase": args.phase,
        "restartable": True,
        "dry_run": bool(args.dry_run),
        "run_id": args.run_id or str(request.get("run_id") or ""),
        "candidate": args.candidate or str(request.get("candidate") or ""),
        "manifest": str(args.manifest or ""),
        "gate": gate,
        "phase_request": _relative_artifact_path(args.phase_request) if args.phase_request else "",
        "config_artifact_run_ids": sorted(artifacts),
        "matrix_status": matrix_status,
        "resolved_candidates": sorted(matrix) if matrix else [],
    }
    if args.dry_run and gate["decision"] != "passed":
        blocked_result: dict[str, object] = {
            "status": "blocked",
            "reason": (
                "corrected budget gate exceeds "
                f"{gate['budget_gpu_hours']:.2f} GPU-hours"
            ),
        }
        if args.phase == "candidate" and request:
            try:
                blocked_result["safety"] = _verify_candidate_dry_run_request(
                    request,
                    splits=load_canonical_splits(),
                )
            except ValueError as error:
                blocked_result["safety"] = {
                    "status": "failed",
                    "reason": str(error),
                    "opened_pixels": 0,
                }
        package["phase_result"] = blocked_result
        write_json_atomic(args.phase_output, package)
        print(f"{args.phase}: restartable package {args.phase_output}")
        return 0
    try:
        package["phase_result"] = _phase_specific_payload(
            args,
            registry=registry,
            rows=rows,
            artifacts=artifacts,
            split_fingerprint=split_fingerprint,
            splits=load_canonical_splits(),
            request=request,
        )
    except ValueError as error:
        if not args.dry_run:
            raise
        package["phase_result"] = {"status": "pending", "reason": str(error)}
    phase_result = package.get("phase_result")
    if isinstance(phase_result, Mapping) and isinstance(phase_result.get("manifest"), str):
        package["manifest"] = phase_result["manifest"]
    if (
        args.dry_run
        and isinstance(phase_result, Mapping)
        and phase_result.get("status") == "ready"
        and str(package.get("matrix_status", "")).startswith("pending:")
    ):
        package["matrix_status"] = (
            f"requested candidate ready; full matrix {package['matrix_status']}"
        )
    if not args.dry_run:
        refreshed_rows = _dry_run_registry_rows(registry)
        refreshed_artifacts = load_config_artifacts(ROOT / "results/evidence/task4")
        try:
            refreshed_matrix = build_experiment_matrix(
                refreshed_rows,
                config_artifacts=refreshed_artifacts,
            )
            package["matrix_status"] = "resolved"
            package["resolved_candidates"] = sorted(refreshed_matrix)
        except ValueError as error:
            package["matrix_status"] = f"pending: {error}"
            package["resolved_candidates"] = []
        package["config_artifact_run_ids"] = sorted(refreshed_artifacts)
    write_json_atomic(args.phase_output, package)
    print(f"{args.phase}: restartable package {args.phase_output}")
    return 0


def _phase_output_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _wait_for_gallery_worker(
    process: subprocess.Popen[bytes],
    *,
    phase_output: Path,
    initial_signature: tuple[int, int, int] | None,
    exit_grace_seconds: float,
    poll_seconds: float = 0.25,
) -> int:
    """Require a gallery worker to exit shortly after publishing its result."""

    published_at: float | None = None
    while True:
        returncode = process.poll()
        if returncode is not None:
            return int(returncode)
        if _phase_output_signature(phase_output) != initial_signature:
            published_at = published_at or time.monotonic()
            if time.monotonic() - published_at >= exit_grace_seconds:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                raise RuntimeError(
                    "gallery worker wrote output but did not exit within "
                    f"{exit_grace_seconds:g} seconds"
                )
        time.sleep(poll_seconds)


def _run_gallery_worker_bounded(
    argv: Sequence[str],
    *,
    phase_output: Path,
    exit_grace_seconds: float,
) -> int:
    initial_signature = _phase_output_signature(phase_output)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *argv,
        "--gallery-worker",
    ]
    process = subprocess.Popen(command, cwd=ROOT)
    return _wait_for_gallery_worker(
        process,
        phase_output=phase_output,
        initial_signature=initial_signature,
        exit_grace_seconds=exit_grace_seconds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Task 4 learned comparison phases.")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--registry", type=Path, default=ROOT / "results/runs.csv")
    parser.add_argument("--split-path", type=Path, default=SPLITS_CSV)
    parser.add_argument("--family", choices=tuple(SMOKE_FAMILIES), action="append")
    parser.add_argument("--child-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--budget-estimate", action="store_true")
    parser.add_argument("--budget-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--budget-family",
        choices=tuple(THROUGHPUT_CONFIGS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--budget-output",
        type=Path,
        default=ROOT / "results/evidence/task4/model_comparison_budget.json",
    )
    parser.add_argument("--budget-gate-combine", action="store_true")
    parser.add_argument("--budget-source", type=Path, action="append", default=[])
    parser.add_argument(
        "--budget-gate",
        type=Path,
        default=ROOT / "results/evidence/task4/model_comparison_budget_gate.json",
    )
    parser.add_argument("--phase-output", type=Path)
    parser.add_argument("--phase-request", type=Path)
    parser.add_argument("--gallery-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--gallery-exit-grace-seconds",
        type=float,
        default=60.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--candidate", choices=("R1", "R2", "R3", "R4", "R5", "B1"))
    parser.add_argument("--run-id")
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(command_arguments)
    if (
        args.phase == "gallery"
        and not args.dry_run
        and not args.gallery_worker
    ):
        if args.phase_output is None:
            raise ValueError("gallery phase requires --phase-output")
        if args.gallery_exit_grace_seconds <= 0:
            raise ValueError("gallery exit grace must be positive")
        return _run_gallery_worker_bounded(
            command_arguments,
            phase_output=args.phase_output,
            exit_grace_seconds=args.gallery_exit_grace_seconds,
        )
    if args.budget_gate_combine:
        if len(args.budget_source) != 2:
            raise ValueError("budget gate combine requires exactly two --budget-source values")
        artifact = write_budget_gate_artifact(
            source_paths=(args.budget_source[0], args.budget_source[1]),
            output_path=args.budget_gate,
        )
        print(
            "budget gate: "
            f"{artifact['selected_estimated_full_matrix_gpu_hours']:.2f}/"
            f"{artifact['budget_gpu_hours']:.2f} GPU-hours; "
            f"fits={artifact['fits_budget']}"
        )
        return 0
    if args.split_path.resolve() != SPLITS_CSV.resolve() and not args.dry_run:
        raise ValueError("Task 4 learned comparisons must use data/processed/splits.csv")
    splits = load_canonical_splits()
    split_fingerprint = cv_assignment_digest(splits)
    development_image_rows(splits)
    registry = RunRegistry(args.registry, project_root=ROOT)
    if args.phase != "smoke":
        return run_downstream_phase(args, registry=registry, split_fingerprint=split_fingerprint)
    if args.dry_run:
        print_dry_run(
            phase=args.phase,
            registry=registry,
            split_fingerprint=split_fingerprint,
        )
        return 0
    if args.phase == "smoke":
        if args.budget_family:
            measure_budget_family(
                family=args.budget_family,
                split_fingerprint=split_fingerprint,
                splits=splits,
                gpu=args.gpu,
                output_path=args.budget_output,
            )
            return 0
        if args.budget_only:
            return run_budget_estimate_in_children(
                split_fingerprint=split_fingerprint,
                gpu=args.gpu,
                output_path=args.budget_output,
            )
        families = args.family or list(SMOKE_FAMILIES)
        if len(families) > 1 and not args.child_smoke:
            failures = 0
            for family in families:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--phase",
                    "smoke",
                    "--family",
                    family,
                    "--child-smoke",
                    "--gpu",
                    "0",
                    "--registry",
                    str(args.registry),
                    "--split-path",
                    str(args.split_path),
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=isolated_child_env(requested_physical_gpu(args.gpu)),
                )
                failures += int(completed.returncode != 0)
            if failures:
                return 1
            if args.budget_estimate:
                return run_budget_estimate_in_children(
                    split_fingerprint=split_fingerprint,
                    gpu=args.gpu,
                    output_path=args.budget_output,
                )
            return 0
        for family in families:
            run_id = run_registered_smoke_attempt(
                registry=registry,
                config=SMOKE_FAMILIES[family],
                split_fingerprint=split_fingerprint,
                gpu=args.gpu,
                output_root=ROOT / "results/evidence/task4/smoke",
            )
            print(f"{family}: {run_id}")
        if args.budget_estimate:
            return run_budget_estimate_in_children(
                split_fingerprint=split_fingerprint,
                gpu=args.gpu,
                output_path=args.budget_output,
            )
        return 0
    raise NotImplementedError(
        f"{args.phase} phase entry point is reserved for Tasks 9-10 after the smoke gate"
    )


if __name__ == "__main__":
    raise SystemExit(main())
