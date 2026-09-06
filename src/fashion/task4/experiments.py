"""Immutable Task 4 experiment matrix and development-only selection rules."""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import pandas as pd

from fashion.task4.benchmark import TimingPolicy
from fashion.task4.models import SCRATCH_WEIGHT_ORIGIN
from fashion.task4.training import (
    AugmentationPolicy,
    CandidateConfig,
    CheckpointRecord,
    SourcePolicy,
    TrainingHyperparameters,
    TrainingSessionConfig,
)

CandidateName: TypeAlias = Literal["R1", "R2", "R3", "R4", "R5", "B1"]
ScratchCandidateName: TypeAlias = Literal["R1", "R2", "R3", "R4", "R5"]
GalleryPolicy: TypeAlias = Literal["teacher", "v1", "two_view"]

CANDIDATE_ORDER: tuple[CandidateName, ...] = ("R1", "R2", "R3", "R4", "R5", "B1")
SCRATCH_CANDIDATE_ORDER: tuple[ScratchCandidateName, ...] = ("R1", "R2", "R3", "R4", "R5")
EXPERIMENT_FACTOR_FIELDS = (
    "architecture",
    "objective",
    "augmentation_policy",
    "weight_origin",
)
STABILITY_FOLDS = (0, 1, 2, 3, 4)
GALLERY_QUALITY_TOLERANCE = 1e-5
INDEX_LIMIT_BYTES = 1024**3
CPU_MEASUREMENT_ROUTE = "spawned_batch_one_checkpoint_extraction_v1"
REQUIRED_DIRECTIONS = (
    ("teacher", "teacher"),
    ("teacher", "v1"),
    ("v1", "teacher"),
    ("v1", "v1"),
)
_VALIDATED_TASK6_TOKEN = object()
_STABILITY_SUMMARY_TOKEN = object()
_GALLERY_PRACTICAL_TOKEN = object()
_POST_STABILITY_DECISION_TOKEN = object()
_FINAL_GALLERY_DECISION_TOKEN = object()

__all__ = (
    "B1_CONFIG",
    "CANDIDATE_ORDER",
    "CPU_MEASUREMENT_ROUTE",
    "EXPERIMENT_FACTOR_FIELDS",
    "GALLERY_QUALITY_TOLERANCE",
    "INDEX_LIMIT_BYTES",
    "R1_CONFIG",
    "R2_CONFIG",
    "R5_CONFIG",
    "SCRATCH_CANDIDATE_ORDER",
    "STABILITY_FOLDS",
    "DeploymentEvidence",
    "PostStabilityDeploymentDecision",
    "FinalGalleryDecision",
    "EvidenceIdentity",
    "ExperimentConfig",
    "ExperimentConfigArtifact",
    "GalleryPracticalEvidence",
    "GalleryOption",
    "StabilityFinalist",
    "StabilityEvidenceInput",
    "StabilityRunPlan",
    "StabilitySummary",
    "Task6ManifestInput",
    "ValidatedStabilityEvidence",
    "ValidatedTask6Evidence",
    "DeploymentDecisionInput",
    "build_experiment_matrix",
    "build_stability_plan",
    "deployment_evidence_from_artifacts",
    "derive_r3_config",
    "derive_r4_config",
    "development_winner_score",
    "gallery_options_from_evidence",
    "link_gallery_result_to_final_decision",
    "derive_gallery_practical_evidence",
    "passes_deployment_gates",
    "pooled_spread",
    "query_normalization_source",
    "select_deployment_candidate",
    "select_gallery_source",
    "validate_final_gallery_decision_artifact",
    "validate_post_stability_deployment_artifact",
    "write_final_gallery_decision_artifact",
    "write_post_stability_deployment_artifact",
    "select_parent",
    "select_stability_finalists",
    "summarize_stability",
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One immutable experiment recipe composed from the training contracts."""

    candidate: CandidateConfig
    hyperparameters: TrainingHyperparameters
    objective: str
    source_policy: SourcePolicy
    augmentation_policy: AugmentationPolicy
    run_kind: str
    parent_run_id: str | None = None

    @property
    def method(self) -> CandidateName:
        return self.candidate.candidate

    @property
    def pretrained(self) -> bool:
        return self.candidate.pretrained

    @property
    def comparison_only(self) -> bool:
        return self.pretrained

    @property
    def weight_origin(self) -> str:
        return self.candidate.weight_origin

    @property
    def factor_signature(self) -> tuple[object, ...]:
        """Return only controlled experiment factors, excluding lineage metadata."""

        return (
            self.candidate.architecture,
            self.objective,
            self.augmentation_policy,
            self.weight_origin,
        )

    def training_session(
        self,
        *,
        run_id: str,
        split_fingerprint: str,
        validation_fold: int = 1,
    ) -> TrainingSessionConfig:
        """Bind this recipe to the existing validated training/session identity."""

        return TrainingSessionConfig(
            run_id=run_id,
            run_kind=self.run_kind,
            candidate=self.candidate,
            hyperparameters=self.hyperparameters,
            objective=self.objective,
            source_policy=self.source_policy,
            augmentation_policy=self.augmentation_policy,
            validation_fold=validation_fold,
            split_fingerprint=split_fingerprint,
            parent_run_id=self.parent_run_id,
        )


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    """Exact run, recipe, split, fold, and selected-checkpoint identity."""

    run_id: str
    method: CandidateName
    fold: int
    config_hash: str
    split_fingerprint: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.method not in CANDIDATE_ORDER:
            raise ValueError("evidence run/method identity is invalid")
        if self.fold not in STABILITY_FOLDS:
            raise ValueError("evidence fold identity must be in range(5)")
        for label, value in (
            ("config hash", self.config_hash),
            ("split fingerprint", self.split_fingerprint),
            ("checkpoint SHA-256", self.checkpoint_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"evidence {label} is malformed")

    @classmethod
    def from_registry_row(cls, row: Mapping[str, Any]) -> EvidenceIdentity:
        return cls(
            run_id=str(_field(row, "run_id")),
            method=str(_field(row, "method")),  # type: ignore[arg-type]
            fold=_integer(_field(row, "fold"), field="fold"),
            config_hash=str(_field(row, "config_hash")),
            split_fingerprint=str(_field(row, "split_fingerprint")),
            checkpoint_sha256=str(_field(row, "checkpoint_sha256")),
        )


@dataclass(frozen=True, slots=True)
class Task6ManifestInput:
    """Canonical inputs required to revalidate one Task 6 evidence manifest."""

    manifest_path: Path
    session: TrainingSessionConfig
    checkpoint: CheckpointRecord
    canonical_splits: pd.DataFrame

    def validated(self) -> ValidatedTask6Evidence:
        return ValidatedTask6Evidence.from_manifest(
            self.manifest_path,
            session=self.session,
            checkpoint=self.checkpoint,
            canonical_splits=self.canonical_splits,
        )


@dataclass(frozen=True, slots=True)
class ValidatedStabilityEvidence:
    """Selection-safe view of one lightweight stability evidence artifact."""

    identity: EvidenceIdentity
    development_winner_score: float
    total_query_count: int
    scorable_query_count: int
    primary_coverage: float

    def __post_init__(self) -> None:
        _finite_float(self.development_winner_score, field="stability score")
        coverage = _finite_float(self.primary_coverage, field="stability coverage")
        total = _integer(self.total_query_count, field="stability total query count")
        scorable = _integer(self.scorable_query_count, field="stability scorable count")
        if total <= 0 or scorable <= 0 or scorable > total:
            raise ValueError("stability evidence scorable query count is malformed")
        if coverage != scorable / total:
            raise ValueError("stability evidence coverage does not match its counts")


@dataclass(frozen=True, slots=True)
class StabilityEvidenceInput:
    """Canonical inputs required to revalidate one lightweight stability artifact."""

    artifact_path: Path
    session: TrainingSessionConfig
    checkpoint: CheckpointRecord
    canonical_splits: pd.DataFrame

    def validated(self) -> ValidatedStabilityEvidence:
        from fashion.task4.learned_evidence import validate_stability_evidence_artifact

        payload = validate_stability_evidence_artifact(
            self.artifact_path,
            session=self.session,
            checkpoint=self.checkpoint,
            canonical_splits=self.canonical_splits,
        )
        identity = payload["identity"]
        selected = payload["selected_metrics"]
        coverage = payload["coverage"]
        if (
            not isinstance(identity, Mapping)
            or not isinstance(selected, Mapping)
            or not isinstance(coverage, Mapping)
        ):
            raise ValueError("stability evidence payload is malformed")
        return ValidatedStabilityEvidence(
            identity=EvidenceIdentity(
                run_id=str(identity["run_id"]),
                method=str(identity["method"]),  # type: ignore[arg-type]
                fold=_integer(identity["fold"], field="fold"),
                config_hash=str(identity["config_hash"]),
                split_fingerprint=str(identity["split_fingerprint"]),
                checkpoint_sha256=str(identity["checkpoint_sha256"]),
            ),
            development_winner_score=float(selected["development_winner_score"]),
            total_query_count=_integer(
                coverage["total_query_count"],
                field="stability total query count",
            ),
            scorable_query_count=_integer(
                coverage["scorable_query_count"],
                field="stability scorable count",
            ),
            primary_coverage=float(coverage["primary_coverage"]),
        )


@dataclass(frozen=True, slots=True)
class ExperimentConfigArtifact:
    """Persisted immutable training recipe tied to one completed checkpoint."""

    identity: EvidenceIdentity
    candidate: CandidateConfig
    hyperparameters: TrainingHyperparameters
    objective: str
    source_policy: SourcePolicy
    augmentation_policy: AugmentationPolicy
    run_kind: str
    parent_run_id: str | None
    canonical_config_json: str
    candidate_config_hash: str

    @classmethod
    def from_session(
        cls,
        session: TrainingSessionConfig,
        *,
        checkpoint_sha256: str,
    ) -> ExperimentConfigArtifact:
        identity = EvidenceIdentity(
            run_id=session.run_id,
            method=session.candidate.candidate,
            fold=session.validation_fold,
            config_hash=session.config_hash,
            split_fingerprint=session.split_fingerprint,
            checkpoint_sha256=checkpoint_sha256,
        )
        return cls(
            identity=identity,
            candidate=session.candidate,
            hyperparameters=session.hyperparameters,
            objective=session.objective,
            source_policy=session.source_policy,
            augmentation_policy=session.augmentation_policy,
            run_kind=session.run_kind,
            parent_run_id=session.parent_run_id,
            canonical_config_json=session.config_json,
            candidate_config_hash=session.config_hash,
        )

    @property
    def config(self) -> ExperimentConfig:
        return ExperimentConfig(
            candidate=self.candidate,
            hyperparameters=self.hyperparameters,
            objective=self.objective,
            source_policy=self.source_policy,
            augmentation_policy=self.augmentation_policy,
            run_kind=self.run_kind,
            parent_run_id=self.parent_run_id,
        )

    def validate(self) -> None:
        session = TrainingSessionConfig(
            run_id=self.identity.run_id,
            run_kind=self.run_kind,
            candidate=self.candidate,
            hyperparameters=self.hyperparameters,
            objective=self.objective,
            source_policy=self.source_policy,
            augmentation_policy=self.augmentation_policy,
            validation_fold=self.identity.fold,
            split_fingerprint=self.identity.split_fingerprint,
            parent_run_id=self.parent_run_id,
        )
        if (
            session.config_json != self.canonical_config_json
            or session.config_hash != self.candidate_config_hash
            or self.identity.config_hash != self.candidate_config_hash
        ):
            raise ValueError("persisted experiment config identity does not match")


_SHARED_HYPERPARAMETERS = TrainingHyperparameters()
_SHARED_SOURCE_POLICY = SourcePolicy.TEACHER_V1_PAIRS


def _fixed_config(
    candidate: CandidateName,
    architecture: Literal["resnet18", "resnet34"],
    *,
    objective: str,
    augmentation_policy: AugmentationPolicy = AugmentationPolicy.NONE,
    run_kind: str = "candidate",
) -> ExperimentConfig:
    return ExperimentConfig(
        candidate=CandidateConfig(candidate, architecture),
        hyperparameters=_SHARED_HYPERPARAMETERS,
        objective=objective,
        source_policy=_SHARED_SOURCE_POLICY,
        augmentation_policy=augmentation_policy,
        run_kind=run_kind,
    )


R1_CONFIG = _fixed_config("R1", "resnet18", objective="vicreg")
R2_CONFIG = _fixed_config("R2", "resnet34", objective="vicreg")
R5_CONFIG = _fixed_config("R5", "resnet18", objective="content_mask_mse")
B1_CONFIG = _fixed_config("B1", "resnet18", objective="vicreg", run_kind="benchmark")


def development_winner_score(summary: pd.DataFrame) -> float:
    """Use the existing evidence selector for equal teacher/V1 same-source nDCG@10."""

    from fashion.task4.learned_evidence import summarize_learned_scores

    return summarize_learned_scores(summary)["development_winner_score"]


def _field(row: Mapping[str, Any], name: str) -> Any:
    try:
        return row[name]
    except KeyError as error:
        raise ValueError(f"registry row is missing {name}") from error


def _false(value: object) -> bool:
    return value is False or value == "false"


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if str(parsed) != str(value):
        raise ValueError(f"{field} must be an integer")
    return parsed


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _is_completed_scratch_candidate(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") == "completed"
        and row.get("run_kind") == "candidate"
        and _integer(row.get("fold"), field="fold") == 1
        and _false(row.get("pretrained"))
        and row.get("deployment_eligibility") == "eligible"
        and row.get("weight_origin") == SCRATCH_WEIGHT_ORIGIN
        and row.get("method") in SCRATCH_CANDIDATE_ORDER
    )


def _method_rank(method: object) -> int:
    try:
        return CANDIDATE_ORDER.index(method)  # type: ignore[arg-type]
    except ValueError as error:
        raise ValueError(f"unknown experiment method: {method}") from error


def _best_run_per_method(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[ScratchCandidateName],
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for method in candidates:
        runs = [
            row
            for row in rows
            if row.get("method") == method and _is_completed_scratch_candidate(row)
        ]
        if not runs:
            raise ValueError(f"completed eligible scratch candidate {method} is required")
        best = min(
            runs,
            key=lambda row: (
                -_finite_float(
                    _field(row, "development_winner_score"),
                    field="development_winner_score",
                ),
                str(_field(row, "run_id")),
            ),
        )
        _validate_candidate_chain(best, rows, config_artifacts, active=frozenset())
        selected.append(best)
    return selected


def _artifact_for(
    row: Mapping[str, Any],
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
) -> ExperimentConfigArtifact:
    run_id = str(_field(row, "run_id"))
    artifact = config_artifacts.get(run_id)
    if not isinstance(artifact, ExperimentConfigArtifact):
        raise ValueError(f"persisted config artifact is required for run {run_id}")
    artifact.validate()
    if artifact.identity != EvidenceIdentity.from_registry_row(row):
        raise ValueError("registry and persisted config artifact identity do not match")
    if (str(row.get("parent_run_id")) if row.get("parent_run_id") else None) != (
        artifact.parent_run_id
    ):
        raise ValueError("registry and persisted config parent lineage do not match")
    return artifact


def _raw_best_runs(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[ScratchCandidateName],
) -> list[Mapping[str, Any]]:
    selected = []
    for method in candidates:
        runs = [
            row
            for row in rows
            if row.get("method") == method and _is_completed_scratch_candidate(row)
        ]
        if not runs:
            raise ValueError(f"completed eligible scratch candidate {method} is required")
        selected.append(
            min(
                runs,
                key=lambda row: (
                    -_finite_float(
                        _field(row, "development_winner_score"),
                        field="development_winner_score",
                    ),
                    str(_field(row, "run_id")),
                ),
            )
        )
    return selected


def _expected_child_config(
    method: ScratchCandidateName,
    parent: Mapping[str, Any] | None,
    parent_artifact: ExperimentConfigArtifact | None,
) -> ExperimentConfig:
    if method == "R1":
        return R1_CONFIG
    if method == "R2":
        return R2_CONFIG
    if method == "R5":
        return R5_CONFIG
    if parent is None or parent_artifact is None:
        raise ValueError(f"{method} requires a validated parent")
    if method == "R3":
        return ExperimentConfig(
            candidate=CandidateConfig("R3", parent_artifact.candidate.architecture),
            hyperparameters=parent_artifact.hyperparameters,
            objective=parent_artifact.objective,
            source_policy=parent_artifact.source_policy,
            augmentation_policy=AugmentationPolicy.GEOMETRY,
            run_kind="candidate",
            parent_run_id=str(_field(parent, "run_id")),
        )
    return ExperimentConfig(
        candidate=CandidateConfig("R4", parent_artifact.candidate.architecture),
        hyperparameters=parent_artifact.hyperparameters,
        objective="vicreg_triplet",
        source_policy=parent_artifact.source_policy,
        augmentation_policy=parent_artifact.augmentation_policy,
        run_kind="candidate",
        parent_run_id=str(_field(parent, "run_id")),
    )


def _validate_candidate_chain(
    row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
    *,
    active: frozenset[str],
) -> ExperimentConfigArtifact:
    artifact = _artifact_for(row, config_artifacts)
    run_id = artifact.identity.run_id
    if run_id in active:
        raise ValueError("adaptive parent chain contains a cycle")
    method = artifact.identity.method
    if method not in SCRATCH_CANDIDATE_ORDER:
        raise ValueError("adaptive chain requires a scratch candidate")
    parent: Mapping[str, Any] | None = None
    parent_artifact: ExperimentConfigArtifact | None = None
    required_parents: tuple[ScratchCandidateName, ...] = ()
    if method == "R3":
        required_parents = ("R1", "R2")
    elif method == "R4":
        required_parents = ("R1", "R2", "R3")
    if required_parents:
        parent_runs = _raw_best_runs(rows, required_parents)
        for candidate in parent_runs:
            _validate_candidate_chain(
                candidate,
                rows,
                config_artifacts,
                active=active | {run_id},
            )
        parent = min(
            parent_runs,
            key=lambda candidate: (
                -_finite_float(
                    _field(candidate, "development_winner_score"),
                    field="development_winner_score",
                ),
                required_parents.index(_field(candidate, "method")),
                str(_field(candidate, "run_id")),
            ),
        )
        parent_artifact = _artifact_for(parent, config_artifacts)
    expected = _expected_child_config(method, parent, parent_artifact)
    expected_session = expected.training_session(
        run_id=run_id,
        split_fingerprint=artifact.identity.split_fingerprint,
        validation_fold=artifact.identity.fold,
    )
    if (
        artifact.parent_run_id != expected.parent_run_id
        or artifact.candidate_config_hash != expected_session.config_hash
        or artifact.canonical_config_json != expected_session.config_json
    ):
        raise ValueError(f"{method} parent chain or one-factor child config does not match")
    return artifact


def select_parent(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
    candidates: Sequence[ScratchCandidateName],
) -> Mapping[str, Any]:
    """Select a completed scratch parent by exact score then candidate order."""

    candidate_tuple = tuple(candidates)
    if not candidate_tuple or len(set(candidate_tuple)) != len(candidate_tuple):
        raise ValueError("parent candidates must be a non-empty unique sequence")
    if any(candidate not in SCRATCH_CANDIDATE_ORDER for candidate in candidate_tuple):
        raise ValueError("parent candidates must be scratch methods R1-R5")
    runs = _best_run_per_method(rows, candidate_tuple, config_artifacts)
    return min(
        runs,
        key=lambda row: (
            -_finite_float(
                _field(row, "development_winner_score"),
                field="development_winner_score",
            ),
            candidate_tuple.index(_field(row, "method")),
            str(_field(row, "run_id")),
        ),
    )


def derive_r3_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
) -> ExperimentConfig:
    """Copy the best R1/R2 architecture and add only seeded geometry."""

    parent = select_parent(
        rows,
        config_artifacts=config_artifacts,
        candidates=("R1", "R2"),
    )
    artifact = _artifact_for(parent, config_artifacts)
    return _expected_child_config("R3", parent, artifact)


def derive_r4_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
) -> ExperimentConfig:
    """Copy the best R1-R3 architecture/geometry and add the family triplet."""

    parent = select_parent(
        rows,
        config_artifacts=config_artifacts,
        candidates=("R1", "R2", "R3"),
    )
    artifact = _artifact_for(parent, config_artifacts)
    return _expected_child_config("R4", parent, artifact)


def build_experiment_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
) -> Mapping[CandidateName, ExperimentConfig]:
    """Resolve adaptive parents and return the complete immutable R1-R5/B1 matrix."""

    resolved = {
        "R1": R1_CONFIG,
        "R2": R2_CONFIG,
        "R3": derive_r3_config(rows, config_artifacts=config_artifacts),
        "R4": derive_r4_config(rows, config_artifacts=config_artifacts),
        "R5": R5_CONFIG,
        "B1": B1_CONFIG,
    }
    return MappingProxyType(resolved)


@dataclass(frozen=True, slots=True)
class StabilityFinalist:
    """One selected fold-1 scratch candidate awaiting five fresh runs."""

    identity: EvidenceIdentity
    candidate_score: float
    config_artifact: ExperimentConfigArtifact

    def __post_init__(self) -> None:
        self.config_artifact.validate()
        if self.identity != self.config_artifact.identity:
            raise ValueError("finalist identity does not match persisted config")
        _finite_float(self.candidate_score, field="candidate score")

    @property
    def method(self) -> ScratchCandidateName:
        return self.identity.method  # type: ignore[return-value]

    @property
    def candidate_run_id(self) -> str:
        return self.identity.run_id


@dataclass(frozen=True, slots=True)
class StabilityRunPlan:
    """Identity for one fresh finalist/fold stability run."""

    method: ScratchCandidateName
    candidate_run_id: str
    run_id: str
    fold: int
    parent_run_id: str
    attempt_token: str
    config_artifact: ExperimentConfigArtifact
    run_kind: Literal["stability"] = "stability"

    def __post_init__(self) -> None:
        self.config_artifact.validate()
        if (
            self.method != self.config_artifact.identity.method
            or self.candidate_run_id != self.config_artifact.identity.run_id
            or self.parent_run_id != self.candidate_run_id
            or self.fold not in STABILITY_FOLDS
            or not self.run_id.strip()
            or not self.attempt_token.strip()
        ):
            raise ValueError("stability run plan identity or lineage is malformed")

    def training_session(self) -> TrainingSessionConfig:
        """Bind a selected recipe to this fresh stability identity."""

        return TrainingSessionConfig(
            run_id=self.run_id,
            run_kind=self.run_kind,
            candidate=self.config_artifact.candidate,
            hyperparameters=self.config_artifact.hyperparameters,
            objective=self.config_artifact.objective,
            source_policy=self.config_artifact.source_policy,
            augmentation_policy=self.config_artifact.augmentation_policy,
            validation_fold=self.fold,
            split_fingerprint=self.config_artifact.identity.split_fingerprint,
            parent_run_id=self.parent_run_id,
        )


def select_stability_finalists(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
) -> tuple[StabilityFinalist, StabilityFinalist]:
    """Choose the top two completed fold-1 scratch candidates."""

    available_methods = tuple(
        method
        for method in SCRATCH_CANDIDATE_ORDER
        if any(row.get("method") == method and _is_completed_scratch_candidate(row) for row in rows)
    )
    runs = _best_run_per_method(rows, available_methods, config_artifacts)
    ranked = sorted(
        runs,
        key=lambda row: (
            -_finite_float(
                _field(row, "development_winner_score"),
                field="development_winner_score",
            ),
            _method_rank(_field(row, "method")),
            str(_field(row, "run_id")),
        ),
    )
    if len(ranked) < 2:
        raise ValueError("at least two completed eligible scratch candidates are required")
    finalists = []
    for row in ranked[:2]:
        artifact = _validate_candidate_chain(
            row,
            rows,
            config_artifacts,
            active=frozenset(),
        )
        finalists.append(
            StabilityFinalist(
                identity=artifact.identity,
                candidate_score=_finite_float(
                    _field(row, "development_winner_score"),
                    field="development_winner_score",
                ),
                config_artifact=artifact,
            )
        )
    return finalists[0], finalists[1]


def build_stability_plan(
    finalist: StabilityFinalist,
    *,
    attempt_token: str | None = None,
) -> tuple[StabilityRunPlan, ...]:
    """Create five deterministic fresh run identities for folds zero through four."""

    token = uuid.uuid4().hex if attempt_token is None else attempt_token
    if not isinstance(token, str) or not token.strip():
        raise ValueError("stability attempt token must not be blank")
    return tuple(
        StabilityRunPlan(
            method=finalist.method,
            candidate_run_id=finalist.candidate_run_id,
            run_id=f"{finalist.candidate_run_id}-stability-{token}-fold-{fold}",
            fold=fold,
            parent_run_id=finalist.candidate_run_id,
            attempt_token=token,
            config_artifact=finalist.config_artifact,
        )
        for fold in STABILITY_FOLDS
    )


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    """Five-fold score mean and sample standard deviation for one finalist."""

    finalist: StabilityFinalist
    run_identities: tuple[EvidenceIdentity, ...]
    mean: float
    standard_deviation: float
    run_config_artifacts: tuple[ExperimentConfigArtifact, ...] = ()
    _aggregation_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._aggregation_token is not _STABILITY_SUMMARY_TOKEN:
            raise TypeError("stability summary requires the five-fold aggregation factory")
        folds = tuple(identity.fold for identity in self.run_identities)
        distinct_run_ids = {identity.run_id for identity in self.run_identities}
        if folds != STABILITY_FOLDS or len(distinct_run_ids) != 5:
            raise ValueError("stability summary requires five distinct fold identities")
        if len(self.run_config_artifacts) != 5:
            raise ValueError("stability summary requires five persisted config artifacts")
        for identity, artifact in zip(
            self.run_identities,
            self.run_config_artifacts,
            strict=True,
        ):
            artifact.validate()
            if (
                artifact.identity != identity
                or artifact.parent_run_id != self.finalist.candidate_run_id
            ):
                raise ValueError("stability summary config lineage does not match finalist")
        if any(
            identity.method != self.finalist.method
            or identity.split_fingerprint != self.finalist.identity.split_fingerprint
            for identity in self.run_identities
        ):
            raise ValueError("stability summary run identity does not match finalist")
        _finite_float(self.mean, field="stability mean")
        spread = _finite_float(
            self.standard_deviation,
            field="stability standard deviation",
        )
        if spread < 0:
            raise ValueError("stability standard deviation must be non-negative")

    @property
    def method(self) -> ScratchCandidateName:
        return self.finalist.method

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(identity.run_id for identity in self.run_identities)


def summarize_stability(
    rows: Sequence[Mapping[str, Any]],
    *,
    finalist: StabilityFinalist,
    config_artifacts: Mapping[str, ExperimentConfigArtifact],
    evidence_manifests: Mapping[str, Task6ManifestInput | StabilityEvidenceInput],
) -> StabilitySummary:
    """Aggregate exactly one fresh completed stability run for each fold."""

    selected = [
        row
        for row in rows
        if row.get("method") == finalist.method
        and row.get("run_kind") == "stability"
        and row.get("status") == "completed"
        and _false(row.get("pretrained"))
        and row.get("deployment_eligibility") == "eligible"
        and row.get("weight_origin") == SCRATCH_WEIGHT_ORIGIN
    ]
    by_fold: dict[int, Mapping[str, Any]] = {}
    for row in selected:
        fold = _integer(_field(row, "fold"), field="fold")
        if fold in by_fold:
            raise ValueError(f"stability fold {fold} must have exactly one completed run")
        by_fold[fold] = row
    if tuple(sorted(by_fold)) != STABILITY_FOLDS:
        raise ValueError("stability requires folds 0 through 4 exactly once")
    ordered = tuple(by_fold[fold] for fold in STABILITY_FOLDS)
    identities: list[EvidenceIdentity] = []
    scores: list[float] = []
    finalist_recipe = finalist.config_artifact
    finalist_recipe.validate()
    for row in ordered:
        artifact = _artifact_for(row, config_artifacts)
        if artifact.run_kind != "stability":
            raise ValueError("stability config artifact has the wrong run kind")
        if artifact.parent_run_id != finalist.candidate_run_id:
            raise ValueError("stability config artifact parent lineage does not match finalist")
        if (
            artifact.candidate != finalist_recipe.candidate
            or artifact.hyperparameters != finalist_recipe.hyperparameters
            or artifact.objective != finalist_recipe.objective
            or artifact.source_policy != finalist_recipe.source_policy
            or artifact.augmentation_policy != finalist_recipe.augmentation_policy
            or artifact.identity.split_fingerprint != finalist.identity.split_fingerprint
        ):
            raise ValueError("stability config does not match the finalist persisted config")
        manifest_input = evidence_manifests.get(artifact.identity.run_id)
        if not isinstance(manifest_input, (Task6ManifestInput, StabilityEvidenceInput)):
            raise ValueError("validated stability evidence manifest is required for every fold")
        evidence = manifest_input.validated()
        if evidence.identity != artifact.identity:
            raise ValueError("stability evidence identity does not match its config artifact")
        identities.append(artifact.identity)
        scores.append(evidence.development_winner_score)
    return StabilitySummary(
        finalist=finalist,
        run_identities=tuple(identities),
        mean=float(statistics.mean(scores)),
        standard_deviation=float(statistics.stdev(scores)),
        run_config_artifacts=tuple(config_artifacts[identity.run_id] for identity in identities),
        _aggregation_token=_STABILITY_SUMMARY_TOKEN,
    )


def pooled_spread(standard_deviation_a: Real, standard_deviation_b: Real) -> float:
    """Return ``sqrt((sd_a² + sd_b²) / 2)``."""

    first = _finite_float(standard_deviation_a, field="standard deviation")
    second = _finite_float(standard_deviation_b, field="standard deviation")
    if first < 0 or second < 0:
        raise ValueError("standard deviations must be non-negative")
    return math.sqrt((first**2 + second**2) / 2.0)


@dataclass(frozen=True, slots=True)
class CPUMeasurementIdentity:
    """Exact validated CPU measurement route and host/thread identity."""

    route: str
    measurement_sha256: str
    cpu: str
    operating_system: str
    thread_count: int

    def __post_init__(self) -> None:
        if self.route != CPU_MEASUREMENT_ROUTE:
            raise ValueError("CPU measurement route does not match")
        if len(self.measurement_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.measurement_sha256
        ):
            raise ValueError("CPU measurement SHA-256 is malformed")
        if not self.cpu.strip() or not self.operating_system.strip() or self.thread_count != 1:
            raise ValueError("CPU measurement hardware/thread identity is malformed")


def _require_uniform_frame_identity(
    frame: pd.DataFrame,
    identity: EvidenceIdentity,
    *,
    label: str,
) -> None:
    expected = {
        "run_id": identity.run_id,
        "method": identity.method,
        "fold": identity.fold,
        "config_hash": identity.config_hash,
        "split_fingerprint": identity.split_fingerprint,
        "checkpoint_sha256": identity.checkpoint_sha256,
    }
    if missing := set(expected).difference(frame.columns):
        raise ValueError(f"{label} is missing identity columns: {sorted(missing)}")
    for column, value in expected.items():
        if frame[column].isna().any() or set(frame[column].astype(str)) != {str(value)}:
            raise ValueError(f"{label} {column} identity does not match")


@dataclass(frozen=True, slots=True)
class ValidatedTask6Evidence:
    """Selection-safe view of one already validated Task 6 manifest package."""

    identity: EvidenceIdentity
    canvas_ndcg_at_10: tuple[tuple[str, float], ...]
    direction_p95_end_to_end_seconds: tuple[tuple[tuple[str, str], float], ...]
    source_index_bytes: tuple[tuple[str, int], ...]
    cpu_measurement: CPUMeasurementIdentity
    development_winner_score: float
    source_robustness_ratio: float
    gallery_quality_at_10: tuple[tuple[str, float], ...] = ()
    _validation_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_token is not _VALIDATED_TASK6_TOKEN:
            raise TypeError("Task 6 selection evidence requires a validated artifact factory")
        canvas_labels = tuple(label for label, _ in self.canvas_ndcg_at_10)
        if len(canvas_labels) != 2 or set(canvas_labels) != {"wide", "tall"}:
            raise ValueError("canvas evidence requires exact wide and tall labels")
        directions = tuple(direction for direction, _ in self.direction_p95_end_to_end_seconds)
        if len(directions) != 4 or set(directions) != set(REQUIRED_DIRECTIONS):
            raise ValueError("cost evidence requires the exact four direction labels")
        sources = tuple(source for source, _ in self.source_index_bytes)
        if len(sources) != 2 or set(sources) != {"teacher", "v1"}:
            raise ValueError("cost evidence requires exact teacher and v1 index labels")
        numeric_values = (
            *(value for _, value in self.canvas_ndcg_at_10),
            *(value for _, value in self.direction_p95_end_to_end_seconds),
            self.development_winner_score,
            self.source_robustness_ratio,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in numeric_values
        ):
            raise ValueError("Task 6 selection metrics must be finite and non-negative")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for _, value in self.source_index_bytes
        ):
            raise ValueError("Task 6 index bytes must be non-negative integers")
        gallery_policies = tuple(policy for policy, _ in self.gallery_quality_at_10)
        if len(gallery_policies) != 3 or set(gallery_policies) != {
            "teacher",
            "v1",
            "two_view",
        }:
            raise ValueError("gallery evidence requires one nDCG@10 quality per policy")

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        session: TrainingSessionConfig,
        checkpoint: CheckpointRecord,
        canonical_splits: pd.DataFrame,
    ) -> ValidatedTask6Evidence:
        """Validate Task 6, then hash and parse the exact validated artifact bytes."""

        from fashion.task4.learned_evidence import validate_learned_manifest

        validated_manifest = validate_learned_manifest(
            manifest_path,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=canonical_splits,
        )
        checkpoint = validated_manifest.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("validated manifest checkpoint identity is malformed")
        identity = EvidenceIdentity(
            run_id=str(_field(validated_manifest, "run_id")),
            method=str(_field(validated_manifest, "method")),  # type: ignore[arg-type]
            fold=_integer(_field(validated_manifest, "fold"), field="fold"),
            config_hash=str(_field(validated_manifest, "config_hash")),
            split_fingerprint=str(_field(validated_manifest, "split_fingerprint")),
            checkpoint_sha256=str(_field(checkpoint, "sha256")),
        )
        artifacts = validated_manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("validated manifest artifact records are malformed")
        records = {
            str(record["name"]): record
            for record in artifacts
            if isinstance(record, Mapping)
            and "name" in record
            and "path" in record
            and "sha256" in record
        }
        required = {"canvas_summary", "gallery_comparison", "cost"}
        if not required.issubset(records):
            raise ValueError("validated manifest lacks selection artifacts")

        def verified_bytes(name: str) -> bytes:
            record = records[name]
            path = Path(manifest_path).parent / str(record["path"])
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != str(record["sha256"]):
                raise ValueError(f"{name} artifact hash changed after Task 6 validation")
            return content

        canvas = pd.read_csv(io.BytesIO(verified_bytes("canvas_summary")))
        gallery = pd.read_csv(io.BytesIO(verified_bytes("gallery_comparison")))
        cost = json.loads(verified_bytes("cost"))
        _require_uniform_frame_identity(canvas, identity, label="canvas summary")
        _require_uniform_frame_identity(gallery, identity, label="gallery comparison")
        timing = pd.DataFrame(cost["timing_summary"])
        p95 = timing.loc[timing["metric"].eq("end_to_end") & timing["percentile"].eq("p95")]
        hardware = cost["hardware"]
        quality = gallery.loc[
            gallery["query_source"].eq("equal_teacher_v1_mean")
            & gallery["metric"].eq("gallery_quality")
            & pd.to_numeric(gallery["k"], errors="coerce").eq(10)
            & gallery["aggregation"].eq("equal_source_mean"),
            ["gallery_policy", "value"],
        ]
        selected_metrics = validated_manifest["selected_metrics"]
        return cls(
            identity=identity,
            canvas_ndcg_at_10=tuple(
                (str(row.query_variant), float(row.ndcg_at_10))
                for row in canvas.loc[canvas["query_variant"].isin(("wide", "tall"))].itertuples(
                    index=False
                )
            ),
            direction_p95_end_to_end_seconds=tuple(
                (
                    (str(row.query_source), str(row.gallery_source)),
                    float(row.value_seconds),
                )
                for row in p95.itertuples(index=False)
            ),
            source_index_bytes=tuple(
                (str(source), int(value)) for source, value in cost["index_bytes"].items()
            ),
            cpu_measurement=CPUMeasurementIdentity(
                route=str(cost["measurement_route"]),
                measurement_sha256=str(cost["measurement_sha256"]),
                cpu=str(hardware["cpu"]),
                operating_system=str(hardware["operating_system"]),
                thread_count=int(hardware["thread_count"]),
            ),
            development_winner_score=float(selected_metrics["development_winner_score"]),
            source_robustness_ratio=float(selected_metrics["source_robustness_ratio"]),
            gallery_quality_at_10=tuple(
                (str(row.gallery_policy), float(row.value))
                for row in quality.itertuples(index=False)
            ),
            _validation_token=_VALIDATED_TASK6_TOKEN,
        )


@dataclass(frozen=True, slots=True)
class DeploymentEvidence:
    """Finalist stability and one identity-bound Task 6 practical evidence package."""

    identity: EvidenceIdentity
    method: ScratchCandidateName
    stability_mean: float
    stability_standard_deviation: float
    source_robustness_ratio: float
    wide_canvas_ndcg_at_10: float
    tall_canvas_ndcg_at_10: float
    direction_p95_end_to_end_seconds: tuple[tuple[tuple[str, str], float], ...]
    source_index_bytes: tuple[tuple[str, int], ...]
    cpu_measurement: CPUMeasurementIdentity
    pretrained: bool
    weight_origin: str
    deployment_eligibility: str = "eligible"

    def __post_init__(self) -> None:
        if self.identity.method != self.method:
            raise ValueError("deployment method does not match evidence identity")
        for label, value in (
            ("stability mean", self.stability_mean),
            ("stability standard deviation", self.stability_standard_deviation),
            ("source robustness ratio", self.source_robustness_ratio),
            ("wide canvas nDCG", self.wide_canvas_ndcg_at_10),
            ("tall canvas nDCG", self.tall_canvas_ndcg_at_10),
        ):
            parsed = _finite_float(value, field=label)
            if parsed < 0:
                raise ValueError(f"{label} must be non-negative")

    @property
    def canvas_behaviour(self) -> float:
        return (self.wide_canvas_ndcg_at_10 + self.tall_canvas_ndcg_at_10) / 2.0

    @property
    def worst_p95_end_to_end_seconds(self) -> float:
        return max(value for _, value in self.direction_p95_end_to_end_seconds)

    @property
    def total_index_bytes(self) -> int:
        return sum(value for _, value in self.source_index_bytes)


def deployment_evidence_from_artifacts(
    registry_row: Mapping[str, Any],
    stability: StabilitySummary,
    evidence: ValidatedTask6Evidence,
) -> DeploymentEvidence:
    """Extract selection inputs from already validated registry/evidence fields."""

    identity = EvidenceIdentity.from_registry_row(registry_row)
    if (
        identity != stability.finalist.identity
        or identity != stability.finalist.config_artifact.identity
        or identity != evidence.identity
    ):
        raise ValueError("registry, stability, config, and Task 6 evidence identity do not match")
    canvas = dict(evidence.canvas_ndcg_at_10)
    return DeploymentEvidence(
        identity=identity,
        method=identity.method,  # type: ignore[arg-type]
        stability_mean=stability.mean,
        stability_standard_deviation=stability.standard_deviation,
        source_robustness_ratio=evidence.source_robustness_ratio,
        wide_canvas_ndcg_at_10=canvas["wide"],
        tall_canvas_ndcg_at_10=canvas["tall"],
        direction_p95_end_to_end_seconds=evidence.direction_p95_end_to_end_seconds,
        source_index_bytes=evidence.source_index_bytes,
        cpu_measurement=evidence.cpu_measurement,
        pretrained=not _false(_field(registry_row, "pretrained")),
        weight_origin=str(_field(registry_row, "weight_origin")),
        deployment_eligibility=str(_field(registry_row, "deployment_eligibility")),
    )


def passes_deployment_gates(candidate: DeploymentEvidence) -> bool:
    """Require scratch provenance, four sub-second directions, and sub-GiB indexes."""

    return (
        candidate.method in SCRATCH_CANDIDATE_ORDER
        and not candidate.pretrained
        and candidate.weight_origin == SCRATCH_WEIGHT_ORIGIN
        and candidate.deployment_eligibility == "eligible"
        and len(candidate.direction_p95_end_to_end_seconds) == 4
        and {direction for direction, _ in candidate.direction_p95_end_to_end_seconds}
        == set(REQUIRED_DIRECTIONS)
        and all(
            math.isfinite(value) and 0 <= value < 1.0
            for _, value in candidate.direction_p95_end_to_end_seconds
        )
        and len(candidate.source_index_bytes) == 2
        and {source for source, _ in candidate.source_index_bytes} == {"teacher", "v1"}
        and all(
            isinstance(value, Integral)
            and not isinstance(value, bool)
            and 0 <= value < INDEX_LIMIT_BYTES
            for _, value in candidate.source_index_bytes
        )
        and candidate.cpu_measurement.route == CPU_MEASUREMENT_ROUTE
        and candidate.cpu_measurement.thread_count == 1
    )


def _select_deployment_evidence(
    candidates: Sequence[DeploymentEvidence],
) -> DeploymentEvidence:
    """Apply gates, pooled-spread judgement, then frozen practical tie-breaks."""

    if len(candidates) != 2 or len({candidate.method for candidate in candidates}) != 2:
        raise ValueError("deployment selection requires two distinct stability finalists")
    eligible = [candidate for candidate in candidates if passes_deployment_gates(candidate)]
    if not eligible:
        raise ValueError("no stability finalist passes the deployment gates")
    if len(eligible) == 1:
        return eligible[0]
    left, right = eligible
    spread = pooled_spread(
        left.stability_standard_deviation,
        right.stability_standard_deviation,
    )
    gap = abs(left.stability_mean - right.stability_mean)
    if gap > spread:
        return min(
            eligible,
            key=lambda candidate: (
                -candidate.stability_mean,
                _method_rank(candidate.method),
            ),
        )
    return min(
        eligible,
        key=lambda candidate: (
            -candidate.source_robustness_ratio,
            -candidate.canvas_behaviour,
            candidate.worst_p95_end_to_end_seconds,
            candidate.total_index_bytes,
            _method_rank(candidate.method),
        ),
    )


@dataclass(frozen=True, slots=True)
class DeploymentDecisionInput:
    """Canonical manifests, configs, and lifecycle rows for one finalist decision."""

    registry_row: Mapping[str, Any]
    finalist: StabilityFinalist
    stability_rows: tuple[Mapping[str, Any], ...]
    stability_config_artifacts: Mapping[str, ExperimentConfigArtifact]
    stability_evidence_manifests: Mapping[str, Task6ManifestInput | StabilityEvidenceInput]
    candidate_evidence_manifest: Task6ManifestInput


def select_deployment_candidate(
    candidates: Sequence[DeploymentDecisionInput],
) -> DeploymentEvidence:
    """Revalidate all candidate/stability manifests at the deployment outcome boundary."""

    if len(candidates) != 2 or any(
        not isinstance(candidate, DeploymentDecisionInput) for candidate in candidates
    ):
        raise ValueError("deployment decision requires two canonical manifest inputs")
    evidence: list[DeploymentEvidence] = []
    for candidate in candidates:
        stability = summarize_stability(
            candidate.stability_rows,
            finalist=candidate.finalist,
            config_artifacts=candidate.stability_config_artifacts,
            evidence_manifests=candidate.stability_evidence_manifests,
        )
        task6_evidence = candidate.candidate_evidence_manifest.validated()
        evidence.append(
            deployment_evidence_from_artifacts(
                candidate.registry_row,
                stability,
                task6_evidence,
            )
        )
    return _select_deployment_evidence(evidence)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _artifact_reference(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class PostStabilityDeploymentDecision:
    """Validated, reproducible post-stability deployment decision."""

    winner: DeploymentEvidence
    summaries: tuple[StabilitySummary, StabilitySummary]
    canonical_json: str = field(repr=False)
    _validation_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_token is not _POST_STABILITY_DECISION_TOKEN:
            raise TypeError("post-stability decision requires canonical deployment inputs")
        if self.winner.method not in SCRATCH_CANDIDATE_ORDER or not passes_deployment_gates(
            self.winner
        ):
            raise ValueError("post-stability decision winner must be an eligible scratch model")


def _post_stability_deployment_decision(
    candidates: Sequence[DeploymentDecisionInput],
    *,
    split_path: str,
) -> PostStabilityDeploymentDecision:
    if split_path != "data/processed/splits.csv":
        raise ValueError("post-stability deployment must use data/processed/splits.csv")
    if len(candidates) != 2 or any(
        not isinstance(candidate, DeploymentDecisionInput) for candidate in candidates
    ):
        raise ValueError("post-stability deployment requires two canonical finalist inputs")

    summaries: list[StabilitySummary] = []
    evidence: list[DeploymentEvidence] = []
    summary_records: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    for candidate in candidates:
        summary = summarize_stability(
            candidate.stability_rows,
            finalist=candidate.finalist,
            config_artifacts=candidate.stability_config_artifacts,
            evidence_manifests=candidate.stability_evidence_manifests,
        )
        task6 = candidate.candidate_evidence_manifest.validated()
        deployment = deployment_evidence_from_artifacts(
            candidate.registry_row,
            summary,
            task6,
        )
        summaries.append(summary)
        evidence.append(deployment)

        fold_records: list[dict[str, object]] = []
        stability_sources: list[dict[str, object]] = []
        for identity in summary.run_identities:
            manifest_input = candidate.stability_evidence_manifests[identity.run_id]
            validated = manifest_input.validated()
            fold_record: dict[str, object] = {
                "run_id": identity.run_id,
                "fold": identity.fold,
                "config_hash": identity.config_hash,
                "checkpoint_sha256": identity.checkpoint_sha256,
                "score": validated.development_winner_score,
            }
            if isinstance(validated, ValidatedStabilityEvidence):
                fold_record.update(
                    {
                        "coverage": validated.primary_coverage,
                        "scorable_query_count": validated.scorable_query_count,
                        "total_query_count": validated.total_query_count,
                    }
                )
            fold_records.append(fold_record)
            stability_sources.append(
                {
                    "run_id": identity.run_id,
                    **(
                        _artifact_reference(manifest_input.artifact_path)
                        if isinstance(manifest_input, StabilityEvidenceInput)
                        else _artifact_reference(manifest_input.manifest_path)
                    ),
                }
            )
        summary_records.append(
            {
                "candidate_run_id": candidate.finalist.candidate_run_id,
                "method": summary.method,
                "mean": summary.mean,
                "sample_standard_deviation": summary.standard_deviation,
                "folds": fold_records,
            }
        )
        source_records.append(
            {
                "candidate_run_id": candidate.finalist.candidate_run_id,
                "candidate_manifest": _artifact_reference(
                    candidate.candidate_evidence_manifest.manifest_path
                ),
                "stability_manifests": stability_sources,
            }
        )

    winner = _select_deployment_evidence(evidence)
    summary_pair = (summaries[0], summaries[1])
    mean_gap = abs(summary_pair[0].mean - summary_pair[1].mean)
    spread = pooled_spread(
        summary_pair[0].standard_deviation,
        summary_pair[1].standard_deviation,
    )
    canvas = {
        "wide": winner.wide_canvas_ndcg_at_10,
        "tall": winner.tall_canvas_ndcg_at_10,
    }
    selected_model = {
        "run_id": winner.identity.run_id,
        "method": winner.method,
        "scratch_gate": (
            not winner.pretrained and winner.weight_origin == SCRATCH_WEIGHT_ORIGIN
        ),
        "cpu_p95_gate": winner.worst_p95_end_to_end_seconds < 1.0,
        "index_size_gate": all(
            value < INDEX_LIMIT_BYTES for _, value in winner.source_index_bytes
        ),
        "all_deployment_gates_passed": passes_deployment_gates(winner),
        "stability_mean": winner.stability_mean,
        "stability_sample_standard_deviation": winner.stability_standard_deviation,
        "source_robustness_ratio": winner.source_robustness_ratio,
        "canvas_ndcg_at_10": canvas,
        "cpu_p95_end_to_end_seconds": {
            f"{query}_to_{gallery}": value
            for (query, gallery), value in winner.direction_p95_end_to_end_seconds
        },
        "source_index_bytes": dict(winner.source_index_bytes),
        "cpu_measurement": {
            "route": winner.cpu_measurement.route,
            "measurement_sha256": winner.cpu_measurement.measurement_sha256,
            "cpu": winner.cpu_measurement.cpu,
            "operating_system": winner.cpu_measurement.operating_system,
            "thread_count": winner.cpu_measurement.thread_count,
        },
    }
    payload: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": "task4_post_stability_deployment_judgement",
        "producer": "fashion.task4.experiments.write_post_stability_deployment_artifact",
        "development_only": True,
        "split_path": split_path,
        "split_fingerprint": winner.identity.split_fingerprint,
        "holdout_opened": False,
        "quarantine_opened": False,
        "official_teacher_test_opened": False,
        "stability_summaries": summary_records,
        "source_artifacts": source_records,
        "mean_gap": mean_gap,
        "pooled_spread": spread,
        "mean_gap_exceeds_pooled_spread": mean_gap > spread,
        "selected_model": selected_model,
    }
    unsigned = _canonical_json(payload)
    payload["decision_sha256"] = hashlib.sha256(unsigned.encode("utf-8")).hexdigest()
    return PostStabilityDeploymentDecision(
        winner=winner,
        summaries=summary_pair,
        canonical_json=_canonical_json(payload),
        _validation_token=_POST_STABILITY_DECISION_TOKEN,
    )


def write_post_stability_deployment_artifact(
    destination: str | Path,
    candidates: Sequence[DeploymentDecisionInput],
    *,
    split_path: str = "data/processed/splits.csv",
) -> Path:
    """Derive and atomically persist the canonical post-stability decision."""

    decision = _post_stability_deployment_decision(candidates, split_path=split_path)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(decision.canonical_json, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_post_stability_deployment_artifact(
    artifact_path: str | Path,
    candidates: Sequence[DeploymentDecisionInput],
    *,
    split_path: str = "data/processed/splits.csv",
) -> PostStabilityDeploymentDecision:
    """Re-derive and require the exact canonical post-stability artifact bytes."""

    expected = _post_stability_deployment_decision(candidates, split_path=split_path)
    try:
        actual = Path(artifact_path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"post-stability deployment artifact cannot be read: {error}") from error
    if actual != expected.canonical_json:
        raise ValueError("post-stability deployment artifact disagrees with derived canonical data")
    return expected


@dataclass(frozen=True, slots=True)
class GalleryOption:
    """Quality and practical cost for one deployed gallery policy."""

    policy: GalleryPolicy
    quality: float
    index_bytes: int
    p95_end_to_end_seconds: float


@dataclass(frozen=True, slots=True)
class GalleryPracticalEvidence:
    """Identity-bound per-policy speed evidence for the gallery comparison."""

    identity: EvidenceIdentity
    policy_p95_end_to_end_seconds: tuple[tuple[str, float], ...]
    _validation_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_token is not _GALLERY_PRACTICAL_TOKEN:
            raise TypeError("gallery practical evidence requires validated Task 6 evidence")
        policies = tuple(policy for policy, _ in self.policy_p95_end_to_end_seconds)
        if len(policies) != 3 or set(policies) != {"teacher", "v1", "two_view"}:
            raise ValueError("gallery timing requires exact teacher, v1, and two_view policies")
        if any(
            not math.isfinite(value) or value < 0 for _, value in self.policy_p95_end_to_end_seconds
        ):
            raise ValueError("gallery timing values must be finite and non-negative")


def derive_gallery_practical_evidence(
    evidence_manifest: Task6ManifestInput,
    *,
    policy_timing_output_path: str | Path | None = None,
    timing_policy: TimingPolicy = TimingPolicy(),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> GalleryPracticalEvidence:
    """Produce and validate measured Task 6 gallery-policy timing evidence."""

    from fashion.task4.learned_evidence import (
        validate_gallery_policy_timing_artifact,
        write_gallery_policy_timing_artifact,
    )

    evidence = evidence_manifest.validated()
    destination = (
        Path(policy_timing_output_path)
        if policy_timing_output_path is not None
        else Path(evidence_manifest.manifest_path).with_name("gallery_policy_timing.json")
    )
    timing_path = write_gallery_policy_timing_artifact(
        destination,
        manifest_path=evidence_manifest.manifest_path,
        session=evidence_manifest.session,
        checkpoint=evidence_manifest.checkpoint,
        canonical_splits=evidence_manifest.canonical_splits,
        policy=timing_policy,
        clock_ns=clock_ns,
    )
    timings = validate_gallery_policy_timing_artifact(
        timing_path,
        session=evidence_manifest.session,
        checkpoint=evidence_manifest.checkpoint,
    )
    return GalleryPracticalEvidence(
        identity=evidence.identity,
        policy_p95_end_to_end_seconds=tuple(
            (policy, timings[policy]) for policy in ("teacher", "v1", "two_view")
        ),
        _validation_token=_GALLERY_PRACTICAL_TOKEN,
    )


def gallery_options_from_evidence(
    evidence: ValidatedTask6Evidence,
    *,
    practical: GalleryPracticalEvidence,
) -> tuple[GalleryOption, ...]:
    """Read validated equal-source gallery quality and derive each policy's cost."""

    if evidence.identity != practical.identity:
        raise ValueError("gallery quality and measured timing identity do not match")
    quality = dict(evidence.gallery_quality_at_10)
    source_index_bytes = dict(evidence.source_index_bytes)
    policy_p95_end_to_end_seconds = dict(practical.policy_p95_end_to_end_seconds)
    storage = {
        "teacher": source_index_bytes["teacher"],
        "v1": source_index_bytes["v1"],
        "two_view": source_index_bytes["teacher"] + source_index_bytes["v1"],
    }
    return tuple(
        GalleryOption(
            policy=policy,
            quality=_finite_float(
                quality[policy],
                field="gallery quality",
            ),
            index_bytes=storage[policy],
            p95_end_to_end_seconds=_finite_float(
                policy_p95_end_to_end_seconds[policy],
                field="gallery p95 end-to-end seconds",
            ),
        )
        for policy in ("teacher", "v1", "two_view")
    )


def _select_gallery_options(options: Sequence[GalleryOption]) -> GalleryOption:
    """Choose quality first; within tolerance prefer smaller then faster."""

    if len(options) != 3 or {option.policy for option in options} != {
        "teacher",
        "v1",
        "two_view",
    }:
        raise ValueError("gallery selection requires teacher, v1, and two_view options")
    if any(
        not math.isfinite(option.quality)
        or not math.isfinite(option.p95_end_to_end_seconds)
        or option.index_bytes < 0
        or option.p95_end_to_end_seconds < 0
        for option in options
    ):
        raise ValueError("gallery quality and cost values must be finite and non-negative")
    best_quality = max(option.quality for option in options)
    tied = [
        option for option in options if best_quality - option.quality <= GALLERY_QUALITY_TOLERANCE
    ]
    policy_order = ("teacher", "v1", "two_view")
    return min(
        tied,
        key=lambda option: (
            option.index_bytes,
            option.p95_end_to_end_seconds,
            policy_order.index(option.policy),
        ),
    )


def select_gallery_source(
    evidence_manifest: Task6ManifestInput,
    *,
    policy_timing_output_path: str | Path | None = None,
    timing_policy: TimingPolicy = TimingPolicy(),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> GalleryOption:
    """Revalidate quality and produce measured policy timing at the outcome boundary."""

    if not isinstance(evidence_manifest, Task6ManifestInput):
        raise ValueError("gallery decision requires canonical manifest input")
    evidence = evidence_manifest.validated()
    practical = derive_gallery_practical_evidence(
        evidence_manifest,
        policy_timing_output_path=policy_timing_output_path,
        timing_policy=timing_policy,
        clock_ns=clock_ns,
    )
    options = gallery_options_from_evidence(evidence, practical=practical)
    return _select_gallery_options(options)


@dataclass(frozen=True, slots=True)
class FinalGalleryDecision:
    """Validated final policy overlay for immutable candidate cost evidence."""

    policy: GalleryPolicy
    canonical_json: str = field(repr=False)
    _validation_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_token is not _FINAL_GALLERY_DECISION_TOKEN:
            raise TypeError("final gallery decision requires canonical evidence")


def _final_gallery_decision(
    *,
    deployment: PostStabilityDeploymentDecision,
    evidence_manifest: Task6ManifestInput,
    timing_artifact_path: str | Path,
) -> FinalGalleryDecision:
    if not isinstance(deployment, PostStabilityDeploymentDecision):
        raise ValueError("final gallery decision requires validated deployment evidence")
    evidence = evidence_manifest.validated()
    if evidence.identity != deployment.winner.identity:
        raise ValueError("final gallery evidence does not match the deployment winner")

    from fashion.task4.learned_evidence import validate_gallery_policy_timing_artifact

    timing_path = Path(timing_artifact_path)
    timings = validate_gallery_policy_timing_artifact(
        timing_path,
        session=evidence_manifest.session,
        checkpoint=evidence_manifest.checkpoint,
    )
    practical = GalleryPracticalEvidence(
        identity=evidence.identity,
        policy_p95_end_to_end_seconds=tuple(
            (policy, timings[policy]) for policy in ("teacher", "v1", "two_view")
        ),
        _validation_token=_GALLERY_PRACTICAL_TOKEN,
    )
    options = gallery_options_from_evidence(evidence, practical=practical)
    selected = _select_gallery_options(options)

    manifest_path = evidence_manifest.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate manifest artifact records are malformed")
    cost_record = next(
        (
            record
            for record in artifacts
            if isinstance(record, Mapping) and record.get("name") == "cost"
        ),
        None,
    )
    if not isinstance(cost_record, Mapping):
        raise ValueError("candidate manifest lacks immutable cost evidence")
    cost_path = manifest_path.parent / str(cost_record.get("path") or "")
    cost_reference = _artifact_reference(cost_path)
    if cost_reference["sha256"] != cost_record.get("sha256"):
        raise ValueError("candidate cost hash disagrees with its immutable manifest")
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    candidate_cost_policy = cost.get("selected_gallery_policy")
    if candidate_cost_policy not in {"teacher", "v1", "two_view"}:
        raise ValueError("candidate cost policy is invalid")

    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "task4_final_gallery_decision",
        "producer": "fashion.task4.experiments.write_final_gallery_decision_artifact",
        "development_only": True,
        "holdout_opened": False,
        "quarantine_opened": False,
        "official_teacher_test_opened": False,
        "deployment_decision": {
            "run_id": deployment.winner.identity.run_id,
            "method": deployment.winner.method,
            "sha256": hashlib.sha256(
                deployment.canonical_json.encode("utf-8")
            ).hexdigest(),
        },
        "candidate_manifest": _artifact_reference(manifest_path),
        "candidate_cost_policy": {
            "policy": candidate_cost_policy,
            "role": "pre_study_cost_assumption",
            **cost_reference,
        },
        "timing_artifact": _artifact_reference(timing_path),
        "policies": [
            {
                "policy": option.policy,
                "quality_at_10": option.quality,
                "p95_end_to_end_seconds": option.p95_end_to_end_seconds,
                "index_bytes": option.index_bytes,
                "query_normalization_source": query_normalization_source(option.policy),
            }
            for option in options
        ],
        "final_policy": {
            "policy": selected.policy,
            "source": "three_policy_development_study",
            "quality_at_10": selected.quality,
            "p95_end_to_end_seconds": selected.p95_end_to_end_seconds,
            "index_bytes": selected.index_bytes,
            "query_normalization_source": query_normalization_source(selected.policy),
        },
    }
    unsigned = _canonical_json(payload)
    payload["decision_sha256"] = hashlib.sha256(unsigned.encode("utf-8")).hexdigest()
    return FinalGalleryDecision(
        policy=selected.policy,
        canonical_json=_canonical_json(payload),
        _validation_token=_FINAL_GALLERY_DECISION_TOKEN,
    )


def write_final_gallery_decision_artifact(
    destination: str | Path,
    *,
    deployment: PostStabilityDeploymentDecision,
    evidence_manifest: Task6ManifestInput,
    timing_artifact_path: str | Path,
) -> Path:
    """Persist the final policy while preserving the candidate cost assumption."""

    decision = _final_gallery_decision(
        deployment=deployment,
        evidence_manifest=evidence_manifest,
        timing_artifact_path=timing_artifact_path,
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(decision.canonical_json, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_final_gallery_decision_artifact(
    artifact_path: str | Path,
    *,
    deployment: PostStabilityDeploymentDecision,
    evidence_manifest: Task6ManifestInput,
    timing_artifact_path: str | Path,
) -> FinalGalleryDecision:
    """Re-derive and require the exact canonical final gallery overlay."""

    expected = _final_gallery_decision(
        deployment=deployment,
        evidence_manifest=evidence_manifest,
        timing_artifact_path=timing_artifact_path,
    )
    try:
        actual = Path(artifact_path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"final gallery decision artifact cannot be read: {error}") from error
    if actual != expected.canonical_json:
        raise ValueError("final gallery decision artifact disagrees with derived canonical data")
    return expected


def link_gallery_result_to_final_decision(
    gallery_result_path: str | Path,
    *,
    decision_path: str | Path,
    decision: FinalGalleryDecision,
) -> Path:
    """Bind a runner result to the validated final decision and old cost role."""

    if not isinstance(decision, FinalGalleryDecision):
        raise ValueError("gallery result link requires a validated final decision")
    final_path = Path(decision_path)
    if final_path.read_text(encoding="utf-8") != decision.canonical_json:
        raise ValueError("final gallery decision bytes disagree with validated decision")
    final_payload = json.loads(decision.canonical_json)
    final_policy = final_payload["final_policy"]
    deployment = final_payload["deployment_decision"]
    candidate_cost = final_payload["candidate_cost_policy"]
    result_path = Path(gallery_result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    phase_result = result.get("phase_result") if isinstance(result, Mapping) else None
    if (
        result.get("phase") != "gallery"
        or not isinstance(phase_result, dict)
        or phase_result.get("deployment_winner") != deployment["run_id"]
        or phase_result.get("selected_gallery") != final_policy["policy"]
        or phase_result.get("selected_gallery_p95_seconds")
        != final_policy["p95_end_to_end_seconds"]
        or phase_result.get("selected_gallery_index_bytes") != final_policy["index_bytes"]
    ):
        raise ValueError("gallery result policy or deployment winner disagrees with final decision")
    phase_result["candidate_cost_policy"] = {
        "policy": candidate_cost["policy"],
        "role": candidate_cost["role"],
        "path": candidate_cost["path"],
        "sha256": candidate_cost["sha256"],
    }
    phase_result["final_decision"] = {
        "path": str(final_path),
        "sha256": hashlib.sha256(decision.canonical_json.encode("utf-8")).hexdigest(),
        "policy": decision.policy,
        "source": final_policy["source"],
    }
    content = _canonical_json(result)
    temporary = result_path.with_name(f".{result_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(result_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result_path


def query_normalization_source(policy: GalleryPolicy) -> Literal["teacher", "v1"]:
    """Use V1 normalization for a deployed two-view gallery."""

    if policy == "teacher":
        return "teacher"
    if policy in {"v1", "two_view"}:
        return "v1"
    raise ValueError("gallery policy must be teacher, v1, or two_view")
