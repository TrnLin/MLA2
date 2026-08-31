"""Verified Task 2 component handoff that keeps final evaluation locked."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping

import pandas as pd
import psutil

from fashion.config import (
    ROOT,
    RUNS_CSV,
    TASK2_EVIDENCE_DIR,
    TASK2_MODEL_MANIFEST_JSON,
)
from fashion.data.hashing import compute_sha256
from fashion.task2.inference import SeasonPrediction
from fashion.task2.refit import (
    _load_verified_development_refit_package,
    _verify_refit_registry,
    load_verified_development_refit_manifest,
)
from fashion.task2.ultimate_judgement import (
    load_verified_selection_freeze,
    load_verified_ultimate_judgement_manifest,
)
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.registry import RUN_COLUMNS, SHA256_PATTERN, RunRegistry

DEFAULT_HANDOFF_DIRECTORY = TASK2_EVIDENCE_DIR / "final_handoff"
DEFAULT_ULTIMATE_MANIFEST = TASK2_EVIDENCE_DIR / "ultimate_judgement/manifest.json"
HANDOFF_LOCK_FILENAME = ".task2-handoff.lock"
HANDOFF_SCHEMA_VERSION = "1.1.0"
REGISTRY_SNAPSHOT_FILENAME = "registry_snapshot.csv"
SEASON_LABELS = ("Fall", "Spring", "Summer", "Winter")

_AUDIT_COLUMNS = (
    "artifact",
    "path",
    "expected_sha256",
    "actual_sha256",
    "status",
    "role",
)
_REQUIRED_AUDIT_ARTIFACTS = frozenset(
    {
        "ultimate_judgement_manifest",
        "selection_freeze",
        "development_refit_manifest",
        "development_refit_bundle",
        "development_refit_history",
        "development_refit_runtime",
        "canonical_splits",
        "canonical_label_maps",
        "inference_source",
        "registry_binding",
    }
)
_SMOKE_FIELDS = {
    "schema_version",
    "input_scope",
    "product_id",
    "image_path",
    "predicted_label",
    "probabilities",
    "probability_order",
    "confidence",
    "review_required",
    "latency_measured",
    "run_id",
    "model_manifest_sha256",
    "bundle_sha256",
    "holdout_opened",
}


def _resolve_within_root(path: str | Path, *, root: Path, scope: str) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{scope} is outside project root: {resolved}") from error
    return resolved


def _relative_path(path: Path, *, root: Path) -> str:
    return _resolve_within_root(path, root=root, scope="handoff artifact").relative_to(
        root
    ).as_posix()


def _rebase_project_default(path: str | Path, *, default: Path, root: Path) -> Path:
    """Map an absolute config default into an explicitly supplied project root."""
    candidate = Path(path)
    if candidate.resolve() == default.resolve() and root != ROOT.resolve():
        return root / default.resolve().relative_to(ROOT.resolve())
    return candidate


def _declaration(path: Path, *, root: Path) -> dict[str, str]:
    resolved = _resolve_within_root(path, root=root, scope="handoff artifact")
    if not resolved.is_file():
        raise FileNotFoundError(f"handoff artifact does not exist: {resolved}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": compute_sha256(resolved),
    }


def _staged_declaration(
    final_path: Path,
    staged_path: Path,
    *,
    root: Path,
) -> dict[str, str]:
    return {
        "path": _relative_path(final_path, root=root),
        "sha256": compute_sha256(staged_path),
    }


def _declared_path(
    declaration: Mapping[str, Any],
    *,
    root: Path,
    scope: str,
) -> Path:
    if set(declaration) != {"path", "sha256"}:
        raise ValueError(f"{scope} must contain only path and sha256")
    resolved = _resolve_within_root(declaration["path"], root=root, scope=scope)
    verify_artifact(resolved, str(declaration["sha256"]))
    return resolved


def _load_json_object(path: Path, scope: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{scope} must be a JSON object")
    return payload


def _audit_file(
    name: str,
    path: Path,
    expected_sha256: str,
    role: str,
    *,
    root: Path,
) -> dict[str, str]:
    resolved = _resolve_within_root(path, root=root, scope=name)
    actual = verify_artifact(resolved, expected_sha256)
    return {
        "artifact": name,
        "path": resolved.relative_to(root).as_posix(),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "status": "PASS",
        "role": role,
    }


def _registry_snapshot(
    refit: Mapping[str, Any],
    registry_path: str | Path,
    *,
    root: Path,
) -> tuple[pd.DataFrame, Path]:
    resolved = _resolve_within_root(registry_path, root=root, scope="refit registry")
    _verify_refit_registry(refit, registry_path=resolved, root=root)
    registry = RunRegistry(resolved).read()
    matches = registry.loc[registry["run_id"].eq(str(refit["run_id"]))]
    if len(matches) != 1:
        raise ValueError("Task 2 registry snapshot requires exactly one refit row")
    return matches.loc[:, RUN_COLUMNS].reset_index(drop=True), resolved


def audit_task2_artifacts(
    *,
    project_root: str | Path = ROOT,
    registry_path: str | Path = RUNS_CSV,
    registry_snapshot_path: str | Path | None = None,
    ultimate_manifest_path: str | Path = DEFAULT_ULTIMATE_MANIFEST,
    model_manifest_path: str | Path = TASK2_MODEL_MANIFEST_JSON,
) -> pd.DataFrame:
    """Verify the frozen decision, refit package, registry row, and deployed source."""
    root = Path(project_root).resolve()
    ultimate, resolved_ultimate, ultimate_artifacts = (
        load_verified_ultimate_judgement_manifest(
            ultimate_manifest_path,
            project_root=root,
        )
    )
    if registry_snapshot_path is None:
        refit, resolved_refit, _ = load_verified_development_refit_manifest(
            model_manifest_path,
            project_root=root,
            registry_path=registry_path,
        )
        registry_frame, resolved_registry = _registry_snapshot(
            refit,
            registry_path,
            root=root,
        )
        registry_sha256 = canonical_sha256(registry_frame.iloc[0].to_dict())
    else:
        refit, resolved_refit, _ = _load_verified_development_refit_package(
            model_manifest_path,
            project_root=root,
        )
        _, resolved_registry = _registry_snapshot(
            refit,
            registry_snapshot_path,
            root=root,
        )
        registry_sha256 = compute_sha256(resolved_registry)
    if (
        ultimate["selected_candidate"] != refit["selected_candidate"]
        or ultimate["selected_experiment_id"] != refit["selected_experiment_id"]
        or ultimate["holdout_opened"] is not False
        or ultimate["holdout_metrics_present"] is not False
        or refit["holdout_opened"] is not False
        or refit["holdout_metrics_present"] is not False
        or refit["scratch"] is not True
        or refit["weights"] is not None
        or refit["final_eligible"] is not True
    ):
        raise ValueError("Task 2 decision and development refit boundaries disagree")
    if ultimate["artifacts"]["selection_freeze"] != refit["selection_freeze"]:
        raise ValueError("Task 2 selection freeze changed between G7 and G8")

    declarations = {
        "selection_freeze": refit["selection_freeze"],
        "development_refit_bundle": refit["bundle"],
        "development_refit_history": refit["artifacts"]["history"],
        "development_refit_runtime": refit["artifacts"]["runtime"],
        "canonical_splits": refit["canonical_inputs"]["splits"],
        "canonical_label_maps": refit["canonical_inputs"]["label_maps"],
    }
    roles = {
        "selection_freeze": "immutable pre-holdout model decision",
        "development_refit_bundle": "scratch image-only model weights",
        "development_refit_history": "24-epoch optimisation diagnostics",
        "development_refit_runtime": "training environment and cost provenance",
        "canonical_splits": "only allowed data partition contract",
        "canonical_label_maps": "frozen Season class order",
    }
    rows = [
        _audit_file(
            "ultimate_judgement_manifest",
            resolved_ultimate,
            compute_sha256(resolved_ultimate),
            "verified G7 winner and rejected alternatives",
            root=root,
        ),
        _audit_file(
            "development_refit_manifest",
            resolved_refit,
            compute_sha256(resolved_refit),
            "registry-bound G8 package contract",
            root=root,
        ),
    ]
    for name, declaration in declarations.items():
        path = _declared_path(declaration, root=root, scope=name)
        rows.append(
            _audit_file(
                name,
                path,
                str(declaration["sha256"]),
                roles[name],
                root=root,
            )
        )
    inference_source = root / "src/fashion/task2/inference.py"
    rows.append(
        _audit_file(
            "inference_source",
            inference_source,
            compute_sha256(inference_source),
            "verified image-only prediction interface",
            root=root,
        )
    )
    rows.append(
        {
            "artifact": "registry_binding",
            "path": resolved_registry.relative_to(root).as_posix(),
            "expected_sha256": registry_sha256,
            "actual_sha256": registry_sha256,
            "status": "PASS",
            "role": f"exact completed row for {refit['run_id']}",
        }
    )
    if ultimate_artifacts["selection_freeze"] != _declared_path(
        refit["selection_freeze"], root=root, scope="selection freeze"
    ):
        raise ValueError("verified G7 and G8 selection-freeze paths disagree")
    audit = pd.DataFrame(rows, columns=_AUDIT_COLUMNS)
    _validate_artifact_audit(audit)
    return audit


def _normalise_audit(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.loc[:, _AUDIT_COLUMNS].astype("string").fillna("").reset_index(drop=True)


def _validate_artifact_audit(audit: pd.DataFrame) -> None:
    if tuple(audit.columns) != _AUDIT_COLUMNS:
        raise ValueError("Task 2 artifact audit columns changed")
    if audit["artifact"].duplicated().any():
        raise ValueError("Task 2 artifact audit contains duplicate rows")
    if frozenset(audit["artifact"]) != _REQUIRED_AUDIT_ARTIFACTS:
        raise ValueError("Task 2 artifact audit is incomplete")
    if not audit["status"].eq("PASS").all():
        raise ValueError("Task 2 artifact audit contains a failed check")
    for column in ("expected_sha256", "actual_sha256"):
        invalid = [value for value in audit[column] if not SHA256_PATTERN.fullmatch(str(value))]
        if invalid:
            raise ValueError(f"Task 2 artifact audit contains invalid {column}")
    if (
        audit["path"].astype(str).str.strip().eq("").any()
        or audit["role"].astype(str).str.strip().eq("").any()
        or not audit["expected_sha256"].eq(audit["actual_sha256"]).all()
    ):
        raise ValueError("Task 2 artifact audit contains an unverifiable declaration")


def _assert_same_audit(
    supplied: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    scope: str,
) -> None:
    _validate_artifact_audit(supplied)
    _validate_artifact_audit(recomputed)
    if not _normalise_audit(supplied).equals(_normalise_audit(recomputed)):
        raise ValueError(f"{scope} audit differs from recomputed artifact evidence")


def _validate_probability_contract(
    probabilities: Mapping[str, Any],
    *,
    predicted_label: Any,
    confidence: Any,
) -> None:
    if tuple(probabilities) != SEASON_LABELS or predicted_label not in SEASON_LABELS:
        raise ValueError("Task 2 inference smoke prediction changed its frozen contract")
    values = [float(probabilities[label]) for label in SEASON_LABELS]
    if (
        not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
        or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("Task 2 inference smoke prediction changed its frozen contract")
    winning_index = max(range(len(values)), key=values.__getitem__)
    selected_probability = values[winning_index]
    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Task 2 inference smoke prediction changed its frozen contract"
        ) from error
    if (
        predicted_label != SEASON_LABELS[winning_index]
        or not math.isfinite(numeric_confidence)
        or not math.isclose(
            numeric_confidence,
            selected_probability,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Task 2 inference smoke prediction changed its frozen contract")


def _validate_smoke_image(
    image_path: str | Path,
    *,
    refit: Mapping[str, Any],
    root: Path,
    expected_product_id: str | None = None,
) -> tuple[str, str]:
    image = _resolve_within_root(image_path, root=root, scope="smoke image")
    relative_image = image.relative_to(root).as_posix()
    splits_path = _declared_path(
        refit["canonical_inputs"]["splits"], root=root, scope="canonical splits"
    )
    splits = pd.read_csv(splits_path, dtype={"id": "string"})
    normalised_paths = splits["path"].astype("string").str.replace("\\", "/", regex=False)
    matches = splits.loc[normalised_paths.eq(relative_image)]
    if len(matches) != 1:
        raise ValueError("Task 2 inference smoke image is not unique in canonical splits")
    row = matches.iloc[0]
    product_id = str(row["id"])
    if (
        row["partition"] != "development"
        or str(row["has_season_label"]).strip().lower() != "true"
        or (expected_product_id is not None and product_id != expected_product_id)
    ):
        raise ValueError("Task 2 inference smoke image must be labelled development data")
    return product_id, relative_image


def _smoke_from_prediction(
    prediction: SeasonPrediction,
    *,
    refit: Mapping[str, Any],
    resolved_refit: Path,
    root: Path,
) -> dict[str, Any]:
    if prediction.manifest_sha256 != compute_sha256(resolved_refit):
        raise ValueError("Task 2 inference smoke used a different model manifest")
    if (
        prediction.run_id != refit["run_id"]
        or prediction.bundle_sha256 != refit["bundle"]["sha256"]
        or prediction.review_required is not None
        or not math.isfinite(float(prediction.latency_ms))
        or float(prediction.latency_ms) < 0.0
    ):
        raise ValueError("Task 2 inference smoke prediction changed its frozen contract")
    _validate_probability_contract(
        prediction.probabilities,
        predicted_label=prediction.predicted_label,
        confidence=prediction.confidence,
    )
    product_id, relative_image = _validate_smoke_image(
        prediction.image_path,
        refit=refit,
        root=root,
    )
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "input_scope": "one labelled development image",
        "product_id": product_id,
        "image_path": relative_image,
        "predicted_label": prediction.predicted_label,
        "probabilities": {
            label: round(float(prediction.probabilities[label]), 10)
            for label in SEASON_LABELS
        },
        "probability_order": list(SEASON_LABELS),
        "confidence": round(float(prediction.confidence), 10),
        "review_required": None,
        "latency_measured": True,
        "run_id": prediction.run_id,
        "model_manifest_sha256": prediction.manifest_sha256,
        "bundle_sha256": prediction.bundle_sha256,
        "holdout_opened": False,
    }


def _validate_smoke_payload(
    smoke: Mapping[str, Any],
    *,
    refit: Mapping[str, Any],
    resolved_refit: Path,
    root: Path,
) -> None:
    if set(smoke) != _SMOKE_FIELDS:
        raise ValueError("Task 2 inference smoke fields changed")
    if (
        smoke["schema_version"] != HANDOFF_SCHEMA_VERSION
        or smoke["input_scope"] != "one labelled development image"
        or smoke["probability_order"] != list(SEASON_LABELS)
        or smoke["review_required"] is not None
        or smoke["latency_measured"] is not True
        or smoke["run_id"] != refit["run_id"]
        or smoke["model_manifest_sha256"] != compute_sha256(resolved_refit)
        or smoke["bundle_sha256"] != refit["bundle"]["sha256"]
        or smoke["holdout_opened"] is not False
    ):
        raise ValueError("Task 2 inference smoke prediction changed its frozen contract")
    probabilities = smoke["probabilities"]
    if not isinstance(probabilities, Mapping):
        raise ValueError("Task 2 inference smoke prediction changed its frozen contract")
    _validate_probability_contract(
        probabilities,
        predicted_label=smoke["predicted_label"],
        confidence=smoke["confidence"],
    )
    _validate_smoke_image(
        smoke["image_path"],
        refit=refit,
        root=root,
        expected_product_id=str(smoke["product_id"]),
    )


def _lock_owner_is_alive(path: Path) -> bool:
    try:
        owner = _load_json_object(path, "Task 2 handoff build lock")
        process_id = int(owner["pid"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    return psutil.pid_exists(process_id)


@contextmanager
def _exclusive_handoff_lock(path: Path) -> Iterator[None]:
    """Serialise package verification, staging, and publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid()}
    acquired = False
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_owner_is_alive(path):
                raise RuntimeError(
                    f"Task 2 handoff build is already running; lock={path}"
                ) from None
            path.unlink(missing_ok=True)
            continue
        try:
            with os.fdopen(descriptor, mode="w", encoding="utf-8") as handle:
                json.dump(owner, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        acquired = True
        break
    if not acquired:
        raise RuntimeError(f"could not acquire Task 2 handoff build lock: {path}")
    try:
        yield
    finally:
        try:
            current_owner = _load_json_object(path, "Task 2 handoff build lock")
        except (json.JSONDecodeError, OSError, ValueError):
            current_owner = None
        if current_owner == owner:
            path.unlink(missing_ok=True)


def _publish_transaction(staged_to_final: Mapping[Path, Path]) -> None:
    """Publish children before the manifest and restore every old file on failure."""
    if not staged_to_final:
        raise ValueError("Task 2 handoff transaction is empty")
    first_stage = next(iter(staged_to_final))
    backup = first_stage.parent / "backup"
    backup.mkdir()
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for index, final in enumerate(staged_to_final.values()):
            if final.exists():
                saved = backup / f"{index}-{final.name}"
                os.replace(final, saved)
                backups[final] = saved
        for staged, final in staged_to_final.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            published.append(final)
    except BaseException:
        for final in reversed(published):
            final.unlink(missing_ok=True)
        for final, saved in backups.items():
            if saved.exists():
                os.replace(saved, final)
        raise


def _legacy_upgrade_allowed(
    existing: Mapping[str, Any],
    desired: Mapping[str, Any],
    *,
    root: Path,
) -> bool:
    """Allow the one unsafe v1 package created before immutability was enforced."""
    old_artifacts = existing.get("artifacts")
    if not isinstance(old_artifacts, Mapping):
        return False
    required_old = {
        "artifact_audit",
        "inference_smoke",
        "selection_freeze",
        "model_manifest",
        "model_bundle",
        "inference_source",
    }
    identity_matches = (
        existing.get("schema_version") == "1.0.0"
        and existing.get("gate") == "TASK2-COMPONENT-HANDOFF"
        and existing.get("status") == "ready_for_group_freeze"
        and existing.get("task2_component_ready") is True
        and existing.get("group_freeze_verified") is False
        and existing.get("notebook_06_unlocked") is False
        and existing.get("holdout_opened") is False
        and existing.get("evaluation_claim_allowed") is False
        and existing.get("model_change_allowed") is False
        and existing.get("run_id") == desired.get("run_id")
        and existing.get("selected_candidate") == desired.get("selected_candidate")
        and existing.get("selected_experiment_id")
        == desired.get("selected_experiment_id")
        and set(old_artifacts) == required_old
    )
    if not identity_matches:
        return False
    for name, declaration in old_artifacts.items():
        _declared_path(declaration, root=root, scope=f"legacy handoff {name}")
    for name in ("selection_freeze", "model_manifest", "model_bundle", "inference_source"):
        if old_artifacts[name] != desired["artifacts"][name]:
            return False
    return True


def _fixed_manifest_fields() -> dict[str, Any]:
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "gate": "TASK2-COMPONENT-HANDOFF",
        "status": "ready_for_group_freeze",
        "task2_component_ready": True,
        "group_freeze_verified": False,
        "notebook_06_unlocked": False,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "evaluation_claim_allowed": False,
        "model_change_allowed": False,
        "labels": list(SEASON_LABELS),
        "inference_inputs": ["image"],
        "review_threshold": None,
        "next_gate": "whole-group freeze before Notebook 06",
    }


def build_task2_handoff_evidence(
    artifact_audit: pd.DataFrame,
    prediction: SeasonPrediction,
    *,
    project_root: str | Path = ROOT,
    registry_path: str | Path = RUNS_CSV,
    ultimate_manifest_path: str | Path = DEFAULT_ULTIMATE_MANIFEST,
    model_manifest_path: str | Path = TASK2_MODEL_MANIFEST_JSON,
    output_directory: str | Path = DEFAULT_HANDOFF_DIRECTORY,
) -> tuple[dict[str, Any], Path]:
    """Write an immutable Task 2 handoff while keeping Notebook 06 locked."""
    root = Path(project_root).resolve()
    ultimate_manifest_path = _rebase_project_default(
        ultimate_manifest_path,
        default=DEFAULT_ULTIMATE_MANIFEST,
        root=root,
    )
    output = _resolve_within_root(
        output_directory,
        root=root,
        scope="Task 2 handoff directory",
    )
    lock_path = output / HANDOFF_LOCK_FILENAME
    with _exclusive_handoff_lock(lock_path):
        recomputed_live_audit = audit_task2_artifacts(
            project_root=root,
            registry_path=registry_path,
            ultimate_manifest_path=ultimate_manifest_path,
            model_manifest_path=model_manifest_path,
        )
        _assert_same_audit(
            artifact_audit,
            recomputed_live_audit,
            scope="Task 2 caller",
        )
        refit, resolved_refit, _ = load_verified_development_refit_manifest(
            model_manifest_path,
            project_root=root,
            registry_path=registry_path,
        )
        smoke = _smoke_from_prediction(
            prediction,
            refit=refit,
            resolved_refit=resolved_refit,
            root=root,
        )
        registry_frame, _ = _registry_snapshot(refit, registry_path, root=root)
        freeze, freeze_path = load_verified_selection_freeze(
            root / str(refit["selection_freeze"]["path"]),
            project_root=root,
        )

        audit_path = output / "artifact_audit.csv"
        smoke_path = output / "inference_smoke.json"
        snapshot_path = output / REGISTRY_SNAPSHOT_FILENAME
        manifest_path = output / "manifest.json"
        staging_root = root / "tmp/task2/handoff"
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            with TemporaryDirectory(prefix="build-", dir=staging_root) as directory:
                staging = Path(directory)
                staged_snapshot = staging / REGISTRY_SNAPSHOT_FILENAME
                staged_audit = staging / "artifact_audit.csv"
                staged_smoke = staging / "inference_smoke.json"
                staged_manifest = staging / "manifest.json"
                atomic_write_csv(staged_snapshot, registry_frame)
                snapshot_sha256 = compute_sha256(staged_snapshot)
                packaged_audit = recomputed_live_audit.copy()
                registry_index = packaged_audit.index[
                    packaged_audit["artifact"].eq("registry_binding")
                ].item()
                packaged_audit.loc[
                    registry_index,
                    ["path", "expected_sha256", "actual_sha256"],
                ] = [
                    snapshot_path.relative_to(root).as_posix(),
                    snapshot_sha256,
                    snapshot_sha256,
                ]
                _validate_artifact_audit(packaged_audit)
                atomic_write_csv(staged_audit, packaged_audit)
                atomic_write_json(staged_smoke, smoke)
                ultimate_row = packaged_audit.loc[
                    packaged_audit["artifact"].eq("ultimate_judgement_manifest")
                ].iloc[0]
                resolved_ultimate = _resolve_within_root(
                    str(ultimate_row["path"]),
                    root=root,
                    scope="ultimate judgement manifest",
                )
                manifest = {
                    **_fixed_manifest_fields(),
                    "selected_candidate": refit["selected_candidate"],
                    "selected_experiment_id": refit["selected_experiment_id"],
                    "run_id": refit["run_id"],
                    "artifacts": {
                        "artifact_audit": _staged_declaration(
                            audit_path, staged_audit, root=root
                        ),
                        "inference_smoke": _staged_declaration(
                            smoke_path, staged_smoke, root=root
                        ),
                        "registry_snapshot": _staged_declaration(
                            snapshot_path, staged_snapshot, root=root
                        ),
                        "ultimate_judgement_manifest": _declaration(
                            resolved_ultimate,
                            root=root,
                        ),
                        "selection_freeze": _declaration(freeze_path, root=root),
                        "model_manifest": _declaration(resolved_refit, root=root),
                        "model_bundle": _declaration(
                            root / str(refit["bundle"]["path"]), root=root
                        ),
                        "inference_source": _declaration(
                            root / "src/fashion/task2/inference.py", root=root
                        ),
                    },
                    "unresolved_risks": list(freeze["limitations"]),
                }
                atomic_write_json(staged_manifest, manifest)

                if manifest_path.exists():
                    existing = _load_json_object(manifest_path, "Task 2 handoff manifest")
                    if existing.get("schema_version") == HANDOFF_SCHEMA_VERSION:
                        load_verified_task2_handoff(manifest_path, project_root=root)
                        if canonical_sha256(existing) != canonical_sha256(manifest):
                            raise ValueError(
                                "Task 2 handoff already exists with different content"
                            )
                        return existing, manifest_path
                    if not _legacy_upgrade_allowed(existing, manifest, root=root):
                        raise ValueError(
                            "Task 2 handoff already exists with different content"
                        )
                else:
                    orphaned = [
                        path
                        for path in (audit_path, smoke_path, snapshot_path)
                        if path.exists()
                    ]
                    if orphaned:
                        raise FileExistsError(
                            "unmanifested Task 2 handoff artifacts exist: "
                            + ", ".join(str(path) for path in orphaned)
                        )

                _publish_transaction(
                    {
                        staged_snapshot: snapshot_path,
                        staged_audit: audit_path,
                        staged_smoke: smoke_path,
                        staged_manifest: manifest_path,
                    }
                )
        finally:
            try:
                staging_root.rmdir()
                staging_root.parent.rmdir()
            except OSError:
                pass
        verified, verified_path, _, _ = load_verified_task2_handoff(
            manifest_path,
            project_root=root,
        )
        return verified, verified_path


def load_verified_task2_handoff(
    path: str | Path = DEFAULT_HANDOFF_DIRECTORY / "manifest.json",
    *,
    project_root: str | Path = ROOT,
    registry_path: str | Path = RUNS_CSV,
) -> tuple[dict[str, Any], Path, pd.DataFrame, dict[str, Any]]:
    """Verify the portable Task 2 handoff and keep final evaluation locked."""
    del registry_path  # Kept only for call-site compatibility; v1.1 uses its snapshot.
    root = Path(project_root).resolve()
    resolved = _resolve_within_root(path, root=root, scope="Task 2 handoff manifest")
    manifest = _load_json_object(resolved, "Task 2 handoff manifest")
    fixed = _fixed_manifest_fields()
    changed = [name for name, expected in fixed.items() if manifest.get(name) != expected]
    if changed:
        raise ValueError(f"Task 2 handoff boundary changed: {changed}")
    required_fields = set(fixed) | {
        "selected_candidate",
        "selected_experiment_id",
        "run_id",
        "artifacts",
        "unresolved_risks",
    }
    if set(manifest) != required_fields:
        raise ValueError("Task 2 handoff fields changed")
    artifacts = manifest["artifacts"]
    required_artifacts = {
        "artifact_audit",
        "inference_smoke",
        "registry_snapshot",
        "ultimate_judgement_manifest",
        "selection_freeze",
        "model_manifest",
        "model_bundle",
        "inference_source",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_artifacts:
        raise ValueError("Task 2 handoff artifact set changed")
    verified = {
        name: _declared_path(declaration, root=root, scope=f"handoff {name}")
        for name, declaration in artifacts.items()
    }
    refit, resolved_refit, _ = _load_verified_development_refit_package(
        verified["model_manifest"],
        project_root=root,
    )
    _verify_refit_registry(
        refit,
        registry_path=verified["registry_snapshot"],
        root=root,
    )
    freeze, freeze_path = load_verified_selection_freeze(
        verified["selection_freeze"],
        project_root=root,
    )
    expected_artifacts = {
        "selection_freeze": _declaration(freeze_path, root=root),
        "model_manifest": _declaration(resolved_refit, root=root),
        "model_bundle": dict(refit["bundle"]),
        "inference_source": _declaration(
            root / "src/fashion/task2/inference.py", root=root
        ),
    }
    artifact_mismatches = [
        name for name, expected in expected_artifacts.items() if artifacts[name] != expected
    ]
    if (
        artifact_mismatches
        or manifest["selected_candidate"] != refit["selected_candidate"]
        or manifest["selected_experiment_id"] != refit["selected_experiment_id"]
        or manifest["run_id"] != refit["run_id"]
        or manifest["unresolved_risks"] != list(freeze["limitations"])
    ):
        raise ValueError("Task 2 handoff no longer matches the verified refit")

    audit = pd.read_csv(verified["artifact_audit"], dtype="string").fillna("")
    recomputed_audit = audit_task2_artifacts(
        project_root=root,
        registry_snapshot_path=verified["registry_snapshot"],
        ultimate_manifest_path=verified["ultimate_judgement_manifest"],
        model_manifest_path=verified["model_manifest"],
    )
    _assert_same_audit(audit, recomputed_audit, scope="Task 2 stored")
    smoke = _load_json_object(verified["inference_smoke"], "Task 2 inference smoke")
    _validate_smoke_payload(
        smoke,
        refit=refit,
        resolved_refit=resolved_refit,
        root=root,
    )
    return manifest, resolved, audit, smoke


__all__ = [
    "DEFAULT_HANDOFF_DIRECTORY",
    "HANDOFF_LOCK_FILENAME",
    "audit_task2_artifacts",
    "build_task2_handoff_evidence",
    "load_verified_task2_handoff",
]
