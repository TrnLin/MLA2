"""The stability-evidence recovery route must stay provenance-safe and never retrain.

A stability run that trained every bound milestone and then died inside the evidence step
can be completed from its persisted checkpoints. Eligibility is deliberately narrow: the
exact planned run ID, the exact recoverable error identity, a complete identity-validated
milestone set, and no evidence already recorded on the row.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import fashion.task4.learned_evidence as learned
import fashion.task4.training as training
from fashion.train.registry import Task4RunRegistry as RunRegistry
from tests.task4.test_stability_evidence_coverage import (  # reuse the frozen fold fixture
    _StabilityFixture,
    _undefined_query_splits,
)

ROOT = Path(__file__).resolve().parents[2]
RECOVERABLE_ERROR_TYPE = "ValueError"
RECOVERABLE_ERROR_MESSAGE = "stability evidence primary coverage is incomplete"
ATTEMPT_TOKEN = "task9-stability-r3-a1"


def _load_runner() -> object:
    spec = importlib.util.spec_from_file_location(
        "run_model_comparisons_recovery",
        ROOT / "scripts/task4/run_model_comparisons.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _budget_gate(runner: object, tmp_path: Path) -> Path:
    gate = tmp_path / "gate.json"
    runner.write_json_atomic(
        gate,
        {
            "schema_version": 1,
            "gate": "task4_model_comparison_budget",
            "decision": "passed",
            "fits_budget": True,
            "selected_estimated_full_matrix_gpu_hours": 80.0,
            "budget_gpu_hours": 98.0,
            "split_fingerprint": "c" * 64,
            "source_artifacts": [],
        },
    )
    return gate


class _RecoveryScenario:
    """A finalist candidate plus one failed fold-0 stability run on its plan."""

    def __init__(
        self,
        runner: object,
        tmp_path: Path,
        *,
        error_message: str = RECOVERABLE_ERROR_MESSAGE,
        evidence_manifest_path: str = "",
        stability_run_kind: str = "stability",
        complete_the_failed_row: bool = False,
    ) -> None:
        self.runner = runner
        self.tmp_path = tmp_path
        self.registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
        self.artifacts: dict[str, object] = {}
        for candidate, score in (("R5", 0.5), ("R1", 0.3)):
            config = runner._candidate_config(candidate)
            session = config.training_session(
                run_id=f"{candidate.lower()}-candidate",
                split_fingerprint="c" * 64,
            )
            self.registry.append(runner._base_registry_row(session))
            checkpoint_path = (
                tmp_path
                / f"results/evidence/task4/checkpoints/{session.run_id}/epoch-100.pt"
            )
            manifest_path = (
                tmp_path
                / f"results/evidence/task4/learned/{session.run_id}/manifest.json"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")
            manifest_path.write_text("{}\n", encoding="utf-8")
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            self.registry.update(
                session.run_id,
                {
                    "status": "completed",
                    "selected_epoch": 100,
                    "development_winner_score": score,
                    "checkpoint_path": str(checkpoint_path.relative_to(tmp_path)),
                    "checkpoint_sha256": checkpoint_sha256,
                    "evidence_manifest_path": str(manifest_path.relative_to(tmp_path)),
                    "completed_at_utc": "2099-08-31T00:00:00Z",
                },
            )
            artifact_path = runner.write_experiment_config_artifact(
                manifest_path.with_name("experiment_config.json"),
                session=session,
                checkpoint_sha256=checkpoint_sha256,
            )
            self.artifacts[session.run_id] = runner._load_experiment_config_artifact(
                artifact_path
            )
        self.finalist_artifact = self.artifacts["r5-candidate"]
        self.finalist = runner.StabilityFinalist(
            identity=self.finalist_artifact.identity,
            candidate_score=0.5,
            config_artifact=self.finalist_artifact,
        )
        self.plan = runner.build_stability_plan(
            self.finalist,
            attempt_token=ATTEMPT_TOKEN,
        )[0]
        self.session = self.plan.training_session()
        self.registered_session = (
            self.session
            if stability_run_kind == "stability"
            else runner._candidate_config(self.plan.method).training_session(
                run_id=self.plan.run_id,
                split_fingerprint="c" * 64,
            )
        )
        self.checkpoints = [
            tmp_path
            / "results/evidence/task4/checkpoints"
            / self.session.run_id
            / f"epoch-{epoch:03d}.pt"
            for epoch in (20, 40, 60, 80, 100)
        ]
        for checkpoint in self.checkpoints:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
        stability_row = runner._base_registry_row(self.registered_session)
        self.registry.append(stability_row)
        if complete_the_failed_row:
            manifest = (
                tmp_path
                / f"results/evidence/task4/stability/{self.session.run_id}/stability_evidence.json"
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}\n", encoding="utf-8")
            self.registry.update(
                self.session.run_id,
                {
                    "status": "completed",
                    "completed_at_utc": "2099-08-31T00:00:00Z",
                    "selected_epoch": 100,
                    "development_winner_score": 0.42,
                    "checkpoint_path": str(self.checkpoints[-1].relative_to(tmp_path)),
                    "checkpoint_sha256": hashlib.sha256(
                        self.checkpoints[-1].read_bytes()
                    ).hexdigest(),
                    "evidence_manifest_path": str(manifest.relative_to(tmp_path)),
                },
            )
        else:
            self.registry.update(
                self.session.run_id,
                {
                    "status": "failed",
                    "completed_at_utc": stability_row["started_at_utc"],
                    "error_type": RECOVERABLE_ERROR_TYPE,
                    "error_message": error_message,
                    **(
                        {"evidence_manifest_path": evidence_manifest_path}
                        if evidence_manifest_path
                        else {}
                    ),
                },
            )

    def write_request(self, *, run_id: str | None = None) -> Path:
        request = self.tmp_path / "stability-recovery-request.json"
        self.runner.write_canonical_stability_evidence_recovery_request(
            request,
            candidate=self.plan.method,
            run_id=self.plan.run_id if run_id is None else run_id,
            fold=self.plan.fold,
            attempt_token=self.plan.attempt_token,
            parent_run_id=self.plan.parent_run_id,
            checkpoint_paths=self.checkpoints,
            variant_index_path=Path(
                "data/processed/task4/external_variant_index.csv.gz"
            ),
            cache_root=Path("results/cache/task4/images"),
            checkpoint_root=Path("results/evidence/task4/checkpoints"),
            evidence_root=Path("results/evidence/task4"),
            feature_cache_root=Path("results/cache/task4/features"),
        )
        return request

    def patch_environment(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        runner = self.runner
        calls: list[str] = []
        monkeypatch.setattr(
            runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []})
        )
        monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
        monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
        monkeypatch.setattr(
            runner, "load_config_artifacts", lambda root: dict(self.artifacts)
        )
        monkeypatch.setattr(
            runner,
            "_verify_candidate_dry_run_request",
            lambda request, *, splits: {"status": "passed", "opened_pixels": 0},
        )
        monkeypatch.setattr(
            runner,
            "reconstruct_training_result",
            lambda *args, **kwargs: calls.append("reconstruct")
            or type(
                "Result",
                (),
                {
                    "checkpoints": [
                        type("Checkpoint", (), {"epoch": epoch, "path": path})()
                        for epoch, path in zip(
                            (20, 40, 60, 80, 100), self.checkpoints, strict=True
                        )
                    ],
                    "best_checkpoint": type("Checkpoint", (), {"sha256": "a" * 64})(),
                },
            )(),
        )
        monkeypatch.setattr(
            runner,
            "_run_training_and_evidence",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("stability recovery must not train")
            ),
        )
        return calls

    def run(self, monkeypatch: pytest.MonkeyPatch, *, request: Path) -> tuple[int, dict]:
        gate = _budget_gate(self.runner, self.tmp_path)
        output = self.tmp_path / "stability-recovery-result.json"
        code = self.runner.main(
            [
                "--phase",
                "stability-evidence-recovery",
                "--dry-run",
                "--registry",
                str(self.registry.path),
                "--budget-gate",
                str(gate),
                "--phase-output",
                str(output),
                "--phase-request",
                str(request),
            ]
        )
        return code, json.loads(output.read_text(encoding="utf-8"))


def test_stability_recovery_dry_run_opens_no_pixels_and_forbids_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(runner, tmp_path)
    request = scenario.write_request()
    calls = scenario.patch_environment(monkeypatch)

    code, package = scenario.run(monkeypatch, request=request)

    assert code == 0
    assert calls == ["reconstruct"]
    result = package["phase_result"]
    assert result["status"] == "ready"
    assert result["mode"] == "stability_evidence_recovery"
    assert result["run_id"] == scenario.session.run_id
    assert result["fold"] == 0
    assert result["opened_pixels"] == 0
    assert result["training_call"] == "forbidden"
    assert result["evidence_scope"] == "lightweight_primary_score_and_coverage"
    assert result["recoverable_error"] == {
        "type": RECOVERABLE_ERROR_TYPE,
        "message": RECOVERABLE_ERROR_MESSAGE,
    }
    assert result["checkpoint_epochs"] == [20, 40, 60, 80, 100]


def test_stability_recovery_rejects_a_run_id_outside_the_finalist_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(runner, tmp_path)
    request = scenario.write_request(run_id="r5-candidate-stability-forged-fold-0")
    calls = scenario.patch_environment(monkeypatch)

    code, package = scenario.run(monkeypatch, request=request)

    assert code == 0
    assert calls == []
    assert package["phase_result"]["status"] == "pending"
    assert "run_id does not match its fold plan" in package["phase_result"]["reason"]


def test_stability_recovery_rejects_a_different_failure_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(
        runner,
        tmp_path,
        error_message="CUDA out of memory",
    )
    request = scenario.write_request()
    calls = scenario.patch_environment(monkeypatch)

    code, package = scenario.run(monkeypatch, request=request)

    assert code == 0
    assert calls == []
    assert package["phase_result"]["status"] == "pending"
    assert (
        "exact failed stability evidence row" in package["phase_result"]["reason"]
    )


def test_stability_recovery_rejects_a_row_that_already_recorded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(
        runner,
        tmp_path,
        evidence_manifest_path="results/evidence/task4/learned/r5-candidate/manifest.json",
    )
    request = scenario.write_request()
    calls = scenario.patch_environment(monkeypatch)

    code, package = scenario.run(monkeypatch, request=request)

    assert code == 0
    assert calls == []
    assert package["phase_result"]["status"] == "pending"
    assert "no completed stability evidence" in package["phase_result"]["reason"]


def test_stability_recovery_refuses_a_row_that_already_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(runner, tmp_path, complete_the_failed_row=True)
    request = scenario.write_request()
    calls = scenario.patch_environment(monkeypatch)

    code, package = scenario.run(monkeypatch, request=request)

    assert code == 0
    assert calls == []
    assert package["phase_result"]["status"] == "pending"
    assert "exact failed stability evidence row" in package["phase_result"]["reason"]


def test_stability_recovery_refuses_a_row_that_is_not_a_stability_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(runner, tmp_path, stability_run_kind="candidate")
    request = scenario.write_request()
    calls = scenario.patch_environment(monkeypatch)

    code, package = scenario.run(monkeypatch, request=request)

    assert code == 0
    assert scenario.registered_session.run_kind == "candidate"
    assert calls == []
    assert package["phase_result"]["status"] == "pending"
    assert "exact failed stability evidence row" in package["phase_result"]["reason"]


def test_stability_recovery_refuses_an_incomplete_milestone_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(runner, tmp_path)
    request = scenario.write_request()
    calls = scenario.patch_environment(monkeypatch)
    scenario.checkpoints[2].unlink()

    code, package = scenario.run(monkeypatch, request=request)

    assert code == 0
    assert calls == []
    assert package["phase_result"]["status"] == "pending"
    assert "checkpoint is missing" in package["phase_result"]["reason"]


def test_stability_recovery_failure_audit_cannot_be_rewritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(runner, tmp_path)
    request = runner.load_stability_evidence_recovery_request(scenario.write_request())
    row = next(
        item
        for item in scenario.registry.read()
        if item["run_id"] == scenario.session.run_id
    )
    result = type(
        "Result",
        (),
        {
            "checkpoints": [
                type("Checkpoint", (), {"epoch": epoch, "path": path})()
                for epoch, path in zip(
                    (20, 40, 60, 80, 100), scenario.checkpoints, strict=True
                )
            ],
            "best_checkpoint": type("Checkpoint", (), {"sha256": "a" * 64})(),
        },
    )()
    audit = tmp_path / "audit.json"

    runner._write_stability_recovery_audit(
        audit, request=request, registry_row=row, result=result
    )
    first = json.loads(audit.read_text(encoding="utf-8"))
    runner._write_stability_recovery_audit(
        audit, request=request, registry_row=row, result=result
    )

    forged = dict(row)
    forged["error_message"] = "CUDA out of memory"
    with pytest.raises(ValueError, match="immutable"):
        runner._write_stability_recovery_audit(
            audit, request=request, registry_row=forged, result=result
        )
    assert json.loads(audit.read_text(encoding="utf-8")) == first


def test_stability_recovery_writes_the_audit_then_recovers_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    scenario = _RecoveryScenario(runner, tmp_path)
    request = scenario.write_request()
    calls = scenario.patch_environment(monkeypatch)
    manifest = (
        tmp_path
        / f"results/evidence/task4/stability/{scenario.session.run_id}/stability_evidence.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        runner, "_candidate_data_context", lambda **kwargs: {
            "caches": {}, "statistics": {}, "statistics_paths": {},
        }
    )
    monkeypatch.setattr(
        runner,
        "pin_single_gpu",
        lambda index: calls.append(f"pin-{index}") or "cuda:0",
    )

    def _build(registry_argument: object, **kwargs: object) -> object:
        captured.update(kwargs)
        captured["audit_exists_before_recovery"] = (
            tmp_path
            / "results/evidence/task4/phase_results"
            / f"{scenario.session.run_id}-stability-recovery-failure-attempt.json"
        ).is_file()
        calls.append("stability-evidence")
        return type(
            "Evidence",
            (),
            {
                "manifest_path": manifest,
                "development_winner_score": 0.25,
                "total_query_count": 10,
                "scorable_query_count": 9,
                "primary_coverage": 0.9,
            },
        )()

    monkeypatch.setattr(runner, "build_stability_evidence", _build)
    monkeypatch.setattr(
        runner,
        "write_experiment_config_artifact",
        lambda path, **kwargs: Path(path),
    )

    output = tmp_path / "results/evidence/task4/phase_results/recovery-result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    code = runner.main(
        [
            "--phase",
            "stability-evidence-recovery",
            "--gpu",
            "1",
            "--registry",
            str(scenario.registry.path),
            "--budget-gate",
            str(_budget_gate(runner, tmp_path)),
            "--phase-output",
            str(output),
            "--phase-request",
            str(request),
        ]
    )

    assert code == 0
    assert calls == ["reconstruct", "pin-1", "stability-evidence"]
    assert captured["audit_exists_before_recovery"] is True
    assert captured["recover_failed_evidence"] is True
    assert captured["recovery_error_type"] == RECOVERABLE_ERROR_TYPE
    assert captured["recovery_error_message"] == RECOVERABLE_ERROR_MESSAGE
    assert str(captured["completed_at"]).endswith("Z")
    package = json.loads(output.read_text(encoding="utf-8"))
    assert package["phase_result"]["mode"] == "stability_evidence_recovery"
    assert package["phase_result"]["primary_coverage"] == 0.9


def test_stability_recovery_completes_the_same_failed_row_from_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    fixture.registry.update(
        fixture.session.run_id,
        {
            "status": "failed",
            "completed_at_utc": "2026-08-31T13:55:31Z",
            "error_type": RECOVERABLE_ERROR_TYPE,
            "error_message": RECOVERABLE_ERROR_MESSAGE,
        },
    )
    for entry_point in ("train_epochs", "run_training_attempt"):
        monkeypatch.setattr(
            training,
            entry_point,
            lambda *args, **kwargs: pytest.fail("stability recovery must not train"),
        )

    built = learned.build_stability_evidence(
        fixture.registry,
        result=fixture.result,
        session=fixture.session,
        splits=fixture.splits,
        caches=fixture.caches,
        statistics=fixture.statistics,
        statistics_paths=fixture.statistics_paths,
        feature_cache_root=fixture.feature_cache_root,
        evidence_root=fixture.evidence_root,
        completed_at="2026-08-31T19:00:00Z",
        recover_failed_evidence=True,
        recovery_error_type=RECOVERABLE_ERROR_TYPE,
        recovery_error_message=RECOVERABLE_ERROR_MESSAGE,
    )

    assert built.registry_row["status"] == "completed"
    assert built.registry_row["error_type"] == RECOVERABLE_ERROR_TYPE
    assert built.registry_row["error_message"] == RECOVERABLE_ERROR_MESSAGE
    assert built.scorable_query_count < built.total_query_count
    reopened = learned.validate_stability_evidence_artifact(
        built.manifest_path,
        session=fixture.session,
        checkpoint=fixture.result.best_checkpoint,
        canonical_splits=fixture.splits,
    )
    assert reopened["identity"]["run_id"] == fixture.session.run_id


def test_stability_recovery_refuses_to_complete_a_different_failure_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    fixture.registry.update(
        fixture.session.run_id,
        {
            "status": "failed",
            "completed_at_utc": "2026-08-31T13:55:31Z",
            "error_type": RECOVERABLE_ERROR_TYPE,
            "error_message": "CUDA out of memory",
        },
    )

    with pytest.raises(ValueError, match="exact failed evidence row"):
        learned.build_stability_evidence(
            fixture.registry,
            result=fixture.result,
            session=fixture.session,
            splits=fixture.splits,
            caches=fixture.caches,
            statistics=fixture.statistics,
            statistics_paths=fixture.statistics_paths,
            feature_cache_root=fixture.feature_cache_root,
            evidence_root=fixture.evidence_root,
            completed_at="2026-08-31T19:00:00Z",
            recover_failed_evidence=True,
            recovery_error_type=RECOVERABLE_ERROR_TYPE,
            recovery_error_message=RECOVERABLE_ERROR_MESSAGE,
        )


def test_stability_recovery_request_requires_the_exact_error_identity() -> None:
    runner = _load_runner()
    payload = {
        "schema_version": 1,
        "artifact_type": "task4_stability_evidence_recovery_request",
        "phase": "stability-evidence-recovery",
        "candidate": "R3",
        "run_id": "r3-candidate-stability-token-fold-0",
        "fold": 0,
        "attempt_token": "token",
        "parent_run_id": "r3-candidate",
        "paths": {
            "variant_index_path": "data/processed/task4/external_variant_index.csv.gz",
            "cache_root": "results/cache/task4/images",
            "checkpoint_root": "results/evidence/task4/checkpoints",
            "evidence_root": "results/evidence/task4",
            "feature_cache_root": "results/cache/task4/features",
        },
        "checkpoint_paths": [
            f"results/evidence/task4/checkpoints/r3/epoch-{epoch:03d}.pt"
            for epoch in (20, 40, 60, 80, 100)
        ],
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": 98.0,
        },
        "recoverable_error": {
            "type": RECOVERABLE_ERROR_TYPE,
            "message": RECOVERABLE_ERROR_MESSAGE,
        },
        "evidence_scope": "lightweight_primary_score_and_coverage",
    }

    assert runner.validate_stability_evidence_recovery_request_payload(payload)

    wrong_error = json.loads(json.dumps(payload))
    wrong_error["recoverable_error"]["message"] = "something else"
    with pytest.raises(ValueError, match="failed-attempt identity"):
        runner.validate_stability_evidence_recovery_request_payload(wrong_error)

    wrong_scope = json.loads(json.dumps(payload))
    wrong_scope["evidence_scope"] = "full"
    with pytest.raises(ValueError, match="evidence scope"):
        runner.validate_stability_evidence_recovery_request_payload(wrong_scope)

    short_checkpoints = json.loads(json.dumps(payload))
    short_checkpoints["checkpoint_paths"] = short_checkpoints["checkpoint_paths"][:4]
    with pytest.raises(ValueError, match="five checkpoint paths"):
        runner.validate_stability_evidence_recovery_request_payload(short_checkpoints)
