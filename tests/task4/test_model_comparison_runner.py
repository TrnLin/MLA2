from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from fashion.train.registry import RunRegistry

RUNNER_PATH = Path(__file__).parents[2] / "scripts/task4/run_model_comparisons.py"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("task4_model_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("runner module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_unknown_phase() -> None:
    runner = _load_runner()

    with pytest.raises(SystemExit):
        runner.main(["--phase", "holdout"])


def test_runner_exposes_no_placeholder_downstream_hooks() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner)

    assert not hasattr(runner, "build_canonical_training_inputs")
    assert not hasattr(runner, "produce_task6_evidence")
    assert not hasattr(runner, "validate_development_evidence_summary")
    assert "must be supplied by Task 9" not in source


def test_cli_accepts_typed_phase_request_json(tmp_path: Path) -> None:
    runner = _load_runner()
    request = tmp_path / "request.json"
    runner.write_canonical_candidate_phase_request(
        request,
        candidate="R1",
        run_id="candidate-r1",
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )

    args = runner._parser().parse_args(
        ["--phase", "candidate", "--phase-request", str(request)]
    )

    assert args.phase_request == request
    payload = json.loads(request.read_text(encoding="utf-8"))
    assert payload["budget_gate"] == {
        "gate": "task4_model_comparison_budget",
        "budget_gpu_hours": 98.0,
    }
    assert payload["hyperparameters"]["amp_growth_interval"] == 40901


def test_phase_request_rejects_noncanonical_or_opaque_fields(tmp_path: Path) -> None:
    runner = _load_runner()
    request = {
        "schema_version": 1,
        "artifact_type": "task4_phase_request",
        "phase": "candidate",
        "candidate": "R1",
        "run_id": "candidate-r1",
        "paths": {
            "variant_index_path": "data/processed/task4/external_variant_index.csv.gz",
            "cache_root": "results/cache/task4/images",
            "checkpoint_root": "results/evidence/task4/checkpoints",
            "evidence_root": "results/evidence/task4",
            "feature_cache_root": "results/cache/task4/features",
        },
        "budget_gate": {
            "gate": "task4_model_comparison_budget",
            "budget_gpu_hours": 98.0,
        },
        "hyperparameters": {
            "seed": 2753,
            "product_batch_size": 64,
            "amp_growth_interval": 40901,
            "planned_epochs": 100,
            "checkpoint_epochs": [20, 40, 60, 80, 100],
        },
    }
    request_path = tmp_path / "request.json"
    runner.write_json_atomic(request_path, request)

    validated = runner.load_phase_request(request_path, phase="candidate")

    assert validated["candidate"] == "R1"
    assert validated["variant_index_path"] == Path(
        "data/processed/task4/external_variant_index.csv.gz"
    )
    assert validated["hyperparameters"]["checkpoint_epochs"] == [20, 40, 60, 80, 100]

    bad = copy.deepcopy(request)
    bad["opaque_token"] = "human-invented"
    bad_path = tmp_path / "bad-request.json"
    runner.write_json_atomic(bad_path, bad)
    with pytest.raises(ValueError, match="schema"):
        runner.load_phase_request(bad_path, phase="candidate")


def test_gallery_request_round_trip_carries_exact_deployment_evidence_inputs(
    tmp_path: Path,
) -> None:
    runner = _load_runner()

    def manifest_spec(run_id: str, *, stability: bool = False) -> dict[str, object]:
        area = "stability" if stability else "learned"
        return {
            "run_id": run_id,
            "config_artifact_path": (
                f"results/evidence/task4/{area}/{run_id}/experiment_config.json"
            ),
            "checkpoint_path": f"results/evidence/task4/checkpoints/{run_id}/epoch-020.pt",
            "manifest_path": (
                f"results/evidence/task4/{area}/{run_id}/"
                f"{'stability_evidence.json' if stability else 'manifest.json'}"
            ),
        }

    deployment_inputs = []
    for method in ("R5", "R3"):
        candidate_run_id = f"{method.lower()}-candidate"
        stability_rows = [
            {
                "run_id": f"{candidate_run_id}-fold-{fold}",
                "method": method,
                "run_kind": "stability",
                "status": "completed",
                "fold": str(fold),
            }
            for fold in range(5)
        ]
        deployment_inputs.append(
            {
                "registry_row": {
                    "run_id": candidate_run_id,
                    "method": method,
                    "run_kind": "candidate",
                    "status": "completed",
                    "fold": "1",
                },
                "candidate_score": 0.5,
                "finalist_config_artifact_path": (
                    f"results/evidence/task4/learned/{candidate_run_id}/experiment_config.json"
                ),
                "stability_rows": stability_rows,
                "stability_manifests": [
                    manifest_spec(str(row["run_id"]), stability=True) for row in stability_rows
                ],
                "candidate_manifest": manifest_spec(candidate_run_id),
            }
        )
    selected = manifest_spec("r5-candidate")
    request_path = tmp_path / "gallery-request.json"

    runner.write_canonical_gallery_phase_request(
        request_path,
        deployment_inputs=deployment_inputs,
        selected_fold1_manifest=selected,
    )
    loaded = runner.load_gallery_phase_request(request_path)

    assert loaded["artifact_type"] == "task4_gallery_phase_request"
    assert [item["registry_row"]["method"] for item in loaded["deployment_inputs"]] == [
        "R5",
        "R3",
    ]
    assert loaded["selected_fold1_manifest"]["run_id"] == "r5-candidate"


def test_stability_manifest_spec_builds_lightweight_validated_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    session = runner.R5_CONFIG.training_session(
        run_id="r5-stability-fold-0",
        split_fingerprint="b" * 64,
        validation_fold=0,
    )
    session = runner.TrainingSessionConfig(
        run_id=session.run_id,
        run_kind="stability",
        candidate=session.candidate,
        hyperparameters=session.hyperparameters,
        objective=session.objective,
        source_policy=session.source_policy,
        augmentation_policy=session.augmentation_policy,
        validation_fold=0,
        split_fingerprint=session.split_fingerprint,
        parent_run_id="r5-candidate",
    )
    artifact = runner.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256="a" * 64,
    )
    checkpoint = type(
        "Loaded",
        (),
        {"epoch": 20, "score": 0.4, "sha256": "a" * 64},
    )()
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: checkpoint)

    evidence_input = runner._stability_input_from_request(
        {
            "run_id": session.run_id,
            "checkpoint_path": "checkpoint.pt",
            "manifest_path": "stability_evidence.json",
        },
        artifacts={session.run_id: artifact},
        splits=pd.DataFrame(),
    )

    assert isinstance(evidence_input, runner.StabilityEvidenceInput)
    assert evidence_input.artifact_path == Path("stability_evidence.json")
    assert isinstance(evidence_input.checkpoint, runner.CheckpointRecord)
    assert evidence_input.checkpoint.path == Path("checkpoint.pt")
    assert evidence_input.checkpoint.run_kind == "stability"


@pytest.mark.parametrize("registry_path", ("results/other.pt", "../outside.pt"))
def test_registry_checkpoint_path_must_match_request_inside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_path: str,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="checkpoint path"):
        runner._registry_checkpoint_path(
            Path("results/request.pt"),
            registry_row={"run_id": "run-1", "checkpoint_path": registry_path},
            run_id="run-1",
        )


def test_gallery_dry_run_revalidates_deployment_and_binds_selected_fold1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[str] = []
    winner = type("Winner", (), {"identity": type("Identity", (), {"run_id": "r5-candidate"})()})()
    monkeypatch.setattr(
        runner,
        "_deployment_inputs_from_request",
        lambda request, *, artifacts, splits: calls.append("inputs") or ("left", "right"),
    )
    monkeypatch.setattr(
        runner,
        "select_deployment_candidate",
        lambda candidates: calls.append("deployment") or winner,
    )

    payload = runner._gallery_phase_payload(
        type(
            "Args",
            (),
            {"dry_run": True, "phase_output": tmp_path / "gallery-dry-run.json"},
        )(),
        artifacts={},
        splits=pd.DataFrame(),
        request={"selected_fold1_manifest": {"run_id": "r5-candidate"}},
    )

    assert calls == ["inputs", "deployment"]
    assert payload == {
        "status": "ready",
        "deployment_winner": "r5-candidate",
        "selected_fold1_manifest": "r5-candidate",
        "gallery_timing": "forbidden",
        "opened_pixels": 0,
    }


def test_gallery_worker_is_terminated_when_it_hangs_after_output(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    output = tmp_path / "gallery-result.json"
    initial_signature = runner._phase_output_signature(output)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import time; "
                f"Path({str(output)!r}).write_text('{{}}'); "
                "time.sleep(60)"
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="wrote output but did not exit"):
        runner._wait_for_gallery_worker(
            process,
            phase_output=output,
            initial_signature=initial_signature,
            exit_grace_seconds=0.1,
            poll_seconds=0.01,
        )

    assert process.poll() is not None


def test_gpu_pinning_exposes_exactly_one_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    selected: list[int] = []
    monkeypatch.setattr(runner.torch.cuda, "set_device", lambda index: selected.append(index))

    device = runner.pin_single_gpu(0)

    assert device == torch.device("cuda:0")
    assert selected == [0]


def test_gpu_pinning_rejects_more_than_one_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="exactly one"):
        runner.pin_single_gpu(0)


def test_child_environment_maps_physical_gpu_to_visible_zero_and_rejects_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()

    env = runner.isolated_child_env(2)

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["TASK4_PHYSICAL_CUDA_DEVICE"] == "2"

    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(RuntimeError, match="distributed"):
        runner.isolated_child_env(0)


def test_gpu_pinning_rejects_cpu_fallback_and_distributed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        runner.pin_single_gpu(0)

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="distributed"):
        runner.pin_single_gpu(0)


def test_sealed_rows_and_teacher_test_paths_are_rejected_before_image_access() -> None:
    runner = _load_runner()
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "partition": "development",
                "teacher_path": "data/raw/teacher/train/images_train/1.jpg",
                "external_path": "data/processed/task4/v1/1.jpg",
            },
            {
                "id": 2,
                "partition": "holdout",
                "teacher_path": "data/raw/teacher/train/images_train/2.jpg",
                "external_path": "data/processed/task4/v1/2.jpg",
            },
        ]
    )

    with pytest.raises(ValueError, match="sealed"):
        runner.reject_sealed_image_rows(frame)

    frame.loc[1, "partition"] = "development"
    frame.loc[1, "teacher_path"] = "data/raw/teacher/test/images_test/2.jpg"
    with pytest.raises(ValueError, match="teacher-test"):
        runner.reject_sealed_image_rows(frame)


def test_smoke_attempt_registers_one_new_run_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
    calls: list[tuple[str, str]] = []

    def smoke(session: Any, device: torch.device, output_dir: Path) -> dict[str, object]:
        calls.append((session.run_id, str(device)))
        checkpoint = output_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        manifest = output_dir / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return {
            "parameter_count": 1,
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": runner.sha256_file(checkpoint),
            "evidence_manifest_path": manifest,
        }

    monkeypatch.setattr(runner, "_execute_synthetic_smoke", smoke)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))

    run_id = runner.run_registered_smoke_attempt(
        registry=registry,
        config=runner.SMOKE_FAMILIES["incremental_encoder"],
        split_fingerprint="b" * 64,
        gpu=0,
        output_root=tmp_path / "evidence",
        device_factory=lambda index: torch.device(f"cuda:{index}"),
    )

    rows = registry.read()
    assert run_id == rows[0]["run_id"]
    assert rows[0]["status"] == "completed"
    assert rows[0]["run_kind"] == "smoke"
    assert rows[0]["method"] == "R1"
    assert calls == [(run_id, "cuda:0")]


def test_dry_run_prints_phase_matrix_and_parent_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    split_path = tmp_path / "splits.csv"
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame())
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "load_config_artifacts", lambda root: {})
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)

    code = runner.main(
        [
            "--phase",
            "smoke",
            "--dry-run",
            "--registry",
            str(tmp_path / "runs.csv"),
            "--split-path",
            str(split_path),
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "phase: smoke" in output
    assert "split_fingerprint: " + "c" * 64 in output
    assert "incremental_encoder -> R1" in output
    assert "autoencoder -> R5" in output
    assert "pretrained_benchmark -> B1" in output
    assert "candidate matrix: pending parents" in output


def test_dry_run_does_not_create_registry_or_lock_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    registry_path = tmp_path / "runs.csv"
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame())
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)

    runner.main(
        [
            "--phase",
            "smoke",
            "--dry-run",
            "--registry",
            str(registry_path),
        ]
    )

    assert not registry_path.exists()
    assert not registry_path.with_name("runs.csv.lock").exists()


def test_real_canonical_split_filters_before_image_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    splits = runner.load_canonical_splits()
    inspected: list[pd.DataFrame] = []

    def spy(frame: pd.DataFrame) -> None:
        inspected.append(frame.copy())

    monkeypatch.setattr(runner, "reject_sealed_image_rows", spy)

    rows = runner.development_image_rows(splits)

    assert set(rows["partition"]) == {"development"}
    assert len(inspected) == 1
    assert set(inspected[0]["partition"]) == {"development"}


def test_runner_path_image_access_sees_development_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    splits = pd.DataFrame(
        [
            {
                "id": 1,
                "partition": "development",
                "cv_fold": 1,
                "path": "data/raw/teacher/train/images_train/1.jpg",
            },
            {
                "id": 2,
                "partition": "holdout",
                "cv_fold": None,
                "path": "data/raw/teacher/train/images_train/2.jpg",
            },
        ]
    )
    inspected: list[pd.DataFrame] = []
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: splits)
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)

    def spy(frame: pd.DataFrame) -> None:
        inspected.append(frame.copy())

    monkeypatch.setattr(runner, "reject_sealed_image_rows", spy)

    runner.main(["--phase", "smoke", "--dry-run", "--registry", str(tmp_path / "runs.csv")])

    assert len(inspected) == 1
    assert inspected[0]["partition"].tolist() == ["development"]


def test_development_image_rows_rejects_teacher_test_paths_mislabeled_development() -> None:
    runner = _load_runner()
    splits = pd.DataFrame(
        [
            {
                "id": 1,
                "partition": "development",
                "cv_fold": 1,
                "path": "data/raw/teacher/test/images_test/1.jpg",
            },
        ]
    )

    with pytest.raises(ValueError, match="teacher-test"):
        runner.development_image_rows(splits)


def test_candidate_variant_join_adds_teacher_sha_and_structure_from_real_schema() -> None:
    runner = _load_runner()
    variant_path = Path("data/processed/task4/external_variant_index.csv.gz")
    raw = pd.read_csv(runner.ROOT / variant_path, keep_default_na=False)
    splits = runner.load_canonical_splits()

    assert "sha256" not in raw.columns

    joined = runner.load_joined_candidate_variant_index(variant_path, splits)
    split_by_id = splits.set_index("id", drop=False)

    for column in (
        "sha256",
        "teacher_sha256",
        "cv_fold",
        "partition",
        "duplicate_group",
        "product_family_group",
    ):
        assert column in joined.columns
    assert set(joined["id"].astype(int)) == set(splits["id"].astype(int))
    assert joined.set_index("id")["sha256"].astype(str).equals(
        split_by_id["sha256"].astype(str)
    )
    assert joined.set_index("id")["teacher_sha256"].astype(str).equals(
        split_by_id["sha256"].astype(str)
    )
    assert joined.loc[joined["partition"].eq("development"), "cv_fold"].notna().all()


def test_candidate_dry_run_preflight_joins_real_variant_without_opening_pixels() -> None:
    runner = _load_runner()
    result = runner._verify_candidate_dry_run_request(
        {"variant_index_path": Path("data/processed/task4/external_variant_index.csv.gz")},
        splits=runner.load_canonical_splits(),
    )

    assert result["status"] == "passed"
    assert result["teacher_sha256_joined_from_splits"] is True
    assert result["opened_pixels"] == 0
    assert result["development_rows"] == 32773


def test_smoke_phase_spawns_one_fresh_process_per_default_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame())
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)

    def run(command: list[str], **kwargs: object) -> Any:
        calls.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", run)

    code = runner.main(
        [
            "--phase",
            "smoke",
            "--registry",
            str(tmp_path / "runs.csv"),
        ]
    )

    assert code == 0
    families = [command[command.index("--family") + 1] for command in calls]
    assert families == [
        "incremental_encoder",
        "autoencoder",
        "pretrained_benchmark",
    ]


def test_budget_estimate_spawns_one_fresh_process_per_throughput_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame())
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)

    def run(command: list[str], **kwargs: object) -> Any:
        calls.append(command)
        output = Path(command[command.index("--budget-output") + 1])
        family = command[command.index("--budget-family") + 1]
        config = runner.THROUGHPUT_CONFIGS[family]
        runner.write_json_atomic(
            output,
            {
                "family": family,
                "method": config.method,
                "step_seconds": 1.0,
                "conservative_step_seconds": 1.0,
                "sample_step_seconds": [1.0],
                "warmup_steps": 2,
                "measured_steps": 1,
                "steps_per_epoch": 409,
                "run_gpu_hours": 11.36,
                "non_training_overhead_gpu_hours": 0.25,
                "parameter_count": 1,
                "deterministic_algorithms": True,
                "device": "cuda:0",
            },
        )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", run)

    code = runner.main(
        [
            "--phase",
            "smoke",
            "--budget-only",
            "--budget-output",
            str(tmp_path / "budget.json"),
        ]
    )

    assert code == 0
    families = [command[command.index("--budget-family") + 1] for command in calls]
    assert families == list(runner.THROUGHPUT_CONFIGS)
    assert (tmp_path / "budget.json").is_file()


def test_non_smoke_phases_write_restartable_packages_after_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    gate = tmp_path / "gate.json"
    runner.write_json_atomic(
        gate,
        {
            "schema_version": 1,
            "gate": "task4_model_comparison_budget",
            "decision": "passed",
            "fits_budget": True,
            "selected_estimated_full_matrix_gpu_hours": 70.0,
            "budget_gpu_hours": 98.0,
            "split_fingerprint": "c" * 64,
            "source_artifacts": [],
        },
    )
    registry = tmp_path / "runs.csv"
    registry.write_text(",".join(runner.RUN_COLUMNS) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []}))
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
    monkeypatch.setattr(runner, "load_config_artifacts", lambda root: {})

    for phase in ("candidate", "stability", "gallery", "evidence"):
        package = tmp_path / f"{phase}.json"
        code = runner.main(
            [
                "--phase",
                phase,
                "--dry-run",
                "--registry",
                str(registry),
                "--budget-gate",
                str(gate),
                "--phase-output",
                str(package),
                "--candidate",
                "R1",
                "--run-id",
                f"task4-{phase}-planned",
                "--manifest",
                "results/evidence/task4/learned/task4-smoke-incremental_encoder-a1ec7342537e/manifest.json",
            ]
        )

        assert code == 0
        payload = json.loads(package.read_text(encoding="utf-8"))
        assert payload["phase"] == phase
        assert payload["gate"]["decision"] == "passed"


def test_candidate_phase_calls_real_training_and_evidence_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    gate = tmp_path / "gate.json"
    registry = tmp_path / "runs.csv"
    registry.write_text(",".join(runner.RUN_COLUMNS) + "\n", encoding="utf-8")
    runner.write_json_atomic(
        gate,
        {
            "schema_version": 1,
            "gate": "task4_model_comparison_budget",
            "decision": "passed",
            "fits_budget": True,
            "selected_estimated_full_matrix_gpu_hours": 17.0,
            "budget_gpu_hours": 98.0,
            "split_fingerprint": "c" * 64,
            "source_artifacts": [],
        },
    )
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []}))
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
    monkeypatch.setattr(runner, "load_config_artifacts", lambda root: {})
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
    monkeypatch.setattr(
        runner,
        "build_experiment_matrix",
        lambda rows, *, config_artifacts: {"R1": runner.R1_CONFIG},
    )
    calls: list[dict[str, object]] = []

    def train_evidence(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "run_id": kwargs["session"].run_id,
            "manifest": "results/evidence/task4/learned/candidate-r1-new/manifest.json",
            "config_artifact": (
                "results/evidence/task4/learned/candidate-r1-new/"
                "experiment_config.json"
            ),
        }

    monkeypatch.setattr(runner, "_run_training_and_evidence", train_evidence)

    code = runner.main(
        [
            "--phase",
            "candidate",
            "--candidate",
            "R1",
            "--run-id",
            "candidate-r1-new",
            "--registry",
            str(registry),
            "--budget-gate",
            str(gate),
            "--phase-output",
            str(tmp_path / "candidate-package.json"),
        ]
    )

    assert code == 0
    package = json.loads((tmp_path / "candidate-package.json").read_text(encoding="utf-8"))
    assert package["manifest"] == "results/evidence/task4/learned/candidate-r1-new/manifest.json"
    assert len(calls) == 1
    assert calls[0]["registry"] is not None
    assert calls[0]["session"].run_id == "candidate-r1-new"


def test_candidate_dry_run_reports_failed_budget_gate_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    gate = tmp_path / "gate.json"
    registry = tmp_path / "runs.csv"
    registry.write_text(",".join(runner.RUN_COLUMNS) + "\n", encoding="utf-8")
    runner.write_json_atomic(
        gate,
        {
            "schema_version": 1,
            "gate": "task4_model_comparison_budget",
            "decision": "failed",
            "fits_budget": False,
            "selected_estimated_full_matrix_gpu_hours": 120.0,
            "budget_gpu_hours": 98.0,
            "split_fingerprint": "c" * 64,
            "source_artifacts": [],
        },
    )
    called_training = False
    request = tmp_path / "request.json"
    runner.write_canonical_candidate_phase_request(
        request,
        candidate="R1",
        run_id="candidate-r1",
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )

    def train_evidence(**kwargs: object) -> dict[str, object]:
        nonlocal called_training
        called_training = True
        return {}

    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []}))
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
    monkeypatch.setattr(runner, "load_config_artifacts", lambda root: {})
    monkeypatch.setattr(
        runner,
        "build_experiment_matrix",
        lambda rows, *, config_artifacts: {"R1": runner.R1_CONFIG},
    )
    monkeypatch.setattr(runner, "_run_training_and_evidence", train_evidence)
    monkeypatch.setattr(
        runner,
        "_verify_candidate_dry_run_request",
        lambda request, *, splits: {
            "status": "passed",
            "sealed_rows_rejected_before_pixels": True,
            "opened_pixels": 0,
        },
    )

    package_path = tmp_path / "candidate-dry-run.json"
    code = runner.main(
        [
            "--phase",
            "candidate",
            "--dry-run",
            "--candidate",
            "R1",
            "--run-id",
            "candidate-r1",
            "--registry",
            str(registry),
            "--budget-gate",
            str(gate),
            "--phase-output",
            str(package_path),
            "--phase-request",
            str(request),
        ]
    )

    package = json.loads(package_path.read_text(encoding="utf-8"))
    assert code == 0
    assert package["gate"]["decision"] == "failed"
    assert package["phase_result"]["status"] == "blocked"
    assert package["phase_result"]["safety"]["opened_pixels"] == 0
    assert called_training is False


def test_base_candidate_dry_run_does_not_require_later_parent_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    gate = tmp_path / "gate.json"
    registry = tmp_path / "runs.csv"
    registry.write_text(",".join(runner.RUN_COLUMNS) + "\n", encoding="utf-8")
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
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []}))
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
    monkeypatch.setattr(runner, "load_config_artifacts", lambda root: {})

    package_path = tmp_path / "candidate-dry-run.json"
    code = runner.main(
        [
            "--phase",
            "candidate",
            "--dry-run",
            "--candidate",
            "R1",
            "--run-id",
            "candidate-r1",
            "--registry",
            str(registry),
            "--budget-gate",
            str(gate),
            "--phase-output",
            str(package_path),
        ]
    )

    package = json.loads(package_path.read_text(encoding="utf-8"))
    assert code == 0
    assert package["phase_result"]["status"] == "ready"
    assert package["phase_result"]["candidate"] == "R1"
    assert package["phase_result"]["run_id"] == "candidate-r1"
    assert str(package["matrix_status"]).startswith("requested candidate ready")


def test_r3_candidate_config_uses_r1_r2_parent_without_completed_r3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    rows = [
        {"run_id": "r1", "method": "R1", "status": "completed"},
        {"run_id": "r2", "method": "R2", "status": "completed"},
    ]
    monkeypatch.setattr(
        runner,
        "build_experiment_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("R3 dry-run must not require the full R1-R5 matrix")
        ),
    )
    monkeypatch.setattr(
        runner,
        "derive_r3_config",
        lambda rows_arg, *, config_artifacts: runner.R1_CONFIG,
    )

    assert (
        runner._candidate_phase_config(rows, artifacts={}, candidate="R3")
        is runner.R1_CONFIG
    )


def test_evidence_recovery_requires_exact_failed_row_and_does_not_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
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
    registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
    session = runner._candidate_config("R5").training_session(
        run_id="task4-candidate-r5-task9-preexec",
        split_fingerprint="c" * 64,
    )
    row = runner._base_registry_row(session)
    registry.append(row)
    registry.update(
        session.run_id,
        {
            "status": "failed",
            "completed_at_utc": row["started_at_utc"],
            "error_type": "ValueError",
            "error_message": "query metric values must be null or finite in [0, 1]",
        },
    )
    checkpoints = [
        tmp_path / f"results/evidence/task4/checkpoints/{session.run_id}/epoch-{epoch:03d}.pt"
        for epoch in (20, 40, 60, 80, 100)
    ]
    for checkpoint in checkpoints:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
    request = tmp_path / "recovery-request.json"
    runner.write_canonical_evidence_recovery_request(
        request,
        candidate="R5",
        run_id=session.run_id,
        checkpoint_paths=checkpoints,
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )
    calls: list[str] = []
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []}))
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
    monkeypatch.setattr(runner, "load_config_artifacts", lambda root: {})
    monkeypatch.setattr(runner, "build_experiment_matrix", lambda rows, *, config_artifacts: {})
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
                    type("Checkpoint", (), {"epoch": epoch})()
                    for epoch in (20, 40, 60, 80, 100)
                ],
                "best_checkpoint": type("Checkpoint", (), {"sha256": "a" * 64})(),
            },
        )(),
    )
    monkeypatch.setattr(
        runner,
        "_run_training_and_evidence",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("recovery must not train")),
    )

    code = runner.main(
        [
            "--phase",
            "evidence-recovery",
            "--dry-run",
            "--registry",
            str(registry.path),
            "--budget-gate",
            str(gate),
            "--phase-output",
            str(tmp_path / "recovery-result.json"),
            "--phase-request",
            str(request),
        ]
    )

    package = json.loads((tmp_path / "recovery-result.json").read_text(encoding="utf-8"))
    assert code == 0
    assert calls == ["reconstruct"]
    assert package["phase_result"]["status"] == "ready"
    assert package["phase_result"]["mode"] == "evidence_recovery"
    assert package["phase_result"]["opened_pixels"] == 0
    assert package["phase_result"]["training_call"] == "forbidden"


def test_stability_fold_request_dry_run_selects_one_lightweight_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "current_git_identity", lambda: ("a" * 40, True))
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
    registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
    completed = []
    for candidate, score in (("R5", 0.5), ("R1", 0.3)):
        config = runner._candidate_config(candidate)
        session = config.training_session(
            run_id=f"{candidate.lower()}-candidate",
            split_fingerprint="c" * 64,
        )
        row = runner._base_registry_row(session)
        registry.append(row)
        checkpoint_path = (
            tmp_path
            / f"results/evidence/task4/checkpoints/{session.run_id}/epoch-100.pt"
        )
        manifest_path = tmp_path / f"results/evidence/task4/learned/{session.run_id}/manifest.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        manifest_path.write_text("{}\n", encoding="utf-8")
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        registry.update(
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
            tmp_path / f"results/evidence/task4/learned/{session.run_id}/experiment_config.json",
            session=session,
            checkpoint_sha256=checkpoint_sha256,
        )
        artifact = runner._load_experiment_config_artifact(artifact_path)
        completed.append((session, artifact))
    request = tmp_path / "stability-request.json"
    runner.write_canonical_stability_phase_request(
        request,
        finalist=runner.StabilityFinalist(
            identity=completed[0][1].identity,
            candidate_score=0.5,
            config_artifact=completed[0][1],
        ),
        plan=runner.build_stability_plan(
            runner.StabilityFinalist(
                identity=completed[0][1].identity,
                candidate_score=0.5,
                config_artifact=completed[0][1],
            ),
            attempt_token="task9-stability-r5",
        )[1],
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )
    calls: list[str] = []
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []}))
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
    monkeypatch.setattr(
        runner,
        "load_config_artifacts",
        lambda root: {item[0].run_id: item[1] for item in completed},
    )
    monkeypatch.setattr(
        runner,
        "_verify_candidate_dry_run_request",
        lambda request, *, splits: {"status": "passed", "opened_pixels": 0},
    )
    monkeypatch.setattr(
        runner,
        "_run_training_and_evidence",
        lambda **kwargs: calls.append("train") or {},
    )

    code = runner.main(
        [
            "--phase",
            "stability",
            "--dry-run",
            "--registry",
            str(registry.path),
            "--budget-gate",
            str(gate),
            "--phase-output",
            str(tmp_path / "stability-dry-run.json"),
            "--phase-request",
            str(request),
        ]
    )

    package = json.loads((tmp_path / "stability-dry-run.json").read_text(encoding="utf-8"))
    assert code == 0
    assert calls == []
    assert (
        package["phase_result"]["selected_run"]
        == "r5-candidate-stability-task9-stability-r5-fold-1"
    )
    assert package["phase_result"]["fold"] == 1
    assert package["phase_result"]["opened_pixels"] == 0
    assert package["phase_result"]["evidence_scope"] == "lightweight_primary_score_and_coverage"


def test_stability_gallery_and_evidence_call_phase_specific_apis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    gate = tmp_path / "gate.json"
    registry = tmp_path / "runs.csv"
    registry.write_text(",".join(runner.RUN_COLUMNS) + "\n", encoding="utf-8")
    runner.write_json_atomic(
        gate,
        {
            "schema_version": 1,
            "gate": "task4_model_comparison_budget",
            "decision": "passed",
            "fits_budget": True,
            "selected_estimated_full_matrix_gpu_hours": 17.0,
            "budget_gpu_hours": 98.0,
            "split_fingerprint": "c" * 64,
            "source_artifacts": [],
        },
    )
    calls: list[str] = []
    monkeypatch.setattr(runner, "load_canonical_splits", lambda: pd.DataFrame({"partition": []}))
    monkeypatch.setattr(runner, "cv_assignment_digest", lambda frame: "c" * 64)
    monkeypatch.setattr(runner, "development_image_rows", lambda frame: frame)
    monkeypatch.setattr(runner, "load_config_artifacts", lambda root: {})
    monkeypatch.setattr(
        runner,
        "select_stability_finalists",
        lambda rows, *, config_artifacts: calls.append("select_finalists")
        or ("left", "right"),
    )
    monkeypatch.setattr(
        runner,
        "build_stability_plan",
        lambda finalist, *, attempt_token: calls.append("stability_plan") or [],
    )
    monkeypatch.setattr(
        runner,
        "select_deployment_candidate",
        lambda candidates: calls.append("deployment_winner") or "winner",
    )
    monkeypatch.setattr(
        runner,
        "select_gallery_source",
        lambda manifest, **kwargs: calls.append("gallery_selection") or "two_view",
    )

    for phase in ("stability", "gallery", "evidence"):
        runner.main(
            [
                "--phase",
                phase,
                "--dry-run",
                "--registry",
                str(registry),
                "--budget-gate",
                str(gate),
                "--phase-output",
                str(tmp_path / f"{phase}.json"),
                "--run-id",
                f"{phase}-run",
                "--manifest",
                "manifest.json",
            ]
        )

    assert calls == [
        "select_finalists",
        "stability_plan",
    ]


def test_config_artifact_loader_reopens_validated_experiment_configs(tmp_path: Path) -> None:
    runner = _load_runner()
    session = runner.TrainingSessionConfig(
        run_id="candidate-r1",
        run_kind="candidate",
        candidate=runner.R1_CONFIG.candidate,
        hyperparameters=runner.R1_CONFIG.hyperparameters,
        objective=runner.R1_CONFIG.objective,
        source_policy=runner.R1_CONFIG.source_policy,
        augmentation_policy=runner.R1_CONFIG.augmentation_policy,
        validation_fold=1,
        split_fingerprint="b" * 64,
    )
    runner.write_experiment_config_artifact(
        tmp_path / "candidate-r1" / "experiment_config.json",
        session=session,
        checkpoint_sha256="a" * 64,
    )

    artifacts = runner.load_config_artifacts(tmp_path)

    assert tuple(artifacts) == ("candidate-r1",)
    assert artifacts["candidate-r1"].identity.run_id == "candidate-r1"
    assert artifacts["candidate-r1"].identity.checkpoint_sha256 == "a" * 64


def test_atomic_json_write_does_not_leave_temp_file(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "artifact.json"

    runner.write_json_atomic(path, {"b": 2, "a": 1})

    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_budget_gate_cli_combines_two_source_artifacts(tmp_path: Path) -> None:
    runner = _load_runner()
    gpu0 = tmp_path / "gpu0.json"
    gpu1 = tmp_path / "gpu1.json"
    output = tmp_path / "gate.json"
    gpu0.write_text(json.dumps(_budget_artifact("cuda:0", 10.0, 12.0, 8.0)), encoding="utf-8")
    gpu1.write_text(json.dumps(_budget_artifact("cuda:1", 11.0, 12.0, 8.0)), encoding="utf-8")

    code = runner.main(
        [
            "--phase",
            "smoke",
            "--budget-gate-combine",
            "--budget-source",
            str(gpu0),
            "--budget-source",
            str(gpu1),
            "--budget-gate",
            str(output),
        ]
    )

    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["selected_device"] == "cuda:1"


def test_budget_estimator_records_warmups_samples_overhead_and_all_paths(tmp_path: Path) -> None:
    runner = _load_runner()
    records = [
        {
            "family": family,
            "method": config.method,
            "step_seconds": 1.0,
            "conservative_step_seconds": 1.0,
            "sample_step_seconds": [0.9, 1.0],
            "warmup_steps": 2,
            "measured_steps": 2,
            "steps_per_epoch": 409,
            "run_gpu_hours": 40900 / 3600,
            "non_training_overhead_gpu_hours": 0.25,
            "parameter_count": 1,
            "deterministic_algorithms": True,
            "device": "cuda:0",
        }
        for family, config in runner.THROUGHPUT_CONFIGS.items()
    ]

    artifact = runner._aggregate_budget_records(
        records,
        split_fingerprint="b" * 64,
        output_path=tmp_path / "budget.json",
    )

    assert tuple(runner.THROUGHPUT_CONFIGS) == (
        "incremental_encoder_r1",
        "incremental_encoder_resnet34",
        "geometry_encoder_r3",
        "triplet_encoder_r4",
        "autoencoder",
        "pretrained_benchmark",
    )
    fold1_total = (
        sum(float(record["run_gpu_hours"]) for record in records)
        + sum(float(record["non_training_overhead_gpu_hours"]) for record in records)
    )
    stability_total = 10 * ((40900 / 3600) + 0.25)
    assert artifact["estimated_full_matrix_gpu_hours"] == pytest.approx(
        fold1_total + stability_total
    )
    assert artifact["non_training_overhead_gpu_hours"] == pytest.approx(1.5)


def test_budget_gate_includes_ten_conservative_stability_runs(tmp_path: Path) -> None:
    runner = _load_runner()
    records = [
        _budget_record(
            device="cuda:0",
            family=family,
            method=config.method,
            run_gpu_hours=hours,
        )
        for (family, config), hours in zip(
            runner.THROUGHPUT_CONFIGS.items(),
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            strict=True,
        )
    ]

    artifact = runner._aggregate_budget_records(
        records,
        split_fingerprint="b" * 64,
        output_path=tmp_path / "budget.json",
    )

    assert artifact["fold1_candidate_runs"] == 6
    assert artifact["conservative_stability_runs"] == 10
    assert artifact["conservative_stability_gpu_hours"] == pytest.approx(10 * (8.0 + 0.25))
    assert artifact["estimated_full_matrix_gpu_hours"] == pytest.approx(39.0 + 1.5 + 82.5)
    assert artifact["fits_budget"] is False


def test_budget_gate_preserves_distinct_physical_gpu_identity(tmp_path: Path) -> None:
    runner = _load_runner()
    gpu0 = tmp_path / "gpu0.json"
    gpu1 = tmp_path / "gpu1.json"
    output = tmp_path / "gate.json"
    gpu0_payload = _budget_artifact("cuda:0", 10.0, 12.0, 8.0)
    gpu1_payload = _budget_artifact("cuda:0", 11.0, 12.0, 8.0)
    gpu0_payload["physical_gpu"] = {
        "ordinal": 0,
        "name": "NVIDIA RTX A6000",
        "uuid": "GPU-0000",
    }
    gpu1_payload["physical_gpu"] = {
        "ordinal": 1,
        "name": "NVIDIA RTX A6000",
        "uuid": "GPU-1111",
    }
    for row in gpu0_payload["results"]:
        row["physical_gpu"] = gpu0_payload["physical_gpu"]
    for row in gpu1_payload["results"]:
        row["physical_gpu"] = gpu1_payload["physical_gpu"]
    runner.write_json_atomic(gpu0, gpu0_payload)
    runner.write_json_atomic(gpu1, gpu1_payload)

    artifact = runner.write_budget_gate_artifact(
        source_paths=(gpu0, gpu1),
        output_path=output,
    )

    assert artifact["selected_physical_gpu"] == {
        "ordinal": 1,
        "name": "NVIDIA RTX A6000",
        "uuid": "GPU-1111",
    }
    assert [source["physical_gpu"]["ordinal"] for source in artifact["source_artifacts"]] == [0, 1]


def _budget_record(
    *,
    device: str,
    family: str,
    method: str,
    run_gpu_hours: float,
) -> dict[str, object]:
    return {
        "family": family,
        "method": method,
        "step_seconds": 1.0,
        "conservative_step_seconds": 1.0,
        "sample_step_seconds": [0.9, 0.95, 1.0, 0.98, 0.97],
        "warmup_steps": 2,
        "measured_steps": 5,
        "steps_per_epoch": 409,
        "run_gpu_hours": run_gpu_hours,
        "non_training_overhead_gpu_hours": 0.25,
        "parameter_count": 1,
        "deterministic_algorithms": True,
        "device": device,
    }


def _budget_artifact(
    device: str,
    incremental: float,
    autoencoder: float,
    b1: float,
) -> dict[str, object]:
    physical_gpu = {
        "ordinal": int(device.rsplit(":", maxsplit=1)[-1]),
        "name": "NVIDIA RTX A6000",
        "uuid": f"GPU-{device.rsplit(':', maxsplit=1)[-1]}",
    }
    results = [
        _budget_record(
            device=device,
            family="incremental_encoder_r1",
            method="R1",
            run_gpu_hours=incremental,
        ),
        _budget_record(
            device=device,
            family="incremental_encoder_resnet34",
            method="R2",
            run_gpu_hours=incremental,
        ),
        _budget_record(
            device=device,
            family="geometry_encoder_r3",
            method="R3",
            run_gpu_hours=incremental,
        ),
        _budget_record(
            device=device,
            family="triplet_encoder_r4",
            method="R4",
            run_gpu_hours=incremental,
        ),
        _budget_record(device=device, family="autoencoder", method="R5", run_gpu_hours=autoencoder),
        _budget_record(
            device=device,
            family="pretrained_benchmark",
            method="B1",
            run_gpu_hours=b1,
        ),
    ]
    for row in results:
        row["physical_gpu"] = physical_gpu
    training_hours = sum(float(record["run_gpu_hours"]) for record in results)
    overhead_hours = sum(float(record["non_training_overhead_gpu_hours"]) for record in results)
    return {
        "schema_version": 1,
        "split_fingerprint": "b" * 64,
        "batch_size": 64,
        "planned_epochs": 100,
        "budget_gpu_hours": 98.0,
        "estimate_rule": (
            "Sum R1, R2, R3 geometry, R4 triplet, R5, and B1 conservative "
            "sampled timings plus declared non-training overhead"
        ),
        "results": results,
        "training_gpu_hours": training_hours,
        "non_training_overhead_gpu_hours": overhead_hours,
        "physical_gpu": physical_gpu,
        "estimated_full_matrix_gpu_hours": training_hours + overhead_hours,
        "fits_budget": True,
    }


def test_budget_gate_artifact_selects_slower_gpu_and_hashes_sources(tmp_path: Path) -> None:
    runner = _load_runner()
    gpu0 = tmp_path / "gpu0.json"
    gpu1 = tmp_path / "gpu1.json"
    output = tmp_path / "gate.json"
    gpu0.write_text(json.dumps(_budget_artifact("cuda:0", 10.0, 12.0, 8.0)), encoding="utf-8")
    gpu1.write_text(json.dumps(_budget_artifact("cuda:1", 11.0, 12.0, 8.0)), encoding="utf-8")

    artifact = runner.write_budget_gate_artifact(
        source_paths=(gpu0, gpu1),
        output_path=output,
    )

    assert output.is_file()
    assert artifact["schema_version"] == 1
    assert artifact["decision"] == "failed"
    assert artifact["selected_device"] == "cuda:1"
    assert artifact["selected_estimated_full_matrix_gpu_hours"] == pytest.approx(188.0)
    assert artifact["selected_conservative_stability_gpu_hours"] == pytest.approx(122.5)
    assert artifact["budget_gpu_hours"] == 98.0
    assert artifact["fits_budget"] is False
    sources = artifact["source_artifacts"]
    assert [source["device"] for source in sources] == ["cuda:0", "cuda:1"]
    assert all(len(source["sha256"]) == 64 for source in sources)


def test_training_uses_protocol_a_milestone_scorer_not_loss_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    session = runner.TrainingSessionConfig(
        run_id="candidate-r1",
        run_kind="candidate",
        candidate=runner.R1_CONFIG.candidate,
        hyperparameters=runner.TrainingHyperparameters(
            planned_epochs=20,
            checkpoint_epochs=(20,),
        ),
        objective=runner.R1_CONFIG.objective,
        source_policy=runner.R1_CONFIG.source_policy,
        augmentation_policy=runner.R1_CONFIG.augmentation_policy,
        validation_fold=1,
        split_fingerprint="b" * 64,
    )
    context = {
        "loader": ["batch"],
        "caches": {"teacher": object(), "v1": object()},
        "statistics": {"teacher": {"source": "teacher"}, "v1": {"source": "v1"}},
        "statistics_paths": {},
        "query_rows": pd.DataFrame({"id": [1], "partition": ["development"]}),
        "path_columns": {"teacher": "teacher_path", "v1": "external_path"},
    }
    monkeypatch.setattr(runner, "_candidate_data_context", lambda **kwargs: context)
    monkeypatch.setattr(runner, "build_optimizer", lambda model, hyperparameters: object())
    monkeypatch.setattr(runner, "WarmupCosineScheduler", lambda *args, **kwargs: object())
    scaler_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "make_grad_scaler",
        lambda *args, **kwargs: scaler_calls.append({"args": args, "kwargs": kwargs})
        or object(),
    )
    monkeypatch.setattr(runner, "pin_single_gpu", lambda gpu: torch.device("cpu"))
    monkeypatch.setattr(
        runner,
        "_base_registry_row",
        lambda bound_session: {"run_id": bound_session.run_id},
    )
    evidence_calls: list[dict[str, object]] = []

    def build_evidence(*args: object, **kwargs: object) -> object:
        evidence_calls.append(kwargs)
        return type(
            "Evidence",
            (),
            {"manifest_path": tmp_path / "manifest.json"},
        )()

    monkeypatch.setattr(runner, "build_learned_evidence", build_evidence)
    monkeypatch.setattr(runner, "record_evidence_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "write_experiment_config_artifact",
        lambda path, **kwargs: tmp_path / "experiment_config.json",
    )
    monkeypatch.setattr(runner, "_relative_artifact_path", lambda path: str(path))
    monkeypatch.setattr(
        runner,
        "save_checkpoint",
        lambda path, **kwargs: runner.CheckpointRecord(
            epoch=kwargs["epoch"],
            path=tmp_path / "epoch-020.pt",
            sha256="a" * 64,
            config_hash=session.config_hash,
            score=kwargs["score"],
            split_fingerprint=session.split_fingerprint,
            weight_origin=session.model_metadata.weight_origin,
            parent_run_id=session.parent_run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
        ),
    )
    scorer_calls: list[dict[str, object]] = []

    def milestone_scorer(evaluate: Any, *, fold: int) -> Any:
        scorer_calls.append({"evaluate": evaluate, "fold": fold})
        return lambda model, epoch: 0.73

    monkeypatch.setattr(runner, "make_milestone_scorer", milestone_scorer)
    monkeypatch.setattr(
        runner,
        "run_training_attempt",
        lambda **kwargs: kwargs["train"](
            torch.device("cuda:0"),
            torch.nn.Linear(1, 1),
            kwargs["session"],
            kwargs["batches"],
        ),
    )
    captured_callbacks: dict[str, Any] = {}

    def train_epochs(**kwargs: Any) -> Any:
        captured_callbacks["score_callback"] = kwargs["score_callback"]
        record = kwargs["checkpoint_callback"](20, kwargs["score_callback"](kwargs["model"], 20))
        return type(
            "Result",
            (),
            {
                "run_id": session.run_id,
                "run_kind": session.run_kind,
                "checkpoints": (record,),
                "best_checkpoint": record,
            },
        )()

    monkeypatch.setattr(runner, "train_epochs", train_epochs)

    output = runner._run_training_and_evidence(
        registry=object(),
        request={},
        session=session,
        splits=pd.DataFrame({"partition": []}),
        phase_output=tmp_path / "candidate.json",
        gpu=0,
    )

    assert output["checkpoint_sha256"] == "a" * 64
    assert len(scorer_calls) >= 1
    assert {call["fold"] for call in scorer_calls} == {1}
    assert scaler_calls[0]["kwargs"]["growth_interval"] == 40901
    assert evidence_calls[0]["device"] == torch.device("cuda:0")
    assert captured_callbacks["score_callback"](torch.nn.Linear(1, 1), 20) == pytest.approx(0.73)


def test_stability_training_uses_lightweight_evidence_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    session = runner.TrainingSessionConfig(
        run_id="stability-r1-fold0",
        run_kind="stability",
        candidate=runner.R1_CONFIG.candidate,
        hyperparameters=runner.TrainingHyperparameters(
            planned_epochs=20,
            checkpoint_epochs=(20,),
        ),
        objective=runner.R1_CONFIG.objective,
        source_policy=runner.R1_CONFIG.source_policy,
        augmentation_policy=runner.R1_CONFIG.augmentation_policy,
        validation_fold=0,
        split_fingerprint="b" * 64,
        parent_run_id="candidate-r1",
    )
    context = {
        "loader": ["batch"],
        "caches": {"teacher": object(), "v1": object()},
        "statistics": {"teacher": {"source": "teacher"}, "v1": {"source": "v1"}},
        "statistics_paths": {},
        "query_rows": pd.DataFrame({"id": [1], "partition": ["development"]}),
        "path_columns": {"teacher": "teacher_path", "v1": "external_path"},
    }
    monkeypatch.setattr(runner, "_candidate_data_context", lambda **kwargs: context)
    monkeypatch.setattr(runner, "build_optimizer", lambda model, hyperparameters: object())
    monkeypatch.setattr(runner, "WarmupCosineScheduler", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "make_grad_scaler", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner,
        "_base_registry_row",
        lambda bound_session: {"run_id": bound_session.run_id},
    )
    monkeypatch.setattr(
        runner,
        "build_learned_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stability must not build full candidate evidence")
        ),
    )
    monkeypatch.setattr(runner, "record_evidence_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "write_experiment_config_artifact",
        lambda path, **kwargs: tmp_path / "experiment_config.json",
    )
    monkeypatch.setattr(runner, "_relative_artifact_path", lambda path: str(path))
    record = runner.CheckpointRecord(
        epoch=20,
        path=tmp_path / "epoch-020.pt",
        sha256="a" * 64,
        config_hash=session.config_hash,
        score=0.73,
        split_fingerprint=session.split_fingerprint,
        weight_origin=session.model_metadata.weight_origin,
        parent_run_id=session.parent_run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
    )
    monkeypatch.setattr(runner, "save_checkpoint", lambda path, **kwargs: record)
    monkeypatch.setattr(
        runner,
        "make_milestone_scorer",
        lambda evaluate, *, fold: lambda model, epoch: 0.73,
    )
    monkeypatch.setattr(
        runner,
        "run_training_attempt",
        lambda **kwargs: kwargs["train"](
            torch.device("cuda:0"),
            torch.nn.Linear(1, 1),
            kwargs["session"],
            kwargs["batches"],
        ),
    )
    monkeypatch.setattr(
        runner,
        "train_epochs",
        lambda **kwargs: type(
            "Result",
            (),
            {
                "run_id": session.run_id,
                "run_kind": session.run_kind,
                "checkpoints": (record,),
                "best_checkpoint": record,
            },
        )(),
    )
    stability_calls: list[dict[str, object]] = []

    def build_stability(*args: object, **kwargs: object) -> object:
        stability_calls.append(kwargs)
        return type(
            "StabilityEvidence",
            (),
            {"manifest_path": tmp_path / "stability.json"},
        )()

    monkeypatch.setattr(runner, "build_stability_evidence", build_stability)

    output = runner._run_training_and_evidence(
        registry=object(),
        request={},
        session=session,
        splits=pd.DataFrame({"partition": []}),
        phase_output=tmp_path / "stability-result.json",
        gpu=0,
    )

    assert output["manifest"] == str(tmp_path / "stability.json")
    assert stability_calls[0]["device"] == torch.device("cuda:0")


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_evidence_resume_refuses_non_running_registry_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    runner = _load_runner()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runner, "ROOT", repo_root)
    request_path = tmp_path / "resume.json"
    checkpoint_paths = [
        repo_root / "results/evidence/task4/checkpoints/candidate-r1" / f"epoch-{epoch:03d}.pt"
        for epoch in (20, 40, 60, 80, 100)
    ]
    runner.write_canonical_evidence_resume_request(
        request_path,
        candidate="R1",
        run_id="candidate-r1",
        checkpoint_paths=checkpoint_paths,
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )
    request = runner.load_evidence_resume_request(request_path)

    class Registry:
        def read(self) -> list[dict[str, object]]:
            return [
                {
                    **runner._candidate_config("R1")
                    .training_session(run_id="candidate-r1", split_fingerprint="b" * 64)
                    .expected_registry_identity.as_dict(),
                    "status": status,
                    "checkpoint_sha256": "a" * 64,
                }
            ]

    with pytest.raises(ValueError, match="running"):
        runner._evidence_resume_phase_payload(
            type(
                "Args",
                (),
                {
                    "dry_run": True,
                    "phase_output": tmp_path / "resume-result.json",
                    "gpu": 0,
                },
            )(),
            registry=Registry(),
            split_fingerprint="b" * 64,
            splits=pd.DataFrame({"partition": []}),
            request=request,
        )


def test_evidence_resume_refuses_missing_checkpoint_before_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runner, "ROOT", repo_root)
    request_path = tmp_path / "resume.json"
    checkpoint_paths = [
        repo_root / "results/evidence/task4/checkpoints/candidate-r1" / f"epoch-{epoch:03d}.pt"
        for epoch in (20, 40, 60, 80, 100)
    ]
    runner.write_canonical_evidence_resume_request(
        request_path,
        candidate="R1",
        run_id="candidate-r1",
        checkpoint_paths=checkpoint_paths,
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )
    request = runner.load_evidence_resume_request(request_path)
    session = runner._candidate_config("R1").training_session(
        run_id="candidate-r1",
        split_fingerprint="b" * 64,
    )
    registry_row = {**session.expected_registry_identity.as_dict(), "status": "running"}

    class Registry:
        def read(self) -> list[dict[str, object]]:
            return [registry_row]

    monkeypatch.setattr(
        runner,
        "build_learned_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("missing checkpoint must block")
        ),
    )

    with pytest.raises(ValueError, match="checkpoint is missing"):
        runner._evidence_resume_phase_payload(
            type(
                "Args",
                (),
                {
                    "dry_run": True,
                    "phase_output": tmp_path / "resume-result.json",
                    "gpu": 0,
                },
            )(),
            registry=Registry(),
            split_fingerprint="b" * 64,
            splits=pd.DataFrame({"partition": []}),
            request=request,
        )


def test_evidence_resume_refuses_mismatched_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runner, "ROOT", repo_root)
    request_path = tmp_path / "resume.json"
    checkpoint_paths = [
        repo_root / "results/evidence/task4/checkpoints/candidate-r1" / f"epoch-{epoch:03d}.pt"
        for epoch in (20, 40, 60, 80, 100)
    ]
    for path in checkpoint_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    runner.write_canonical_evidence_resume_request(
        request_path,
        candidate="R1",
        run_id="candidate-r1",
        checkpoint_paths=checkpoint_paths,
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )
    request = runner.load_evidence_resume_request(request_path)
    session = runner._candidate_config("R1").training_session(
        run_id="candidate-r1",
        split_fingerprint="b" * 64,
    )
    registry_row = {**session.expected_registry_identity.as_dict(), "status": "running"}

    class Registry:
        def read(self) -> list[dict[str, object]]:
            return [registry_row]

    monkeypatch.setattr(
        runner,
        "reconstruct_training_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("checkpoint config hash does not match")
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_learned_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched checkpoint must block")
        ),
    )

    with pytest.raises(ValueError, match="checkpoint config hash"):
        runner._evidence_resume_phase_payload(
            type(
                "Args",
                (),
                {
                    "dry_run": True,
                    "phase_output": tmp_path / "resume-result.json",
                    "gpu": 0,
                },
            )(),
            registry=Registry(),
            split_fingerprint="b" * 64,
            splits=pd.DataFrame({"partition": []}),
            request=request,
        )


def test_evidence_resume_reconstructs_and_completes_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runner, "ROOT", repo_root)
    request_path = tmp_path / "resume.json"
    checkpoint_paths = [
        repo_root / "results/evidence/task4/checkpoints/candidate-r1" / f"epoch-{epoch:03d}.pt"
        for epoch in (20, 40, 60, 80, 100)
    ]
    for path in checkpoint_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    runner.write_canonical_evidence_resume_request(
        request_path,
        candidate="R1",
        run_id="candidate-r1",
        checkpoint_paths=checkpoint_paths,
        variant_index_path=Path("data/processed/task4/external_variant_index.csv.gz"),
        cache_root=Path("results/cache/task4/images"),
        checkpoint_root=Path("results/evidence/task4/checkpoints"),
        evidence_root=Path("results/evidence/task4"),
        feature_cache_root=Path("results/cache/task4/features"),
    )
    request = runner.load_evidence_resume_request(request_path)
    session = runner._candidate_config("R1").training_session(
        run_id="candidate-r1",
        split_fingerprint="b" * 64,
    )
    registry_row = {**session.expected_registry_identity.as_dict(), "status": "running"}

    class Registry:
        def read(self) -> list[dict[str, object]]:
            return [registry_row]

    monkeypatch.setattr(
        runner,
        "run_training_attempt",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("resume must not train")),
    )
    monkeypatch.setattr(runner, "_candidate_data_context", lambda **kwargs: {
        "caches": {"teacher": object(), "v1": object()},
        "statistics": {"teacher": {}, "v1": {}},
        "statistics_paths": {"teacher": tmp_path / "teacher.json", "v1": tmp_path / "v1.json"},
        "query_rows": pd.DataFrame({"id": [1], "partition": ["development"]}),
        "path_columns": {"teacher": "teacher_path", "v1": "external_path"},
    })
    monkeypatch.setattr(runner, "pin_single_gpu", lambda gpu: torch.device("cuda:0"))
    record = runner.CheckpointRecord(
        epoch=100,
        path=checkpoint_paths[-1],
        sha256="a" * 64,
        config_hash=session.config_hash,
        score=0.9,
        split_fingerprint=session.split_fingerprint,
        weight_origin=session.model_metadata.weight_origin,
        parent_run_id=session.parent_run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
    )
    result = type(
        "Result",
        (),
        {
            "run_id": session.run_id,
            "run_kind": session.run_kind,
            "checkpoints": (record,),
            "best_checkpoint": record,
        },
    )()
    reconstruct_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "reconstruct_training_result",
        lambda paths, **kwargs: reconstruct_calls.append({"paths": paths, **kwargs}) or result,
    )
    evidence_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "build_learned_evidence",
        lambda *args, **kwargs: evidence_calls.append(kwargs)
        or type("Evidence", (), {"manifest_path": tmp_path / "manifest.json"})(),
    )
    monkeypatch.setattr(
        runner,
        "write_experiment_config_artifact",
        lambda path, **kwargs: tmp_path / "experiment_config.json",
    )
    monkeypatch.setattr(runner, "_relative_artifact_path", lambda path: str(path))

    output = runner._evidence_resume_phase_payload(
        type(
            "Args",
            (),
            {
                "dry_run": False,
                "phase_output": tmp_path / "resume-result.json",
                "gpu": 0,
            },
        )(),
        registry=Registry(),
        split_fingerprint="b" * 64,
        splits=pd.DataFrame({"partition": []}),
        request=request,
    )

    assert output["run_id"] == "candidate-r1"
    assert output["checkpoint_sha256"] == "a" * 64
    assert reconstruct_calls[0]["registry_row"] == registry_row
    assert evidence_calls[0]["device"] == torch.device("cuda:0")
