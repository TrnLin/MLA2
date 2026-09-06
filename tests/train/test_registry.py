from __future__ import annotations

import csv
import fcntl
import hashlib
import multiprocessing
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import fashion.train.registry as registry_module
from fashion.train.recovery import interrupt_orphaned_run
from fashion.train.registry import (
    RUN_COLUMNS,
    TASK2_RUN_COLUMNS,
    TASK4_RUN_COLUMNS,
    DuplicateRunError,
    ImmutableRunError,
    RunRecord,
    RunRegistryError,
    new_run_id,
    tracked_run,
)
from fashion.train.registry import RunRegistry as Task2RunRegistry
from fashion.train.registry import (
    Task4RunRegistry as RunRegistry,
)


def _running_row(run_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "1",
        "run_id": run_id,
        "parent_run_id": "",
        "started_at_utc": "2026-08-29T01:02:03Z",
        "completed_at_utc": "",
        "task": "task4",
        "run_kind": "candidate",
        "status": "running",
        "fold": 1,
        "method": "R1",
        "architecture": "resnet18",
        "objective": "vicreg",
        "source_policy": "teacher_v1_pairs",
        "pretrained": False,
        "weight_origin": "random_initialization",
        "deployment_eligibility": "eligible",
        "seed": 2753,
        "embedding_dim": 128,
        "planned_epochs": 100,
        "selected_epoch": "",
        "config_hash": "a" * 64,
        "split_fingerprint": "b" * 64,
        "git_commit": "c" * 40,
        "dirty_tree": False,
        "parameter_count": "",
        "checkpoint_path": "",
        "checkpoint_sha256": "",
        "development_winner_score": "",
        "cross_source_score": "",
        "source_robustness_ratio": "",
        "protocol_b_recall_at_10": "",
        "p95_end_to_end_seconds": "",
        "index_bytes": "",
        "evidence_manifest_path": "",
        "error_type": "",
        "error_message": "",
    }
    row.update(overrides)
    return row


def _append_in_process(csv_path: str, run_id: str, start: object) -> None:
    start.wait()
    RunRegistry(Path(csv_path)).append(_running_row(run_id))


def _hold_registry_lock(
    lock_path: str,
    acquired: Any,
    release: Any,
) -> None:
    with Path(lock_path).open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        acquired.set()
        release.wait(timeout=15)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _mutate_registry_in_process(
    csv_path: str,
    operation: str,
    attempted: Any,
    finished: Any,
) -> None:
    registry = RunRegistry(Path(csv_path))
    real_flock = registry_module.fcntl.flock

    def signal_exclusive_lock_attempt(file_descriptor: int, operation_code: int) -> Any:
        if operation_code == fcntl.LOCK_EX:
            attempted.set()
        return real_flock(file_descriptor, operation_code)

    registry_module.fcntl.flock = signal_exclusive_lock_attempt
    if operation == "append":
        registry.append(_running_row("contender"))
    else:
        registry.update("existing", {"parameter_count": 11_689_512})
    finished.set()


def _completion_updates(root: Path, run_id: str) -> dict[str, object]:
    artifact_dir = root / "results" / "evidence" / "task4" / run_id
    artifact_dir.mkdir(parents=True)
    checkpoint = artifact_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = artifact_dir / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return {
        "status": "completed",
        "completed_at_utc": "2026-08-29T02:02:03Z",
        "selected_epoch": 80,
        "checkpoint_path": f"results/evidence/task4/{run_id}/model.pt",
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "evidence_manifest_path": f"results/evidence/task4/{run_id}/manifest.json",
    }


def test_registry_import_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import fashion.train.registry; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_registry_import_check_preserves_parent_torch_module() -> None:
    import torch

    loaded_torch = torch

    test_registry_import_does_not_import_torch()

    assert sys.modules["torch"] is loaded_torch


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_kind", "training"),
        ("status", "queued"),
        ("fold", 5),
        ("pretrained", "sometimes"),
        ("embedding_dim", 0),
        ("planned_epochs", 0),
        ("config_hash", "not-a-sha256"),
        ("split_fingerprint", ""),
    ],
)
def test_append_rejects_invalid_schema_values(
    tmp_path: Path, field: str, value: object
) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")

    with pytest.raises(RunRegistryError, match=field):
        registry.append(_running_row("bad-schema", **{field: value}))


def test_append_requires_every_registry_field(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    row = _running_row("missing-field")
    del row["objective"]

    with pytest.raises(RunRegistryError, match="objective"):
        registry.append(row)


def test_append_writes_running_attempt_with_deterministic_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "nested" / "runs.csv"
    registry = RunRegistry(csv_path)

    written = registry.append(_running_row("run-1"))

    assert written["run_id"] == "run-1"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == RUN_COLUMNS
        assert list(reader) == [
            {
                **{column: "" for column in RUN_COLUMNS},
                **{
                    key: str(value).lower() if isinstance(value, bool) else str(value)
                    for key, value in _running_row("run-1").items()
                },
            }
        ]


def test_append_rejects_duplicate_run_id(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("same-id"))

    with pytest.raises(RunRegistryError, match="duplicate run_id"):
        registry.append(_running_row("same-id"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "R2"),
        ("parent_run_id", "other"),
        ("seed", 99),
        ("pretrained", True),
        ("started_at_utc", "2026-08-30T00:00:00Z"),
    ],
)
def test_update_rejects_changes_to_identity_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("immutable"))

    with pytest.raises(RunRegistryError, match="immutable"):
        registry.update("immutable", {field: value})


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_runs_reject_all_further_status_transitions(
    tmp_path: Path, status: str
) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("terminal"))
    if status == "completed":
        registry.update("terminal", _completion_updates(tmp_path, "terminal"))
    elif status == "failed":
        registry.update(
            "terminal",
            {
                "status": "failed",
                "completed_at_utc": "2026-08-29T02:02:03Z",
                "error_type": "RuntimeError",
                "error_message": "non-finite loss",
            },
        )
    elif status == "cancelled":
        registry.update(
            "terminal",
            {
                "status": "cancelled",
                "completed_at_utc": "2026-08-29T02:02:03Z",
            },
        )

    target = "failed" if status != "failed" else "cancelled"
    with pytest.raises(RunRegistryError, match="transition"):
        registry.update("terminal", {"status": target})


def test_completed_run_records_artifacts_and_metrics(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("completed"))

    updates = _completion_updates(tmp_path, "completed")
    updates.update(
        {
            "parameter_count": 11_689_512,
            "development_winner_score": 0.42,
            "cross_source_score": 0.4,
            "source_robustness_ratio": 0.952,
            "protocol_b_recall_at_10": 0.7,
            "p95_end_to_end_seconds": 0.2,
            "index_bytes": 1234,
        }
    )
    updated = registry.update(
        "completed",
        updates,
    )

    assert updated["status"] == "completed"
    assert updated["selected_epoch"] == "80"
    assert registry.read()[0] == updated


@pytest.mark.parametrize(
    "missing_field",
    ["checkpoint_path", "checkpoint_sha256", "evidence_manifest_path", "selected_epoch"],
)
def test_completion_rejects_missing_artifacts(tmp_path: Path, missing_field: str) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("incomplete"))
    updates = _completion_updates(tmp_path, "incomplete")
    updates[missing_field] = ""

    with pytest.raises(RunRegistryError, match=missing_field):
        registry.update("incomplete", updates)


def test_completion_rejects_artifact_paths_that_do_not_exist(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("missing-artifact"))
    updates = _completion_updates(tmp_path, "missing-artifact")
    (tmp_path / str(updates["checkpoint_path"])).unlink()

    with pytest.raises(RunRegistryError, match="checkpoint_path"):
        registry.update("missing-artifact", updates)


def test_failed_run_remains_visible_and_retry_links_to_it(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("failed-run"))
    registry.update(
        "failed-run",
        {
            "status": "failed",
            "completed_at_utc": "2026-08-29T02:02:03Z",
            "error_type": "RuntimeError",
            "error_message": "CUDA out of memory",
        },
    )

    registry.append(_running_row("retry-run", parent_run_id="failed-run"))

    rows = registry.read()
    assert [(row["run_id"], row["status"]) for row in rows] == [
        ("failed-run", "failed"),
        ("retry-run", "running"),
    ]
    assert rows[1]["parent_run_id"] == "failed-run"


def test_append_rejects_missing_parent_run(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row("running-parent"))

    with pytest.raises(RunRegistryError, match="parent_run_id"):
        registry.append(_running_row("bad-retry", parent_run_id="missing"))


@pytest.mark.parametrize("eligibility", ["eligible", "deployment_eligible"])
def test_pretrained_run_cannot_be_deployment_eligible(
    tmp_path: Path, eligibility: str
) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")

    with pytest.raises(RunRegistryError, match="pretrained"):
        registry.append(
            _running_row(
                "pretrained",
                run_kind="benchmark",
                method="B1",
                pretrained=True,
                weight_origin="ResNet18_Weights.IMAGENET1K_V1",
                deployment_eligibility=eligibility,
            )
        )


def test_pretrained_comparison_only_run_is_valid(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")

    row = registry.append(
        _running_row(
            "benchmark",
            run_kind="benchmark",
            method="B1",
            pretrained=True,
            weight_origin="ResNet18_Weights.IMAGENET1K_V1",
            deployment_eligibility="comparison_only",
        )
    )

    assert row["deployment_eligibility"] == "comparison_only"


@pytest.mark.parametrize(
    "contents",
    [
        "wrong,header\nvalue,value\n",
        ",".join(TASK4_RUN_COLUMNS)
        + "\n"
        + ",".join([""] * len(TASK4_RUN_COLUMNS))
        + "\n",
    ],
)
def test_read_rejects_malformed_existing_csv(tmp_path: Path, contents: str) -> None:
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(RunRegistryError, match="malformed"):
        RunRegistry(csv_path).read()


def test_read_rejects_duplicate_ids_in_existing_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "runs.csv"
    registry = RunRegistry(csv_path)
    registry.append(_running_row("duplicate"))
    first_row = csv_path.read_text(encoding="utf-8").splitlines()[1]
    with csv_path.open("a", encoding="utf-8") as handle:
        handle.write(first_row + "\n")

    with pytest.raises(RunRegistryError, match="duplicate run_id"):
        registry.read()


def _write_malformed_quote_registry(tmp_path: Path) -> tuple[RunRegistry, bytes]:
    csv_path = tmp_path / "runs.csv"
    registry = RunRegistry(csv_path)
    registry.append(_running_row("quoted-run"))
    valid = csv_path.read_bytes()
    malformed = valid.replace(b"quoted-run,", b'"quoted-run"x,')
    assert malformed != valid
    csv_path.write_bytes(malformed)
    return registry, malformed


def test_read_rejects_malformed_quote_csv_without_changing_file(tmp_path: Path) -> None:
    registry, original_bytes = _write_malformed_quote_registry(tmp_path)

    with pytest.raises(RunRegistryError, match="malformed"):
        registry.read()

    assert registry.path.read_bytes() == original_bytes


def test_append_rejects_malformed_quote_csv_without_changing_file(tmp_path: Path) -> None:
    registry, original_bytes = _write_malformed_quote_registry(tmp_path)

    with pytest.raises(RunRegistryError, match="malformed"):
        registry.append(_running_row("must-not-be-written"))

    assert registry.path.read_bytes() == original_bytes


@pytest.mark.parametrize("operation", ["append", "update"])
def test_append_and_update_wait_for_exclusive_registry_lock(
    tmp_path: Path, operation: str
) -> None:
    csv_path = tmp_path / "runs.csv"
    registry = RunRegistry(csv_path)
    registry.append(_running_row("existing"))
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    attempted = context.Event()
    finished = context.Event()
    holder = context.Process(
        target=_hold_registry_lock,
        args=(str(registry.lock_path), acquired, release),
    )
    contender = context.Process(
        target=_mutate_registry_in_process,
        args=(str(csv_path), operation, attempted, finished),
    )

    holder.start()
    assert acquired.wait(timeout=5), "holder did not acquire the registry lock"
    contender.start()
    assert attempted.wait(timeout=5), "contender did not attempt its mutation"
    assert not finished.wait(timeout=0.25), "mutation finished while the lock was held"
    release.set()
    assert finished.wait(timeout=5), "mutation did not finish after lock release"
    holder.join(timeout=5)
    contender.join(timeout=5)

    assert holder.exitcode == 0
    assert contender.exitcode == 0
    rows = registry.read()
    if operation == "append":
        assert [row["run_id"] for row in rows] == ["existing", "contender"]
    else:
        assert rows[0]["parameter_count"] == "11689512"


def test_failed_temporary_write_preserves_csv_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "runs.csv"
    registry = RunRegistry(csv_path)
    registry.append(_running_row("existing"))
    original_bytes = csv_path.read_bytes()

    def fail_during_rows_write(writer: csv.DictWriter, rows: object) -> None:
        del writer, rows
        raise OSError("injected temporary write failure")

    monkeypatch.setattr(csv.DictWriter, "writerows", fail_during_rows_write)

    with pytest.raises(OSError, match="injected temporary write failure"):
        registry.append(_running_row("must-not-be-written"))

    assert csv_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".runs.csv.*.tmp")) == []


def test_two_process_contention_preserves_both_complete_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "runs.csv"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_append_in_process, args=(str(csv_path), f"run-{index}", start))
        for index in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)

    assert [process.exitcode for process in processes] == [0, 0]
    rows = RunRegistry(csv_path).read()
    assert {row["run_id"] for row in rows} == {"run-0", "run-1"}
    assert all(row["status"] == "running" for row in rows)


DIGESTS = {
    "config_sha256": "a" * 64,
    "split_sha256": "b" * 64,
    "label_map_sha256": "c" * 64,
    "implementation_sha256": "d" * 64,
}


def _record(run_id: str = "c1-f0-s2753-test") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        experiment_id="c1-screen",
        fold=0,
        seed=2753,
        model_family="smallcnn",
        **DIGESTS,
    )


def test_tracked_run_records_completed_metrics(tmp_path: Path) -> None:
    registry = Task2RunRegistry(tmp_path / "runs.csv")

    with tracked_run(registry, _record()) as run:
        run.epochs_completed = 2
        run.primary_metric_name = "macro_f1"
        run.primary_metric_value = 0.625
        run.metrics = {"macro_f1": 0.625, "loss": 0.8}

    rows = registry.read()
    assert tuple(rows.columns) == TASK2_RUN_COLUMNS
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "completed"
    assert rows.loc[0, "primary_metric_value"] == "0.625"
    assert rows.loc[0, "metrics"] == '{"loss":0.8,"macro_f1":0.625}'
    assert rows.loc[0, "git_commit"]
    assert rows.loc[0, "finished_at_utc"].endswith("Z")


def test_tracked_run_records_failure_and_reraises(tmp_path: Path) -> None:
    registry = Task2RunRegistry(tmp_path / "runs.csv")

    with pytest.raises(RuntimeError, match="out of memory"):
        with tracked_run(registry, _record()) as run:
            run.epochs_completed = 1
            raise RuntimeError("out of memory")

    rows = registry.read()
    assert rows.loc[0, "status"] == "failed"
    assert rows.loc[0, "error_type"] == "RuntimeError"
    assert rows.loc[0, "error_message"] == "out of memory"


def test_orphaned_running_run_can_be_marked_interrupted(tmp_path: Path) -> None:
    registry = Task2RunRegistry(tmp_path / "runs.csv")
    record = _record("p1-f1-s2753-orphaned")
    record.started_at_utc = "2026-08-26T17:42:28Z"
    registry.append(record)

    interrupt_orphaned_run(
        registry,
        record.run_id,
        reason="training host process exited before Python cleanup",
    )

    row = registry.read().iloc[0]
    assert row["status"] == "interrupted"
    assert row["error_type"] == "ExternalProcessTermination"
    assert row["error_message"] == "training host process exited before Python cleanup"
    assert row["started_at_utc"] == "2026-08-26T17:42:28Z"
    assert row["finished_at_utc"].endswith("Z")


def test_duplicate_ids_and_final_rewrites_are_rejected(tmp_path: Path) -> None:
    registry = Task2RunRegistry(tmp_path / "runs.csv")
    with tracked_run(registry, _record()):
        pass

    with pytest.raises(DuplicateRunError):
        registry.append(_record())

    finished = _record()
    finished.status = "completed"
    finished.finished_at_utc = "2026-08-26T00:00:00Z"
    with pytest.raises(ImmutableRunError):
        registry.finalize(finished)


def test_start_identity_cannot_change(tmp_path: Path) -> None:
    registry = Task2RunRegistry(tmp_path / "runs.csv")
    record = _record()
    registry.append(record)
    record.fold = 1
    record.status = "completed"
    record.finished_at_utc = "2026-08-26T00:00:00Z"

    with pytest.raises(ImmutableRunError, match="fold"):
        registry.finalize(record)


def test_new_run_ids_are_readable_and_unique() -> None:
    first = new_run_id("C1 screen", fold=2, seed=2753)
    second = new_run_id("C1 screen", fold=2, seed=2753)

    assert first.startswith("C1-screen-f2-s2753-")
    assert first != second


def test_record_rejects_invalid_provenance_hash() -> None:
    record = _record()
    record.split_sha256 = "not-a-digest"
    with pytest.raises(ValueError, match="split_sha256"):
        record.to_row()


def test_registry_retries_transient_windows_write_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = registry_module.atomic_write_csv
    attempts = 0

    def deny_once(path: Path, frame: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = PermissionError(5, "Access is denied", str(path))
            error.winerror = 5
            raise error
        real_write(path, frame)

    monkeypatch.setattr(registry_module, "atomic_write_csv", deny_once)
    registry = Task2RunRegistry(tmp_path / "runs.csv")

    registry.append(_record())

    assert attempts == 2
    assert registry.read().loc[0, "status"] == "running"


def test_task2_and_task4_rows_share_one_union_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "runs.csv"
    task4_registry = RunRegistry(csv_path)
    task2_registry = Task2RunRegistry(csv_path)

    task4_registry.append(_running_row("task4-run"))
    task2_registry.append(_record("task2-run"))

    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == RUN_COLUMNS
    assert RUN_COLUMNS == registry_module.RUN_COLUMNS
    assert task4_registry.read()[0]["run_id"] == "task4-run"
    assert task2_registry.read().loc[0, "run_id"] == "task2-run"


def test_run_ids_are_unique_across_task_registry_views(tmp_path: Path) -> None:
    csv_path = tmp_path / "runs.csv"
    RunRegistry(csv_path).append(_running_row("shared-id"))

    with pytest.raises(DuplicateRunError, match="shared-id"):
        Task2RunRegistry(csv_path).append(_record("shared-id"))


def test_task2_orphan_recovery_preserves_task4_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "runs.csv"
    task4_registry = RunRegistry(csv_path)
    task2_registry = Task2RunRegistry(csv_path)
    task4_registry.append(_running_row("task4-run"))
    task2_registry.append(_record("task2-run"))

    interrupt_orphaned_run(task2_registry, "task2-run", reason="host stopped")

    assert task4_registry.read()[0]["run_id"] == "task4-run"
    assert task2_registry.read().loc[0, "status"] == "interrupted"
