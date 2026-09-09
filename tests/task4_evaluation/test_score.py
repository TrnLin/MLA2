from __future__ import annotations

import inspect
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import fashion.task4_evaluation.audit as audit_module
import fashion.task4_evaluation.score as score_module
from fashion.data.hashing import compute_sha256
from fashion.task4_evaluation import (
    load_verified_holdout_evaluation,
    score_holdout,
)
from fashion.task4_evaluation.blind import build_blind_holdout_evidence
from fashion.task4_evaluation.encoders import R5_METHOD, RANDOM_FLOOR_METHOD
from fashion.train.artifacts import ArtifactVerificationError
from tests.task4_evaluation.test_blind import (
    CHECKPOINT_SHA256,
    METHODS,
    RUN_ID,
    _commit_fixture_change,
    _encoder_factory,
    _write_project,
)

FINAL_RELATIVE = Path("results/evidence/task4/final_evaluation")
FIGURE_RELATIVE = Path("results/figures/task4/final_evaluation")
APPROVAL_REF = "refs/tags/task4-holdout-scoring-approved-v1"


@pytest.fixture(autouse=True)
def _isolated_runtime_for_synthetic_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        score_module,
        "_runtime_is_isolated",
        lambda: True,
    )


def _write_synthetic_protected_labels(root: Path, splits: pd.DataFrame) -> None:
    holdout_ids = sorted(
        splits.loc[splits["partition"].eq("holdout"), "id"].astype(int).tolist()
    )
    undefined_ids = set(holdout_ids[-2:])
    raw = pd.DataFrame(
        {
            "id": splits["id"].astype(int),
            "gender": "Unisex",
            "articleType": [
                "Shoes" if int(product_id) in undefined_ids else "Tshirts"
                for product_id in splits["id"]
            ],
            "season": "Summer",
            "usage": "Casual",
        }
    )
    path = root / "data/raw/teacher/train/styles_train.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exclude_path = root / ".git/info/exclude"
    exclude_path.write_text(
        exclude_path.read_text(encoding="utf-8")
        + "\n/data/raw/teacher/train/styles_train.csv\n",
        encoding="utf-8",
    )
    raw.to_csv(path, index=False, lineterminator="\n")


def _prepare_blind_project(
    root: Path,
    *,
    approve: bool = True,
) -> dict[str, Any]:
    splits = _write_project(root)
    receipt = build_blind_holdout_evidence(
        project_root=root,
        encoder_factory=_encoder_factory({}, []),
    )
    _write_synthetic_protected_labels(root, splits)
    result = {"root": root, "splits": splits, "receipt": receipt}
    if approve:
        source_commit, evidence_commit, changed_paths = _commit_blind_evidence(root)
        result.update(
            source_commit=source_commit,
            evidence_commit=evidence_commit,
            changed_paths=changed_paths,
        )
    return result


def _create_approval_tag(root: Path, target: str = "HEAD") -> None:
    subprocess.run(
        ["git", "tag", APPROVAL_REF.removeprefix("refs/tags/"), target],
        cwd=root,
        check=True,
    )


def _move_approval_tag(root: Path, target: str = "HEAD") -> None:
    subprocess.run(
        ["git", "tag", "--force", APPROVAL_REF.removeprefix("refs/tags/"), target],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _commit_blind_evidence(
    root: Path,
    *,
    approve: bool = True,
) -> tuple[str, str, list[str]]:
    output = root / FINAL_RELATIVE
    expected_paths = sorted(
        [
            *(record["path"] for record in json.loads(
                (output / "prediction_receipt.json").read_text(encoding="utf-8")
            )["artifacts"].values()),
            (FINAL_RELATIVE / "prediction_receipt.json").as_posix(),
        ]
    )
    assert sorted(
        path.relative_to(root).as_posix()
        for path in output.iterdir()
        if path.is_file()
    ) == expected_paths
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _commit_fixture_change(root, output, "freeze blind evidence")
    evidence_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if approve:
        _create_approval_tag(root)
    return source_commit, evidence_commit, expected_paths


def _prepare_deployment_project(root: Path) -> dict[str, Any]:
    splits = _write_project(root)
    package = root / "models/task4_r5"
    weights = b"captured portable weights"
    (package / "weights.pt").write_bytes(weights)
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "task4_r5_inference_package",
        "method": "R5",
        "architecture": "resnet18",
        "objective": "content_mask_mse",
        "pretrained": False,
        "weight_origin": "random_initialization",
        "embedding_dim": 128,
        "source_checkpoint": {
            "run_id": RUN_ID,
            "score": 0.5,
            "sha256": CHECKPOINT_SHA256,
        },
        "weights": {
            "path": "weights.pt",
            "sha256": compute_sha256(package / "weights.pt"),
            "bytes": len(weights),
        },
        "normalization": {
            "teacher": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
            "v1": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
        },
    }
    model_manifest_path = package / "manifest.json"
    model_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _commit_fixture_change(root, package, "add deployment snapshot fixture")
    receipt = build_blind_holdout_evidence(
        project_root=root,
        encoder_factory=_encoder_factory({}, []),
    )
    _write_synthetic_protected_labels(root, splits)
    _commit_blind_evidence(root)
    gallery_directory = root / "models/task4_holdout_gallery"
    gallery_manifest_path = gallery_directory / "manifest.json"
    gallery_manifest = json.loads(gallery_manifest_path.read_text(encoding="utf-8"))
    return {
        "root": root,
        "splits": splits,
        "receipt": receipt,
        "weights": weights,
        "model_manifest_bytes": model_manifest_path.read_bytes(),
        "gallery_manifest_bytes": gallery_manifest_path.read_bytes(),
        "gallery_files": {
            name: (gallery_directory / record["path"]).read_bytes()
            for name, record in gallery_manifest["files"].items()
        },
    }


def _fake_deployment(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": R5_METHOD,
                "parameter_count": 11_176_512,
                "model_package_bytes": 44_800_000,
                "gallery_bytes": 20_000,
                "timed_queries": 4,
                "single_query_cpu_p50_ms": 10.0,
                "single_query_cpu_p95_ms": 12.0,
            }
        ]
    )


def _score_synthetic_project(root: Path) -> dict[str, Any]:
    _prepare_blind_project(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(score_module, "_runtime_is_isolated", lambda: True)
        patch.setattr(score_module, "_measure_deployment", _fake_deployment)
        manifest = score_holdout(
            evaluation_unlocked=True,
            project_root=root,
        )
    return {"root": root, "manifest": manifest}


@pytest.fixture(scope="module")
def scored_project(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _score_synthetic_project(tmp_path_factory.mktemp("task4-score-complete"))


def test_explicit_unlock_false_raises_before_any_io(tmp_path: Path) -> None:
    missing_root = tmp_path / "does-not-exist"
    with pytest.raises(
        ValueError,
        match="^holdout scoring requires evaluation_unlocked=True$",
    ):
        score_holdout(project_root=missing_root)


def test_score_api_has_no_isolation_bypass_parameter() -> None:
    assert "require_isolated_runtime" not in inspect.signature(score_holdout).parameters


def test_score_api_rejects_isolation_bypass_before_any_io(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="require_isolated_runtime"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path / "missing",
            require_isolated_runtime=False,
        )


def test_default_score_requires_isolated_python_before_any_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        score_module,
        "_runtime_is_isolated",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="isolated Python"):
        score_holdout(evaluation_unlocked=True, project_root=tmp_path / "missing")


def test_missing_scoring_approval_tag_fails_before_marker_or_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path, approve=False)
    _commit_blind_evidence(tmp_path, approve=False)
    raw_reads = 0

    def count_raw_reads(label: str, _path: Path) -> None:
        nonlocal raw_reads
        if label == "raw_teacher_csv":
            raw_reads += 1

    monkeypatch.setattr(score_module, "_after_verified_bytes_captured", count_raw_reads)
    with pytest.raises(RuntimeError, match="scoring approval ref is missing"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )

    assert raw_reads == 0
    assert not (tmp_path / FINAL_RELATIVE / "unlock_attempt.json").exists()


def test_scoring_approval_tag_at_wrong_commit_fails_before_marker_or_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _prepare_blind_project(tmp_path, approve=False)
    source_commit, _evidence_commit, _changed_paths = _commit_blind_evidence(
        tmp_path,
        approve=False,
    )
    assert source_commit == project["receipt"]["git"]["commit"]
    _create_approval_tag(tmp_path, source_commit)
    raw_reads = 0

    def count_raw_reads(label: str, _path: Path) -> None:
        nonlocal raw_reads
        if label == "raw_teacher_csv":
            raw_reads += 1

    monkeypatch.setattr(score_module, "_after_verified_bytes_captured", count_raw_reads)
    with pytest.raises(RuntimeError, match="does not approve current HEAD"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )

    assert raw_reads == 0
    assert not (tmp_path / FINAL_RELATIVE / "unlock_attempt.json").exists()


def test_untracked_python_shadow_fails_before_marker_or_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    (tmp_path / "sitecustomize.py").write_text(
        "raise RuntimeError('shadowed')\n",
        encoding="utf-8",
    )
    raw_reads = 0

    def count_raw_reads(label: str, _path: Path) -> None:
        nonlocal raw_reads
        if label == "raw_teacher_csv":
            raw_reads += 1

    monkeypatch.setattr(score_module, "_after_verified_bytes_captured", count_raw_reads)
    with pytest.raises(RuntimeError, match="nonignored changes or untracked files"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )

    assert raw_reads == 0
    assert not (tmp_path / FINAL_RELATIVE / "unlock_attempt.json").exists()


def test_post_blind_gate_rejects_symbolic_source_before_git_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        score_module,
        "_git_state",
        lambda _root: {"commit": "a" * 40, "tracked_files_dirty": False},
    )

    def unexpected_git(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("malformed source must fail before git commands")

    monkeypatch.setattr(score_module.subprocess, "run", unexpected_git)
    with pytest.raises(RuntimeError, match="blind source commit is invalid"):
        score_module._verify_post_blind_git_state(
            root=tmp_path,
            receipt={"git": {"commit": "HEAD"}, "artifacts": {}},
            receipt_path=tmp_path / FINAL_RELATIVE / "prediction_receipt.json",
        )


def test_committed_blind_evidence_passes_post_blind_git_gate(tmp_path: Path) -> None:
    project = _prepare_blind_project(tmp_path)
    source_commit = project["source_commit"]
    evidence_commit = project["evidence_commit"]
    expected_paths = project["changed_paths"]

    package = score_module._verify_blind_package(
        root=tmp_path,
        receipt_path=tmp_path / FINAL_RELATIVE / "prediction_receipt.json",
    )

    assert package.receipt["git"]["commit"] == source_commit
    assert package.git_state == {
        "commit": evidence_commit,
        "tracked_files_dirty": False,
    }
    assert package.blind_source_commit == source_commit
    assert package.post_blind_changed_paths == expected_paths
    assert package.scoring_approval_ref == APPROVAL_REF
    assert package.scoring_approval_commit == evidence_commit


def test_committed_disallowed_runtime_change_fails_before_marker_or_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    runtime_path = tmp_path / "src/fashion/task4_evaluation/encoders.py"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text("# disallowed post-blind runtime change\n", encoding="utf-8")
    _commit_fixture_change(tmp_path, runtime_path, "change runtime after blind prediction")
    _move_approval_tag(tmp_path)
    raw_reads = 0

    def count_raw_reads(label: str, _path: Path) -> None:
        nonlocal raw_reads
        if label == "raw_teacher_csv":
            raw_reads += 1

    monkeypatch.setattr(
        score_module,
        "_after_verified_bytes_captured",
        count_raw_reads,
    )
    with pytest.raises(RuntimeError, match="post-blind changed paths are not allowed"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )

    assert raw_reads == 0
    assert not (tmp_path / FINAL_RELATIVE / "unlock_attempt.json").exists()


def test_non_ancestor_blind_source_commit_fails_before_marker_or_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path, approve=False)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    unrelated_commit = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task 4 Test",
            "-c",
            "user.email=task4@example.test",
            "commit-tree",
            tree,
            "-m",
            "unrelated blind source",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt_path = tmp_path / FINAL_RELATIVE / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["git"]["commit"] = unrelated_commit
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _commit_blind_evidence(tmp_path)
    raw_reads = 0

    def count_raw_reads(label: str, _path: Path) -> None:
        nonlocal raw_reads
        if label == "raw_teacher_csv":
            raw_reads += 1

    monkeypatch.setattr(
        score_module,
        "_after_verified_bytes_captured",
        count_raw_reads,
    )
    with pytest.raises(RuntimeError, match="blind source commit is not an ancestor"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )

    assert raw_reads == 0
    assert not (tmp_path / FINAL_RELATIVE / "unlock_attempt.json").exists()


def test_successful_committed_score_records_and_audits_git_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _prepare_blind_project(tmp_path)
    source_commit = project["source_commit"]
    evidence_commit = project["evidence_commit"]
    changed_paths = project["changed_paths"]
    monkeypatch.setattr(score_module, "_measure_deployment", _fake_deployment)

    manifest = score_holdout(
        evaluation_unlocked=True,
        project_root=tmp_path,
    )

    output = tmp_path / FINAL_RELATIVE
    attempt = json.loads((output / "unlock_attempt.json").read_text(encoding="utf-8"))
    unlock = json.loads((output / "unlock_receipt.json").read_text(encoding="utf-8"))
    expected_git = {"commit": evidence_commit, "tracked_files_dirty": False}
    for payload in (attempt, unlock, manifest):
        assert payload["git"] == expected_git
        assert payload["blind_source_commit"] == source_commit
        assert payload["post_blind_changed_paths"] == changed_paths
        assert payload["scoring_approval_ref"] == APPROVAL_REF
        assert payload["scoring_approval_commit"] == evidence_commit
    assert load_verified_holdout_evaluation(project_root=tmp_path) == manifest


@pytest.mark.parametrize(
    "marker",
    ["unlock_attempt.json", "unlock_receipt.json", "evaluation_manifest.json"],
)
def test_later_state_refuses_one_way_scoring_before_label_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    output = tmp_path / FINAL_RELATIVE
    output.mkdir(parents=True)
    (output / marker).write_text("{}\n", encoding="utf-8")

    def unexpected_label_read(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("protected labels must stay sealed")

    monkeypatch.setattr(
        score_module,
        "load_splits_for_final_evaluation",
        unexpected_label_read,
    )
    with pytest.raises(RuntimeError, match="one-way"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )


def test_live_ranking_mutation_after_capture_fails_clean_gate_before_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    mutated = False

    def mutate_after_capture(label: str, path: Path) -> None:
        nonlocal mutated
        if label == "holdout_primary_rankings" and not mutated:
            path.write_bytes(b"not,the,verified,rankings\n")
            mutated = True

    monkeypatch.setattr(
        score_module,
        "_after_verified_bytes_captured",
        mutate_after_capture,
        raising=False,
    )
    monkeypatch.setattr(score_module, "_measure_deployment", _fake_deployment)
    with pytest.raises(RuntimeError, match="nonignored changes"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )
    assert mutated is True
    assert not (tmp_path / FINAL_RELATIVE / "unlock_attempt.json").exists()


def test_split_and_raw_snapshots_parse_captured_bytes_not_changed_paths(
    tmp_path: Path,
) -> None:
    project = _prepare_blind_project(tmp_path)
    split_path = tmp_path / "data/processed/splits.csv"
    raw_path = tmp_path / "data/raw/teacher/train/styles_train.csv"
    split_bytes = split_path.read_bytes()
    raw_bytes = raw_path.read_bytes()

    split_path.write_text("broken split\n", encoding="utf-8")
    raw_path.write_text("id,articleType\n101,Changed\n", encoding="utf-8")
    unlocked = score_module._load_unlocked_from_snapshot_bytes(split_bytes, raw_bytes)

    expected_ids = set(
        project["splits"].loc[
            project["splits"]["partition"].eq("holdout"), "id"
        ].astype(int)
    )
    holdout = unlocked.loc[unlocked["partition"].eq("holdout")]
    assert set(holdout["id"].astype(int)) == expected_ids
    assert set(holdout["articleType"]) == {"Tshirts", "Shoes"}


def test_deployment_uses_captured_model_gallery_snapshots_after_live_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _prepare_deployment_project(tmp_path)
    original_package = tmp_path / "models/task4_r5"
    original_gallery = tmp_path / "models/task4_holdout_gallery"
    mutated: set[str] = set()

    class _Parameter:
        def numel(self) -> int:
            return 123

    class _SnapshotModel:
        def parameters(self) -> list[_Parameter]:
            return [_Parameter()]

        def encode(self, batch: torch.Tensor) -> torch.Tensor:
            result = torch.ones((len(batch), 128), dtype=torch.float32)
            return result / torch.linalg.vector_norm(result, dim=1, keepdim=True)

    def load_snapshot(package: Path, *, device: str) -> _SnapshotModel:
        assert package.resolve() != original_package.resolve()
        assert device == "cpu"
        assert (package / "manifest.json").read_bytes() == project["model_manifest_bytes"]
        assert (package / "weights.pt").read_bytes() == project["weights"]
        return _SnapshotModel()

    def mutate_live_after_capture(label: str, path: Path) -> None:
        if label == "deployment_model_weights" and label not in mutated:
            path.write_bytes(b"replacement weights")
            mutated.add(label)
        if label == "deployment_gallery_features.npy" and label not in mutated:
            path.write_bytes(b"replacement gallery features")
            mutated.add(label)

    monkeypatch.setattr(
        score_module,
        "load_r5_inference_package",
        load_snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        score_module,
        "_after_verified_bytes_captured",
        mutate_live_after_capture,
    )
    manifest = score_holdout(
        evaluation_unlocked=True,
        project_root=tmp_path,
    )
    assert manifest["status"] == "complete"
    assert mutated == {
        "deployment_model_weights",
        "deployment_gallery_features.npy",
    }
    deployment = pd.read_csv(tmp_path / FINAL_RELATIVE / "deployment_summary.csv").iloc[0]
    assert int(deployment["parameter_count"]) == 123
    assert int(deployment["model_package_bytes"]) == (
        len(project["model_manifest_bytes"]) + len(project["weights"])
    )
    assert int(deployment["gallery_bytes"]) == (
        len(project["gallery_manifest_bytes"])
        + sum(len(payload) for payload in project["gallery_files"].values())
    )
    assert (original_gallery / "features.npy").read_bytes() == b"replacement gallery features"


def test_unlock_attempt_claim_has_exactly_one_concurrent_winner(
    tmp_path: Path,
) -> None:
    output = tmp_path / FINAL_RELATIVE
    output.mkdir(parents=True)
    attempt_path = output / "unlock_attempt.json"
    payload = {
        "schema_version": "1.0.0",
        "evaluation_id": "race-test",
        "state": "label_access_started",
    }
    barrier = Barrier(2)

    def claim() -> str:
        barrier.wait()
        try:
            score_module._claim_unlock_attempt(
                attempt_path,
                payload,
                root=tmp_path,
            )
        except RuntimeError:
            return "blocked"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: claim(), range(2)))
    assert sorted(outcomes) == ["blocked", "claimed"]
    assert attempt_path.is_file()


def test_two_production_scorers_have_one_winner_and_one_raw_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    barrier = Barrier(2)
    count_lock = Lock()
    raw_reads = 0
    boundary_calls = 0

    class _WinnerStopped(RuntimeError):
        pass

    def meet_before_claim() -> None:
        nonlocal boundary_calls
        with count_lock:
            boundary_calls += 1
        try:
            barrier.wait(timeout=15.0)
        except BrokenBarrierError as error:
            raise AssertionError(
                "both scorers did not reach the pre-claim barrier within 15 seconds"
            ) from error

    def count_raw_read(label: str, _path: Path) -> None:
        nonlocal raw_reads
        if label == "raw_teacher_csv":
            with count_lock:
                raw_reads += 1

    def stop_winner(_split_bytes: bytes, _raw_bytes: bytes) -> pd.DataFrame:
        raise _WinnerStopped("winner stopped after raw capture")

    monkeypatch.setattr(
        score_module,
        "_before_unlock_attempt_claim",
        meet_before_claim,
        raising=False,
    )
    monkeypatch.setattr(
        score_module,
        "_after_verified_bytes_captured",
        count_raw_read,
    )
    monkeypatch.setattr(
        score_module,
        "_load_unlocked_from_snapshot_bytes",
        stop_winner,
    )

    def run() -> str:
        try:
            score_holdout(
                evaluation_unlocked=True,
                project_root=tmp_path,
            )
        except _WinnerStopped:
            return "winner"
        except RuntimeError as error:
            assert "unlock attempt already claimed" in str(error)
            return "loser"
        raise AssertionError("scorer unexpectedly completed")

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: run(), range(2)))
    assert sorted(outcomes) == ["loser", "winner"]
    assert boundary_calls == 2
    assert raw_reads == 1
    assert (tmp_path / FINAL_RELATIVE / "unlock_attempt.json").is_file()
    with pytest.raises(RuntimeError, match="one-way"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )
    assert raw_reads == 1


def test_loader_failure_leaves_attempt_and_blocks_retry_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    loader_calls = 0
    raw_reads = 0

    def failing_loader(_split_bytes: bytes, _raw_bytes: bytes) -> pd.DataFrame:
        nonlocal loader_calls
        loader_calls += 1
        raise RuntimeError("forced snapshot loader failure")

    def count_raw_reads(label: str, _path: Path) -> None:
        nonlocal raw_reads
        if label == "raw_teacher_csv":
            raw_reads += 1

    monkeypatch.setattr(
        score_module,
        "_load_unlocked_from_snapshot_bytes",
        failing_loader,
        raising=False,
    )
    monkeypatch.setattr(
        score_module,
        "_after_verified_bytes_captured",
        count_raw_reads,
    )
    with pytest.raises(RuntimeError, match="forced snapshot loader failure"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )

    output = tmp_path / FINAL_RELATIVE
    assert (output / "unlock_attempt.json").is_file()
    assert not (output / "unlock_receipt.json").exists()
    with pytest.raises(RuntimeError, match="one-way"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )
    assert loader_calls == 1
    assert raw_reads == 1


def test_changed_blind_csv_fails_before_labels_are_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _prepare_blind_project(tmp_path)
    record = project["receipt"]["artifacts"]["holdout_primary_rankings"]
    ranking_path = tmp_path / record["path"]
    ranking_path.write_bytes(ranking_path.read_bytes() + b"changed")

    def unexpected_label_read(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("protected labels must stay sealed")

    monkeypatch.setattr(
        score_module,
        "load_splits_for_final_evaluation",
        unexpected_label_read,
    )
    with pytest.raises(ArtifactVerificationError, match="holdout_primary_rankings"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )


def test_changed_file_valued_input_fails_before_labels_are_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    split_path = tmp_path / "data/processed/splits.csv"
    split_path.write_bytes(split_path.read_bytes() + b"\n")

    def unexpected_label_read(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("protected labels must stay sealed")

    monkeypatch.setattr(
        score_module,
        "load_splits_for_final_evaluation",
        unexpected_label_read,
    )
    with pytest.raises(ArtifactVerificationError, match="splits"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )


def test_file_input_records_are_bound_to_their_exact_paths_before_label_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    receipt_path = tmp_path / FINAL_RELATIVE / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split_path = tmp_path / "data/processed/splits.csv"
    decoy_path = tmp_path / "data/processed/decoy-splits.csv"
    decoy_path.write_bytes(split_path.read_bytes())
    receipt["inputs"]["splits"]["path"] = "data/processed/decoy-splits.csv"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def unexpected_label_read(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("protected labels must stay sealed")

    monkeypatch.setattr(
        score_module,
        "load_splits_for_final_evaluation",
        unexpected_label_read,
    )
    with pytest.raises(ArtifactVerificationError, match="splits.*path"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )


def test_incomplete_ranking_is_rejected_even_if_receipt_hash_is_rewritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    receipt_path = tmp_path / FINAL_RELATIVE / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = receipt["artifacts"]["holdout_family_rankings"]
    ranking_path = tmp_path / record["path"]
    rankings = pd.read_csv(ranking_path)
    rankings.iloc[1:].to_csv(ranking_path, index=False, lineterminator="\n")
    record["sha256"] = compute_sha256(ranking_path)
    record["bytes"] = ranking_path.stat().st_size
    record["rows"] = len(rankings) - 1
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def unexpected_label_read(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("protected labels must stay sealed")

    monkeypatch.setattr(
        score_module,
        "load_splits_for_final_evaluation",
        unexpected_label_read,
    )
    with pytest.raises(ValueError, match="complete consecutive ranks"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )


def test_duplicate_gallery_manifest_row_is_not_exact_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    receipt_path = tmp_path / FINAL_RELATIVE / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = receipt["artifacts"]["gallery_manifest"]
    gallery_path = tmp_path / record["path"]
    gallery = pd.read_csv(gallery_path)
    gallery = pd.concat([gallery, gallery.iloc[[0]]], ignore_index=True)
    gallery.to_csv(gallery_path, index=False, lineterminator="\n")
    record["sha256"] = compute_sha256(gallery_path)
    record["bytes"] = gallery_path.stat().st_size
    record["rows"] = len(gallery)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def unexpected_label_read(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("protected labels must stay sealed")

    monkeypatch.setattr(
        score_module,
        "load_splits_for_final_evaluation",
        unexpected_label_read,
    )
    with pytest.raises(ValueError, match="gallery manifest row coverage"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )


def test_unlock_receipt_survives_post_unlock_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)

    def forced_failure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("forced post-unlock failure")

    monkeypatch.setattr(score_module, "_calculate_scored_evidence", forced_failure)
    with pytest.raises(RuntimeError, match="forced post-unlock failure"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )

    unlock_path = tmp_path / FINAL_RELATIVE / "unlock_receipt.json"
    assert unlock_path.is_file()
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    assert unlock["holdout_opened"] is True
    assert not (tmp_path / FINAL_RELATIVE / "evaluation_manifest.json").exists()


def test_source_change_during_unlock_stops_metrics_but_keeps_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_blind_project(tmp_path)
    original_git_state = score_module._git_state
    calls = 0

    def changing_git_state(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        state = original_git_state(root)
        if calls == 2:
            state["tracked_files_dirty"] = True
        return state

    def unexpected_metrics(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("metrics must not run after source changes")

    monkeypatch.setattr(score_module, "_git_state", changing_git_state)
    monkeypatch.setattr(score_module, "_calculate_scored_evidence", unexpected_metrics)
    with pytest.raises(RuntimeError, match="tracked source changed during holdout unlock"):
        score_holdout(
            evaluation_unlocked=True,
            project_root=tmp_path,
        )
    assert (tmp_path / FINAL_RELATIVE / "unlock_receipt.json").is_file()


def test_scorecard_has_nine_frozen_clean_combinations(
    scored_project: dict[str, Any],
) -> None:
    scorecard = pd.read_csv(scored_project["root"] / FINAL_RELATIVE / "holdout_scorecard.csv")
    assert len(scorecard) == 9
    assert (
        scorecard.loc[scorecard["method"].eq(RANDOM_FLOOR_METHOD), "direction"].tolist()
        == ["teacher"]
    )
    assert len(scorecard.loc[scorecard["method"].ne(RANDOM_FLOOR_METHOD)]) == 8


def test_undefined_queries_stay_nan_and_are_counted_excluded(
    scored_project: dict[str, Any],
) -> None:
    per_query = pd.read_csv(
        scored_project["root"] / FINAL_RELATIVE / "holdout_per_query_primary.csv"
    )
    selected = per_query.loc[
        per_query["method"].eq(R5_METHOD)
        & per_query["direction"].eq("teacher")
        & per_query["condition"].eq("clean")
    ]
    assert selected["ndcg_at_10"].isna().sum() == 2

    scorecard = pd.read_csv(scored_project["root"] / FINAL_RELATIVE / "holdout_scorecard.csv")
    r5_teacher = scorecard.loc[
        scorecard["method"].eq(R5_METHOD) & scorecard["direction"].eq("teacher")
    ].iloc[0]
    assert int(r5_teacher["scored_queries"]) == 10
    assert int(r5_teacher["excluded_queries"]) == 2


def test_r5_bootstrap_interval_brackets_its_point_estimate(
    scored_project: dict[str, Any],
) -> None:
    intervals = pd.read_csv(
        scored_project["root"] / FINAL_RELATIVE / "holdout_bootstrap_intervals.csv"
    )
    r5 = intervals.loc[intervals["metric"].eq("r5_mean_ndcg_at_10")].iloc[0]
    assert r5["lower_95"] <= r5["median"] <= r5["upper_95"]


def test_known_better_r5_has_positive_r5_minus_comparator_median() -> None:
    query_ids = [1, 2, 3, 4]
    groups = pd.DataFrame(
        {
            "id": query_ids,
            "product_family_group": ["a", "a", "b", "b"],
        }
    )
    rows = []
    for method in METHODS:
        score = 1.0 if method == R5_METHOD else 0.25
        rows.extend(
            {
                "method": method,
                "direction": "teacher",
                "condition": "clean",
                "query_id": query_id,
                "ndcg_at_10": score,
            }
            for query_id in query_ids
        )
    spec = SimpleNamespace(
        methods=METHODS,
        bootstrap_replicates=100,
        bootstrap_seed=2753,
    )
    intervals = score_module._bootstrap_intervals(
        pd.DataFrame(rows),
        groups,
        spec,
    )
    comparator = intervals.loc[
        intervals["metric"].eq(
            "r5_minus_spatial_hsv_edge_probe_ndcg_at_10"
        )
    ].iloc[0]
    assert comparator["median"] > 0


def test_selective_retrieval_breaks_distance_ties_by_query_id() -> None:
    ranking_rows = []
    for query_id in (1, 2, 3, 4):
        ranking_rows.extend(
            [
                {
                    "method": R5_METHOD,
                    "direction": "teacher",
                    "condition": "clean",
                    "query_id": query_id,
                    "candidate_id": 100 + query_id,
                    "distance": 0.5,
                    "rank": 1,
                },
                {
                    "method": R5_METHOD,
                    "direction": "teacher",
                    "condition": "clean",
                    "query_id": query_id,
                    "candidate_id": 200 + query_id,
                    "distance": 0.6,
                    "rank": 2,
                },
            ]
        )
    primary = pd.DataFrame(
        {
            "method": R5_METHOD,
            "direction": "teacher",
            "condition": "clean",
            "query_id": [1, 2, 3, 4],
            "ndcg_at_10": [0.1, 0.2, 0.9, 1.0],
        }
    )
    family = pd.DataFrame(
        {
            "method": R5_METHOD,
            "direction": "teacher",
            "query_id": [1, 2, 3, 4],
            "recall_at_10": [0.2, 0.4, 0.8, 1.0],
        }
    )
    selective = score_module._selective_retrieval(
        pd.DataFrame(ranking_rows),
        primary,
        family,
    )
    half = selective.loc[selective["target_coverage"].eq(0.5)].iloc[0]
    assert half["selective_ndcg_at_10"] == pytest.approx(0.15)
    assert half["selective_recall_at_10"] == pytest.approx(0.3)


def test_development_reference_is_kept_but_not_compared(
    scored_project: dict[str, Any],
) -> None:
    scorecard = pd.read_csv(scored_project["root"] / FINAL_RELATIVE / "holdout_scorecard.csv")
    r5 = scorecard.loc[scorecard["method"].eq(R5_METHOD)]
    assert r5["development_reference_ndcg_at_10"].notna().all()
    assert r5["development_reference_comparable"].eq(False).all()  # noqa: E712
    assert r5["holdout_minus_development_ndcg_at_10"].isna().all()


def test_random_source_robustness_is_explicitly_not_applicable(
    scored_project: dict[str, Any],
) -> None:
    source = pd.read_csv(
        scored_project["root"] / FINAL_RELATIVE / "holdout_source_robustness.csv"
    )
    random = source.loc[source["method"].eq(RANDOM_FLOOR_METHOD)].iloc[0]
    assert np.isnan(random["holdout_source_robustness_ratio"])
    assert (
        random["status"]
        == "not_applicable_source_independent_single_frozen_ranking"
    )


def test_six_figures_are_nonempty_and_manifest_tracked(
    scored_project: dict[str, Any],
) -> None:
    figure_names = {
        "holdout_scorecard.png",
        "holdout_bootstrap_intervals.png",
        "holdout_selective_retrieval.png",
        "holdout_slices_robustness.png",
        "holdout_error_examples.png",
        "holdout_source_robustness.png",
    }
    figure_paths = [scored_project["root"] / FIGURE_RELATIVE / name for name in figure_names]
    assert all(path.is_file() and path.stat().st_size > 0 for path in figure_paths)
    tracked = {
        Path(record["path"]).name
        for record in scored_project["manifest"]["artifacts"].values()
    }
    assert figure_names <= tracked
    assert len(scored_project["manifest"]["artifacts"]) == 20


def test_complete_manifest_and_receipts_agree_on_no_change_flags(
    scored_project: dict[str, Any],
) -> None:
    root = scored_project["root"]
    manifest = scored_project["manifest"]
    prediction = json.loads(
        (root / FINAL_RELATIVE / "prediction_receipt.json").read_text(encoding="utf-8")
    )
    unlock = json.loads(
        (root / FINAL_RELATIVE / "unlock_receipt.json").read_text(encoding="utf-8")
    )
    attempt = json.loads(
        (root / FINAL_RELATIVE / "unlock_attempt.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert attempt["state"] == "label_access_started"
    assert unlock["unlock_attempt"] == manifest["unlock_attempt"]
    assert prediction["model_changed"] is False
    assert prediction["retuning_allowed"] is False
    for key in (
        "model_retrained",
        "winner_changed",
        "metric_changed",
        "conditions_changed",
        "retuning_allowed",
    ):
        assert unlock[key] is False
        assert manifest["no_change_after_unlock"][key] is False
    assert unlock["holdout_opened"] is True


def test_audit_fails_after_one_byte_manifest_tracked_mutation(tmp_path: Path) -> None:
    project = _score_synthetic_project(tmp_path)
    verified = load_verified_holdout_evaluation(project_root=tmp_path)
    assert verified["status"] == "complete"
    record = project["manifest"]["artifacts"]["holdout_scorecard"]
    path = tmp_path / record["path"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ArtifactVerificationError, match="holdout_scorecard"):
        load_verified_holdout_evaluation(project_root=tmp_path)


def test_audit_fails_after_one_byte_unlock_attempt_mutation(tmp_path: Path) -> None:
    project = _score_synthetic_project(tmp_path)
    verified = load_verified_holdout_evaluation(project_root=tmp_path)
    assert verified["status"] == "complete"
    attempt_path = tmp_path / project["manifest"]["unlock_attempt"]["path"]
    attempt_path.write_bytes(attempt_path.read_bytes() + b"x")
    with pytest.raises(ArtifactVerificationError, match="unlock_attempt"):
        load_verified_holdout_evaluation(project_root=tmp_path)


def test_audit_fails_after_one_byte_blind_artifact_mutation(tmp_path: Path) -> None:
    _score_synthetic_project(tmp_path)
    verified = load_verified_holdout_evaluation(project_root=tmp_path)
    assert verified["status"] == "complete"
    receipt_path = tmp_path / FINAL_RELATIVE / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    record = receipt["artifacts"]["runtime"]
    path = tmp_path / record["path"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ArtifactVerificationError, match="runtime"):
        load_verified_holdout_evaluation(project_root=tmp_path)


def test_audit_parses_captured_receipt_bytes_after_live_path_mutates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _score_synthetic_project(tmp_path)
    mutated = False

    def mutate_after_capture(label: str, path: Path) -> None:
        nonlocal mutated
        if label == "unlock_receipt" and not mutated:
            path.write_text('{"replacement": true}\n', encoding="utf-8")
            mutated = True

    monkeypatch.setattr(
        audit_module,
        "_after_audit_bytes_captured",
        mutate_after_capture,
    )
    manifest = load_verified_holdout_evaluation(project_root=tmp_path)
    assert mutated is True
    assert manifest["status"] == "complete"
