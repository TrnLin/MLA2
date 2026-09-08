"""Regression checks use copied evidence; they never unlock the real holdout."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from filelock import FileLock

from fashion.config import ROOT
from fashion.task2 import final_evaluation_runner as runner
from fashion.task2.final_evaluation import load_final_evaluation_spec
from fashion.train.artifacts import ArtifactVerificationError


@pytest.fixture
def evidence_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    evidence = runner.FINAL_EVALUATION_DIR
    receipt = json.loads(runner.PREDICTION_RECEIPT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(runner.EVALUATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    paths = set(evidence.iterdir())
    for ledger in (receipt["artifacts"], receipt["inputs"], manifest["artifacts"]):
        for record in ledger.values():
            if isinstance(record, dict):
                paths.add(ROOT / record["path"])
    paths.update(
        {
            runner.TASK2_SELECTION_FREEZE_JSON,
            ROOT / "results/evidence/task2/b0_majority/pooled_metrics.json",
        }
    )
    for source in paths:
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in (
        "FINAL_EVALUATION_DIR", "FINAL_EVALUATION_FIGURE_DIR", "PREDICTION_RECEIPT_PATH",
        "UNLOCK_RECEIPT_PATH", "EVALUATION_MANIFEST_PATH", "TASK2_SELECTION_FREEZE_JSON",
        "SPLITS_CSV", "LABEL_MAPS_JSON", "TASK2_MODEL_MANIFEST_JSON", "FROZEN_REGISTRY_PATH",
    ):
        monkeypatch.setattr(runner, name, tmp_path / getattr(runner, name).relative_to(ROOT))
    spec = load_final_evaluation_spec()
    monkeypatch.setattr(
        runner, "load_final_evaluation_spec",
        lambda **kwargs: replace(
            spec, config_path=tmp_path / spec.config_path.relative_to(ROOT),
            official_output_path=tmp_path / spec.official_output_path.relative_to(ROOT),
        ),
    )
    return tmp_path


@pytest.mark.parametrize("phase", ["predict", "score"])
def test_completed_evaluation_rejects_reexecution_before_data_access(
    evidence_copy: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("one-shot guard must run before loading data or opening labels")

    monkeypatch.setattr(runner, "load_splits", forbidden)
    monkeypatch.setattr(runner, "load_splits_for_final_evaluation", forbidden)
    with pytest.raises(ValueError, match="already|existing|one-shot"):
        if phase == "predict":
            runner.build_blind_prediction_evidence(project_root=evidence_copy)
        else:
            runner.score_internal_holdout(evaluation_unlocked=True, project_root=evidence_copy)


def test_replay_verifies_nested_blind_artifacts(evidence_copy: Path) -> None:
    path = runner.FINAL_EVALUATION_DIR / "holdout_b0_predictions.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ArtifactVerificationError, match="hash|SHA|sha|mismatch"):
        runner.load_verified_final_evaluation(project_root=evidence_copy)


def test_replay_rejects_changed_legacy_scorecard(evidence_copy: Path) -> None:
    path = runner.FINAL_EVALUATION_DIR / "holdout_scorecard.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["model"].eq("I2 frozen"), "accuracy"] = 0.999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="scorecard"):
        runner.load_verified_final_evaluation(project_root=evidence_copy)


def test_replay_does_not_need_raw_labels_or_images(evidence_copy: Path) -> None:
    assert not (evidence_copy / "data/raw").exists()
    assert runner.load_verified_final_evaluation(project_root=evidence_copy)["status"] == "complete"


@pytest.fixture
def empty_destination(tmp_path, monkeypatch):
    directory = tmp_path / "evaluation"
    monkeypatch.setattr(runner, "FINAL_EVALUATION_DIR", directory)
    monkeypatch.setattr(runner, "FINAL_EVALUATION_FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(runner, "PREDICTION_RECEIPT_PATH", directory / "prediction_receipt.json")
    monkeypatch.setattr(runner, "UNLOCK_RECEIPT_PATH", directory / "unlock_receipt.json")
    monkeypatch.setattr(runner, "EVALUATION_MANIFEST_PATH", directory / "evaluation_manifest.json")
    monkeypatch.setattr(runner, "load_final_evaluation_spec", lambda **kw: SimpleNamespace(
        official_output_path=tmp_path / "official.csv", evaluation_id="unit-test",
    ))
    return tmp_path


def test_interrupted_prediction_cannot_restart(empty_destination):
    calls = []

    @runner._one_shot_phase("predict")
    def failing(**kwargs):
        calls.append(True)
        raise RuntimeError("simulated process failure")

    with pytest.raises(RuntimeError, match="simulated"):
        failing(project_root=empty_destination)
    assert (runner.FINAL_EVALUATION_DIR / "prediction_started.json").is_file()
    with pytest.raises(ValueError, match="existing"):
        failing(project_root=empty_destination)
    assert len(calls) == 1


def test_concurrent_evaluation_writer_is_rejected(empty_destination):
    @runner._one_shot_phase("predict")
    def forbidden(**kwargs):
        pytest.fail("concurrent writer entered prediction")

    with FileLock(str(runner.FINAL_EVALUATION_DIR.with_suffix(".lock"))):
        with pytest.raises(ValueError, match="already active"):
            forbidden(project_root=empty_destination)
    assert not runner.FINAL_EVALUATION_DIR.exists()


def test_missing_artifact_ledger_entry_is_rejected(evidence_copy):
    path = runner.EVALUATION_MANIFEST_PATH
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["holdout_metrics"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger is incomplete"):
        runner.load_verified_final_evaluation(project_root=evidence_copy)
