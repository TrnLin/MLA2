"""Epoch-25 SAM must reproduce the original prefix and verify its shorter checkpoint."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from test_task3_gender_sam import CLASSES, synthetic_gender  # noqa: F401

torch = pytest.importorskip("torch")

from fashion.train import task3_gender_sam as original  # noqa: E402
from fashion.train import task3_gender_sam25 as screen  # noqa: E402
from fashion.train.config import Task3BaselineConfig  # noqa: E402
from fashion.train.sam import POLICY, SAMStep, policy_for_epochs  # noqa: E402
from fashion.train.task3_g2_audit import inspect_gender_run  # noqa: E402


def test_only_stopping_budget_and_diagnostics_change(monkeypatch):
    monkeypatch.setattr(screen, "label_contract", lambda root: {})
    spec = screen.sam25_spec()
    old = original.SAMSpec("{}")
    changed = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "screen_rule_version",
        "training_epochs",
        "cosine_t_max",
        "selection_basis",
        "sam_policy",
    }
    assert {k: v for k, v in spec.to_dict().items() if k not in changed} == {
        k: v for k, v in old.to_dict().items() if k not in changed
    }
    assert screen.sam25_config(spec, fold=0, device_name="cuda").epochs == 25
    assert spec.to_dict()["cosine_t_max"] == 30
    assert policy_for_epochs(30) == POLICY
    assert {
        k: v for k, v in policy_for_epochs(25).items() if k not in {"version", "diagnostic_epochs"}
    } == {k: v for k, v in POLICY.items() if k not in {"version", "diagnostic_epochs"}}
    for fold, device in ((1, "cuda"), (0, "cpu")):
        with pytest.raises(ValueError, match="only folds"):
            screen.sam25_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="frozen recipe"):
        screen.sam25_config(old, fold=0, device_name="cuda")
    with pytest.raises(ValueError, match="25- or 30-epoch"):
        policy_for_epochs(24)


def test_public_screen_requests_exactly_25_epoch_artifacts(monkeypatch):
    spec = screen.SAM25Spec("{}")
    monkeypatch.setattr(screen, "check_gender_sam25_sources", lambda **kw: ({}, CLASSES, spec, {}))
    monkeypatch.setattr(screen, "sam25_config", lambda *a, **kw: None)
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: None)
    monkeypatch.setattr(screen, "_source_identity", lambda *a, **kw: {})
    monkeypatch.setattr(screen, "training_splits", lambda *a, **kw: "canonical")
    monkeypatch.setattr(screen, "_run_verified_label_screen", lambda **kw: kw)
    kwargs = screen.run_gender_sam25_screen(
        output_root="output",
        registry_path="runs",
        source_registry_path="source",
        precision_directory="precision",
    )
    assert kwargs["candidate_epochs"] == 25
    assert kwargs["parent_group"] == "MixUp20"
    assert kwargs["verify_candidate"] is screen.verify_sam25_evidence
    assert kwargs["refine_report"] is original.apply_improvement_rules


def test_25_epoch_run_matches_original_weights_and_is_auditable(
    synthetic_gender,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from fashion.train import task3_baseline as engine

    torch.set_num_threads(2)
    captured = {}
    end_epoch = SAMStep.end_epoch
    step = SAMStep.step
    registry_path = tmp_path / "runs.csv"

    def capture(self, mixed):
        stats = end_epoch(self, mixed)
        if stats["epoch"] == 25 and self.policy == POLICY:
            captured.update({k: v.detach().clone() for k, v in self.model.state_dict().items()})
        return stats

    def registered_step(self, *a, **kw):
        rows = pd.read_csv(registry_path)
        assert rows.status.eq("running").sum() == 1
        return step(self, *a, **kw)

    monkeypatch.setattr(SAMStep, "end_epoch", capture)
    monkeypatch.setattr(SAMStep, "step", registered_step)
    spec25 = screen.SAM25Spec("{}")
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "metrics.json").write_text(
        json.dumps({"run_id": spec25.parent_run_id_for_fold(0), "validation_fold": 0})
    )
    for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
        (parent / name).write_text("Never load these comparison weights")
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    results = {}
    for epochs, module, spec in ((30, original, original.SAMSpec("{}")), (25, screen, spec25)):
        config = replace(Task3BaselineConfig(target="gender"), epochs=epochs, num_workers=0)
        monkeypatch.setattr(engine, "Task3BaselineConfig", lambda config=config, **kw: config)
        monkeypatch.setattr(
            module,
            "sam_config" if epochs == 30 else "sam25_config",
            lambda *a, config=config, **kw: config,
        )
        monkeypatch.setattr(module, "training_splits", lambda *a, **kw: synthetic_gender)
        monkeypatch.setattr(
            module,
            "require_sam_prerequisites" if epochs == 30 else "require_sam25_prerequisites",
            lambda *a, **kw: {
                "precision": {
                    "status": {"runtime_default_settings": {}},
                    "artifact_sha256": "test",
                },
                "parent_directory": parent,
                "prerequisite_sha256": "test",
            },
        )
        results[epochs] = engine.run_task3_baseline_fold(
            "gender",
            0,
            root=tmp_path,
            output_root=tmp_path / "output",
            registry_path=registry_path,
            device_name="cpu",
            child_spec=spec,
            prerequisite_path=audit,
        )
    directory = Path(results[25]["run_dir"])
    checkpoint = torch.load(directory / "final_epoch.pt", map_location="cpu", weights_only=False)
    assert captured and all(
        torch.equal(v, captured[k]) for k, v in checkpoint["model_state_dict"].items()
    )
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 25
    assert checkpoint["config"]["epochs"] == 25
    assert checkpoint["config"]["cosine_t_max"] == 30
    assert results[25]["metrics"]["early_stopped"] is False
    assert [
        v.item()
        for k, v in checkpoint["model_state_dict"].items()
        if k.endswith("num_batches_tracked")
    ] == [25] * 4
    current = pd.read_csv(directory / "history.csv")
    previous = pd.read_csv(Path(results[30]["run_dir"]) / "history.csv").iloc[:25]
    for field in ("learning_rate", "train_loss", "sam_second_loss", "validation_macro_f1"):
        np.testing.assert_array_equal(current[field], previous[field])
    registry = pd.read_csv(registry_path, keep_default_na=False)
    assert registry.status.tolist() == ["complete", "complete"]
    run = inspect_gender_run(
        directory,
        registry=registry,
        splits=synthetic_gender,
        classes=CLASSES,
        root=tmp_path,
        expected_epochs=25,
    )
    screen.verify_sam25_evidence(run, fold=0, splits=synthetic_gender, directory=directory)
    with pytest.raises(ValueError, match="history is incomplete"):
        inspect_gender_run(
            directory, registry=registry, splits=synthetic_gender, classes=CLASSES, root=tmp_path
        )
    with pytest.raises(ValueError, match="frozen contract"):
        original.verify_sam_evidence(run, fold=0, splits=synthetic_gender, directory=directory)
    other = Path(results[30]["run_dir"])
    with pytest.raises(ValueError, match="shorter history"):
        inspect_gender_run(
            other,
            registry=registry,
            splits=synthetic_gender,
            classes=CLASSES,
            root=tmp_path,
            expected_epochs=25,
        )
    history = (directory / "history.csv").read_text()
    current.loc[24, "learning_rate"] = 0.00001
    current.to_csv(directory / "history.csv", index=False)
    with pytest.raises(ValueError, match="cosine schedule"):
        screen.verify_sam25_evidence(run, fold=0, splits=synthetic_gender, directory=directory)
    (directory / "history.csv").write_text(history)
    bad = copy.deepcopy(run)
    bad["metrics"]["selected_epoch"] = 24
    with pytest.raises(ValueError, match="checkpoint"):
        screen.verify_sam25_evidence(bad, fold=0, splits=synthetic_gender, directory=directory)


def test_sam25_notebook_is_unexecuted_and_keeps_original_schedule():
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(root / "notebooks/04aj_task3_gender_sam25_screen.ipynb", 4)
    nbformat.validate(notebook)
    source = "\n".join(c.source for c in notebook.cells)
    assert source.count("result = run_gender_sam25_screen(") == 1
    assert "T_max=30" in source and "all 19 checks" in source.lower()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04aj", "exec")
            assert cell.execution_count is None and not cell.outputs
