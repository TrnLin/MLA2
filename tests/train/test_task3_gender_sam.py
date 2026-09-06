"""Frozen SAM screen and a complete synthetic-image scratch training run."""

import json
from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("torch")

from fashion.config import TARGET_COLUMNS  # noqa: E402
from fashion.data import load_splits  # noqa: E402
from fashion.data.hashing import compute_sha256  # noqa: E402
from fashion.train import task3_gender_sam as screen  # noqa: E402
from fashion.train.config import Task3BaselineConfig  # noqa: E402
from fashion.train.sam import POLICY, SAMStep  # noqa: E402
from fashion.train.task3_gender_mixup import MixUpSpec  # noqa: E402

CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]


def test_frozen_spec_changes_only_the_learning_step(monkeypatch):
    monkeypatch.setattr(screen, "label_contract", lambda root: {"labels": "frozen"})
    spec = screen.sam_spec()
    old = MixUpSpec(spec.label_contract_json)
    identity = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
        "parent_run_ids",
        "screen_rule_version",
        "sam_policy",
        "improvement_rules",
    }
    assert {k: v for k, v in spec.to_dict().items() if k not in identity} == {
        k: v for k, v in old.to_dict().items() if k not in identity
    }
    assert spec.to_dict()["mixup_policy"]["alpha"] == 0.2
    assert spec.to_dict()["sam_policy"] == POLICY
    assert screen.sam_config(spec, fold=0, device_name="cuda").epochs == 30
    for fold, device in [(1, "cuda"), (0, "cpu")]:
        with pytest.raises(ValueError, match="only folds"):
            screen.sam_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="frozen recipe"):
        screen.sam_config(old, fold=0, device_name="cuda")


def test_screen_uses_mixup20_parents_and_sam_receipt(monkeypatch):
    spec = screen.SAMSpec("{}")
    sources = {"MixUp20": {0: "parent0", 4: "parent4"}}
    monkeypatch.setattr(
        screen, "check_gender_sam_sources", lambda **kw: (sources, CLASSES, spec, {})
    )
    monkeypatch.setattr(screen, "sam_config", lambda *a, **kw: None)
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: None)
    monkeypatch.setattr(screen, "_source_identity", lambda *a, **kw: {"sam": "identity"})
    monkeypatch.setattr(screen, "training_splits", lambda *a, **kw: "frozen-split")
    monkeypatch.setattr(screen, "_run_verified_label_screen", lambda **kw: kw)
    args = screen.run_gender_sam_screen(
        output_root="output",
        registry_path="runs.csv",
        source_registry_path="parents.csv",
        precision_directory="precision",
        mixup_directory="04af",
    )
    assert args["parent_group"] == "MixUp20" and args["candidate_group"] == "SAM005"
    assert args["verify_candidate"] is screen.verify_sam_evidence
    from test_task3_gender_stronger_mixup import _report

    report = _report()
    report["incremental_comparison"]["candidate"]["macro_f1"] = 0.74
    assert args["refine_report"](report)["status"] == "pass"
    report["incremental_comparison"]["candidate"]["macro_f1"] = 0.739
    assert args["refine_report"](report)["status"] == "fail"


def test_public_trainer_rejects_cpu_before_creating_run(tmp_path, monkeypatch):
    from fashion.train.task3_baseline import run_task3_baseline_fold

    monkeypatch.setattr(screen, "label_contract", lambda root: {})
    with pytest.raises(ValueError, match="CUDA"):
        run_task3_baseline_fold(
            "gender",
            0,
            output_root=tmp_path,
            root=tmp_path,
            child_spec=screen.sam_spec(),
            device_name="cpu",
        )
    assert not list(tmp_path.rglob("config.json"))


@pytest.fixture
def synthetic_gender(tmp_path):
    rng = np.random.default_rng(2753)
    rows = []
    for fold in range(5):
        for label in CLASSES:
            image_id = len(rows) + 1
            image = tmp_path / f"image{image_id}.png"
            Image.fromarray(rng.integers(0, 220, (80, 60, 3), dtype=np.uint8)).save(image)
            row = dict(
                id=image_id,
                path=image.name,
                sha256=compute_sha256(image),
                product_family_group=f"family{image_id}",
                duplicate_group=f"image{image_id}",
                product_name_key=f"item {image_id}",
                partition="development",
                cv_fold=fold,
                is_cross_role_exact_duplicate=False,
                is_cross_role_near_duplicate=False,
                has_conflicting_target_labels=False,
                conflicting_targets="",
                quarantine_reason="",
            )
            for target in TARGET_COLUMNS:
                row[target] = label if target == "gender" else ""
                row[f"has_{target}_label"] = target == "gender"
            rows.append(row)
    protected = dict(
        rows[0],
        id=999,
        partition="holdout",
        cv_fold="",
        path="sealed.png",
        sha256="sealed",
        product_family_group="sealed",
        duplicate_group="sealed",
        product_name_key="sealed",
    )
    protected.update({f"has_{t}_label": False for t in TARGET_COLUMNS})
    protected.update({t: "" for t in TARGET_COLUMNS})
    rows.append(protected)
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(processed / "splits.csv", index=False)
    (processed / "label_maps.json").write_text(
        json.dumps(
            {
                "gender": {
                    "source_scope": "development",
                    "classes": CLASSES,
                    "label_to_index": dict(zip(CLASSES, range(5))),
                }
            }
        )
    )
    return load_splits(processed / "splits.csv")


@pytest.fixture
def trained_sam(synthetic_gender, tmp_path, monkeypatch, request):
    import torch

    from fashion.train import task3_baseline as engine

    torch.set_num_threads(2)
    config = replace(Task3BaselineConfig(target="gender"), num_workers=0)
    spec = screen.SAMSpec("{}")
    monkeypatch.setattr(engine, "Task3BaselineConfig", lambda **kw: config)
    # Only hardware/source boundaries are substituted. Real model, data, 30 epochs,
    # SAM, MixUp, metrics, checkpoints and registry execute on synthetic images.
    monkeypatch.setattr(screen, "sam_config", lambda *a, **kw: config)
    monkeypatch.setattr(screen, "training_splits", lambda *a, **kw: synthetic_gender)
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": spec.parent_run_id_for_fold(0),
                "validation_fold": 0,
            }
        )
    )
    for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
        (parent / name).write_text("Parent weights must never be loaded")
    audit = tmp_path / "source_audit.json"
    audit.write_text("{}")
    monkeypatch.setattr(
        screen,
        "require_sam_prerequisites",
        lambda *a, **kw: {
            "precision": {"status": {"runtime_default_settings": {}}, "artifact_sha256": "test"},
            "parent_directory": parent,
            "prerequisite_sha256": compute_sha256(audit),
        },
    )
    original_step = SAMStep.step
    before_updates = []

    def checked_step(self, *a, **kw):
        row = pd.read_csv(tmp_path / "runs.csv").iloc[0]
        assert row.status == "running" and bool(row.scratch)
        before_updates.append(row.last_completed_stage)
        if getattr(request, "param", None) == "fail":
            closure = a[0]

            def bad_loss():
                output, loss = closure()
                return output, loss * float("nan")

            return original_step(self, bad_loss, **kw)
        return original_step(self, *a, **kw)

    monkeypatch.setattr(SAMStep, "step", checked_step)

    def fit():
        return engine.run_task3_baseline_fold(
            "gender",
            0,
            output_root=tmp_path / "output",
            root=tmp_path,
            registry_path=tmp_path / "runs.csv",
            device_name="cpu",
            child_spec=spec,
            prerequisite_path=audit,
        )

    if getattr(request, "param", None) == "fail":
        with pytest.raises(FloatingPointError, match="non-finite"):
            fit()
        return None, before_updates, synthetic_gender
    result = fit()
    return result, before_updates, synthetic_gender


@pytest.mark.parametrize("trained_sam", ["fail"], indirect=True)
def test_nonfinite_sam_marks_run_failed_without_checkpoint(trained_sam, tmp_path):
    row = pd.read_csv(tmp_path / "runs.csv").iloc[0]
    assert row.status == "failed"
    assert row.last_completed_stage == "registered_before_first_optimizer_step"
    assert not list((tmp_path / "output").rglob("final_epoch.pt"))


def test_complete_scratch_training_saves_verified_sam_evidence(trained_sam, tmp_path):
    import torch

    result, updates, splits = trained_sam
    directory = Path(result["run_dir"])
    assert len(updates) == 30 and updates[0] == "registered_before_first_optimizer_step"
    registry = pd.read_csv(tmp_path / "runs.csv")
    assert registry.status.tolist() == ["complete"]
    assert registry.training_product_count.item() == 20
    assert registry.validation_product_count.item() == 5
    assert not registry.submission_eligible.item()
    checkpoint = torch.load(directory / "final_epoch.pt", weights_only=False, map_location="cpu")
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 30
    assert checkpoint["config"]["sam_policy"] == POLICY
    counters = [
        v.item()
        for k, v in checkpoint["model_state_dict"].items()
        if k.endswith("num_batches_tracked")
    ]
    assert counters == [30] * 4
    assert checkpoint["config"]["parameter_count"] == 390181
    run = {"config": checkpoint["config"], "metrics": result["metrics"]}
    screen.verify_sam_evidence(run, fold=0, splits=splits, directory=directory)
    diagnostics = json.loads((directory / "clean_epoch_diagnostics.json").read_text())
    assert all(len(row["clean_training"]["per_class"]) == 5 for row in diagnostics)
    assert all(row["checkpoint_selection"] == "diagnostic_only" for row in diagnostics)
    receipt = directory / "sam_training.json"
    payload = json.loads(receipt.read_text())
    payload["epochs"][0]["optimizer_steps"] = 2
    receipt.write_text(json.dumps(payload))
    run["metrics"]["sam_receipt_sha256"] = compute_sha256(receipt)
    with pytest.raises(ValueError, match="accounting"):
        screen.verify_sam_evidence(run, fold=0, splits=splits, directory=directory)


def test_notebook_is_one_unexecuted_sam_trial():
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(root / "notebooks/04ai_task3_gender_sam_screen.ipynb", 4)
    nbformat.validate(notebook)
    source = "\n".join(c.source for c in notebook.cells)
    assert source.count("result = run_gender_sam_screen(") == 1
    assert "74%" in source and "all 19 required checks" in source
    assert "alpha040" not in source and "train_test_split" not in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04ai", "exec")
            assert cell.execution_count is None and not cell.outputs


@pytest.mark.parametrize("fault", [None, "code", "policy", "parent", "precision"])
def test_sam_prerequisites_bind_code_policy_and_direct_parent(
    synthetic_gender, tmp_path, monkeypatch, fault
):
    monkeypatch.setattr(screen, "label_contract", lambda root: {})
    spec = screen.sam_spec()
    sources = {"MixUp20": {f: {"directory": str(tmp_path / f"parent{f}")} for f in (0, 4)}}
    evidence = {"artifact_sha256": {"precision": "original"}}
    paths = {"mixup_directory": str(tmp_path), "precision_directory": str(tmp_path)}
    (tmp_path / "screen_decision.json").write_text('{"status": "pass"}')
    source_dir = tmp_path / "src/fashion/train"
    source_dir.mkdir(parents=True)
    names = (
        "sam.py",
        "task3_gender_sam.py",
        "mixup.py",
        "task3_gender_mixup.py",
        "task3_gender_stronger_mixup.py",
        "task3_baseline.py",
        "task3_gender_name_truth.py",
        "task3_gender_narrow.py",
    )
    for name in names:
        (source_dir / name).write_text(name)
    monkeypatch.setattr(screen, "training_splits", lambda *a, **kw: synthetic_gender)
    monkeypatch.setattr(screen, "label_source_identity", lambda *a, **kw: {"paths": paths})
    monkeypatch.setattr(
        screen, "check_gender_sam_sources", lambda **kw: (sources, CLASSES, spec, evidence)
    )
    precision = json.loads(json.dumps(evidence))
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: precision)
    identity = screen._source_identity(sources, spec, evidence, paths, root=tmp_path)
    assert set(identity["implementation_sha256"]) == set(names)
    assert all(row["policy"]["alpha"] == 0.2 for row in identity["mixup_contracts"].values())
    if fault == "policy":
        identity["spec"]["sam_policy"]["rho"] = 0.1
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps({"identity": identity}))
    if fault == "code":
        (source_dir / "sam.py").write_text("changed code")
    if fault == "precision":
        precision["artifact_sha256"]["precision"] = "changed"
    parent = tmp_path / ("wrong" if fault == "parent" else "parent0")
    if fault:
        with pytest.raises(ValueError, match="evidence changed|verified 04af"):
            screen.require_sam_prerequisites(
                audit, spec=spec, fold=0, root=tmp_path, parent_run_directory=parent
            )
    else:
        checked = screen.require_sam_prerequisites(
            audit, spec=spec, fold=0, root=tmp_path, parent_run_directory=parent
        )
        assert checked["parent_directory"] == parent
        assert checked["prerequisite_sha256"] == compute_sha256(audit)
