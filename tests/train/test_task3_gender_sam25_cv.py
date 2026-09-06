"""Five scratch SAM25 folds, complete OOF coverage and strict reuse boundaries."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from test_task3_gender_sam import CLASSES, synthetic_gender  # noqa: F401

torch = pytest.importorskip("torch")

from fashion.data.hashing import compute_sha256  # noqa: E402
from fashion.train import task3_gender_sam25 as screen  # noqa: E402
from fashion.train import task3_gender_sam25_cv as cv  # noqa: E402
from fashion.train.config import Task3BaselineConfig  # noqa: E402
from fashion.train.registry import REGISTRY_COLUMNS  # noqa: E402


def test_five_fold_recipe_keeps_all_training_controls(monkeypatch):
    monkeypatch.setattr(cv, "label_contract", lambda root: {})
    monkeypatch.setattr(screen, "label_contract", lambda root: {})
    parents = tuple(f"g2-f{f}" for f in range(5))
    spec = cv.sam25_cv_spec(parents)
    before, after = screen.SAM25Spec("{}").to_dict(), spec.to_dict()
    metadata = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
        "parent_run_ids",
        "parent_folds",
        "parent_role",
        "recipe_source_run_ids",
        "recipe_source_commit",
        "screen_rule_version",
        "execution_scope",
        "improvement_rules",
    }
    assert {k: v for k, v in after.items() if k not in metadata} == {
        k: v for k, v in before.items() if k not in metadata
    }
    for f in range(5):
        assert spec.parent_run_id_for_fold(f) == parents[f]
        assert cv.sam25_cv_config(spec, fold=f, device_name="cuda").epochs == 25
    assert after["cosine_t_max"] == 30
    assert after["recipe_source_run_ids"] == {str(f): r for f, r in cv.SCREEN_RUN_IDS.items()}
    assert "improvement_rules" not in after
    for bad in (("same",) * 5, parents[:2]):
        with pytest.raises(ValueError, match="five distinct"):
            cv.sam25_cv_spec(bad)
    with pytest.raises(ValueError, match="canonical folds"):
        cv.sam25_cv_config(spec, fold=5, device_name="cuda")
    with pytest.raises(ValueError, match="CUDA"):
        cv.sam25_cv_config(spec, fold=0, device_name="cpu")
    with pytest.raises(ValueError, match="only folds"):
        screen.sam25_config(screen.SAM25Spec("{}"), fold=1, device_name="cuda")


def test_source_check_rejects_another_screen_before_loading_parents(tmp_path, monkeypatch):
    (tmp_path / "screen_decision.json").write_text('{"status":"pass"}')
    monkeypatch.setattr(cv, "check_gender_sam25_sources", lambda **kw: pytest.fail("read parents"))
    with pytest.raises(ValueError, match="exact reviewed"):
        cv.check_gender_sam25_cv_sources(sam25_directory=tmp_path)


def test_screen_audit_uses_recorded_code_and_verified_predictions(tmp_path, monkeypatch):
    old_spec = screen.SAM25Spec("{}")
    old_identity = {"implementation_sha256": {"mixup.py": "old", "task3_baseline.py": "old"}}
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps({"identity": old_identity}))
    rows = pd.DataFrame(
        {
            "true_index": [0, 1],
            **{
                f"probability_{i}_{name}": [0.8 if i == 0 else 0.05, 0.8 if i == 1 else 0.05]
                for i, name in enumerate(CLASSES)
            },
        }
    )
    score = cv.oof_metrics(rows, CLASSES)
    decision_path = tmp_path / "screen_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "candidate": score,
                "run_ids": {str(f): r for f, r in cv.SCREEN_RUN_IDS.items()},
            }
        )
    )
    monkeypatch.setattr(cv, "SCREEN_DECISION_SHA256", compute_sha256(decision_path))
    registry = tmp_path / "runs.csv"
    pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(registry, index=False)
    sources = {
        "G2": {f: {"run_id": f"g2-f{f}"} for f in range(5)},
        "MixUp20": {f: {"run_id": f"mixup-f{f}"} for f in (0, 4)},
    }
    monkeypatch.setattr(
        cv, "check_gender_sam25_sources", lambda **kw: (sources, CLASSES, old_spec, {})
    )
    monkeypatch.setattr(
        cv,
        "screen_source_identity",
        lambda *a, **kw: {"implementation_sha256": {"mixup.py": "new", "task3_baseline.py": "new"}},
    )
    monkeypatch.setattr(cv, "_historical_hash", lambda *a: "old")
    monkeypatch.setattr(cv, "load_gender_name_truth_variant", lambda root: "verified labels")
    monkeypatch.setattr(cv, "label_contract", lambda root: {})
    inspected, checked = [], []

    def inspect(directory, **kwargs):
        assert kwargs["expected_epochs"] == 25
        inspected.append(directory.name)
        return {"run_id": directory.name}

    monkeypatch.setattr(cv, "inspect_gender_run", inspect)
    monkeypatch.setattr(
        cv, "_verify_training_evidence", lambda *a, **kw: checked.append(kw["expected_epochs"])
    )
    monkeypatch.setattr(cv, "verify_sam25_evidence", lambda *a, **kw: checked.append(kw["fold"]))
    monkeypatch.setattr(
        cv,
        "_screen_evaluation",
        lambda run, **kw: {
            **run,
            "predictions": rows.iloc[[0 if run["run_id"] == cv.SCREEN_RUN_IDS[0] else 1]],
        },
    )
    _, _, spec, _ = cv.check_gender_sam25_cv_sources(
        sam25_directory=tmp_path, root=tmp_path, source_registry_path=registry
    )
    assert inspected == list(cv.SCREEN_RUN_IDS.values()) and checked == [25, 0, 25, 4]
    assert spec.comparison_run_ids == tuple(f"g2-f{f}" for f in range(5))
    audit.write_text(json.dumps({"identity": {"implementation_sha256": {"mixup.py": "tampered"}}}))
    with pytest.raises(ValueError, match="source audit"):
        cv.check_gender_sam25_cv_sources(
            sam25_directory=tmp_path, root=tmp_path, source_registry_path=registry
        )


@pytest.mark.parametrize("memory", [0, -1, float("nan"), 3_000_000_000])
def test_resource_boundary_rejects_invalid_or_excessive_memory(memory):
    with pytest.raises(ValueError, match="3 GB"):
        cv._require_resources({"peak_memory_bytes": memory, "parameter_count": 390181})


def test_same_fold_references_and_duplicate_completions(monkeypatch, tmp_path):
    spec = cv.SAM25CVSpec("{}", tuple(f"g2-f{f}" for f in range(5)))
    rows = pd.DataFrame(
        [
            {
                "experiment_id": spec.experiment_id,
                "validation_fold": f,
                "status": status,
                "run_id": run,
            }
            for f, status, run in (
                (0, "complete", "cv0"),
                (1, "failed", "failed1"),
                (2, "running", "running2"),
            )
        ]
    )
    assert cv._completed_fold(spec, 0, destination=tmp_path, registry=rows) == tmp_path / "cv0"
    assert cv._completed_fold(spec, 1, destination=tmp_path, registry=rows) is None
    assert cv._completed_fold(spec, 2, destination=tmp_path, registry=rows) is None
    with pytest.raises(ValueError, match="Multiple completed"):
        cv._completed_fold(
            spec,
            0,
            destination=tmp_path,
            registry=pd.concat([rows, rows.iloc[[0]]], ignore_index=True),
        )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"identity": {"paths": {"precision_directory": "precision"}}}))
    monkeypatch.setattr(cv, "sam25_cv_config", lambda *a, **kw: None)
    monkeypatch.setattr(
        cv, "require_narrow_prerequisites", lambda *a, **kw: {"artifact_sha256": {}}
    )
    monkeypatch.setattr(
        cv,
        "check_gender_sam25_cv_sources",
        lambda **kw: (
            {"G2": {1: {"directory": str(tmp_path / "parent1")}}},
            CLASSES,
            spec,
            {"artifact_sha256": {}},
        ),
    )
    monkeypatch.setattr(
        cv, "_source_identity", lambda *a, **kw: json.loads(audit.read_text())["identity"]
    )
    with pytest.raises(ValueError, match="same-fold"):
        cv.require_sam25_cv_prerequisites(
            audit, spec=spec, fold=1, parent_run_directory=tmp_path / "parent0"
        )


def test_all_five_real_scratch_runs_complete_and_resume(
    synthetic_gender,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from fashion.train import task3_baseline as engine
    from fashion.train import task3_gender_dropout_darkening as controls

    torch.set_num_threads(2)
    config = replace(Task3BaselineConfig(target="gender"), epochs=25, num_workers=0)
    spec = cv.SAM25CVSpec("{}", tuple(f"g2-f{f}" for f in range(5)))
    evidence = {
        "status": {"runtime_default_settings": controls.TRAINING_PRECISION},
        "artifact_sha256": {},
    }
    parents = {}
    for fold in range(5):
        directory = tmp_path / f"parent{fold}"
        directory.mkdir()
        (directory / "metrics.json").write_text(
            json.dumps({"run_id": spec.parent_run_id_for_fold(fold), "validation_fold": fold})
        )
        for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
            (directory / name).write_text("Comparison only: loading these weights must fail")
        parents[fold] = {"directory": str(directory), "run_id": spec.parent_run_id_for_fold(fold)}
    labels = tmp_path / cv.VARIANT_RELATIVE_PATH
    labels.mkdir(parents=True)
    (labels / "summary.json").write_text("{}")
    source_registry = tmp_path / "source.csv"
    pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(source_registry, index=False)
    monkeypatch.setattr(
        cv, "check_gender_sam25_cv_sources", lambda **kw: ({"G2": parents}, CLASSES, spec, evidence)
    )
    monkeypatch.setattr(cv, "training_splits", lambda *a, **kw: synthetic_gender)
    monkeypatch.setattr(cv, "_source_identity", lambda *a, **kw: {"recipe": spec.to_dict()})
    monkeypatch.setattr(cv, "sam25_cv_config", lambda *a, **kw: config)
    monkeypatch.setattr(cv, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    monkeypatch.setattr(
        cv,
        "require_sam25_cv_prerequisites",
        lambda path, fold, **kw: {
            "precision": evidence,
            "parent_directory": parents[fold]["directory"],
            "prerequisite_sha256": compute_sha256(path),
        },
    )
    monkeypatch.setattr(engine, "Task3BaselineConfig", lambda **kw: config)
    monkeypatch.setattr(controls, "Task3BaselineConfig", lambda **kw: config)
    # Hardware and external evidence boundaries only. Run the real model/data,
    # SAM/MixUp, 25 epochs per fold, registry, checkpoint and full source auditor.
    monkeypatch.setattr(cv, "_require_resources", lambda metrics: None)
    monkeypatch.setattr(
        cv,
        "evaluate_gender_ieee",
        lambda run, **kw: {**run, "metrics": {**run["metrics"], "comparison_precision": cv.POLICY}},
    )
    monkeypatch.setattr(cv, "write_original_label_diagnostic", lambda *a, **kw: None)
    real_train = engine.run_task3_baseline_fold
    started = []

    def cpu_train(target, fold, **kwargs):
        started.append(fold)
        return real_train(target, fold, **dict(kwargs, device_name="cpu"))

    monkeypatch.setattr(engine, "run_task3_baseline_fold", cpu_train)
    kwargs = dict(
        root=tmp_path,
        output_root=tmp_path / "output",
        registry_path=tmp_path / "runs.csv",
        source_registry_path=source_registry,
        precision_directory="mocked",
    )
    result = cv.run_gender_sam25_cv(**kwargs)
    assert started == [0, 1, 2, 3, 4]
    registry = pd.read_csv(kwargs["registry_path"], keep_default_na=False)
    assert registry.status.tolist() == ["complete"] * 5
    assert registry.scratch.all()
    aggregate = Path(result["aggregate_directory"])
    oof = pd.read_csv(aggregate / "oof_predictions.csv", keep_default_na=False)
    assert len(oof) == oof.id.nunique() == 25
    assert set(oof.id) == set(range(1, 26))
    assert oof.groupby("cv_fold").size().tolist() == [5] * 5
    assert result["scopes"]["all_five"]["metrics"]["support"] == 25
    assert result["scopes"]["additional_folds"]["metrics"]["support"] == 15
    assert result["scopes"]["screen_folds"]["metrics"]["support"] == 10
    assert result["independent_test_evidence"] is False
    manifest = json.loads(Path(result["model_manifest"]).read_text())
    destination = Path(result["model_manifest"]).parent
    assert manifest["checkpoint_epoch"] == 25 and len(manifest["folds"]) == 5
    assert manifest["class_names"] == CLASSES
    for entry in manifest["folds"]:
        for record in entry["files"].values():
            assert compute_sha256(destination / record["path"]) == record["sha256"]
        checkpoint = torch.load(
            destination / entry["files"]["final_epoch.pt"]["path"], weights_only=False
        )
        assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 25
        assert checkpoint["config"]["cosine_t_max"] == 30
    before = kwargs["registry_path"].read_bytes()
    assert cv.run_gender_sam25_cv(**kwargs) == result
    assert started == [0, 1, 2, 3, 4] and kwargs["registry_path"].read_bytes() == before
    # Check summaries reject duplicate IDs, even with otherwise valid five-fold keys.
    runs = {}
    for entry in manifest["folds"]:
        directory = destination / entry["run_id"]
        run = cv.inspect_gender_run(
            directory,
            registry=registry,
            splits=synthetic_gender,
            classes=CLASSES,
            root=tmp_path,
            expected_epochs=25,
        )
        run["metrics"]["comparison_precision"] = cv.POLICY
        runs[entry["fold"]] = run
    with pytest.raises(ValueError, match="all five"):
        cv.summarize_cv({0: runs[0]}, splits=synthetic_gender, classes=CLASSES)
    bad = copy.deepcopy(runs)
    bad[1]["predictions"] = bad[0]["predictions"]
    with pytest.raises(ValueError, match="unique IDs"):
        cv.summarize_cv(bad, splits=synthetic_gender, classes=CLASSES)
    # The CV metadata and comparison references must not change the actual recipe.
    old_spec = screen.SAM25Spec("{}")
    old_parent = tmp_path / "old_parent"
    old_parent.mkdir()
    (old_parent / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": old_spec.parent_run_id_for_fold(0),
                "validation_fold": 0,
            }
        )
    )
    for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
        (old_parent / name).write_text("Never load comparison weights")
    monkeypatch.setattr(screen, "sam25_config", lambda *a, **kw: config)
    monkeypatch.setattr(screen, "training_splits", lambda *a, **kw: synthetic_gender)
    monkeypatch.setattr(
        screen,
        "require_sam25_prerequisites",
        lambda path, **kw: {
            "precision": evidence,
            "parent_directory": old_parent,
            "prerequisite_sha256": compute_sha256(path),
        },
    )
    old = real_train(
        "gender",
        0,
        root=tmp_path,
        output_root=tmp_path / "prefix_check",
        registry_path=tmp_path / "prefix_runs.csv",
        device_name="cpu",
        child_spec=old_spec,
        prerequisite_path=destination / "source_audit.json",
    )
    original_weights = torch.load(Path(old["run_dir"]) / "final_epoch.pt", weights_only=False)[
        "model_state_dict"
    ]
    current_weights = torch.load(
        destination / manifest["folds"][0]["files"]["final_epoch.pt"]["path"], weights_only=False
    )["model_state_dict"]
    assert all(
        torch.equal(value, original_weights[name]) for name, value in current_weights.items()
    )
    # A damaged completed model must stop resume rather than silently retrain it.
    broken = destination / manifest["folds"][0]["files"]["final_epoch.pt"]["path"]
    broken.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="registry hash mismatch"):
        cv.run_gender_sam25_cv(**kwargs)
    assert started == [0, 1, 2, 3, 4]


def test_cv_notebook_is_five_fold_and_unexecuted():
    root = Path(__file__).resolve().parents[2]
    book = nbformat.read(root / "notebooks/04ak_task3_gender_sam25_five_fold.ipynb", 4)
    nbformat.validate(book)
    text = "\n".join(c.source for c in book.cells)
    assert text.count("result = run_gender_sam25_cv(") == 1
    assert "T_max=30" in text and "five" in text.lower()
    assert "run_gender_sam25_screen(" not in text
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04ak", "exec")
            assert cell.execution_count is None and not cell.outputs
