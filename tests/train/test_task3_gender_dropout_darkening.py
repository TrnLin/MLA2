"""Frozen two-parent lineage, resume safety and incremental comparison contracts."""

import ast
import copy
import json
from dataclasses import replace
from pathlib import Path

import nbformat
import pytest
from test_task3_gender_narrow import _case, _gates

import fashion.train.task3_gender_dropout_darkening as screen
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import dataset_v2_spec, run_task3_dataset_v2_screen
from fashion.train.task3_gender_narrow import TRAINING_PRECISION, evaluate_gender_narrow_screen


def test_recipe_and_sparse_parents_preserve_baseline_controls():
    spec = dataset_v2_spec(screen.NAME, screen.PARENT_RUN_IDS)
    for fold, parent in zip((0, 4), screen.PARENT_RUN_IDS, strict=True):
        assert spec.parent_run_id_for_fold(fold) == parent
        assert screen.dropout_darkening_config(
            spec, fold=fold, device_name="cuda"
        ) == Task3BaselineConfig(target="gender")
    assert spec.classifier_dropout == 0.30
    assert spec.training_augmentation == "translation_2px_p05_mild_darkening_p025"
    assert spec.saved_tensors_on_cpu is False
    assert spec.to_dict()["parent_folds"] == [0, 4]
    for ids in ([], ["one"], ["one", "one"], [str(f) for f in range(5)]):
        with pytest.raises(ValueError, match="2 distinct"):
            dataset_v2_spec(screen.NAME, ids)
    with pytest.raises(ValueError, match="5 distinct"):
        dataset_v2_spec("gender_dropout_030", screen.PARENT_RUN_IDS)
    with pytest.raises(ValueError, match="predeclared"):
        replace(spec, classifier_dropout=0.4)
    for fold, device in ((1, "cuda"), (0, "cpu")):
        with pytest.raises(ValueError, match="only folds"):
            screen.dropout_darkening_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="completed parents"):
        screen.dropout_darkening_config(
            dataset_v2_spec(screen.NAME, ["a", "b"]), fold=0, device_name="cuda"
        )
    with pytest.raises(ValueError, match="run_gender_dropout_darkening_screen"):
        run_task3_dataset_v2_screen(
            screen.NAME, parent_run_ids=screen.PARENT_RUN_IDS, output_root="unused"
        )
    with pytest.raises(ValueError, match="required before training"):
        screen.require_dropout_darkening_prerequisites(None, spec=spec, fold=0)


def _source_fixture(tmp_path):
    child, sources, classes = _case()
    sources["Drop30"] = copy.deepcopy(sources["G2"])
    spec = dataset_v2_spec(screen.NAME, screen.PARENT_RUN_IDS)
    evidence = {"artifact_sha256": {"precision": "verified"}}
    for group in sources.values():
        for run in group.values():
            run.update(
                sha256={"config.json": run["run_id"]}, directory=str(tmp_path / run["run_id"])
            )
    for fold, parent in zip((0, 4), screen.PARENT_RUN_IDS, strict=True):
        sources["Drop30"][fold]["run_id"] = parent
        sources["Drop30"][fold]["directory"] = str(tmp_path / parent)
        sources["Drop30"][fold]["robustness"]["run_id"] = parent
        child[fold].update(
            fold=fold,
            config={
                **Task3BaselineConfig(target="gender").to_dict(),
                "child_experiment": spec.to_dict(),
                "parent_run_id": parent,
                "training_precision_settings": TRAINING_PRECISION,
                "precision_evidence_sha256": evidence["artifact_sha256"],
            },
        )
        child[fold]["metrics"]["parameter_count"] = 390181
    split = tmp_path / "data/processed/splits.csv"
    split.parent.mkdir(parents=True)
    split.write_text("canonical split\n")
    registry = tmp_path / "runs.csv"
    registry.write_text("run_id\n")
    paths = dict(
        g2_directory=tmp_path,
        e6_directory=tmp_path,
        dropout_directory=tmp_path,
        source_registry_path=registry,
        precision_directory=tmp_path,
    )
    return child, sources, classes, spec, evidence, paths


@pytest.mark.parametrize("fault", [None, "parent", "precision", "fold"])
def test_source_check_requires_exact_completed_dropout_parents(tmp_path, monkeypatch, fault):
    _, sources, classes, spec, evidence, paths = _source_fixture(tmp_path)
    parent_spec = dataset_v2_spec("gender_dropout_030", [f"G2-{f}" for f in range(5)])
    dropout = sources.pop("Drop30")
    for fold, run in dropout.items():
        run.update(
            fold=fold,
            config={
                **Task3BaselineConfig(target="gender").to_dict(),
                "child_experiment": parent_spec.to_dict(),
                "parent_run_id": sources["G2"][fold]["run_id"],
                "training_precision_settings": TRAINING_PRECISION,
                "precision_evidence_sha256": evidence["artifact_sha256"],
            },
        )
    if fault == "parent":
        dropout[4]["config"]["parent_run_id"] = "wrong"
    elif fault == "precision":
        dropout[4]["config"]["training_precision_settings"] = {}
    elif fault == "fold":
        dropout[4]["fold"] = 1
    monkeypatch.setattr(
        screen,
        "check_gender_narrow_sources",
        lambda **kw: (sources, classes, parent_spec, evidence),
    )
    monkeypatch.setattr(screen, "load_splits", lambda *a: None)
    inspected = []

    def inspect(path, **kwargs):
        inspected.append(path.name)
        return dropout[dict(zip(screen.PARENT_RUN_IDS, (0, 4), strict=True))[path.name]]

    monkeypatch.setattr(screen, "inspect_gender_run", inspect)
    if fault:
        with pytest.raises(ValueError, match="parent|precision|fold"):
            screen.check_gender_dropout_darkening_sources(**paths, root=tmp_path)
    else:
        checked, _, checked_spec, _ = screen.check_gender_dropout_darkening_sources(
            **paths, root=tmp_path
        )
        assert checked_spec == spec
        assert set(checked["Drop30"]) == {0, 4}
    assert inspected == list(screen.PARENT_RUN_IDS)


def test_audit_survives_registry_append_but_rejects_changed_sources_and_parent(
    tmp_path, monkeypatch
):
    _, sources, classes, spec, evidence, paths = _source_fixture(tmp_path)
    audit = tmp_path / "source_audit.json"
    identity = screen._source_identity(sources, spec, evidence, paths, root=tmp_path)
    screen._save_source_audit(audit, identity, registry_path=paths["source_registry_path"])
    before = audit.read_bytes()
    with paths["source_registry_path"].open("a") as handle:
        handle.write("new-child-run\n")
    screen._save_source_audit(audit, identity, registry_path=paths["source_registry_path"])
    assert audit.read_bytes() == before
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    monkeypatch.setattr(
        screen,
        "check_gender_dropout_darkening_sources",
        lambda **kw: (sources, classes, spec, evidence),
    )
    result = screen.require_dropout_darkening_prerequisites(audit, spec=spec, fold=4, root=tmp_path)
    assert result["parent_directory"].name == screen.PARENT_RUN_IDS[1]
    with pytest.raises(ValueError, match="direct parent directory"):
        screen.require_dropout_darkening_prerequisites(
            audit, spec=spec, fold=4, root=tmp_path, parent_run_directory=tmp_path / "wrong"
        )
    sources["Drop30"][4]["sha256"]["config.json"] = "changed"
    with pytest.raises(ValueError, match="evidence changed"):
        screen.require_dropout_darkening_prerequisites(audit, spec=spec, fold=4, root=tmp_path)
    with pytest.raises(ValueError, match="audit differs"):
        screen._save_source_audit(
            audit,
            screen._source_identity(sources, spec, evidence, paths, root=tmp_path),
            registry_path=paths["source_registry_path"],
        )


@pytest.mark.parametrize("mode", ["fresh", "reuse", "bad_reuse", "memory", "reference_failure"])
def test_runner_trains_only_two_folds_and_checks_reuse(tmp_path, monkeypatch, mode):
    child, sources, classes, spec, evidence, paths = _source_fixture(tmp_path)
    monkeypatch.setattr(
        screen,
        "check_gender_dropout_darkening_sources",
        lambda **kw: (sources, classes, spec, evidence),
    )
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    monkeypatch.setattr(screen, "load_splits", lambda *a: None)
    events = []
    audit = tmp_path / spec.artifact_dir / "gender/source_audit.json"

    def result_for_fold(fold):
        child[fold]["config"]["refinement_prerequisite_sha256"] = screen.compute_sha256(audit)
        return {"run_dir": str(tmp_path / f"child-{fold}")}

    def reusable(spec, fold, **kw):
        if mode not in {"reuse", "bad_reuse"}:
            return None
        result = result_for_fold(fold)
        if mode == "bad_reuse":
            child[fold]["config"]["parent_run_id"] = "wrong"
        return result

    def fit(**kw):
        fold = kw["validation_fold"]
        events.append(("fit", fold))
        assert kw["parent_run_directory"] == sources["Drop30"][fold]["directory"]
        assert kw["registry_path"] == paths["source_registry_path"]
        assert kw["registry_mirrors"] == (tmp_path / "mirror.csv",)
        assert kw["prerequisite_path"] == audit
        assert kw["child_spec"] == spec
        if mode == "memory":
            child[fold]["metrics"]["peak_memory_bytes"] = 3_000_000_000
        return result_for_fold(fold)

    def evaluate(run, **kw):
        events.append(("evaluate", run["run_id"]))
        if mode == "reference_failure":
            raise ValueError("reference failed")
        return run

    monkeypatch.setattr(screen, "_reusable_fold", reusable)
    monkeypatch.setattr(screen, "_train_fold", fit)
    monkeypatch.setattr(screen, "inspect_gender_run", lambda path, **kw: child[int(path.name[-1])])
    monkeypatch.setattr(screen, "evaluate_gender_ieee", evaluate)
    # Keep genuine decision/comparison arithmetic, with fewer draws in the orchestration test.
    monkeypatch.setattr(
        screen,
        "evaluate_gender_narrow_screen",
        lambda *a, **kw: evaluate_gender_narrow_screen(*a, **kw, repetitions=20),
    )
    compare = screen.compare_with_dropout
    monkeypatch.setattr(screen, "compare_with_dropout", lambda *a: compare(*a, repetitions=20))
    args = dict(
        **paths,
        output_root=tmp_path,
        registry_path=paths["source_registry_path"],
        root=tmp_path,
        registry_mirrors=(tmp_path / "mirror.csv",),
    )
    if mode in {"bad_reuse", "reference_failure"}:
        with pytest.raises(ValueError, match="direct parent|reference failed"):
            screen.run_gender_dropout_darkening_screen(**args)
        assert not any(event[0] == "fit" for event in events)
        return
    result = screen.run_gender_dropout_darkening_screen(**args)
    assert [value for event, value in events if event == "fit"] == (
        [] if mode == "reuse" else [0] if mode == "memory" else [0, 4]
    )
    assert events[:6] == [
        ("evaluate", sources[name][fold]["run_id"])
        for name in ("G2", "E6", "Drop30")
        for fold in (0, 4)
    ]
    assert result["status"] == ("fail" if mode == "memory" else "pass")
    saved = json.loads((audit.parent / "screen_decision.json").read_text())
    assert saved == result
    if mode != "memory":
        assert (audit.parent / "dropout_corruption_comparison.csv").is_file()


def test_incremental_comparison_exposes_raw_scores_and_gates_stay_fixed():
    child, sources, classes = _case()
    for run in child.values():
        run["metrics"]["parameter_count"] = 390181
    result = evaluate_gender_narrow_screen(
        child, sources, classes, experiment_name=screen.NAME, repetitions=20
    )
    assert result["status"] == "pass"
    assert result["rule_version"] == screen.RULE_VERSION
    incremental = screen.compare_with_dropout(child, sources["G2"], classes, repetitions=20)
    assert incremental["validation_delta"] == 0
    assert all(
        row["raw_corrupted_delta"] == pytest.approx(row["induced_change_delta"])
        for row in incremental["corruptions"]
    )
    assert incremental["folds"][0]["gap_reduction"] == pytest.approx(0.09)
    for run in child.values():
        run["metrics"]["final_train_eval_macro_f1"] += 0.08
        run["metrics"]["final_train_validation_macro_f1_gap"] += 0.08
    gates = _gates(
        evaluate_gender_narrow_screen(
            child, sources, classes, experiment_name=screen.NAME, repetitions=20
        )
    )
    assert gates["validation_delta"] == "pass"
    assert gates["mean_gap_reduction"] == "fail"


def test_baseline_parent_resolver_handles_fold_four_without_torch():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "src/fashion/train/task3_baseline.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_child_parent_run_id"
    )
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "parent resolver", "exec"), scope)
    spec = dataset_v2_spec(screen.NAME, screen.PARENT_RUN_IDS)
    assert scope["_child_parent_run_id"](spec, 4) == screen.PARENT_RUN_IDS[1]
    with pytest.raises(ValueError, match="No completed parent"):
        scope["_child_parent_run_id"](spec, 1)


def test_notebook_is_unexecuted_and_has_one_two_fold_screen():
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(
        root / "notebooks/04aa_task3_gender_dropout_darkening_screen.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert code.count("run_gender_dropout_darkening_screen(") == 1
    assert "DROPOUT_DIR" in code and "registry_mirrors=(LOCAL_REGISTRY,)" in code
    assert "train_test_split" not in code and "pretrained=True" not in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04aa", "exec")
            assert cell.execution_count is None and cell.outputs == []


def test_actual_trainer_rejects_missing_audit_before_any_fit(tmp_path):
    pytest.importorskip("torch")
    from fashion.train.task3_baseline import run_task3_baseline_fold

    with pytest.raises(ValueError, match="required before training"):
        run_task3_baseline_fold(
            "gender",
            0,
            output_root=tmp_path,
            root=tmp_path,
            child_spec=dataset_v2_spec(screen.NAME, screen.PARENT_RUN_IDS),
        )
    assert not list(tmp_path.rglob("runs.csv"))
