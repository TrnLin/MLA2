"""Stronger dropout must change one factor and keep verified 04aa lineage."""

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import nbformat
import pytest
from test_task3_gender_dropout_darkening import _source_fixture
from test_task3_gender_narrow import _case, _gates

import fashion.train.task3_gender_dropout_darkening as shared
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import (
    check_task3_dataset_v2_setup,
    dataset_v2_spec,
    run_task3_dataset_v2_screen,
)
from fashion.train.task3_gender_stronger_dropout import (
    NAME,
    PARENT_RUN_IDS,
    evaluate_gender_stronger_dropout_screen,
    run_gender_stronger_dropout_screen,
    stronger_dropout_config,
)


def test_only_dropout_changes_and_old_config_digest_is_preserved():
    old = dataset_v2_spec(shared.NAME, shared.PARENT_RUN_IDS)
    new = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    metadata = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
        "parent_run_ids",
    }
    assert {k for k in asdict(old) if asdict(old)[k] != asdict(new)[k]} - metadata == {
        "classifier_dropout"
    }
    assert new.classifier_dropout == 0.45 and old.classifier_dropout == 0.30
    assert new.parent_folds == (0, 4) and new.parent_run_id_for_fold(4) == PARENT_RUN_IDS[1]
    assert not new.saved_tensors_on_cpu
    config = stronger_dropout_config(new, fold=4, device_name="cuda")
    assert config == Task3BaselineConfig(target="gender")
    payload = {"baseline_controls": config.to_dict(), "child_experiment": old.to_dict()}
    assert (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        == "cfbed3f0fed4"
    )
    for spec, fold, device in [
        (old, 0, "cuda"),
        (new, 1, "cuda"),
        (new, 0, "cpu"),
        (dataset_v2_spec(NAME, shared.PARENT_RUN_IDS), 0, "cuda"),
    ]:
        with pytest.raises(ValueError):
            stronger_dropout_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="predeclared"):
        replace(new, classifier_dropout=0.5)
    for function in (check_task3_dataset_v2_setup, run_task3_dataset_v2_screen):
        with pytest.raises(ValueError, match="stronger_dropout"):
            function(NAME, parent_run_ids=PARENT_RUN_IDS, output_root="unused")
    with pytest.raises(ValueError, match="required before training"):
        shared.require_dropout_darkening_prerequisites(None, spec=new, fold=0)


def _completed_darkening(tmp_path):
    previous, sources, classes, old_spec, evidence, paths = _source_fixture(tmp_path)
    directory = tmp_path / "completed-darkening"
    directory.mkdir()
    audit = directory / "source_audit.json"
    shared._save_source_audit(
        audit,
        shared._source_identity(sources, old_spec, evidence, paths, root=tmp_path),
        registry_path=paths["source_registry_path"],
    )
    for fold, run_id in zip((0, 4), PARENT_RUN_IDS, strict=True):
        run = previous[fold]
        run.update(
            run_id=run_id, directory=str(directory / run_id), sha256={"config.json": f"dark-{fold}"}
        )
        run["predictions"]["run_id"] = run_id
        run["robustness"]["run_id"] = run_id
        run["config"]["refinement_prerequisite_sha256"] = shared.compute_sha256(audit)
    return previous, sources, classes, evidence, {**paths, "darkening_directory": directory}


@pytest.mark.parametrize("fault", [None, "parent", "fold", "dropout", "audit"])
def test_completed_darkening_parents_are_checked_before_use(tmp_path, monkeypatch, fault):
    previous, sources, classes, evidence, paths = _completed_darkening(tmp_path)
    if fault == "parent":
        previous[4]["config"]["parent_run_id"] = "wrong"
    elif fault == "fold":
        previous[4]["fold"] = 1
    elif fault == "dropout":
        previous[4]["config"]["child_experiment"]["classifier_dropout"] = 0.45
    elif fault == "audit":
        sources["Drop30"][4]["sha256"]["config.json"] = "changed"
    monkeypatch.setattr(
        shared,
        "inspect_gender_run",
        lambda path, **kw: previous[dict(zip(PARENT_RUN_IDS, (0, 4), strict=True))[path.name]],
    )

    def verify():
        shared._add_completed_darkening_parents(
            sources,
            classes=classes,
            evidence=evidence,
            directory=paths["darkening_directory"],
            registry=None,
            splits=None,
            root=tmp_path,
        )

    if fault:
        with pytest.raises(ValueError):
            verify()
    else:
        verify()
        assert sources["Drop30Dark"][4]["run_id"] == PARENT_RUN_IDS[1]


def test_stronger_prerequisites_resolve_fold_four_and_keep_audit_on_resume(tmp_path, monkeypatch):
    previous, sources, classes, evidence, paths = _completed_darkening(tmp_path)
    sources["Drop30Dark"] = previous
    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    audit = tmp_path / "new-source-audit.json"
    identity = shared._source_identity(sources, spec, evidence, paths, root=tmp_path)
    shared._save_source_audit(audit, identity, registry_path=paths["source_registry_path"])
    before = audit.read_bytes()
    with paths["source_registry_path"].open("a") as handle:
        handle.write("completed-new-fold-0\n")
    shared._save_source_audit(audit, identity, registry_path=paths["source_registry_path"])
    assert audit.read_bytes() == before
    monkeypatch.setattr(shared, "require_narrow_prerequisites", lambda *a, **kw: evidence)

    def checked(**kwargs):
        assert kwargs["experiment_name"] == NAME
        assert Path(kwargs["darkening_directory"]) == paths["darkening_directory"]
        return sources, classes, spec, evidence

    monkeypatch.setattr(shared, "check_gender_dropout_darkening_sources", checked)
    result = shared.require_dropout_darkening_prerequisites(audit, spec=spec, fold=4, root=tmp_path)
    assert result["parent_directory"].name == PARENT_RUN_IDS[1]
    with pytest.raises(ValueError, match="direct parent directory"):
        shared.require_dropout_darkening_prerequisites(
            audit,
            spec=spec,
            fold=4,
            root=tmp_path,
            parent_run_directory=sources["Drop30"][4]["directory"],
        )


@pytest.mark.parametrize("mode", ["fresh", "reuse", "bad_reuse", "memory"])
def test_stronger_runner_uses_direct_darkening_parents_and_only_two_folds(
    tmp_path, monkeypatch, mode
):
    previous, sources, classes, evidence, paths = _completed_darkening(tmp_path)
    sources["Drop30Dark"] = previous
    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    child = copy.deepcopy(previous)
    audit = tmp_path / spec.artifact_dir / "gender/source_audit.json"
    for fold, run in child.items():
        run["run_id"] = f"stronger-{fold}"
        run["predictions"]["run_id"] = run["run_id"]
        run["robustness"]["run_id"] = run["run_id"]
        run["config"].update(
            child_experiment=spec.to_dict(), parent_run_id=PARENT_RUN_IDS[(0, 4).index(fold)]
        )
    events = []
    monkeypatch.setattr(
        shared,
        "check_gender_dropout_darkening_sources",
        lambda **kw: (sources, classes, spec, evidence),
    )
    monkeypatch.setattr(shared, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    monkeypatch.setattr(shared, "load_splits", lambda *a: None)

    def result(fold):
        child[fold]["config"]["refinement_prerequisite_sha256"] = shared.compute_sha256(audit)
        return {"run_dir": str(tmp_path / f"stronger-{fold}")}

    def reuse(spec, fold, **kw):
        if mode not in {"reuse", "bad_reuse"}:
            return None
        value = result(fold)
        if mode == "bad_reuse":
            child[fold]["config"]["parent_run_id"] = shared.PARENT_RUN_IDS[0]
        return value

    def fit(**kw):
        fold = kw["validation_fold"]
        events.append(("fit", fold))
        assert kw["parent_run_directory"] == previous[fold]["directory"]
        assert kw["registry_path"] == paths["source_registry_path"]
        assert kw["registry_mirrors"] == (tmp_path / "mirror.csv",)
        assert kw["child_spec"].classifier_dropout == 0.45
        if mode == "memory":
            child[fold]["metrics"]["peak_memory_bytes"] = 3_000_000_000
        return result(fold)

    def evaluate(run, **kw):
        events.append(("evaluate", run["run_id"]))
        return run

    monkeypatch.setattr(shared, "_reusable_fold", reuse)
    monkeypatch.setattr(shared, "_train_fold", fit)
    monkeypatch.setattr(shared, "inspect_gender_run", lambda path, **kw: child[int(path.name[-1])])
    monkeypatch.setattr(shared, "evaluate_gender_ieee", evaluate)
    screen_evaluate, compare = shared.evaluate_gender_narrow_screen, shared.compare_with_dropout
    monkeypatch.setattr(
        shared,
        "evaluate_gender_narrow_screen",
        lambda *a, **kw: screen_evaluate(*a, **kw, repetitions=20),
    )
    monkeypatch.setattr(shared, "compare_with_dropout", lambda *a: compare(*a, repetitions=20))
    args = dict(
        **paths,
        registry_path=paths["source_registry_path"],
        output_root=tmp_path,
        root=tmp_path,
        registry_mirrors=(tmp_path / "mirror.csv",),
    )
    if mode == "bad_reuse":
        with pytest.raises(ValueError, match="direct parent"):
            run_gender_stronger_dropout_screen(**args)
        assert not any(kind == "fit" for kind, value in events)
        return
    report = run_gender_stronger_dropout_screen(**args)
    assert [v for k, v in events if k == "fit"] == (
        [] if mode == "reuse" else [0] if mode == "memory" else [0, 4]
    )
    assert events[:6] == [
        ("evaluate", sources[name][f]["run_id"])
        for name in ("G2", "E6", "Drop30Dark")
        for f in (0, 4)
    ]
    if mode != "memory":
        assert report["rule_version"] == "gdrop045dark_loss003_gap005_v1"
        assert report["direct_parent_comparison"]["name"] == "Drop30Dark"
        assert "Drop30Dark" in report["incremental_comparison"]["comparison"]
        assert json.loads((audit.parent / "screen_decision.json").read_text()) == report


def test_stronger_dropout_keeps_gap_and_class_guards():
    child, sources, classes = _case()
    for run in child.values():
        run["metrics"]["parameter_count"] = 390181
    report = evaluate_gender_stronger_dropout_screen(child, sources, classes, repetitions=20)
    assert report["status"] == "pass"
    for run in child.values():
        run["metrics"]["final_train_eval_macro_f1"] += 0.08
        run["metrics"]["final_train_validation_macro_f1_gap"] += 0.08
    assert (
        _gates(evaluate_gender_stronger_dropout_screen(child, sources, classes, repetitions=20))[
            "mean_gap_reduction"
        ]
        == "fail"
    )


def test_new_notebook_compiles_and_runs_one_locked_trial():
    root = Path(__file__).resolve().parents[2]
    n = nbformat.read(
        root / "notebooks/04ab_task3_gender_stronger_dropout_screen.ipynb", as_version=4
    )
    nbformat.validate(n)
    code = "\n".join(c.source for c in n.cells if c.cell_type == "code")
    assert code.count("run_gender_stronger_dropout_screen(") == 1
    assert "darkening_directory=DARKENING_DIR" in code
    assert "spec.classifier_dropout == 0.45" in code
    assert "train_test_split" not in code and "pretrained=True" not in code
    for c in n.cells:
        if c.cell_type == "code":
            compile(c.source, "04ab", "exec")
            assert c.execution_count is None and not c.outputs


def test_actual_model_receives_stronger_dropout_and_direct_trainer_requires_audit(tmp_path):
    torch = pytest.importorskip("torch")
    from fashion.train.task3_baseline import _build_task3_model, run_task3_baseline_fold

    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    model = _build_task3_model(Task3BaselineConfig(target="gender"), spec)
    assert model.classifier_dropout.p == 0.45
    assert sum(p.numel() for p in model.parameters()) == 390181
    model.eval()
    features = torch.ones(10, 256)
    assert torch.equal(model.classifier_dropout(features), features)
    with pytest.raises(ValueError, match="required before training"):
        run_task3_baseline_fold("gender", 0, output_root=tmp_path, child_spec=spec, root=tmp_path)
