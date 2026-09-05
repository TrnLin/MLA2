"""Check source lineage, matching label bases, reuse and the two-fold stop."""

import copy
import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from test_task3_gender_narrow import _case

import fashion.train.task3_gender_name_truth as screen
from fashion.data.gender_name_truth import VARIANT_RELATIVE_PATH
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_gender_narrow import evaluate_gender_narrow_screen


def _fixture(tmp_path, monkeypatch):
    child, sources, classes = _case()
    sources["Gray10"] = copy.deepcopy(sources["G2"])
    contract = {"variant_id": "gender_name_truth_v1", "labels_sha256": "verified"}
    spec = screen.NameTruthSpec(json.dumps(contract, sort_keys=True))
    evidence = {"artifact_sha256": {"precision": "verified"}}
    monkeypatch.setattr(screen, "label_contract", lambda root: contract)
    for group in sources.values():
        for fold, run in group.items():
            run.update(
                fold=fold,
                sha256={"config.json": run["run_id"]},
                directory=str(tmp_path / run["run_id"]),
            )
    for fold, parent in zip((0, 4), screen.PARENT_RUN_IDS, strict=True):
        sources["Gray10"][fold]["run_id"] = parent
        sources["Gray10"][fold]["directory"] = str(tmp_path / parent)
        sources["Gray10"][fold]["predictions"]["run_id"] = parent
        sources["Gray10"][fold]["robustness"]["run_id"] = parent
        child[fold].update(
            fold=fold,
            config={
                **Task3BaselineConfig(target="gender").to_dict(),
                "child_experiment": spec.to_dict(),
                "gender_label_variant": contract,
                "parent_run_id": parent,
                "training_precision_settings": screen.TRAINING_PRECISION,
                "precision_evidence_sha256": evidence["artifact_sha256"],
            },
        )
        child[fold]["metrics"]["parameter_count"] = 390181
    split = tmp_path / "data/processed/splits.csv"
    split.parent.mkdir(parents=True)
    split.write_text("canonical\n")
    variant = tmp_path / VARIANT_RELATIVE_PATH
    variant.mkdir(parents=True)
    (variant / "summary.json").write_text(json.dumps(contract))
    registry = tmp_path / "runs.csv"
    registry.write_text("run_id\n")
    paths = dict(
        g2_directory=tmp_path,
        e6_directory=tmp_path,
        dropout_directory=tmp_path,
        darkening_directory=tmp_path,
        grayscale_directory=tmp_path,
        precision_directory=tmp_path,
        source_registry_path=registry,
    )
    return child, sources, classes, spec, evidence, paths, contract


@pytest.mark.parametrize("mode", ["fresh", "reuse", "bad_reuse", "memory", "mixed_labels"])
def test_runner_matches_labels_before_comparing_and_stops_after_two_folds(
    tmp_path,
    monkeypatch,
    mode,
):
    child, sources, classes, spec, evidence, paths, contract = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        screen, "check_gender_name_truth_sources", lambda **kw: (sources, classes, spec, evidence)
    )
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    frame = pd.DataFrame({"gender": ["Girls"]})
    monkeypatch.setattr(screen, "training_splits", lambda *a, **kw: frame)
    events = []
    audit = tmp_path / spec.artifact_dir / "gender/source_audit.json"

    def result(fold):
        child[fold]["config"]["refinement_prerequisite_sha256"] = screen.compute_sha256(audit)
        return {"run_dir": str(tmp_path / f"child-{fold}")}

    def reuse(spec, fold, **kw):
        if mode not in {"reuse", "bad_reuse"}:
            return None
        value = result(fold)
        if mode == "bad_reuse":
            child[fold]["config"]["gender_label_variant"] = {}
        return value

    def fit(**kw):
        fold = kw["validation_fold"]
        events.append(("fit", fold))
        assert kw["child_spec"] == spec
        assert kw["parent_run_directory"] == sources["Gray10"][fold]["directory"]
        assert kw["registry_mirrors"] == (tmp_path / "mirror.csv",)
        if mode == "memory":
            child[fold]["metrics"]["peak_memory_bytes"] = 3_000_000_000
        return result(fold)

    def inspect(path, **kw):
        assert kw["splits"] is frame
        return child[int(path.name[-1])]

    def evaluate(run, **kw):
        assert kw["splits"] is frame and kw["label_variant"] == contract
        assert kw["output"].parent.name == "comparison_name_truth_ieee"
        events.append(("evaluate", run["run_id"]))
        run["evaluation_manifest"] = {"identity": {"label_variant": contract}}
        if mode == "mixed_labels" and run["run_id"] == "E6-0":
            run["evaluation_manifest"]["identity"]["label_variant"] = {}
        return run

    monkeypatch.setattr(screen, "_reusable_fold", reuse)
    monkeypatch.setattr(screen, "_train_fold", fit)
    monkeypatch.setattr(screen, "inspect_gender_run", inspect)
    monkeypatch.setattr(screen, "evaluate_gender_ieee", evaluate)
    monkeypatch.setattr(screen, "write_original_label_diagnostic", lambda *a, **kw: None)
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
        registry_mirrors=(tmp_path / "mirror.csv",),
        root=tmp_path,
    )
    if mode in {"bad_reuse", "mixed_labels"}:
        with pytest.raises(ValueError, match="verified name-truth|different gender labels"):
            screen.run_gender_name_truth_screen(**args)
        assert not any(kind == "fit" for kind, _ in events)
        return
    report = screen.run_gender_name_truth_screen(**args)
    assert [v for k, v in events if k == "fit"] == (
        [] if mode == "reuse" else [0] if mode == "memory" else [0, 4]
    )
    assert events[:6] == [
        ("evaluate", sources[n][f]["run_id"]) for n in ("G2", "E6", "Gray10") for f in (0, 4)
    ]
    assert json.loads((audit.parent / "screen_decision.json").read_text()) == report
    if mode != "memory":
        assert len(report["checks"]) == 19 and report["status"] == "pass"
        assert report["comparison_label_basis"] == contract
        assert not report["independent_blind_test"]
        assert report["rule_version"] == screen.RULE_VERSION


@pytest.mark.parametrize("fault", [None, "parent", "audit"])
def test_completed_grayscale_sources_are_checked_against_original_labels(
    tmp_path,
    monkeypatch,
    fault,
):
    _, sources, classes, _, evidence, paths, _ = _fixture(tmp_path, monkeypatch)
    previous = sources.pop("Gray10")
    old_spec = dataset_v2_spec(screen.GRAYSCALE_NAME, screen.DARKENING_RUN_IDS)
    identity = {"paths": {k: str(v) for k, v in paths.items() if k != "grayscale_directory"}}
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps({"identity": identity}))
    monkeypatch.setattr(
        screen, "grayscale_source_identity", lambda *a, **kw: {} if fault == "audit" else identity
    )
    for fold, run in previous.items():
        run["config"] = {
            **Task3BaselineConfig(target="gender").to_dict(),
            "child_experiment": old_spec.to_dict(),
            "parent_run_id": screen.DARKENING_RUN_IDS[(0, 4).index(fold)],
            "training_precision_settings": screen.TRAINING_PRECISION,
            "precision_evidence_sha256": evidence["artifact_sha256"],
            "refinement_prerequisite_sha256": screen.compute_sha256(audit),
        }
    if fault == "parent":
        previous[4]["config"]["parent_run_id"] = "wrong"
    monkeypatch.setattr(
        screen,
        "check_gender_grayscale_sources",
        lambda **kw: (sources, classes, old_spec, evidence),
    )
    canonical = pd.DataFrame({"gender": ["Women"]})
    monkeypatch.setattr(screen, "load_splits", lambda *a: canonical)
    inspected = []

    def inspect(path, **kw):
        assert kw["splits"] is canonical
        inspected.append(path.name)
        return previous[(0, 4)[screen.PARENT_RUN_IDS.index(path.name)]]

    monkeypatch.setattr(screen, "inspect_gender_run", inspect)
    if fault:
        with pytest.raises(ValueError, match="audit differs|parent disagrees"):
            screen.check_gender_name_truth_sources(**paths, root=tmp_path)
    else:
        checked, _, spec, _ = screen.check_gender_name_truth_sources(**paths, root=tmp_path)
        assert inspected == list(screen.PARENT_RUN_IDS)
        assert set(checked["Gray10"]) == {0, 4} and spec.name == screen.NAME


def test_prerequisite_replay_rejects_changed_labels_and_wrong_parent(tmp_path, monkeypatch):
    _, sources, classes, spec, evidence, paths, contract = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        screen, "check_gender_name_truth_sources", lambda **kw: (sources, classes, spec, evidence)
    )
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    audit = tmp_path / "audit.json"
    screen._save_source_audit(
        audit,
        screen._source_identity(
            sources,
            spec,
            evidence,
            paths,
            root=tmp_path,
        ),
        registry_path=paths["source_registry_path"],
    )
    checked = screen.require_name_truth_prerequisites(audit, spec=spec, fold=4, root=tmp_path)
    assert checked["parent_directory"].name == screen.PARENT_RUN_IDS[1]
    with pytest.raises(ValueError, match="verified grayscale parent"):
        screen.require_name_truth_prerequisites(
            audit,
            spec=spec,
            fold=4,
            root=tmp_path,
            parent_run_directory=tmp_path / "wrong",
        )
    contract["labels_sha256"] = "changed"
    with pytest.raises(ValueError, match="labels or frozen recipe changed"):
        screen.require_name_truth_prerequisites(audit, spec=spec, fold=4, root=tmp_path)


def test_notebook_is_one_unexecuted_label_trial():
    root = Path(__file__).resolve().parents[2]
    nb = nbformat.read(root / "notebooks/04ad_task3_gender_name_truth_screen.ipynb", as_version=4)
    nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert code.count("run_gender_name_truth_screen(") == 1
    assert "build_gender_name_truth_variant(REPO_DIR)" in code
    assert "train_test_split" not in code and "pretrained=True" not in code
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04ad", "exec")
            assert cell.execution_count is None and not cell.outputs
