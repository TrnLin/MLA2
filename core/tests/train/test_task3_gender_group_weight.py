"""Training-only group weights, sealed evaluation and completed-parent lineage."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_task3_gender_name_truth import _fixture

import fashion.train.task3_gender_group_weight as screen
import fashion.train.task3_gender_name_truth as parent
from fashion.train.config import Task3BaselineConfig


def _training():
    rows = [("Shoes", "Men")] * 100 + [("Shoes", "Unisex")] * 4
    rows += [("Bags", "Men")] * 2 + [("Bags", "Unisex")] * 18
    rows += [("Rare", "Girls")] * 2
    frame = pd.DataFrame(rows, columns=["articleType", "gender"])
    frame["id"] = range(len(frame))
    frame["partition"] = "development"
    frame["cv_fold"] = 1
    frame["product_family_group"] = frame.id.map(lambda x: f"family{x}")
    return frame


def test_weights_cap_minority_boost_and_preserve_article_mass_without_resampling():
    original = _training()
    frame, contract = screen.add_gender_article_weights(original, validation_fold=0)
    pd.testing.assert_frame_equal(frame[original.columns], original)
    assert screen.WEIGHT_COLUMN not in original
    assert frame.raw_group_weight.min() == 1 and frame.raw_group_weight.max() == 3
    assert np.allclose(
        frame.groupby("articleType")[screen.WEIGHT_COLUMN].sum(),
        original.groupby("articleType").size(),
    )
    weights = frame.groupby(["articleType", "gender"])[screen.WEIGHT_COLUMN].first()
    assert weights["Shoes", "Unisex"] / weights["Shoes", "Men"] == pytest.approx(3)
    assert weights["Bags", "Men"] / weights["Bags", "Unisex"] == pytest.approx(3)
    assert weights["Rare", "Girls"] == 1
    assert contract["row_mean"] == pytest.approx(1)
    assert contract["maximum"] <= 3 and contract["minimum"] >= 1 / 3
    reordered, _ = screen.add_gender_article_weights(
        original.sample(frac=1, random_state=4), validation_fold=0
    )
    pd.testing.assert_series_equal(
        frame.set_index("id")[screen.WEIGHT_COLUMN],
        reordered.set_index("id")[screen.WEIGHT_COLUMN].sort_index(),
    )


@pytest.mark.parametrize(
    "fault", ["validation", "test", "duplicate", "bad_gender", "null", "fractional"]
)
def test_weights_reject_leakage_and_invalid_rows(fault):
    frame = _training()
    if fault == "validation":
        frame.loc[0, "cv_fold"] = 0
    elif fault == "test":
        frame.loc[0, "partition"] = "held_out"
    elif fault == "duplicate":
        frame.loc[1, "id"] = 0
    elif fault == "bad_gender":
        frame.loc[0, "gender"] = "unknown"
    elif fault == "null":
        frame.loc[0, "articleType"] = None
    else:
        frame["cv_fold"] = 1.5
    with pytest.raises(ValueError, match="valid training rows"):
        screen.add_gender_article_weights(frame, validation_fold=0)


def test_only_loss_weighting_changes_parent_controls(monkeypatch):
    contract = {"variant_id": "gender_name_truth_v1", "labels_sha256": "frozen"}
    monkeypatch.setattr(screen, "label_contract", lambda root: contract)
    spec = screen.group_weight_spec()
    old = parent.NameTruthSpec(spec.label_contract_json)
    metadata = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "parent_run_ids",
        "parent_artifact_dir",
        "changed_factor",
        "screen_rule_version",
        "sample_weight_strategy",
        "sample_weight_policy",
    }
    assert {k: v for k, v in spec.to_dict().items() if k not in metadata} == {
        k: v for k, v in old.to_dict().items() if k not in metadata
    }
    assert spec.classifier_dropout == 0.30 and spec.to_dict()["grayscale_probability"] == 0.10
    assert spec.parent_run_id_for_fold(4) == screen.PARENT_RUN_IDS[1]
    assert spec.sample_weight_strategy == screen.STRATEGY
    assert screen.group_weight_config(spec, fold=0, device_name="cuda") == Task3BaselineConfig(
        target="gender"
    )
    for fold, device in [(1, "cuda"), (0, "cpu")]:
        with pytest.raises(ValueError, match="only folds"):
            screen.group_weight_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="recipe changed"):
        screen.group_weight_config(old, fold=0, device_name="cuda")


@pytest.mark.parametrize("fault", [None, "labels", "parent", "audit"])
def test_completed_name_truth_sources_have_exact_lineage(tmp_path, monkeypatch, fault):
    _, sources, classes, old_spec, evidence, paths, contract = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(screen, "label_contract", lambda root: contract)
    identity = {"paths": {key: str(value) for key, value in paths.items()}}
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps({"identity": identity}))
    monkeypatch.setattr(
        screen, "parent_source_identity", lambda *a, **kw: {} if fault == "audit" else identity
    )
    monkeypatch.setattr(
        screen,
        "check_gender_name_truth_sources",
        lambda **kw: (sources, classes, old_spec, evidence),
    )
    splits = pd.DataFrame({"gender": ["Girls"]})
    monkeypatch.setattr(screen, "load_gender_name_truth_variant", lambda root: splits)

    def inspect(path, **kw):
        assert kw["splits"] is splits
        fold = (0, 4)[screen.PARENT_RUN_IDS.index(path.name)]
        return dict(
            run_id=path.name,
            fold=fold,
            config={
                **Task3BaselineConfig(target="gender").to_dict(),
                "child_experiment": old_spec.to_dict(),
                "parent_run_id": "bad"
                if fault == "parent"
                else parent.PARENT_RUN_IDS[(0, 4).index(fold)],
                "training_precision_settings": parent.TRAINING_PRECISION,
                "precision_evidence_sha256": evidence["artifact_sha256"],
                "refinement_prerequisite_sha256": screen.compute_sha256(audit),
                "gender_label_variant": {} if fault == "labels" else contract,
            },
        )

    monkeypatch.setattr(screen, "inspect_gender_run", inspect)
    if fault:
        with pytest.raises(ValueError, match="labels differ|parent disagrees|audit differs"):
            screen.check_gender_group_weight_sources(
                name_truth_directory=tmp_path, root=tmp_path, **paths
            )
    else:
        checked, _, spec, _ = screen.check_gender_group_weight_sources(
            name_truth_directory=tmp_path, root=tmp_path, **paths
        )
        assert set(checked["NameTruth"]) == {0, 4} and spec.name == screen.NAME


def test_weight_artifact_replay_rejects_tampering(tmp_path, monkeypatch):
    frame, contract = screen.add_gender_article_weights(_training(), validation_fold=0)
    path = tmp_path / "training_selection.csv"
    frame[screen.SELECTION_COLUMNS].to_csv(path, index=False)
    monkeypatch.setattr(screen, "get_cv_split", lambda *a: (_training(), None))
    monkeypatch.setattr(screen, "get_samples", lambda frame, **kw: frame)
    run = {
        "config": {
            "sample_weight_strategy": screen.STRATEGY,
            "training_selection_contract": {
                "sample_weight_contract": contract,
                "artifact_sha256": screen.compute_sha256(path),
            },
        }
    }
    screen.verify_weight_evidence(run, fold=0, splits=None, directory=tmp_path)
    path.write_text(path.read_text().replace("Unisex", "Women"))
    with pytest.raises(ValueError, match="weights differ"):
        screen.verify_weight_evidence(run, fold=0, splits=None, directory=tmp_path)


@pytest.mark.parametrize("mode", ["fresh", "reuse", "bad_weights", "memory"])
def test_shared_runner_uses_name_truth_parent_and_two_folds(tmp_path, monkeypatch, mode):
    child, sources, classes, _, evidence, paths, contract = _fixture(tmp_path, monkeypatch)
    spec = screen.GroupWeightSpec(json.dumps(contract, sort_keys=True))
    sources["NameTruth"] = copy.deepcopy(sources["Gray10"])
    for fold in (0, 4):
        run_id = spec.parent_run_id_for_fold(fold)
        direct = sources["NameTruth"][fold]
        direct.update(run_id=run_id, directory=str(tmp_path / run_id))
        direct["predictions"]["run_id"] = run_id
        direct["robustness"]["run_id"] = run_id
        child[fold]["config"].update(child_experiment=spec.to_dict(), parent_run_id=run_id)
    audit = tmp_path / spec.artifact_dir / "gender/source_audit.json"
    fitted, verified = [], []

    def result(fold):
        child[fold]["config"]["refinement_prerequisite_sha256"] = screen.compute_sha256(audit)
        return {"run_dir": str(tmp_path / f"child-{fold}")}

    def fit(**kw):
        fold = kw["validation_fold"]
        fitted.append(fold)
        assert kw["parent_run_directory"] == sources["NameTruth"][fold]["directory"]
        if mode == "memory":
            child[fold]["metrics"]["peak_memory_bytes"] = 3_000_000_000
        return result(fold)

    def verify(run, *, fold, **kw):
        verified.append(fold)
        if mode == "bad_weights":
            raise ValueError("bad weights")

    def evaluate(run, **kw):
        assert kw["label_variant"] == contract
        run["evaluation_manifest"] = {"identity": {"label_variant": contract}}
        return run

    monkeypatch.setattr(
        parent,
        "_reusable_fold",
        lambda spec, fold, **kw: result(fold) if mode in {"reuse", "bad_weights"} else None,
    )
    monkeypatch.setattr(parent, "_train_fold", fit)
    monkeypatch.setattr(parent, "inspect_gender_run", lambda path, **kw: child[int(path.name[-1])])
    monkeypatch.setattr(parent, "evaluate_gender_ieee", evaluate)
    monkeypatch.setattr(parent, "write_original_label_diagnostic", lambda *a, **kw: None)
    score = parent.evaluate_gender_narrow_screen
    compare = parent.compare_with_dropout
    monkeypatch.setattr(
        parent, "evaluate_gender_narrow_screen", lambda *a, **kw: score(*a, **kw, repetitions=20)
    )
    monkeypatch.setattr(parent, "compare_with_dropout", lambda *a: compare(*a, repetitions=20))
    args = dict(
        sources=sources,
        classes=classes,
        spec=spec,
        evidence=evidence,
        identity={},
        splits=pd.DataFrame(),
        source_registry_path=paths["source_registry_path"],
        output_root=tmp_path,
        registry_path=paths["source_registry_path"],
        registry_mirrors=(),
        root=tmp_path,
        device_name="cuda",
        parent_group="NameTruth",
        candidate_group="GroupWeight",
        verify_candidate=verify,
    )
    if mode == "bad_weights":
        with pytest.raises(ValueError, match="bad weights"):
            parent._run_verified_label_screen(**args)
        assert not fitted
        return
    report = parent._run_verified_label_screen(**args)
    assert fitted == ([] if mode == "reuse" else [0] if mode == "memory" else [0, 4])
    assert verified == ([0] if mode == "memory" else [0, 4])
    if mode != "memory":
        assert len(report["checks"]) == 19
        assert report["rule_version"] == screen.RULE_VERSION
        assert set(report["direct_parent_run_ids"].values()) == set(screen.PARENT_RUN_IDS)
        assert "NameTruth" in report["incremental_comparison"]["comparison"]
