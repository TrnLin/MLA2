"""Stronger blending must protect useful scores and preserve the completed 0.2 recipe."""

import copy
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from test_task3_gender_mixup import CLASSES, MAPPING, _training
from test_task3_gender_name_truth import _fixture

import fashion.train.task3_gender_name_truth as parent
import fashion.train.task3_gender_stronger_mixup as screen
from fashion.train.config import Task3BaselineConfig
from fashion.train.mixup import POLICY, TrainingMixUp, policy_for_alpha
from fashion.train.task3_gender_mixup import MixUpSpec, verify_mixup_evidence


def test_strength_is_the_only_training_change(monkeypatch):
    monkeypatch.setattr(screen, "label_contract", lambda root: {"labels_sha256": "frozen"})
    spec = screen.mixup40_spec()
    old = MixUpSpec(spec.label_contract_json)
    changed = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
        "parent_run_ids",
        "screen_rule_version",
        "mixup_policy",
        "improvement_rules",
    }
    assert {k: v for k, v in spec.to_dict().items() if k not in changed} == {
        k: v for k, v in old.to_dict().items() if k not in changed
    }
    assert policy_for_alpha() == POLICY == old.to_dict()["mixup_policy"]
    new_policy = spec.to_dict()["mixup_policy"]
    assert new_policy["alpha"] == 0.4
    assert {k: v for k, v in new_policy.items() if k not in {"alpha", "version"}} == {
        k: v for k, v in POLICY.items() if k not in {"alpha", "version"}
    }
    assert spec.parent_run_id_for_fold(4) == screen.PARENT_RUN_IDS[1]
    assert screen.mixup40_config(spec, fold=0, device_name="cuda") == Task3BaselineConfig(
        target="gender"
    )
    for fold, device in [(1, "cuda"), (0, "cpu")]:
        with pytest.raises(ValueError, match="only folds"):
            screen.mixup40_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="frozen recipe"):
        screen.mixup40_config(old, fold=0, device_name="cuda")
    with pytest.raises(ValueError, match="frozen MixUp"):
        policy_for_alpha(0.8)


@pytest.mark.parametrize("alpha", [0.2, 0.4])
def test_actual_blending_uses_selected_beta_strength(alpha):
    torch = pytest.importorskip("torch")
    mix = TrainingMixUp(_training(), validation_fold=0, label_to_index=MAPPING, alpha=alpha)
    mix.begin_epoch(1)
    rng = np.random.Generator(np.random.PCG64(2753 ^ 0x4D495855))
    expected_lambda = float(rng.beta(alpha, alpha))
    order = rng.permutation(10)
    images, labels = torch.arange(10, dtype=torch.float32).reshape(10, 1), torch.arange(10) % 5
    actual, partners, lam = mix.apply(images, labels, torch.arange(10))
    assert lam == expected_lambda
    torch.testing.assert_close(actual, lam * images + (1 - lam) * images[order])
    torch.testing.assert_close(partners, labels[order])
    mix.end_epoch()


def _report():
    return {
        "status": "pass",
        "checks": [{"gate": f"old_{i}", "status": "pass"} for i in range(14)]
        + [{"gate": name, "status": "pass"} for name in sorted(screen.DIAGNOSTIC_GATES)],
        "incremental_comparison": {
            "folds": [
                {"fold": f, "gap_reduction": 0.025, "delta_validation_f1": 0.005} for f in (0, 4)
            ],
            "validation_delta": 0.005,
            "validation_interval": {"lower_95": -0.005},
            "class_f1_delta": dict.fromkeys(CLASSES, 0.005),
            "candidate": {
                "macro_f1": 0.815,
                "per_class": [{"class_name": "Unisex", "recall": 0.52}],
            },
            "dropout": {"per_class": [{"class_name": "Unisex", "recall": 0.51}]},
        },
    }


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "gap",
        "one_fold_gap",
        "val",
        "unisex",
        "historical",
        "nan",
    ],
)
def test_smaller_gap_alone_cannot_pass_the_new_rules(fault):
    report = _report()
    comparison = report["incremental_comparison"]
    if fault == "gap":
        for row in comparison["folds"]:
            row["gap_reduction"] = 0.01
    elif fault == "one_fold_gap":
        comparison["folds"][0]["gap_reduction"] = 0.0
    elif fault == "val":
        comparison["candidate"]["macro_f1"] = 0.7399
    elif fault == "unisex":
        comparison["candidate"]["per_class"][0]["recall"] = 0.50
    elif fault == "historical":
        report["checks"][0]["status"] = "fail"
        report["status"] = "fail"
    elif fault == "nan":
        comparison["candidate"]["macro_f1"] = float("nan")
    original_checks = copy.deepcopy(report["checks"])
    result = screen.apply_improvement_rules(report)
    assert len(result["checks"]) == 19
    assert result["status"] == ("fail" if fault else "pass")
    assert report["checks"] == original_checks


@pytest.mark.parametrize("f1", [0.74, 0.75, 0.808909])
def test_validation_floor_allows_relative_f1_losses(f1):
    report = _report()
    report["status"] = "fail"
    for item in report["checks"]:
        if item["gate"] in screen.DIAGNOSTIC_GATES:
            item["status"] = "fail"
    comparison = report["incremental_comparison"]
    comparison["candidate"]["macro_f1"] = f1
    comparison["validation_delta"] = f1 - 0.808909
    comparison["validation_interval"]["lower_95"] = -0.10
    comparison["class_f1_delta"]["Girls"] = -0.10
    for row in comparison["folds"]:
        row["delta_validation_f1"] = -0.10
    result = screen.apply_improvement_rules(report)
    assert result["status"] == "pass"
    assert result["historical_screen_status"] == "fail"
    assert len(result["diagnostic_checks"]) == 5
    assert all(c["status"] == "fail" for c in result["diagnostic_checks"])
    assert result["required_validation_f1"] == 0.74


def test_completed_parent_hashes_come_from_its_commit(monkeypatch, tmp_path):
    calls = []

    def git(command, *, cwd):
        assert cwd == tmp_path
        assert command[:2] == ["git", "show"]
        assert command[2].startswith(screen.PARENT_COMMIT + ":src/fashion/train/")
        calls.append(command[2])
        return b"original parent implementation"

    monkeypatch.setattr(screen.subprocess, "check_output", git)
    assert set(screen.recorded_mixup20_hashes(tmp_path).values()) == {
        hashlib.sha256(b"original parent implementation").hexdigest()
    }
    assert len(calls) == 3


@pytest.mark.parametrize("fault", [None, "audit", "labels", "parent", "receipt", "decision"])
def test_completed_mixup_parent_lineage_uses_original_code(tmp_path, monkeypatch, fault):
    _, sources, classes, _, evidence, paths, contract = _fixture(tmp_path, monkeypatch)
    spec = MixUpSpec(json.dumps(contract, sort_keys=True))
    monkeypatch.setattr(screen, "label_contract", lambda root: contract)
    identity = {"paths": paths, "mixup_implementation_sha256": {"mixup.py": "original"}}
    identity["paths"] = {k: str(v) for k, v in paths.items()}
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps({"identity": identity}))
    (tmp_path / "screen_decision.json").write_text(
        json.dumps(
            {
                "status": "fail" if fault == "decision" else "pass",
                "run_ids": dict(zip(["0", "4"], screen.PARENT_RUN_IDS)),
            }
        )
    )
    monkeypatch.setattr(
        screen, "check_gender_mixup_sources", lambda **kw: (sources, classes, spec, evidence)
    )
    monkeypatch.setattr(
        screen,
        "mixup20_source_identity",
        lambda *a, **kw: {
            **identity,
            "mixup_implementation_sha256": {"mixup.py": "new trial code"},
        },
    )
    monkeypatch.setattr(
        screen,
        "recorded_mixup20_hashes",
        lambda root: {"mixup.py": "wrong" if fault == "audit" else "original"},
    )
    splits = pd.DataFrame({"gender": ["Girls"]})
    monkeypatch.setattr(screen, "load_gender_name_truth_variant", lambda root: splits)
    checked = []

    def inspect(path, **kw):
        assert kw["splits"] is splits
        fold = (0, 4)[screen.PARENT_RUN_IDS.index(path.name)]
        return {
            "run_id": path.name,
            "fold": fold,
            "config": {
                **Task3BaselineConfig(target="gender").to_dict(),
                "child_experiment": spec.to_dict(),
                "parent_run_id": "wrong"
                if fault == "parent"
                else spec.parent_run_id_for_fold(fold),
                "training_precision_settings": parent.TRAINING_PRECISION,
                "precision_evidence_sha256": evidence["artifact_sha256"],
                "refinement_prerequisite_sha256": screen.compute_sha256(audit),
                "gender_label_variant": {} if fault == "labels" else contract,
            },
        }

    def receipt(run, **kw):
        checked.append(kw["fold"])
        if fault == "receipt":
            raise ValueError("invalid receipt")

    monkeypatch.setattr(screen, "inspect_gender_run", inspect)
    monkeypatch.setattr(screen, "verify_mixup_evidence", receipt)
    if fault:
        with pytest.raises(
            ValueError, match="audit differs|different labels|parent disagrees|receipt|passing"
        ):
            screen.check_gender_stronger_mixup_sources(
                mixup_directory=tmp_path, root=tmp_path, **paths
            )
    else:
        result, _, new, _ = screen.check_gender_stronger_mixup_sources(
            mixup_directory=tmp_path, root=tmp_path, **paths
        )
        assert checked == [0, 4] and set(result["MixUp20"]) == {0, 4}
        assert new.name == screen.NAME


def test_receipts_cannot_be_reused_across_strengths(tmp_path, monkeypatch):
    import fashion.train.task3_gender_mixup as base

    mix = TrainingMixUp(_training(), validation_fold=0, label_to_index=MAPPING, alpha=0.4)
    for epoch in range(1, 31):
        mix.begin_epoch(epoch)
        mix.plan(list(range(10)), [i % 5 for i in range(10)])
        mix.end_epoch()
    path = tmp_path / "mixup_training.json"
    path.write_text(json.dumps(mix.receipt()))
    pd.DataFrame(
        {
            "epoch": range(1, 31),
            "train_macro_f1": np.nan,
            "train_metric_scope": POLICY["online_train_f1"],
        }
    ).to_csv(tmp_path / "history.csv", index=False)
    monkeypatch.setattr(base, "get_cv_split", lambda *a: (_training(), None))
    monkeypatch.setattr(base, "get_samples", lambda frame, **kw: frame)
    run = {
        "config": {"mixup_contract": mix.contract},
        "metrics": {"mixup_receipt_sha256": screen.compute_sha256(path)},
    }
    screen.verify_mixup40_evidence(run, fold=0, splits=None, directory=tmp_path)
    with pytest.raises(ValueError, match="frozen contract"):
        verify_mixup_evidence(run, fold=0, splits=None, directory=tmp_path)


def test_stronger_notebook_is_a_single_unexecuted_trial():
    root = Path(__file__).resolve().parents[2]
    nb = nbformat.read(root / "notebooks/04ah_task3_gender_stronger_mixup_screen.ipynb", 4)
    nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert code.count("run_gender_stronger_mixup_screen(") == 1
    assert "mixup_directory=MIXUP_DIR" in code
    assert "train_test_split" not in code and "pretrained=True" not in code
    assert "All 19 checks" in "\n".join(c.source for c in nb.cells)
    for c in nb.cells:
        if c.cell_type == "code":
            compile(c.source, "04ah", "exec")
            assert c.execution_count is None and not c.outputs


@pytest.mark.parametrize("mode", ["fresh", "reuse", "bad_mixup", "memory"])
def test_stronger_runner_checks_reused_runs_and_applies_new_rules(tmp_path, monkeypatch, mode):
    child, sources, classes, _, evidence, paths, contract = _fixture(tmp_path, monkeypatch)
    spec = screen.MixUp40Spec(json.dumps(contract, sort_keys=True))
    sources["MixUp20"] = copy.deepcopy(sources["Gray10"])
    for fold in (0, 4):
        run_id = spec.parent_run_id_for_fold(fold)
        direct = sources["MixUp20"][fold]
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
        assert kw["parent_run_directory"] == sources["MixUp20"][fold]["directory"]
        if mode == "memory":
            child[fold]["metrics"]["peak_memory_bytes"] = 3_000_000_000
        return result(fold)

    def verify(run, *, fold, **kw):
        verified.append(fold)
        if mode == "bad_mixup":
            raise ValueError("bad MixUp")

    def evaluate(run, **kw):
        assert kw["label_variant"] == contract
        run["evaluation_manifest"] = {"identity": {"label_variant": contract}}
        return run

    monkeypatch.setattr(
        parent,
        "_reusable_fold",
        lambda spec, fold, **kw: result(fold) if mode in {"reuse", "bad_mixup"} else None,
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
        parent_group="MixUp20",
        candidate_group="MixUp40",
        verify_candidate=verify,
        refine_report=screen.apply_improvement_rules,
    )
    if mode == "bad_mixup":
        with pytest.raises(ValueError, match="bad MixUp"):
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
        assert "MixUp20" in report["incremental_comparison"]["comparison"]
