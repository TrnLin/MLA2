"""MixUp math, training isolation, reproducibility and frozen parent controls."""

import copy
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from test_task3_gender_name_truth import _fixture

import fashion.train.task3_gender_mixup as screen
import fashion.train.task3_gender_name_truth as parent
from fashion.train.config import Task3BaselineConfig
from fashion.train.mixup import POLICY, TrainingMixUp, training_contract
from fashion.train.task3_gender_name_truth import NameTruthSpec

CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
MAPPING = dict(zip(CLASSES, range(5)))


def _training():
    return pd.DataFrame(
        {
            "id": range(10),
            "partition": "development",
            "cv_fold": 1,
            "product_family_group": [f"f{i}" for i in range(10)],
            "gender": CLASSES * 2,
        }
    )


def _mixer():
    return TrainingMixUp(_training(), validation_fold=0, label_to_index=MAPPING)


@pytest.mark.parametrize("fault", ["validation", "holdout", "duplicate", "label", "null"])
def test_mixup_refuses_invalid_training_scope(fault):
    frame = _training()
    column, value = {
        "validation": ("cv_fold", 0),
        "holdout": ("partition", "holdout"),
        "duplicate": ("id", 1),
        "label": ("gender", "Unknown"),
        "null": ("product_family_group", None),
    }[fault]
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="fold-training"):
        training_contract(frame, validation_fold=0)


def test_mixup_random_stream_and_coverage_are_reproducible():
    first, second = _mixer(), _mixer()
    np.random.seed(123)
    expected_global = np.random.random(4)
    np.random.seed(123)
    for epoch in (1, 2):
        first.begin_epoch(epoch)
        second.begin_epoch(epoch)
        for ids in (list(range(6)), list(range(6, 10))):
            labels = [i % 5 for i in ids]
            lam, order = first.plan(ids, labels)
            other_lam, other_order = second.plan(ids, labels)
            assert 0 <= lam <= 1 and sorted(order) == list(range(len(ids)))
            assert lam == other_lam and np.array_equal(order, other_order)
        first.end_epoch()
        second.end_epoch()
    np.testing.assert_array_equal(np.random.random(4), expected_global)
    assert first.receipt() == second.receipt()
    epochs = first.receipt()["epochs"]
    assert all(e["rows"] == 10 and e["batches"] == 2 for e in epochs)
    assert epochs[0]["mix_plan_sha256"] != epochs[1]["mix_plan_sha256"]


def test_mixup_rejects_wrong_batch_labels_repeated_rows_and_partial_epochs():
    mix = _mixer()
    with pytest.raises(ValueError, match="non-training"):
        mix.plan([0], [0])
    mix.begin_epoch(1)
    for ids, labels in [([999], [0]), ([0], [1]), ([0, 0], [0, 0])]:
        with pytest.raises(ValueError, match="non-training"):
            mix.plan(ids, labels)
    mix.plan([0], [0])
    with pytest.raises(ValueError, match="non-training"):
        mix.plan([0], [0])
    with pytest.raises(ValueError, match="exactly once"):
        mix.end_epoch()
    with pytest.raises(ValueError, match="unfinished"):
        mix.receipt()


def test_training_pass_matches_soft_label_loss_and_gradients_without_mutating_inputs():
    torch = pytest.importorskip("torch")
    from fashion.train.task3_baseline import _pass

    torch.manual_seed(17)
    model = torch.nn.Linear(3, 5)
    reference = torch.nn.Linear(3, 5)
    reference.load_state_dict(model.state_dict())
    images = torch.arange(30, dtype=torch.float32).reshape(10, 3) / 30
    original = images.clone()
    labels = torch.tensor(list(range(5)) * 2)
    planner = _mixer()
    planner.begin_epoch(1)
    mixed, partners, lam = planner.apply(images, labels, torch.arange(10))
    # Independent soft-target CE must give the same loss and SGD update.
    soft = lam * torch.nn.functional.one_hot(labels, 5) + (1 - lam) * (
        torch.nn.functional.one_hot(partners, 5)
    )
    expected_loss = -(soft * reference(mixed).log_softmax(dim=1)).sum(1).mean()
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected_loss.backward()
    expected_optimizer.step()
    mix = _mixer()
    mix.begin_epoch(1)
    random_state = torch.get_rng_state().clone()
    actual_loss, train_labels, probabilities, _ = _pass(
        model,
        [{"image": images, "label": labels, "id": torch.arange(10)}],
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        mixup=mix,
    )
    mix.end_epoch()
    assert actual_loss == pytest.approx(expected_loss.item())
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(images, original)
    assert train_labels.size == 0 and probabilities.shape == (0, 5)
    assert torch.equal(torch.get_rng_state(), random_state)


def test_mixed_images_use_the_same_lambda_and_permutation_as_labels():
    torch = pytest.importorskip("torch")
    mix, reference = _mixer(), _mixer()
    mix.begin_epoch(1)
    reference.begin_epoch(1)
    images = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
    labels = torch.arange(10) % 5
    lam, order = reference.plan(list(range(10)), labels.tolist())
    actual, partners, weight = mix.apply(images, labels, torch.arange(10))
    assert weight == lam
    torch.testing.assert_close(actual, lam * images + (1 - lam) * images[order])
    torch.testing.assert_close(partners, labels[order])


def test_evaluation_rejects_mixup_and_plain_evaluation_stays_unchanged():
    torch = pytest.importorskip("torch")
    from fashion.train.task3_baseline import _pass

    model = torch.nn.Linear(3, 5)
    image, labels = torch.rand(10, 3), torch.arange(10) % 5
    batch = {
        "image": image,
        "label": labels,
        "id": torch.arange(10),
        "cv_fold": torch.zeros(10, dtype=torch.int64),
        "product_family_group": [f"f{i}" for i in range(10)],
        "path": ["x"] * 10,
    }
    with pytest.raises(ValueError, match="requires training"):
        _pass(model, [batch], torch.nn.CrossEntropyLoss(), torch.device("cpu"), mixup=_mixer())
    for criterion in [
        torch.nn.CrossEntropyLoss(weight=torch.ones(5)),
        torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        torch.nn.CrossEntropyLoss(reduction="sum"),
    ]:
        with pytest.raises(ValueError, match="plain unweighted"):
            _pass(
                model,
                [batch],
                criterion,
                torch.device("cpu"),
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                mixup=_mixer(),
            )
    loss, actual_labels, probabilities, trace = _pass(
        model, [batch], torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert loss == pytest.approx(torch.nn.functional.cross_entropy(model(image), labels).item())
    np.testing.assert_allclose(probabilities, model(image).softmax(1).detach().numpy())
    np.testing.assert_array_equal(actual_labels, labels.numpy())
    assert trace["id"] == list(range(10)) and not model.training


def test_spec_keeps_unweighted_parent_controls_and_rejects_other_trials(monkeypatch):
    monkeypatch.setattr(screen, "label_contract", lambda root: {"labels_sha256": "frozen"})
    spec = screen.mixup_spec()
    parent = NameTruthSpec(spec.label_contract_json)
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
    }
    assert {k: v for k, v in spec.to_dict().items() if k not in changed} == {
        k: v for k, v in parent.to_dict().items() if k not in changed
    }
    assert spec.sample_weight_strategy == "none"
    assert spec.to_dict()["mixup_policy"] == POLICY
    assert spec.parent_run_id_for_fold(4) == screen.PARENT_RUN_IDS[1]
    assert screen.mixup_config(spec, fold=0, device_name="cuda") == Task3BaselineConfig(
        target="gender"
    )
    for fold, device in [(1, "cuda"), (0, "cpu")]:
        with pytest.raises(ValueError, match="only folds"):
            screen.mixup_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="recipe changed"):
        screen.mixup_config(parent, fold=0, device_name="cuda")


def test_receipt_checks_reject_modified_evidence(tmp_path, monkeypatch):
    mix = _mixer()
    for epoch in range(1, 31):
        mix.begin_epoch(epoch)
        mix.plan(list(range(10)), [i % 5 for i in range(10)])
        mix.end_epoch()
    receipt = tmp_path / "mixup_training.json"
    receipt.write_text(json.dumps(mix.receipt()))
    history_path = tmp_path / "history.csv"
    pd.DataFrame(
        {
            "epoch": range(1, 31),
            "train_macro_f1": np.nan,
            "train_metric_scope": POLICY["online_train_f1"],
        }
    ).to_csv(history_path, index=False)
    monkeypatch.setattr(screen, "get_cv_split", lambda *a: (_training(), None))
    monkeypatch.setattr(screen, "get_samples", lambda frame, **kw: frame)
    run = {
        "config": {"mixup_contract": mix.contract},
        "metrics": {"mixup_receipt_sha256": screen.compute_sha256(receipt)},
    }
    screen.verify_mixup_evidence(run, fold=0, splits=None, directory=tmp_path)
    for fault in ("hash", "rows", "history"):
        payload = mix.receipt()
        if fault == "rows":
            payload["epochs"][0] = {**payload["epochs"][0], "rows": 9}
        receipt.write_text(json.dumps(payload))
        run["metrics"]["mixup_receipt_sha256"] = (
            "bad" if fault == "hash" else screen.compute_sha256(receipt)
        )
        if fault == "history":
            frame = pd.read_csv(history_path)
            frame["train_macro_f1"] = 0.99
            frame.to_csv(history_path, index=False)
        with pytest.raises(ValueError, match="evidence differs|coverage"):
            screen.verify_mixup_evidence(run, fold=0, splits=None, directory=tmp_path)


def test_mixup_notebook_is_one_unexecuted_two_fold_trial():
    root = Path(__file__).resolve().parents[2]
    nb = nbformat.read(root / "notebooks/04af_task3_gender_mixup_screen.ipynb", as_version=4)
    nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert code.count("run_gender_mixup_screen(") == 1
    assert "name_truth_directory=NAME_TRUTH_DIR" in code
    assert "train_test_split" not in code and "pretrained=True" not in code
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04af", "exec")
            assert cell.execution_count is None and not cell.outputs


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
            screen.check_gender_mixup_sources(name_truth_directory=tmp_path, root=tmp_path, **paths)
    else:
        checked, _, spec, _ = screen.check_gender_mixup_sources(
            name_truth_directory=tmp_path, root=tmp_path, **paths
        )
        assert set(checked["NameTruth"]) == {0, 4} and spec.name == screen.NAME


@pytest.mark.parametrize("mode", ["fresh", "reuse", "bad_mixup", "memory"])
def test_mixup_runner_uses_name_truth_parent_and_two_folds(tmp_path, monkeypatch, mode):
    child, sources, classes, _, evidence, paths, contract = _fixture(tmp_path, monkeypatch)
    spec = screen.MixUpSpec(json.dumps(contract, sort_keys=True))
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
        parent_group="NameTruth",
        candidate_group="MixUp",
        verify_candidate=verify,
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
        assert "NameTruth" in report["incremental_comparison"]["comparison"]
