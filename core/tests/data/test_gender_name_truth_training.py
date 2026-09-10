"""The training/evaluation contract must bind corrected labels, never split membership."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from test_gender_name_truth import project as project

from fashion.data import get_cv_split, load_splits
from fashion.data.gender_name_truth import (
    VARIANT_RELATIVE_PATH,
    build_gender_name_truth_variant,
    load_gender_name_truth_variant,
)
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_gender_dropout_darkening import DARKENING_RUN_IDS, GRAYSCALE_NAME
from fashion.train.task3_gender_ieee import evaluation_identity, verify_evaluation_labels
from fashion.train.task3_gender_name_truth import (
    NAME,
    PARENT_RUN_IDS,
    NameTruthSpec,
    label_contract,
    name_truth_config,
    name_truth_spec,
    require_name_truth_prerequisites,
    rescore_original_labels,
    training_splits,
    write_original_label_diagnostic,
)


def test_frozen_controls_and_old_digest_stay_unchanged(project):
    build_gender_name_truth_variant(project)
    spec = name_truth_spec(project)
    old = dataset_v2_spec(GRAYSCALE_NAME, DARKENING_RUN_IDS)
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
        "gender_label_variant",
        "independent_blind_test",
    }
    assert {k: v for k, v in old.to_dict().items() if k not in metadata} == {
        k: v for k, v in spec.to_dict().items() if k not in metadata
    }
    assert spec.name == NAME and spec.classifier_dropout == 0.30
    assert spec.to_dict()["grayscale_probability"] == 0.10
    assert spec.parent_run_id_for_fold(4) == PARENT_RUN_IDS[1]
    config = name_truth_config(spec, fold=0, device_name="cuda", root=project)
    assert config == Task3BaselineConfig(target="gender")

    def digest(child):
        return hashlib.sha256(
            json.dumps(
                {
                    "baseline_controls": config.to_dict(),
                    "child_experiment": child.to_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]

    assert digest(old) == "eb37119b7e68"
    assert digest(spec) != digest(old)
    for fold, device in [(1, "cuda"), (0, "cpu")]:
        with pytest.raises(ValueError, match="only folds"):
            name_truth_config(spec, fold=fold, device_name=device, root=project)
    with pytest.raises(ValueError, match="audit is required"):
        require_name_truth_prerequisites(None, spec=spec, fold=0, root=project)


def test_training_and_validation_use_new_labels_on_the_original_ids(project):
    build_gender_name_truth_variant(project)
    spec = name_truth_spec(project)
    original = load_splits(project / "data/processed/splits.csv")
    actual = training_splits(spec, root=project)
    expected = load_gender_name_truth_variant(project)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.gender.ne(original.gender).sum() == 4
    pd.testing.assert_frame_equal(actual.drop(columns="gender"), original.drop(columns="gender"))
    for fold in (0, 4):
        for before, after in zip(
            get_cv_split(original, fold), get_cv_split(actual, fold), strict=True
        ):
            assert before.id.tolist() == after.id.tolist()
    bad = dict(label_contract(project), labels_sha256="different")
    with pytest.raises(ValueError, match="frozen training"):
        training_splits(NameTruthSpec(json.dumps(bad)), root=project)
    with (project / VARIANT_RELATIVE_PATH / "labels.csv").open("a") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        training_splits(spec, root=project)


def test_evaluation_rejects_missing_contract_old_labels_and_changed_hash(project, monkeypatch):
    build_gender_name_truth_variant(project)
    corrected = load_gender_name_truth_variant(project)
    contract = label_contract(project)
    verify_evaluation_labels(corrected, label_variant=contract, root=project)
    with pytest.raises(ValueError, match="explicit label variant"):
        verify_evaluation_labels(corrected, label_variant=None, root=project)
    with pytest.raises(ValueError, match="differs from the name-truth"):
        verify_evaluation_labels(
            load_splits(project / "data/processed/splits.csv"),
            label_variant=contract,
            root=project,
        )
    with pytest.raises(ValueError, match="differs from the verified"):
        verify_evaluation_labels(corrected, label_variant={}, root=project)
    monkeypatch.setattr("fashion.train.task3_gender_ieee.compute_sha256", lambda path: str(path))
    run = {"run_id": "run", "sha256": {"checkpoint": "hash"}}
    old = evaluation_identity(run, root=project)
    new = evaluation_identity(run, root=project, label_variant=contract)
    assert "label_variant" not in old and new["label_variant"] == contract
    assert old != new


def test_actual_model_and_trainer_require_label_audit_before_training(project):
    pytest.importorskip("torch")
    from fashion.train.task3_baseline import _build_task3_model, run_task3_baseline_fold

    build_gender_name_truth_variant(project)
    spec = name_truth_spec(project)
    model = _build_task3_model(Task3BaselineConfig(target="gender"), spec)
    assert sum(p.numel() for p in model.parameters()) == 390181
    assert model.classifier_dropout.p == 0.30
    with pytest.raises(ValueError, match="audit is required"):
        run_task3_baseline_fold("gender", 0, output_root=project, child_spec=spec, root=project)


def test_original_label_diagnostic_reuses_exact_probabilities_and_checks_hashes(project):
    from fashion.data.hashing import compute_sha256
    from fashion.train.task3_clean_slate import _prediction_frame
    from fashion.train.task3_decisions import CORE_CORRUPTIONS, oof_metrics

    build_gender_name_truth_variant(project)
    splits = load_gender_name_truth_variant(project)
    original = load_splits(project / "data/processed/splits.csv")
    classes = ["Boys", "Girls", "Men", "Unisex", "Women"]
    directory = project / "evaluation"
    directory.mkdir()
    training, validation = get_cv_split(splits, 0)

    def predictions(frame):
        probabilities = np.full((len(frame), 5), 0.025)
        truth = frame.gender.map(dict(zip(classes, range(5)))).to_numpy(dtype=int)
        probabilities[np.arange(len(frame)), truth] = 0.9
        return _prediction_frame(
            frame,
            target="gender",
            classes=classes,
            probabilities=probabilities,
            run_id="saved",
        )

    p = predictions(validation)
    teacher = rescore_original_labels(
        p,
        corrected=validation,
        original=get_cv_split(original, 0)[1],
        classes=classes,
        run_id="saved",
    )
    assert oof_metrics(p, classes)["macro_f1"] > oof_metrics(teacher, classes)["macro_f1"]
    pd.testing.assert_frame_equal(
        p.drop(columns=["true_label", "true_index"]),
        teacher.drop(columns=["true_label", "true_index"]),
        check_dtype=False,
    )
    files = {}
    for name, frame in (
        ("clean_train_predictions.csv", training),
        ("oof_predictions.csv", validation),
        *((f"{name}_predictions.csv", validation) for name in CORE_CORRUPTIONS),
    ):
        path = directory / name
        predictions(frame).to_csv(path, index=False)
        files[name] = compute_sha256(path)
    manifest = {"files": files, "identity": {"label_variant": label_contract(project)}}
    (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))
    group = {
        "model": {
            0: {
                "run_id": "saved",
                "evaluation_directory": str(directory),
                "evaluation_manifest": manifest,
            }
        }
    }
    write_original_label_diagnostic(
        group,
        splits=splits,
        classes=classes,
        destination=directory,
        root=project,
    )
    report = json.loads((directory / "original_label_diagnostic.json").read_text())
    assert report["probabilities_and_predictions_unchanged"]
    table = pd.read_csv(directory / "label_basis_comparison.csv")
    assert len(table) == 2 + len(CORE_CORRUPTIONS)
    assert set(table.view) == {"clean_train", "clean_validation", *CORE_CORRUPTIONS}
    with (directory / "oof_predictions.csv").open("a") as handle:
        handle.write("tamper")
    with pytest.raises(ValueError, match="probabilities changed"):
        write_original_label_diagnostic(
            group,
            splits=splits,
            classes=classes,
            destination=directory,
            root=project,
        )
