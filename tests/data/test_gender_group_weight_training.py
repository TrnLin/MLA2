"""The new weighting path uses the same labels/IDs and leaves evaluation untouched."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from test_gender_name_truth import project as project

from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import build_gender_name_truth_variant
from fashion.train.task3_gender_group_weight import (
    PARENT_RUN_IDS,
    WEIGHT_COLUMN,
    GroupWeightSpec,
    add_gender_article_weights,
    group_weight_spec,
    preview_gender_article_weights,
    require_group_weight_prerequisites,
    training_splits,
)
from fashion.train.task3_gender_name_truth import name_truth_spec
from fashion.train.task3_gender_name_truth import training_splits as parent_splits


def test_weights_do_not_change_labels_or_any_fold_membership(project):
    build_gender_name_truth_variant(project)
    spec = group_weight_spec(project)
    splits = training_splits(spec, root=project)
    pd.testing.assert_frame_equal(splits, parent_splits(name_truth_spec(project), root=project))
    before = splits.copy(deep=True)
    for fold in (0, 4):
        training = get_samples(get_cv_split(splits, fold)[0], target="gender")
        weighted, _ = add_gender_article_weights(training, validation_fold=fold)
        pd.testing.assert_frame_equal(weighted[training.columns], training)
    pd.testing.assert_frame_equal(splits, before)
    assert WEIGHT_COLUMN not in splits
    preview = preview_gender_article_weights(splits)
    assert set(preview.validation_fold) == {0, 4}
    assert np.allclose(
        preview.groupby("validation_fold").weighted_rows.sum(),
        preview.groupby("validation_fold").training_rows.sum(),
    )
    changed = splits.copy()
    # Change validation product types only. The fold-0 training weights cannot change.
    changed.loc[changed.cv_fold.eq(0), "articleType"] = "Different product"
    changed_preview = preview_gender_article_weights(changed)
    pd.testing.assert_frame_equal(
        preview.query("validation_fold == 0").reset_index(drop=True),
        changed_preview.query("validation_fold == 0").reset_index(drop=True),
    )
    with pytest.raises(ValueError, match="audit is required"):
        require_group_weight_prerequisites(None, spec=spec, fold=0, root=project)
    with pytest.raises(ValueError, match="frozen label"):
        training_splits(GroupWeightSpec(json.dumps({"labels_sha256": "changed"})), root=project)


def test_trainer_requires_source_audit_and_preserves_architecture(project):
    pytest.importorskip("torch")
    from fashion.train.config import Task3BaselineConfig
    from fashion.train.task3_baseline import _build_task3_model, run_task3_baseline_fold

    build_gender_name_truth_variant(project)
    spec = group_weight_spec(project)
    model = _build_task3_model(Task3BaselineConfig(target="gender"), spec)
    assert sum(p.numel() for p in model.parameters()) == 390_181
    assert model.classifier_dropout.p == 0.30
    with pytest.raises(ValueError, match="audit is required"):
        run_task3_baseline_fold("gender", 0, output_root=project, child_spec=spec, root=project)


def test_weighted_training_dataset_does_not_weight_clean_evaluation(project):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from fashion.train.data import Task3ImageDataset
    from fashion.train.loss import SampleWeightedCrossEntropy

    build_gender_name_truth_variant(project)
    splits = training_splits(group_weight_spec(project), root=project)
    training = get_samples(get_cv_split(splits, 0)[0], target="gender")
    weighted, _ = add_gender_article_weights(training, validation_fold=0)
    for relative in weighted.path:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 80), (100, 150, 200)).save(path)
    classes = ["Boys", "Girls", "Men", "Unisex", "Women"]
    kwargs = dict(
        target="gender",
        label_to_index=dict(zip(classes, range(5))),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        root=project,
    )
    train = Task3ImageDataset(weighted, sample_weight_column=WEIGHT_COLUMN, **kwargs)
    clean = Task3ImageDataset(weighted, **kwargs)
    assert "sample_weight" in train[0] and "sample_weight" not in clean[0]
    for index in range(len(train)):
        assert torch.equal(train[index]["image"], clean[index]["image"])
    weights = torch.tensor(weighted[WEIGHT_COLUMN].to_numpy(), dtype=torch.float32)
    labels = torch.tensor(weighted.gender.map(kwargs["label_to_index"]).to_numpy())
    logits = torch.arange(len(labels) * 5, dtype=torch.float32).reshape(-1, 5) / 7
    per_row = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    actual = SampleWeightedCrossEntropy(torch.ones(5))(logits, labels, weights)
    assert actual.item() == pytest.approx(((per_row * weights).sum() / weights.sum()).item())


def test_notebook_has_one_unexecuted_weighting_trial():
    root = Path(__file__).resolve().parents[2]
    nb = nbformat.read(root / "notebooks/04ae_task3_gender_group_weight_screen.ipynb", as_version=4)
    nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert code.count("run_gender_group_weight_screen(") == 1
    assert "name_truth_directory=NAME_TRUTH_DIR" in code
    assert "preview_gender_article_weights" in code
    assert "train_test_split" not in code and "pretrained=True" not in code
    assert len(PARENT_RUN_IDS) == 2
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04ae", "exec")
            assert cell.execution_count is None and not cell.outputs


def test_prerequisites_bind_exact_weight_fit_code_labels_and_parent(project, monkeypatch):
    import fashion.train.task3_gender_group_weight as screen

    build_gender_name_truth_variant(project)
    spec = group_weight_spec(project)
    implementation = project / "src/fashion/train/task3_gender_group_weight.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("frozen implementation")
    sources = {
        "NameTruth": {
            fold: {
                "run_id": spec.parent_run_id_for_fold(fold),
                "sha256": {"config.json": f"verified-{fold}"},
                "directory": str(project / f"parent-{fold}"),
            }
            for fold in (0, 4)
        }
    }
    evidence = {"artifact_sha256": {"precision": "verified"}}
    paths = {"precision_directory": str(project / "precision")}
    identity = screen._source_identity(sources, spec, evidence, paths, root=project)
    audit = project / "source_audit.json"
    audit.write_text(json.dumps({"identity": identity}))
    monkeypatch.setattr(
        screen, "check_gender_group_weight_sources", lambda **kw: (sources, [], spec, evidence)
    )
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    verified = require_group_weight_prerequisites(audit, spec=spec, fold=0, root=project)
    assert verified["parent_directory"] == project / "parent-0"
    with pytest.raises(ValueError, match="verified name-truth parent"):
        require_group_weight_prerequisites(
            audit, spec=spec, fold=0, root=project, parent_run_directory=project / "parent-4"
        )
    tampered = json.loads(audit.read_text())
    tampered["identity"]["training_weight_contracts"]["0"]["maximum"] = 999
    audit.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="evidence changed"):
        require_group_weight_prerequisites(audit, spec=spec, fold=0, root=project)
    audit.write_text(json.dumps({"identity": identity}))
    implementation.write_text("changed implementation")
    with pytest.raises(ValueError, match="evidence changed"):
        require_group_weight_prerequisites(audit, spec=spec, fold=0, root=project)
