"""Exercise the real five-fold MixUp runner on small synthetic images."""

import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest
import torch

from fashion.config import ROOT
from fashion.data.gender_name_truth import build_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train import task3_gender_mixup_cv as cv


@pytest.fixture
def cv_fixture(tmp_path, monkeypatch):
    from test_task3_gender_sam import synthetic_gender

    synthetic_gender.__wrapped__(tmp_path)
    path = tmp_path / "data/processed/splits.csv"
    rows = pd.read_csv(path, keep_default_na=False)
    rows["productDisplayName"] = rows.gender + " item"
    rows.to_csv(path, index=False)
    summary = build_gender_name_truth_variant(tmp_path)
    source = json.loads((ROOT / cv.SOURCE_CONFIG).read_text())
    source["num_workers"] = 0
    source["gender_label_variant"].update(
        canonical_split_sha256=compute_sha256(path),
        labels_sha256=summary["files"]["labels.csv"],
    )
    target = tmp_path / cv.SOURCE_CONFIG
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(source))
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    monkeypatch.setattr(cv, "SOURCE_SHA256", compute_sha256(target))
    monkeypatch.setattr(cv, "EXPECTED_ROWS", 25)
    monkeypatch.setattr(cv, "_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(cv.subprocess, "check_output", lambda *a, **kw: "test-commit\n")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield tmp_path
    torch.set_num_threads(previous)


def test_source_is_exact_original_mixup_recipe():
    assert compute_sha256(ROOT / cv.SOURCE_CONFIG) == cv.SOURCE_SHA256
    source = json.loads((ROOT / cv.SOURCE_CONFIG).read_text())
    assert source["epochs"] == 30
    assert source["child_experiment"]["mixup_policy"]["alpha"] == 0.2
    assert "sam_policy" not in source["child_experiment"]


def test_preflight_preserves_folds_and_rejects_changed_images(cv_fixture):
    config, payload, splits = cv.prepare_cv(root=cv_fixture)
    assert config.epochs == payload["cosine_t_max"] == 30
    assert payload["classifier_dropout"] == 0.3
    assert payload["source_role"] == "recipe_only_no_checkpoint_loaded"
    assert payload["mixup_policy"]["alpha"] == 0.2
    assert not (cv_fixture / "sealed.png").exists()
    for fold in cv.FOLDS:
        training, validation = cv.task3_target_frames(splits, target="gender", validation_fold=fold)
        assert len(training) == 20 and len(validation) == 5
        assert not set(training.id) & set(validation.id)
        assert set(training.cv_fold) == set(cv.FOLDS) - {fold}
    (cv_fixture / training.iloc[0].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input image"):
        cv.prepare_cv(root=cv_fixture)


def test_colab_runtime_required_before_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Colab L4"):
        cv.run_gender_mixup_cv(output_root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_real_five_folds_30_epochs_registry_predictions_and_restart(cv_fixture, monkeypatch):
    from fashion.train.sam import SAMStep

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not load weights or use SAM")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(SAMStep, "step", forbidden)
    optimizer_steps = []
    original_step = torch.optim.AdamW.step

    def count_step(optimizer, *args, **kwargs):
        optimizer_steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", count_step)
    root = cv_fixture
    mirror = root / "mirror/runs.csv"
    # Real model, gradients, MixUp, images, statistics and scoring: 150 tiny epochs.
    result = cv.run_gender_mixup_cv(root=root, output_root=root / "out", registry_mirrors=(mirror,))
    assert len(optimizer_steps) == 5 * 30  # One batch and one AdamW update per tiny epoch.
    assert result["folds"] == [0, 1, 2, 3, 4] and result["oof_rows"] == 25
    output = Path(result["summary_path"]).parent
    rows = pd.read_csv(root / "results/runs.csv", keep_default_na=False)
    assert len(rows) == 5 and rows.status.eq("complete").all()
    assert rows.training_product_count.eq(20).all()
    assert rows.validation_product_count.eq(5).all()
    pd.testing.assert_frame_equal(rows, pd.read_csv(mirror, keep_default_na=False))
    all_predictions = pd.read_csv(output / "oof_predictions.csv")
    assert all_predictions.id.nunique() == 25 and 999 not in set(all_predictions.id)
    assert len(pd.read_csv(output / "robustness.csv")) == 5
    for model in result["models"]:
        directory = output / model["directory"]
        history = pd.read_csv(directory / "history.csv")
        assert history.epoch.tolist() == list(range(1, 31))
        assert history.train_macro_f1.isna().all()
        assert history.selected_checkpoint.tolist() == [False] * 29 + [True]
        receipt = json.loads((directory / "mixup_training.json").read_text())
        assert len(receipt["epochs"]) == 30
        assert all(epoch["rows"] == 20 for epoch in receipt["epochs"])
        assert not (directory / "sam_training.json").exists()
        norm = json.loads((directory / "normalization.json").read_text())
        assert norm["validation_fold"] == model["validation_fold"]
        assert norm["fit_scope"] == "fold_training_content_pixels_only"
        scores = json.loads((directory / "metrics.json").read_text())
        assert scores["selected_epoch"] == 30 and not scores["holdout_evaluated"]
        assert scores["final_train_validation_macro_f1_gap"] == pytest.approx(
            scores["final_train_eval_macro_f1"] - scores["macro_f1"]
        )
    # Completed folds are read and verified, never retrained.
    monkeypatch.setattr(cv, "_fit", forbidden)
    assert (
        cv.run_gender_mixup_cv(root=root, output_root=root / "out")["run_ids_by_fold"]
        == result["run_ids_by_fold"]
    )
    assert len(pd.read_csv(root / "results/runs.csv")) == 5
    directory = output / result["models"][0]["directory"]
    (directory / "history.csv").write_text("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        cv.run_gender_mixup_cv(root=root, output_root=root / "out")


def test_failed_fold_is_recorded_and_restarts_fresh(cv_fixture, monkeypatch):
    root = cv_fixture
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(kwargs["run_id"])
        raise KeyboardInterrupt("test interruption")

    monkeypatch.setattr(cv, "_fit", interrupt)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            cv.run_gender_mixup_cv(root=root, output_root=root / "out")
    assert len(set(calls)) == 2
    rows = pd.read_csv(root / "results/runs.csv")
    assert len(rows) == 2 and rows.status.eq("failed").all()


def test_colab_notebook_is_valid_and_uses_shared_drive_contract():
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/gender_mixup_five_fold.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "google.colab import drive" in source
    assert "data/task3-data.zip" in source
    assert "registry_mirrors=(LOCAL_REGISTRY,)" in source
    assert "run_gender_mixup_cv(" in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "colab-cell", "exec")
            assert all(output.output_type != "error" for output in cell.outputs)
            if cell.outputs:
                assert isinstance(cell.execution_count, int) and cell.execution_count > 0
