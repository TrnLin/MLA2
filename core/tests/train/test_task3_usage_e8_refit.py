"""Frozen E8 translation/weight contracts and a real synthetic scratch refit."""

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from fashion.config import ROOT, TARGET_COLUMNS  # noqa: E402
from fashion.data import get_samples, load_splits  # noqa: E402
from fashion.data.hashing import compute_sha256  # noqa: E402
from fashion.train import task3_usage_e8_refit as refit  # noqa: E402
from fashion.train.registry import RunRegistry  # noqa: E402


def _refresh_source(root, monkeypatch, source):
    path = root / refit.SOURCE
    path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(refit, "SOURCE_SHA256", compute_sha256(path))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(2753)
    rows = []
    images = tmp_path / "data/raw/teacher/train/images_train"
    images.mkdir(parents=True)
    # Casual has extra rows so plain CE cannot accidentally pass as weighted CE.
    for fold in range(5):
        for label in [*refit.CLASSES, *(["Casual"] * 4)]:
            image_id = len(rows) + 1
            image = images / f"{image_id}.png"
            Image.fromarray(rng.integers(0, 220, (40, 60, 3), dtype=np.uint8)).save(image)
            row = dict(
                id=image_id,
                path=str(image.relative_to(tmp_path)),
                sha256=compute_sha256(image),
                product_family_group=f"f{image_id}",
                duplicate_group=f"d{image_id}",
                product_name_key=f"n{image_id}",
                partition="development",
                cv_fold=fold,
                is_cross_role_exact_duplicate=False,
                is_cross_role_near_duplicate=False,
                has_conflicting_target_labels=False,
                conflicting_targets="",
                quarantine_reason="",
            )
            for target in TARGET_COLUMNS:
                row[target] = label if target == "usage" else ""
                row[f"has_{target}_label"] = target == "usage"
            rows.append(row)
    for image_id, partition in [(998, "quarantine"), (999, "holdout")]:
        protected = dict(
            rows[0],
            id=image_id,
            partition=partition,
            cv_fold="",
            path=f"sealed{image_id}.png",
            sha256=f"sealed{image_id}",
            product_family_group=f"sealed{image_id}",
            duplicate_group=f"sealed{image_id}",
            product_name_key=f"sealed{image_id}",
        )
        protected.update({t: "" for t in TARGET_COLUMNS})
        protected.update({f"has_{t}_label": False for t in TARGET_COLUMNS})
        if partition == "quarantine":
            protected["quarantine_reason"] = "synthetic protection"
        rows.append(protected)
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(processed / "splits.csv", index=False)
    (processed / "label_maps.json").write_text(
        json.dumps(
            {
                "usage": {
                    "source_scope": "development",
                    "classes": refit.CLASSES,
                    "label_to_index": {n: i for i, n in enumerate(refit.CLASSES)},
                }
            }
        )
    )
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    source = json.loads((ROOT / refit.SOURCE).read_text())
    for name in source["source_contracts"]:
        source["source_contracts"][name] = compute_sha256(tmp_path / name)
    training = get_samples(
        load_splits(processed / "splits.csv"), partition="development", target="usage"
    ).reset_index(drop=True)
    source["training_rows"] = len(training)
    source["training_rows_sha256"] = hashlib.sha256(
        training[refit.ROW_COLUMNS].to_csv(index=False).encode()
    ).hexdigest()
    for fold in source["folds"]:
        counts = [
            int(training.loc[training.cv_fold.ne(fold["fold"]), "usage"].eq(n).sum())
            for n in refit.CLASSES
        ]
        fold["config"]["class_counts"] = counts
        fold["config"]["class_weights"] = refit.effective_number_class_weights(counts).tolist()
        fold["config_sha256"] = hashlib.sha256(
            (json.dumps(fold["config"], sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest()
    inference = tmp_path / source["inference_recipe"]["path"]
    inference.parent.mkdir(parents=True)
    inference.write_bytes((ROOT / source["inference_recipe"]["path"]).read_bytes())
    (tmp_path / refit.SOURCE).parent.mkdir(parents=True)
    _refresh_source(tmp_path, monkeypatch, source)
    monkeypatch.setattr(refit, "EXPECTED_ROWS", 65)
    monkeypatch.setattr(refit, "_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(refit.subprocess, "check_output", lambda *a, **kw: "fixture\n")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield tmp_path
    torch.set_num_threads(previous)


def test_saved_source_recipes_are_exact_and_weights_use_all_development(fixture):
    config, payload, training = refit.prepare_refit(root=fixture)
    assert config.epochs == 30 and config.to_dict() == payload["base_config"]
    assert len(training) == 65 and set(training.cv_fold) == set(range(5))
    assert training.partition.eq("development").all()
    assert not set(training.id) & {998, 999}
    assert not list(fixture.glob("sealed*.png"))
    assert payload["training_augmentation"] == "translation_uniform_2px_p05"
    assert payload["effective_loss_name"] == "effective_number_cross_entropy"
    contract = payload["class_weight_contract"]
    counts = np.array([25, *([5] * 8)])
    expected = (1 - 0.999) / (1 - 0.999**counts)
    expected = np.minimum(expected / expected.mean(), 5.0)
    np.testing.assert_allclose(contract["class_weights"], expected, rtol=0, atol=1e-14)
    assert contract["class_counts"] == counts.tolist()
    assert contract["beta"] == 0.999 and contract["cap"] == 5.0
    assert contract["training_rows_sha256"] == payload["training_rows_sha256"]


@pytest.mark.parametrize("change", ["snapshot", "inference", "split", "image"])
def test_changed_source_or_input_fails(fixture, change):
    source = json.loads((fixture / refit.SOURCE).read_text())
    paths = {
        "snapshot": refit.SOURCE,
        "inference": Path(source["inference_recipe"]["path"]),
        "split": Path("data/processed/splits.csv"),
        "image": Path("data/raw/teacher/train/images_train/1.png"),
    }
    with (fixture / paths[change]).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="changed|differs"):
        refit.prepare_refit(root=fixture)


@pytest.mark.parametrize(
    "field,value",
    [("training_augmentation", "none"), ("class_weight_beta", 0.9999), ("class_weight_cap", 3.0)],
)
def test_wrong_e8_recipe_fails_even_if_snapshot_is_rehashed(fixture, monkeypatch, field, value):
    source = json.loads((fixture / refit.SOURCE).read_text())
    source["folds"][0]["config"]["child_experiment"][field] = value
    _refresh_source(fixture, monkeypatch, source)
    with pytest.raises(ValueError, match="recipe"):
        refit.prepare_refit(root=fixture)


def test_real_refit_saves_translation_weights_checkpoint_and_cannot_fit_twice(fixture, monkeypatch):
    registry = fixture / "results/runs.csv"
    mirror = fixture / "mirror/runs.csv"
    old = RunRegistry(registry, mirrors=[mirror])
    old.start({"run_id": "existing_e1", "target": "usage", "experiment_id": "e1"})
    old.complete("existing_e1", {})
    before = old._read_rows()[0]
    real_pass = refit._pass

    def checked_pass(model, batches, criterion, device, **kwargs):
        assert criterion.weight is not None and torch.unique(criterion.weight).numel() > 1
        assert criterion.label_smoothing == 0 and criterion.reduction == "mean"
        return real_pass(model, batches, criterion, device, **kwargs)

    monkeypatch.setattr(refit, "_pass", checked_pass)
    real_dataset = refit.Task3ImageDataset

    def checked_dataset(*args, **kwargs):
        assert kwargs["augmentation"] == "translation_uniform_2px_p05"
        return real_dataset(*args, **kwargs)

    monkeypatch.setattr(refit, "Task3ImageDataset", checked_dataset)
    result = refit.run_usage_e8_refit(
        root=fixture, output_root=fixture / "out", registry_mirrors=[mirror]
    )
    directory = Path(result["manifest_path"]).parent
    checkpoint = torch.load(directory / "final_epoch.pt", weights_only=True)
    model = refit.Task3BaselineCNN(refit.Task3BaselineConfig(target="usage"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 391209
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 30
    assert checkpoint["normalization"]["total_pixels"] == 65 * 40 * 60
    weights = json.loads((directory / "class_weights.json").read_text())
    assert checkpoint["class_weight_contract"] == weights
    history = pd.read_csv(directory / "history.csv")
    expected_lr = 1e-5 + 0.5 * (0.001 - 1e-5) * (1 + np.cos(np.pi * np.arange(30) / 30))
    np.testing.assert_allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-12)
    assert history.selected_checkpoint.tolist() == [False] * 29 + [True]
    assert history.training_rows.eq(65).all()
    assert result["training_completed"] and not result["evaluation_completed"]
    assert not result["new_blind_evaluation"] and not list(directory.glob("*predictions*"))
    for path in (registry, mirror):
        rows = RunRegistry(path)._read_rows()
        assert len(rows) == 2 and rows[0] == before and rows[1]["status"] == "complete"
        assert rows[1]["validation_product_count"] == "0"
        assert rows[1]["validation_fold"] == ""
    saved = {p: p.read_bytes() for p in [registry, mirror, *directory.iterdir()] if p.is_file()}
    monkeypatch.setattr(refit, "_fit", lambda *a, **kw: pytest.fail("Second physical fit"))
    assert refit.run_usage_e8_refit(
        root=fixture, output_root=fixture / "out", registry_mirrors=[mirror]
    )["reused"]
    assert all(p.read_bytes() == data for p, data in saved.items())
    with pytest.raises(ValueError, match="already registered"):
        refit.run_usage_e8_refit(root=fixture, output_root=fixture / "another")
    (directory / "class_weights.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        refit.run_usage_e8_refit(root=fixture, output_root=fixture / "out")


def test_interrupted_run_stops_without_overwriting(fixture, monkeypatch):
    def stop(*args, **kwargs):
        raise KeyboardInterrupt("synthetic stop")

    monkeypatch.setattr(refit, "_fit", stop)
    with pytest.raises(KeyboardInterrupt):
        refit.run_usage_e8_refit(root=fixture, output_root=fixture / "out")
    rows = RunRegistry(fixture / "results/runs.csv")._read_rows()
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        refit.run_usage_e8_refit(root=fixture, output_root=fixture / "out")


def test_notebook_schema_and_code():
    nb = nbformat.read(ROOT / "notebooks/task3_training/usage_e8_refit.ipynb", as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "usage_e8_refit.ipynb", "exec")
