"""Freeze equal-weight SAM25 predictions before opening evaluation labels."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import json
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fashion.config import ROOT
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.images import load_and_transform_image
from fashion.train.config import Task3BaselineConfig
from fashion.train.model import Task3GeM3CNN
from fashion.train.task3_decisions import probability_columns
from fashion.train.task3_gender_diagnostic import verify_checkpoint_metadata

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/gender_sam25_cv_20260906"
REVIEW = ROOT / "reports/task3/gender_sam25_cv_result_20260906"
SPLITS = ROOT / "data/processed/splits.csv"
TEST_MANIFEST = ROOT / "data/processed/prediction_manifest.csv"
TEST_TEMPLATE = ROOT / "data/raw/teacher/test/styles_prediction.csv"
BATCH_SIZE = 128
REPRODUCTION_TOLERANCE = 1e-4


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(value, path):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text())


class UnlabelledImages(Dataset):
    def __init__(self, frame, normalization):
        self.paths = [resolve_task3_path(name, root=ROOT) for name in frame.path]
        self.normalization = normalization

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = load_and_transform_image(
            self.paths[index],
            image_size=(80, 60),
            mean=self.normalization["mean"],
            std=self.normalization["std"],
        )
        return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def model_for(fold, manifest):
    entry = manifest["folds"][fold]
    assert entry["fold"] == fold
    directory = EVIDENCE / entry["run_id"]
    for filename, record in entry["files"].items():
        assert record["path"] == f"{entry['run_id']}/{filename}"
        assert compute_sha256(EVIDENCE / record["path"]) == record["sha256"]
    config = read(directory / "config.json")
    normalization = read(directory / "normalization.json")
    checkpoint = torch.load(directory / "final_epoch.pt", map_location="cpu", weights_only=True)
    verify_checkpoint_metadata(
        checkpoint,
        run_id=entry["run_id"],
        config=config,
        classes=manifest["class_names"],
        normalization=normalization,
    )
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 25
    assert checkpoint["checkpoint_policy"] == config["checkpoint_rule"] == "final_epoch"
    assert config["scratch"] and config["submission_eligible"]
    assert config["target"] == "gender" and config["input_view"] == "full"
    assert config["effective_model_family"] == "task3_small_cnn_gem_p3"
    assert config["child_experiment"] == manifest["recipe"]
    assert config["image_height"] == 80 and config["image_width"] == 60
    kwargs = {field.name: config[field.name] for field in fields(Task3BaselineConfig)}
    kwargs["channels"] = tuple(kwargs["channels"])
    model = Task3GeM3CNN(Task3BaselineConfig(**kwargs))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 390181
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    model.eval().requires_grad_(False)
    return model, normalization


def predict(model, frame, normalization, *, label):
    loader = DataLoader(
        UnlabelledImages(frame, normalization),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, images in enumerate(loader, 1):
            outputs.append(torch.softmax(model(images), dim=1).numpy().astype(np.float64))
            if batch_number % 20 == 0:
                print(
                    f"{label}: {min(batch_number * BATCH_SIZE, len(frame)):,}/{len(frame):,} images",
                    flush=True,
                )
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    values = np.concatenate(outputs)
    assert values.shape == (len(frame), 5) and np.isfinite(values).all()
    assert np.all(values >= 0) and np.all(values <= 1)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    return values, time.perf_counter() - started


def prediction_frame(frame, values, classes):
    result = frame[["id"]].reset_index(drop=True).copy()
    result["gender"] = np.asarray(classes)[values.argmax(axis=1)]
    result["confidence"] = values.max(axis=1)
    result[probability_columns(classes)] = values
    return result


def main():
    if (REPORT / "prediction_freeze.json").exists():
        raise RuntimeError("These predictions are already frozen; preserve the saved evaluation.")
    if (REPORT / "inference_recipe.json").exists():
        raise RuntimeError("An evaluation was already started; inspect it before retrying.")
    torch.set_num_threads(2)
    torch.manual_seed(2753)
    torch.use_deterministic_algorithms(True)
    torch.backends.fp32_precision = "ieee"
    manifest = read(EVIDENCE / "model_manifest.json")
    assert manifest == read(REVIEW / "model_manifest.json")
    classes = manifest["class_names"]
    assert classes == ["Boys", "Girls", "Men", "Unisex", "Women"]
    assert [f["fold"] for f in manifest["folds"]] == list(range(5))
    assert manifest["checkpoint_epoch"] == 25
    for name in (
        "src/fashion/data/images.py",
        "src/fashion/train/model.py",
        "src/fashion/train/config.py",
    ):
        expected = read(REVIEW / "ieee0/evaluation_manifest.json")["identity"]["dependencies"][name]
        assert compute_sha256(resolve_task3_path(name, root=ROOT)) == expected, name

    splits = load_splits(SPLITS)
    assert (
        compute_sha256(SPLITS)
        == manifest["recipe"]["gender_label_variant"]["canonical_split_sha256"]
    )
    development = splits.loc[splits.partition.eq("development")]
    protected = splits.loc[splits.partition.eq("holdout")]
    assert protected.gender.eq("").all()
    holdout = (
        protected[["id", "path", "sha256", "product_family_group"]]
        .sort_values("id")
        .reset_index(drop=True)
    )
    test = pd.read_csv(TEST_MANIFEST, keep_default_na=False)
    template = pd.read_csv(TEST_TEMPLATE, usecols=["id"], keep_default_na=False)
    assert template.id.is_unique and test.id.is_unique and set(template.id) == set(test.id)
    test = test.set_index("id").loc[template.id].reset_index()[["id", "path", "sha256"]]
    assert len(holdout) == 5778 and len(test) == 5829
    assert holdout.id.is_unique and test.id.is_unique
    for a, b in ((holdout, development), (test, development), (test, holdout)):
        assert not set(a.id).intersection(b.id)
        assert not set(a.sha256).intersection(b.sha256)
    assert not set(holdout.product_family_group).intersection(development.product_family_group)
    for frame in (holdout, test):
        for row in frame.itertuples():
            assert compute_sha256(resolve_task3_path(row.path, root=ROOT)) == row.sha256, row.id
    holdout.to_csv(REPORT / "holdout_image_manifest.csv", index=False)
    test.to_csv(REPORT / "test_image_manifest.csv", index=False)
    print(
        "Verified 5,778 holdout images and 5,829 test images; exact files and IDs are disjoint from training.",
        flush=True,
    )

    recipe = {
        "declared_at_utc": now(),
        "user_authorization": "Evaluate the fixed model on reserved holdout and teacher test; use high-resolution dataset labels for test scoring.",
        "target": "gender",
        "model_manifest": str((EVIDENCE / "model_manifest.json").relative_to(ROOT)),
        "model_manifest_sha256": compute_sha256(EVIDENCE / "model_manifest.json"),
        "folds": manifest["folds"],
        "checkpoint_epoch": 25,
        "classes": classes,
        "aggregation": "Equal arithmetic mean of all five saved fold softmax vectors, then argmax",
        "fold_weights": [0.2] * 5,
        "preprocessing": "Full teacher RGB images; 80x60 letterbox; each fold's saved normalization; no augmentation",
        "holdout_rows": len(holdout),
        "test_rows": len(test),
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "batch_size": BATCH_SIZE,
        "dtype": "float32",
        "autocast": False,
        "backend_precision": torch.backends.fp32_precision,
        "training_or_tuning": False,
        "evaluation_labels_used_for_inference": False,
        "primary_label_plan": {
            "holdout": "Original gender labels in teacher styles_train.csv, joined on the canonical holdout IDs",
            "test": "Original gender labels in high-resolution fashion-dataset/styles.csv, joined by product ID; cross-check per-product JSON",
        },
        "secondary_label_plan": "Audit high-resolution holdout labels and the already-frozen single explicit product-name cue rule; never use model predictions to choose labels.",
        "reproduction_plan": "For each fold: first 32 validation IDs per true class; compare CPU probabilities with saved IEEE validation probabilities before held-out inference.",
        "reproduction_max_abs_tolerance": REPRODUCTION_TOLERANCE,
        "input_sha256": {
            str(path.relative_to(ROOT)): compute_sha256(path)
            for path in (
                Path(__file__),
                SPLITS,
                TEST_MANIFEST,
                TEST_TEMPLATE,
                ROOT / "src/fashion/data/images.py",
                ROOT / "src/fashion/train/model.py",
                ROOT / "src/fashion/train/config.py",
                ROOT / "src/fashion/train/metrics.py",
                ROOT / "src/fashion/data/gender_name_truth.py",
                REVIEW / "aggregate/oof_predictions.csv",
                REPORT / "holdout_image_manifest.csv",
                REPORT / "test_image_manifest.csv",
            )
        },
    }
    save_json(recipe, REPORT / "inference_recipe.json")
    oof = pd.read_csv(
        REVIEW / "aggregate/oof_predictions.csv",
        keep_default_na=False,
        float_precision="round_trip",
    )
    checks = []
    for fold in range(5):
        model, normalization = model_for(fold, manifest)
        sample = (
            oof.loc[oof.cv_fold.eq(fold)]
            .sort_values("id")
            .groupby("true_label", sort=False)
            .head(32)
            .sort_values("id")
        )
        assert len(sample) == 160
        values, seconds = predict(model, sample, normalization, label=f"CPU check fold {fold}")
        reference = sample[probability_columns(classes)].to_numpy(dtype=float)
        difference = np.abs(values - reference)
        maximum = float(difference.max())
        flips = int(np.count_nonzero(values.argmax(1) != reference.argmax(1)))
        assert maximum <= REPRODUCTION_TOLERANCE, (fold, maximum)
        checks.append(
            {
                "fold": fold,
                "rows": len(sample),
                "sample_ids": sample.id.tolist(),
                "max_abs_probability_difference": maximum,
                "prediction_flips": flips,
                "seconds": seconds,
            }
        )
        print(
            f"CPU check fold {fold}: max probability difference {maximum:.3g}, {flips} changed classes",
            flush=True,
        )
        del model
    save_json(checks, REPORT / "cpu_reproduction_checks.json")

    outputs = [
        REPORT / "inference_recipe.json",
        REPORT / "cpu_reproduction_checks.json",
        REPORT / "holdout_image_manifest.csv",
        REPORT / "test_image_manifest.csv",
    ]
    probability_sets = {"holdout": [], "test": []}
    timings = []
    for fold in range(5):
        model, normalization = model_for(fold, manifest)
        for scope, frame in (("holdout", holdout), ("test", test)):
            values, seconds = predict(model, frame, normalization, label=f"{scope} fold {fold}")
            path = REPORT / f"gender_{scope}_fold_{fold}.csv"
            prediction_frame(frame, values, classes).to_csv(path, index=False)
            outputs.append(path)
            probability_sets[scope].append(values)
            timings.append({"fold": fold, "scope": scope, "rows": len(frame), "seconds": seconds})
            print(f"{scope} fold {fold}: complete in {seconds:.1f}s", flush=True)
        del model
    for scope, frame in (("holdout", holdout), ("test", test)):
        averaged = np.mean(np.stack(probability_sets[scope]), axis=0)
        result = prediction_frame(frame, averaged, classes)
        probability_path = REPORT / f"gender_{scope}_probabilities.csv"
        label_path = REPORT / f"gender_{scope}_predictions.csv"
        result.to_csv(probability_path, index=False)
        result[["id", "gender"]].to_csv(label_path, index=False)
        outputs.extend((probability_path, label_path))
    save_json(timings, REPORT / "inference_timing.json")
    outputs.append(REPORT / "inference_timing.json")
    freeze = {
        "frozen_at_utc": now(),
        "holdout_rows": len(holdout),
        "test_rows": len(test),
        "holdout_and_test_reference_labels_opened": False,
        "weights_and_buffers_unchanged": True,
        "outputs": {path.name: compute_sha256(path) for path in outputs},
    }
    save_json(freeze, REPORT / "prediction_freeze.json")
    print(
        "All holdout and test predictions are frozen. Ready to open labels for scoring.", flush=True
    )


if __name__ == "__main__":
    main()
