"""Predict the teacher test images without opening their reference labels."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.data.images import load_and_transform_image
from fashion.train.config import Task3BaselineConfig
from fashion.train.model import Task3BaselineCNN

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/usage_expanded_e8_20260906"
MANIFEST = ROOT / "data/processed/prediction_manifest.csv"
TEMPLATE = ROOT / "data/raw/teacher/test/styles_prediction.csv"
SPLITS = ROOT / "data/processed/teacher_plus_rare_usage_20260906/splits.csv"
CLASSES = (
    "Casual", "Ethnic", "Formal", "Home", "NA", "Party", "Smart Casual", "Sports", "Travel"
)
PROBABILITY_COLUMNS = [f"probability_{name}" for name in CLASSES]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class UnlabelledTestImages(Dataset):
    def __init__(self, manifest, normalization):
        self.paths = [resolve_task3_path(value, root=ROOT) for value in manifest.path]
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


def prediction_frame(ids, probabilities):
    assert probabilities.shape == (len(ids), len(CLASSES))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    indices = probabilities.argmax(axis=1)
    result = pd.DataFrame({"id": ids, "usage": np.asarray(CLASSES)[indices]})
    result["confidence"] = probabilities.max(axis=1)
    result[PROBABILITY_COLUMNS] = probabilities
    return result


def main():
    torch.set_num_threads(2)
    if (REPORT / "prediction_freeze.json").exists():
        raise RuntimeError("Predictions are already frozen; keep the original evaluation.")

    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    template = pd.read_csv(TEMPLATE, usecols=["id"], keep_default_na=False)
    split_ids = pd.read_csv(SPLITS, usecols=["id"], keep_default_na=False)
    assert template.id.is_unique and manifest.id.is_unique
    assert set(template.id) == set(manifest.id)
    assert not set(template.id).intersection(split_ids.id)
    manifest = manifest.set_index("id").loc[template.id].reset_index()
    assert manifest.id.tolist() == template.id.tolist()
    for row in manifest.itertuples():
        assert compute_sha256(resolve_task3_path(row.path, root=ROOT)) == row.sha256, row.id
    print(f"Verified all {len(manifest):,} test images and their hashes.", flush=True)

    runs = pd.read_csv(EVIDENCE / "results/runs.csv", keep_default_na=False)
    runs = runs.sort_values("validation_fold")
    assert runs.validation_fold.tolist() == list(range(5))
    checkpoints = []
    for row in runs.itertuples():
        path = EVIDENCE / row.run_id / "final_epoch.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["run_id"] == row.run_id
        assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 30
        assert checkpoint["checkpoint_policy"] == "final_epoch"
        assert checkpoint["class_names"] == list(CLASSES)
        config = checkpoint["config"]
        assert config["target"] == "usage" and config["scratch"]
        assert config["effective_model_family"] == "task3_small_cnn"
        assert config["input_view"] == "full"
        assert config["expanded_dataset"]["split_sha256"] == compute_sha256(SPLITS)
        checkpoints.append((int(row.validation_fold), path, checkpoint))

    recipe = {
        "declared_at_utc": utc_now(),
        "target": "usage",
        "test_rows": len(manifest),
        "class_names": list(CLASSES),
        "aggregation": "Equal arithmetic mean of five softmax probability vectors; argmax",
        "fold_weights": [0.2] * 5,
        "preprocessing": "Saved fold normalization; full RGB; 80x60; no augmentation",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "checkpoint_rule": "Saved final epoch 30 for all five folds",
        "labels_used_for_inference": False,
        "training_or_tuning": False,
        "test_and_training_split_ids_disjoint": True,
        "test_image_hashes_verified": len(manifest),
        "inputs": {
            str(path.relative_to(ROOT)): compute_sha256(path)
            for path in (MANIFEST, TEMPLATE, SPLITS, Path(__file__),
                         ROOT / "src/fashion/data/images.py",
                         ROOT / "src/fashion/train/model.py",
                         ROOT / "src/fashion/train/config.py")
        },
        "checkpoints": [
            {
                "fold": fold,
                "path": str(path.relative_to(ROOT)),
                "sha256": compute_sha256(path),
                "run_id": checkpoint["run_id"],
                "normalization": checkpoint["normalization"],
            }
            for fold, path, checkpoint in checkpoints
        ],
    }
    save_json(recipe, REPORT / "inference_recipe.json")
    probabilities_by_fold = []
    output_paths = []
    for fold, _, checkpoint in checkpoints:
        config_kwargs = {
            field.name: checkpoint["config"][field.name]
            for field in fields(Task3BaselineConfig)
        }
        config_kwargs["channels"] = tuple(config_kwargs["channels"])
        model = Task3BaselineCNN(Task3BaselineConfig(**config_kwargs))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        dataset = UnlabelledTestImages(manifest, checkpoint["normalization"])
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        batches = []
        with torch.inference_mode():
            for batch_number, images in enumerate(loader, start=1):
                batches.append(torch.softmax(model(images), dim=1).numpy())
                if batch_number % 50 == 0:
                    print(f"Fold {fold}: {min(batch_number * 32, len(dataset)):,} images", flush=True)
        probabilities = np.concatenate(batches)
        path = REPORT / f"usage_test_fold_{fold}.csv"
        prediction_frame(manifest.id, probabilities).to_csv(path, index=False)
        probabilities_by_fold.append(probabilities.astype(np.float64))
        output_paths.append(path)
        print(f"Fold {fold} complete: {len(probabilities):,} predictions.", flush=True)

    mean_probabilities = np.mean(np.stack(probabilities_by_fold), axis=0)
    result = prediction_frame(manifest.id, mean_probabilities)
    probability_path = REPORT / "usage_test_probabilities.csv"
    prediction_path = REPORT / "usage_test_predictions.csv"
    result.to_csv(probability_path, index=False)
    result[["id", "usage"]].to_csv(prediction_path, index=False)
    output_paths.extend([probability_path, prediction_path, REPORT / "inference_recipe.json"])
    freeze = {
        "frozen_at_utc": utc_now(),
        "reference_labels_opened": False,
        "test_rows": len(result),
        "outputs": {path.name: compute_sha256(path) for path in output_paths},
        "predicted_class_counts": result.usage.value_counts().to_dict(),
    }
    save_json(freeze, REPORT / "prediction_freeze.json")
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    main()
