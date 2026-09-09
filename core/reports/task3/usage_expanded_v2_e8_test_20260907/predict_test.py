"""Freeze test predictions from the five completed E8 models trained with 687 additions."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import importlib.util
import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train.config import Task3BaselineConfig
from fashion.train.model import Task3BaselineCNN
from fashion.train.registry import RunRegistry
from fashion.train.task3_decisions import probability_columns

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/usage_expanded_v2_e8_20260906"
MANIFEST = ROOT / "data/processed/prediction_manifest.csv"
TEMPLATE = ROOT / "data/raw/teacher/test/styles_prediction.csv"
PRIOR_INFERENCE = ROOT / "reports/task3/usage_expanded_e8_test_20260906/predict_test.py"
CLASSES = v2.CLASSES
spec = importlib.util.spec_from_file_location("original_test_inference", PRIOR_INFERENCE)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def model_from_checkpoint(checkpoint):
    settings = {
        field.name: checkpoint["config"][field.name] for field in fields(Task3BaselineConfig)
    }
    settings["channels"] = tuple(settings["channels"])
    model = Task3BaselineCNN(Task3BaselineConfig(**settings))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def main():
    torch.set_num_threads(2)
    if (REPORT / "prediction_freeze.json").exists():
        raise RuntimeError("This evaluation already has frozen predictions; preserve them.")
    assert tuple(shared.CLASSES) == tuple(CLASSES)
    splits, contract = v2.validate_dataset(check_images=False)
    rows = sorted(
        RunRegistry(EVIDENCE / "results/runs.csv")._read_rows(),
        key=lambda row: int(row["validation_fold"]),
    )
    assert len(rows) == 5 and [int(row["validation_fold"]) for row in rows] == list(range(5))
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    template = pd.read_csv(TEMPLATE, usecols=["id"], keep_default_na=False)
    assert manifest.id.is_unique and template.id.is_unique
    assert len(template) == 5829 and set(manifest.id) == set(template.id)
    assert not set(template.id).intersection(splits.id)
    manifest = manifest.set_index("id").loc[template.id].reset_index()
    for row in manifest.itertuples():
        assert compute_sha256(resolve_task3_path(row.path, root=ROOT)) == row.sha256, row.id
    development = splits.loc[splits.partition.eq("development")]
    assert not set(manifest.sha256).intersection(development.sha256)
    exact_matches = manifest[["id", "sha256"]].merge(
        splits[["id", "sha256", "partition"]],
        on="sha256",
        suffixes=("_test", "_split"),
    )
    assert exact_matches.partition.eq("quarantine").all()
    exact_matches.to_csv(REPORT / "test_hash_matches_quarantine.csv", index=False)
    print(
        f"Verified {len(manifest):,} test images; no development ID/hash overlap.",
        flush=True,
    )

    recipe = dict(
        declared_at_utc=now(),
        target="usage",
        experiment_id=v2.EXPERIMENT,
        dataset=contract,
        test_rows=len(manifest),
        class_names=list(CLASSES),
        aggregation="Equal arithmetic mean of five saved fold softmax probability vectors; argmax",
        fold_weights=[0.2] * 5,
        checkpoint_rule="Saved final epoch 30; folds 0 through 4",
        preprocessing="Same full RGB 80x60 transform; saved normalization per fold",
        device="cpu",
        torch_version=str(torch.__version__),
        labels_used_for_inference=False,
        training_or_tuning=False,
        test_labels_previously_seen=True,
        test_exact_hash_matches_development=0,
        test_exact_hash_matches_quarantine=int(exact_matches.id_test.nunique()),
        evaluation_reason="User requested the official test result for E8 with all 687 additions",
        numerical_reproduction_tolerance=0.005,
        checkpoints=[],
        reproduction_checks=[],
        inputs={
            str(p.relative_to(ROOT)): compute_sha256(p)
            for p in (
                MANIFEST,
                TEMPLATE,
                resolve_task3_path(v2.DATA_DIRECTORY, root=ROOT) / "splits.csv",
                EVIDENCE / "results/runs.csv",
                Path(__file__),
                PRIOR_INFERENCE,
                ROOT / "src/fashion/data/images.py",
                ROOT / "src/fashion/train/model.py",
                ROOT / "src/fashion/train/config.py",
            )
        },
    )
    checkpoints = []
    for row in rows:
        fold = int(row["validation_fold"])
        directory = EVIDENCE / row["run_id"]
        verified = v2._verified_completed_fold(
            row=row,
            path=directory,
            splits=splits,
            fold=fold,
            spec=v2.expanded_usage_spec(),
            split_digest=v2.SPLIT_SHA256,
        )
        path = directory / "final_epoch.pt"
        assert compute_sha256(path) == row["checkpoint_sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["run_id"] == row["run_id"]
        assert checkpoint["class_names"] == list(CLASSES)
        assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 30
        assert checkpoint["checkpoint_policy"] == "final_epoch"
        config = checkpoint["config"]
        assert config == json.loads((directory / "config.json").read_text())
        assert config["target"] == "usage" and config["scratch"]
        assert (
            config["effective_model_family"] == "task3_small_cnn" and config["input_view"] == "full"
        )
        assert config["expanded_dataset"]["split_sha256"] == v2.SPLIT_SHA256
        saved_normalization = json.loads((directory / "normalization.json").read_text())
        assert all(
            saved_normalization[key] == value for key, value in checkpoint["normalization"].items()
        )
        assert saved_normalization["validation_fold"] == fold
        assert saved_normalization["fit_scope"] == "fold_training_content_pixels_only"
        model = model_from_checkpoint(checkpoint)
        _, validation = v2.training_scope(splits, fold)
        controls = (
            validation.loc[validation.source_dataset.eq("teacher")]
            .groupby("usage", sort=True)
            .head(2)
        )
        data = shared.UnlabelledTestImages(controls, checkpoint["normalization"])
        with torch.inference_mode():
            probabilities = torch.softmax(
                model(torch.stack([data[i] for i in range(len(data))])), dim=1
            ).numpy()
        saved = v2.read_predictions(verified["prediction_path"]).set_index("id").loc[controls.id]
        drift = float(np.abs(probabilities - saved[probability_columns(CLASSES)].to_numpy()).max())
        assert drift < recipe["numerical_reproduction_tolerance"], (fold, drift)
        assert np.array_equal(probabilities.argmax(axis=1), saved.predicted_index.to_numpy())
        recipe["checkpoints"].append(
            dict(
                fold=fold,
                run_id=row["run_id"],
                path=str(path.relative_to(ROOT)),
                sha256=row["checkpoint_sha256"],
                normalization=checkpoint["normalization"],
            )
        )
        recipe["reproduction_checks"].append(
            dict(
                fold=fold,
                rows=len(data),
                maximum_probability_drift=drift,
                all_class_decisions_match=True,
            )
        )
        checkpoints.append((fold, checkpoint))
        print(
            f"Verified fold {fold} checkpoint and {len(data)} saved validation decisions.",
            flush=True,
        )
    save_json(recipe, REPORT / "inference_recipe.json")

    outputs, fold_probabilities = (
        [REPORT / "inference_recipe.json", REPORT / "test_hash_matches_quarantine.csv"],
        [],
    )
    for fold, checkpoint in checkpoints:
        model = model_from_checkpoint(checkpoint)
        data = shared.UnlabelledTestImages(manifest, checkpoint["normalization"])
        loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=0)
        batches = []
        with torch.inference_mode():
            for number, images in enumerate(loader, start=1):
                batches.append(torch.softmax(model(images), dim=1).numpy())
                if number % 60 == 0:
                    print(
                        f"Fold {fold}: {min(number * 32, len(data)):,}/{len(data):,} test images",
                        flush=True,
                    )
        probabilities = np.concatenate(batches)
        path = REPORT / f"usage_test_fold_{fold}.csv"
        shared.prediction_frame(manifest.id, probabilities).to_csv(path, index=False)
        outputs.append(path)
        fold_probabilities.append(probabilities.astype(np.float64))
        print(f"Fold {fold} test predictions complete.", flush=True)

    result = shared.prediction_frame(manifest.id, np.mean(np.stack(fold_probabilities), axis=0))
    result.to_csv(REPORT / "usage_test_probabilities.csv", index=False)
    result[["id", "usage"]].to_csv(REPORT / "usage_test_predictions.csv", index=False)
    outputs.extend((REPORT / "usage_test_probabilities.csv", REPORT / "usage_test_predictions.csv"))
    for path, digest in recipe["inputs"].items():
        assert compute_sha256(resolve_task3_path(path, root=ROOT)) == digest
    for checkpoint in recipe["checkpoints"]:
        assert compute_sha256(resolve_task3_path(checkpoint["path"], root=ROOT)) == checkpoint["sha256"]
    freeze = dict(
        frozen_at_utc=now(),
        reference_labels_opened_in_this_inference=False,
        test_labels_previously_seen=True,
        test_rows=len(result),
        outputs={path.name: compute_sha256(path) for path in outputs},
        predicted_class_counts=result.usage.value_counts().to_dict(),
    )
    save_json(freeze, REPORT / "prediction_freeze.json")
    print("All test predictions frozen; ready to score.", flush=True)


if __name__ == "__main__":
    main()
