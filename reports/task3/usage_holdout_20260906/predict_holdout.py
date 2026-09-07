"""Run the same three frozen Usage models on the canonical unlabelled holdout."""

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
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.model import Task3BaselineCNN

REPORT = Path(__file__).resolve().parent
ORIGINAL_TEST = ROOT / "reports/task3/usage_teacher_vs_expanded_test_20260906"
EXPANDED_TEST = ROOT / "reports/task3/usage_expanded_e8_test_20260906"
SPLITS = ROOT / "data/processed/splits.csv"
EXPANDED_SPLITS = ROOT / "data/processed/teacher_plus_rare_usage_20260906/splits.csv"
MODELS = ("E1", "E8", "Expanded")

spec = importlib.util.spec_from_file_location(
    "unlabelled_inference", EXPANDED_TEST / "predict_test.py"
)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    if (REPORT / "prediction_freeze.json").exists():
        raise RuntimeError("Holdout predictions are already frozen; preserve this evaluation.")
    torch.set_num_threads(2)
    recipes = []
    for directory in (ORIGINAL_TEST, EXPANDED_TEST):
        freeze = json.loads((directory / "prediction_freeze.json").read_text())
        for name, digest in freeze["outputs"].items():
            assert compute_sha256(directory / name) == digest, name
        recipes.append(json.loads((directory / "inference_recipe.json").read_text()))
    checkpoints = [
        *recipes[0]["checkpoints"],
        *[{**row, "model": "Expanded"} for row in recipes[1]["checkpoints"]],
    ]
    assert recipes[0]["class_names"] == recipes[1]["class_names"] == list(shared.CLASSES)
    for name in MODELS:
        chosen = sorted((c for c in checkpoints if c["model"] == name), key=lambda c: c["fold"])
        assert [c["fold"] for c in chosen] == list(range(5))
        for checkpoint in chosen:
            assert compute_sha256(resolve_task3_path(checkpoint["path"], root=ROOT)) == checkpoint["sha256"]

    splits = load_splits(SPLITS)
    assert splits.loc[splits.partition.eq("holdout"), "usage"].eq("").all()
    holdout = (
        splits.loc[splits.partition.eq("holdout"), ["id", "path", "sha256", "product_family_group"]]
        .sort_values("id")
        .reset_index(drop=True)
    )
    assert len(holdout) == 5778 and holdout.id.is_unique
    development = pd.read_csv(
        EXPANDED_SPLITS,
        usecols=["id", "partition", "sha256", "product_family_group"],
        keep_default_na=False,
    )
    development = development.loc[development.partition.eq("development")]
    assert not set(holdout.id).intersection(development.id)
    assert not set(holdout.product_family_group).intersection(development.product_family_group)
    assert not set(holdout.sha256).intersection(development.sha256)
    for row in holdout.itertuples():
        assert compute_sha256(resolve_task3_path(row.path, root=ROOT)) == row.sha256
    holdout.to_csv(REPORT / "holdout_image_manifest.csv", index=False)
    print(
        f"Verified {len(holdout):,} holdout images; no training ID, family or exact-file overlap.",
        flush=True,
    )

    recipe = {
        "declared_at_utc": now(),
        "user_authorization": "ok can you test it on the holdout set?",
        "evaluation_scope": "Canonical teacher holdout; Usage only",
        "holdout_rows": len(holdout),
        "class_names": list(shared.CLASSES),
        "models": list(MODELS),
        "checkpoints": checkpoints,
        "checkpoint_rule": "The same saved final-epoch checkpoints used for the test comparison",
        "aggregation": "Equal mean of five saved fold softmax vectors, then argmax",
        "fold_weights": [0.2] * 5,
        "preprocessing": "Full RGB 80x60; checkpoint-specific normalization; no augmentation",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "training_or_tuning": False,
        "holdout_labels_used_for_inference": False,
        "training_id_family_and_exact_file_overlap": False,
        "inputs": {
            str(path.relative_to(ROOT)): compute_sha256(path)
            for path in (
                SPLITS,
                EXPANDED_SPLITS,
                Path(__file__),
                ORIGINAL_TEST / "inference_recipe.json",
                EXPANDED_TEST / "inference_recipe.json",
                EXPANDED_TEST / "predict_test.py",
                ROOT / "src/fashion/data/images.py",
                ROOT / "src/fashion/train/model.py",
                ROOT / "src/fashion/train/config.py",
                ROOT / "src/fashion/data/dataset.py",
                REPORT / "holdout_image_manifest.csv",
            )
        },
    }
    save_json(recipe, REPORT / "inference_recipe.json")
    output_paths = [REPORT / "holdout_image_manifest.csv", REPORT / "inference_recipe.json"]
    for name in MODELS:
        directory = REPORT / name
        directory.mkdir(exist_ok=True)
        outputs = []
        for source in sorted(
            (c for c in checkpoints if c["model"] == name), key=lambda c: c["fold"]
        ):
            checkpoint = torch.load(resolve_task3_path(source["path"], root=ROOT), map_location="cpu", weights_only=True)
            assert checkpoint["run_id"] == source["run_id"]
            assert checkpoint["class_names"] == list(shared.CLASSES)
            assert checkpoint["normalization"] == source["normalization"]
            config = checkpoint["config"]
            assert config["target"] == "usage" and config["scratch"]
            assert config.get("input_view", "full") == "full"
            config_kwargs = {
                field.name: config[field.name] for field in fields(Task3BaselineConfig)
            }
            config_kwargs["channels"] = tuple(config_kwargs["channels"])
            model = Task3BaselineCNN(Task3BaselineConfig(**config_kwargs))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            dataset = shared.UnlabelledTestImages(holdout, checkpoint["normalization"])
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            batches = []
            with torch.inference_mode():
                for images in loader:
                    batches.append(torch.softmax(model(images), dim=1).numpy())
            probabilities = np.concatenate(batches)
            path = directory / f"usage_holdout_fold_{source['fold']}.csv"
            shared.prediction_frame(holdout.id, probabilities).to_csv(path, index=False)
            output_paths.append(path)
            outputs.append(probabilities.astype(np.float64))
            print(
                f"{name} fold {source['fold']}: {len(probabilities):,} predictions saved",
                flush=True,
            )
        averaged = np.mean(np.stack(outputs), axis=0)
        result = shared.prediction_frame(holdout.id, averaged)
        path = directory / "usage_holdout_probabilities.csv"
        result.to_csv(path, index=False)
        output_paths.append(path)
        path = directory / "usage_holdout_predictions.csv"
        result[["id", "usage"]].to_csv(path, index=False)
        output_paths.append(path)

    freeze = {
        "frozen_at_utc": now(),
        "holdout_rows": len(holdout),
        "holdout_reference_labels_opened": False,
        "outputs": {str(p.relative_to(REPORT)): compute_sha256(p) for p in output_paths},
    }
    save_json(freeze, REPORT / "prediction_freeze.json")
    print(
        "All three holdout predictions are frozen. Ready to open Usage labels for scoring.",
        flush=True,
    )


if __name__ == "__main__":
    main()
