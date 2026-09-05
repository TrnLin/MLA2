"""Separate, reproducible IEEE-FP32 evaluations of registered gender checkpoints."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data import get_cv_split, get_samples
from fashion.data.hashing import compute_sha256
from fashion.train.config import NARROW_GEM3_FAMILY, Task3BaselineConfig
from fashion.train.task3_decisions import (
    CORE_CORRUPTIONS,
    oof_metrics,
    probability_columns,
    robustness_changes,
    validate_oof,
)
from fashion.train.task3_gender_diagnostic import verify_checkpoint_metadata, verify_input_images
from fashion.train.task3_gender_precision import PRECISION_PATHS

POLICY = "saved_checkpoint_ieee_fp32_batch128_v1"


def save_ieee_predictions(predictions, path, classes):
    """Preserve FP32 values in CSV and score the exact persisted probabilities."""
    predictions = predictions.copy()
    columns = ["confidence", *probability_columns(classes)]
    predictions[columns] = predictions[columns].astype(np.float64)
    predictions.to_csv(path, index=False)
    saved = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    return oof_metrics(saved, classes)


def evaluation_identity(run, *, root):
    """Tie a reusable evaluation to its source files, data and evaluation code."""
    root = Path(root)
    dependencies = (
        "src/fashion/train/task3_gender_ieee.py",
        "src/fashion/train/task3_gender_precision.py",
        "src/fashion/train/task3_gender_diagnostic.py",
        "src/fashion/train/task3_baseline.py",
        "src/fashion/train/config.py",
        "src/fashion/train/model.py",
        "src/fashion/train/data.py",
        "src/fashion/train/metrics.py",
        "src/fashion/data/dataset.py",
        "src/fashion/data/images.py",
        "data/processed/splits.csv",
        "data/processed/label_maps.json",
    )
    return {
        "policy": POLICY,
        "run_id": run["run_id"],
        "source_sha256": run["sha256"],
        "dependencies": {name: compute_sha256(root / name) for name in dependencies},
    }


def load_ieee_evaluation(directory, *, run, splits, classes, identity):
    """Read a complete hash-verified evaluation and check its canonical validation scope."""
    directory = Path(directory)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    if manifest["identity"] != identity or not manifest["weights_and_buffers_unchanged"]:
        raise ValueError("IEEE evaluation identity or frozen model state disagrees")
    required = {
        "metrics.json",
        "robustness.csv",
        "clean_train_predictions.csv",
        "oof_predictions.csv",
    }
    required.update(f"{name}_predictions.csv" for name in CORE_CORRUPTIONS)
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete IEEE evaluation artifact manifest")
    if (
        manifest["precision_settings"] != {key: "ieee" for key in PRECISION_PATHS}
        or not manifest["settings_restored"]
    ):
        raise ValueError("IEEE evaluation precision contract disagrees")
    for name, digest in manifest["files"].items():
        if compute_sha256(directory / name) != digest:
            raise ValueError(f"IEEE evaluation artifact changed: {name}")
    metrics = json.loads((directory / "metrics.json").read_text())
    if metrics.get("comparison_precision") != POLICY:
        raise ValueError("IEEE metrics have the wrong evaluation precision")
    _, expected = get_cv_split(splits, run["fold"])
    expected = get_samples(expected, target="gender")
    predictions = validate_oof(
        pd.read_csv(
            directory / "oof_predictions.csv", keep_default_na=False, float_precision="round_trip"
        ),
        expected,
        target="gender",
        classes=classes,
        run_ids_by_fold={run["fold"]: run["run_id"]},
    )
    recalculated = oof_metrics(predictions, classes)
    for name in ("macro_f1", "nll", "ece_15"):
        if not np.isclose(metrics[name], recalculated[name], rtol=0, atol=1e-10):
            raise ValueError(f"IEEE saved {name} differs from probabilities")
    robust = pd.read_csv(directory / "robustness.csv", keep_default_na=False)
    robustness_changes(
        robust,
        clean_by_fold={run["fold"]: metrics["macro_f1"]},
        run_ids_by_fold={run["fold"]: run["run_id"]},
    )
    return {
        **run,
        "metrics": metrics,
        "predictions": predictions,
        "robustness": robust,
        "evaluation_manifest": manifest,
        "evaluation_directory": str(directory),
    }


def evaluate_gender_ieee(run, *, splits, classes, output, root):
    """No optimizer: score clean train/validation and every standard corruption."""
    import torch

    from fashion.train.data import CORE_CORRUPTIONS, Task3ImageDataset
    from fashion.train.model import Task3GeM3CNN
    from fashion.train.task3_baseline import _json_dump, _loader, _pass, _prediction_frame
    from fashion.train.task3_gender_precision import ieee_precision, precision_settings

    root, output = Path(root), Path(output)
    source = Path(run["directory"])
    identity = evaluation_identity(run, root=root)
    identity["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    training, validation = (
        get_samples(f, target="gender") for f in get_cv_split(splits, run["fold"])
    )
    verified_images = verify_input_images(pd.concat([training, validation]), root)
    if (output / "evaluation_manifest.json").is_file():
        return load_ieee_evaluation(
            output, run=run, splits=splits, classes=classes, identity=identity
        )
    output.mkdir(parents=True, exist_ok=True)
    payload = run["config"]
    if payload["effective_model_family"] not in {"task3_small_cnn_gem_p3", NARROW_GEM3_FAMILY}:
        raise ValueError("IEEE comparison only supports the saved GeM gender models")
    values = {f.name: payload[f.name] for f in fields(Task3BaselineConfig)}
    values["channels"] = tuple(values["channels"])
    config = Task3BaselineConfig(**values)
    stats = json.loads((source / "normalization.json").read_text())
    checkpoint = torch.load(source / "final_epoch.pt", map_location="cpu", weights_only=True)
    verify_checkpoint_metadata(
        checkpoint, run_id=run["run_id"], config=payload, classes=classes, normalization=stats
    )
    model = Task3GeM3CNN(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    device = torch.device("cuda")
    model.to(device).eval().requires_grad_(False)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    original_precision = precision_settings(torch)
    files = []

    def predict(frame, *, filename, corruption=None):
        dataset = Task3ImageDataset(
            frame,
            target="gender",
            label_to_index={c: i for i, c in enumerate(classes)},
            mean=stats["mean"],
            std=stats["std"],
            root=root,
            image_size=(80, 60),
            image_view="full",
            corruption=corruption,
        )
        loader = _loader(dataset, config=config, shuffle=False, device=device)
        loss, labels, probabilities, trace = _pass(
            model, loader, torch.nn.CrossEntropyLoss(), device
        )
        predictions = _prediction_frame(labels, probabilities, trace, classes, run["run_id"])
        metrics = save_ieee_predictions(predictions, output / filename, classes)
        files.append(filename)
        return {**metrics, "loss": loss}

    print(f"IEEE evaluation: {run['run_id']}", flush=True)
    with ieee_precision(torch), torch.autocast(device_type="cuda", enabled=False):
        ieee_settings = precision_settings(torch)
        clean = predict(validation, filename="oof_predictions.csv")
        train = predict(training, filename="clean_train_predictions.csv")
        rows = []
        for corruption in CORE_CORRUPTIONS:
            value = predict(
                validation, filename=f"{corruption}_predictions.csv", corruption=corruption
            )
            rows.append(
                {
                    "run_id": run["run_id"],
                    "validation_fold": run["fold"],
                    "corruption": corruption,
                    "loss": value["loss"],
                    "macro_f1": value["macro_f1"],
                    "macro_f1_change": value["macro_f1"] - clean["macro_f1"],
                }
            )
    if precision_settings(torch) != original_precision:
        raise RuntimeError("Evaluation failed to restore training precision settings")
    unchanged = all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
    if not unchanged:
        raise RuntimeError("IEEE evaluation changed model state")
    peak = int(torch.cuda.max_memory_allocated())
    if not 0 < peak < 3_000_000_000:
        raise RuntimeError("IEEE evaluation exceeded the strict 3 GB GPU memory limit")
    metrics = {
        **run["metrics"],
        **clean,
        "final_train_eval_loss": train["loss"],
        "final_train_eval_macro_f1": train["macro_f1"],
        "final_train_validation_macro_f1_gap": train["macro_f1"] - clean["macro_f1"],
        "final_train_eval_metrics": train,
        "comparison_precision": POLICY,
        "historical_registered_macro_f1": run["metrics"]["macro_f1"],
        "historical_registered_train_macro_f1": run["metrics"]["final_train_eval_macro_f1"],
    }
    _json_dump(metrics, output / "metrics.json")
    pd.DataFrame(rows).to_csv(output / "robustness.csv", index=False)
    files.extend(["metrics.json", "robustness.csv"])
    _json_dump(
        {
            "identity": identity,
            "precision_settings": ieee_settings,
            "settings_restored": True,
            "weights_and_buffers_unchanged": unchanged,
            "verified_input_images": verified_images,
            "evaluation_peak_memory_bytes": peak,
            "runtime": {
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "resource_metrics_note": (
                "Training time and training peak memory stay from the registered run."
            ),
            "files": {name: compute_sha256(output / name) for name in files},
        },
        output / "evaluation_manifest.json",
    )
    return load_ieee_evaluation(output, run=run, splits=splits, classes=classes, identity=identity)
