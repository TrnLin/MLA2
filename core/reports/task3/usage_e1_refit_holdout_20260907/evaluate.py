"""Evaluate the fixed Usage E1 refit against its saved five-model holdout average."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from fashion.config import ROOT, TEACHER_TRAIN_CSV
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.images import load_and_transform_image
from fashion.train.metrics import classification_metrics
from fashion.train.model import Task3BaselineCNN
from fashion.train.task3_gender_precision import ieee_precision
from fashion.train.task3_usage_e1_refit import CLASSES, ROW_COLUMNS, prepare_refit

REPORT = Path(__file__).resolve().parent
REFIT = ROOT / "results/evidence/task3/usage_e1_refit_20260907"
AVERAGE = ROOT / "reports/task3/usage_holdout_20260906"
COLS = [f"probability_{name}" for name in CLASSES]


def read(path):
    return json.loads(path.read_text())


def save(value, name):
    (REPORT / name).write_text(json.dumps(value, indent=2) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


class Images(Dataset):
    def __init__(self, frame, stats):
        self.frame, self.stats = frame, stats

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        image = load_and_transform_image(
            ROOT / self.frame.iloc[index].path,
            image_size=(80, 60),
            mean=self.stats["mean"],
            std=self.stats["std"],
        )
        return torch.from_numpy(image.transpose(2, 0, 1).copy())


def verify_inputs():
    manifest = read(REFIT / "model_manifest.json")
    assert manifest["status"] == "complete" and manifest["training_completed"]
    assert manifest["selected_epoch"] == 30 and not manifest["validation_used"]
    for name, record in manifest["files"].items():
        assert record["path"] == name
        assert compute_sha256(REFIT / name) == record["sha256"], name
    config, payload, training = prepare_refit(root=ROOT)
    assert payload == read(REFIT / "config.json")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert digest == manifest["config_hash"]
    saved_rows = pd.read_csv(REFIT / "training_rows.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(
        saved_rows.astype(str), training[ROW_COLUMNS].astype(str), check_dtype=False
    )
    history = pd.read_csv(REFIT / "history.csv")
    assert history.epoch.tolist() == list(range(1, 31))
    assert history.training_rows.eq(32772).all()
    assert history.selected_checkpoint.tolist() == [False] * 29 + [True]
    expected_lr = 1e-5 + (0.001 - 1e-5) * (1 + np.cos(np.pi * np.arange(30) / 30)) / 2
    np.testing.assert_allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-12)
    assert np.isfinite(history.train_loss).all()
    stats = read(REFIT / "normalization.json")
    assert (
        stats["padding_excluded"]
        and stats["training_rows_sha256"] == payload["training_rows_sha256"]
    )
    checkpoint = torch.load(REFIT / "final_epoch.pt", map_location="cpu", weights_only=True)
    assert checkpoint["run_id"] == manifest["run_id"] and checkpoint["config"] == payload
    assert checkpoint["class_names"] == manifest["class_names"] == CLASSES
    assert checkpoint["epochs_completed"] == checkpoint["selected_epoch"] == 30
    assert checkpoint["checkpoint_policy"] == "final_epoch"
    assert all(stats[key] == value for key, value in checkpoint["normalization"].items())
    model = Task3BaselineCNN(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 391209
    model.eval().requires_grad_(False)
    registry = pd.read_csv(REFIT / "source_runs.csv", keep_default_na=False, dtype=str)
    registered = registry.loc[registry.run_id.eq(manifest["run_id"])]
    assert len(registered) == 1
    row = registered.iloc[0]
    assert row.status == "complete" and row.target == "usage" and row.task == "task3"
    assert row.config_hash == manifest["config_hash"]
    assert row.checkpoint_sha256 == manifest["files"]["final_epoch.pt"]["sha256"]
    splits = load_splits(ROOT / "data/processed/splits.csv")
    protected = splits.loc[splits.partition.eq("holdout")]
    assert protected.usage.eq("").all()
    holdout = (
        protected[["id", "path", "sha256", "product_family_group"]]
        .sort_values("id")
        .reset_index(drop=True)
    )
    assert len(holdout) == 5778 and holdout.id.is_unique
    for key in ("id", "product_family_group", "sha256"):
        assert set(holdout[key]).isdisjoint(training[key]), key
    for row in holdout.itertuples():
        assert compute_sha256(ROOT / row.path) == row.sha256
    freeze = read(AVERAGE / "prediction_freeze.json")
    names = ["holdout_image_manifest.csv", "E1/usage_holdout_probabilities.csv"] + [
        f"E1/usage_holdout_fold_{fold}.csv" for fold in range(5)
    ]
    for name in names:
        assert compute_sha256(AVERAGE / name) == freeze["outputs"][name], name
    previous_rows = pd.read_csv(AVERAGE / "holdout_image_manifest.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(holdout, previous_rows[holdout.columns], check_dtype=False)
    average = pd.read_csv(AVERAGE / "E1/usage_holdout_probabilities.csv", keep_default_na=False)
    assert average.id.tolist() == holdout.id.tolist()
    folded = []
    for fold in range(5):
        frame = pd.read_csv(AVERAGE / f"E1/usage_holdout_fold_{fold}.csv", keep_default_na=False)
        assert frame.id.tolist() == holdout.id.tolist()
        # Fold CSVs round-trip float32 outputs; the original mean used float64.
        folded.append(frame[COLS].to_numpy(dtype=np.float32).astype(np.float64))
    np.testing.assert_allclose(np.mean(folded, axis=0), average[COLS], rtol=0, atol=1e-12)
    return manifest, model, stats, holdout, average, history


def main():
    if (REPORT / "prediction_freeze.json").exists():
        raise RuntimeError("This refit evaluation is frozen; preserve its saved predictions.")
    torch.set_num_threads(2)
    manifest, model, stats, holdout, average, history = verify_inputs()
    print("Verified completed refit, 30 epochs, training boundaries and old average", flush=True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    outputs = []
    started = time.perf_counter()
    with ieee_precision(torch), torch.inference_mode():
        for batch in DataLoader(
            Images(holdout, stats), batch_size=32, shuffle=False, num_workers=0
        ):
            outputs.append(torch.softmax(model(batch), dim=1).numpy())
    seconds = time.perf_counter() - started
    probabilities = np.concatenate(outputs).astype(np.float64)
    np.testing.assert_allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    predictions = holdout[["id"]].copy()
    predictions[COLS] = probabilities
    predictions["usage"] = np.asarray(CLASSES)[probabilities.argmax(1)]
    predictions.to_csv(REPORT / "refit_holdout_probabilities.csv", index=False)
    holdout.to_csv(REPORT / "holdout_image_manifest.csv", index=False)
    save(
        {
            "frozen_at_utc": now(),
            "user_authorization": (
                "review the result and compare its reserved holdout performance "
                "to the average one; Usage E1 refit"
            ),
            "new_blind_evaluation": False,
            "checkpoint_sha256": manifest["files"]["final_epoch.pt"]["sha256"],
            "class_names": CLASSES,
            "prediction_rule": (
                "One saved final-epoch model; eval; saved normalization; softmax then argmax"
            ),
            "device": "cpu",
            "torch": str(torch.__version__),
            "threads": 2,
            "batch_size": 32,
            "precision": "IEEE float32",
            "inference_seconds_including_image_preprocessing": seconds,
            "teacher_test_evaluated": False,
            "weights_and_buffers_unchanged": True,
            "inputs": {
                str(path.relative_to(ROOT)): compute_sha256(path)
                for path in [
                    REFIT / "model_manifest.json",
                    REFIT / "source_runs.csv",
                    AVERAGE / "prediction_freeze.json",
                    Path(__file__),
                    ROOT / "data/processed/splits.csv",
                ]
            },
            "outputs": {
                name: compute_sha256(REPORT / name)
                for name in ("refit_holdout_probabilities.csv", "holdout_image_manifest.csv")
            },
        },
        "prediction_freeze.json",
    )
    # The holdout was already opened for the old model. Only Usage labels are scored here.
    prior_metrics = read(AVERAGE / "holdout_metrics.json")
    assert compute_sha256(TEACHER_TRAIN_CSV) == prior_metrics["source_sha256"]
    ids, reference = set(holdout.id), {}
    with TEACHER_TRAIN_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            product_id = int(row["id"])
            if product_id in ids:
                assert product_id not in reference and row["usage"].strip() in CLASSES
                reference[product_id] = row["usage"].strip()
    assert set(reference) == ids
    truth = holdout.id.map(reference)
    old_labels = pd.read_csv(AVERAGE / "holdout_predictions_and_labels.csv", keep_default_na=False)
    assert old_labels.id.tolist() == holdout.id.tolist()
    assert truth.tolist() == old_labels.actual_usage.tolist()
    y = truth.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    metrics, summary, classes = {}, [], []
    for name, values in [
        ("Five-model average", average[COLS].to_numpy()),
        ("Single refit", probabilities),
    ]:
        result = classification_metrics(y, values, CLASSES)
        predicted = np.asarray(CLASSES)[values.argmax(1)]
        np.testing.assert_allclose(result["accuracy"], accuracy_score(truth, predicted))
        np.testing.assert_allclose(
            result["macro_f1"],
            f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0),
        )
        result["correct"] = int((predicted == truth.to_numpy()).sum())
        metrics[name] = result
        summary.append(
            {
                "model": name,
                **{
                    key: result[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "balanced_accuracy",
                        "correct",
                        "nll",
                        "ece_15",
                    )
                },
            }
        )
        for item in result["per_class"]:
            item = dict(item)
            item["correct"] = int(((truth == item["class_name"]) & (predicted == truth)).sum())
            classes.append({"model": name, **item})
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "nll", "ece_15"):
        np.testing.assert_allclose(
            metrics["Five-model average"][key], prior_metrics["models"]["E1"][key]
        )
    comparison = holdout[["id", "product_family_group"]].copy()
    comparison["actual_usage"] = truth
    comparison["average_prediction"] = np.asarray(CLASSES)[average[COLS].to_numpy().argmax(1)]
    comparison["refit_prediction"] = predictions.usage
    old_correct = comparison.average_prediction.eq(truth)
    new_correct = comparison.refit_prediction.eq(truth)
    comparison.to_csv(REPORT / "holdout_predictions_and_labels.csv", index=False)
    paired = {
        "fixed_by_refit": int((new_correct & ~old_correct).sum()),
        "new_refit_errors": int((old_correct & ~new_correct).sum()),
        "both_wrong": int((~old_correct & ~new_correct).sum()),
    }
    pd.DataFrame(summary).to_csv(REPORT / "holdout_comparison.csv", index=False)
    per_class = pd.DataFrame(classes)
    per_class.to_csv(REPORT / "holdout_per_class.csv", index=False)
    save(
        {
            "run_id": manifest["run_id"],
            "checks_passed": True,
            "checkpoint_sha256": manifest["files"]["final_epoch.pt"]["sha256"],
            "training": manifest["metrics"],
            "holdout_images": len(holdout),
            "scored_at_utc": now(),
            "reference_labels_sha256": compute_sha256(TEACHER_TRAIN_CSV),
            "teacher_test_evaluated": False,
            "new_blind_evaluation": False,
            "metrics": metrics,
            "paired_changes": paired,
            "inference_seconds_including_image_preprocessing": seconds,
        },
        "evaluation.json",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.8]})
    axes[0].plot(history.epoch, history.train_loss, color="#366d9b", linewidth=2)
    axes[0].set(
        title="Refit training completed", xlabel="Epoch", ylabel="Training loss", xlim=(1, 30)
    )
    scores = (
        per_class.pivot(index="class_name", columns="model", values="f1").reindex(CLASSES) * 100
    )
    scores.plot.bar(ax=axes[1], rot=30, color=["#366d9b", "#c37c36"], width=0.8)
    axes[1].set(
        title="Same 5,778 reserved holdout images", ylabel="Class F1 (%)", xlabel="", ylim=(0, 105)
    )
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].text(
        0.01,
        0.98,
        "Nine fixed classes; Home has no holdout images",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(REPORT / "holdout_comparison.png", dpi=160)
    figure_path = ROOT / "results/figures/task3/usage_e1_refit_holdout_20260907.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    save(
        {
            "outputs": {
                name: compute_sha256(REPORT / name)
                for name in (
                    "evaluation.json",
                    "holdout_comparison.csv",
                    "holdout_per_class.csv",
                    "holdout_predictions_and_labels.csv",
                    "holdout_comparison.png",
                    "prediction_freeze.json",
                )
            }
        },
        "evaluation_provenance.json",
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(per_class.to_string(index=False), flush=True)
    print(json.dumps(paired), flush=True)


if __name__ == "__main__":
    main()
