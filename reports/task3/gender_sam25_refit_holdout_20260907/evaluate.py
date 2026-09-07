"""Compare the saved full-development SAM25 refit with the frozen five-model average."""

import json
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fashion.config import ROOT
from fashion.data import load_splits
from fashion.data.dataset import load_splits_for_final_evaluation
from fashion.data.gender_name_truth import load_gender_name_truth_variant, product_name_gender_cues
from fashion.data.hashing import compute_sha256
from fashion.data.images import load_and_transform_image
from fashion.train.config import Task3BaselineConfig
from fashion.train.metrics import classification_metrics
from fashion.train.mixup import training_contract
from fashion.train.model import Task3GeM3CNN
from fashion.train.task3_gender_precision import ieee_precision

REPORT = Path(__file__).resolve().parent
REFIT = ROOT / "results/evidence/task3/gender_sam25_refit_20260907"
AVERAGE = ROOT / "reports/task3/gender_sam25_holdout_test_20260906"
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
COLS = [f"probability_{i}_{name}" for i, name in enumerate(CLASSES)]


def read(path):
    return json.loads(path.read_text())


def save(value, name):
    (REPORT / name).write_text(json.dumps(value, indent=2) + "\n")


class Images(Dataset):
    def __init__(self, frame, stats):
        self.frame, self.stats = frame, stats

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        array = load_and_transform_image(
            ROOT / self.frame.iloc[index].path,
            image_size=(80, 60),
            mean=self.stats["mean"],
            std=self.stats["std"],
        )
        return torch.from_numpy(array.transpose(2, 0, 1).copy())


def main():
    torch.set_num_threads(2)
    manifest = read(REFIT / "model_manifest.json")
    assert manifest["status"] == "complete" and manifest["selected_epoch"] == 25
    for name, record in manifest["files"].items():
        assert compute_sha256(REFIT / name) == record["sha256"], name
    config = read(REFIT / "config.json")
    for name in ("train/model.py", "train/config.py", "data/images.py"):
        assert compute_sha256(ROOT / "src/fashion" / name) == config["implementation_sha256"][name]
    corrected = load_gender_name_truth_variant(ROOT)
    training = corrected.loc[corrected.partition.eq("development")].reset_index(drop=True)
    contract = training_contract(training, validation_fold=None, scope="development_refit")
    assert contract == config["mixup_contract"] and len(training) == 32773
    saved_rows = pd.read_csv(REFIT / "training_rows.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(
        training[saved_rows.columns].astype(str),
        saved_rows.astype(str),
        check_dtype=False,
    )
    history = pd.read_csv(REFIT / "history.csv")
    assert history.epoch.tolist() == list(range(1, 26))
    assert history.training_rows.eq(32773).all() and history.optimizer_steps.eq(257).all()
    assert history.selected_checkpoint.tolist() == [False] * 24 + [True]
    expected_lr = 1e-5 + 0.5 * (0.001 - 1e-5) * (1 + np.cos(np.pi * np.arange(25) / 30))
    np.testing.assert_allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-12)
    sam, mixup = read(REFIT / "sam_training.json"), read(REFIT / "mixup_training.json")
    assert len(sam["epochs"]) == len(mixup["epochs"]) == 25
    for s, m in zip(sam["epochs"], mixup["epochs"]):
        assert s["epoch"] == m["epoch"] and s["rows"] == m["rows"] == 32773
        assert s["batches"] == m["batches"] == s["optimizer_steps"] == 257
        assert s["forward_backward_passes"] == 514
    stats = read(REFIT / "normalization.json")
    checkpoint = torch.load(REFIT / "final_epoch.pt", map_location="cpu", weights_only=True)
    assert checkpoint["config"] == config and checkpoint["run_id"] == manifest["run_id"]
    assert checkpoint["class_names"] == CLASSES
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 25
    for key, value in checkpoint["normalization"].items():
        assert stats[key] == value
    kwargs = {f.name: config[f.name] for f in fields(Task3BaselineConfig)}
    kwargs["channels"] = tuple(kwargs["channels"])
    model = Task3GeM3CNN(Task3BaselineConfig(**kwargs), classifier_dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    splits = load_splits(ROOT / "data/processed/splits.csv")
    holdout = splits.loc[splits.partition.eq("holdout")].sort_values("id").reset_index(drop=True)
    assert len(holdout) == 5778 and set(holdout.id).isdisjoint(training.id)
    assert set(holdout.product_family_group).isdisjoint(training.product_family_group)
    for row in holdout.itertuples():
        assert compute_sha256(ROOT / row.path) == row.sha256
    freeze = read(AVERAGE / "prediction_freeze.json")
    for name in ["gender_holdout_probabilities.csv", "holdout_image_manifest.csv"] + [
        f"gender_holdout_fold_{f}.csv" for f in range(5)
    ]:
        assert compute_sha256(AVERAGE / name) == freeze["outputs"][name]
    average = (
        pd.read_csv(AVERAGE / "gender_holdout_probabilities.csv").set_index("id").loc[holdout.id]
    )
    folded = [
        pd.read_csv(AVERAGE / f"gender_holdout_fold_{f}.csv").set_index("id").loc[holdout.id, COLS]
        for f in range(5)
    ]
    np.testing.assert_allclose(np.mean(folded, axis=0), average[COLS], rtol=0, atol=1e-12)
    print("Verified refit artifacts, all 25 epochs, and the saved five-model average", flush=True)
    outputs = []
    with ieee_precision(torch), torch.inference_mode():
        for batch in DataLoader(Images(holdout, stats), batch_size=128, shuffle=False):
            outputs.append(torch.softmax(model(batch), dim=1).numpy())
    probabilities = np.concatenate(outputs).astype(np.float64)
    np.testing.assert_allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    predictions = holdout[["id"]].copy()
    predictions[COLS] = probabilities
    predictions["predicted_gender"] = np.asarray(CLASSES)[probabilities.argmax(1)]
    predictions.to_csv(REPORT / "refit_holdout_probabilities.csv", index=False)
    # This is an explicitly requested follow-up on an already opened reserved holdout.
    labels = load_splits_for_final_evaluation(evaluation_unlocked=True)
    labels = labels.set_index("id").loc[holdout.id]
    prior_labels = pd.read_csv(AVERAGE / "holdout_predictions_and_labels.csv").set_index("id")
    assert labels.gender.eq(prior_labels.loc[holdout.id, "reference_gender"]).all()
    cues = product_name_gender_cues(labels.productDisplayName)
    diagnostic = cues.name_gender.where(cues.cue_count.eq(1), labels.gender)
    table, classes, metrics = [], [], {}
    for basis, actual in [
        ("Original labels", labels.gender),
        ("Fixed name-rule diagnostic", diagnostic),
    ]:
        y = actual.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
        for name, p in [
            ("Five-model average", average[COLS].to_numpy()),
            ("Single refit", probabilities),
        ]:
            result = classification_metrics(y, p, CLASSES)
            metrics[f"{basis}: {name}"] = result
            table.append(
                {
                    "labels": basis,
                    "model": name,
                    "images": len(y),
                    **{k: result[k] for k in ["accuracy", "macro_f1", "nll", "ece_15"]},
                }
            )
            classes.extend([{"labels": basis, "model": name, **row} for row in result["per_class"]])
    summary = pd.DataFrame(table)
    summary.to_csv(REPORT / "holdout_comparison.csv", index=False)
    per_class = pd.DataFrame(classes)
    per_class.to_csv(REPORT / "holdout_per_class.csv", index=False)
    y = labels.gender.to_numpy()
    old_pred = np.asarray(CLASSES)[average[COLS].to_numpy().argmax(1)]
    new_pred = predictions.predicted_gender.to_numpy()
    changes = {
        "fixed_by_refit": int(((new_pred == y) & (old_pred != y)).sum()),
        "new_refit_errors": int(((new_pred != y) & (old_pred == y)).sum()),
        "both_wrong": int(((new_pred != y) & (old_pred != y)).sum()),
    }
    save(
        {
            "run_id": manifest["run_id"],
            "checks_passed": True,
            "training": read(REFIT / "metrics.json"),
            "checkpoint_sha256": compute_sha256(REFIT / "final_epoch.pt"),
            "paired_changes": changes,
            "holdout_images": 5778,
            "teacher_test_evaluated": False,
            "metrics": metrics,
        },
        "evaluation.json",
    )
    primary = per_class.loc[per_class.labels.eq("Original labels")]
    chart = primary.pivot(index="class_name", columns="model", values="f1").reindex(CLASSES) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    chart.plot.bar(ax=ax, rot=0, color=["#366d9b", "#c37c36"])
    ax.set(
        ylabel="F1 (%)",
        xlabel="",
        ylim=(0, 105),
        title="Gender reserved holdout · 5,778 images · original labels",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT / "holdout_comparison.png", dpi=160)
    figure_path = ROOT / "results/figures/task3/gender_sam25_refit_holdout_20260907.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    save(
        {
            "checkpoint_sha256": compute_sha256(REFIT / "final_epoch.pt"),
            "canonical_splits_sha256": compute_sha256(ROOT / "data/processed/splits.csv"),
            "average_probabilities_sha256": compute_sha256(
                AVERAGE / "gender_holdout_probabilities.csv"
            ),
            "outputs": {
                name: compute_sha256(REPORT / name)
                for name in (
                    "refit_holdout_probabilities.csv",
                    "holdout_comparison.csv",
                    "holdout_per_class.csv",
                    "evaluation.json",
                    "holdout_comparison.png",
                )
            },
        },
        "evaluation_provenance.json",
    )
    print(summary.to_string(index=False))
    print(primary.to_string(index=False))
    print(changes)


if __name__ == "__main__":
    main()
