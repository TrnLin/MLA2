"""Verify saved five-fold SAM25 records and plot curves without training."""

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_samples, load_splits
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.task3_decisions import oof_metrics, validate_oof
from fashion.train.task3_gender_sam25 import verify_sam25_evidence
from fashion.train.task3_gender_sam25_cv import plot_cv_summary

HERE = Path(__file__).resolve().parent
CODE_COMMIT = "f2123af2a2d3aad11a80bafcec27b66fa12feac7"


def read(path):
    return json.loads(path.read_text())


def equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, atol=1e-12, rtol=1e-10), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


report = read(HERE / "cv_summary.json")
manifest = read(HERE / "model_manifest.json")
audit = read(HERE / "source_audit.json")
classes = manifest["class_names"]
assert report["status"] == "complete_for_review"
assert [f["fold"] for f in report["folds"]] == list(range(5))
assert [f["fold"] for f in manifest["folds"]] == list(range(5))
assert manifest["checkpoint_epoch"] == 25
assert manifest["all_development_refit"] is False
assert audit["optimizer_steps_at_audit"] == 0
equal(audit["identity"]["spec"], manifest["recipe"])
for name, sha in audit["identity"]["implementation_sha256"].items():
    content = subprocess.check_output(
        ["git", "show", f"{CODE_COMMIT}:src/fashion/train/{name}"], cwd=ROOT
    )
    assert hashlib.sha256(content).hexdigest() == sha, name
for name, sha in audit["identity"]["data_sha256"].items():
    assert compute_sha256(resolve_task3_path(name, root=ROOT)) == sha, name

splits = load_gender_name_truth_variant()
expected = get_samples(splits, partition="development", target="gender")
predictions = validate_oof(
    pd.read_csv(HERE / "aggregate/oof_predictions.csv", keep_default_na=False),
    expected,
    target="gender",
    classes=classes,
    run_ids_by_fold={r["fold"]: r["run_id"] for r in report["folds"]},
)
assert len(predictions) == 32773
for name, scope in report["scopes"].items():
    subset = predictions.loc[predictions.cv_fold.isin(scope["folds"])]
    equal(oof_metrics(subset, classes), scope["metrics"])
    equal(
        np.mean([r["gap"] for r in report["folds"] if r["fold"] in scope["folds"]]),
        scope["mean_clean_gap"],
    )
equal(read(HERE / "aggregate/metrics.json"), report["scopes"]["all_five"]["metrics"])
equal(
    pd.read_csv(HERE / "aggregate/confusion_matrix.csv", index_col=0).values.tolist(),
    report["scopes"]["all_five"]["metrics"]["confusion_matrix"],
)

fold_rows = []
histories = {}
fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True, sharey="row")
for fold, row in enumerate(report["folds"]):
    directory = HERE / f"fold{fold}"
    config, metrics = read(directory / "config.json"), read(directory / "metrics.json")
    ieee = read(HERE / f"ieee{fold}/metrics.json")
    evaluation = read(HERE / f"ieee{fold}/evaluation_manifest.json")
    assert evaluation["files"]["metrics.json"] == compute_sha256(HERE / f"ieee{fold}/metrics.json")
    assert evaluation["identity"]["run_id"] == row["run_id"]
    assert evaluation["weights_and_buffers_unchanged"] is True
    assert set(evaluation["precision_settings"].values()) == {"ieee"}
    for filename in ("config.json", "normalization.json", "history.csv", "metrics.json"):
        assert evaluation["identity"]["source_sha256"][filename] == compute_sha256(
            directory / filename
        )
    assert (
        evaluation["identity"]["source_sha256"]["final_epoch.pt"]
        == manifest["folds"][fold]["files"]["final_epoch.pt"]["sha256"]
    )
    history = pd.read_csv(directory / "history.csv")
    diagnostics = read(directory / "clean_epoch_diagnostics.json")
    histories[fold] = history
    verify_sam25_evidence(
        {"config": config, "metrics": metrics}, fold=fold, splits=splits, directory=directory
    )
    assert config["scratch"] is True and config["submission_eligible"] is True
    assert config["parameter_count"] == metrics["parameter_count"] == 390181
    assert config["refinement_prerequisite_sha256"] == compute_sha256(HERE / "source_audit.json")
    equal(config["child_experiment"], manifest["recipe"])
    assert manifest["folds"][fold]["run_id"] == metrics["run_id"] == row["run_id"]
    for name in ("config.json", "normalization.json"):
        record = manifest["folds"][fold]["files"][name]
        assert record["path"] == f"{row['run_id']}/{name}"
        assert record["sha256"] == compute_sha256(directory / name)
    current = oof_metrics(predictions.loc[predictions.cv_fold.eq(fold)], classes)
    equal(current["macro_f1"], row["validation_f1"])
    for key, value in current.items():
        equal(value, ieee[key])
    equal(ieee["final_train_eval_macro_f1"], row["train_f1"])
    equal(row["train_f1"] - row["validation_f1"], row["gap"])
    assert 0 < row["peak_memory_bytes"] < 3_000_000_000
    for diagnostic in diagnostics:
        epoch = diagnostic["epoch"]
        equal(
            diagnostic["clean_training"]["macro_f1"], history.iloc[epoch - 1].clean_train_macro_f1
        )
        equal(diagnostic["validation"]["macro_f1"], history.iloc[epoch - 1].validation_macro_f1)
    train = [d["clean_training"]["macro_f1"] * 100 for d in diagnostics]
    epochs = [d["epoch"] for d in diagnostics]
    axes[0, fold].plot(
        history.epoch, history.validation_macro_f1 * 100, color="#24788b", label="Validation"
    )
    axes[0, fold].plot(epochs, train, "o--", color="#ab6734", label="Clean training")
    axes[0, fold].set(title=f"Fold {fold}", ylim=(35, 100))
    axes[1, fold].plot(history.epoch, history.validation_loss, color="#24788b")
    axes[1, fold].plot(
        epochs, [d["clean_training"]["nll"] for d in diagnostics], "o--", color="#ab6734"
    )
    axes[1, fold].set(xlabel="Epoch", ylim=(0, 0.7))
    fold_rows.append(
        {
            **row,
            "best_observed_validation_epoch": int(
                history.loc[history.validation_macro_f1.idxmax(), "epoch"]
            ),
            "best_observed_validation_f1": float(history.validation_macro_f1.max()),
            "final_validation_loss": float(history.iloc[-1].validation_loss),
            "epoch20_gap": diagnostics[-2]["gap"],
        }
    )
axes[0, 0].set_ylabel("Macro-F1 (%)")
axes[1, 0].set_ylabel("Clean cross-entropy loss")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("SAM25 learning curves — fixed epoch 25; clean training checks at 10, 15, 20, 25")
fig.tight_layout()
fig.savefig(HERE / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
plot_cv_summary(report, HERE / "fold_summary.png")

old = read(HERE.parent / "task3_gender_sam25_result_20260906/incremental_comparison.json")
equal(report["scopes"]["screen_folds"]["metrics"], old["candidate"])
prefix = {}
for fold in (0, 4):
    previous = pd.read_csv(
        HERE.parent / f"task3_gender_sam25_result_20260906/fold{fold}/history.csv"
    )
    columns = [
        "learning_rate",
        "train_loss",
        "validation_loss",
        "validation_macro_f1",
        "sam_second_loss",
    ]
    prefix[fold] = all(np.array_equal(histories[fold][c], previous[c]) for c in columns)
assert all(prefix.values())

robust = pd.read_csv(HERE / "aggregate/robustness.csv")
assert len(robust) == 25 and not robust.duplicated(["validation_fold", "corruption"]).any()
for row in robust.itertuples():
    equal(row.macro_f1 - report["folds"][row.validation_fold]["validation_f1"], row.macro_f1_change)
equal(
    robust.groupby("corruption").macro_f1_change.mean().to_dict(),
    report["mean_induced_corruption_change"],
)

teacher = load_splits().set_index("id").loc[predictions.id, "gender"]
original = predictions.copy()
original["true_index"] = teacher.map({name: i for i, name in enumerate(classes)}).to_numpy()
teacher_metrics = oof_metrics(original, classes)
basis = pd.read_csv(HERE / "aggregate/label_basis_comparison.csv")
for fold in range(5):
    equal(
        oof_metrics(original.loc[original.cv_fold.eq(fold)], classes)["macro_f1"],
        basis.loc[
            basis.fold.eq(fold) & basis.view.eq("clean_validation"), "original_teacher_macro_f1"
        ].item(),
    )
summary = {
    "status": "saved_records_verified",
    "oof_rows": len(predictions),
    "folds": fold_rows,
    "pooled_metrics": report["scopes"]["all_five"]["metrics"],
    "original_teacher_label_metrics": teacher_metrics,
    "matching_screen_histories": prefix,
    "training_minutes": sum(r["train_seconds"] for r in report["folds"]) / 60,
    "peak_allocated_gpu_mb": max(r["peak_memory_bytes"] for r in report["folds"]) / 1e6,
    "mean_clean_training_f1": np.mean([r["train_f1"] for r in report["folds"]]),
    "fold_validation_f1_sample_std": np.std([r["validation_f1"] for r in report["folds"]], ddof=1),
    "errors": int(predictions.true_index.ne(predictions.predicted_index).sum()),
    "checkpoint_tensors_downloaded": False,
    "registry_reaudited": False,
    "new_inference_performed": False,
}
(HERE / "verified_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
pd.DataFrame(fold_rows).to_csv(HERE / "fold_review.csv", index=False)
print(
    json.dumps(
        {
            k: v
            for k, v in summary.items()
            if k not in {"folds", "pooled_metrics", "original_teacher_label_metrics"}
        },
        indent=2,
    )
)
print("Teacher-label pooled F1:", teacher_metrics["macro_f1"])
print(pd.DataFrame(fold_rows).to_string(index=False))
