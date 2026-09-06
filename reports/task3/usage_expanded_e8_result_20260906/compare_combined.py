"""Evaluate frozen E8 checkpoints on added validation images for a fair comparison."""

from fashion.task3_paths import resolve_task3_path

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.data import Task3ImageDataset
from fashion.train.model import Task3BaselineCNN
from fashion.train.task3_baseline import _pass, _prediction_frame
from fashion.train.task3_decisions import oof_metrics, probability_columns, validate_oof
from fashion.train.task3_usage_expanded import (
    CLASSES,
    E8_DIRECTORY,
    check_e8_sources,
    read_predictions,
    source_predictions,
    training_scope,
    validate_dataset,
    write_json,
)

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/usage_expanded_e8_20260906"


def infer(directory, expected):
    checkpoint = torch.load(directory / "final_epoch.pt", map_location="cpu", weights_only=True)
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 30
    assert checkpoint["class_names"] == list(CLASSES)
    model = Task3BaselineCNN(Task3BaselineConfig(target="usage"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    stats = checkpoint["normalization"]
    dataset = Task3ImageDataset(
        expected,
        target="usage",
        label_to_index=dict(zip(CLASSES, range(len(CLASSES)), strict=True)),
        mean=stats["mean"],
        std=stats["std"],
        root=ROOT,
        augmentation="none",
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    _, labels, probabilities, trace = _pass(
        model, loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    predictions = _prediction_frame(labels, probabilities, trace, CLASSES, checkpoint["run_id"])
    return source_predictions(predictions, expected)


def check_reproduced(actual, saved, label):
    actual = actual.sort_values("id").reset_index(drop=True)
    saved = saved.loc[saved.id.isin(actual.id)].sort_values("id").reset_index(drop=True)
    assert actual.id.tolist() == saved.id.tolist()
    columns = probability_columns(CLASSES)
    difference = np.abs(actual[columns].to_numpy() - saved[columns].to_numpy())
    # The saved predictions used CUDA, while this small review uses CPU kernels.
    # Require the same class decisions and bound probability drift to 0.5 points.
    assert float(difference.max()) < 0.005, (label, float(difference.max()))
    assert actual.predicted_index.tolist() == saved.predicted_index.tolist(), label
    return {
        "scope": label,
        "rows": len(actual),
        "max_probability_difference": float(difference.max()),
    }


def main():
    torch.set_num_threads(2)
    splits, _ = validate_dataset(check_images=False)
    parents = check_e8_sources(
        directory=ROOT / "results/evidence/task3" / E8_DIRECTORY,
        registry_path=ROOT / "results/runs.csv",
    )
    runs = pd.read_csv(EVIDENCE / "results/runs.csv", keep_default_na=False).sort_values(
        "validation_fold"
    )
    expanded = read_predictions(EVIDENCE / "aggregate/oof_predictions.csv")
    additions, expected_rows, controls, fold_table = [], [], [], []
    for fold in range(5):
        _, validation = training_scope(splits, fold)
        added = validation.loc[validation.source_dataset.ne("teacher")].copy()
        for row in added.itertuples():
            assert compute_sha256(resolve_task3_path(row.path, root=ROOT)) == row.sha256
        old_directory = ROOT / "results/evidence/task3" / E8_DIRECTORY / parents[fold]["run_id"]
        old_added = infer(old_directory, added)
        additions.append(old_added)
        expected_rows.append(validation)
        control = (
            validation.loc[validation.source_dataset.eq("teacher")]
            .groupby("usage", sort=False)
            .head(2)
        )
        controls.append(
            check_reproduced(
                infer(old_directory, control),
                parents[fold]["predictions"],
                f"E8 fold {fold} teacher control",
            )
        )
        new_directory = EVIDENCE / runs.loc[runs.validation_fold.eq(fold), "run_id"].iloc[0]
        controls.append(
            check_reproduced(
                infer(new_directory, added), expanded, f"expanded fold {fold} added images"
            )
        )
        parent_teacher = parents[fold]["predictions"].copy()
        parent_teacher["source_dataset"] = "teacher"
        parent_fold = pd.concat([parent_teacher, old_added], ignore_index=True)
        candidate_fold = expanded.loc[expanded.cv_fold.eq(fold)]
        assert set(parent_fold.id) == set(candidate_fold.id)
        old_score = oof_metrics(parent_fold, CLASSES)["macro_f1"]
        new_score = oof_metrics(candidate_fold, CLASSES)["macro_f1"]
        fold_table.append(
            {
                "fold": fold,
                "e8_combined_f1": old_score,
                "expanded_combined_f1": new_score,
                "change": new_score - old_score,
            }
        )
        print(
            f"Fold {fold}: identical combined validation images; "
            f"E8 {old_score:.4f} -> expanded {new_score:.4f}",
            flush=True,
        )
    teacher = pd.concat([parents[fold]["predictions"] for fold in range(5)], ignore_index=True)
    teacher["source_dataset"] = "teacher"
    old_added = pd.concat(additions, ignore_index=True)
    parent_combined = validate_oof(
        pd.concat([teacher, old_added], ignore_index=True),
        pd.concat(expected_rows, ignore_index=True),
        target="usage",
        classes=CLASSES,
        run_ids_by_fold={fold: parents[fold]["run_id"] for fold in range(5)},
    )
    assert set(parent_combined.id) == set(expanded.id)
    old_metrics = oof_metrics(parent_combined, CLASSES)
    new_metrics = oof_metrics(expanded, CLASSES)
    old_added.to_csv(REPORT / "e8_added_validation_predictions.csv", index=False)
    pd.DataFrame(fold_table).to_csv(REPORT / "combined_fold_comparison.csv", index=False)
    summary = {
        "purpose": "Secondary added-source diagnostic; not a model-selection criterion",
        "project_goal": "Good predictions on the teacher's unseen test set",
        "selection_scope": "Use original teacher validation for model selection",
        "same_combined_validation_rows": len(parent_combined),
        "e8_combined": old_metrics,
        "expanded_combined": new_metrics,
        "combined_macro_f1_change": new_metrics["macro_f1"] - old_metrics["macro_f1"],
        "e8_added": oof_metrics(old_added, CLASSES),
        "expanded_added": oof_metrics(expanded.loc[expanded.source_dataset.ne("teacher")], CLASSES),
        "combined_fold_wins": sum(row["change"] > 0 for row in fold_table),
        "controls": controls,
        "method": (
            "Frozen E8 fold checkpoints infer only their assigned added validation images. "
            "Existing verified teacher OOF predictions are reused. "
            "No training, tuning, normalization refit or holdout access."
        ),
        "cpu_cuda_reproduction_checks": {
            "maximum_allowed_absolute_probability_difference": 0.005,
            "identical_class_decisions_required": True,
        },
    }
    write_json(summary, REPORT / "combined_comparison.json")
    print(
        json.dumps(
            {
                "rows": len(parent_combined),
                "e8_combined_f1": old_metrics["macro_f1"],
                "expanded_combined_f1": new_metrics["macro_f1"],
                "change": summary["combined_macro_f1_change"],
                "combined_fold_wins": summary["combined_fold_wins"],
                "old_added_correct": int(
                    np.trace(np.array(summary["e8_added"]["confusion_matrix"]))
                ),
                "new_added_correct": int(
                    np.trace(np.array(summary["expanded_added"]["confusion_matrix"]))
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
