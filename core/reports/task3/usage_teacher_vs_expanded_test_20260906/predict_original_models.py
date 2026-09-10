"""Choose two teacher-only references from validation, then freeze test predictions."""

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
from fashion.train.task3_decisions import oof_metrics, probability_columns, validate_oof

REPORT = Path(__file__).resolve().parent
PRIOR = ROOT / "reports/task3/usage_expanded_e8_test_20260906"
EVIDENCE = ROOT / "results/evidence/task3"
REGISTRIES = [ROOT / "results/runs.csv", EVIDENCE / "results/runs.csv"]
TEACHER_SPLIT = ROOT / "data/processed/splits.csv"
LABEL_MAP = ROOT / "data/processed/label_maps.json"
MANIFEST = ROOT / "data/processed/prediction_manifest.csv"
TEMPLATE = ROOT / "data/raw/teacher/test/styles_prediction.csv"
EXPERIMENTS = {
    "E1": "t3_primary_baseline_smallcnn",
    "E2": "t3_usage_class_balanced_smallcnn",
    "E3": "t3_usage_classifier_dropout_smallcnn",
    "E4": "t3_usage_tinyresnet18_pm",
    "E5": "t3_usage_label_smoothing_smallcnn",
    "E6": "t3_usage_focal_gamma1_smallcnn",
    "E7": "t3_usage_tinyconvnext18",
    "E8": "t3_usage_translation_2px_smallcnn",
    "E9": "t3_usage_exception_balance_smallcnn",
}

# Reuse the exact unlabelled image and probability-writing contract of the first test run.
spec = importlib.util.spec_from_file_location("expanded_inference", PRIOR / "predict_test.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
CLASSES = shared.CLASSES


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def local_artifact(recorded_path):
    return EVIDENCE / recorded_path.split("/MLA2/task3/", 1)[1]


def choose_from_validation():
    records = pd.concat(
        [pd.read_csv(path, keep_default_na=False) for path in REGISTRIES], ignore_index=True
    )
    records = records.loc[
        records.target.eq("usage") & records.experiment_id.isin(EXPERIMENTS.values())
    ]
    for _, duplicate_rows in records.groupby("run_id"):
        for column in ("metrics_json", "checkpoint_sha256", "prediction_sha256", "split_digest"):
            assert duplicate_rows[column].nunique() == 1, column
    records = records.drop_duplicates("run_id")
    expected = load_splits(TEACHER_SPLIT)
    expected = expected.loc[expected.partition.eq("development") & expected.usage.ne("")]
    ranking = []
    for name, experiment in EXPERIMENTS.items():
        rows = records.loc[records.experiment_id.eq(experiment)].sort_values("validation_fold")
        assert rows.validation_fold.astype(int).tolist() == list(range(5)), name
        assert rows.status.eq("complete").all()
        assert rows.scratch.astype(str).str.lower().eq("true").all()
        assert rows.submission_eligible.astype(str).str.lower().eq("true").all()
        assert rows.debug.astype(str).str.lower().eq("false").all()
        assert rows.split_digest.eq(compute_sha256(TEACHER_SPLIT)).all()
        assert rows.label_map_digest.eq(compute_sha256(LABEL_MAP)).all()
        matrix = np.zeros((9, 9), dtype=np.int64)
        for row in rows.itertuples():
            metrics = json.loads(row.metrics_json)
            fold_matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            fold_expected = expected.loc[expected.cv_fold.eq(int(row.validation_fold))]
            assert fold_matrix.shape == (9, 9)
            np.testing.assert_array_equal(
                fold_matrix.sum(axis=1), [int(fold_expected.usage.eq(c).sum()) for c in CLASSES]
            )
            matrix += fold_matrix
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        f1 = np.divide(
            2 * np.diag(matrix), denominator, out=np.zeros(9, dtype=float), where=denominator > 0
        )
        ranking.append(
            {
                "model": name,
                "experiment_id": experiment,
                "folds": "0,1,2,3,4",
                "validation_rows": int(matrix.sum()),
                "correct": int(np.trace(matrix)),
                "accuracy": float(np.trace(matrix) / matrix.sum()),
                "macro_f1": float(f1.mean()),
            }
        )
    ranking = pd.DataFrame(ranking)
    ranking.to_csv(REPORT / "validation_ranking.csv", index=False)
    accuracy_winner = ranking.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
    f1_winner = ranking.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    assert accuracy_winner.model == "E1" and f1_winner.model == "E8"
    selection = {
        "declared_at_utc": now(),
        "scope": "Completed individual teacher-only experiments with all five canonical folds",
        "validation_population": len(expected),
        "accuracy_reference": accuracy_winner.to_dict(),
        "macro_f1_reference": f1_winner.to_dict(),
        "incomplete_screens": "S1, S2, U1, U2 and U3 have only folds 0 and 4",
        "diagnostic_exclusions": (
            "EDA probes use a smaller selected population; the previously declined E2/E3/E8 "
            "blend is a multi-experiment diagnostic, not an individual experiment"
        ),
        "test_labels_used_for_selection": False,
        "test_labels_were_seen_in_previous_expanded_model_comparison": True,
        "inference_rule": "Equal mean of the five saved fold softmax vectors, then argmax",
        "training_or_tuning": False,
        "source_hashes": {str(p.relative_to(ROOT)): compute_sha256(p) for p in REGISTRIES},
    }
    save_json(selection, REPORT / "selection_contract.json")
    print(ranking.sort_values("macro_f1", ascending=False).to_string(index=False), flush=True)
    return records, expected, [accuracy_winner.model, f1_winner.model]


def main():
    torch.set_num_threads(2)
    if (REPORT / "prediction_freeze.json").exists():
        raise RuntimeError("Original-model test predictions are already frozen.")
    records, expected, chosen = choose_from_validation()
    template = pd.read_csv(TEMPLATE, usecols=["id"], keep_default_na=False)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    assert template.id.is_unique and manifest.id.is_unique
    assert set(template.id) == set(manifest.id)
    assert not set(template.id).intersection(pd.read_csv(TEACHER_SPLIT, usecols=["id"]).id)
    manifest = manifest.set_index("id").loc[template.id].reset_index()
    for row in manifest.itertuples():
        assert compute_sha256(resolve_task3_path(row.path, root=ROOT)) == row.sha256

    recipe = {
        "declared_at_utc": now(),
        "test_rows": len(manifest),
        "class_names": list(CLASSES),
        "models": chosen,
        "checkpoints": [],
        "reproduction_checks": [],
        "preprocessing": "Same full RGB 80x60 transform; each checkpoint's saved normalization",
        "aggregation": "Equal arithmetic mean of all five saved fold probabilities; argmax",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "test_labels_used_for_inference": False,
        "inputs": {
            str(path.relative_to(ROOT)): compute_sha256(path)
            for path in (
                TEACHER_SPLIT,
                LABEL_MAP,
                MANIFEST,
                TEMPLATE,
                Path(__file__),
                PRIOR / "predict_test.py",
                REPORT / "selection_contract.json",
                ROOT / "src/fashion/data/images.py",
                ROOT / "src/fashion/train/model.py",
                ROOT / "src/fashion/train/config.py",
            )
        },
    }
    output_paths = [REPORT / "validation_ranking.csv", REPORT / "selection_contract.json"]
    for name in chosen:
        rows = records.loc[records.experiment_id.eq(EXPERIMENTS[name])].sort_values(
            "validation_fold"
        )
        model_outputs = []
        output = REPORT / name
        output.mkdir(exist_ok=True)
        for row in rows.itertuples():
            checkpoint_path = local_artifact(row.checkpoint_path)
            prediction_path = local_artifact(row.prediction_path)
            assert compute_sha256(checkpoint_path) == row.checkpoint_sha256
            assert compute_sha256(prediction_path) == row.prediction_sha256
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert checkpoint["run_id"] == row.run_id
            assert checkpoint["class_names"] == list(CLASSES)
            config = checkpoint["config"]
            assert "expanded_dataset" not in config
            assert config.get("input_view", "full") == "full"
            assert config["epochs"] == 30 and config["checkpoint_rule"] == "final_epoch"
            config_kwargs = {
                field.name: config[field.name] for field in fields(Task3BaselineConfig)
            }
            config_kwargs["channels"] = tuple(config_kwargs["channels"])
            model = Task3BaselineCNN(Task3BaselineConfig(**config_kwargs))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()

            fold_expected = expected.loc[expected.cv_fold.eq(int(row.validation_fold))]
            saved = validate_oof(
                pd.read_csv(prediction_path, keep_default_na=False, float_precision="round_trip"),
                fold_expected,
                target="usage",
                classes=CLASSES,
                run_ids_by_fold={int(row.validation_fold): row.run_id},
                allow_legacy_na=True,
            )
            measured = oof_metrics(saved, CLASSES)
            registered = json.loads(row.metrics_json)
            for key in ("accuracy", "macro_f1"):
                np.testing.assert_allclose(measured[key], registered[key], atol=1e-12)

            controls = fold_expected.groupby("usage", sort=True).head(2)
            control_data = shared.UnlabelledTestImages(controls, checkpoint["normalization"])
            with torch.inference_mode():
                control_probabilities = torch.softmax(
                    model(torch.stack([control_data[i] for i in range(len(control_data))])), dim=1
                ).numpy()
            prior = saved.set_index("id").loc[controls.id]
            drift = float(
                np.abs(control_probabilities - prior[probability_columns(CLASSES)].to_numpy()).max()
            )
            assert drift < 0.005
            assert np.array_equal(control_probabilities.argmax(axis=1), prior.predicted_index)
            recipe["reproduction_checks"].append(
                {
                    "model": name,
                    "fold": int(row.validation_fold),
                    "rows": len(controls),
                    "maximum_probability_drift_cpu_vs_saved_gpu": drift,
                    "all_class_decisions_match": True,
                }
            )
            recipe["checkpoints"].append(
                {
                    "model": name,
                    "fold": int(row.validation_fold),
                    "run_id": row.run_id,
                    "path": str(checkpoint_path.relative_to(ROOT)),
                    "sha256": row.checkpoint_sha256,
                    "normalization": checkpoint["normalization"],
                }
            )
            save_json(recipe, REPORT / "inference_recipe.json")

            dataset = shared.UnlabelledTestImages(manifest, checkpoint["normalization"])
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            batches = []
            with torch.inference_mode():
                for images in loader:
                    batches.append(torch.softmax(model(images), dim=1).numpy())
            probabilities = np.concatenate(batches)
            path = output / f"usage_test_fold_{int(row.validation_fold)}.csv"
            shared.prediction_frame(manifest.id, probabilities).to_csv(path, index=False)
            output_paths.append(path)
            model_outputs.append(probabilities.astype(np.float64))
            print(
                f"{name} fold {row.validation_fold}: {len(probabilities):,} predictions saved",
                flush=True,
            )
        mean_probabilities = np.mean(np.stack(model_outputs), axis=0)
        result = shared.prediction_frame(manifest.id, mean_probabilities)
        path = output / "usage_test_probabilities.csv"
        result.to_csv(path, index=False)
        output_paths.append(path)
        path = output / "usage_test_predictions.csv"
        result[["id", "usage"]].to_csv(path, index=False)
        output_paths.append(path)

    output_paths.append(REPORT / "inference_recipe.json")
    freeze = {
        "frozen_at_utc": now(),
        "test_rows": len(manifest),
        "reference_labels_opened_by_this_inference_script": False,
        "test_labels_were_seen_in_previous_expanded_model_comparison": True,
        "outputs": {str(p.relative_to(REPORT)): compute_sha256(p) for p in output_paths},
    }
    save_json(freeze, REPORT / "prediction_freeze.json")
    print("Both original-model predictions are saved and frozen; ready to score.", flush=True)


if __name__ == "__main__":
    main()
