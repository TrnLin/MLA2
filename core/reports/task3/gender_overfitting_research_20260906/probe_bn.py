"""A fixed training-only normalization probe of the saved scratch MixUp models."""

import copy
import json
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.data import Task3ImageDataset
from fashion.train.metrics import classification_metrics
from fashion.train.model import Task3GeM3CNN
from fashion.train.registry import RunRegistry
from fashion.train.task3_gender_diagnostic import verify_checkpoint_metadata, verify_input_images

OUTPUT = ROOT / "reports/task3/gender_overfitting_research_20260906"
SOURCE = ROOT / "reports/task3/gender_mixup_result_20260906"
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
MAPPING = {name: i for i, name in enumerate(CLASSES)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    torch.set_num_threads(6)
    torch.manual_seed(2753)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((OUTPUT / "probe_plan.json").read_text())
    splits = load_gender_name_truth_variant(ROOT)
    source_registry = pd.read_csv(SOURCE / "runs.csv", keep_default_na=False)
    registry = RunRegistry(ROOT / "results/runs.csv")
    verify_input_images(splits.loc[splits.partition.eq("development")], ROOT)
    summaries, predictions, buffer_changes = [], [], []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    for fold in plan["folds"]:
        source = next((SOURCE / "saved").glob(f"t3_gender_name_truth_mixup_alpha020_*_f{fold}_*"))
        source_record = source_registry.loc[source_registry.run_id.eq(source.name)].iloc[0]
        checkpoint_path = source / "final_epoch.pt"
        checkpoint_hash = compute_sha256(checkpoint_path)
        assert checkpoint_hash == source_record.checkpoint_sha256
        payload = json.loads((source / "config.json").read_text())
        stats = json.loads((source / "normalization.json").read_text())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        verify_checkpoint_metadata(
            checkpoint, run_id=source.name, config=payload, classes=CLASSES, normalization=stats
        )
        config_values = {f.name: payload[f.name] for f in fields(Task3BaselineConfig)}
        config_values["channels"] = tuple(config_values["channels"])
        model = Task3GeM3CNN(Task3BaselineConfig(**config_values), classifier_dropout=0.3)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().requires_grad_(False)
        original = copy.deepcopy(model)
        training, validation = [get_samples(f, target="gender") for f in get_cv_split(splits, fold)]
        assert not set(training.id) & set(validation.id)
        assert not set(training.product_family_group) & set(validation.product_family_group)
        assert (
            training.partition.eq("development").all()
            and validation.partition.eq("development").all()
        )

        def loader(frame, shuffle=False):
            dataset = Task3ImageDataset(
                frame,
                target="gender",
                label_to_index=MAPPING,
                mean=stats["mean"],
                std=stats["std"],
                root=ROOT,
                image_size=(80, 60),
            )
            return DataLoader(
                dataset,
                batch_size=128,
                shuffle=shuffle,
                num_workers=2,
                generator=torch.Generator().manual_seed(2753),
            )

        run_id = f"t3_gender_mixup020_clean_bn_probe_f{fold}_{timestamp}"
        destination = OUTPUT / run_id
        destination.mkdir(exist_ok=False)
        record = dict(
            plan,
            source_run_id=source.name,
            source_checkpoint_sha256=checkpoint_hash,
            source_code_root=str(Path(__import__("fashion").__file__).parent),
            probe_script_sha256=compute_sha256(Path(__file__)),
            training_rows=len(training),
            validation_rows=len(validation),
        )
        write_json(destination / "config.json", record)
        registry.start(
            {
                "run_id": run_id,
                "experiment_id": "t3_gender_mixup020_clean_bn_probe",
                "hypothesis_id": "clean_batchnorm_statistics_mismatch",
                "parent_run_ids": [source.name],
                "task": 3,
                "target": "gender",
                "validation_fold": fold,
                "seed": 2753,
                "debug": True,
                "scratch": True,
                "submission_eligible": False,
                "config_path": str(destination / "config.json"),
                "config_hash": compute_sha256(destination / "config.json"),
                "split_digest": compute_sha256(ROOT / "data/processed/splits.csv"),
                "training_product_count": len(training),
                "validation_product_count": len(validation),
                "training_family_count": training.product_family_group.nunique(),
                "validation_family_count": validation.product_family_group.nunique(),
                "model_family": "task3_small_cnn_gem_p3_clean_bn_probe",
                "parameter_count": 390181,
                "environment_json": {"torch": torch.__version__, "device": "cpu", "threads": 6},
                "last_completed_stage": "registered_before_statistics_fit",
            }
        )
        started = time.perf_counter()
        try:
            bns = {
                name: layer
                for name, layer in model.named_modules()
                if isinstance(layer, torch.nn.BatchNorm2d)
            }
            assert len(bns) == 4
            for bn in bns.values():
                bn.reset_running_stats()
                bn.train()
            seen = []
            with torch.inference_mode():
                for index, batch in enumerate(loader(training, shuffle=True)):
                    count = len(batch["id"])
                    for bn in bns.values():
                        bn.momentum = count / (len(seen) + count)
                    model(batch["image"])
                    seen.extend(batch["id"].tolist())
                    if index % 50 == 0:
                        print(f"fold {fold}: clean BN pass {len(seen)}/{len(training)}", flush=True)
            assert len(seen) == len(set(seen)) == len(training)
            assert set(seen) == set(training.id)
            calibration_seconds = time.perf_counter() - started
            for name, parameter in model.named_parameters():
                assert torch.equal(parameter, dict(original.named_parameters())[name])
            for name, bn in bns.items():
                old = dict(original.named_modules())[name]
                buffer_changes.append(
                    {
                        "fold": fold,
                        "layer": name,
                        "median_variance_ratio": float(
                            torch.median(bn.running_var / old.running_var)
                        ),
                        "mean_shift_in_old_std": float(
                            (
                                (bn.running_mean - old.running_mean).abs() / old.running_var.sqrt()
                            ).mean()
                        ),
                    }
                )
            model.eval()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "diagnostic_only": True,
                    "source_run_id": source.name,
                    "plan": plan,
                },
                destination / "bn_probe.pt",
            )
            fold_results = {}
            for scope, frame in (("train", training), ("validation", validation)):
                arrays = {"original_cpu": [], "clean_bn_cpu": []}
                ids, labels = [], []
                with torch.inference_mode():
                    for index, batch in enumerate(loader(frame)):
                        for name, net in (("original_cpu", original), ("clean_bn_cpu", model)):
                            arrays[name].append(torch.softmax(net(batch["image"]), 1).numpy())
                        ids.extend(batch["id"].tolist())
                        labels.extend(batch["label"].tolist())
                        if index % 75 == 0:
                            print(
                                f"fold {fold}: {scope} paired replay {len(ids)}/{len(frame)}",
                                flush=True,
                            )
                for name, values in arrays.items():
                    probabilities = np.concatenate(values)
                    score = classification_metrics(np.array(labels), probabilities, CLASSES)
                    fold_results[f"{name}.{scope}"] = score
                    summaries.append(
                        {"fold": fold, "model": name, "scope": scope, "macro_f1": score["macro_f1"]}
                    )
                    rows = pd.DataFrame({"id": ids, "true_index": labels})
                    rows["predicted_index"] = probabilities.argmax(1)
                    rows["fold"] = fold
                    rows["model"] = name
                    rows["scope"] = scope
                    for col, class_name in enumerate(CLASSES):
                        rows[f"probability_{col}_{class_name}"] = probabilities[:, col].astype(
                            np.float64
                        )
                    rows.to_csv(destination / f"{name}_{scope}.csv", index=False)
                    if scope == "validation":
                        predictions.append(rows)
                saved_name = (
                    "clean_train_predictions.csv" if scope == "train" else "oof_predictions.csv"
                )
                saved_path = SOURCE / "saved/comparison_name_truth_ieee" / source.name / saved_name
                saved = pd.read_csv(saved_path).set_index("id").loc[ids]
                assert np.array_equal(saved.true_index, labels)
                cpu_predictions = np.concatenate(arrays["original_cpu"]).argmax(1)
                fold_results[f"{scope}_cpu_vs_saved_prediction_changes"] = int(
                    np.count_nonzero(cpu_predictions != saved.predicted_index)
                )
                saved_probabilities = saved[
                    [f"probability_{j}_{c}" for j, c in enumerate(CLASSES)]
                ].to_numpy()
                fold_results[f"{scope}_cpu_vs_saved_max_probability_delta"] = float(
                    np.max(np.abs(np.concatenate(arrays["original_cpu"]) - saved_probabilities))
                )
            fold_results["calibration_seconds"] = calibration_seconds
            fold_results["diagnostic_total_seconds"] = time.perf_counter() - started
            write_json(destination / "metrics.json", fold_results)
            assert compute_sha256(checkpoint_path) == checkpoint_hash
            registry.complete(
                run_id,
                {
                    "checkpoint_path": str(destination / "bn_probe.pt"),
                    "checkpoint_sha256": compute_sha256(destination / "bn_probe.pt"),
                    "prediction_path": str(destination / "clean_bn_cpu_validation.csv"),
                    "prediction_sha256": compute_sha256(
                        destination / "clean_bn_cpu_validation.csv"
                    ),
                    "metrics_json": fold_results,
                    "train_seconds": calibration_seconds,
                    "last_completed_stage": "clean_bn_fit_and_paired_cpu_replay_complete",
                },
            )
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seconds": time.perf_counter() - started,
                        "scores": summaries[-4:],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            registry.fail(run_id, error, last_completed_stage="normalization_probe")
            raise
    pd.DataFrame(summaries).to_csv(OUTPUT / "bn_probe_scores.csv", index=False)
    pd.DataFrame(buffer_changes).to_csv(OUTPUT / "bn_buffer_changes.csv", index=False)
    oof = pd.concat(predictions)
    pooled = {}
    for name, rows in oof.groupby("model"):
        pooled[name] = classification_metrics(
            rows.true_index.to_numpy(),
            rows[[f"probability_{j}_{c}" for j, c in enumerate(CLASSES)]].to_numpy(),
            CLASSES,
        )
    write_json(OUTPUT / "bn_probe_pooled.json", pooled)
    print(json.dumps({name: score["macro_f1"] for name, score in pooled.items()}), flush=True)


if __name__ == "__main__":
    main()
