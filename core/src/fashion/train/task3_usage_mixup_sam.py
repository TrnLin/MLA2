"""Fresh Usage models with MixUp + SAM, restricted to development folds 0 and 4."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.mixup import training_contract
from fashion.train.mixup import usage_policy as mixup_policy
from fashion.train.registry import RunRegistry
from fashion.train.task3_experiments import Task3ChildSpec

FOLDS = (0, 4)
NAME = "usage_expanded_v2_mixup_sam"
EXPERIMENT = "t3_usage_expanded_v2_mixup_sam"
ARTIFACT_DIRECTORY = Path("experiments") / EXPERIMENT
BASELINE_RUN_IDS = (
    "t3_usage_expanded_v2_e8_usage_smallcnn_f0_s2753_3c4369c6ce6e_20260906T141147Zfb606e",
    "t3_usage_expanded_v2_e8_usage_smallcnn_f1_s2753_3c4369c6ce6e_20260906T142131Zb5daae",
    "t3_usage_expanded_v2_e8_usage_smallcnn_f2_s2753_3c4369c6ce6e_20260906T143118Zfb9b48",
    "t3_usage_expanded_v2_e8_usage_smallcnn_f3_s2753_3c4369c6ce6e_20260906T144112Z005070",
    "t3_usage_expanded_v2_e8_usage_smallcnn_f4_s2753_3c4369c6ce6e_20260906T145103Z229e7e",
)


def _fields():
    fields = asdict(v2.expanded_usage_spec())
    fields.update(
        name=NAME,
        experiment_id=EXPERIMENT,
        hypothesis_id="t3_usage_mixup_sam_generalization",
        artifact_dir=str(ARTIFACT_DIRECTORY),
        run_prefix=EXPERIMENT,
        changed_factor="joint_mixup_alpha020_and_sam_rho005",
        parent_artifact_dir=str(v2.ARTIFACT_DIRECTORY),
        parent_run_ids=BASELINE_RUN_IDS,
    )
    return fields


class UsageMixUpSAMSpec(Task3ChildSpec):
    """Retain E8 and the v2 data; add only the two requested training methods."""

    def __post_init__(self):
        if asdict(self) != _fields():
            raise ValueError("Usage MixUp + SAM differs from the frozen screen recipe")

    def to_dict(self):
        from fashion.train.sam import usage_policy as sam_policy

        return {
            **asdict(self),
            "parent_run_ids": list(BASELINE_RUN_IDS),
            "dataset": v2.dataset_identity(),
            "folds": list(FOLDS),
            "initialization": "fresh_random_weights_for_each_fold",
            "reference_use": "comparison_only; no checkpoint weights are loaded",
            "mixup_policy": mixup_policy(),
            "sam_policy": sam_policy(),
        }


def screen_spec():
    return UsageMixUpSAMSpec(**_fields())


def screen_config(spec, *, fold, device_name):
    if isinstance(fold, bool) or fold not in FOLDS:
        raise ValueError("The Usage MixUp + SAM screen permits only folds 0 and 4")
    if not isinstance(spec, UsageMixUpSAMSpec) or spec.to_dict() != screen_spec().to_dict():
        raise ValueError("Usage MixUp + SAM differs from the frozen screen recipe")
    if device_name != "cuda":
        raise ValueError("Usage MixUp + SAM training requires a CUDA GPU")
    return Task3BaselineConfig(target="usage")


def usage_registry_path(output_root):
    return Path(output_root) / ARTIFACT_DIRECTORY / "usage/results/runs.csv"


def check_reference(*, directory, registry_path, splits):
    """Verify the two completed v2 controls without loading model weights."""
    rows = RunRegistry(registry_path)._read_rows()
    references = {}
    for fold in FOLDS:
        matches = [row for row in rows if row["run_id"] == BASELINE_RUN_IDS[fold]]
        if len(matches) != 1:
            raise ValueError(f"Expected one completed v2 baseline for fold {fold}")
        result = v2._verified_completed_fold(
            row=matches[0],
            path=Path(directory) / BASELINE_RUN_IDS[fold],
            splits=splits,
            fold=fold,
            spec=v2.expanded_usage_spec(),
            split_digest=v2.SPLIT_SHA256,
        )
        result["predictions"] = v2.read_predictions(result["prediction_path"])
        references[fold] = result
    return references


def verify_training_evidence(result, *, splits, fold, spec):
    """Check actual batch coverage, both gradient passes and the final-epoch history."""
    directory = Path(result["run_dir"])
    metrics = result["metrics"]
    config = json.loads((directory / "config.json").read_text())
    training, _ = v2.training_scope(splits, fold)
    expected_contract = training_contract(
        training, validation_fold=fold, seed=config["seed"], alpha=0.2, target="usage"
    )
    records = {}
    for filename, key in (
        ("mixup_training.json", "mixup_receipt_sha256"),
        ("sam_training.json", "sam_receipt_sha256"),
        ("clean_epoch_diagnostics.json", "clean_epoch_diagnostics_sha256"),
    ):
        if compute_sha256(directory / filename) != metrics.get(key):
            raise ValueError(f"Usage training receipt changed: {filename}")
        records[filename] = json.loads((directory / filename).read_text())
    mixed = records["mixup_training.json"]
    sam = records["sam_training.json"]
    policy = spec.to_dict()["sam_policy"]
    history = pd.read_csv(directory / "history.csv")
    epochs = config["epochs"]
    if (
        config.get("mixup_contract") != expected_contract
        or mixed.get("contract") != expected_contract
        or config.get("sam_policy") != policy
        or sam.get("policy") != policy
        or len(mixed.get("epochs", [])) != epochs
        or len(sam.get("epochs", [])) != epochs
        or history.epoch.tolist() != list(range(1, epochs + 1))
        or not history.train_macro_f1.isna().all()
        or not history.train_metric_scope.eq("not_applicable_mixed_inputs").all()
        or history.loc[history.selected_checkpoint, "epoch"].tolist() != [epochs]
        or [row["epoch"] for row in records["clean_epoch_diagnostics.json"]]
        != policy["diagnostic_epochs"]
    ):
        raise ValueError("Usage MixUp + SAM training contract or history differs")
    batches = math.ceil(len(training) / config["batch_size"])
    for index, (mix, step, (_, trace)) in enumerate(
        zip(mixed["epochs"], sam["epochs"], history.iterrows(), strict=True), start=1
    ):
        if (
            any(row.get("epoch") != index for row in (mix, step))
            or any(row.get("rows") != len(training) for row in (mix, step))
            or any(row.get("batches") != batches for row in (mix, step))
            or step.get("optimizer_steps") != batches
            or step.get("forward_backward_passes") != 2 * batches
            or not np.isclose(
                step.get("loss_denominator_sum", np.nan),
                np.dot(config["class_counts"], config["class_weights"]),
                rtol=1e-6,
            )
            or len(mix.get("mix_plan_sha256", "")) != 64
            or not 0 <= mix.get("lambda_sum", -1) <= batches
            or not 0 <= step.get("gradient_norm_min", -1) <= step.get("gradient_norm_max", -1)
            or not all(
                np.isfinite(step.get(key, np.nan))
                for key in ("first_loss", "second_loss", "gradient_norm_min", "gradient_norm_max")
            )
            or not np.isclose(step["first_loss"], trace.train_loss, rtol=0, atol=1e-10)
            or not np.isclose(step["second_loss"], trace.sam_second_loss, rtol=0, atol=1e-10)
        ):
            raise ValueError(f"Usage MixUp + SAM batch accounting differs at epoch {index}")


def completed_fold(*, fold, output_root, registry_path, splits, spec):
    rows = RunRegistry(registry_path)._read_rows()
    if any(row["experiment_id"] != EXPERIMENT for row in rows):
        raise ValueError("The Usage MixUp + SAM run log contains another experiment")
    matches = [
        row for row in rows if row["status"] == "complete" and int(row["validation_fold"]) == fold
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Multiple completed Usage screens exist for fold {fold}")
    row = matches[0]
    result = v2._verified_completed_fold(
        row=row,
        path=Path(output_root) / ARTIFACT_DIRECTORY / "usage" / row["run_id"],
        splits=splits,
        fold=fold,
        spec=spec,
        split_digest=v2.SPLIT_SHA256,
    )
    verify_training_evidence(result, splits=splits, fold=fold, spec=spec)
    return result


def build_comparison(results, *, splits, references, contract):
    folds = [int(run["metrics"]["validation_fold"]) for run in results]
    if folds != list(FOLDS) or set(references) != set(FOLDS):
        raise ValueError("Screen comparison requires exactly folds 0 and 4")
    predictions = pd.concat(
        [v2.read_predictions(run["prediction_path"]) for run in results], ignore_index=True
    )
    expected = pd.concat([v2.training_scope(splits, fold)[1] for fold in FOLDS])
    predictions = v2.source_predictions(predictions, expected)
    reference = pd.concat([references[fold]["predictions"] for fold in FOLDS], ignore_index=True)
    reference = v2.source_predictions(reference, expected)
    scores = v2.source_metrics(predictions)
    baseline = v2.source_metrics(reference)
    teacher = scores["teacher"]["metrics"]
    old = baseline["teacher"]["metrics"]
    rows = []
    for run in results:
        fold = run["metrics"]["validation_fold"]
        sources = run["metrics"]["source_metrics"]
        val = sources["validation"]["teacher"]["metrics"]["macro_f1"]
        train = sources["clean_training"]["teacher"]["metrics"]["macro_f1"]
        control = references[fold]["metrics"]["source_metrics"]["validation"]["teacher"]["metrics"]
        rows.append(
            dict(
                fold=fold,
                baseline_teacher_f1=control["macro_f1"],
                candidate_teacher_f1=val,
                teacher_f1_change=val - control["macro_f1"],
                clean_teacher_train_f1=train,
                clean_teacher_gap=train - val,
            )
        )
    per_class = [
        {
            "class_name": current["class_name"],
            "support": current["support"],
            "baseline_f1": prior["f1"],
            "candidate_f1": current["f1"],
            "baseline_recall": prior["recall"],
            "candidate_recall": current["recall"],
        }
        for current, prior in zip(teacher["per_class"], old["per_class"], strict=True)
    ]
    return {
        "folds": list(FOLDS),
        "dataset": contract,
        "primary_scope": "same teacher validation images, folds 0 and 4, all nine classes",
        "sources": scores,
        "baseline_sources": baseline,
        "teacher_macro_f1_change": teacher["macro_f1"] - old["macro_f1"],
        "fold_comparison": rows,
        "teacher_per_class": per_class,
        "limits": [
            "A two-fold development screen; no automatic model promotion or five-fold training.",
            "MixUp and SAM change together, so their separate effects are not isolated.",
            "A smaller training gap alone is not evidence of better validation predictions.",
            "External-source scores are diagnostics, not the primary teacher score.",
            "Rare teacher classes have few examples; earlier holdout/test results already exist.",
        ],
    }


def run_usage_mixup_sam(
    *,
    output_root,
    baseline_directory,
    baseline_registry_path,
    root=ROOT,
    registry_mirrors=(),
    folds=FOLDS,
):
    """Train exactly two fresh models, or reuse their fully verified completed results."""
    import torch

    from fashion.train.task3_baseline import _aggregate_target, run_task3_baseline_fold

    if tuple(folds) != FOLDS or any(isinstance(fold, bool) for fold in folds):
        raise ValueError("This screen trains exactly folds 0 and 4, in that order")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before running Usage MixUp + SAM")
    root, output_root = Path(root), Path(output_root)
    spec = screen_spec()
    screen_config(spec, fold=0, device_name="cuda")
    registry_path = usage_registry_path(output_root)
    protected_logs = {
        Path(baseline_registry_path).resolve(),
        (output_root / "results/runs.csv").resolve(),
        (root / "results/runs.csv").resolve(),
        registry_path.resolve(),
    }
    for mirror in registry_mirrors:
        if Path(mirror).resolve() in protected_logs:
            raise ValueError("Use a separate screen-only local results/runs.csv mirror")
    for path in (registry_path, *registry_mirrors):
        if any(row["experiment_id"] != EXPERIMENT for row in RunRegistry(path)._read_rows()):
            raise ValueError("A Usage screen run log cannot overwrite another experiment")
    splits, contract = v2.validate_dataset(root=root, check_images=True)
    references = check_reference(
        directory=baseline_directory, registry_path=baseline_registry_path, splits=splits
    )
    experiment_dir = output_root / ARTIFACT_DIRECTORY / "usage"
    v2.write_json(
        {
            "dataset": contract,
            "recipe": spec.to_dict(),
            "baseline": Task3BaselineConfig(target="usage").to_dict(),
            "references": {str(fold): references[fold]["run_id"] for fold in FOLDS},
        },
        experiment_dir / "prerequisites.json",
    )
    results = []
    for fold in FOLDS:
        result = completed_fold(
            fold=fold,
            output_root=output_root,
            registry_path=registry_path,
            splits=splits,
            spec=spec,
        )
        if result is None:
            print(f"Usage MixUp + SAM: fresh weights, fold {fold}, 30 epochs", flush=True)
            run_task3_baseline_fold(
                "usage",
                fold,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                child_spec=spec,
                parent_run_directory=Path(baseline_directory) / BASELINE_RUN_IDS[fold],
            )
            result = completed_fold(
                fold=fold,
                output_root=output_root,
                registry_path=registry_path,
                splits=splits,
                spec=spec,
            )
            if result is None:
                raise RuntimeError(f"Usage fold {fold} did not register a complete result")
        else:
            print(f"Reusing verified completed fold {fold}: {result['run_id']}", flush=True)
        results.append(result)
        torch.cuda.empty_cache()
    comparison = build_comparison(results, splits=splits, references=references, contract=contract)
    aggregate = _aggregate_target(
        "usage",
        results,
        output_root=output_root,
        root=root,
        artifact_dir=spec.artifact_dir,
        experiment_id=EXPERIMENT,
        hypothesis_id=spec.hypothesis_id,
        child_spec=spec,
        model_family=spec.model_family,
        parameter_count=baseline_parameter_count("usage"),
        aggregate_dir_name="aggregate_folds_0_4",
    )
    destination = experiment_dir / "aggregate_folds_0_4"
    v2.write_json(comparison, destination / "teacher_comparison.json")
    for key in ("fold_comparison", "teacher_per_class"):
        pd.DataFrame(comparison[key]).to_csv(destination / f"{key}.csv", index=False)
    return {
        "fold_results": results,
        "aggregate": aggregate,
        "comparison": comparison,
        "comparison_path": str(destination / "teacher_comparison.json"),
        "registry_path": str(registry_path),
    }


def save_learning_curves(result, *, path):
    """Display both saved folds and keep mixed training losses clearly labelled."""
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, run in enumerate(result["fold_results"]):
        fold = run["metrics"]["validation_fold"]
        history = pd.read_csv(Path(run["run_dir"]) / "history.csv")
        axes[0, column].plot(history.epoch, history.validation_macro_f1, color="#176b87")
        axes[0, column].set(title=f"Fold {fold}: combined validation", ylabel="Nine-class macro-F1")
        axes[0, column].set_ylim(0, 1)
        for name, label in (
            ("train_loss", "Mixed train: first pass"),
            ("sam_second_loss", "Mixed train: SAM pass"),
            ("validation_loss", "Unmixed validation"),
        ):
            axes[1, column].plot(history.epoch, history[name], label=label, linewidth=1.5)
        axes[1, column].set(ylabel="Class-weighted cross-entropy", xlabel="Epoch")
        axes[1, column].legend(fontsize=9)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Usage MixUp + SAM — fresh weights, folds 0 and 4")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--baseline-registry", type=Path, required=True)
    args = parser.parse_args()
    result = run_usage_mixup_sam(
        root=args.root,
        output_root=args.output_root,
        baseline_directory=args.baseline_directory,
        baseline_registry_path=args.baseline_registry,
    )
    print(result["comparison_path"])


if __name__ == "__main__":
    main()
