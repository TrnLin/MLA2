"""Repeat the Usage MixUp + SAM screen with 130 reviewed image replacements."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fashion.config import ROOT, TARGET_COLUMNS
from fashion.data import get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train import task3_usage_mixup_sam as previous
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.registry import RunRegistry
from fashion.train.task3_decisions import oof_metrics
from fashion.train.task3_experiments import Task3ChildSpec

DATASET = "teacher_plus_rare_usage_v3_20260906"
DATA_DIRECTORY = Path("data/processed") / DATASET
SPLIT_SHA256 = "19d514fbd56f48459b3ec207f49bb3e59f1c23bc901804fa0c32f13eb3066ecc"
GEOMETRY_SHA256 = "da01f7e831ac415d9a1ed398be43be35dd00afa8265da630b2a1baa9dd441f2b"
NAME = "usage_replaced_v3_mixup_sam"
EXPERIMENT = "t3_usage_replaced_v3_mixup_sam"
ARTIFACT_DIRECTORY = Path("experiments") / EXPERIMENT
FOLDS = (0, 4)
BASELINE_RUN_IDS = (
    "t3_usage_expanded_v2_mixup_sam_usage_smallcnn_f0_s2753_dd6018a005e9_20260906T154107Z42a2aa",
    "t3_usage_expanded_v2_mixup_sam_usage_smallcnn_f4_s2753_dd6018a005e9_20260906T155155Z92ec97",
)
CLASSES, MAP_SHA256, TEACHER_SHA256 = v2.CLASSES, v2.MAP_SHA256, v2.TEACHER_SHA256
COHORT_COLUMN, GEOMETRY_COLUMNS = v2.COHORT_COLUMN, v2.GEOMETRY_COLUMNS
training_scope, source_predictions, source_metrics = (
    v2.training_scope,
    v2.source_predictions,
    v2.source_metrics,
)
save_learning_curves = previous.save_learning_curves


def dataset_identity():
    return {
        "name": DATASET,
        "split_path": str(DATA_DIRECTORY / "splits.csv"),
        "split_sha256": SPLIT_SHA256,
        "label_map_sha256": MAP_SHA256,
        "teacher_split_sha256": TEACHER_SHA256,
        "previous_split_sha256": v2.SPLIT_SHA256,
        "source_geometry_sha256": GEOMETRY_SHA256,
        "same_usage_target": True,
        "teacher_folds_preserved": True,
        "retained_external_folds_preserved": True,
        "same_external_class_totals": True,
        "replaced_images": 130,
        "source_cohort_meanings": {
            "teacher": "original teacher rows",
            "previous_added": "557 retained v2 outside images",
            "new_added": "130 reviewed replacement images",
        },
        "initialization": "fresh_scratch_weights",
    }


def _fields():
    fields = asdict(previous.screen_spec())
    fields.update(
        name=NAME,
        experiment_id=EXPERIMENT,
        hypothesis_id="t3_usage_replacements_with_fixed_mixup_sam",
        artifact_dir=str(ARTIFACT_DIRECTORY),
        run_prefix=EXPERIMENT,
        changed_factor="replace_130_external_images_with_reviewed_missing_forms",
        parent_artifact_dir=str(previous.ARTIFACT_DIRECTORY),
        parent_run_ids=BASELINE_RUN_IDS,
    )
    return fields


class ReplacedUsageV3Spec(Task3ChildSpec):
    """Change the data while preserving every completed MixUp + SAM control."""

    def __post_init__(self):
        if asdict(self) != _fields():
            raise ValueError("Usage v3 differs from the frozen MixUp + SAM recipe")

    def parent_run_id_for_fold(self, fold):
        if isinstance(fold, bool) or fold not in FOLDS:
            raise ValueError("Usage v3 permits only folds 0 and 4")
        return dict(zip(FOLDS, BASELINE_RUN_IDS, strict=True))[fold]

    def to_dict(self):
        prior = previous.screen_spec().to_dict()
        return {
            **asdict(self),
            "parent_run_ids": list(BASELINE_RUN_IDS),
            "dataset": dataset_identity(),
            "folds": list(FOLDS),
            "initialization": "fresh_random_weights_for_each_fold",
            "reference_use": "comparison_only; no checkpoint weights are loaded",
            "mixup_policy": prior["mixup_policy"],
            "sam_policy": prior["sam_policy"],
        }


def screen_spec():
    return ReplacedUsageV3Spec(**_fields())


def screen_config(spec, *, fold, device_name):
    if isinstance(fold, bool) or fold not in FOLDS:
        raise ValueError("Usage v3 permits only folds 0 and 4")
    if not isinstance(spec, ReplacedUsageV3Spec) or spec.to_dict() != screen_spec().to_dict():
        raise ValueError("Usage v3 differs from the frozen MixUp + SAM recipe")
    if device_name != "cuda":
        raise ValueError("Usage v3 MixUp + SAM training requires a CUDA GPU")
    return Task3BaselineConfig(target="usage")


def validate_dataset(*, root=ROOT, check_images=True):
    """Verify the frozen exchange, family folds, target masks and source geometry."""
    root = Path(root)
    files = {
        Path("data/processed/splits.csv"): TEACHER_SHA256,
        Path("data/processed/label_maps.json"): MAP_SHA256,
        v2.DATA_DIRECTORY / "splits.csv": v2.SPLIT_SHA256,
        DATA_DIRECTORY / "splits.csv": SPLIT_SHA256,
        DATA_DIRECTORY / "label_maps.json": MAP_SHA256,
        DATA_DIRECTORY / "image_geometry.csv": GEOMETRY_SHA256,
    }
    for path, digest in files.items():
        if compute_sha256(root / path) != digest:
            raise ValueError(f"Reviewed Usage v3 data changed: {path}")
    splits = load_splits(root / DATA_DIRECTORY / "splits.csv")
    teacher = load_splits(root / "data/processed/splits.csv")
    older = load_splits(root / v2.DATA_DIRECTORY / "splits.csv")
    retained = older.loc[older.id.isin(splits.id)]
    removed = older.loc[~older.id.isin(splits.id)]
    new = ~splits.id.isin(older.id)
    external = splits.source_dataset.ne("teacher")
    v2._preserve_rows(splits, teacher)
    v2._preserve_rows(splits, retained)
    if (
        len(splits) != 39_299
        or int(external.sum()) != 687
        or int(new.sum()) != 130
        or len(removed) != 130
        or removed.source_dataset.eq("teacher").any()
        or splits.loc[new, "id"].min() <= older.id.max()
        or splits.loc[new, "usage"].value_counts().to_dict()
        != {"Smart Casual": 62, "Party": 44, "Travel": 17, "Home": 7}
        or splits.usage.value_counts().to_dict() != older.usage.value_counts().to_dict()
    ):
        raise ValueError("Usage v3 must exchange the reviewed 130 outside images within classes")
    mapping = load_label_maps(root / DATA_DIRECTORY / "label_maps.json")["usage"]
    if tuple(mapping["classes"]) != CLASSES or mapping["label_to_index"] != {
        label: index for index, label in enumerate(CLASSES)
    }:
        raise ValueError("Usage v3 must preserve the nine classes and indices")
    if not splits.loc[external, "partition"].eq("development").all():
        raise ValueError("Outside images may only enter development")
    if splits.loc[external, "usage"].isin({"", "NA"}).any():
        raise ValueError("No blank or NA outside Usage labels were admitted")
    for target in TARGET_COLUMNS:
        if not splits.loc[external, f"has_{target}_label"].eq(target == "usage").all():
            raise ValueError(f"Invalid outside target mask: {target}")
        if target != "usage" and splits.loc[external, target].ne("").any():
            raise ValueError(f"Outside images do not have {target} labels")
    protected = splits.partition.isin({"holdout", "quarantine"})
    if splits.loc[protected, list(TARGET_COLUMNS)].ne("").any().any():
        raise ValueError("Protected labels must stay blank")
    splits[COHORT_COLUMN] = np.where(external, "previous_added", "teacher")
    splits.loc[new, COHORT_COLUMN] = "new_added"
    geometry = pd.read_csv(root / DATA_DIRECTORY / "image_geometry.csv", keep_default_na=False)
    if geometry.external_id.duplicated().any() or set(geometry.external_id) != set(
        splits.loc[external, "external_id"]
    ):
        raise ValueError("Source geometry must cover every outside image exactly once")
    geometry = geometry.set_index("external_id")
    indexed = splits.loc[external].set_index("external_id").loc[geometry.index]
    for column in ("path", "sha256", *GEOMETRY_COLUMNS):
        actual, expected = indexed[column], geometry[column]
        if column in GEOMETRY_COLUMNS:
            actual, expected = pd.to_numeric(actual), pd.to_numeric(expected)
        if not actual.eq(expected).all():
            raise ValueError(f"Source geometry differs from the split: {column}")
    left, top, width, height = (geometry[c].to_numpy(dtype=float) for c in GEOMETRY_COLUMNS)
    if not (
        (left >= 0)
        & (top >= 0)
        & (width > 0)
        & (height > 0)
        & (left + width <= 60)
        & (top + height <= 80)
    ).all():
        raise ValueError("Source geometry lies outside the RGB 60x80 image")
    for column in GEOMETRY_COLUMNS:
        if splits.loc[~external, column].ne("").any():
            raise ValueError("Outside source borders cannot override teacher geometry")
        splits[column] = pd.to_numeric(splits[column].replace("", np.nan), errors="raise")
    folds = []
    for fold in range(5):
        training, validation = training_scope(splits, fold)
        row = {"fold": fold, "training_rows": len(training), "validation_rows": len(validation)}
        for cohort in ("teacher", "previous_added", "new_added"):
            for side, frame in (("training", training), ("validation", validation)):
                row[f"{cohort}_{side}_rows"] = int(frame[COHORT_COLUMN].eq(cohort).sum())
        folds.append(row)
    development = get_samples(splits, partition="development", target="usage")
    if check_images:
        for row in development.itertuples(index=False):
            path = root / row.path
            if compute_sha256(path) != row.sha256:
                raise ValueError(f"Development image bytes changed: {path}")
            with Image.open(path) as image:
                image.load()
                if row.source_dataset != "teacher" and (
                    image.mode != "RGB" or image.size != (60, 80)
                ):
                    raise ValueError(f"Outside image must be RGB 60x80: {path}")
    return splits, {
        **dataset_identity(),
        "total_rows": len(splits),
        "usage_development_rows": len(development),
        "teacher_rows": int((~external).sum()),
        "retained_external_rows": int((external & ~new).sum()),
        "new_added_rows": int(new.sum()),
        "folds": folds,
    }


def usage_registry_path(output_root):
    return Path(output_root) / ARTIFACT_DIRECTORY / "usage/results/runs.csv"


def check_reference(*, directory, registry_path, root=ROOT):
    """Verify the completed v2 SAM controls using their original data and receipts."""
    old_splits, _ = v2.validate_dataset(root=root, check_images=False)
    rows = RunRegistry(registry_path)._read_rows()
    references = {}
    for fold, run_id in zip(FOLDS, BASELINE_RUN_IDS, strict=True):
        matches = [row for row in rows if row["run_id"] == run_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one completed v2 MixUp + SAM control for fold {fold}")
        result = v2._verified_completed_fold(
            row=matches[0],
            path=Path(directory) / run_id,
            splits=old_splits,
            fold=fold,
            spec=previous.screen_spec(),
            split_digest=v2.SPLIT_SHA256,
        )
        previous.verify_training_evidence(
            result, splits=old_splits, fold=fold, spec=previous.screen_spec()
        )
        result["predictions"] = v2.read_predictions(result["prediction_path"])
        references[fold] = result
    return references


def completed_fold(*, fold, output_root, registry_path, splits, spec):
    rows = RunRegistry(registry_path)._read_rows()
    if any(row["experiment_id"] != EXPERIMENT for row in rows):
        raise ValueError("Usage v3 run log contains another experiment")
    matches = [r for r in rows if r["status"] == "complete" and int(r["validation_fold"]) == fold]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Multiple completed Usage v3 runs exist for fold {fold}")
    result = v2._verified_completed_fold(
        row=matches[0],
        path=Path(output_root) / ARTIFACT_DIRECTORY / "usage" / matches[0]["run_id"],
        splits=splits,
        fold=fold,
        spec=spec,
        split_digest=SPLIT_SHA256,
    )
    previous.verify_training_evidence(result, splits=splits, fold=fold, spec=spec)
    return result


def build_comparison(results, *, splits, references, contract):
    if [int(r["metrics"]["validation_fold"]) for r in results] != list(FOLDS) or set(
        references
    ) != set(FOLDS):
        raise ValueError("Usage v3 comparison requires exactly folds 0 and 4")
    current = source_predictions(
        pd.concat([v2.read_predictions(r["prediction_path"]) for r in results], ignore_index=True),
        pd.concat([training_scope(splits, f)[1] for f in FOLDS]),
    )
    old = pd.concat([references[f]["predictions"] for f in FOLDS], ignore_index=True)
    teacher = current.loc[current.source_dataset.eq("teacher")]
    old_teacher = old.loc[old.source_dataset.eq("teacher")]
    if old_teacher.id.duplicated().any() or set(old_teacher.id) != set(teacher.id):
        raise ValueError("Teacher controls must exactly cover the same validation IDs")
    old_teacher = old_teacher.set_index("id").loc[teacher.id].reset_index()
    if not np.array_equal(old_teacher.true_index.to_numpy(), teacher.true_index.to_numpy()):
        raise ValueError("Teacher reference labels differ")
    scores, baseline = source_metrics(current), oof_metrics(old_teacher, CLASSES)
    folds = []
    for run in results:
        fold = int(run["metrics"]["validation_fold"])
        now = oof_metrics(teacher.loc[teacher.cv_fold.eq(fold)], CLASSES)
        before = oof_metrics(old_teacher.loc[old_teacher.cv_fold.eq(fold)], CLASSES)
        clean = run["metrics"]["source_metrics"]["clean_training"]["teacher"]["metrics"]
        old_clean = references[fold]["metrics"]["source_metrics"]["clean_training"]["teacher"][
            "metrics"
        ]
        folds.append(
            {
                "fold": fold,
                "baseline_teacher_f1": before["macro_f1"],
                "candidate_teacher_f1": now["macro_f1"],
                "teacher_f1_change": now["macro_f1"] - before["macro_f1"],
                "clean_teacher_train_f1": clean["macro_f1"],
                "clean_teacher_gap": clean["macro_f1"] - now["macro_f1"],
                "baseline_clean_teacher_gap": old_clean["macro_f1"] - before["macro_f1"],
            }
        )
    current_teacher = scores["teacher"]["metrics"]
    return {
        "folds": list(FOLDS),
        "dataset": contract,
        "sources": scores,
        "baseline_sources": {"teacher": {"rows": len(old_teacher), "metrics": baseline}},
        "primary_scope": "same teacher validation IDs, folds 0 and 4, all nine Usage classes",
        "teacher_macro_f1_change": current_teacher["macro_f1"] - baseline["macro_f1"],
        "fold_comparison": folds,
        "teacher_per_class": [
            {
                "class_name": a["class_name"],
                "support": a["support"],
                "baseline_f1": b["f1"],
                "candidate_f1": a["f1"],
                "baseline_recall": b["recall"],
                "candidate_recall": a["recall"],
            }
            for a, b in zip(current_teacher["per_class"], baseline["per_class"], strict=True)
        ],
        "limits": [
            "Only the dataset changes; MixUp, SAM, architecture and training controls stay fixed.",
            "Weights and normalization are refitted on each new training fold only.",
            "Retained/new outside-source scores describe different groups, not teacher gains.",
            "Tiny teacher classes and previously inspected evaluation results limit conclusions.",
            "No automatic model promotion or training of the other three folds.",
        ],
    }


def run_usage_replaced_v3(
    *,
    output_root,
    baseline_directory,
    baseline_registry_path,
    root=ROOT,
    registry_mirrors=(),
    folds=FOLDS,
):
    """Fit two fresh models or reuse only fully verified completed v3 runs."""
    import torch

    from fashion.train.task3_baseline import _aggregate_target, run_task3_baseline_fold

    if tuple(folds) != FOLDS or any(isinstance(fold, bool) for fold in folds):
        raise ValueError("Usage v3 trains exactly folds 0 and 4, in that order")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before running Usage v3 MixUp + SAM")
    root, output_root = Path(root), Path(output_root)
    spec = screen_spec()
    screen_config(spec, fold=0, device_name="cuda")
    registry_path = usage_registry_path(output_root)
    protected = {
        Path(baseline_registry_path).resolve(),
        registry_path.resolve(),
        (root / "results/runs.csv").resolve(),
        (output_root / "results/runs.csv").resolve(),
    }
    if registry_path.resolve() == Path(baseline_registry_path).resolve():
        raise ValueError("The Usage v3 log must be separate from its reference")
    if any(Path(p).resolve() in protected for p in registry_mirrors):
        raise ValueError("Use a separate v3-only local results/runs.csv mirror")
    for path in (registry_path, *registry_mirrors):
        if any(row["experiment_id"] != EXPERIMENT for row in RunRegistry(path)._read_rows()):
            raise ValueError("A Usage v3 run log cannot overwrite another experiment")
    splits, contract = validate_dataset(root=root, check_images=True)
    references = check_reference(
        directory=baseline_directory, registry_path=baseline_registry_path, root=root
    )
    destination = output_root / ARTIFACT_DIRECTORY / "usage"
    v2.write_json(
        {
            "dataset": contract,
            "recipe": spec.to_dict(),
            "references": {str(f): references[f]["run_id"] for f in FOLDS},
        },
        destination / "prerequisites.json",
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
            print(f"Usage v3 MixUp + SAM: fresh weights, fold {fold}, 30 epochs", flush=True)
            run_task3_baseline_fold(
                "usage",
                fold,
                root=root,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                child_spec=spec,
                parent_run_directory=Path(baseline_directory) / spec.parent_run_id_for_fold(fold),
            )
            result = completed_fold(
                fold=fold,
                output_root=output_root,
                registry_path=registry_path,
                splits=splits,
                spec=spec,
            )
            if result is None:
                raise RuntimeError(f"Usage v3 fold {fold} did not register completion")
        else:
            print(f"Reusing verified Usage v3 fold {fold}: {result['run_id']}", flush=True)
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
    destination = destination / "aggregate_folds_0_4"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--baseline-registry", type=Path, required=True)
    args = parser.parse_args()
    result = run_usage_replaced_v3(
        root=args.root,
        output_root=args.output_root,
        baseline_directory=args.baseline_directory,
        baseline_registry_path=args.baseline_registry,
    )
    print(result["comparison_path"])


if __name__ == "__main__":
    main()
