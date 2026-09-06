"""Repeat Usage E8 from scratch on the explicitly versioned combined dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.task3_decisions import oof_metrics, validate_oof
from fashion.train.task3_experiments import (
    Task3ChildSpec,
    effective_number_class_weights,
    usage_translation_2px_spec,
)

DATASET = "teacher_plus_rare_usage_20260906"
DATA_DIRECTORY = Path("data/processed") / DATASET
EXPERIMENT = "t3_usage_expanded_e8"
ARTIFACT_DIRECTORY = Path("experiments") / EXPERIMENT
E8_DIRECTORY = Path("experiments/t3_usage_e8_translation/usage")
CLASSES = ("Casual", "Ethnic", "Formal", "Home", "NA", "Party", "Smart Casual", "Sports", "Travel")
E8_RUN_IDS = (
    "t3_usage_e8_translation_usage_smallcnn_f0_s2753_be3cf562b9ab_20260831T173123Zd848db",
    "t3_usage_e8_translation_usage_smallcnn_f1_s2753_be3cf562b9ab_20260831T174039Zabd68d",
    "t3_usage_e8_translation_usage_smallcnn_f2_s2753_be3cf562b9ab_20260831T174953Z528158",
    "t3_usage_e8_translation_usage_smallcnn_f3_s2753_be3cf562b9ab_20260831T175907Z3587b1",
    "t3_usage_e8_translation_usage_smallcnn_f4_s2753_be3cf562b9ab_20260831T180832Z708926",
)
TEACHER_SHA256 = "d76a49c6dc7999b4f286e94838a92c603d68c4f66179fde081028948f6a187db"
SPLIT_SHA256 = "7d5f3a986ef8e00f53ee1c86d408f1c3a155d6b0a0ab99bef4c9f7eb27dbda77"
MAP_SHA256 = "51b7623c3a2c1a8f9ebb9b0d8b4b8a50a24ea57522c1f8f4dd15b94c1996a327"
SOURCE_MANIFEST = Path("data/external/rare_usage_20260906/splits.csv")
SOURCE_SHA256 = "e951f13baf7916d74513015aaef2d79aa845a50278e6904f1c946f4803051e79"


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(path)


def read_predictions(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def dataset_identity():
    return {
        "name": DATASET,
        "split_path": str(DATA_DIRECTORY / "splits.csv"),
        "split_sha256": SPLIT_SHA256,
        "label_map_sha256": MAP_SHA256,
        "teacher_split_sha256": TEACHER_SHA256,
        "source_geometry_sha256": SOURCE_SHA256,
        "same_usage_target": True,
        "teacher_folds_preserved": True,
        "initialization": "fresh_scratch_weights",
    }


def _expanded_fields():
    fields = asdict(usage_translation_2px_spec(E8_RUN_IDS))
    fields.update(
        name="usage_expanded_e8",
        experiment_id=EXPERIMENT,
        hypothesis_id="t3_usage_teacher_plus_rare_images",
        artifact_dir=str(ARTIFACT_DIRECTORY),
        run_prefix=EXPERIMENT,
        changed_factor="training_dataset",
        parent_artifact_dir="experiments/t3_usage_e8_translation",
    )
    return fields


class ExpandedUsageSpec(Task3ChildSpec):
    """Reuse every frozen E8 model and loss field; change only dataset identity."""

    def __post_init__(self):
        if asdict(self) != _expanded_fields():
            raise ValueError("Expanded Usage must preserve the frozen E8 recipe")

    def to_dict(self):
        return {**asdict(self), "parent_run_ids": list(E8_RUN_IDS), "dataset": dataset_identity()}


def expanded_usage_spec():
    return ExpandedUsageSpec(**_expanded_fields())


def training_scope(splits, fold):
    if fold not in range(5):
        raise ValueError("Usage validation fold must be between 0 and 4")
    training, validation = (get_samples(f, target="usage") for f in get_cv_split(splits, fold))
    if set(training.product_family_group) & set(validation.product_family_group):
        raise ValueError("A product family crosses the saved fold")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def validate_dataset(*, root=ROOT, check_images=True):
    """Require the reviewed dataset bytes and unchanged original teacher rows."""
    root = Path(root)
    for path, digest in (
        (Path("data/processed/splits.csv"), TEACHER_SHA256),
        (Path("data/processed/label_maps.json"), MAP_SHA256),
        (DATA_DIRECTORY / "splits.csv", SPLIT_SHA256),
        (DATA_DIRECTORY / "label_maps.json", MAP_SHA256),
        (SOURCE_MANIFEST, SOURCE_SHA256),
    ):
        if compute_sha256(root / path) != digest:
            raise ValueError(f"Reviewed Usage data changed: {path}")
    splits = load_splits(root / DATA_DIRECTORY / "splits.csv")
    teacher = load_splits(root / "data/processed/splits.csv")
    retained = splits.loc[splits.source_dataset.eq("teacher"), teacher.columns]
    pd.testing.assert_frame_equal(
        retained.reset_index(drop=True).astype(str), teacher.reset_index(drop=True).astype(str)
    )
    mapping = load_label_maps(root / DATA_DIRECTORY / "label_maps.json")["usage"]
    if tuple(mapping["classes"]) != CLASSES:
        raise ValueError("The nine Usage classes changed")
    # The admitted PNGs already contain letterboxing. Recover its reviewed
    # geometry for RGB statistics only; labels and folds come from combined splits.
    geometry = pd.read_csv(root / SOURCE_MANIFEST, keep_default_na=False)
    columns = ["content_left", "content_top", "content_width", "content_height"]
    geometry = geometry.set_index("external_id")
    for column in columns:
        splits[column] = splits.external_id.map(geometry[column])
    folds = []
    for fold in range(5):
        training, validation = training_scope(splits, fold)
        folds.append(
            {
                "fold": fold,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "added_training_rows": int(training.source_dataset.ne("teacher").sum()),
                "added_validation_rows": int(validation.source_dataset.ne("teacher").sum()),
            }
        )
    if check_images:
        development = get_samples(splits.loc[splits.partition.eq("development")], target="usage")
        for row in development.itertuples():
            path = root / row.path
            if not path.is_file():
                raise FileNotFoundError(f"Missing training image: {path}")
            if compute_sha256(path) != row.sha256:
                raise ValueError(f"Training image bytes changed: {path}")
    return splits, {**dataset_identity(), "folds": folds}


def source_predictions(predictions, expected):
    """Attach origin only after exact-ID validation; never silently drop a row."""
    clean = validate_oof(predictions, expected, target="usage", classes=CLASSES)
    clean = clean.drop(columns="source_dataset", errors="ignore")
    return clean.merge(expected[["id", "source_dataset"]], on="id", validate="one_to_one")


def source_metrics(predictions):
    scopes = {
        "combined": predictions,
        "teacher": predictions.loc[predictions.source_dataset.eq("teacher")],
        "added": predictions.loc[predictions.source_dataset.ne("teacher")],
    }
    return {
        name: {"rows": len(frame), "metrics": oof_metrics(frame, CLASSES) if len(frame) else None}
        for name, frame in scopes.items()
    }


def check_e8_sources(*, directory, registry_path, root=ROOT):
    """Verify E8's registered recipe and scores. Checkpoint weights are never loaded."""
    root, directory = Path(root), Path(directory)
    teacher = load_splits(root / "data/processed/splits.csv")
    registry = pd.read_csv(registry_path, keep_default_na=False)
    sources = {}
    base = Task3BaselineConfig(target="usage").to_dict()
    for fold, run_id in enumerate(E8_RUN_IDS):
        rows = registry.loc[registry.run_id.eq(run_id)]
        if len(rows) != 1:
            raise ValueError(f"Expected one registered E8 source: {run_id}")
        row = rows.iloc[0]
        if (
            row.status != "complete"
            or row.target != "usage"
            or int(row.validation_fold) != fold
            or int(row.seed) != 2753
            or str(row.scratch).lower() != "true"
            or str(row.debug).lower() != "false"
            or row.split_digest != TEACHER_SHA256
            or row.label_map_digest != MAP_SHA256
        ):
            raise ValueError(
                "E8 source is not a completed scratch run on the original teacher data"
            )
        path = directory / run_id
        names = (
            "config.json",
            "metrics.json",
            "normalization.json",
            "history.csv",
            "oof_predictions.csv",
            "robustness.csv",
            "final_epoch.pt",
        )
        hashes = {name: compute_sha256(path / name) for name in names}
        if (
            hashes["final_epoch.pt"] != row.checkpoint_sha256
            or hashes["oof_predictions.csv"] != row.prediction_sha256
        ):
            raise ValueError("E8 artifacts differ from the run registry")
        config = json.loads((path / "config.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        child = config["child_experiment"]
        expected_child = usage_translation_2px_spec(child["parent_run_ids"]).to_dict()
        if (
            any(config.get(k) != v for k, v in base.items())
            or child != expected_child
            or row.experiment_id != expected_child["experiment_id"]
        ):
            raise ValueError("E8 recipe differs from the frozen 30-epoch translation run")
        if metrics != json.loads(row.metrics_json):
            raise ValueError("E8 metrics differ from the run registry")
        training, expected = training_scope(teacher, fold)
        counts = [int(training.usage.eq(c).sum()) for c in CLASSES]
        weights = effective_number_class_weights(counts, beta=0.999, cap=5.0)
        if config["class_counts"] != counts or not np.allclose(config["class_weights"], weights):
            raise ValueError("E8 class weights differ from its training rows")
        predictions = validate_oof(
            read_predictions(path / "oof_predictions.csv"),
            expected,
            target="usage",
            classes=CLASSES,
            run_ids_by_fold={fold: run_id},
            allow_legacy_na=True,
        )
        measured = oof_metrics(predictions, CLASSES)
        for key in ("macro_f1", "nll", "brier", "ece_15"):
            if not np.isclose(measured[key], metrics[key], atol=1e-7, rtol=0):
                raise ValueError(f"E8 saved probabilities disagree with {key}")
        sources[fold] = {
            "run_id": run_id,
            "sha256": hashes,
            "predictions": predictions,
            "metrics": measured,
        }
    return sources


def reusable_fold(*, fold, output_root, registry_path, splits, spec):
    """Resume only an intact, complete run with this exact dataset and recipe."""
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        return None
    registry = pd.read_csv(registry_path, keep_default_na=False)
    rows = registry.loc[
        registry.experiment_id.eq(EXPERIMENT)
        & registry.status.eq("complete")
        & pd.to_numeric(registry.validation_fold, errors="coerce").eq(fold)
    ]
    expected_config = spec.to_dict()
    base = Task3BaselineConfig(target="usage").to_dict()
    _, expected = training_scope(splits, fold)
    for row in reversed(list(rows.itertuples())):
        path = Path(output_root) / ARTIFACT_DIRECTORY / "usage" / row.run_id
        if not (path / "config.json").is_file():
            continue
        config = json.loads((path / "config.json").read_text())
        if config.get("child_experiment") != expected_config or any(
            config.get(k) != v for k, v in base.items()
        ):
            continue
        if row.split_digest != SPLIT_SHA256 or row.label_map_digest != MAP_SHA256:
            raise ValueError("Completed expanded run has a different data digest")
        for name, digest in (
            ("final_epoch.pt", row.checkpoint_sha256),
            ("oof_predictions.csv", row.prediction_sha256),
        ):
            if compute_sha256(path / name) != digest:
                raise ValueError(f"Completed expanded run artifact changed: {path / name}")
        metrics = json.loads((path / "metrics.json").read_text())
        if metrics != json.loads(row.metrics_json) or metrics["selected_epoch"] != 30:
            raise ValueError("Completed expanded run metrics changed")
        validate_oof(
            read_predictions(path / "oof_predictions.csv"),
            expected,
            target="usage",
            classes=CLASSES,
            run_ids_by_fold={fold: row.run_id},
        )
        for name, digest in metrics["expanded_artifact_sha256"].items():
            if compute_sha256(path / name) != digest:
                raise ValueError(f"Completed expanded diagnostic changed: {name}")
        return {
            "run_id": row.run_id,
            "run_dir": str(path),
            "metrics": metrics,
            "prediction_path": str(path / "oof_predictions.csv"),
            "robustness_path": str(path / "robustness.csv"),
            "metrics_path": str(path / "metrics.json"),
        }
    return None


def run_expanded_usage(
    *,
    output_root,
    registry_path,
    e8_directory,
    source_registry_path,
    root=ROOT,
    registry_mirrors=(),
    folds=(0, 1, 2, 3, 4),
    resume=True,
):
    """Run sequential GPU folds, append each to the registry, then compare teacher OOF."""
    import torch

    from fashion.train.task3_baseline import _aggregate_target, run_task3_baseline_fold

    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise ValueError("Choose distinct folds from 0,1,2,3,4")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before running the expanded Usage experiment")
    root, output_root = Path(root), Path(output_root)
    splits, contract = validate_dataset(root=root)
    sources = check_e8_sources(
        directory=e8_directory, registry_path=source_registry_path, root=root
    )
    spec = expanded_usage_spec()
    experiment_dir = output_root / ARTIFACT_DIRECTORY / "usage"
    write_json(
        {
            "dataset": contract,
            "baseline": Task3BaselineConfig(target="usage").to_dict(),
            "child": spec.to_dict(),
            "sources": {
                str(f): {k: v for k, v in s.items() if k != "predictions"}
                for f, s in sources.items()
            },
        },
        experiment_dir / "prerequisites.json",
    )
    results = []
    for fold in folds:
        result = (
            reusable_fold(
                fold=fold,
                output_root=output_root,
                registry_path=registry_path,
                splits=splits,
                spec=spec,
            )
            if resume
            else None
        )
        if result is None:
            result = run_task3_baseline_fold(
                "usage",
                fold,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                child_spec=spec,
                parent_run_directory=Path(e8_directory) / E8_RUN_IDS[fold],
            )
        else:
            print(f"Reusing verified fold {fold}: {result['run_id']}", flush=True)
        results.append(result)
        torch.cuda.empty_cache()
    aggregate_name = (
        "aggregate"
        if set(folds) == set(range(5))
        else ("aggregate_folds_" + "_".join(str(f) for f in sorted(folds)))
    )
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
        aggregate_dir_name=aggregate_name,
    )
    predictions = pd.concat(
        [read_predictions(r["prediction_path"]) for r in results], ignore_index=True
    )
    scopes = source_metrics(predictions)
    parent = pd.concat([sources[f]["predictions"] for f in folds], ignore_index=True)
    child_teacher = predictions.loc[predictions.source_dataset.eq("teacher")]
    if set(parent.id) != set(child_teacher.id):
        raise ValueError("Teacher-only comparison must use the exact same image IDs")
    parent_metrics = oof_metrics(parent, CLASSES)
    child_metrics = scopes["teacher"]["metrics"]
    comparison = {
        "folds": list(folds),
        "dataset": contract,
        "sources": scopes,
        "teacher_reference_e8": parent_metrics,
        "teacher_macro_f1_change": child_metrics["macro_f1"] - parent_metrics["macro_f1"],
        "teacher_per_class": [
            {
                "class_name": c["class_name"],
                "support": c["support"],
                "e8_f1": p["f1"],
                "expanded_f1": c["f1"],
                "change": c["f1"] - p["f1"],
            }
            for c, p in zip(child_metrics["per_class"], parent_metrics["per_class"], strict=True)
        ],
        "limits": [
            "Added-source scores have a different class mix and small samples.",
            "Smart Casual has two added families; Travel has one added image.",
            "No added NA images. Holdout and prediction images remain sealed.",
            "Changing training data may lower scores; this run does not auto-promote.",
        ],
    }
    summary_path = experiment_dir / aggregate_name / "teacher_comparison.json"
    write_json(comparison, summary_path)
    pd.DataFrame(comparison["teacher_per_class"]).to_csv(
        summary_path.with_name("teacher_per_class.csv"), index=False
    )
    return {
        "fold_results": results,
        "aggregate": aggregate,
        "comparison": comparison,
        "comparison_path": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--e8-directory", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_expanded_usage(
        output_root=args.output_root,
        registry_path=args.registry,
        e8_directory=args.e8_directory,
        source_registry_path=args.source_registry,
        root=args.root,
        folds=args.folds,
        resume=not args.no_resume,
    )
    print(result["comparison_path"])


if __name__ == "__main__":
    main()
