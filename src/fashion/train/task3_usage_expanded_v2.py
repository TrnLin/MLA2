"""Train the frozen E8 recipe on teacher images plus all 687 admitted Usage images."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fashion.config import ROOT, TARGET_COLUMNS
from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded as previous
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.registry import RunRegistry
from fashion.train.task3_decisions import oof_metrics, validate_oof
from fashion.train.task3_experiments import Task3ChildSpec, effective_number_class_weights

DATASET = "teacher_plus_rare_usage_v2_20260906"
DATA_DIRECTORY = Path("data/processed") / DATASET
EXPERIMENT = "t3_usage_expanded_v2_e8"
ARTIFACT_DIRECTORY = Path("experiments") / EXPERIMENT
SPLIT_SHA256 = "0bd490672543b7022fa7a883be9b006d63681d7c168e497e0388616836696776"
GEOMETRY_SHA256 = "564f1141a475a321f6a7bb5661062957655bcad4c907ceb7b3bbc1c11fa1dc42"
MAP_SHA256 = previous.MAP_SHA256
TEACHER_SHA256 = previous.TEACHER_SHA256
CLASSES = previous.CLASSES
E8_RUN_IDS = previous.E8_RUN_IDS
PREVIOUS_RUN_IDS = (
    "t3_usage_expanded_e8_usage_smallcnn_f0_s2753_6eb56854d557_20260906T091936Zd9a955",
    "t3_usage_expanded_e8_usage_smallcnn_f1_s2753_6eb56854d557_20260906T092900Z43e2aa",
    "t3_usage_expanded_e8_usage_smallcnn_f2_s2753_6eb56854d557_20260906T093831Zb99abd",
    "t3_usage_expanded_e8_usage_smallcnn_f3_s2753_6eb56854d557_20260906T094801Z1a6eea",
    "t3_usage_expanded_e8_usage_smallcnn_f4_s2753_6eb56854d557_20260906T095731Zaa2e15",
)
COHORT_COLUMN = "usage_source_cohort"
GEOMETRY_COLUMNS = ("content_left", "content_top", "content_width", "content_height")
DIAGNOSTIC_FILES = {
    "config.json",
    "normalization.json",
    "history.csv",
    "robustness.csv",
    "training_predictions.csv",
    "source_metrics.json",
}
write_json = previous.write_json
read_predictions = previous.read_predictions


def dataset_identity():
    return {
        "name": DATASET,
        "split_path": str(DATA_DIRECTORY / "splits.csv"),
        "split_sha256": SPLIT_SHA256,
        "label_map_sha256": MAP_SHA256,
        "teacher_split_sha256": TEACHER_SHA256,
        "previous_split_sha256": previous.SPLIT_SHA256,
        "source_geometry_sha256": GEOMETRY_SHA256,
        "same_usage_target": True,
        "teacher_folds_preserved": True,
        "previous_external_folds_preserved": True,
        "initialization": "fresh_scratch_weights",
    }


def _expanded_fields():
    fields = asdict(previous.expanded_usage_spec())
    fields.update(
        name="usage_expanded_v2_e8",
        experiment_id=EXPERIMENT,
        hypothesis_id="t3_usage_teacher_plus_687_rare_images",
        artifact_dir=str(ARTIFACT_DIRECTORY),
        run_prefix=EXPERIMENT,
    )
    return fields


class ExpandedUsageV2Spec(Task3ChildSpec):
    """Change the dataset identity while retaining every E8 training control."""

    def __post_init__(self):
        if asdict(self) != _expanded_fields():
            raise ValueError("Expanded Usage v2 must preserve the frozen E8 recipe")

    def to_dict(self):
        return {**asdict(self), "parent_run_ids": list(E8_RUN_IDS), "dataset": dataset_identity()}


def expanded_usage_spec():
    return ExpandedUsageV2Spec(**_expanded_fields())


def training_scope(splits, fold):
    if isinstance(fold, bool) or fold not in range(5):
        raise ValueError("Usage validation fold must be between 0 and 4")
    training, validation = (get_samples(f, target="usage") for f in get_cv_split(splits, fold))
    for column in ("product_family_group", "extension_family_group"):
        if column not in splits:
            continue
        groups = set(training[column].dropna().astype(str)) - {""}
        other = set(validation[column].dropna().astype(str)) - {""}
        if groups & other:
            raise ValueError(f"A {column} crosses the saved fold")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def _preserve_rows(combined, original):
    retained = combined.set_index("id").loc[original.id, original.columns.drop("id")]
    pd.testing.assert_frame_equal(
        retained.reset_index().astype(str), original.reset_index(drop=True).astype(str)
    )


def validate_dataset(*, root=ROOT, check_images=True):
    """Check frozen files, old folds, source borders and only development image pixels."""
    root = Path(root)
    expected_files = {
        Path("data/processed/splits.csv"): TEACHER_SHA256,
        Path("data/processed/label_maps.json"): MAP_SHA256,
        previous.DATA_DIRECTORY / "splits.csv": previous.SPLIT_SHA256,
        DATA_DIRECTORY / "splits.csv": SPLIT_SHA256,
        DATA_DIRECTORY / "label_maps.json": MAP_SHA256,
        DATA_DIRECTORY / "image_geometry.csv": GEOMETRY_SHA256,
    }
    for path, digest in expected_files.items():
        if compute_sha256(root / path) != digest:
            raise ValueError(f"Reviewed Usage v2 data changed: {path}")
    splits = load_splits(root / DATA_DIRECTORY / "splits.csv")
    teacher = load_splits(root / "data/processed/splits.csv")
    older = load_splits(root / previous.DATA_DIRECTORY / "splits.csv")
    _preserve_rows(splits, teacher)
    _preserve_rows(splits, older)
    mapping = load_label_maps(root / DATA_DIRECTORY / "label_maps.json")["usage"]
    if tuple(mapping["classes"]) != CLASSES or mapping["label_to_index"] != {
        label: index for index, label in enumerate(CLASSES)
    }:
        raise ValueError("The existing nine Usage classes and indices must be preserved")

    external = splits.source_dataset.ne("teacher")
    new = ~splits.id.isin(older.id)
    if len(splits) != 39299 or external.sum() != 687 or new.sum() != 567:
        raise ValueError("Usage v2 requires all teacher, previous 120 and new 567 rows")
    if not splits.loc[external, "partition"].eq("development").all():
        raise ValueError("External images may only enter development")
    if splits.loc[external, "usage"].isin({"", "NA"}).any():
        raise ValueError("No blank or NA external Usage labels were admitted")
    for target in TARGET_COLUMNS:
        if not splits.loc[external, f"has_{target}_label"].eq(target == "usage").all():
            raise ValueError(f"Invalid external target mask: {target}")
        if target != "usage" and splits.loc[external, target].ne("").any():
            raise ValueError(f"External images do not have {target} labels")
    protected = splits.partition.isin({"holdout", "quarantine"})
    if splits.loc[protected, list(TARGET_COLUMNS)].ne("").any().any():
        raise ValueError("Protected labels must stay blank")
    splits[COHORT_COLUMN] = np.where(external, "previous_added", "teacher")
    splits.loc[new, COHORT_COLUMN] = "new_added"

    geometry = pd.read_csv(root / DATA_DIRECTORY / "image_geometry.csv", keep_default_na=False)
    if geometry.external_id.duplicated().any() or set(geometry.external_id) != set(
        splits.loc[external, "external_id"]
    ):
        raise ValueError("Source-border geometry must cover every external image once")
    geometry = geometry.set_index("external_id")
    indexed = splits.loc[external].set_index("external_id").loc[geometry.index]
    for column in ("path", "sha256", *GEOMETRY_COLUMNS):
        actual = indexed[column]
        expected = geometry[column]
        if column in GEOMETRY_COLUMNS:
            actual = pd.to_numeric(actual, errors="raise")
            expected = pd.to_numeric(expected, errors="raise")
        if not actual.eq(expected).all():
            raise ValueError(f"Source-border geometry differs from the split: {column}")
    left, top, width, height = (geometry[c].to_numpy(dtype=float) for c in GEOMETRY_COLUMNS)
    valid = (
        (left >= 0)
        & (top >= 0)
        & (width > 0)
        & (height > 0)
        & (left + width <= 60)
        & (top + height <= 80)
    )
    if not valid.all():
        raise ValueError("Source-border geometry lies outside the 60x80 image")
    # Empty teacher cells must become NaN so the shared normalizer uses its ordinary mask.
    for column in GEOMETRY_COLUMNS:
        if splits.loc[~external, column].ne("").any():
            raise ValueError("External source borders cannot override teacher image geometry")
        splits[column] = pd.to_numeric(splits[column].replace("", np.nan), errors="raise")

    folds = []
    for fold in range(5):
        training, validation = training_scope(splits, fold)
        row = {"fold": fold, "training_rows": len(training), "validation_rows": len(validation)}
        for cohort in ("teacher", "previous_added", "new_added"):
            row[f"{cohort}_training_rows"] = int(training[COHORT_COLUMN].eq(cohort).sum())
            row[f"{cohort}_validation_rows"] = int(validation[COHORT_COLUMN].eq(cohort).sum())
        folds.append(row)
    development = get_samples(splits.loc[splits.partition.eq("development")], target="usage")
    if check_images:
        for row in development.itertuples(index=False):
            path = root / row.path
            if compute_sha256(path) != row.sha256:
                raise ValueError(f"Training image bytes changed: {path}")
            with Image.open(path) as image:
                image.load()
                if row.source_dataset != "teacher" and (
                    image.mode != "RGB" or image.size != (60, 80)
                ):
                    raise ValueError(f"External image must be RGB 60x80: {path}")
    contract = {
        **dataset_identity(),
        "total_rows": len(splits),
        "usage_development_rows": len(development),
        "teacher_rows": int((~external).sum()),
        "previous_added_rows": int((external & ~new).sum()),
        "new_added_rows": int(new.sum()),
        "holdout_rows": int(splits.partition.eq("holdout").sum()),
        "quarantine_rows": int(splits.partition.eq("quarantine").sum()),
        "folds": folds,
    }
    return splits, contract


def source_predictions(predictions, expected):
    clean = previous.source_predictions(predictions, expected)
    clean = clean.drop(columns=COHORT_COLUMN, errors="ignore")
    return clean.merge(expected[["id", COHORT_COLUMN]], on="id", validate="one_to_one")


def source_metrics(predictions):
    scopes = {"combined": predictions}
    scopes.update(
        {
            c: predictions.loc[predictions[COHORT_COLUMN].eq(c)]
            for c in ("teacher", "previous_added", "new_added")
        }
    )
    scopes["added"] = predictions.loc[predictions.source_dataset.ne("teacher")]
    return {
        name: {"rows": len(frame), "metrics": oof_metrics(frame, CLASSES) if len(frame) else None}
        for name, frame in scopes.items()
    }


def usage_registry_path(output_root):
    return Path(output_root) / ARTIFACT_DIRECTORY / "usage/results/runs.csv"


def _verified_completed_fold(*, row, path, splits, fold, spec, split_digest):
    """Verify completion from the registry, recipe, saved files and exact validation IDs."""
    base = Task3BaselineConfig(target="usage").to_dict()
    if (
        row["status"] != "complete"
        or row["experiment_id"] != spec.experiment_id
        or row["target"] != "usage"
        or int(row["validation_fold"]) != fold
        or int(row["seed"]) != base["seed"]
        or row["scratch"] != "true"
        or row["debug"] != "false"
        or row["split_digest"] != split_digest
        or row["label_map_digest"] != MAP_SHA256
    ):
        raise ValueError(f"Completed Usage run identity differs: {row['run_id']}")
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("child_experiment") != spec.to_dict() or any(
        config.get(key) != value for key, value in base.items()
    ):
        raise ValueError(f"Completed Usage recipe differs: {row['run_id']}")
    metrics = json.loads((path / "metrics.json").read_text())
    if (
        metrics != json.loads(row["metrics_json"])
        or metrics["run_id"] != row["run_id"]
        or metrics["validation_fold"] != fold
        or metrics["selected_epoch"] != base["epochs"]
        or metrics["epochs_completed"] != base["epochs"]
        or metrics["checkpoint_policy"] != "final_epoch"
    ):
        raise ValueError(f"Completed Usage metrics differ: {row['run_id']}")
    hashes = metrics.get("expanded_artifact_sha256", {})
    if set(hashes) != DIAGNOSTIC_FILES:
        raise ValueError("Completed Usage diagnostic bundle is incomplete")
    hashes = {
        **hashes,
        "final_epoch.pt": row["checkpoint_sha256"],
        "oof_predictions.csv": row["prediction_sha256"],
    }
    for name, digest in hashes.items():
        if compute_sha256(path / name) != digest:
            raise ValueError(f"Completed Usage artifact changed: {path / name}")
    training, validation = training_scope(splits, fold)
    if (int(row["training_product_count"]), int(row["validation_product_count"])) != (
        len(training),
        len(validation),
    ):
        raise ValueError("Completed Usage row counts differ from its saved fold")
    counts = [int(training.usage.eq(label).sum()) for label in CLASSES]
    weights = effective_number_class_weights(counts, beta=0.999, cap=5.0)
    if config["class_counts"] != counts or not np.allclose(config["class_weights"], weights):
        raise ValueError("Completed Usage weights differ from its training fold")
    predictions = validate_oof(
        read_predictions(path / "oof_predictions.csv"),
        validation,
        target="usage",
        classes=CLASSES,
        run_ids_by_fold={fold: row["run_id"]},
    )
    if not predictions.source_dataset.eq(
        predictions.id.map(validation.set_index("id").source_dataset)
    ).all():
        raise ValueError("Completed Usage prediction sources differ")
    if COHORT_COLUMN in validation:
        if (
            COHORT_COLUMN not in predictions
            or not predictions[COHORT_COLUMN]
            .eq(predictions.id.map(validation.set_index("id")[COHORT_COLUMN]))
            .all()
        ):
            raise ValueError("Completed Usage prediction cohorts differ")
    measured = oof_metrics(predictions, CLASSES)
    for key in ("macro_f1", "nll", "brier", "ece_15"):
        if not np.isclose(measured[key], metrics[key], atol=1e-7, rtol=0):
            raise ValueError(f"Completed Usage predictions disagree with {key}")
    return {
        "run_id": row["run_id"],
        "run_dir": str(path),
        "metrics": metrics,
        "prediction_path": str(path / "oof_predictions.csv"),
        "robustness_path": str(path / "robustness.csv"),
        "metrics_path": str(path / "metrics.json"),
    }


def reusable_fold(*, fold, output_root, registry_path, splits, spec):
    rows = RunRegistry(registry_path)._read_rows()
    if any(row["experiment_id"] != EXPERIMENT for row in rows):
        raise ValueError("The Usage v2 run log contains another experiment")
    completed = [
        row for row in rows if row["status"] == "complete" and int(row["validation_fold"]) == fold
    ]
    if not completed:
        return None
    row = completed[-1]
    return _verified_completed_fold(
        row=row,
        path=Path(output_root) / ARTIFACT_DIRECTORY / "usage" / row["run_id"],
        splits=splits,
        fold=fold,
        spec=spec,
        split_digest=SPLIT_SHA256,
    )


def check_references(
    *, e8_directory, source_registry_path, previous_directory, previous_registry_path, root=ROOT
):
    """Read both frozen references for comparison; never initialize from their weights."""
    e8 = previous.check_e8_sources(
        directory=e8_directory, registry_path=source_registry_path, root=root
    )
    older, _ = previous.validate_dataset(root=root, check_images=False)
    rows = RunRegistry(previous_registry_path)._read_rows()
    expanded = {}
    for fold, run_id in enumerate(PREVIOUS_RUN_IDS):
        matched = [row for row in rows if row["run_id"] == run_id]
        if len(matched) != 1:
            raise ValueError(f"Expected one completed first-expansion source: {run_id}")
        result = _verified_completed_fold(
            row=matched[0],
            path=Path(previous_directory) / run_id,
            splits=older,
            fold=fold,
            spec=previous.expanded_usage_spec(),
            split_digest=previous.SPLIT_SHA256,
        )
        expanded[fold] = {
            "run_id": run_id,
            "predictions": read_predictions(result["prediction_path"]),
            "metrics": result["metrics"],
        }
    return {"teacher_e8": e8, "previous_expansion": expanded}


def build_comparison(predictions, *, splits, folds, references, contract):
    expected = get_samples(
        splits.loc[splits.partition.eq("development") & splits.cv_fold.isin(folds)], target="usage"
    )
    predictions = source_predictions(predictions, expected)
    scopes = source_metrics(predictions)
    teacher = predictions.loc[predictions.source_dataset.eq("teacher")]
    reference_scores = {}
    for name, source in references.items():
        old = pd.concat([source[fold]["predictions"] for fold in folds], ignore_index=True)
        if name == "previous_expansion":
            old = old.loc[old.source_dataset.eq("teacher")]
        if old.id.duplicated().any() or set(old.id) != set(teacher.id):
            raise ValueError("Teacher comparisons must use exactly the same validation IDs")
        old = old.set_index("id").loc[teacher.id].reset_index()
        if not np.array_equal(old.true_index.to_numpy(), teacher.true_index.to_numpy()):
            raise ValueError("Teacher reference labels differ")
        reference_scores[name] = oof_metrics(old, CLASSES)
    new_score = scopes["teacher"]["metrics"]
    per_class = []
    for index, row in enumerate(new_score["per_class"]):
        record = {"class_name": row["class_name"], "support": row["support"], "v2_f1": row["f1"]}
        for name, scores in reference_scores.items():
            record[f"{name}_f1"] = scores["per_class"][index]["f1"]
            record[f"change_from_{name}"] = row["f1"] - record[f"{name}_f1"]
        per_class.append(record)
    return {
        "folds": list(folds),
        "dataset": contract,
        "sources": scopes,
        "primary_selection_scope": "same original teacher validation IDs and nine Usage classes",
        "teacher_references": reference_scores,
        "teacher_macro_f1_changes": {
            name: new_score["macro_f1"] - scores["macro_f1"]
            for name, scores in reference_scores.items()
        },
        "teacher_per_class": per_class,
        "limits": [
            "External-source scores have a different source and class mix; they are diagnostic.",
            "Product-text and collection labels are not teacher annotations.",
            "A 49-image Smart Casual family makes new validation counts uneven across folds.",
            "No new NA images; teacher rare-class samples remain small.",
            "This run uses development only and does not select a deployment model.",
            "Earlier holdout/test results already exist; later checks cannot be claimed unseen.",
        ],
    }


def run_expanded_usage_v2(
    *,
    output_root,
    e8_directory,
    source_registry_path,
    previous_directory,
    previous_registry_path,
    root=ROOT,
    registry_path=None,
    registry_mirrors=(),
    folds=(0, 1, 2, 3, 4),
    resume=True,
):
    """Train five sequential scratch folds and compare only matched teacher validation rows."""
    import torch

    from fashion.train.task3_baseline import _aggregate_target, run_task3_baseline_fold

    folds = tuple(folds)
    if (
        not folds
        or len(set(folds)) != len(folds)
        or any(isinstance(fold, bool) or fold not in range(5) for fold in folds)
    ):
        raise ValueError("Choose distinct folds from 0,1,2,3,4")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before running Usage v2")
    root, output_root = Path(root), Path(output_root)
    destination = usage_registry_path(output_root)
    if registry_path is not None and Path(registry_path).resolve() != destination.resolve():
        raise ValueError(f"Usage v2 must use its own run log: {destination}")
    registry_path = destination
    protected_logs = {
        (output_root / "results/runs.csv").resolve(),
        Path(source_registry_path).resolve(),
        Path(previous_registry_path).resolve(),
        registry_path.resolve(),
    }
    if any(Path(mirror).resolve() in protected_logs for mirror in registry_mirrors):
        raise ValueError("Use only a separate local results/runs.csv mirror")
    for mirror in registry_mirrors:
        mirror_rows = RunRegistry(mirror)._read_rows()
        if any(row["experiment_id"] != EXPERIMENT for row in mirror_rows):
            raise ValueError("A local mirror cannot overwrite another experiment's run log")
    rows = RunRegistry(registry_path)._read_rows()
    if any(row["experiment_id"] != EXPERIMENT for row in rows):
        raise ValueError("The Usage v2 run log contains another experiment")
    splits, contract = validate_dataset(root=root, check_images=True)
    references = check_references(
        e8_directory=e8_directory,
        source_registry_path=source_registry_path,
        previous_directory=previous_directory,
        previous_registry_path=previous_registry_path,
        root=root,
    )
    spec = expanded_usage_spec()
    experiment_dir = output_root / ARTIFACT_DIRECTORY / "usage"
    write_json(
        {
            "dataset": contract,
            "baseline": Task3BaselineConfig(target="usage").to_dict(),
            "child": spec.to_dict(),
            "references": {
                name: {str(fold): record["run_id"] for fold, record in source.items()}
                for name, source in references.items()
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
            print(f"Reusing verified Usage v2 fold {fold}: {result['run_id']}", flush=True)
        results.append(result)
        torch.cuda.empty_cache()
    predictions = pd.concat(
        [read_predictions(r["prediction_path"]) for r in results], ignore_index=True
    )
    comparison = build_comparison(
        predictions, splits=splits, folds=folds, references=references, contract=contract
    )
    aggregate_name = (
        "aggregate"
        if set(folds) == set(range(5))
        else ("aggregate_folds_" + "_".join(str(fold) for fold in sorted(folds)))
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
    comparison_path = experiment_dir / aggregate_name / "teacher_comparison.json"
    write_json(comparison, comparison_path)
    pd.DataFrame(comparison["teacher_per_class"]).to_csv(
        comparison_path.with_name("teacher_per_class.csv"), index=False
    )
    return {
        "fold_results": results,
        "aggregate": aggregate,
        "comparison": comparison,
        "comparison_path": str(comparison_path),
        "registry_path": str(registry_path),
    }


def save_learning_curves(result, *, path):
    """Show the combined validation history; final teacher-only scores stay separate."""
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for run in result["fold_results"]:
        history = pd.read_csv(Path(run["run_dir"]) / "history.csv")
        label = f"Fold {run['metrics']['validation_fold']}"
        for axis, column in zip(axes, ("validation_macro_f1", "validation_loss"), strict=True):
            axis.plot(history["epoch"], history[column], label=label, linewidth=1.6)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Nine-class macro-F1")
    axes[0].set_title("Combined validation score")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Ordinary cross-entropy loss")
    axes[1].set_title("Combined validation loss")
    axes[1].legend(frameon=False)
    figure.suptitle("Usage E8 v2 — teacher + 687 added images")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--e8-directory", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--previous-directory", type=Path, required=True)
    parser.add_argument("--previous-registry", type=Path, required=True)
    parser.add_argument("--local-registry", type=Path)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_expanded_usage_v2(
        root=args.root,
        output_root=args.output_root,
        e8_directory=args.e8_directory,
        source_registry_path=args.source_registry,
        previous_directory=args.previous_directory,
        previous_registry_path=args.previous_registry,
        registry_mirrors=(args.local_registry,) if args.local_registry else (),
        folds=args.folds,
        resume=not args.no_resume,
    )
    print(result["comparison_path"])


if __name__ == "__main__":
    main()
