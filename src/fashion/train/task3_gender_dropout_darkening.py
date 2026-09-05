"""Two-fold scratch refinement: add G-D1 darkening to completed G-Drop30."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import _reusable_fold, _write_json, dataset_v2_spec
from fashion.train.task3_decisions import oof_metrics, paired_family_bootstrap
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_ieee import POLICY, evaluate_gender_ieee
from fashion.train.task3_gender_narrow import (
    FOLDS,
    MEMORY_LIMIT,
    TRAINING_PRECISION,
    check_gender_narrow_sources,
    evaluate_gender_narrow_screen,
    require_narrow_prerequisites,
)
from fashion.train.task3_gender_weight_decay import _pool, _robustness, _verify_baseline_controls

NAME = "gender_dropout_030_mild_darkening"
RULE_VERSION = "gdrop030dark_loss003_gap005_v1"
PARENT_RUN_IDS = (
    "t3_gender_dropout_030_gender_smallcnngem3_f0_s2753_2ab5e633206b_20260905T111550Zce42ad",
    "t3_gender_dropout_030_gender_smallcnngem3_f4_s2753_2ab5e633206b_20260905T112541Z8145b5",
)
STRONGER_NAME = "gender_dropout_045_mild_darkening"
GRAYSCALE_NAME = "gender_dropout_030_mild_darkening_grayscale_010"
DARKENING_RUN_IDS = (
    "t3_gender_dropout_030_mild_darkening_gender_smallcnngem3_f0_s2753_cfbed3f0fed4_20260905T123721Z5d546f",
    "t3_gender_dropout_030_mild_darkening_gender_smallcnngem3_f4_s2753_cfbed3f0fed4_20260905T124728Z7dfb12",
)


def _direct_parent_group(name):
    if name == NAME:
        return "Drop30", PARENT_RUN_IDS
    if name in {STRONGER_NAME, GRAYSCALE_NAME}:
        return "Drop30Dark", DARKENING_RUN_IDS
    raise ValueError("Unknown frozen dropout refinement")


def dropout_darkening_config(spec, *, fold, device_name):
    """Keep full-width G2 controls; dropout and darkening live in the child spec."""
    _, parents = _direct_parent_group(spec.name)
    if spec.to_dict() != dataset_v2_spec(spec.name, parents).to_dict():
        raise ValueError("Dropout plus darkening requires the frozen recipe and completed parents")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("Dropout plus darkening requires CUDA and only folds 0 and 4")
    return Task3BaselineConfig(target="gender")


def _verify_training_evidence(run, spec, parent_id, evidence, *, audit_sha256=None):
    _verify_baseline_controls(run["config"], Task3BaselineConfig(target="gender"))
    config = run["config"]
    if config.get("child_experiment") != spec.to_dict() or config.get("parent_run_id") != parent_id:
        raise ValueError("Dropout refinement configuration or direct parent disagrees")
    if (
        config.get("training_precision_settings") != TRAINING_PRECISION
        or config.get("precision_evidence_sha256") != evidence["artifact_sha256"]
    ):
        raise ValueError("Dropout refinement training precision evidence changed")
    if audit_sha256 is not None and config.get("refinement_prerequisite_sha256") != audit_sha256:
        raise ValueError("Dropout refinement source audit changed")


def check_gender_dropout_darkening_sources(
    *,
    g2_directory,
    e6_directory,
    dropout_directory,
    source_registry_path,
    precision_directory,
    root=ROOT,
    darkening_directory=None,
    experiment_name=NAME,
):
    """Check all saved evidence without fitting; failed research parents are allowed."""
    _, parents = _direct_parent_group(experiment_name)
    if experiment_name in {STRONGER_NAME, GRAYSCALE_NAME} and darkening_directory is None:
        raise ValueError("The completed 04aa darkening directory is required")
    if experiment_name == NAME and darkening_directory is not None:
        raise ValueError("The original 04aa recipe does not take darkening parents")
    sources, classes, parent_spec, evidence = check_gender_narrow_sources(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        source_registry_path=source_registry_path,
        precision_directory=precision_directory,
        root=root,
        experiment_name="gender_dropout_030",
    )
    registry = pd.read_csv(source_registry_path, keep_default_na=False)
    splits = load_splits(Path(root) / "data/processed/splits.csv")
    sources["Drop30"] = {}
    for fold, run_id in zip(FOLDS, PARENT_RUN_IDS, strict=True):
        directory = Path(dropout_directory) / run_id
        run = inspect_gender_run(
            directory, registry=registry, splits=splits, classes=classes, root=root
        )
        if run["fold"] != fold or run["run_id"] != run_id:
            raise ValueError("Completed dropout parent is assigned to the wrong fold")
        _verify_training_evidence(run, parent_spec, sources["G2"][fold]["run_id"], evidence)
        run["directory"] = str(directory)
        sources["Drop30"][fold] = run
    if experiment_name in {STRONGER_NAME, GRAYSCALE_NAME}:
        _add_completed_darkening_parents(
            sources,
            classes=classes,
            evidence=evidence,
            directory=darkening_directory,
            registry=registry,
            splits=splits,
            root=root,
        )
    return sources, classes, dataset_v2_spec(experiment_name, parents), evidence


def _add_completed_darkening_parents(
    sources,
    *,
    classes,
    evidence,
    directory,
    registry,
    splits,
    root,
):
    """Bind further refinements to both exact 04aa runs and their original source audit."""
    directory = Path(directory)
    audit = directory / "source_audit.json"
    previous = json.loads(audit.read_text())["identity"]
    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    if _source_identity(sources, spec, evidence, previous["paths"], root=root) != previous:
        raise ValueError("The completed 04aa source audit differs from the verified parents")
    audit_sha256 = compute_sha256(audit)
    direct = {}
    for fold, run_id in zip(FOLDS, DARKENING_RUN_IDS, strict=True):
        run = inspect_gender_run(
            directory / run_id,
            registry=registry,
            splits=splits,
            classes=classes,
            root=root,
        )
        if run["fold"] != fold or run["run_id"] != run_id:
            raise ValueError("Completed darkening parent is assigned to the wrong fold")
        _verify_training_evidence(
            run,
            spec,
            sources["Drop30"][fold]["run_id"],
            evidence,
            audit_sha256=audit_sha256,
        )
        run["directory"] = str(directory / run_id)
        direct[fold] = run
    sources["Drop30Dark"] = direct


def _source_identity(sources, spec, evidence, paths, *, root):
    """Registry rows are checked afresh; appended child rows do not change this identity."""
    return {
        "spec": spec.to_dict(),
        "baseline_controls": dropout_darkening_config(spec, fold=0, device_name="cuda").to_dict(),
        "paths": {key: str(Path(value).resolve()) for key, value in paths.items()},
        "source_sha256": {
            run["run_id"]: run["sha256"] for group in sources.values() for run in group.values()
        },
        "precision_evidence_sha256": evidence["artifact_sha256"],
        "split_sha256": compute_sha256(Path(root) / "data/processed/splits.csv"),
        "folds": list(FOLDS),
        "rule_version": spec.to_dict()["screen_rule_version"],
        "training_precision": TRAINING_PRECISION,
        "comparison_precision": POLICY,
    }


def _save_source_audit(path, identity, *, registry_path):
    if path.exists():
        if json.loads(path.read_text())["identity"] != identity:
            raise ValueError(
                "Existing dropout refinement source audit differs; do not reuse this screen"
            )
        return
    _write_json(
        {
            "identity": identity,
            "initial_source_registry_sha256": compute_sha256(registry_path),
            "optimizer_steps_at_audit": 0,
        },
        path,
    )


def require_dropout_darkening_prerequisites(
    path,
    *,
    spec,
    fold,
    parent_run_directory=None,
    root=ROOT,
    device_name="cuda",
):
    """Direct trainer entry must pass the same source and runtime checks as Run All."""
    dropout_darkening_config(spec, fold=fold, device_name=device_name)
    if path is None:
        raise ValueError("The dropout refinement source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, checked_spec, evidence = check_gender_dropout_darkening_sources(
        **paths,
        root=root,
        experiment_name=spec.name,
    )
    if (
        checked_spec.to_dict() != spec.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or _source_identity(sources, spec, evidence, paths, root=root) != identity
    ):
        raise ValueError("Dropout refinement prerequisite evidence changed")
    parent_group, _ = _direct_parent_group(spec.name)
    parent = Path(sources[parent_group][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("Dropout refinement requires its verified direct parent directory")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def compare_with_dropout(child, dropout, classes, *, repetitions=10_000):
    """Descriptive incremental comparisons; the acceptance gates remain versus G2/E6."""
    if set(child) != set(FOLDS) or set(dropout) != set(FOLDS):
        raise ValueError("Incremental comparison requires exactly folds 0 and 4")
    for group in (child, dropout):
        if any(run["metrics"].get("comparison_precision") != POLICY for run in group.values()):
            raise ValueError("Incremental comparison requires matched IEEE evaluation")
    candidate, parent = (oof_metrics(_pool(group), classes) for group in (child, dropout))
    candidate_class = {row["class_name"]: row["f1"] for row in candidate["per_class"]}
    parent_class = {row["class_name"]: row["f1"] for row in parent["per_class"]}
    folds, corruptions = [], []
    # Validate corruption coverage and score-change arithmetic before summarizing raw F1.
    child_changes, parent_changes = _robustness(child), _robustness(dropout)
    for fold in FOLDS:
        c, p = child[fold]["metrics"], dropout[fold]["metrics"]
        row = {"fold": fold}
        for label, key in (
            ("train_f1", "final_train_eval_macro_f1"),
            ("validation_f1", "macro_f1"),
            ("gap", "final_train_validation_macro_f1_gap"),
        ):
            row[f"candidate_{label}"] = c[key]
            row[f"dropout_{label}"] = p[key]
            row[f"delta_{label}"] = c[key] - p[key]
        row["gap_reduction"] = -row["delta_gap"]
        folds.append(row)
        c_scores = child[fold]["robustness"].set_index("corruption")
        p_scores = dropout[fold]["robustness"].set_index("corruption")
        for corruption in child_changes.index:
            c_raw, p_raw = (
                float(c_scores.loc[corruption, "macro_f1"]),
                float(p_scores.loc[corruption, "macro_f1"]),
            )
            c_change, p_change = c_raw - c["macro_f1"], p_raw - p["macro_f1"]
            corruptions.append(
                {
                    "fold": fold,
                    "corruption": corruption,
                    "candidate_clean_f1": c["macro_f1"],
                    "dropout_clean_f1": p["macro_f1"],
                    "candidate_corrupted_f1": c_raw,
                    "dropout_corrupted_f1": p_raw,
                    "raw_corrupted_delta": c_raw - p_raw,
                    "candidate_induced_change": c_change,
                    "dropout_induced_change": p_change,
                    "induced_change_delta": c_change - p_change,
                }
            )
    return {
        "comparison": "candidate minus matched IEEE Drop30; descriptive, no extra acceptance gates",
        "candidate": candidate,
        "dropout": parent,
        "validation_delta": candidate["macro_f1"] - parent["macro_f1"],
        "validation_interval": paired_family_bootstrap(
            _pool(child),
            _pool(dropout),
            classes=classes,
            repetitions=repetitions,
            seed=2753,
        ),
        "class_f1_delta": {name: candidate_class[name] - parent_class[name] for name in classes},
        "folds": folds,
        "corruptions": corruptions,
        "mean_induced_change_delta": {
            name: child_changes[name] - parent_changes[name] for name in child_changes.index
        },
        "independent_test_evidence": False,
    }


def _train_fold(**kwargs):
    from fashion.train.task3_baseline import run_task3_baseline_fold

    return run_task3_baseline_fold("gender", **kwargs)


def run_gender_dropout_darkening_screen(
    *,
    g2_directory,
    e6_directory,
    dropout_directory,
    source_registry_path,
    precision_directory,
    output_root,
    registry_path,
    registry_mirrors=(),
    root=ROOT,
    device_name="cuda",
    darkening_directory=None,
    experiment_name=NAME,
):
    """Fit or verify two registered runs, evaluate matched references, and stop for review."""
    root, output_root = Path(root), Path(output_root)
    paths = dict(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        dropout_directory=dropout_directory,
        source_registry_path=source_registry_path,
        precision_directory=precision_directory,
    )
    parent_group, _ = _direct_parent_group(experiment_name)
    if darkening_directory is not None:
        paths["darkening_directory"] = darkening_directory
    sources, classes, spec, evidence = check_gender_dropout_darkening_sources(
        **paths,
        root=root,
        experiment_name=experiment_name,
    )
    dropout_darkening_config(spec, fold=0, device_name=device_name)
    require_narrow_prerequisites(precision_directory, root=root)
    destination = output_root / spec.artifact_dir / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    audit = destination / "source_audit.json"
    _save_source_audit(
        audit,
        _source_identity(sources, spec, evidence, paths, root=root),
        registry_path=source_registry_path,
    )
    audit_sha256 = compute_sha256(audit)
    splits = load_splits(root / "data/processed/splits.csv")
    matched = {name: {} for name in ("G2", "E6", parent_group)}

    def evaluate(run):
        return evaluate_gender_ieee(
            run,
            splits=splits,
            classes=classes,
            root=root,
            output=destination / "comparison_ieee_v2" / run["run_id"],
        )

    for name in matched:
        for fold in FOLDS:
            matched[name][fold] = evaluate(sources[name][fold])
    child = {}
    for fold in FOLDS:
        result = _reusable_fold(spec, fold, output_root=output_root)
        if result is None:
            result = _train_fold(
                validation_fold=fold,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                device_name=device_name,
                child_spec=spec,
                parent_run_directory=sources[parent_group][fold]["directory"],
                prerequisite_path=audit,
            )
        run = inspect_gender_run(
            Path(result["run_dir"]),
            registry=pd.read_csv(registry_path, keep_default_na=False),
            splits=splits,
            classes=classes,
            root=root,
        )
        if run["fold"] != fold:
            raise ValueError("Dropout refinement candidate is assigned to the wrong fold")
        _verify_training_evidence(
            run,
            spec,
            sources[parent_group][fold]["run_id"],
            evidence,
            audit_sha256=audit_sha256,
        )
        memory = run["metrics"]["peak_memory_bytes"]
        if not np.isfinite(memory) or not 0 < memory < MEMORY_LIMIT:
            stopped = {
                "status": "fail",
                "phase": "screen",
                "fold": fold,
                "reason": "GPU memory must stay below 3 GB; later folds were not started.",
            }
            _write_json(stopped, destination / "screen_decision.json")
            return stopped
        run["directory"] = result["run_dir"]
        child[fold] = evaluate(run)
    report = evaluate_gender_narrow_screen(child, matched, classes, experiment_name=experiment_name)
    report["registry_and_artifact_integrity"] = True
    report["run_ids"] = {str(f): child[f]["run_id"] for f in FOLDS}
    report["incremental_comparison"] = compare_with_dropout(child, matched[parent_group], classes)
    if experiment_name in {STRONGER_NAME, GRAYSCALE_NAME}:
        report["direct_parent_comparison"] = {
            "name": "Drop30Dark",
            "classifier_dropout": 0.30,
            "training_augmentation": dataset_v2_spec(NAME, PARENT_RUN_IDS).training_augmentation,
            "run_ids": {str(f): matched[parent_group][f]["run_id"] for f in FOLDS},
        }
        report["incremental_comparison"]["comparison"] = (
            "candidate minus matched IEEE Drop30Dark (dropout 0.30 plus mild darkening); "
            "descriptive, no extra acceptance gates"
        )
    _write_json(report, destination / "screen_decision.json")
    _write_json(report["incremental_comparison"], destination / "incremental_comparison.json")
    pd.DataFrame(report["folds"]).to_csv(destination / "clean_gap_comparison.csv", index=False)
    pd.DataFrame(report["incremental_comparison"]["corruptions"]).to_csv(
        destination / "dropout_corruption_comparison.csv",
        index=False,
    )
    _pool(child).sort_values("id").to_csv(destination / "ieee_oof_predictions.csv", index=False)
    return report
