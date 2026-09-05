"""Evaluate the completed G2 confirmation from saved files, without training."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import (
    DECISION_BOOTSTRAP_REPETITIONS,
    check,
    decision,
    oof_metrics,
    paired_family_bootstrap,
    robustness_changes,
    validate_oof,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_metrics(saved: dict[str, Any], calculated: dict[str, Any]) -> None:
    for key in ("macro_f1", "nll", "ece_15", "brier", "accuracy"):
        if not np.isclose(float(saved[key]), float(calculated[key]), atol=1e-7, rtol=0):
            raise ValueError(f"saved {key} disagrees with OOF probabilities")
    if saved["confusion_matrix"] != calculated["confusion_matrix"]:
        raise ValueError("saved confusion matrix disagrees with OOF predictions")


def inspect_gender_run(
    run_dir: Path,
    *,
    registry: pd.DataFrame,
    splits: pd.DataFrame,
    classes: list[str],
    root: Path,
) -> dict[str, Any]:
    """Verify source hashes and canonical scope; never unpickle a checkpoint."""
    run_id = run_dir.name
    matches = registry.loc[registry["run_id"].eq(run_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one registry row for {run_id}; found {len(matches)}")
    row = matches.iloc[0]
    if row["status"] != "complete" or str(row["scratch"]).lower() != "true":
        raise ValueError(f"run is not complete and scratch-trained: {run_id}")
    if str(row["debug"]).lower() != "false" or int(row["seed"]) != 2753:
        raise ValueError(f"run is debug or uses a different seed: {run_id}")
    paths = {
        name: run_dir / name
        for name in (
            "config.json",
            "normalization.json",
            "history.csv",
            "metrics.json",
            "oof_predictions.csv",
            "robustness.csv",
            "final_epoch.pt",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: compute_sha256(path) for name, path in paths.items()}
    for name, field in (
        ("oof_predictions.csv", "prediction_sha256"),
        ("final_epoch.pt", "checkpoint_sha256"),
    ):
        if hashes[name] != row[field]:
            raise ValueError(f"registry hash mismatch: {run_id}/{name}")
    for name, field in (
        ("config.json", "config_path"),
        ("history.csv", "history_path"),
        ("oof_predictions.csv", "prediction_path"),
        ("final_epoch.pt", "checkpoint_path"),
    ):
        recorded = Path(str(row[field]))
        if recorded.name != name or recorded.parent.name != run_id:
            raise ValueError(f"registry artifact location disagrees: {run_id}/{field}")
    if paths["final_epoch.pt"].stat().st_size != int(row["checkpoint_bytes"]):
        raise ValueError(f"checkpoint size differs from registry: {run_id}")
    with zipfile.ZipFile(paths["final_epoch.pt"]) as archive:
        if archive.testzip() is not None or not any(
            name.endswith("/data.pkl") for name in archive.namelist()
        ):
            raise ValueError(f"invalid PyTorch checkpoint archive: {run_id}")
    for field, path in (
        ("split_digest", root / "data/processed/splits.csv"),
        ("label_map_digest", root / "data/processed/label_maps.json"),
    ):
        if str(row[field]) != compute_sha256(path):
            raise ValueError(f"registry {field} disagrees with local data: {run_id}")
    metrics = _json(paths["metrics.json"])
    config = _json(paths["config.json"])
    if metrics != json.loads(row["metrics_json"]):
        raise ValueError(f"metrics file differs from registered metrics: {run_id}")
    fold = int(metrics["validation_fold"])
    if (
        metrics["run_id"] != run_id
        or int(row["validation_fold"]) != fold
        or row["target"] != "gender"
        or metrics["target"] != "gender"
        or config["target"] != "gender"
        or int(config["seed"]) != 2753
    ):
        raise ValueError(f"run metadata disagrees: {run_id}")
    controls = {f.name: config[f.name] for f in fields(Task3BaselineConfig)}
    controls["channels"] = tuple(controls["channels"])
    baseline = Task3BaselineConfig(**controls).to_dict()
    digest_input = {"baseline_controls": baseline, "child_experiment": config["child_experiment"]}
    digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    if digest != row["config_hash"] or f"_{digest}_" not in run_id:
        raise ValueError(f"configuration digest disagrees with the registry: {run_id}")
    lineage = [config["parent_run_id"]]
    if json.loads(row["parent_run_ids"]) != lineage or metrics["parent_run_ids"] != lineage:
        raise ValueError(f"parent lineage disagrees: {run_id}")
    training, expected = get_cv_split(splits, fold)
    training = get_samples(training, target="gender")
    expected = get_samples(expected, target="gender")
    for prefix, frame in (("training", training), ("validation", expected)):
        if (
            int(row[f"{prefix}_product_count"]) != len(frame)
            or int(row[f"{prefix}_family_count"]) != frame["product_family_group"].nunique()
        ):
            raise ValueError(f"registry {prefix} scope count disagrees: {run_id}")
    counts = [int(training["gender"].eq(label).sum()) for label in classes]
    if config["class_counts"] != counts or metrics["class_counts"] != counts:
        raise ValueError(f"fold-training class counts disagree: {run_id}")
    normalization = _json(paths["normalization.json"])
    if (
        normalization["fit_scope"] != "fold_training_content_pixels_only"
        or int(normalization["validation_fold"]) != fold
        or not normalization["padding_excluded"]
    ):
        raise ValueError(f"normalization scope disagrees: {run_id}")
    history = pd.read_csv(paths["history.csv"], keep_default_na=False)
    if (
        history["epoch"].tolist() != list(range(1, 31))
        or config["checkpoint_rule"] != "final_epoch"
        or metrics.get("selected_epoch", 30) != 30
    ):
        raise ValueError(f"G2/E6 final-epoch history is incomplete: {run_id}")
    predictions = validate_oof(
        pd.read_csv(paths["oof_predictions.csv"], keep_default_na=False),
        expected,
        target="gender",
        classes=classes,
        run_ids_by_fold={fold: run_id},
    )
    recalculated = oof_metrics(predictions, classes)
    _same_metrics(metrics, recalculated)
    if not np.isclose(
        metrics["final_train_eval_macro_f1"] - recalculated["macro_f1"],
        metrics["final_train_validation_macro_f1_gap"],
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError(f"clean train/validation gap arithmetic disagrees: {run_id}")
    robustness = pd.read_csv(paths["robustness.csv"], keep_default_na=False)
    robustness_changes(
        robustness, clean_by_fold={fold: recalculated["macro_f1"]}, run_ids_by_fold={fold: run_id}
    )
    return {
        "run_id": run_id,
        "fold": fold,
        "metrics": metrics,
        "config": config,
        "predictions": predictions,
        "robustness": robustness,
        "sha256": hashes,
        "config_digest": digest,
    }


def audit_g2_confirmation(
    *,
    g2_directory: str | Path,
    e6_directory: str | Path,
    registry_path: str | Path,
    report_directory: str | Path,
    root: str | Path = ROOT,
    bootstrap_repetitions: int = DECISION_BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Check both confirmation scopes and save every frozen G2 rule separately.

    Invalid source evidence raises before any acceptance report is written.
    The source registry, checkpoints, metrics, and notebooks are never modified.
    """
    root, g2_directory, e6_directory = Path(root), Path(g2_directory), Path(e6_directory)
    report_directory = Path(report_directory)
    splits = load_splits(root / "data/processed/splits.csv")
    classes = list(load_label_maps(root / "data/processed/label_maps.json")["gender"]["classes"])
    registry = pd.read_csv(registry_path, keep_default_na=False)
    bundles: dict[str, dict[int, dict[str, Any]]] = {}
    for name, directory, prefix in (
        ("G2", g2_directory, "t3_gender_v2_g2_translation_"),
        ("E6", e6_directory, "t3_gender_e6_gem_p3_"),
    ):
        found: dict[int, dict[str, Any]] = {}
        for run_dir in sorted(directory.glob(prefix + "*")):
            bundle = inspect_gender_run(
                run_dir, registry=registry, splits=splits, classes=classes, root=root
            )
            if bundle["fold"] in found:
                raise ValueError(f"multiple {name} candidates for fold {bundle['fold']}")
            found[bundle["fold"]] = bundle
        if set(found) != set(range(5)):
            raise ValueError(f"{name} needs exactly five verified folds; found {sorted(found)}")
        bundles[name] = found
    parent_ids = [bundles["E6"][fold]["run_id"] for fold in range(5)]
    spec = dataset_v2_spec("gender_v2_translation", parent_ids).to_dict()
    for fold, bundle in bundles["G2"].items():
        if bundle["config"]["child_experiment"] != spec:
            raise ValueError("G2 configuration differs from its frozen one-change specification")
        if bundle["config"]["parent_run_id"] != parent_ids[fold]:
            raise ValueError("G2 does not use the matched E6 fold")
    pooled = {
        name: pd.concat([runs[f]["predictions"] for f in range(5)], ignore_index=True)
        for name, runs in bundles.items()
    }
    scopes: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for scope, folds in (("fresh_folds_1_2_3", (1, 2, 3)), ("five_fold", tuple(range(5)))):
        child = pooled["G2"].loc[pooled["G2"]["cv_fold"].isin(folds)]
        parent = pooled["E6"].loc[pooled["E6"]["cv_fold"].isin(folds)]
        child_metrics, parent_metrics = oof_metrics(child, classes), oof_metrics(parent, classes)
        aggregate_dir = g2_directory / ("aggregate_" + scope)
        _same_metrics(_json(aggregate_dir / "metrics.json"), child_metrics)
        expected = get_samples(splits, partition="development", target="gender")
        expected = expected.loc[expected["cv_fold"].isin(folds)]
        aggregate_oof = validate_oof(
            pd.read_csv(aggregate_dir / "oof_predictions.csv", keep_default_na=False),
            expected,
            target="gender",
            classes=classes,
            run_ids_by_fold={f: bundles["G2"][f]["run_id"] for f in folds},
        )
        pd.testing.assert_frame_equal(
            aggregate_oof, child.sort_values("id").reset_index(drop=True), check_exact=True
        )
        interval = paired_family_bootstrap(
            child, parent, classes=classes, repetitions=bootstrap_repetitions
        )
        changes = np.array(
            [
                bundles["G2"][f]["metrics"]["macro_f1"] - bundles["E6"][f]["metrics"]["macro_f1"]
                for f in folds
            ]
        )
        scopes[scope] = {
            "folds": list(folds),
            "candidate": child_metrics,
            "parent": parent_metrics,
            "paired_family_bootstrap": interval,
        }
        fresh = len(folds) == 3
        score = float(child_metrics["macro_f1"])
        delta = score - float(parent_metrics["macro_f1"])
        checks.extend(
            [
                check(
                    scope + ".macro_f1",
                    delta if fresh else score,
                    "> 0 matched change" if fresh else ">= 0.7462",
                    delta > 0 if fresh else score >= 0.7462,
                ),
                check(
                    scope + ".family_bootstrap_lower",
                    interval["lower_95"],
                    "> 0",
                    interval["lower_95"] > 0,
                ),
                check(
                    scope + ".improved_folds",
                    int((changes > 0).sum()),
                    ">= 2" if fresh else ">= 4",
                    (changes > 0).sum() >= (2 if fresh else 4),
                ),
                check(
                    scope + ".worst_fold_change",
                    float(changes.min()),
                    ">= -0.005",
                    changes.min() >= -0.005,
                ),
            ]
        )
    child_metrics = scopes["five_fold"]["candidate"]
    fold_rows = []
    for f in range(5):
        c, p = bundles["G2"][f]["metrics"], bundles["E6"][f]["metrics"]
        fold_rows.append(
            {
                "fold": f,
                "candidate_f1": c["macro_f1"],
                "parent_f1": p["macro_f1"],
                "candidate_gap": c["final_train_validation_macro_f1_gap"],
                "parent_gap": p["final_train_validation_macro_f1_gap"],
            }
        )
    frame = pd.DataFrame(fold_rows)
    sd = float(frame["candidate_f1"].std(ddof=1))
    gap = float(frame["candidate_gap"].mean())
    reduction = float(frame["parent_gap"].mean() - gap)
    worsening = float((frame["candidate_gap"] - frame["parent_gap"]).max())
    by_class = {row["class_name"]: row["f1"] for row in child_metrics["per_class"]}
    minority = float(np.mean([by_class[c] for c in ("Boys", "Girls", "Unisex")]))
    checks.extend(
        [
            check("fold_sd", sd, "<= 0.0209", sd <= 0.0209),
            check("clean_gap", gap, "<= 0.2466", gap <= 0.2466),
            check("gap_reduction", reduction, ">= 0.020", reduction >= 0.020),
            check("worst_fold_gap_worsening", worsening, "<= 0.005", worsening <= 0.005),
            check("minority_mean_f1", minority, ">= 0.6013", minority >= 0.6013),
            check("Men_f1", by_class["Men"], ">= 0.9188", by_class["Men"] >= 0.9188),
            check("Women_f1", by_class["Women"], ">= 0.8950", by_class["Women"] >= 0.8950),
            check("nll", child_metrics["nll"], "<= 0.4525", child_metrics["nll"] <= 0.4525),
            check(
                "ece_15", child_metrics["ece_15"], "<= 0.0842", child_metrics["ece_15"] <= 0.0842
            ),
        ]
    )
    robust = {}
    for name, runs in bundles.items():
        robust[name] = robustness_changes(
            pd.concat([runs[f]["robustness"] for f in range(5)], ignore_index=True),
            clean_by_fold={f: runs[f]["metrics"]["macro_f1"] for f in range(5)},
            run_ids_by_fold={f: runs[f]["run_id"] for f in range(5)},
        )
    for corruption, change in (robust["G2"] - robust["E6"]).items():
        bound = 0.030 if corruption == "translation_003" else -0.020
        checks.append(
            check("robustness." + corruption, float(change), f">= {bound}", change >= bound)
        )
    size_fixed = all(
        int(bundle["metrics"]["parameter_count"]) == baseline_parameter_count("gender")
        and int(bundle["config"]["parameter_count"]) == baseline_parameter_count("gender")
        for runs in bundles.values()
        for bundle in runs.values()
    )
    checks.append(
        check(
            "model_size_fixed",
            baseline_parameter_count("gender"),
            "same 390181 trainable parameters as E6",
            size_fixed,
        )
    )
    checks.append(
        check(
            "registry_and_artifact_integrity",
            10,
            "all five G2 and five matched E6 bundles verified",
            True,
        )
    )
    report = {
        "status": decision(checks),
        "checks": checks,
        "scopes": scopes,
        "folds": fold_rows,
        "robustness_mean_changes": {k: v.to_dict() for k, v in robust.items()},
        "run_ids": {k: [v[f]["run_id"] for f in range(5)] for k, v in bundles.items()},
        "registry_sha256": compute_sha256(registry_path),
        "artifact_sha256": {b["run_id"]: b["sha256"] for v in bundles.values() for b in v.values()},
        "no_training": True,
        "checkpoint_inference_performed": False,
        "checkpoint_verification": "registry_sha256_size_and_zip_crc; no unpickling",
        "limitations": [
            "Repeated development-fold selection is not final-test evidence.",
            "One seed; bootstrap does not measure training-seed variation.",
            "Clean-training and corruption scores are verified saved-run evidence; "
            "no new inference.",
        ],
    }
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "g2_decision.json").write_text(json.dumps(report, indent=2) + "\n")
    write_deterministic_csv(pd.DataFrame(checks), report_directory / "g2_gates.csv", index=False)
    write_deterministic_csv(frame, report_directory / "g2_fold_comparison.csv", index=False)
    return report
