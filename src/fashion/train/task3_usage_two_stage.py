"""Frozen Usage U3: 30 epochs of feature learning, then a 10-epoch balanced head."""

import hashlib
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.registry import RunRegistry
from fashion.train.task3_decisions import (
    CORE_CORRUPTIONS,
    check,
    decision,
    oof_metrics,
    paired_family_bootstrap,
    probability_columns,
    robustness_changes,
    validate_oof,
)
from fashion.train.task3_experiments import effective_number_class_weights
from fashion.train.task3_usage_two_stage_runtime import run_fold_process

EXPERIMENT = "t3_usage_u3_two_stage_cnn"
FOLDS = (0, 4)
MEMORY_LIMIT = 7 * 1024**3
HOST_MEMORY_LIMIT = 16 * 1024**3
SECONDS_LIMIT = 90 * 60
RULE = "usage_two_stage_e2_routes_v1"
CLASSES = ("Casual", "Ethnic", "Formal", "Home", "NA", "Party", "Smart Casual", "Sports", "Travel")
RARE_CLASSES = ("NA", "Party", "Smart Casual", "Travel")
E2_RUN_IDS = {
    0: "t3_usage_e2_class_balanced_ce_usage_smallcnn_f0_s2753_5461e048c3b3_20260830T115815Z356f6d",
    4: "t3_usage_e2_class_balanced_ce_usage_smallcnn_f4_s2753_5461e048c3b3_20260830T123218Z94db47",
}


def write_json(value, path):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(path)


def recipe():
    """No tuning knobs: the scored checkpoint is always the final Stage B epoch."""
    return {
        "experiment_id": EXPERIMENT,
        "baseline": Task3BaselineConfig(target="usage").to_dict(),
        "stage_a_epochs": 30,
        "stage_a_loss": "cross_entropy",
        "stage_b_epochs": 10,
        "stage_b_loss": "effective_number_cross_entropy",
        "stage_b_beta": 0.999,
        "stage_b_cap": 5.0,
        "stage_b_reset_seed": 2753,
        "stage_b_optimizer": "fresh_AdamW_head_only",
        "stage_b_schedule": "CosineAnnealingLR_10_epochs_0.001_to_0.00001",
        "stage_b_cache": "outer_training_only_clean_eval_features",
        "stage_b_batchnorm": "eval_frozen_parameters_and_buffers",
        "sample_weighting": "loss_only_one_visit_per_row_per_epoch",
        "classifier_dropout": 0.0,
        "checkpoint_policy": "final_stage_b_epoch_10",
        "stage_a_role": "mechanism_control_not_candidate_selection",
        "precision": "fp32_no_autocast_runtime_tf32_flags_recorded",
        "screen_folds": list(FOLDS),
        "seconds_limit": SECONDS_LIMIT,
        "host_memory_limit_bytes": HOST_MEMORY_LIMIT,
        "gpu_memory_limit_bytes": MEMORY_LIMIT,
        "rule_version": RULE,
    }


def configuration_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def training_scope(splits, fold):
    if fold not in FOLDS:
        raise ValueError("Usage two-stage screen allows only folds 0 and 4")
    train, validation = (get_samples(f, target="usage") for f in get_cv_split(splits, fold))
    if set(train.product_family_group) & set(validation.product_family_group):
        raise ValueError("A product family crosses the canonical fold")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def class_weights_for_training(training):
    counts = [int(training.usage.eq(name).sum()) for name in CLASSES]
    return counts, effective_number_class_weights(counts, beta=0.999, cap=5.0).tolist()


def save_predictions(frame, path):
    """Preserve float32 outputs exactly through CSV; score the persisted values."""
    frame = frame.copy()
    columns = ["confidence", *probability_columns(CLASSES)]
    frame[columns] = frame[columns].astype(np.float64)
    frame.to_csv(path, index=False)
    return oof_metrics(read_predictions(path), CLASSES)


def read_predictions(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def check_usage_two_stage_sources(*, e2_directory, registry_path, root=ROOT):
    """Read-only check of the two fixed E2 source bundles; no PyTorch needed."""
    root, directory = Path(root), Path(e2_directory)
    splits = load_splits(root / "data/processed/splits.csv")
    mapping = load_label_maps(root / "data/processed/label_maps.json")["usage"]
    if tuple(mapping["classes"]) != CLASSES or mapping["label_to_index"] != {
        name: i for i, name in enumerate(CLASSES)
    }:
        raise ValueError("The canonical nine-class Usage label map changed")
    registry = pd.read_csv(registry_path, keep_default_na=False)
    sources = {}
    for fold, run_id in E2_RUN_IDS.items():
        rows = registry[registry.run_id.eq(run_id)]
        if len(rows) != 1:
            raise ValueError(f"Expected one registered E2 source: {run_id}")
        row = rows.iloc[0]
        if (
            row.status != "complete"
            or row.target != "usage"
            or row.experiment_id != "t3_usage_class_balanced_smallcnn"
            or int(row.validation_fold) != fold
            or int(row.seed) != 2753
            or str(row.scratch).lower() != "true"
            or str(row.debug).lower() != "false"
        ):
            raise ValueError("Source is not a completed scratch E2 fold")
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
        for name, field in (
            ("final_epoch.pt", "checkpoint_sha256"),
            ("oof_predictions.csv", "prediction_sha256"),
        ):
            if hashes[name] != row[field]:
                raise ValueError(f"E2 artifact differs from registry: {name}")
        for name, field in (
            ("splits.csv", "split_digest"),
            ("label_maps.json", "label_map_digest"),
        ):
            if compute_sha256(root / "data/processed" / name) != row[field]:
                raise ValueError("E2 data contract changed")
        metrics = json.loads((path / "metrics.json").read_text())
        config = json.loads((path / "config.json").read_text())
        if metrics != json.loads(row.metrics_json):
            raise ValueError("E2 metrics differ from registry")
        base = Task3BaselineConfig(target="usage").to_dict()
        if any(config.get(k) != v for k, v in base.items()):
            raise ValueError("E2 baseline controls changed")
        child = config["child_experiment"]
        if (
            child["name"] != "usage_class_balanced"
            or child["class_weight_beta"] != 0.999
            or child["class_weight_cap"] != 5.0
            or child["training_augmentation"] != "none"
        ):
            raise ValueError("E2 loss recipe changed")
        digest = configuration_hash({"baseline_controls": base, "child_experiment": child})[:12]
        if row.config_hash != digest or f"_{digest}_" not in run_id:
            raise ValueError("E2 configuration digest differs from registry")
        training, expected = training_scope(splits, fold)
        counts, weights = class_weights_for_training(training)
        if config["class_counts"] != counts or not np.allclose(config["class_weights"], weights):
            raise ValueError("E2 weights differ from canonical training rows")
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
            if not np.isclose(metrics[key], measured[key], atol=1e-7, rtol=0):
                raise ValueError(f"E2 {key} does not match its saved probabilities")
        robust = pd.read_csv(path / "robustness.csv", keep_default_na=False)
        robustness_changes(
            robust, clean_by_fold={fold: measured["macro_f1"]}, run_ids_by_fold={fold: run_id}
        )
        sources[fold] = {
            "run_id": run_id,
            "sha256": hashes,
            "predictions": predictions,
            "metrics": measured,
            "robustness": robust,
        }
    pooled = oof_metrics(pd.concat([r["predictions"] for r in sources.values()]), CLASSES)
    if not np.isclose(pooled["macro_f1"], 0.40731939187942284, atol=1e-10, rtol=0):
        raise ValueError("Fixed E2 screen reference changed")
    return sources, splits


def evaluate_usage_two_stage(child, sources, splits, *, repetitions=10_000):
    """Reuse Usage's score/probability routes; never apply U2's forced-zero Home rule."""
    if set(child) != set(FOLDS) or set(sources) != set(FOLDS):
        raise ValueError("The screen needs both canonical folds 0 and 4")
    frames = []
    for fold, run in child.items():
        _, expected = training_scope(splits, fold)
        frames.append(
            validate_oof(
                run["predictions"],
                expected,
                target="usage",
                classes=CLASSES,
                run_ids_by_fold={fold: run["run_id"]},
            )
        )
    candidate = pd.concat(frames, ignore_index=True)
    parent = pd.concat([r["predictions"] for r in sources.values()], ignore_index=True)
    cm, pm = oof_metrics(candidate, CLASSES), oof_metrics(parent, CLASSES)
    interval = paired_family_bootstrap(
        candidate, parent, classes=CLASSES, repetitions=repetitions, seed=2753
    )
    improvements = {k: (pm[k] - cm[k]) / pm[k] if pm[k] > 0 else 0.0 for k in ("nll", "brier")}
    route_a = cm["macro_f1"] >= 0.417319 and interval["lower_95"] > 0
    route_b = cm["macro_f1"] >= 0.402319 and max(improvements.values()) >= 0.10
    checks = [
        check("canonical_oof_integrity", len(candidate), "exact folds 0 and 4", True),
        check(
            "route_a_or_b",
            {"a": route_a, "b": route_b},
            "F1>=.417319 and CI>0, or F1>=.402319 and NLL/Brier improves >=10%",
            bool(route_a or route_b),
        ),
        check("ece_15", cm["ece_15"], "<=.05", cm["ece_15"] <= 0.05),
    ]
    deltas = {
        c["class_name"]: c["f1"] - p["f1"]
        for c, p in zip(cm["per_class"], pm["per_class"], strict=True)
        if c["class_name"] != "Home"
    }
    checks.append(
        check(
            "class_no_harm", deltas, "each non-Home F1 delta >=-.03", min(deltas.values()) >= -0.03
        )
    )
    caps = []
    for label, frame in [
        ("pooled", candidate),
        *[(str(f), candidate[candidate.cv_fold.eq(f)]) for f in FOLDS],
    ]:
        for name in RARE_CLASSES:
            count, support = (
                int(frame.predicted_label.eq(name).sum()),
                int(frame.true_label.eq(name).sum()),
            )
            caps.append({"scope": label, "class": name, "predicted": count, "support": support})
    checks.append(
        check(
            "rare_prediction_cap",
            caps,
            "predictions<=5x support pooled/per fold",
            all(c["predicted"] <= 5 * c["support"] for c in caps),
        )
    )
    changes = []
    for group in (child, sources):
        changes.append(
            robustness_changes(
                pd.concat([r["robustness"] for r in group.values()], ignore_index=True),
                clean_by_fold={
                    f: oof_metrics(r["predictions"], CLASSES)["macro_f1"] for f, r in group.items()
                },
                run_ids_by_fold={f: r["run_id"] for f, r in group.items()},
            )
        )
    for name, delta in (changes[0] - changes[1]).items():
        checks.append(
            check(f"robustness.{name}", float(delta), "induced delta vs E2 >=-.02", delta >= -0.02)
        )
    for fold, run in child.items():
        m = run["metrics"]
        valid = (
            0 < m["fold_wall_seconds"] <= SECONDS_LIMIT
            and 0 < m["peak_host_memory_bytes"] <= HOST_MEMORY_LIMIT
            and 0 < m["peak_memory_bytes"] <= MEMORY_LIMIT
        )
        checks.append(
            check(
                f"fold_{fold}.resources",
                {
                    k: m[k]
                    for k in ("fold_wall_seconds", "peak_host_memory_bytes", "peak_memory_bytes")
                },
                "<=90 minutes, <=16 GiB host RSS and <=7 GiB GPU",
                bool(valid),
            )
        )
        checks.append(
            check(
                f"fold_{fold}.frozen_backbone",
                m["backbone_unchanged"],
                "Stage B parameters and BatchNorm buffers unchanged",
                m["backbone_unchanged"],
            )
        )
    return {
        "status": decision(checks),
        "rule_version": RULE,
        "checks": checks,
        "candidate_metrics": cm,
        "matched_parent_metrics": pm,
        "paired_family_bootstrap": interval,
        "probability_relative_improvements": improvements,
        "clean_gap_route": "unavailable: matched clean E2 training scores not measured",
        "independent_test_evidence": False,
        "screen_only": True,
        "next_step": "Review before any additional folds, refit or held-out evaluation.",
    }


def run_usage_two_stage_screen(
    *,
    e2_directory,
    source_registry_path,
    output_root,
    registry_path,
    root=ROOT,
    registry_mirrors=(),
):
    """Register, train and verify only two fixed runs; reuse complete bundles strictly."""
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    sources, splits = check_usage_two_stage_sources(
        e2_directory=e2_directory, registry_path=source_registry_path, root=root
    )
    output = output_root / "experiments" / EXPERIMENT / "usage"
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(f): {"run_id": r["run_id"], "sha256": r["sha256"]} for f, r in sources.items()
    }
    dependencies = [
        "task3_usage_two_stage.py",
        "task3_usage_two_stage_fit.py",
        "task3_usage_two_stage_runtime.py",
        "model.py",
        "data.py",
        "metrics.py",
        "task3_baseline.py",
        "task3_experiments.py",
        "task3_decisions.py",
        "config.py",
        "augmentation.py",
        "loss.py",
        "registry.py",
        "../config.py",
        "../data/dataset.py",
        "../data/images.py",
        "../data/hashing.py",
    ]
    contract = {
        "recipe": recipe(),
        "parents": source_hashes,
        "split_sha256": compute_sha256(root / "data/processed/splits.csv"),
        "label_map_sha256": compute_sha256(root / "data/processed/label_maps.json"),
        "code_sha256": {n: compute_sha256(root / "src/fashion/train" / n) for n in dependencies},
    }
    write_json(contract, output / "source_audit.json")
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    child = {}
    for fold in FOLDS:
        config = {**contract, "fold": fold}
        digest = configuration_hash(config)
        candidates = []
        if Path(registry_path).is_file():
            rows = pd.read_csv(registry_path, keep_default_na=False)
            candidates = rows[
                (rows.experiment_id == EXPERIMENT)
                & (rows.validation_fold.astype(str) == str(fold))
                & (rows.config_hash == digest)
                & (rows.status == "complete")
            ].run_id.tolist()
        if len(candidates) > 1:
            raise ValueError("Multiple complete two-stage runs match this fold; review first")
        if candidates:
            child[fold] = load_two_stage_run(output / candidates[0], config, registry_path, splits)
            continue
        stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        run_id = f"{EXPERIMENT}_f{fold}_s2753_{digest[:12]}_{stamp}"
        directory = output / run_id
        directory.mkdir()
        write_json(config, directory / "config.json")
        train, validation = training_scope(splits, fold)
        registry.start(
            {
                "run_id": run_id,
                "experiment_id": EXPERIMENT,
                "hypothesis_id": "separate_feature_learning_from_class_rebalancing",
                "parent_run_ids": [sources[fold]["run_id"]],
                "task": "task3",
                "target": "usage",
                "validation_fold": fold,
                "seed": 2753,
                "scratch": True,
                "debug": False,
                "submission_eligible": True,
                "config_hash": digest,
                "config_path": directory / "config.json",
                "split_digest": contract["split_sha256"],
                "label_map_digest": contract["label_map_sha256"],
                "training_product_count": len(train),
                "validation_product_count": len(validation),
                "training_family_count": train.product_family_group.nunique(),
                "validation_family_count": validation.product_family_group.nunique(),
                "model_family": "task3_small_cnn_two_stage",
                "parameter_count": baseline_parameter_count("usage"),
                "history_path": directory / "history.csv",
                "last_completed_stage": "registered_before_first_optimizer_step",
            }
        )
        write_json(
            {"run_id": run_id, "directory": str(directory), "root": str(root)},
            directory / "request.json",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["FASHION_PROJECT_ROOT"] = str(root)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            resources = run_fold_process(
                [
                    sys.executable,
                    "-m",
                    "fashion.train.task3_usage_two_stage_fit",
                    "--request",
                    str(directory / "request.json"),
                ],
                cwd=root,
                log_path=directory / "worker.log",
                seconds=SECONDS_LIMIT,
                memory_bytes=HOST_MEMORY_LIMIT,
                env=env,
            )
            metrics = json.loads((directory / "metrics.json").read_text())
            metrics.update(resources)
            if metrics["peak_memory_bytes"] > MEMORY_LIMIT or not metrics["backbone_unchanged"]:
                raise ValueError("Stage B violated the memory or frozen-backbone contract")
            write_json(metrics, directory / "metrics.json")
            artifacts = {
                str(p.relative_to(directory)): compute_sha256(p)
                for p in directory.rglob("*")
                if p.is_file() and p.name not in {"manifest.json", "worker.log", "request.json"}
            }
            write_json(
                {"config_hash": digest, "run_id": run_id, "files": artifacts},
                directory / "manifest.json",
            )
            load_two_stage_run(directory, config, registry_path, splits, pending=True)
            registry.complete(
                run_id,
                {
                    "checkpoint_path": directory / "final_epoch.pt",
                    "checkpoint_sha256": compute_sha256(directory / "final_epoch.pt"),
                    "prediction_path": directory / "oof_predictions.csv",
                    "prediction_sha256": compute_sha256(directory / "oof_predictions.csv"),
                    "checkpoint_bytes": (directory / "final_epoch.pt").stat().st_size,
                    "train_seconds": metrics["train_seconds"],
                    "peak_memory_bytes": metrics["peak_memory_bytes"],
                    "metrics_json": metrics,
                    "environment_json": metrics["environment"],
                    "last_completed_stage": "stage_b_and_diagnostics_complete",
                },
            )
        except BaseException as error:
            progress = directory / "progress.json"
            stage = (
                json.loads(progress.read_text()).get("stage", "worker_start")
                if progress.is_file()
                else "worker_start"
            )
            registry.fail(run_id, error, last_completed_stage=stage)
            write_json(
                {
                    "status": "fail",
                    "fold": fold,
                    "error": str(error),
                    "stage": stage,
                    "later_folds_started": False,
                },
                output / "screen_decision.json",
            )
            raise
        child[fold] = load_two_stage_run(directory, config, registry_path, splits)
    report = evaluate_usage_two_stage(child, sources, splits)
    report["run_ids"] = {str(f): r["run_id"] for f, r in child.items()}
    write_json(report, output / "screen_decision.json")
    pd.concat([r["predictions"] for r in child.values()]).sort_values("id").to_csv(
        output / "oof_predictions.csv", index=False
    )
    pd.DataFrame(report["candidate_metrics"]["per_class"]).to_csv(
        output / "per_class.csv", index=False
    )
    return report


def load_two_stage_run(directory, config, registry_path, splits, *, pending=False):
    """Fail closed on missing, altered or wrong-recipe cached training evidence."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    run_id = directory.name
    if (
        manifest["config_hash"] != configuration_hash(config)
        or manifest["run_id"] != run_id
        or json.loads((directory / "config.json").read_text()) != config
    ):
        raise ValueError("Two-stage cached configuration changed")
    required = {
        "config.json",
        "metrics.json",
        "normalization.json",
        "history.csv",
        "final_epoch.pt",
        "stage_a.pt",
        "stage_a_metrics.json",
        "stage_a_train_predictions.csv",
        "stage_a_oof_predictions.csv",
        "oof_predictions.csv",
        "clean_train_predictions.csv",
        "robustness.csv",
        "feature_cache_manifest.json",
        *(f"corruptions/{c}.csv" for c in CORE_CORRUPTIONS),
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Two-stage artifact bundle is incomplete")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or compute_sha256(directory / relative) != digest
        ):
            raise ValueError(f"Two-stage artifact changed: {name}")
    rows = pd.read_csv(registry_path, keep_default_na=False)
    row = rows[rows.run_id.eq(run_id)]
    if (
        len(row) != 1
        or row.iloc[0].status != ("running" if pending else "complete")
        or row.iloc[0].config_hash != configuration_hash(config)
    ):
        raise ValueError("Two-stage registry proof is missing")
    row = row.iloc[0]
    metrics = json.loads((directory / "metrics.json").read_text())
    if (not pending and metrics != json.loads(row.metrics_json)) or not metrics[
        "backbone_unchanged"
    ]:
        raise ValueError("Two-stage registered metrics or backbone proof disagree")
    if not pending and (
        row.checkpoint_sha256 != manifest["files"]["final_epoch.pt"]
        or row.prediction_sha256 != manifest["files"]["oof_predictions.csv"]
    ):
        raise ValueError("Two-stage registry artifact hashes disagree")
    train, expected = training_scope(splits, config["fold"])
    cache = json.loads((directory / "feature_cache_manifest.json").read_text())
    if (
        cache["scope"] != "outer_training_only"
        or cache["ids"] != train.id.astype(int).tolist()
        or cache["product_family_groups"] != train.product_family_group.astype(str).tolist()
        or cache["shape"] != [len(train), 256]
        or cache["backbone_sha256_before"] != metrics["backbone_sha256_after"]
    ):
        raise ValueError("Two-stage feature cache scope or frozen backbone proof changed")
    history = pd.read_csv(directory / "history.csv", keep_default_na=False)
    if (
        list(zip(history.stage, history.epoch, strict=True))
        != [*(("A", e) for e in range(1, 31)), *(("B", e) for e in range(1, 11))]
        or metrics["selected_stage"] != "B"
        or metrics["selected_epoch"] != 10
    ):
        raise ValueError("Two-stage final epoch policy changed")
    counts, weights = class_weights_for_training(train)
    if (
        metrics["class_counts"] != counts
        or not np.allclose(metrics["class_weights"], weights)
        or metrics["parameter_count"] != baseline_parameter_count("usage")
        or metrics["stage_b_trainable_parameters"] != 256 * 9 + 9
    ):
        raise ValueError("Two-stage model or balanced head recipe changed")
    stage_a = json.loads((directory / "stage_a_metrics.json").read_text())
    if metrics["stage_a"] != stage_a:
        raise ValueError("Two-stage Stage A evidence disagrees")
    for name, frame, saved in (
        ("oof_predictions.csv", expected, metrics),
        ("clean_train_predictions.csv", train, metrics["clean_training"]),
        ("stage_a_train_predictions.csv", train, stage_a["training"]),
        ("stage_a_oof_predictions.csv", expected, stage_a["validation"]),
    ):
        predictions = validate_oof(
            read_predictions(directory / name), frame, target="usage", classes=CLASSES
        )
        if not predictions.run_id.eq(run_id).all():
            raise ValueError("Two-stage predictions have the wrong run ID")
        actual = oof_metrics(predictions, CLASSES)
        for key in ("macro_f1", "nll", "brier", "ece_15"):
            if not np.isclose(actual[key], saved[key], atol=1e-10, rtol=0):
                raise ValueError(f"Two-stage {name} metric does not reproduce: {key}")
    robust = pd.read_csv(directory / "robustness.csv", keep_default_na=False)
    robustness_changes(
        robust,
        clean_by_fold={config["fold"]: metrics["macro_f1"]},
        run_ids_by_fold={config["fold"]: run_id},
    )
    for name in CORE_CORRUPTIONS:
        predictions = validate_oof(
            read_predictions(directory / "corruptions" / f"{name}.csv"),
            expected,
            target="usage",
            classes=CLASSES,
            run_ids_by_fold={config["fold"]: run_id},
        )
        score = oof_metrics(predictions, CLASSES)["macro_f1"]
        if not np.isclose(
            score, robust.loc[robust.corruption.eq(name), "macro_f1"].item(), atol=1e-10, rtol=0
        ):
            raise ValueError("Two-stage corruption probabilities do not reproduce")
    return {
        "run_id": run_id,
        "metrics": metrics,
        "predictions": read_predictions(directory / "oof_predictions.csv"),
        "robustness": robust,
        "directory": str(directory),
    }
