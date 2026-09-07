"""Recover expanded Usage runs into a log that other notebooks do not write."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.registry import REGISTRY_COLUMNS, RunRegistry
from fashion.train.task3_decisions import oof_metrics, validate_oof
from fashion.train.task3_usage_expanded import (
    ARTIFACT_DIRECTORY,
    CLASSES,
    E8_RUN_IDS,
    EXPERIMENT,
    MAP_SHA256,
    SPLIT_SHA256,
    expanded_usage_spec,
    read_predictions,
    training_scope,
    validate_dataset,
    write_json,
)


def usage_registry_path(output_root):
    """One experiment owns this CSV; each runtime also keeps its local results/runs.csv."""
    return Path(output_root) / ARTIFACT_DIRECTORY / "usage/results/runs.csv"


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REGISTRY_COLUMNS:
            raise ValueError(f"Unexpected registry columns: {path}")
        rows = list(reader)
    ids = [row["run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate run IDs in {path}")
    return rows


def _completed_row(path, splits, previous):
    """Recover completion only from a whole, consistent final-epoch evidence bundle."""
    config = json.loads((path / "config.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    base = Task3BaselineConfig(target="usage").to_dict()
    spec = expanded_usage_spec()
    if config.get("child_experiment") != spec.to_dict() or any(
        config.get(k) != value for k, value in base.items()
    ):
        raise ValueError(f"Saved Usage recipe differs: {path.name}")
    fold = int(metrics["validation_fold"])
    if fold not in range(5) or metrics["run_id"] != path.name:
        raise ValueError(f"Saved run identity differs: {path.name}")
    if (
        metrics["experiment_id"] != EXPERIMENT
        or metrics["target"] != "usage"
        or metrics["selected_epoch"] != base["epochs"]
        or metrics["epochs_completed"] != base["epochs"]
        or metrics["checkpoint_policy"] != "final_epoch"
    ):
        raise ValueError(f"Saved run did not complete the frozen recipe: {path.name}")
    required = {
        "config.json",
        "normalization.json",
        "history.csv",
        "robustness.csv",
        "training_predictions.csv",
        "source_metrics.json",
    }
    hashes = metrics.get("expanded_artifact_sha256", {})
    if set(hashes) != required:
        raise ValueError(f"Saved diagnostic bundle is incomplete: {path.name}")
    for name, digest in hashes.items():
        if compute_sha256(path / name) != digest:
            raise ValueError(f"Saved Usage artifact changed: {path / name}")
    training, expected = training_scope(splits, fold)
    predictions = validate_oof(
        read_predictions(path / "oof_predictions.csv"),
        expected,
        target="usage",
        classes=CLASSES,
        run_ids_by_fold={fold: path.name},
    )
    measured = oof_metrics(predictions, CLASSES)
    for key in ("macro_f1", "nll", "brier", "ece_15"):
        if not np.isclose(measured[key], metrics[key], atol=1e-7, rtol=0):
            raise ValueError(f"Saved {key} differs from the predictions: {path.name}")
    checkpoint_path = path / "final_epoch.pt"
    checkpoint_hash = compute_sha256(checkpoint_path)
    prediction_hash = compute_sha256(path / "oof_predictions.csv")
    if previous and previous["status"] == "complete":
        if (
            previous["checkpoint_sha256"] != checkpoint_hash
            or previous["prediction_sha256"] != prediction_hash
            or json.loads(previous["metrics_json"]) != metrics
        ):
            raise ValueError(f"Saved files differ from the completed registry row: {path.name}")
    else:
        # No original checkpoint hash survived. Cross-check the local checkpoint's
        # own config and completion marker before recording a reconstructed row.
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint["run_id"] != path.name
            or checkpoint["config"] != config
            or checkpoint["selected_epoch"] != base["epochs"]
            or checkpoint["epochs_completed"] != base["epochs"]
            or checkpoint["checkpoint_policy"] != "final_epoch"
            or checkpoint["class_names"] != list(CLASSES)
        ):
            raise ValueError(f"Checkpoint does not match its completed run: {path.name}")
        del checkpoint
    digest_payload = {"baseline_controls": base, "child_experiment": spec.to_dict()}
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    row = {key: "" for key in REGISTRY_COLUMNS}
    if previous:
        row.update(previous)
    row.update(
        {
            "run_id": path.name,
            "experiment_id": EXPERIMENT,
            "hypothesis_id": spec.hypothesis_id,
            "parent_run_ids": json.dumps([E8_RUN_IDS[fold]]),
            "task": "task3",
            "target": "usage",
            "validation_fold": str(fold),
            "seed": str(base["seed"]),
            "status": "complete",
            "debug": "false",
            "scratch": "true",
            "submission_eligible": "true",
            "config_hash": digest,
            "config_path": str(path / "config.json"),
            "split_digest": SPLIT_SHA256,
            "label_map_digest": MAP_SHA256,
            "training_product_count": str(len(training)),
            "validation_product_count": str(len(expected)),
            "training_family_count": str(training.product_family_group.nunique()),
            "validation_family_count": str(expected.product_family_group.nunique()),
            "model_family": spec.model_family,
            "parameter_count": str(baseline_parameter_count("usage")),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "prediction_path": str(path / "oof_predictions.csv"),
            "prediction_sha256": prediction_hash,
            "history_path": str(path / "history.csv"),
            "metrics_json": json.dumps(metrics),
            "train_seconds": str(metrics["train_seconds"]),
            "peak_memory_bytes": str(metrics["peak_memory_bytes"]),
            "checkpoint_bytes": str(checkpoint_path.stat().st_size),
            "last_completed_stage": "recovered_verified_complete_bundle",
        }
    )
    return row


def prepare_usage_registry(*, output_root, source_paths=(), root=ROOT, interrupted_run_ids=()):
    """Back up old logs, verify completed folds, then write the private Usage log.

    Source logs are read-only. Incomplete runs remain incomplete. This function
    neither fits a model nor changes checkpoints or their recorded metrics.
    """
    output_root = Path(output_root)
    destination = usage_registry_path(output_root)
    sources = list(dict.fromkeys([destination, *(Path(p) for p in source_paths)]))
    sources = [p for p in sources if p.is_file()]
    evidence = output_root / ARTIFACT_DIRECTORY / "usage"
    recovery_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = evidence / "registry_recovery" / recovery_id
    backup.mkdir(parents=True, exist_ok=False)
    records = {}
    for index, source in enumerate(sources):
        snapshot = backup / f"source_{index}.csv"
        shutil.copyfile(source, snapshot)
        for row in _read_csv(snapshot):
            if row["experiment_id"] != EXPERIMENT:
                continue
            existing = records.get(row["run_id"])
            if existing is None or (
                existing["status"] != "complete" and row["status"] == "complete"
            ):
                records[row["run_id"]] = row
    splits, _ = validate_dataset(root=root, check_images=False)
    recovered, incomplete = [], []
    for path in sorted(evidence.glob(f"{EXPERIMENT}_usage_smallcnn_f*")):
        if not path.is_dir():
            continue
        if not (path / "metrics.json").is_file():
            incomplete.append(path.name)
            continue
        metrics = json.loads((path / "metrics.json").read_text())
        if "expanded_artifact_sha256" not in metrics:
            incomplete.append(path.name)
            continue
        previous = records.get(path.name)
        records[path.name] = _completed_row(path, splits, previous)
        recovered.append(
            {
                "run_id": path.name,
                "fold": int(records[path.name]["validation_fold"]),
                "original_complete_row_found": bool(previous and previous["status"] == "complete"),
            }
        )
    # Missing or unfinished bundles must never be promoted from a stale CSV alone.
    for run_id, row in records.items():
        if row["status"] == "complete" and not any(r["run_id"] == run_id for r in recovered):
            raise ValueError(f"Completed registry row has no verified bundle: {run_id}")
    for run_id in interrupted_run_ids:
        if run_id not in incomplete:
            continue
        path = evidence / run_id
        config = json.loads((path / "config.json").read_text())
        if config.get("child_experiment") != expanded_usage_spec().to_dict():
            raise ValueError(f"Interrupted run is not this expanded Usage experiment: {run_id}")
        fold = E8_RUN_IDS.index(config["parent_run_id"])
        training, validation = training_scope(splits, fold)
        row = records.setdefault(run_id, {key: "" for key in REGISTRY_COLUMNS})
        row.update(
            {
                "run_id": run_id,
                "experiment_id": EXPERIMENT,
                "task": "task3",
                "target": "usage",
                "validation_fold": str(fold),
                "seed": str(config["seed"]),
                "debug": "false",
                "scratch": "true",
                "submission_eligible": "true",
                "model_family": config["effective_model_family"],
                "training_product_count": str(len(training)),
                "validation_product_count": str(len(validation)),
                "training_family_count": str(training.product_family_group.nunique()),
                "validation_family_count": str(validation.product_family_group.nunique()),
                "status": "failed",
                "timestamp_end": datetime.now(UTC).isoformat(),
                "config_path": str(path / "config.json"),
                "history_path": str(path / "history.csv"),
                "split_digest": SPLIT_SHA256,
                "label_map_digest": MAP_SHA256,
                "exception_type": "RegistryRowLost",
                "exception_message": "Training stopped after the shared CSV lost this row",
                "last_completed_stage": "interrupted_history_preserved",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="usage-registry-") as scratch:
        candidate = RunRegistry(Path(scratch) / "runs.csv")
        candidate._write_rows(list(records.values()))
        # The destination is owned only by Usage. No shared log is written here.
        registry = RunRegistry(destination)
        with registry._locked():
            registry._write_rows(_read_csv(candidate.path))
    receipt = {
        "registry_path": str(destination),
        "sources": [str(p) for p in sources],
        "backups": str(backup),
        "verified_complete_runs": recovered,
        "incomplete_runs": incomplete,
        "training_started": False,
    }
    write_json(receipt, backup / "recovery.json")
    return receipt


def repair_connected_usage_session(
    namespace, *, expected_completed_folds=(), interrupted_run_ids=()
):
    """Switch the existing notebook's next training call without editing its frozen code."""
    required = ("REPO_DIR", "DRIVE_TASK_DIR", "LOCAL_REGISTRY", "DRIVE_REGISTRY")
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError("Run this repair inside the connected, failed Usage notebook")
    output_root = Path(namespace["DRIVE_TASK_DIR"])
    receipt = prepare_usage_registry(
        output_root=output_root,
        source_paths=(
            namespace["LOCAL_REGISTRY"],
            namespace["DRIVE_REGISTRY"],
            output_root / "results/runs.csv",
        ),
        root=namespace["REPO_DIR"],
        interrupted_run_ids=interrupted_run_ids,
    )
    completed = sorted({row["fold"] for row in receipt["verified_complete_runs"]})
    if not set(expected_completed_folds).issubset(completed):
        raise RuntimeError(
            f"Expected completed folds {list(expected_completed_folds)}, verified {completed}. "
            "Keep the session connected and inspect the saved files before restarting training."
        )
    namespace["DRIVE_REGISTRY"] = Path(receipt["registry_path"])
    namespace["registry_recovery"] = receipt
    print(f"Verified completed folds: {completed}")
    print(f"Usage-only run log: {receipt['registry_path']}")
    print(f"Old CSV backups: {receipt['backups']}")
    print("Now rerun only the training cell with FOLDS = (0, 1, 2, 3, 4).")
    print("Completed folds will be reused; each unfinished fold starts from scratch.")
    return receipt


if __name__ == "__main__":
    repair_connected_usage_session(
        globals(),
        expected_completed_folds=(0, 1),
        interrupted_run_ids=(
            "t3_usage_expanded_e8_usage_smallcnn_f2_s2753_6eb56854d557_20260906T082459Z0a0f7d",
        ),
    )
