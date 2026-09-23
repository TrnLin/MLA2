"""Small, isolated post-submission RF-head search on frozen I2 fold embeddings.

This study never opens the internal holdout, changes the submitted I2 bundle, or
reuses validation examples as training examples within a fold. Its exploratory
winner is not an independently validated improvement: the grid is selected on
the same development OOF folds used to describe its performance.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from fashion.config import LABEL_MAPS_JSON, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data.dataset import get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.torch import ImageTransformSpec, build_task_loaders
from fashion.task2.experiments import _expected_validation, _validate_output_oof
from fashion.task2.post_submission_experiments import (
    BASE_I2_EXPERIMENT_ID,
    CANONICAL_FOLDS,
    POST_SUBMISSION_IMPLEMENTATION_PATHS,
    _artifact_path,
    _build_i2_model,
    _extract_embeddings,
    _load_base_i2_config,
    _load_i2_checkpoint,
    _resolve_device,
)
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import build_run_cache_key, find_cached_run, implementation_sha256
from fashion.train.metrics import (
    SEASON_LABELS,
    multiclass_metrics,
    paired_group_bootstrap,
    validate_metrics_match_oof,
    validate_oof,
)
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIG_PATH = ROOT / "configs/task2/post_submission_rf_grid.json"
EVIDENCE_DIR = ROOT / "results/evidence/task2/post_submission/rf_grid"
FIGURE_DIR = ROOT / "results/figures/task2/post_submission/rf_grid"
TEMP_DIR = ROOT / "tmp/task2/post_submission/rf_grid"
IMPLEMENTATION_PATHS = (
    *POST_SUBMISSION_IMPLEMENTATION_PATHS,
    "src/fashion/task2/post_submission_rf_grid.py",
)


@dataclass(frozen=True)
class Candidate:
    id: str
    min_samples_leaf: int
    max_features: str | float


def load_grid_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load a deliberately narrow, predeclared grid; reject accidental expansion."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "1.0.0"
        or payload.get("role") != "post_submission_development_only"
    ):
        raise ValueError("RF grid schema or post-submission boundary changed")
    if payload.get("folds") != list(CANONICAL_FOLDS) or payload.get("seed") != 2753:
        raise ValueError("RF grid must use canonical folds and the primary seed")
    if payload.get("fixed_parameters") != {
        "n_estimators": 300,
        "criterion": "gini",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": -1,
        "random_state_policy": "seed_plus_fold",
    }:
        raise ValueError("RF grid fixed parameters changed")
    expected = (
        Candidate("r0_baseline", 2, "sqrt"),
        Candidate("r1_leaf1", 1, "sqrt"),
        Candidate("r2_leaf4", 4, "sqrt"),
        Candidate("r3_features025", 2, 0.25),
        Candidate("r4_leaf4_features025", 4, 0.25),
    )
    if payload.get("candidates") != [item.__dict__ for item in expected]:
        raise ValueError("RF grid candidates changed from the predeclared comparison")
    selection = payload.get("selection", {})
    if selection != {
        "minimum_macro_f1_gain": 0.003,
        "minimum_spring_f1_delta": -0.005,
        "require_positive_paired_lower_95": True,
        "bootstrap_replicates": 10000,
    }:
        raise ValueError("RF grid decision thresholds changed")
    return payload


def _verified_baseline_sources(
    config: dict[str, Any], registry: RunRegistry
) -> tuple[dict[int, dict[str, Any]], dict[int, pd.DataFrame]]:
    summary_path = ROOT / config["baseline_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_ids = summary["random_forest_runs"]
    if len(run_ids) != 5 or len(set(run_ids)) != 5:
        raise ValueError("pinned RF baseline must contain exactly five distinct runs")
    rows = registry.read().set_index("run_id")
    split_sha256 = compute_sha256(SPLITS_CSV)
    label_map_sha256 = compute_sha256(LABEL_MAPS_JSON)
    sources: dict[int, dict[str, Any]] = {}
    baseline_oof: dict[int, pd.DataFrame] = {}
    for fold, run_id in enumerate(run_ids):
        row = rows.loc[run_id]
        if (
            row["status"] != "completed"
            or int(row["fold"]) != fold
            or int(row["seed"]) != config["seed"]
            or row["experiment_id"] != "postsubmit-i2-embedding-random-forest"
            or row["split_sha256"] != split_sha256
            or row["label_map_sha256"] != label_map_sha256
        ):
            raise ValueError(f"pinned RF baseline registry identity changed at fold {fold}")
        prediction_path = _artifact_path(row["prediction_path"], data_root=ROOT)
        history_path = _artifact_path(row["history_path"], data_root=ROOT)
        verify_artifact(prediction_path, row["prediction_sha256"])
        verify_artifact(history_path, row["history_sha256"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        parameters = history["random_forest"]["parameters"]
        if (
            parameters["n_estimators"] != 300
            or parameters["min_samples_leaf"] != 2
            or parameters["max_features"] != "sqrt"
            or parameters["class_weight"] != "balanced_subsample"
        ):
            raise ValueError(f"pinned RF baseline parameters changed at fold {fold}")
        boundary = history["experiment_boundary"]
        source_row = rows.loc[boundary["source_run_id"]]
        if (
            source_row["task"] != "task2"
            or source_row["status"] != "completed"
            or int(source_row["fold"]) != fold
            or int(source_row["seed"]) != config["seed"]
            or source_row["experiment_id"] != BASE_I2_EXPERIMENT_ID
            or source_row["split_sha256"] != split_sha256
            or source_row["label_map_sha256"] != label_map_sha256
        ):
            raise ValueError(f"pinned I2 source registry identity changed at fold {fold}")
        if source_row["checkpoint_sha256"] != boundary["source_checkpoint_sha256"]:
            raise ValueError(f"pinned I2 checkpoint digest changed at fold {fold}")
        checkpoint_path = _artifact_path(source_row["checkpoint_path"], data_root=ROOT)
        verify_artifact(checkpoint_path, source_row["checkpoint_sha256"])
        sources[fold] = {
            "run_id": str(boundary["source_run_id"]),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": str(source_row["checkpoint_sha256"]),
            "baseline_run_id": run_id,
            "baseline_prediction_sha256": str(row["prediction_sha256"]),
        }
        baseline_oof[fold] = pd.read_csv(prediction_path)
    return sources, baseline_oof


def _canonical_fold_samples(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = get_samples(load_splits(SPLITS_CSV), partition="development", target="season")
    training = development.loc[development["cv_fold"].ne(fold)]
    validation = development.loc[development["cv_fold"].eq(fold)]
    if len(training) + len(validation) != 32753 or training.empty or validation.empty:
        raise ValueError("RF grid fold does not cover the 32,753 valid development rows")
    return training, validation


def _validate_feature_arrays(arrays: dict[str, np.ndarray], fold: int) -> None:
    training, validation = _canonical_fold_samples(fold)
    train_ids = arrays["train_ids"].astype(np.int64)
    val_ids = arrays["validation_ids"].astype(np.int64)
    if len(set(train_ids.tolist())) != len(training) or set(train_ids.tolist()) != set(
        training["id"].astype(int)
    ):
        raise ValueError("cached training feature IDs differ from canonical fold")
    expected_validation = _expected_validation(fold=fold, splits_path=SPLITS_CSV, target="season")
    if tuple(val_ids.tolist()) != tuple(expected_validation["id"].astype(int).tolist()):
        raise ValueError("cached validation feature IDs differ from canonical order")
    labels = load_label_maps(LABEL_MAPS_JSON)["season"]["classes"]
    for prefix, samples, ids in (
        ("train", training, train_ids),
        ("validation", validation, val_ids),
    ):
        x = arrays[f"{prefix}_x"]
        y = arrays[f"{prefix}_y"]
        if x.shape != (len(ids), 256) or y.shape != (len(ids),) or not np.isfinite(x).all():
            raise ValueError(f"invalid {prefix} embedding matrix")
        if x.dtype != np.float32 or y.dtype != np.int64 or (y < 0).any() or (y >= 4).any():
            raise ValueError(f"invalid {prefix} embedding dtype or target")
        target_map = samples.set_index("id")["season"].to_dict()
        expected_y = np.array(
            [labels.index(target_map[int(identifier)]) for identifier in ids], dtype=np.int64
        )
        if not np.array_equal(y, expected_y):
            raise ValueError(f"cached {prefix} targets differ from canonical Season labels")


def _fold_embeddings(
    fold: int, source: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], str]:
    base = _load_base_i2_config(config["base_i2_config"], project_root=ROOT)
    identity = {
        "source_run_id": source["run_id"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "split_sha256": compute_sha256(SPLITS_CSV),
        "label_map_sha256": compute_sha256(LABEL_MAPS_JSON),
        "implementation_sha256": implementation_sha256(*IMPLEMENTATION_PATHS, root=ROOT),
        "base_config": base.to_dict(),
        "fold": fold,
        "seed": config["seed"],
        "batch_size": config["embedding_batch_size"],
    }
    digest = canonical_sha256(identity)
    cache_dir = TEMP_DIR / "embeddings"
    matrix_path = cache_dir / f"fold-{fold}.npz"
    manifest_path = cache_dir / f"fold-{fold}.json"
    if matrix_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity_sha256") == digest:
            verify_artifact(matrix_path, manifest["matrix_sha256"])
            with np.load(matrix_path, allow_pickle=False) as data:
                arrays = {
                    name: data[name]
                    for name in (
                        "train_x",
                        "train_y",
                        "train_ids",
                        "validation_x",
                        "validation_y",
                        "validation_ids",
                    )
                }
            _validate_feature_arrays(arrays, fold)
            print(f"[RF grid] fold {fold}: loaded verified frozen embeddings", flush=True)
            return arrays, digest
    loaders = build_task_loaders(
        validation_fold=fold,
        image_size=base.data.image_size,
        batch_size=config["embedding_batch_size"],
        target="season",
        augmentation="none",
        seed=config["seed"],
        num_workers=base.data.num_workers,
        validation_batch_size=config["embedding_batch_size"],
        pin_memory=base.data.pin_memory,
        root=ROOT,
        splits_path=SPLITS_CSV,
        label_map_path=LABEL_MAPS_JSON,
    )
    mappings = load_label_maps(LABEL_MAPS_JSON)
    model = _build_i2_model(
        season_classes=len(loaders.labels),
        article_type_classes=int(mappings["articleType"]["num_classes"]),
    )
    _load_i2_checkpoint(model, source["checkpoint_path"])
    device = _resolve_device(base.optimisation.device)
    print(f"[RF grid] fold {fold}: extracting frozen features on {device}", flush=True)
    train_x, train_y, train_ids = _extract_embeddings(model, loaders.train, device=device)
    validation_x, validation_y, validation_ids = _extract_embeddings(
        model, loaders.validation, device=device
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    arrays = {
        "train_x": train_x,
        "train_y": train_y,
        "train_ids": np.asarray(train_ids, dtype=np.int64),
        "validation_x": validation_x,
        "validation_y": validation_y,
        "validation_ids": np.asarray(validation_ids, dtype=np.int64),
    }
    _validate_feature_arrays(arrays, fold)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    atomic_write_bytes(matrix_path, buffer.getvalue())
    atomic_write_json(
        manifest_path,
        {
            "identity": identity,
            "identity_sha256": digest,
            "matrix_sha256": compute_sha256(matrix_path),
        },
    )
    return arrays, digest


def _assert_baseline_parity(new: pd.DataFrame, old: pd.DataFrame, fold: int) -> None:
    old = old.set_index("id").loc[new["id"]].reset_index()
    if not np.array_equal(new["y_pred"].to_numpy(), old["y_pred"].to_numpy()):
        raise ValueError(f"fold {fold} RF baseline predictions did not reproduce pinned run")
    columns = [f"prob_{label}" for label in SEASON_LABELS]
    if not np.allclose(
        new[columns].to_numpy(float), old[columns].to_numpy(float), rtol=0.0, atol=1e-10
    ):
        raise ValueError(f"fold {fold} RF baseline probabilities did not reproduce pinned run")


def _run_fold_candidate(
    fold: int,
    candidate: Candidate,
    arrays: dict[str, np.ndarray],
    embedding_digest: str,
    source: dict[str, Any],
    old_baseline: pd.DataFrame,
    config: dict[str, Any],
    registry: RunRegistry,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    experiment_id = f"{config['study_id']}-{candidate.id}"
    settings = {
        "study": config,
        "candidate": candidate.__dict__,
        "embedding_digest": embedding_digest,
        "source_run_id": source["run_id"],
    }
    key = build_run_cache_key(
        settings,
        fold=fold,
        seed=config["seed"],
        implementation_paths=IMPLEMENTATION_PATHS,
        split_path=SPLITS_CSV,
        label_map_path=LABEL_MAPS_JSON,
        root=ROOT,
    )
    cached = find_cached_run(
        registry, key, required_artifacts=("prediction", "history"), artifact_root=ROOT
    )
    if cached is not None and cached.row["experiment_id"] == experiment_id:
        oof = pd.read_csv(_artifact_path(cached.row["prediction_path"], data_root=ROOT))
        history = json.loads(
            _artifact_path(cached.row["history_path"], data_root=ROOT).read_text(encoding="utf-8")
        )
        if (
            history.get("run_id") != cached.run_id
            or history.get("candidate") != candidate.__dict__
            or history.get("source_i2_run_id") != source["run_id"]
            or history.get("source_checkpoint_sha256") != source["checkpoint_sha256"]
            or history.get("embedding_digest") != embedding_digest
            or history.get("holdout_used") is not False
        ):
            raise ValueError(f"cached RF grid history identity changed at fold {fold}")
        _validate_output_oof(
            oof,
            expected=_expected_validation(fold=fold, splits_path=SPLITS_CSV, target="season"),
            labels=SEASON_LABELS,
        )
        metrics = validate_metrics_match_oof(oof, history["metrics"], labels=SEASON_LABELS)
        if not math.isclose(
            float(cached.row["primary_metric_value"]),
            float(metrics["macro_f1"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"cached RF grid macro-F1 differs from registry at fold {fold}")
        if candidate.id == "r0_baseline":
            _assert_baseline_parity(oof, old_baseline, fold)
        print(f"[RF grid] fold {fold} {candidate.id}: verified run cache", flush=True)
        return oof, history
    run_id = new_run_id(experiment_id, fold, config["seed"])
    record = RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        fold=fold,
        seed=config["seed"],
        config_sha256=key.config_sha256,
        split_sha256=key.split_sha256,
        label_map_sha256=key.label_map_sha256,
        implementation_sha256=key.implementation_sha256,
        stage="post_submission_rf_grid",
        model_family="frozen_i2_embedding_random_forest",
        benchmark_only=True,
        final_eligible=False,
        scratch=True,
        transform_id=ImageTransformSpec(
            image_size=_load_base_i2_config(
                config["base_i2_config"], project_root=ROOT
            ).data.image_size,
            augmentation="none",
        ).transform_id,
        loss_id="random_forest_gini_on_frozen_i2_embedding",
        epochs_requested=0,
        primary_metric_name="macro_f1",
    )
    run_started = time.perf_counter()
    with tracked_run(registry, record) as run:
        forest = RandomForestClassifier(
            n_estimators=300,
            criterion="gini",
            max_features=candidate.max_features,
            min_samples_leaf=candidate.min_samples_leaf,
            class_weight="balanced_subsample",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=config["seed"] + fold,
        )
        start = time.perf_counter()
        forest.fit(arrays["train_x"], arrays["train_y"])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw = forest.predict_proba(arrays["validation_x"])
        predict_seconds = time.perf_counter() - start
        probs = np.zeros((len(arrays["validation_x"]), 4), dtype=np.float64)
        for column, class_index in enumerate(forest.classes_):
            probs[:, int(class_index)] = raw[:, column]
        label_array = np.asarray(SEASON_LABELS, dtype=object)
        oof = pd.DataFrame(
            {
                "id": arrays["validation_ids"],
                "fold": fold,
                "seed": config["seed"],
                "y_true": label_array[arrays["validation_y"]],
                "y_pred": label_array[probs.argmax(axis=1)],
            }
        )
        for index, label in enumerate(SEASON_LABELS):
            oof[f"prob_{label}"] = probs[:, index]
        _validate_output_oof(
            oof,
            expected=_expected_validation(fold=fold, splits_path=SPLITS_CSV, target="season"),
            labels=SEASON_LABELS,
        )
        if candidate.id == "r0_baseline":
            _assert_baseline_parity(oof, old_baseline, fold)
        metrics = multiclass_metrics(oof["y_true"], probabilities=probs, labels=SEASON_LABELS)
        serialized = io.BytesIO()
        joblib.dump(forest, serialized, compress=3)
        history = {
            "run_id": run_id,
            "candidate": candidate.__dict__,
            "source_i2_run_id": source["run_id"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "embedding_digest": embedding_digest,
            "metrics": metrics,
            "fit_seconds": fit_seconds,
            "validation_predict_seconds": predict_seconds,
            "rf_head_validation_ms_per_image": 1000.0 * predict_seconds / len(oof),
            "serialized_forest_head_bytes": len(serialized.getbuffer()),
            "tree_node_count": int(sum(tree.tree_.node_count for tree in forest.estimators_)),
            "oob_accuracy_training_only": float(forest.oob_score_),
            "post_submission": True,
            "holdout_used": False,
        }
        output_dir = TEMP_DIR / "runs" / run_id
        prediction_path = atomic_write_csv(output_dir / "oof.csv", oof)
        history_path = atomic_write_json(output_dir / "history.json", history)
        run.epochs_completed = 0
        run.primary_metric_value = float(metrics["macro_f1"])
        run.metrics = metrics
        run.parameter_count = history["tree_node_count"]
        run.prediction_path = prediction_path.relative_to(ROOT).as_posix()
        run.prediction_sha256 = compute_sha256(prediction_path)
        run.history_path = history_path.relative_to(ROOT).as_posix()
        run.history_sha256 = compute_sha256(history_path)
        run.runtime_seconds = time.perf_counter() - run_started
    print(
        f"[RF grid] fold {fold} {candidate.id}: "
        f"macro-F1={metrics['macro_f1']:.4f}, fit={fit_seconds:.1f}s",
        flush=True,
    )
    return oof, history


def _summarise(
    results: dict[str, list[tuple[pd.DataFrame, dict[str, Any]]]], config: dict[str, Any]
) -> dict[str, Any]:
    development = get_samples(load_splits(SPLITS_CSV), partition="development", target="season")
    if (
        development["product_family_group"].isna().any()
        or development["product_family_group"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("RF grid development contains a missing product-family group")
    truth = development.set_index("id")["season"].to_dict()
    group = development.set_index("id")["product_family_group"].astype(str).to_dict()
    expected_ids = set(truth)
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    pooled: dict[str, pd.DataFrame] = {}
    for candidate_id, folds in results.items():
        frames = []
        for fold, (oof, history) in enumerate(folds):
            metrics = history["metrics"]
            fold_rows.append(
                {
                    "candidate": candidate_id,
                    "fold": fold,
                    "run_id": history["run_id"],
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "spring_f1": metrics["per_class"]["Spring"]["f1"],
                }
            )
            cost_rows.append(
                {
                    "candidate": candidate_id,
                    "fold": fold,
                    "run_id": history["run_id"],
                    "fit_seconds": history["fit_seconds"],
                    "rf_head_validation_ms_per_image": history["rf_head_validation_ms_per_image"],
                    "serialized_forest_head_bytes": history["serialized_forest_head_bytes"],
                    "tree_node_count": history["tree_node_count"],
                    "oob_accuracy_training_only": history["oob_accuracy_training_only"],
                }
            )
            frames.append(oof)
        combined = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)
        validate_oof(
            combined, expected_ids=expected_ids, expected_targets=truth, labels=SEASON_LABELS
        )
        metrics = multiclass_metrics(
            combined["y_true"],
            probabilities=combined[[f"prob_{label}" for label in SEASON_LABELS]].to_numpy(float),
            labels=SEASON_LABELS,
        )
        model_rows.append(
            {
                "candidate": candidate_id,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "spring_f1": metrics["per_class"]["Spring"]["f1"],
                "nll": metrics["nll"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "n_oof": len(combined),
            }
        )
        for label, values in metrics["per_class"].items():
            class_rows.append({"candidate": candidate_id, "class": label, **values})
        pooled[candidate_id] = combined
    baseline = pooled["r0_baseline"]
    comparisons = {
        candidate_id: (baseline["y_pred"].to_numpy(), frame["y_pred"].to_numpy())
        for candidate_id, frame in pooled.items()
        if candidate_id != "r0_baseline"
    }
    draws = paired_group_bootstrap(
        baseline["y_true"],
        np.asarray([group[int(identifier)] for identifier in baseline["id"]]),
        comparisons,
        labels=SEASON_LABELS,
        replicates=config["selection"]["bootstrap_replicates"],
        random_seed=config["seed"],
    )
    intervals = []
    for candidate_id, frame in draws.groupby("comparison_id"):
        values = frame["b_minus_a_macro_f1"].to_numpy(float)
        intervals.append(
            {
                "candidate": candidate_id,
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_median": float(np.median(values)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "replicates": len(values),
            }
        )
    models = pd.DataFrame(model_rows)
    models["delta_vs_r0"] = models["macro_f1"] - float(
        models.set_index("candidate").loc["r0_baseline", "macro_f1"]
    )
    baseline_spring = float(models.set_index("candidate").loc["r0_baseline", "spring_f1"])
    models["spring_delta_vs_r0"] = models["spring_f1"] - baseline_spring
    intervals_frame = pd.DataFrame(intervals)
    models = models.merge(intervals_frame[["candidate", "ci95_lower"]], on="candidate", how="left")
    models["eligible_exploratory"] = (
        (models["candidate"] != "r0_baseline")
        & (models["delta_vs_r0"] >= config["selection"]["minimum_macro_f1_gain"])
        & (models["spring_delta_vs_r0"] >= config["selection"]["minimum_spring_f1_delta"])
        & (models["ci95_lower"] > 0)
    )
    eligible = models.loc[models["eligible_exploratory"]].sort_values(
        ["macro_f1", "candidate"], ascending=[False, True]
    )
    winner = None if eligible.empty else str(eligible.iloc[0]["candidate"])
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_comparison": atomic_write_csv(EVIDENCE_DIR / "model_comparison.csv", models),
        "fold_comparison": atomic_write_csv(
            EVIDENCE_DIR / "fold_comparison.csv", pd.DataFrame(fold_rows)
        ),
        "per_class": atomic_write_csv(EVIDENCE_DIR / "per_class.csv", pd.DataFrame(class_rows)),
        "cost": atomic_write_csv(EVIDENCE_DIR / "cost.csv", pd.DataFrame(cost_rows)),
        "paired_bootstrap_intervals": atomic_write_csv(
            EVIDENCE_DIR / "paired_bootstrap_intervals.csv", intervals_frame
        ),
    }
    figure, axis = plt.subplots(figsize=(8.5, 4), constrained_layout=True)
    order = models.sort_values("macro_f1", ascending=True)
    axis.barh(
        order["candidate"],
        order["macro_f1"],
        color=["#28658e" if value == "r0_baseline" else "#57a878" for value in order["candidate"]],
    )
    for index, score in enumerate(order["macro_f1"]):
        axis.text(float(score) + 0.0004, index, f"{score:.4f}", va="center", fontsize=9)
    axis.set_xlim(
        max(0.0, float(order["macro_f1"].min()) - 0.015),
        min(1.0, float(order["macro_f1"].max()) + 0.025),
    )
    axis.set_xlabel("Pooled five-fold development OOF macro-F1")
    axis.set_title("Frozen I2 embedding + Random Forest: predeclared small grid")
    axis.grid(axis="x", alpha=0.2)
    figure_path = FIGURE_DIR / "model_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    paths["figure"] = figure_path
    summary = {
        "schema_version": "1.0.0",
        "study_id": config["study_id"],
        "post_submission": True,
        "holdout_used_for_selection": False,
        "exploratory_winner": winner,
        "decision": "retain_existing_full_development_rf"
        if winner is None
        else "candidate_for_separate_full_development_refit",
        "selection_warning": (
            "Candidate choice and reported OOF performance use the same five "
            "development folds; interval is descriptive, not independent "
            "post-selection proof."
        ),
        "baseline_macro_f1": float(models.set_index("candidate").loc["r0_baseline", "macro_f1"]),
        "results": models.astype(object).where(pd.notna(models), None).to_dict(orient="records"),
        "run_ids": {
            candidate_id: [history["run_id"] for _, history in folds]
            for candidate_id, folds in results.items()
        },
        "artifacts": {
            name: {"path": path.relative_to(ROOT).as_posix(), "sha256": compute_sha256(path)}
            for name, path in paths.items()
        },
    }
    atomic_write_json(EVIDENCE_DIR / "summary.json", summary)
    return summary


def run_grid(
    *, fold_filter: int | None = None, candidate_filter: str | None = None
) -> dict[str, Any] | None:
    config = load_grid_config()
    candidates = [Candidate(**raw) for raw in config["candidates"]]
    if fold_filter is not None and fold_filter not in CANONICAL_FOLDS:
        raise ValueError("fold filter must be one of the five canonical folds")
    if candidate_filter is not None and candidate_filter not in {item.id for item in candidates}:
        raise ValueError("candidate filter is not in the predeclared grid")
    registry = RunRegistry(RUNS_CSV)
    sources, old_baseline = _verified_baseline_sources(config, registry)
    results: dict[str, list[tuple[pd.DataFrame, dict[str, Any]]]] = {
        item.id: [] for item in candidates
    }
    for fold in CANONICAL_FOLDS:
        if fold_filter is not None and fold != fold_filter:
            continue
        arrays, embedding_digest = _fold_embeddings(fold, sources[fold], config)
        for candidate in candidates:
            if candidate_filter is not None and candidate.id != candidate_filter:
                continue
            result = _run_fold_candidate(
                fold,
                candidate,
                arrays,
                embedding_digest,
                sources[fold],
                old_baseline[fold],
                config,
                registry,
            )
            results[candidate.id].append(result)
    if fold_filter is not None or candidate_filter is not None:
        return None
    return _summarise(results, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--candidate", type=str)
    args = parser.parse_args()
    summary = run_grid(fold_filter=args.fold, candidate_filter=args.candidate)
    if summary is not None:
        print(
            json.dumps(
                {
                    "decision": summary["decision"],
                    "winner": summary["exploratory_winner"],
                    "baseline_macro_f1": summary["baseline_macro_f1"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
