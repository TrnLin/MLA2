"""Full-development Random Forest head for the post-submission I2 study.

The existing I2 encoder was already refitted on all development images. This
module freezes that verified encoder and fits only the forest on its 256-D
features. The original final package and G9 evaluation stay immutable.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from fashion.config import LABEL_MAPS_JSON, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data.dataset import get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.torch import EncodedClassificationDataset
from fashion.task2.inference import SeasonBundle, load_season_bundle
from fashion.task2.post_submission_experiments import (
    DEFAULT_RANDOM_FOREST_CONFIG,
    PRIMARY_SEED,
    _extract_embeddings,
    load_random_forest_experiment_spec,
)
from fashion.train.artifacts import (
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import implementation_sha256
from fashion.train.metrics import SEASON_LABELS
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run

EXPERIMENT_ID = "postsubmit-i2-embedding-rf-full-development"
SOURCE_REGISTRY = ROOT / "results/evidence/task2/final_handoff/registry_snapshot.csv"
SOURCE_OOF_SUMMARY = ROOT / "results/evidence/task2/post_submission/boosting_summary.json"
DEFAULT_FOREST_PATH = ROOT / "models/task2_postsubmit_i2_rf.joblib"
DEFAULT_MANIFEST_PATH = ROOT / "models/task2_postsubmit_i2_rf.manifest.json"
DEFAULT_HISTORY_PATH = (
    ROOT / "results/evidence/task2/post_submission/full_refit/fit_history.json"
)
IMPLEMENTATION_PATHS = (
    "src/fashion/task2/post_submission_rf_refit.py",
    "src/fashion/task2/post_submission_experiments.py",
    "src/fashion/task2/inference.py",
    "src/fashion/data/dataset.py",
    "src/fashion/data/images.py",
    "src/fashion/data/torch.py",
    "src/fashion/models/season.py",
)
ExecutionMode = Literal["run", "load", "run_or_load"]


@dataclass(frozen=True)
class VerifiedRFFit:
    """Hash-verified two-part inference package: original encoder plus new RF."""

    manifest: dict[str, Any]
    encoder: SeasonBundle
    forest: RandomForestClassifier
    forest_path: Path
    manifest_path: Path


def _inside_root(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"post-submission artifact leaves project root: {path}")
    return resolved


def _record(path: Path, root: Path) -> dict[str, Any]:
    resolved = _inside_root(path, root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": compute_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_record(record: dict[str, Any], root: Path, expected: Path) -> Path:
    if set(record) != {"path", "sha256", "bytes"}:
        raise ValueError("post-submission artifact declaration changed")
    path = _inside_root(record["path"], root)
    if path != _inside_root(expected, root):
        raise ValueError("post-submission artifact path changed")
    verify_artifact(path, str(record["sha256"]))
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError("post-submission artifact byte count changed")
    return path


def _rf_parameters(spec: Any) -> dict[str, Any]:
    return {
        "n_estimators": spec.n_estimators,
        "criterion": "gini",
        "max_features": spec.max_features,
        "min_samples_leaf": spec.min_samples_leaf,
        "class_weight": spec.class_weight,
        "bootstrap": True,
        "oob_score": spec.oob_score,
        "n_jobs": spec.n_jobs,
        "random_state": PRIMARY_SEED,
    }


def _verified_inputs(root: Path, *, device: str) -> tuple[SeasonBundle, Any, pd.DataFrame]:
    encoder = load_season_bundle(
        registry_path=_inside_root(SOURCE_REGISTRY, root),
        project_root=root,
        device=device,
    )
    spec = load_random_forest_experiment_spec(
        _inside_root(DEFAULT_RANDOM_FOREST_CONFIG, root)
    )
    if tuple(encoder.labels) != SEASON_LABELS or encoder.model.embedding_dimension != 256:
        raise ValueError("full-development I2 encoder contract changed")
    splits = load_splits(_inside_root(SPLITS_CSV, root))
    development = get_samples(splits, partition="development", target="season")
    development = development.sort_values("id", kind="stable").reset_index(drop=True)
    training_ids = development["id"].astype(int).tolist()
    if len(training_ids) != 32_753 or len(set(training_ids)) != len(training_ids):
        raise ValueError("full-development Season ID coverage changed")
    if set(development["partition"]) != {"development"}:
        raise ValueError("RF training includes a protected partition")
    expected_hash = encoder.transform.stats.training_id_sha256
    if canonical_sha256(training_ids) != expected_hash:
        raise ValueError("RF training IDs differ from the verified I2 refit")
    return encoder, spec, development


def _selection_record(root: Path) -> dict[str, Any]:
    path = _inside_root(SOURCE_OOF_SUMMARY, root)
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if (
        summary.get("role") != "post_submission_exploration_only"
        or summary.get("winner", {}).get("model") != "I2 embedding + RF"
        or summary.get("protocol", {}).get("evaluation")
        != "same 32753 development OOF IDs and canonical folds"
    ):
        raise ValueError("development-only RF selection evidence changed")
    return _record(path, root)


def _atomic_joblib(path: Path, forest: RandomForestClassifier) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(forest, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_verified_rf_fit(
    *,
    project_root: str | Path = ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    registry_path: str | Path = RUNS_CSV,
    device: str = "cpu",
) -> VerifiedRFFit:
    """Verify local forest bytes, source encoder, config, and completed run."""
    root = Path(project_root).resolve()
    path = _inside_root(manifest_path, root)
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {
        "schema_version", "role", "status", "experiment_id", "run_id",
        "training_rows", "training_id_sha256", "embedding_dimension", "labels",
        "seed", "rf_parameters", "source_encoder", "selection_evidence",
        "inputs", "implementation_sha256", "forest", "history",
    }
    if set(manifest) != required or (
        manifest["schema_version"] != "1.0.0"
        or manifest["role"] != "post_submission_full_development_refit"
        or manifest["status"] != "complete"
        or manifest["experiment_id"] != EXPERIMENT_ID
        or manifest["training_rows"] != 32_753
        or manifest["embedding_dimension"] != 256
        or tuple(manifest["labels"]) != SEASON_LABELS
        or manifest["seed"] != PRIMARY_SEED
    ):
        raise ValueError("post-submission RF manifest identity changed")
    encoder, spec, development = _verified_inputs(root, device=device)
    if manifest["training_id_sha256"] != canonical_sha256(
        development["id"].astype(int).tolist()
    ) or manifest["rf_parameters"] != _rf_parameters(spec):
        raise ValueError("RF training scope or parameters changed")
    _verify_record(manifest["source_encoder"]["manifest"], root, encoder.manifest_path)
    _verify_record(manifest["source_encoder"]["bundle"], root, encoder.bundle_path)
    if manifest["source_encoder"]["run_id"] != encoder.run_id:
        raise ValueError("RF source I2 run changed")
    _verify_record(manifest["selection_evidence"], root, SOURCE_OOF_SUMMARY)
    for name, expected in (
        ("splits", SPLITS_CSV),
        ("label_maps", LABEL_MAPS_JSON),
        ("rf_config", DEFAULT_RANDOM_FOREST_CONFIG),
    ):
        _verify_record(manifest["inputs"][name], root, expected)
    expected_implementation = implementation_sha256(*IMPLEMENTATION_PATHS, root=root)
    if manifest["implementation_sha256"] != expected_implementation:
        raise ValueError("RF refit implementation changed since fit")
    forest_path = _verify_record(manifest["forest"], root, DEFAULT_FOREST_PATH)
    history_path = _verify_record(manifest["history"], root, DEFAULT_HISTORY_PATH)
    registry = RunRegistry(_inside_root(registry_path, root)).read()
    rows = registry.loc[registry["run_id"].eq(manifest["run_id"])]
    if len(rows) != 1:
        raise ValueError("RF refit registry row is missing or duplicated")
    row = rows.iloc[0]
    if any(
        str(row[name]) != expected
        for name, expected in (
            ("task", "task2"),
            ("stage", "post_submission_full_development_refit"),
            ("experiment_id", EXPERIMENT_ID),
            ("status", "completed"),
            ("fold", ""),
            ("seed", str(PRIMARY_SEED)),
            ("benchmark_only", "true"),
            ("final_eligible", "false"),
            ("config_sha256", manifest["inputs"]["rf_config"]["sha256"]),
            ("split_sha256", manifest["inputs"]["splits"]["sha256"]),
            ("label_map_sha256", manifest["inputs"]["label_maps"]["sha256"]),
            ("implementation_sha256", expected_implementation),
            ("checkpoint_sha256", manifest["forest"]["sha256"]),
            ("history_sha256", manifest["history"]["sha256"]),
        )
    ) or _inside_root(str(row["checkpoint_path"]), root) != forest_path or (
        _inside_root(str(row["history_path"]), root) != history_path
    ):
        raise ValueError("RF refit registry provenance changed")
    # joblib uses pickle internally. Load only after all trusted local hashes match.
    forest = joblib.load(forest_path)
    if not isinstance(forest, RandomForestClassifier) or any(
        forest.get_params(deep=False)[key] != value
        for key, value in _rf_parameters(spec).items()
    ) or forest.n_features_in_ != 256 or not np.array_equal(
        forest.classes_, np.arange(len(SEASON_LABELS))
    ) or len(forest.estimators_) != spec.n_estimators:
        raise ValueError("RF model structure differs from the frozen declaration")
    return VerifiedRFFit(manifest, encoder, forest, forest_path, path)


def refit_rf_on_all_development(
    *,
    mode: ExecutionMode = "run_or_load",
    project_root: str | Path = ROOT,
    device: str = "auto",
) -> VerifiedRFFit:
    """Fit the fixed RF head once on every valid development Season image."""
    if mode not in {"run", "load", "run_or_load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    root = Path(project_root).resolve()
    manifest_path = _inside_root(DEFAULT_MANIFEST_PATH, root)
    forest_path = _inside_root(DEFAULT_FOREST_PATH, root)
    history_path = _inside_root(DEFAULT_HISTORY_PATH, root)
    if manifest_path.exists():
        if mode == "run":
            raise FileExistsError("RF refit already exists; use load to preserve provenance")
        return load_verified_rf_fit(project_root=root, device=device)
    if mode == "load":
        raise FileNotFoundError(manifest_path)
    if forest_path.exists() or history_path.exists():
        raise FileExistsError("partial RF refit artifacts need audit before another fit")

    encoder, spec, development = _verified_inputs(root, device=device)
    selection = _selection_record(root)
    training_ids = development["id"].astype(int).tolist()
    parameters = _rf_parameters(spec)
    implementation = implementation_sha256(*IMPLEMENTATION_PATHS, root=root)
    inputs = {
        "splits": _record(_inside_root(SPLITS_CSV, root), root),
        "label_maps": _record(_inside_root(LABEL_MAPS_JSON, root), root),
        "rf_config": _record(_inside_root(DEFAULT_RANDOM_FOREST_CONFIG, root), root),
    }
    source = {
        "run_id": encoder.run_id,
        "manifest": _record(encoder.manifest_path, root),
        "bundle": _record(encoder.bundle_path, root),
    }
    run_id = new_run_id(EXPERIMENT_ID, None, PRIMARY_SEED)
    run = RunRecord(
        run_id=run_id,
        experiment_id=EXPERIMENT_ID,
        fold=None,
        seed=PRIMARY_SEED,
        config_sha256=inputs["rf_config"]["sha256"],
        split_sha256=inputs["splits"]["sha256"],
        label_map_sha256=inputs["label_maps"]["sha256"],
        implementation_sha256=implementation,
        stage="post_submission_full_development_refit",
        model_family="frozen_i2_encoder_plus_random_forest",
        benchmark_only=True,
        final_eligible=False,
        scratch=True,
        transform_id="verified_i2_development_inference_no_augmentation",
        loss_id="random_forest_gini_on_frozen_i2_embedding",
        epochs_requested=0,
        primary_metric_name="not_applicable_full_refit",
    )
    started = time.perf_counter()
    with tracked_run(RunRegistry(_inside_root(RUNS_CSV, root)), run) as tracked:
        dataset = EncodedClassificationDataset(
            development,
            transform=encoder.transform,
            target="season",
            label_to_index={label: index for index, label in enumerate(SEASON_LABELS)},
            root=root,
        )
        loader = DataLoader(
            dataset,
            batch_size=spec.embedding_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=encoder.device.type == "cuda",
        )
        features, targets, observed_ids = _extract_embeddings(
            encoder.model, loader, device=encoder.device
        )
        if observed_ids != training_ids or features.shape != (32_753, 256) or (
            not np.isfinite(features).all()
        ):
            raise ValueError("RF refit embeddings do not cover the canonical development set")
        expected_targets = development["season"].map(
            {label: index for index, label in enumerate(SEASON_LABELS)}
        ).to_numpy(dtype=np.int64)
        if not np.array_equal(targets, expected_targets):
            raise ValueError("RF refit labels differ from canonical development labels")
        forest = RandomForestClassifier(**parameters)
        forest.fit(features, targets)
        if not np.array_equal(forest.classes_, np.arange(4)):
            raise ValueError("RF refit did not learn the four canonical classes")
        if not math.isfinite(float(forest.oob_score_)):
            raise ValueError("RF training diagnostic is non-finite")
        _atomic_joblib(forest_path, forest)
        history = {
            "run_id": run_id,
            "role": "post_submission_full_development_refit",
            "selection": "five_fold_development_oof_only",
            "holdout_used_for_fit": False,
            "training_rows": len(features),
            "training_id_sha256": canonical_sha256(training_ids),
            "embedding_dimension": features.shape[1],
            "embedding_dtype": str(features.dtype),
            "class_counts": {
                label: int((targets == index).sum())
                for index, label in enumerate(SEASON_LABELS)
            },
            "forest_parameters": parameters,
            "oob_accuracy_training_diagnostic_only": float(forest.oob_score_),
            "tree_node_count": int(
                sum(tree.tree_.node_count for tree in forest.estimators_)
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
        atomic_write_json(history_path, history)
        tracked.epochs_completed = 0
        tracked.runtime_seconds = time.perf_counter() - started
        tracked.parameter_count = history["tree_node_count"]
        tracked.checkpoint_path = forest_path.relative_to(root).as_posix()
        tracked.checkpoint_sha256 = compute_sha256(forest_path)
        tracked.history_path = history_path.relative_to(root).as_posix()
        tracked.history_sha256 = compute_sha256(history_path)

    manifest = {
        "schema_version": "1.0.0",
        "role": "post_submission_full_development_refit",
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "training_rows": len(development),
        "training_id_sha256": canonical_sha256(training_ids),
        "embedding_dimension": 256,
        "labels": list(SEASON_LABELS),
        "seed": PRIMARY_SEED,
        "rf_parameters": parameters,
        "source_encoder": source,
        "selection_evidence": selection,
        "inputs": inputs,
        "implementation_sha256": implementation,
        "forest": _record(forest_path, root),
        "history": _record(history_path, root),
    }
    atomic_write_json(manifest_path, manifest)
    return load_verified_rf_fit(project_root=root, device=device)
