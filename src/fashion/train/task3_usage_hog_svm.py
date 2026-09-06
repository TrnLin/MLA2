"""Frozen Usage U2 screen: full-RGB HOG with a calibrated linear SVM."""

from __future__ import annotations

import json
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from fashion.config import LABEL_MAPS_JSON, RANDOM_SEED, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.data.splits import cv_assignment_digest
from fashion.data.task3_clean_slate_eda import build_clean_slate_audit_contract
from fashion.train.metrics import classification_metrics
from fashion.train.registry import RunRegistry
from fashion.train.resource_worker import run_supervised
from fashion.train.task3_clean_slate import (
    CLEAN_SLATE_SCREEN_FOLDS,
    HOST_MEMORY_LIMIT_BYTES,
    SCREEN_SECONDS_PER_FOLD_LIMIT,
    _canonical_inner_splits,
    _classes,
    _completed_fold,
    _expand_probabilities,
    _feature_contract_valid,
    _fold_paths,
    _json_dump,
    _matrix_rows,
    _peak_memory_bytes,
    _pickle_dump,
    _prediction_frame,
    _registry_start,
    _relative,
    _screen_folds,
    _stable_digest,
    _teacher_path,
    _valid,
    build_fixed_feature_cache,
    fixed_feature_vector,
)
from fashion.train.task3_decisions import (
    CORE_CORRUPTIONS,
    DECISION_BOOTSTRAP_REPETITIONS,
    oof_metrics,
    robustness_changes,
    validate_oof,
)
from fashion.train.task3_experiments import effective_number_class_weights
from fashion.train.task3_usage_hog_decision import (
    SUPPORTED_USAGE_CLASSES,
    U2_GATE_VERSION,
    evaluate_usage_u2,
)

USAGE_HOG_EXPERIMENT_ID = "t3_usage_v2_u2_full_rgb_hog_svm"
USAGE_HOG_HYPOTHESIS_ID = "full_rgb_hog_linear_margin_reduces_usage_overfit"
USAGE_HOG_ARTIFACT_ROOT = "experiments/t3_usage_v2_u2_full_rgb_hog_svm"


@dataclass(frozen=True)
class UsageHogSvmConfig:
    """One frozen screen; there is no feature or C search."""

    target: str = "usage"
    feature_view: str = "full_rgb_hog"
    c: float = 1.0
    inner_folds: int = 4
    calibration: str = "sigmoid"
    class_weight_strategy: str = "effective_number_sample_weight"
    class_weight_beta: float = 0.999
    class_weight_cap: float = 5.0
    max_iterations: int = 20_000
    seed: int = RANDOM_SEED
    scratch: bool = True
    submission_eligible: bool = True
    contract_version: str = U2_GATE_VERSION
    weighting_scope: str = "each_inner_base_training_subset"
    calibration_weighting: str = "unweighted_natural_inner_validation_distribution"
    unsupported_home: str = "exclude_fit_and_calibration; zero_probability; official_score_retained"
    supported_classes: tuple[str, ...] = SUPPORTED_USAGE_CLASSES
    robustness_corruptions: tuple[str, ...] = CORE_CORRUPTIONS
    resource_enforcement: str = "spawned_fold_wall_deadline_and_hard_address_space_v1"
    solver_threads: int = 1


class WeightedScaledLinearSVC(ClassifierMixin, BaseEstimator):
    """Fit weighted scaling and a one-vs-rest linear SVM inside each CV split."""

    def __init__(
        self,
        *,
        c: float = 1.0,
        max_iterations: int = 20_000,
        seed: int = 2753,
        class_weight_beta: float = 0.999,
        class_weight_cap: float = 5.0,
        required_classes: tuple[str, ...] | None = None,
    ):
        self.c = c
        self.max_iterations = max_iterations
        self.seed = seed
        self.class_weight_beta = class_weight_beta
        self.class_weight_cap = class_weight_cap
        self.required_classes = required_classes

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> WeightedScaledLinearSVC:
        labels = np.asarray(labels).astype(str)
        self.weight_classes_ = tuple(self.required_classes or sorted(set(labels)))
        if set(labels) != set(self.weight_classes_):
            raise ValueError("an inner SVM training subset lacks a supported class")
        self.training_counts_ = np.array([(labels == name).sum() for name in self.weight_classes_])
        self.class_weights_ = effective_number_class_weights(
            self.training_counts_, beta=self.class_weight_beta, cap=self.class_weight_cap
        )
        if sample_weight is None:
            lookup = dict(zip(self.weight_classes_, self.class_weights_, strict=True))
            sample_weight = np.array([lookup[name] for name in labels], dtype=float)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(features, sample_weight=sample_weight)
        self.estimator_ = LinearSVC(
            C=self.c,
            dual="auto",
            max_iter=self.max_iterations,
            random_state=self.seed,
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            self.estimator_.fit(
                self.scaler_.transform(features), labels, sample_weight=sample_weight
            )
        self.solver_warnings_ = [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in recorded
        ]
        self.solver_iterations_ = int(self.estimator_.n_iter_)
        self.solver_converged_ = not any(
            issubclass(warning.category, ConvergenceWarning) for warning in recorded
        )
        for warning in recorded:
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.scaler_.n_features_in_
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return self.estimator_.decision_function(self.scaler_.transform(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(self.scaler_.transform(features))


def _calibrated_parameter_count(model: CalibratedClassifierCV) -> int:
    count = 0
    for calibrated in model.calibrated_classifiers_:
        estimator = calibrated.estimator.estimator_
        count += int(estimator.coef_.size + estimator.intercept_.size)
        count += 2 * len(calibrated.calibrators)
    return count


def _inner_training_scope(
    training: pd.DataFrame, *, outer_fold: int
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    fitting = training.loc[training["usage"].ne("Home")].reset_index(drop=True)
    inner_splits = _canonical_inner_splits(fitting, outer_fold=outer_fold)
    for inner_train, calibration in inner_splits:
        for role, positions in (("base training", inner_train), ("calibration", calibration)):
            if set(fitting.iloc[positions]["usage"].astype(str)) != set(SUPPORTED_USAGE_CLASSES):
                raise ValueError(f"outer fold {outer_fold} {role} lacks a supported Usage class")
    return fitting, inner_splits


def _load_parent_evidence(
    anchor_path: Path,
    *,
    parent_run_ids: Sequence[str],
    parent_registry_path: Path,
    splits: pd.DataFrame,
    classes: Sequence[str],
    root: Path,
) -> dict[str, Any]:
    """Check E2 lineage and saved corruption evidence before spending any fit time."""
    if len(parent_run_ids) != 5 or len(set(parent_run_ids)) != 5:
        raise ValueError("Usage U2 requires five distinct E2 parent run IDs")
    expected = get_samples(splits, partition="development", target="usage")
    parent = validate_oof(
        pd.read_csv(anchor_path, keep_default_na=False),
        expected,
        target="usage",
        classes=classes,
        run_ids_by_fold=dict(enumerate(parent_run_ids)),
        allow_legacy_na=True,
    )
    registry = pd.read_csv(parent_registry_path, keep_default_na=False)
    directory = anchor_path.parent.parent
    robustness, gaps, hashes = [], {}, {}
    for fold, run_id in enumerate(parent_run_ids):
        matches = registry.loc[registry["run_id"].eq(run_id)]
        if len(matches) != 1:
            raise ValueError(f"expected one E2 registry row for {run_id}")
        row = matches.iloc[0]
        if (
            row["status"] != "complete"
            or row["target"] != "usage"
            or row["experiment_id"] != "t3_usage_class_balanced_smallcnn"
            or int(row["validation_fold"]) != fold
            or int(row["seed"]) != 2753
            or str(row["scratch"]).lower() != "true"
            or str(row["debug"]).lower() != "false"
        ):
            raise ValueError(f"parent is not the completed scratch E2 fold: {run_id}")
        for field, path in (
            ("split_digest", root / "data/processed/splits.csv"),
            ("label_map_digest", root / "data/processed/label_maps.json"),
        ):
            if row[field] != compute_sha256(path):
                raise ValueError(f"E2 {field} disagrees with current data")
        run_dir = directory / run_id
        for name, field in (
            ("oof_predictions.csv", "prediction_sha256"),
            ("final_epoch.pt", "checkpoint_sha256"),
        ):
            actual = compute_sha256(run_dir / name)
            if actual != row[field]:
                raise ValueError(f"E2 {name} hash differs from its registry row")
            hashes[f"{run_id}/{name}"] = actual
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if metrics != json.loads(row["metrics_json"]):
            raise ValueError(f"E2 metrics differ from the registered values: {run_id}")
        fold_expected = expected[expected["cv_fold"].eq(fold)]
        fold_oof = validate_oof(
            pd.read_csv(run_dir / "oof_predictions.csv", keep_default_na=False),
            fold_expected,
            target="usage",
            classes=classes,
            run_ids_by_fold={fold: run_id},
            allow_legacy_na=True,
        )
        pd.testing.assert_frame_equal(
            parent[parent.cv_fold.eq(fold)].reset_index(drop=True), fold_oof, check_exact=True
        )
        calculated = oof_metrics(fold_oof, classes)
        for metric in ("macro_f1", "nll", "brier", "ece_15"):
            if not np.isclose(float(metrics[metric]), calculated[metric], atol=1e-7, rtol=0):
                raise ValueError(f"E2 {metric} disagrees with its saved probabilities")
        robustness.append(pd.read_csv(run_dir / "robustness.csv", keep_default_na=False))
        hashes[f"{run_id}/robustness.csv"] = compute_sha256(run_dir / "robustness.csv")
        # Older E2 curves are online training scores, so they cannot unlock this branch.
        if "final_train_eval_macro_f1" in metrics:
            gaps[fold] = float(metrics["final_train_eval_macro_f1"] - calculated["macro_f1"])
    all_robustness = pd.concat(robustness, ignore_index=True)
    robustness_changes(
        all_robustness,
        clean_by_fold={
            f: oof_metrics(parent[parent.cv_fold.eq(f)], classes)["macro_f1"] for f in range(5)
        },
        run_ids_by_fold=dict(enumerate(parent_run_ids)),
    )
    return {
        "predictions": parent,
        "robustness": all_robustness,
        "clean_gaps": {f: gaps[f] for f in (0, 4)} if {0, 4}.issubset(gaps) else None,
        "artifact_sha256": hashes,
        "registry_sha256": compute_sha256(parent_registry_path),
        "anchor_sha256": compute_sha256(anchor_path),
    }


def _check_resources(started: float) -> None:
    if time.perf_counter() - started > SCREEN_SECONDS_PER_FOLD_LIMIT:
        raise RuntimeError("U2 exceeded the frozen 90-minute per-fold time limit")
    if _peak_memory_bytes() > HOST_MEMORY_LIMIT_BYTES:
        raise RuntimeError("U2 exceeded the frozen 7-GiB host memory limit")


def _evaluate_corruptions(
    model: CalibratedClassifierCV,
    validation: pd.DataFrame,
    *,
    classes: Sequence[str],
    root: Path,
    run_id: str,
    run_dir: Path,
    clean_f1: float,
    started: float,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Evaluate all old corruptions in small batches and save paired probabilities."""
    records = validation.to_dict("records")
    labels = (
        validation["usage"]
        .map(dict(zip(classes, range(len(classes)), strict=True)))
        .to_numpy(dtype=int)
    )
    rows, hashes = [], {}
    for corruption in CORE_CORRUPTIONS:
        probabilities = []

        def extract(record: Mapping[str, Any]) -> np.ndarray:
            with Image.open(_teacher_path(root, str(record["path"]))) as source:
                return fixed_feature_vector(source, view="full_rgb_hog", corruption=corruption)

        with ThreadPoolExecutor(max_workers=6) as executor:
            for start in range(0, len(records), 256):
                _check_resources(started)
                matrix = np.stack(list(executor.map(extract, records[start : start + 256])))
                probabilities.append(
                    _expand_probabilities(model.predict_proba(matrix), model.classes_, classes)
                )
        values = np.concatenate(probabilities)
        score = classification_metrics(labels, values, classes)
        path = run_dir / "corruptions" / f"{corruption}.csv"
        path.parent.mkdir(exist_ok=True)
        write_deterministic_csv(
            _prediction_frame(
                validation, target="usage", classes=classes, probabilities=values, run_id=run_id
            ),
            path,
            index=False,
        )
        hashes[path.relative_to(run_dir).as_posix()] = compute_sha256(path)
        rows.append(
            {
                "run_id": run_id,
                "validation_fold": int(validation["cv_fold"].iloc[0]),
                "corruption": corruption,
                "macro_f1": score["macro_f1"],
                "macro_f1_change": score["macro_f1"] - clean_f1,
                "nll": score["nll"],
                "ece_15": score["ece_15"],
            }
        )
        print(f"[task3-usage-hog] {corruption}: macro-F1={score['macro_f1']:.6f}", flush=True)
    return pd.DataFrame(rows), hashes


def prepare_usage_hog_features(
    *,
    root: str | Path = ROOT,
    output_root: str | Path,
    workers: int | None = None,
    local_work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build or reuse a label-blind pure full-RGB HOG cache."""
    root = Path(root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    audit = build_clean_slate_audit_contract(splits, root=root)
    cache = build_fixed_feature_cache(
        splits,
        view="full_rgb_hog",
        audit_contract_hash=str(audit["audit_contract_hash"]),
        output_dir=Path(output_root) / USAGE_HOG_ARTIFACT_ROOT / "feature_cache",
        root=root,
        workers=workers,
        local_work_dir=local_work_dir,
    )
    return {"audit_contract": audit, "usage": cache}


def _run_fold(
    fold: int,
    *,
    splits: pd.DataFrame,
    label_maps: Mapping[str, Mapping[str, Any]],
    cache: Mapping[str, Any],
    audit_hash: str,
    parent_run_ids: Sequence[str],
    output_root: Path,
    registry_path: Path,
    registry_mirrors: Sequence[str | Path],
    root: Path,
    reuse_completed: bool,
) -> dict[str, Any]:
    config = UsageHogSvmConfig()
    training_all, validation_all = get_cv_split(splits, fold)
    training = _valid(training_all, "usage").reset_index(drop=True)
    validation = _valid(validation_all, "usage").reset_index(drop=True)
    if set(training["product_family_group"]).intersection(validation["product_family_group"]):
        raise ValueError("an outer product family crosses the usage fold")
    classes = _classes(label_maps, "usage")
    if set(classes) != {*SUPPORTED_USAGE_CLASSES, "Home"}:
        raise ValueError("U2 requires the nine fixed Usage classes")
    fitting, inner_splits = _inner_training_scope(training, outer_fold=fold)
    config_payload = {
        **asdict(config),
        "validation_fold": fold,
        "audit_contract_hash": audit_hash,
        "screen_folds": list(CLEAN_SLATE_SCREEN_FOLDS),
        "feature_contract": Path(str(cache["contract_path"])).name,
        "feature_contract_sha256": compute_sha256(str(cache["contract_path"])),
        "parent_run_ids": list(parent_run_ids),
        "single_changed_factor": "cnn_pixels_to_fixed_full_rgb_hog_linear_svm",
        "augmentation": "none",
    }
    config_hash = _stable_digest(config_payload)
    target_dir = output_root / USAGE_HOG_ARTIFACT_ROOT / "usage"
    if reuse_completed and (
        completed := _completed_fold(
            target_dir=target_dir,
            fold=fold,
            config_hash=config_hash,
            registry_path=registry_path,
        )
    ):
        _verify_completed_u2_fold(
            completed,
            registry_path=registry_path,
            expected=validation,
            fitting=fitting,
            classes=classes,
            root=root,
        )
        return completed

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"t3_usage_v2_u2_hogsvm_usage_hogsvm_f{fold}_s{config.seed}_"
        f"{config_hash[:12]}_{timestamp}{uuid.uuid4().hex[:6]}"
    )
    paths = _fold_paths(target_dir, run_id)
    _json_dump(config_payload, paths["config"])
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    _registry_start(
        registry,
        run_id=run_id,
        experiment_id=USAGE_HOG_EXPERIMENT_ID,
        hypothesis_id=USAGE_HOG_HYPOTHESIS_ID,
        target="usage",
        fold=fold,
        config_hash=config_hash,
        config_path=paths["config"],
        history_path=paths["history"],
        training=fitting,
        validation=validation,
        model_family="full_rgb_hog_calibrated_linear_svm",
        root=root,
        parent_run_ids=parent_run_ids,
    )
    try:
        result, wall_seconds = run_supervised(
            _fit_registered_fold,
            {
                "fold": fold,
                "fitting": fitting,
                "training": training,
                "validation": validation,
                "classes": classes,
                "inner_splits": inner_splits,
                "cache": cache,
                "config": config,
                "config_payload": config_payload,
                "config_hash": config_hash,
                "audit_hash": audit_hash,
                "paths": paths,
                "run_id": run_id,
                "registry_path": registry_path,
                "registry_mirrors": registry_mirrors,
                "root": root,
            },
            seconds=SCREEN_SECONDS_PER_FOLD_LIMIT,
            memory_bytes=HOST_MEMORY_LIMIT_BYTES,
        )
        # Only the supervising process can mark a run complete, after the worker
        # exits within its budget. A killed worker can never leave a complete row.
        result["metrics"]["fold_wall_seconds"] = wall_seconds
        result["metrics"]["resource_enforcement"] = config.resource_enforcement
        result["metrics"]["hard_address_space_limit_bytes"] = HOST_MEMORY_LIMIT_BYTES
        _json_dump(result["metrics"], paths["metrics"])
        completion = result.pop("completion_fields")
        completion["metrics_json"] = result["metrics"]
        registry.complete(run_id, completion)
        return result
    except BaseException as error:
        registry.fail(run_id, error, last_completed_stage="supervised_fold_incomplete")
        raise


def _fit_registered_fold(
    *,
    fold,
    fitting,
    training,
    validation,
    classes,
    inner_splits,
    cache,
    config,
    config_payload,
    config_hash,
    audit_hash,
    paths,
    run_id,
    registry_path,
    registry_mirrors,
    root,
) -> dict[str, Any]:
    """Worker-only fitting and diagnostics for an already registered run."""
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    started = time.perf_counter()
    last_stage = "registered_before_first_model_fit"
    try:
        x_train = _matrix_rows(fitting, cache)
        x_validation = _matrix_rows(validation, cache)
        y_train = fitting["usage"].astype(str).to_numpy()
        estimator = WeightedScaledLinearSVC(
            c=config.c,
            max_iterations=config.max_iterations,
            seed=config.seed + fold,
            class_weight_beta=config.class_weight_beta,
            class_weight_cap=config.class_weight_cap,
            required_classes=SUPPORTED_USAGE_CLASSES,
        )
        model = CalibratedClassifierCV(
            estimator=estimator,
            method=config.calibration,
            cv=inner_splits,
            ensemble=True,
            n_jobs=1,
        )
        print(
            f"[task3-usage-hog] fold={fold}: fitting {config.inner_folds} calibrated SVMs",
            flush=True,
        )
        # The estimator computes weights from its own inner-training y. Passing
        # weights here would also reweight the held-out sigmoid calibration.
        with threadpool_limits(limits=config.solver_threads):
            model.fit(x_train, y_train)
        _check_resources(started)
        validation_probabilities = _expand_probabilities(
            model.predict_proba(x_validation), model.classes_, classes
        )
        training_probabilities = _expand_probabilities(
            model.predict_proba(_matrix_rows(training, cache)), model.classes_, classes
        )
        last_stage = "outer_model_fit_complete"
        registry.update(run_id, {"last_completed_stage": last_stage})

        predictions = _prediction_frame(
            validation,
            target="usage",
            classes=classes,
            probabilities=validation_probabilities,
            run_id=run_id,
        )
        write_deterministic_csv(predictions, paths["predictions"], index=False)
        class_lookup = {name: index for index, name in enumerate(classes)}
        validation_labels = validation["usage"].astype(str).map(class_lookup).to_numpy(dtype=int)
        training_labels = training["usage"].astype(str).map(class_lookup).to_numpy(dtype=int)
        metrics = classification_metrics(validation_labels, validation_probabilities, classes)
        train_metrics = classification_metrics(training_labels, training_probabilities, classes)
        parameter_count = _calibrated_parameter_count(model)
        inner_history = []
        for calibrated, (_, calibration_rows) in zip(
            model.calibrated_classifiers_, inner_splits, strict=True
        ):
            base = calibrated.estimator
            calibration_frame = fitting.iloc[calibration_rows]
            for index, name in enumerate(base.weight_classes_):
                inner_history.append(
                    {
                        "inner_calibration_fold": int(calibration_frame["cv_fold"].iloc[0]),
                        "class_name": name,
                        "base_training_count": int(base.training_counts_[index]),
                        "effective_number_weight": float(base.class_weights_[index]),
                        "calibration_count": int(calibration_frame["usage"].eq(name).sum()),
                        "calibration_weighted": False,
                        "solver_iterations": base.solver_iterations_,
                        "solver_converged": base.solver_converged_,
                        "solver_warnings": json.dumps(base.solver_warnings_),
                    }
                )
        write_deterministic_csv(pd.DataFrame(inner_history), paths["history"], index=False)
        _pickle_dump(
            {
                "run_id": run_id,
                "config": config_payload,
                "class_names": classes,
                "model": model,
                "feature_view": config.feature_view,
            },
            paths["checkpoint"],
        )
        train_seconds = time.perf_counter() - started
        robustness, artifact_hashes = _evaluate_corruptions(
            model,
            validation,
            classes=classes,
            root=root,
            run_id=run_id,
            run_dir=paths["run_dir"],
            clean_f1=float(metrics["macro_f1"]),
            started=started,
        )
        robustness_path = paths["run_dir"] / "robustness.csv"
        write_deterministic_csv(robustness, robustness_path, index=False)
        _check_resources(started)
        fold_wall_seconds = time.perf_counter() - started
        metrics.update(
            {
                "run_id": run_id,
                "target": "usage",
                "validation_fold": fold,
                "experiment_id": USAGE_HOG_EXPERIMENT_ID,
                "hypothesis_id": USAGE_HOG_HYPOTHESIS_ID,
                "model_family": "full_rgb_hog_calibrated_linear_svm",
                "config_hash": config_hash,
                "audit_contract_hash": audit_hash,
                "final_train_macro_f1": train_metrics["macro_f1"],
                "final_train_validation_macro_f1_gap": float(
                    train_metrics["macro_f1"] - metrics["macro_f1"]
                ),
                "parameter_count": parameter_count,
                "train_seconds": train_seconds,
                "diagnostic_seconds": fold_wall_seconds - train_seconds,
                "fold_wall_seconds": fold_wall_seconds,
                "peak_memory_bytes": _peak_memory_bytes(),
                "screen_scope": "canonical_outer_folds_0_and_4",
                "robustness_scope": "all_five_core_corruptions_on_outer_validation",
                "class_weight_scope": "each_inner_base_training_subset",
                "calibration_weighted": False,
                "unsupported_home_probability": 0.0,
                "fitting_rows": len(fitting),
                "clean_training_evaluation_rows": len(training),
                "excluded_home_fitting_rows": int(training["usage"].eq("Home").sum()),
                "clean_training_measurement": (
                    "ensemble_on_outer_training_with_mixed_fit_and_calibration_exposure"
                ),
            }
        )
        metrics["prediction_sha256"] = compute_sha256(paths["predictions"])
        metrics["checkpoint_sha256"] = compute_sha256(paths["checkpoint"])
        artifact_hashes.update(
            {
                path.relative_to(paths["run_dir"]).as_posix(): compute_sha256(path)
                for path in (
                    paths["config"],
                    paths["history"],
                    paths["checkpoint"],
                    paths["predictions"],
                    robustness_path,
                )
            }
        )
        metrics["artifact_sha256"] = artifact_hashes
        _json_dump(metrics, paths["metrics"])
        return {
            "run_id": run_id,
            "run_dir": str(paths["run_dir"]),
            "prediction_path": str(paths["predictions"]),
            "metrics_path": str(paths["metrics"]),
            "metrics": metrics,
            "completion_fields": {
                "parameter_count": parameter_count,
                "checkpoint_path": _relative(paths["checkpoint"], root),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "prediction_path": _relative(paths["predictions"], root),
                "prediction_sha256": metrics["prediction_sha256"],
                "metrics_json": metrics,
                "train_seconds": train_seconds,
                "peak_memory_bytes": metrics["peak_memory_bytes"],
                "checkpoint_bytes": paths["checkpoint"].stat().st_size,
                "last_completed_stage": "screen_fold_complete",
            },
        }
    except BaseException as error:
        registry.fail(run_id, error, last_completed_stage=last_stage)
        raise


def _verify_completed_u2_fold(
    result: Mapping[str, Any],
    *,
    registry_path: Path,
    expected: pd.DataFrame,
    fitting: pd.DataFrame,
    classes: Sequence[str],
    root: Path,
) -> pd.DataFrame:
    run_dir = Path(str(result["run_dir"]))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    run_id = str(result["run_id"])
    registry = pd.read_csv(registry_path, keep_default_na=False)
    matches = registry[registry["run_id"].eq(run_id)]
    if len(matches) != 1 or matches.iloc[0]["status"] != "complete":
        raise ValueError("U2 needs exactly one completed registry row per fold")
    row = matches.iloc[0]
    config = json.loads((run_dir / "config.json").read_text())
    if (
        config.get("contract_version") != U2_GATE_VERSION
        or config.get("resource_enforcement") != UsageHogSvmConfig().resource_enforcement
    ):
        raise ValueError("U2 cannot reuse a legacy calibration/gate/resource contract")
    if _stable_digest(config) != row["config_hash"] or row["config_hash"] != metrics["config_hash"]:
        raise ValueError("U2 config hash disagrees with its registry or metrics")
    if json.loads(row["metrics_json"]) != metrics:
        raise ValueError("U2 metrics differ from the registered metrics")
    if metrics["run_id"] != run_id or row["experiment_id"] != USAGE_HOG_EXPERIMENT_ID:
        raise ValueError("U2 run identity disagrees")
    if row["target"] != "usage" or int(row["validation_fold"]) != int(expected["cv_fold"].iloc[0]):
        raise ValueError("U2 registry scope disagrees")
    if str(row["scratch"]).lower() != "true" or str(row["debug"]).lower() != "false":
        raise ValueError("U2 registry must record a non-debug scratch run")
    if json.loads(row["parent_run_ids"]) != config["parent_run_ids"]:
        raise ValueError("U2 registered parent lineage disagrees")
    for name, path in (
        ("split_digest", root / "data/processed/splits.csv"),
        ("label_map_digest", root / "data/processed/label_maps.json"),
    ):
        if row[name] != compute_sha256(path):
            raise ValueError(f"U2 {name} no longer matches current data")
    for prefix, frame in (("training", fitting), ("validation", expected)):
        if (
            int(row[f"{prefix}_product_count"]) != len(frame)
            or int(row[f"{prefix}_family_count"]) != frame["product_family_group"].nunique()
        ):
            raise ValueError(f"U2 registry {prefix} counts disagree")
    required_hashes = {
        "config.json",
        "solver_history.csv",
        "model.pkl",
        "oof_predictions.csv",
        "robustness.csv",
        *[f"corruptions/{c}.csv" for c in CORE_CORRUPTIONS],
    }
    hashes = metrics.get("artifact_sha256", {})
    if set(hashes) != required_hashes:
        raise ValueError("U2 artifact manifest must include all corruption predictions")
    for name in sorted(required_hashes):
        if compute_sha256(run_dir / name) != hashes[name]:
            raise ValueError(f"U2 artifact hash changed: {name}")
    if (
        row["checkpoint_sha256"] != hashes["model.pkl"]
        or row["prediction_sha256"] != hashes["oof_predictions.csv"]
    ):
        raise ValueError("U2 registry artifact hashes disagree")
    predictions = validate_oof(
        pd.read_csv(run_dir / "oof_predictions.csv", keep_default_na=False),
        expected,
        target="usage",
        classes=classes,
        run_ids_by_fold={int(row["validation_fold"]): run_id},
    )
    calculated = oof_metrics(predictions, classes)
    for name in ("macro_f1", "nll", "brier", "ece_15"):
        if not np.isclose(metrics[name], calculated[name], atol=1e-8, rtol=0):
            raise ValueError(f"U2 {name} differs from saved probabilities")
    robustness = pd.read_csv(run_dir / "robustness.csv", keep_default_na=False)
    fold = int(row["validation_fold"])
    robustness_changes(
        robustness, clean_by_fold={fold: calculated["macro_f1"]}, run_ids_by_fold={fold: run_id}
    )
    for corruption in CORE_CORRUPTIONS:
        perturbed = validate_oof(
            pd.read_csv(run_dir / "corruptions" / f"{corruption}.csv", keep_default_na=False),
            expected,
            target="usage",
            classes=classes,
            run_ids_by_fold={fold: run_id},
        )
        home_column = f"probability_{list(classes).index('Home')}_Home"
        if not perturbed[home_column].eq(0).all():
            raise ValueError("U2 corruption predictions must keep P(Home)=0")
        perturbed_score = oof_metrics(perturbed, classes)["macro_f1"]
        saved_score = float(
            robustness.loc[robustness.corruption.eq(corruption), "macro_f1"].iloc[0]
        )
        if not np.isclose(perturbed_score, saved_score, atol=1e-8, rtol=0):
            raise ValueError("U2 corrupted probabilities disagree with robustness scores")
    return robustness


def check_usage_hog_svm_setup(
    *,
    root: str | Path = ROOT,
    folds: Iterable[int] = CLEAN_SLATE_SCREEN_FOLDS,
    parent_run_ids: Sequence[str] | None = None,
    anchor_prediction_path: str | Path | None = None,
    parent_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Perform a zero-fit preflight for Usage U2."""
    fold_list = _screen_folds(folds)
    root = Path(root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    blockers = []
    for fold in fold_list:
        training, validation = get_cv_split(splits, fold)
        if set(_valid(training, "usage")["product_family_group"]).intersection(
            _valid(validation, "usage")["product_family_group"]
        ):
            raise ValueError(f"a usage family crosses fold {fold}")
        try:
            _inner_training_scope(_valid(training, "usage"), outer_fold=fold)
        except ValueError as error:
            blockers.append(str(error))
    classes = _classes(label_maps, "usage")
    if set(classes) != {*SUPPORTED_USAGE_CLASSES, "Home"}:
        blockers.append("the label map must contain all nine fixed Usage classes")
    parent_audit = "not_requested"
    if any(
        value is not None
        for value in (parent_run_ids, anchor_prediction_path, parent_registry_path)
    ):
        if parent_run_ids is None or anchor_prediction_path is None or parent_registry_path is None:
            blockers.append("parent lineage, OOF path and registry path are all required")
        else:
            _load_parent_evidence(
                Path(anchor_prediction_path),
                parent_run_ids=parent_run_ids,
                parent_registry_path=Path(parent_registry_path),
                splits=splits,
                classes=classes,
                root=root,
            )
            parent_audit = "verified"
    descriptor = fixed_feature_vector(
        np.full((80, 60, 3), 255, dtype=np.uint8), view="full_rgb_hog"
    )
    estimated_bytes = int(splits["partition"].eq("development").sum()) * descriptor.nbytes
    if estimated_bytes > HOST_MEMORY_LIMIT_BYTES:
        raise RuntimeError("estimated full-RGB HOG cache exceeds the host memory limit")
    return {
        "ready": not blockers,
        "training_blockers": blockers,
        "folds": list(fold_list),
        "usage_classes": _classes(label_maps, "usage"),
        "model": "full-RGB HOG + weighted StandardScaler + calibrated LinearSVC(C=1)",
        "calibration": "unweighted, natural inner-fold frequency",
        "class_weight_scope": "each inner base-training subset",
        "unsupported_class": (
            "Home: excluded from fit/calibration, zero probability, official score retained"
        ),
        "parent_evidence": parent_audit,
        "feature_columns": len(descriptor),
        "estimated_cache_bytes": estimated_bytes,
        "host_memory_limit_bytes": HOST_MEMORY_LIMIT_BYTES,
        "resource_enforcement": UsageHogSvmConfig().resource_enforcement,
        "hard_memory_scope": "worker_address_space_including_mappings_stricter_than_RSS",
        "model_fits": 0,
        "optimizer_steps": 0,
        "execution_device": "cpu",
    }


def run_usage_hog_svm_screen(
    *,
    prepared_features: Mapping[str, Any],
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    folds: Iterable[int] = CLEAN_SLATE_SCREEN_FOLDS,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    anchor_prediction_path: str | Path | None = None,
    parent_registry_path: str | Path | None = None,
    reuse_completed: bool = True,
) -> dict[str, Any]:
    """Run the frozen two-fold Usage U2 screen and pool matched OOF rows."""
    if len(parent_run_ids) != 5 or len(set(parent_run_ids)) != 5:
        raise ValueError("Usage U2 requires five distinct E2 parent run IDs")
    fold_list = _screen_folds(folds)
    root = Path(root)
    output_root = Path(output_root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    classes = _classes(label_maps, "usage")
    if anchor_prediction_path is None:
        raise ValueError("U2 requires its matched E2 predictions before any model fit")
    parent_registry = (
        Path(parent_registry_path)
        if parent_registry_path
        else root / "results/evidence/task3/results/runs.csv"
    )
    parent = _load_parent_evidence(
        Path(anchor_prediction_path),
        parent_run_ids=parent_run_ids,
        parent_registry_path=parent_registry,
        splits=splits,
        classes=classes,
        root=root,
    )
    preflight = check_usage_hog_svm_setup(root=root, folds=fold_list)
    if not preflight["ready"]:
        raise ValueError(f"U2 preflight failed: {preflight['training_blockers']}")
    audit_hash = str(prepared_features["audit_contract"]["audit_contract_hash"])
    if prepared_features["audit_contract"]["split_digest"] != cv_assignment_digest(splits):
        raise ValueError("U2 feature audit uses a different canonical split")
    cache = prepared_features["usage"]
    contract_path = Path(str(cache["contract_path"]))
    if not _feature_contract_valid(
        contract_path, {"audit_contract_hash": audit_hash, "view": "full_rgb_hog"}
    ):
        raise ValueError("U2 HOG cache fails its source/hash contract")
    contract = json.loads(contract_path.read_text())
    for field in ("matrix_path", "ids_path"):
        path = Path(str(cache[field]))
        if path.resolve().parent != contract_path.resolve().parent or contract[
            "artifact_sha256"
        ].get(path.name) != compute_sha256(path):
            raise ValueError("U2 cache paths do not identify the verified artifacts")
    ids = pd.read_csv(str(cache["ids_path"]), keep_default_na=False)["id"].astype(int).tolist()
    if ids != sorted(splits.loc[splits.partition.eq("development"), "id"].astype(int).tolist()):
        raise ValueError("U2 feature cache must contain exactly the development IDs")
    results = [
        _run_fold(
            fold,
            splits=splits,
            label_maps=label_maps,
            cache=prepared_features["usage"],
            audit_hash=audit_hash,
            parent_run_ids=parent_run_ids,
            output_root=output_root,
            registry_path=Path(registry_path),
            registry_mirrors=registry_mirrors,
            root=root,
            reuse_completed=reuse_completed,
        )
        for fold in fold_list
    ]
    robust_rows = []
    for fold, result in zip(fold_list, results, strict=True):
        training, validation = get_cv_split(splits, fold)
        fitting, _ = _inner_training_scope(_valid(training, "usage"), outer_fold=fold)
        robust_rows.append(
            _verify_completed_u2_fold(
                result,
                registry_path=Path(registry_path),
                expected=_valid(validation, "usage"),
                fitting=fitting,
                classes=classes,
                root=root,
            )
        )
    predictions = pd.concat(
        [pd.read_csv(str(r["prediction_path"]), keep_default_na=False) for r in results],
        ignore_index=True,
    )
    expected = get_samples(splits, partition="development", target="usage")
    expected = expected.loc[expected.cv_fold.isin(fold_list)]
    gate = evaluate_usage_u2(
        predictions,
        parent["predictions"].loc[parent["predictions"].cv_fold.isin(fold_list)],
        expected=expected,
        classes=classes,
        run_ids_by_fold={f: str(r["run_id"]) for f, r in zip(fold_list, results, strict=True)},
        parent_run_ids_by_fold={f: parent_run_ids[f] for f in fold_list},
        fold_metrics={f: r["metrics"] for f, r in zip(fold_list, results, strict=True)},
        candidate_robustness=pd.concat(robust_rows, ignore_index=True),
        parent_robustness=parent["robustness"].loc[
            parent["robustness"].validation_fold.isin(fold_list)
        ],
        artifact_integrity=True,
        parent_clean_gaps=parent["clean_gaps"],
        bootstrap_repetitions=DECISION_BOOTSTRAP_REPETITIONS,
    )
    metrics = {
        **gate["candidate_metrics"],
        "screen_gate": gate,
        "experiment_id": USAGE_HOG_EXPERIMENT_ID,
        "hypothesis_id": USAGE_HOG_HYPOTHESIS_ID,
        "model_family": "full_rgb_hog_calibrated_linear_svm",
        "fold_run_ids": [r["run_id"] for r in results],
        "validation_folds": list(fold_list),
        "screen_only": True,
        "not_a_five_fold_result": True,
        "fold_macro_f1": [r["metrics"]["macro_f1"] for r in results],
        "fold_macro_f1_sample_sd": float(
            np.std([r["metrics"]["macro_f1"] for r in results], ddof=1)
        ),
        "macro_f1_without_home": gate["macro_f1_without_home"],
        "parent_evidence_sha256": {
            "artifacts": parent["artifact_sha256"],
            "registry": parent["registry_sha256"],
            "anchor": parent["anchor_sha256"],
        },
    }
    directory = output_root / USAGE_HOG_ARTIFACT_ROOT / "usage" / "aggregate_folds_0_4"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "prediction_path": directory / "oof_predictions.csv",
        "metrics_path": directory / "metrics.json",
        "per_class_path": directory / "per_class.csv",
        "confusion_path": directory / "confusion_matrix.csv",
        "screen_gate_path": directory / "screen_gate.json",
    }
    write_deterministic_csv(predictions.sort_values("id"), paths["prediction_path"], index=False)
    write_deterministic_csv(
        pd.DataFrame(metrics["per_class"]), paths["per_class_path"], index=False
    )
    write_deterministic_csv(
        pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
        .rename_axis("true_label")
        .reset_index(),
        paths["confusion_path"],
        index=False,
    )
    write_deterministic_csv(
        pd.DataFrame(gate["checks"]), directory / "decision_checks.csv", index=False
    )
    _json_dump(gate, paths["screen_gate_path"])
    _json_dump(metrics, paths["metrics_path"])
    return {
        "target": "usage",
        "fold_run_ids": metrics["fold_run_ids"],
        "metrics": metrics,
        **{key: str(value) for key, value in paths.items()},
    }
