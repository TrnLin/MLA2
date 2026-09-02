"""First clean-slate Task 3 screen with separate target-specific models.

Gender uses a calibrated linear SVM over fixed foreground HOG, colour, and
shape features. Usage predicts article-type probabilities from full-image
fixed features, then applies a fold-training-only smoothed type-to-usage map.
Neither model uses pretrained weights or protected target rows.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import resource
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from skimage.feature import hog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fashion.config import (
    LABEL_MAPS_JSON,
    RANDOM_SEED,
    ROOT,
    RUNS_CSV,
    SPLITS_CSV,
    TEACHER_TRAIN_IMAGE_DIR,
)
from fashion.data import get_cv_split, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.data.task3_clean_slate_eda import (
    build_clean_slate_audit_contract,
    foreground_proposal,
    foreground_views,
)
from fashion.train.metrics import classification_metrics
from fashion.train.registry import RunRegistry

CLEAN_SLATE_SCREEN_FOLDS = (0, 4)
CLEAN_SLATE_ARTIFACT_ROOT = "experiments/task3_clean_slate_screen_1"
GENDER_EXPERIMENT_ID = "t3_clean_slate_s1_gender_hog_svm"
USAGE_EXPERIMENT_ID = "t3_clean_slate_s1_usage_type_posterior"
GENDER_HYPOTHESIS_ID = "fixed_features_reduce_gender_variance"
USAGE_HYPOTHESIS_ID = "predicted_type_uncertainty_explains_usage"
HOST_MEMORY_LIMIT_BYTES = 7 * 1024**3
SCREEN_SECONDS_PER_FOLD_LIMIT = 90 * 60


@dataclass(frozen=True)
class GenderHogSvmConfig:
    """Frozen first-screen configuration for gender."""

    target: str = "gender"
    feature_view: str = "foreground_masked"
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    inner_folds: int = 4
    calibration: str = "sigmoid"
    class_weight: str = "balanced"
    seed: int = RANDOM_SEED
    scratch: bool = True
    submission_eligible: bool = True


@dataclass(frozen=True)
class UsageTypePosteriorConfig:
    """Frozen first-screen configuration for structured usage prediction."""

    target: str = "usage"
    intermediate_target: str = "articleType"
    feature_view: str = "full"
    alpha_grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3)
    inner_folds: int = 4
    type_usage_smoothing: float = 5.0
    global_prior_pseudocount: float = 0.5
    seed: int = RANDOM_SEED
    scratch: bool = True
    submission_eligible: bool = True


def _log(message: str) -> None:
    print(f"[task3-clean-slate] {message}", flush=True)


def _json_dump(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _pickle_dump(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _stable_digest(payload: Mapping[str, Any]) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": _package_version("pillow"),
        "scikit_image": _package_version("scikit-image"),
        "scikit_learn": _package_version("scikit-learn"),
        "platform": platform.platform(),
        "execution_device": "cpu",
    }


def _peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _classes(label_maps: Mapping[str, Mapping[str, Any]], target: str) -> list[str]:
    classes = [str(value) for value in label_maps[target]["classes"]]
    if len(classes) != len(set(classes)) or not classes:
        raise ValueError(f"invalid fixed class order for {target}")
    return classes


def _valid(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    column = f"has_{target}_label"
    if column not in frame:
        raise ValueError(f"missing {column}")
    return frame.loc[_as_bool(frame[column])].copy()


def _screen_folds(folds: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(value) for value in folds)
    if values != CLEAN_SLATE_SCREEN_FOLDS:
        raise ValueError("the first clean-slate screen must use folds 0 and 4 in that order")
    return values


def _teacher_path(root: Path, relative: str) -> Path:
    teacher_root = (root / TEACHER_TRAIN_IMAGE_DIR.relative_to(ROOT)).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(teacher_root)
    except ValueError as error:
        raise ValueError(
            f"clean-slate features require teacher development images: {relative}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _corrupt(image: Image.Image, corruption: str | None) -> Image.Image:
    if corruption is None:
        return image
    if corruption == "jpeg_75":
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")
    if corruption == "brightness_085":
        return ImageEnhance.Brightness(image).enhance(0.85)
    if corruption == "brightness_115":
        return ImageEnhance.Brightness(image).enhance(1.15)
    if corruption == "translation_003":
        shift_x = max(1, round(image.width * 0.03))
        shift_y = max(1, round(image.height * 0.03))
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, (shift_x, shift_y))
        return canvas
    if corruption == "grayscale":
        return image.convert("L").convert("RGB")
    raise ValueError(f"unknown corruption: {corruption}")


def _colour_shape_features(array: np.ndarray) -> np.ndarray:
    proposal = foreground_proposal(array)
    mask = proposal.mask
    hsv = np.asarray(Image.fromarray(array).convert("HSV"), dtype=np.float32) / 255.0
    foreground = hsv[mask]
    if not len(foreground):
        foreground = hsv.reshape(-1, 3)
    features: list[float] = []
    for channel in range(3):
        counts, _ = np.histogram(foreground[:, channel], bins=8, range=(0.0, 1.0))
        denominator = max(1, int(counts.sum()))
        features.extend((counts / denominator).astype(float))

    row_edges = np.linspace(0, array.shape[0], 4, dtype=int)
    column_edges = np.linspace(0, array.shape[1], 5, dtype=int)
    for row in range(3):
        for column in range(4):
            cell = hsv[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            cell_mask = mask[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            values = cell[cell_mask]
            if not len(values):
                features.extend([0.0] * 6)
            else:
                features.extend(values.mean(axis=0).astype(float))
                features.extend(values.std(axis=0).astype(float))

    top, left, bottom, right = proposal.bbox
    height, width = array.shape[:2]
    box_height = (bottom - top) / height
    box_width = (right - left) / width
    features.extend(
        [
            float(mask.mean()),
            float(proposal.raw_fraction),
            float(box_height),
            float(box_width),
            float(box_height * box_width),
            float(((left + right) / 2) / width),
            float(((top + bottom) / 2) / height),
            float(box_width / max(box_height, 1e-8)),
            float(bool(proposal.fallback_reason)),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def fixed_feature_vector(
    image: Image.Image | np.ndarray,
    *,
    view: str,
    corruption: str | None = None,
) -> np.ndarray:
    """Extract the frozen HOG, foreground-colour, and shape representation."""

    if view not in {"full", "foreground_masked"}:
        raise ValueError("view must be full or foreground_masked")
    opened = (
        ImageOps.exif_transpose(image).convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    )
    opened = _corrupt(opened, corruption).resize((60, 80), Image.Resampling.BILINEAR)
    array = np.asarray(opened, dtype=np.uint8)
    selected = foreground_views(array)[view]
    gradients = hog(
        selected.astype(np.float32) / 255.0,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
        channel_axis=-1,
    ).astype(np.float32)
    return np.concatenate([gradients, _colour_shape_features(array)]).astype(np.float32)


def _feature_contract_valid(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if any(observed.get(key) != value for key, value in expected.items()):
        return False
    for relative, digest in observed.get("artifact_sha256", {}).items():
        artifact = path.parent / relative
        if not artifact.is_file() or compute_sha256(artifact) != digest:
            return False
    return bool(observed.get("artifact_sha256"))


def build_fixed_feature_cache(
    splits: pd.DataFrame,
    *,
    view: str,
    audit_contract_hash: str,
    output_dir: str | Path,
    root: str | Path = ROOT,
    workers: int | None = None,
    local_work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build locally, then publish one teacher-only, label-blind feature matrix."""

    root = Path(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    development = splits.loc[splits["partition"].eq("development")].copy()
    development.sort_values("id", inplace=True)
    development.reset_index(drop=True, inplace=True)
    ids_path = output / f"{view}_ids.csv"
    matrix_path = output / f"{view}_{audit_contract_hash[:16]}.npy"
    contract_path = output / f"{view}_{audit_contract_hash[:16]}.contract.json"
    expected = {
        "artifact": "task3_clean_slate_fixed_features",
        "audit_contract_hash": audit_contract_hash,
        "view": view,
        "rows": int(len(development)),
        "ordered_ids_sha256": hashlib.sha256(
            development["id"].astype("int64").to_numpy().tobytes()
        ).hexdigest(),
        "configuration": "HOG9_cell8_block2_L2Hys_plus_HSV_grid_and_foreground_shape_v1",
    }
    if _feature_contract_valid(contract_path, expected):
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        ids = pd.read_csv(ids_path, keep_default_na=False)
        if (
            matrix.shape[0] == len(development)
            and ids["id"].astype(int).tolist() == development["id"].astype(int).tolist()
        ):
            return {
                "view": view,
                "matrix_path": str(matrix_path),
                "ids_path": str(ids_path),
                "contract_path": str(contract_path),
                "rows": int(matrix.shape[0]),
                "columns": int(matrix.shape[1]),
                "reused": True,
            }

    records = development[["id", "path"]].to_dict("records")

    def extract(record: Mapping[str, Any]) -> np.ndarray:
        path = _teacher_path(root, str(record["path"]))
        with Image.open(path) as source:
            return fixed_feature_vector(source, view=view)

    first = extract(records[0])
    work_dir = (
        Path(local_work_dir)
        if local_work_dir is not None
        else Path(tempfile.gettempdir()) / "fashion-task3-clean-slate"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{view}_{audit_contract_hash[:16]}_",
        suffix=".npy",
        dir=work_dir,
    )
    os.close(descriptor)
    local_temporary = Path(temporary_name)
    durable_staging = matrix_path.with_suffix(".tmp.npy")
    _log(f"building feature view={view} on local disk: {local_temporary}")
    try:
        matrix = np.lib.format.open_memmap(
            local_temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), len(first)),
        )
        try:
            matrix[0] = first
            worker_count = workers or min(12, (os.cpu_count() or 4) * 2)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for position, vector in enumerate(executor.map(extract, records[1:]), start=1):
                    if vector.shape != first.shape:
                        raise ValueError("fixed feature dimensions changed between teacher images")
                    matrix[position] = vector
                    if position % 2500 == 0:
                        _log(f"feature view={view}: {position:,}/{len(records):,} images")
            matrix.flush()
        finally:
            del matrix

        if local_temporary.stat().st_dev == output.stat().st_dev:
            _log(f"moving completed feature view={view} into the local artifact store")
            os.replace(local_temporary, matrix_path)
        else:
            _log(f"copying completed feature view={view} to durable storage")
            shutil.copyfile(local_temporary, durable_staging)
            os.replace(durable_staging, matrix_path)
    finally:
        local_temporary.unlink(missing_ok=True)
        durable_staging.unlink(missing_ok=True)
    write_deterministic_csv(development[["id"]], ids_path, index=False)
    _json_dump(
        {
            **expected,
            "columns": int(len(first)),
            "dtype": "float32",
            "artifact_sha256": {
                matrix_path.name: compute_sha256(matrix_path),
                ids_path.name: compute_sha256(ids_path),
            },
        },
        contract_path,
    )
    return {
        "view": view,
        "matrix_path": str(matrix_path),
        "ids_path": str(ids_path),
        "contract_path": str(contract_path),
        "rows": int(len(records)),
        "columns": int(len(first)),
        "reused": False,
    }


def prepare_clean_slate_screen_features(
    *,
    root: str | Path = ROOT,
    output_root: str | Path,
    workers: int | None = None,
    local_work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the teacher scope once and prepare both separate model inputs."""

    root = Path(root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    audit = build_clean_slate_audit_contract(splits, root=root)
    cache_dir = Path(output_root) / CLEAN_SLATE_ARTIFACT_ROOT / "feature_cache"
    gender = build_fixed_feature_cache(
        splits,
        view="foreground_masked",
        audit_contract_hash=str(audit["audit_contract_hash"]),
        output_dir=cache_dir,
        root=root,
        workers=workers,
        local_work_dir=local_work_dir,
    )
    usage = build_fixed_feature_cache(
        splits,
        view="full",
        audit_contract_hash=str(audit["audit_contract_hash"]),
        output_dir=cache_dir,
        root=root,
        workers=workers,
        local_work_dir=local_work_dir,
    )
    return {"audit_contract": audit, "gender": gender, "usage": usage}


def _matrix_rows(frame: pd.DataFrame, cache: Mapping[str, Any]) -> np.ndarray:
    ids = pd.read_csv(str(cache["ids_path"]), keep_default_na=False)
    lookup = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids["id"].astype(int))
    positions = frame["id"].astype(int).map(lookup)
    if positions.isna().any():
        raise ValueError("a requested row is absent from the fixed feature cache")
    matrix = np.load(str(cache["matrix_path"]), mmap_mode="r", allow_pickle=False)
    return np.asarray(matrix[positions.astype(int).to_numpy()], dtype=np.float32)


def _canonical_inner_splits(
    frame: pd.DataFrame, *, outer_fold: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reuse the four saved non-outer folds for inner model selection."""

    fold_values = pd.to_numeric(frame["cv_fold"], errors="raise").to_numpy(dtype=int)
    expected = sorted(set(range(5)).difference({outer_fold}))
    observed = sorted(np.unique(fold_values).tolist())
    if observed != expected:
        raise ValueError(
            f"outer fold {outer_fold} expected canonical inner folds {expected}, found {observed}"
        )
    groups = frame["product_family_group"].astype(str).to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_fold in expected:
        validation = np.flatnonzero(fold_values == inner_fold)
        training = np.flatnonzero(fold_values != inner_fold)
        if set(groups[training]).intersection(groups[validation]):
            raise ValueError("an inner product family crosses training and validation")
        splits.append((training, validation))
    return splits


def _expand_probabilities(
    probabilities: np.ndarray,
    observed_classes: Sequence[str],
    fixed_classes: Sequence[str],
) -> np.ndarray:
    result = np.zeros((len(probabilities), len(fixed_classes)), dtype=np.float64)
    fixed_lookup = {name: index for index, name in enumerate(fixed_classes)}
    for source, name in enumerate(observed_classes):
        if str(name) not in fixed_lookup:
            raise ValueError(f"model produced an unknown class: {name}")
        result[:, fixed_lookup[str(name)]] = probabilities[:, source]
    row_sums = result.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("expanded probability rows must have positive mass")
    return result / row_sums


def smoothed_type_usage_mapping(
    article_labels: Sequence[str],
    usage_labels: Sequence[str],
    *,
    article_classes: Sequence[str],
    usage_classes: Sequence[str],
    strength: float,
    global_pseudocount: float,
) -> np.ndarray:
    """Fit P(usage|articleType) with a fold-only empirical-Bayes prior."""

    if strength <= 0 or global_pseudocount <= 0:
        raise ValueError("mapping smoothing values must be positive")
    article_lookup = {name: index for index, name in enumerate(article_classes)}
    usage_lookup = {name: index for index, name in enumerate(usage_classes)}
    counts = np.zeros((len(article_classes), len(usage_classes)), dtype=np.float64)
    for article, usage in zip(article_labels, usage_labels, strict=True):
        counts[article_lookup[str(article)], usage_lookup[str(usage)]] += 1.0
    global_counts = counts.sum(axis=0) + global_pseudocount
    global_prior = global_counts / global_counts.sum()
    totals = counts.sum(axis=1, keepdims=True)
    mapping = (counts + strength * global_prior) / (totals + strength)
    if not np.allclose(mapping.sum(axis=1), 1.0):
        raise RuntimeError("type-to-usage probability rows do not sum to one")
    return mapping


def _gender_estimator(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=seed,
        ),
    )


def _usage_type_estimator(alpha: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            average=True,
            n_jobs=-1,
            random_state=seed,
        ),
    )


def _prediction_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    classes: Sequence[str],
    probabilities: np.ndarray,
    run_id: str,
) -> pd.DataFrame:
    label_lookup = {name: index for index, name in enumerate(classes)}
    labels = frame[target].astype(str).map(label_lookup)
    if labels.isna().any():
        raise ValueError(f"unknown {target} label in prediction scope")
    predicted = probabilities.argmax(axis=1)
    result = frame[["id", "cv_fold", "product_family_group", "path"]].copy()
    result["run_id"] = run_id
    result["true_index"] = labels.astype(int).to_numpy()
    result["true_label"] = frame[target].astype(str).to_numpy()
    result["predicted_index"] = predicted
    result["predicted_label"] = [classes[index] for index in predicted]
    result["confidence"] = probabilities.max(axis=1)
    for index, name in enumerate(classes):
        result[f"probability_{index}_{name}"] = probabilities[:, index]
    return result


def _model_parameter_count(model: Any) -> int:
    count = 0
    if isinstance(model, CalibratedClassifierCV):
        for calibrated in model.calibrated_classifiers_:
            estimator = calibrated.estimator[-1]
            count += int(estimator.coef_.size + estimator.intercept_.size)
            count += 2 * len(calibrated.calibrators)
        return count
    estimator = model[-1]
    return int(estimator.coef_.size + estimator.intercept_.size)


def _completed_fold(
    *,
    target_dir: Path,
    fold: int,
    config_hash: str,
    registry_path: Path,
) -> dict[str, Any] | None:
    if not registry_path.is_file():
        return None
    registry = pd.read_csv(registry_path, keep_default_na=False)
    complete_ids = set(registry.loc[registry["status"].eq("complete"), "run_id"].astype(str))
    for metrics_path in sorted(target_dir.glob("*/metrics.json"), reverse=True):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(metrics.get("run_id", ""))
        run_dir = metrics_path.parent
        prediction_path = run_dir / "oof_predictions.csv"
        checkpoint_path = run_dir / "model.pkl"
        if (
            run_id in complete_ids
            and int(metrics.get("validation_fold", -1)) == fold
            and metrics.get("config_hash") == config_hash
            and prediction_path.is_file()
            and checkpoint_path.is_file()
            and metrics.get("prediction_sha256") == compute_sha256(prediction_path)
            and metrics.get("checkpoint_sha256") == compute_sha256(checkpoint_path)
        ):
            _log(f"reusing complete target={metrics['target']} fold={fold}: {run_id}")
            return {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "prediction_path": str(prediction_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
    return None


def _registry_start(
    registry: RunRegistry,
    *,
    run_id: str,
    experiment_id: str,
    hypothesis_id: str,
    target: str,
    fold: int,
    config_hash: str,
    config_path: Path,
    history_path: Path,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    model_family: str,
    root: Path,
) -> None:
    registry.start(
        {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "parent_run_ids": [],
            "task": "task3",
            "target": target,
            "validation_fold": fold,
            "seed": RANDOM_SEED,
            "debug": False,
            "scratch": True,
            "submission_eligible": True,
            "config_hash": config_hash,
            "config_path": _relative(config_path, root),
            "split_digest": compute_sha256(root / SPLITS_CSV.relative_to(ROOT)),
            "label_map_digest": compute_sha256(root / LABEL_MAPS_JSON.relative_to(ROOT)),
            "training_product_count": len(training),
            "validation_product_count": len(validation),
            "training_family_count": training["product_family_group"].nunique(),
            "validation_family_count": validation["product_family_group"].nunique(),
            "model_family": model_family,
            "history_path": _relative(history_path, root),
            "environment_json": _environment(),
            "last_completed_stage": "registered_before_first_model_fit",
        }
    )


def _fold_paths(target_dir: Path, run_id: str) -> dict[str, Path]:
    run_dir = target_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "history": run_dir / "solver_history.csv",
        "checkpoint": run_dir / "model.pkl",
        "predictions": run_dir / "oof_predictions.csv",
        "metrics": run_dir / "metrics.json",
        "mapping": run_dir / "type_usage_mapping.csv",
    }


def _run_gender_fold(
    fold: int,
    *,
    splits: pd.DataFrame,
    label_maps: Mapping[str, Mapping[str, Any]],
    cache: Mapping[str, Any],
    audit_hash: str,
    output_root: Path,
    registry_path: Path,
    registry_mirrors: Sequence[str | Path],
    root: Path,
    reuse_completed: bool,
) -> dict[str, Any]:
    config = GenderHogSvmConfig()
    training_all, validation_all = get_cv_split(splits, fold)
    training = _valid(training_all, "gender").reset_index(drop=True)
    validation = _valid(validation_all, "gender").reset_index(drop=True)
    if set(training["product_family_group"]).intersection(validation["product_family_group"]):
        raise ValueError("an outer product family crosses the gender fold")
    classes = _classes(label_maps, "gender")
    config_payload = {
        **asdict(config),
        "c_grid": list(config.c_grid),
        "validation_fold": fold,
        "audit_contract_hash": audit_hash,
        "screen_folds": list(CLEAN_SLATE_SCREEN_FOLDS),
        "feature_contract": Path(str(cache["contract_path"])).name,
    }
    config_hash = _stable_digest(config_payload)
    target_dir = output_root / CLEAN_SLATE_ARTIFACT_ROOT / "gender"
    if reuse_completed and (
        completed := _completed_fold(
            target_dir=target_dir,
            fold=fold,
            config_hash=config_hash,
            registry_path=registry_path,
        )
    ):
        _log(f"gender fold={fold}: reusing completed run {completed['run_id']}")
        return completed

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"t3_cs1_gender_hogsvm_f{fold}_s{config.seed}_{config_hash[:12]}_"
        f"{timestamp}{uuid.uuid4().hex[:6]}"
    )
    paths = _fold_paths(target_dir, run_id)
    _json_dump(config_payload, paths["config"])
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    _registry_start(
        registry,
        run_id=run_id,
        experiment_id=GENDER_EXPERIMENT_ID,
        hypothesis_id=GENDER_HYPOTHESIS_ID,
        target="gender",
        fold=fold,
        config_hash=config_hash,
        config_path=paths["config"],
        history_path=paths["history"],
        training=training,
        validation=validation,
        model_family="fixed_hog_colour_shape_calibrated_linear_svm",
        root=root,
    )
    started = time.perf_counter()
    last_stage = "registered_before_first_model_fit"
    try:
        x_train = _matrix_rows(training, cache)
        x_validation = _matrix_rows(validation, cache)
        y_train = training["gender"].astype(str).to_numpy()
        inner_splits = _canonical_inner_splits(training, outer_fold=fold)
        if len(inner_splits) != config.inner_folds:
            raise RuntimeError("gender inner-fold count changed")
        history: list[dict[str, Any]] = []
        total_inner_fits = len(config.c_grid) * len(inner_splits)
        completed_inner_fits = 0
        _log(f"gender fold={fold}: starting {total_inner_fits} inner SVM fits")
        for c_value in config.c_grid:
            for inner_fold, (inner_training, inner_validation) in enumerate(inner_splits):
                estimator = _gender_estimator(c_value, config.seed + fold + inner_fold)
                estimator.fit(x_train[inner_training], y_train[inner_training])
                predicted = estimator.predict(x_train[inner_validation]).astype(str)
                inner_macro_f1 = f1_score(
                    y_train[inner_validation],
                    predicted,
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )
                history.append(
                    {
                        "candidate_c": c_value,
                        "inner_fold": inner_fold,
                        "macro_f1": inner_macro_f1,
                    }
                )
                completed_inner_fits += 1
                _log(
                    f"gender fold={fold}: inner fit {completed_inner_fits}/"
                    f"{total_inner_fits}, C={c_value:g}, macro_f1={inner_macro_f1:.4f}"
                )
        history_frame = pd.DataFrame(history)
        means = history_frame.groupby("candidate_c")["macro_f1"].mean()
        best_c = float(sorted(means.index, key=lambda value: (-means[value], value))[0])
        write_deterministic_csv(history_frame, paths["history"], index=False)
        last_stage = "inner_model_selection_complete"
        registry.update(run_id, {"last_completed_stage": last_stage})

        model = CalibratedClassifierCV(
            estimator=_gender_estimator(best_c, config.seed + fold),
            method=config.calibration,
            cv=inner_splits,
            ensemble=True,
        )
        _log(f"gender fold={fold}: selected C={best_c:g}; fitting calibrated outer model")
        model.fit(x_train, y_train)
        validation_probabilities = _expand_probabilities(
            model.predict_proba(x_validation), model.classes_, classes
        )
        training_probabilities = _expand_probabilities(
            model.predict_proba(x_train), model.classes_, classes
        )
        last_stage = "outer_model_fit_complete"
        registry.update(run_id, {"last_completed_stage": last_stage})

        predictions = _prediction_frame(
            validation,
            target="gender",
            classes=classes,
            probabilities=validation_probabilities,
            run_id=run_id,
        )
        write_deterministic_csv(predictions, paths["predictions"], index=False)
        label_lookup = {name: index for index, name in enumerate(classes)}
        validation_labels = validation["gender"].astype(str).map(label_lookup).to_numpy(dtype=int)
        training_labels = training["gender"].astype(str).map(label_lookup).to_numpy(dtype=int)
        metrics = classification_metrics(validation_labels, validation_probabilities, classes)
        train_metrics = classification_metrics(training_labels, training_probabilities, classes)
        parameter_count = _model_parameter_count(model)
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
        metrics.update(
            {
                "run_id": run_id,
                "target": "gender",
                "validation_fold": fold,
                "experiment_id": GENDER_EXPERIMENT_ID,
                "hypothesis_id": GENDER_HYPOTHESIS_ID,
                "model_family": "fixed_hog_colour_shape_calibrated_linear_svm",
                "config_hash": config_hash,
                "audit_contract_hash": audit_hash,
                "best_c": best_c,
                "final_train_macro_f1": train_metrics["macro_f1"],
                "final_train_validation_macro_f1_gap": float(
                    train_metrics["macro_f1"] - metrics["macro_f1"]
                ),
                "parameter_count": parameter_count,
                "train_seconds": train_seconds,
                "peak_memory_bytes": _peak_memory_bytes(),
                "screen_scope": "canonical_outer_folds_0_and_4",
                "robustness_scope": "deferred_to_full_five_fold_stage",
            }
        )
        metrics["prediction_sha256"] = compute_sha256(paths["predictions"])
        metrics["checkpoint_sha256"] = compute_sha256(paths["checkpoint"])
        _json_dump(metrics, paths["metrics"])
        registry.complete(
            run_id,
            {
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
        )
        _log(f"gender fold={fold}: complete, validation_macro_f1={metrics['macro_f1']:.4f}")
        return {
            "run_id": run_id,
            "run_dir": str(paths["run_dir"]),
            "prediction_path": str(paths["predictions"]),
            "metrics_path": str(paths["metrics"]),
            "metrics": metrics,
        }
    except BaseException as error:
        registry.fail(run_id, error, last_completed_stage=last_stage)
        raise


def _run_usage_fold(
    fold: int,
    *,
    splits: pd.DataFrame,
    label_maps: Mapping[str, Mapping[str, Any]],
    cache: Mapping[str, Any],
    audit_hash: str,
    output_root: Path,
    registry_path: Path,
    registry_mirrors: Sequence[str | Path],
    root: Path,
    reuse_completed: bool,
) -> dict[str, Any]:
    config = UsageTypePosteriorConfig()
    training_all, validation_all = get_cv_split(splits, fold)
    training = _valid(_valid(training_all, "usage"), "articleType").reset_index(drop=True)
    validation = _valid(validation_all, "usage")[
        ["id", "cv_fold", "product_family_group", "path", "usage"]
    ].reset_index(drop=True)
    if set(training["product_family_group"]).intersection(validation["product_family_group"]):
        raise ValueError("an outer product family crosses the usage fold")
    usage_classes = _classes(label_maps, "usage")
    article_classes = _classes(label_maps, "articleType")
    config_payload = {
        **asdict(config),
        "alpha_grid": list(config.alpha_grid),
        "validation_fold": fold,
        "audit_contract_hash": audit_hash,
        "screen_folds": list(CLEAN_SLATE_SCREEN_FOLDS),
        "feature_contract": Path(str(cache["contract_path"])).name,
    }
    config_hash = _stable_digest(config_payload)
    target_dir = output_root / CLEAN_SLATE_ARTIFACT_ROOT / "usage"
    if reuse_completed and (
        completed := _completed_fold(
            target_dir=target_dir,
            fold=fold,
            config_hash=config_hash,
            registry_path=registry_path,
        )
    ):
        _log(f"usage fold={fold}: reusing completed run {completed['run_id']}")
        return completed

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"t3_cs1_usage_typepost_f{fold}_s{config.seed}_{config_hash[:12]}_"
        f"{timestamp}{uuid.uuid4().hex[:6]}"
    )
    paths = _fold_paths(target_dir, run_id)
    _json_dump(config_payload, paths["config"])
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    _registry_start(
        registry,
        run_id=run_id,
        experiment_id=USAGE_EXPERIMENT_ID,
        hypothesis_id=USAGE_HYPOTHESIS_ID,
        target="usage",
        fold=fold,
        config_hash=config_hash,
        config_path=paths["config"],
        history_path=paths["history"],
        training=training,
        validation=validation,
        model_family="image_type_posterior_to_smoothed_usage",
        root=root,
    )
    started = time.perf_counter()
    last_stage = "registered_before_first_model_fit"
    try:
        x_train = _matrix_rows(training, cache)
        x_validation = _matrix_rows(validation, cache)
        usage_train = training["usage"].astype(str).to_numpy()
        article_train = training["articleType"].astype(str).to_numpy()
        inner_splits = _canonical_inner_splits(training, outer_fold=fold)
        if len(inner_splits) != config.inner_folds:
            raise RuntimeError("usage inner-fold count changed")
        history: list[dict[str, Any]] = []
        total_inner_fits = len(config.alpha_grid) * len(inner_splits)
        completed_inner_fits = 0
        _log(f"usage fold={fold}: starting {total_inner_fits} inner article-type fits")
        for alpha in config.alpha_grid:
            for inner_fold, (inner_training, inner_validation) in enumerate(inner_splits):
                model = _usage_type_estimator(alpha, config.seed + fold + inner_fold)
                model.fit(x_train[inner_training], article_train[inner_training])
                type_probabilities = _expand_probabilities(
                    model.predict_proba(x_train[inner_validation]),
                    model[-1].classes_,
                    article_classes,
                )
                mapping = smoothed_type_usage_mapping(
                    article_train[inner_training],
                    usage_train[inner_training],
                    article_classes=article_classes,
                    usage_classes=usage_classes,
                    strength=config.type_usage_smoothing,
                    global_pseudocount=config.global_prior_pseudocount,
                )
                usage_probabilities = type_probabilities @ mapping
                predicted = np.asarray(usage_classes)[usage_probabilities.argmax(axis=1)]
                inner_macro_f1 = f1_score(
                    usage_train[inner_validation],
                    predicted,
                    labels=usage_classes,
                    average="macro",
                    zero_division=0,
                )
                history.append(
                    {
                        "candidate_alpha": alpha,
                        "inner_fold": inner_fold,
                        "end_to_end_usage_macro_f1": inner_macro_f1,
                    }
                )
                completed_inner_fits += 1
                _log(
                    f"usage fold={fold}: inner fit {completed_inner_fits}/"
                    f"{total_inner_fits}, alpha={alpha:g}, macro_f1={inner_macro_f1:.4f}"
                )
        history_frame = pd.DataFrame(history)
        means = history_frame.groupby("candidate_alpha")["end_to_end_usage_macro_f1"].mean()
        best_alpha = float(sorted(means.index, key=lambda value: (-means[value], value))[0])
        write_deterministic_csv(history_frame, paths["history"], index=False)
        last_stage = "inner_model_selection_complete"
        registry.update(run_id, {"last_completed_stage": last_stage})

        model = _usage_type_estimator(best_alpha, config.seed + fold)
        _log(f"usage fold={fold}: selected alpha={best_alpha:g}; fitting outer article-type model")
        model.fit(x_train, article_train)
        mapping = smoothed_type_usage_mapping(
            article_train,
            usage_train,
            article_classes=article_classes,
            usage_classes=usage_classes,
            strength=config.type_usage_smoothing,
            global_pseudocount=config.global_prior_pseudocount,
        )
        validation_type_probabilities = _expand_probabilities(
            model.predict_proba(x_validation), model[-1].classes_, article_classes
        )
        training_type_probabilities = _expand_probabilities(
            model.predict_proba(x_train), model[-1].classes_, article_classes
        )
        validation_probabilities = validation_type_probabilities @ mapping
        training_probabilities = training_type_probabilities @ mapping
        last_stage = "outer_model_fit_complete"
        registry.update(run_id, {"last_completed_stage": last_stage})

        predictions = _prediction_frame(
            validation,
            target="usage",
            classes=usage_classes,
            probabilities=validation_probabilities,
            run_id=run_id,
        )
        predictions["predicted_article_type"] = np.asarray(article_classes)[
            validation_type_probabilities.argmax(axis=1)
        ]
        predictions["article_type_confidence"] = validation_type_probabilities.max(axis=1)
        write_deterministic_csv(predictions, paths["predictions"], index=False)
        mapping_frame = pd.DataFrame(mapping, index=article_classes, columns=usage_classes)
        mapping_frame.index.name = "articleType"
        write_deterministic_csv(mapping_frame.reset_index(), paths["mapping"], index=False)
        usage_lookup = {name: index for index, name in enumerate(usage_classes)}
        validation_labels = validation["usage"].astype(str).map(usage_lookup).to_numpy(dtype=int)
        training_labels = training["usage"].astype(str).map(usage_lookup).to_numpy(dtype=int)
        metrics = classification_metrics(validation_labels, validation_probabilities, usage_classes)
        train_metrics = classification_metrics(
            training_labels, training_probabilities, usage_classes
        )
        parameter_count = _model_parameter_count(model)
        _pickle_dump(
            {
                "run_id": run_id,
                "config": config_payload,
                "article_type_classes": article_classes,
                "usage_classes": usage_classes,
                "article_type_model": model,
                "type_usage_mapping": mapping,
                "feature_view": config.feature_view,
            },
            paths["checkpoint"],
        )
        train_seconds = time.perf_counter() - started
        without_home = [row["f1"] for row in metrics["per_class"] if row["class_name"] != "Home"]
        metrics.update(
            {
                "run_id": run_id,
                "target": "usage",
                "validation_fold": fold,
                "experiment_id": USAGE_EXPERIMENT_ID,
                "hypothesis_id": USAGE_HYPOTHESIS_ID,
                "model_family": "image_type_posterior_to_smoothed_usage",
                "config_hash": config_hash,
                "audit_contract_hash": audit_hash,
                "best_alpha": best_alpha,
                "macro_f1_without_home": float(np.mean(without_home)),
                "final_train_macro_f1": train_metrics["macro_f1"],
                "final_train_validation_macro_f1_gap": float(
                    train_metrics["macro_f1"] - metrics["macro_f1"]
                ),
                "parameter_count": parameter_count,
                "train_seconds": train_seconds,
                "peak_memory_bytes": _peak_memory_bytes(),
                "screen_scope": "canonical_outer_folds_0_and_4",
                "robustness_scope": "deferred_to_full_five_fold_stage",
                "mapping_sha256": compute_sha256(paths["mapping"]),
            }
        )
        metrics["prediction_sha256"] = compute_sha256(paths["predictions"])
        metrics["checkpoint_sha256"] = compute_sha256(paths["checkpoint"])
        _json_dump(metrics, paths["metrics"])
        registry.complete(
            run_id,
            {
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
        )
        _log(f"usage fold={fold}: complete, validation_macro_f1={metrics['macro_f1']:.4f}")
        return {
            "run_id": run_id,
            "run_dir": str(paths["run_dir"]),
            "prediction_path": str(paths["predictions"]),
            "metrics_path": str(paths["metrics"]),
            "metrics": metrics,
        }
    except BaseException as error:
        registry.fail(run_id, error, last_completed_stage=last_stage)
        raise


def _aggregate_screen(
    target: str,
    fold_results: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    root: Path,
    model_family: str,
    experiment_id: str,
    hypothesis_id: str,
    anchor_prediction_path: str | Path | None,
    artifact_root: str = CLEAN_SLATE_ARTIFACT_ROOT,
    seconds_per_fold_limit: float = SCREEN_SECONDS_PER_FOLD_LIMIT,
    memory_limit_bytes: int = HOST_MEMORY_LIMIT_BYTES,
) -> dict[str, Any]:
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    classes = _classes(label_maps, target)
    predictions = pd.concat(
        [
            pd.read_csv(str(result["prediction_path"]), keep_default_na=False)
            for result in fold_results
        ],
        ignore_index=True,
    )
    if predictions["id"].duplicated().any():
        raise ValueError("screen OOF predictions contain duplicate IDs")
    if tuple(sorted(predictions["cv_fold"].astype(int).unique())) != CLEAN_SLATE_SCREEN_FOLDS:
        raise ValueError("screen OOF predictions must contain only folds 0 and 4")
    probability_columns = [f"probability_{index}_{name}" for index, name in enumerate(classes)]
    probabilities = predictions[probability_columns].to_numpy(dtype=np.float64)
    labels = predictions["true_index"].to_numpy(dtype=np.int64)
    metrics = classification_metrics(labels, probabilities, classes)
    fold_scores = [float(result["metrics"]["macro_f1"]) for result in fold_results]
    metrics.update(
        {
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "model_family": model_family,
            "fold_run_ids": [str(result["run_id"]) for result in fold_results],
            "validation_folds": list(CLEAN_SLATE_SCREEN_FOLDS),
            "fold_macro_f1": fold_scores,
            "fold_macro_f1_sample_sd": float(np.std(fold_scores, ddof=1)),
            "screen_only": True,
            "not_a_five_fold_result": True,
        }
    )
    if target == "usage":
        metrics["macro_f1_without_home"] = float(
            np.mean([row["f1"] for row in metrics["per_class"] if row["class_name"] != "Home"])
        )

    aggregate_dir = output_root / artifact_root / target / "aggregate_folds_0_4"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = aggregate_dir / "oof_predictions.csv"
    metrics_path = aggregate_dir / "metrics.json"
    per_class_path = aggregate_dir / "per_class.csv"
    confusion_path = aggregate_dir / "confusion_matrix.csv"
    gate_path = aggregate_dir / "screen_gate.json"
    write_deterministic_csv(predictions.sort_values("id"), predictions_path, index=False)
    write_deterministic_csv(pd.DataFrame(metrics["per_class"]), per_class_path, index=False)
    write_deterministic_csv(
        pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
        .rename_axis("true_label")
        .reset_index(),
        confusion_path,
        index=False,
    )

    anchor_path = Path(anchor_prediction_path) if anchor_prediction_path else None
    gate: dict[str, Any] = {
        "status": "not_evaluated_anchor_missing",
        "training_scope_integrity": True,
        "resource_limit_pass": all(
            float(result["metrics"]["train_seconds"]) <= seconds_per_fold_limit
            and int(result["metrics"]["peak_memory_bytes"]) <= memory_limit_bytes
            for result in fold_results
        ),
        "probability_integrity": bool(
            np.isfinite(probabilities).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
            and predictions["predicted_label"].nunique() > 1
        ),
    }
    if anchor_path is not None and anchor_path.is_file():
        anchor = pd.read_csv(anchor_path, keep_default_na=False)
        anchor = predictions[["id", "true_index", "true_label"]].merge(
            anchor,
            on="id",
            how="left",
            validate="one_to_one",
            suffixes=("_candidate", "_anchor"),
        )
        if anchor[probability_columns].isna().any().any():
            raise ValueError("matched historical anchor is missing screen IDs")

        candidate_indices = pd.to_numeric(anchor["true_index_candidate"], errors="coerce")
        anchor_indices = pd.to_numeric(anchor["true_index_anchor"], errors="coerce")
        valid_indices = (
            candidate_indices.notna()
            & anchor_indices.notna()
            & candidate_indices.eq(candidate_indices.round())
            & anchor_indices.eq(anchor_indices.round())
            & candidate_indices.between(0, len(classes) - 1)
            & anchor_indices.between(0, len(classes) - 1)
        )
        if not valid_indices.all():
            raise ValueError("historical anchor contains an invalid numeric label index")
        candidate_indices = candidate_indices.astype(np.int64)
        anchor_indices = anchor_indices.astype(np.int64)
        if not candidate_indices.eq(anchor_indices).all():
            raise ValueError("historical anchor labels disagree with the screen labels")

        canonical_labels = candidate_indices.map(dict(enumerate(classes)))
        candidate_labels = anchor["true_label_candidate"].astype(str)
        anchor_labels = anchor["true_label_anchor"].astype(str)
        if not candidate_labels.eq(canonical_labels).all():
            raise ValueError("screen labels disagree with their fixed numeric indices")
        legacy_na_labels = anchor_labels.eq("") & canonical_labels.eq("NA")
        anchor_label_matches = anchor_labels.eq(canonical_labels) | legacy_na_labels
        if not anchor_label_matches.all():
            raise ValueError("historical anchor labels disagree with the screen labels")
        anchor.loc[legacy_na_labels, "true_label_anchor"] = "NA"

        anchor_metrics = classification_metrics(
            candidate_indices.to_numpy(dtype=np.int64),
            anchor[probability_columns].to_numpy(dtype=np.float64),
            classes,
        )
        candidate_by_class = {row["class_name"]: row for row in metrics["per_class"]}
        anchor_by_class = {row["class_name"]: row for row in anchor_metrics["per_class"]}
        class_deltas = pd.DataFrame(
            [
                {
                    "class_name": name,
                    "support": candidate_by_class[name]["support"],
                    "candidate_f1": candidate_by_class[name]["f1"],
                    "anchor_f1": anchor_by_class[name]["f1"],
                    "f1_delta": candidate_by_class[name]["f1"] - anchor_by_class[name]["f1"],
                }
                for name in classes
            ]
        )
        write_deterministic_csv(
            class_deltas, aggregate_dir / "matched_anchor_per_class.csv", index=False
        )
        _json_dump(anchor_metrics, aggregate_dir / "matched_anchor_metrics.json")
        score_pass = float(metrics["macro_f1"]) >= float(anchor_metrics["macro_f1"]) - 0.020
        class_pass = bool(
            class_deltas.loc[class_deltas["support"].gt(0), "f1_delta"].ge(-0.030).all()
        )
        gate.update(
            {
                "status": "pass"
                if all(
                    (
                        score_pass,
                        class_pass,
                        gate["resource_limit_pass"],
                        gate["probability_integrity"],
                    )
                )
                else "fail",
                "candidate_macro_f1": metrics["macro_f1"],
                "matched_anchor_macro_f1": anchor_metrics["macro_f1"],
                "macro_f1_delta": float(metrics["macro_f1"] - anchor_metrics["macro_f1"]),
                "score_noninferiority_pass": score_pass,
                "supported_class_noninferiority_pass": class_pass,
                "anchor_label_integrity": "canonical_index_verified",
                "anchor_literal_na_label_repairs": int(legacy_na_labels.sum()),
            }
        )
    _json_dump(gate, gate_path)
    metrics["screen_gate"] = gate
    _json_dump(metrics, metrics_path)
    return {
        "target": target,
        "fold_run_ids": metrics["fold_run_ids"],
        "prediction_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "per_class_path": str(per_class_path),
        "confusion_path": str(confusion_path),
        "screen_gate_path": str(gate_path),
        "metrics": metrics,
    }


def run_clean_slate_gender_screen(
    *,
    prepared_features: Mapping[str, Any],
    output_root: str | Path,
    folds: Iterable[int] = CLEAN_SLATE_SCREEN_FOLDS,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    anchor_prediction_path: str | Path | None = None,
    reuse_completed: bool = True,
) -> dict[str, Any]:
    """Run the folds 0/4 gender HOG-SVM screen and pool its matched OOF rows."""

    fold_list = _screen_folds(folds)
    root = Path(root)
    output_root = Path(output_root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    audit_hash = str(prepared_features["audit_contract"]["audit_contract_hash"])
    results = [
        _run_gender_fold(
            fold,
            splits=splits,
            label_maps=label_maps,
            cache=prepared_features["gender"],
            audit_hash=audit_hash,
            output_root=output_root,
            registry_path=Path(registry_path),
            registry_mirrors=registry_mirrors,
            root=root,
            reuse_completed=reuse_completed,
        )
        for fold in fold_list
    ]
    return _aggregate_screen(
        "gender",
        results,
        output_root=output_root,
        root=root,
        model_family="fixed_hog_colour_shape_calibrated_linear_svm",
        experiment_id=GENDER_EXPERIMENT_ID,
        hypothesis_id=GENDER_HYPOTHESIS_ID,
        anchor_prediction_path=anchor_prediction_path,
    )


def run_clean_slate_usage_screen(
    *,
    prepared_features: Mapping[str, Any],
    output_root: str | Path,
    folds: Iterable[int] = CLEAN_SLATE_SCREEN_FOLDS,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    anchor_prediction_path: str | Path | None = None,
    reuse_completed: bool = True,
) -> dict[str, Any]:
    """Run the folds 0/4 structured usage screen and pool its matched OOF rows."""

    fold_list = _screen_folds(folds)
    root = Path(root)
    output_root = Path(output_root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    audit_hash = str(prepared_features["audit_contract"]["audit_contract_hash"])
    results = [
        _run_usage_fold(
            fold,
            splits=splits,
            label_maps=label_maps,
            cache=prepared_features["usage"],
            audit_hash=audit_hash,
            output_root=output_root,
            registry_path=Path(registry_path),
            registry_mirrors=registry_mirrors,
            root=root,
            reuse_completed=reuse_completed,
        )
        for fold in fold_list
    ]
    return _aggregate_screen(
        "usage",
        results,
        output_root=output_root,
        root=root,
        model_family="image_type_posterior_to_smoothed_usage",
        experiment_id=USAGE_EXPERIMENT_ID,
        hypothesis_id=USAGE_HYPOTHESIS_ID,
        anchor_prediction_path=anchor_prediction_path,
    )


def check_clean_slate_screen_setup(
    *,
    root: str | Path = ROOT,
    folds: Iterable[int] = CLEAN_SLATE_SCREEN_FOLDS,
) -> dict[str, Any]:
    """Perform a zero-fit preflight for the separate clean-slate models."""

    fold_list = _screen_folds(folds)
    root = Path(root)
    splits_path = root / SPLITS_CSV.relative_to(ROOT)
    label_maps_path = root / LABEL_MAPS_JSON.relative_to(ROOT)
    if not splits_path.is_file() or not label_maps_path.is_file():
        raise FileNotFoundError("processed split or label map is missing")
    splits = load_splits(splits_path)
    label_maps = load_label_maps(label_maps_path)
    fold_rows: list[dict[str, Any]] = []
    for fold in fold_list:
        training, validation = get_cv_split(splits, fold)
        for target in ("gender", "usage"):
            target_training = _valid(training, target)
            target_validation = _valid(validation, target)
            crossing = set(target_training["product_family_group"]).intersection(
                target_validation["product_family_group"]
            )
            if crossing:
                raise ValueError(f"a {target} family crosses fold {fold}")
            fold_rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "training_rows": len(target_training),
                    "validation_rows": len(target_validation),
                }
            )
    dummy = np.full((80, 60, 3), 255, dtype=np.uint8)
    feature_columns = len(fixed_feature_vector(dummy, view="full"))
    development_rows = int(splits["partition"].eq("development").sum())
    estimated_cache_bytes = development_rows * feature_columns * np.dtype(np.float32).itemsize * 2
    if estimated_cache_bytes > HOST_MEMORY_LIMIT_BYTES:
        raise RuntimeError("estimated fixed-feature caches exceed the 7 GiB screen limit")
    return {
        "training_screen_ready": True,
        "training_blockers": [],
        "human_observability_review_status": "deferred_non_blocking",
        "folds": list(fold_list),
        "fold_rows": fold_rows,
        "gender_model": "foreground HOG + colour/shape + calibrated linear SVM",
        "usage_model": "image article-type posterior + fold-only smoothed type-to-usage map",
        "gender_classes": _classes(label_maps, "gender"),
        "usage_classes": _classes(label_maps, "usage"),
        "article_type_classes": len(_classes(label_maps, "articleType")),
        "feature_columns_per_view": feature_columns,
        "estimated_two_view_cache_bytes": estimated_cache_bytes,
        "host_memory_limit_bytes": HOST_MEMORY_LIMIT_BYTES,
        "model_fits": 0,
        "optimizer_steps": 0,
        "execution_device": "cpu",
    }
