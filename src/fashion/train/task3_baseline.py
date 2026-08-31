"""Check and execute the Task 3 primary learnable baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fashion.config import LABEL_MAPS_JSON, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data import load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import (
    Task3BaselineConfig,
    baseline_parameter_count,
    compact_blur_cnn_macs,
    compact_blur_cnn_parameter_count,
    config_digest,
    tinyconvnext18_macs,
    tinyconvnext18_parameter_count,
    tinyhrnet20_macs,
    tinyhrnet20_parameter_count,
    tinyresnet18_pm_macs,
    tinyresnet18_pm_parameter_count,
)
from fashion.train.data import (
    CORE_CORRUPTIONS,
    Task3ImageDataset,
    fit_fold_rgb_stats,
    task3_target_frames,
)
from fashion.train.loss import WeightedFocalCrossEntropy, WeightedLabelSmoothedCrossEntropy
from fashion.train.metrics import classification_metrics
from fashion.train.model import (
    Task3BaselineCNN,
    Task3CompactBlurCNN,
    Task3GeM3CNN,
    Task3TinyConvNeXt18,
    Task3TinyHRNet20,
    Task3TinyResNet18PM,
)
from fashion.train.registry import RunRegistry
from fashion.train.task3_experiments import Task3ChildSpec, effective_number_class_weights

BASELINE_EXPERIMENT_ID = "t3_primary_baseline_smallcnn"
VERIFIED_COLAB_RUNTIME = {
    "python_major_minor": "3.13",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "numpy": "2.1.3",
    "pandas": "2.2.3",
    "pillow": "11.3.0",
}


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _log(message: str) -> None:
    print(f"[task3] {message}", flush=True)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_environment(device: torch.device) -> dict[str, object]:
    """Capture the runtime fields needed to interpret cost and reproducibility."""
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": _package_version("Pillow"),
        "sklearn": _package_version("scikit-learn"),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "gpu": gpu_name,
        "cuda_runtime": torch.version.cuda or "",
        "vram_bytes": int(properties.total_memory) if properties is not None else 0,
    }


def validate_verified_colab_runtime(environment: dict[str, object]) -> None:
    """Fail before training when the managed Colab stack differs from the tested stack."""
    actual = {
        "python_major_minor": ".".join(str(environment["python"]).split(".")[:2]),
        "torch": str(environment["torch"]).split("+")[0],
        "torchvision": str(environment["torchvision"]).split("+")[0],
        "numpy": str(environment["numpy"]),
        "pandas": str(environment["pandas"]),
        "pillow": str(environment["pillow"]),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in VERIFIED_COLAB_RUNTIME.items()
        if actual[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "Colab runtime differs from requirements/colab-task3-runtime.txt: "
            + json.dumps(mismatches, sort_keys=True)
        )


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _loader(
    dataset: Task3ImageDataset,
    *,
    config: Task3BaselineConfig,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def _class_spec(
    label_maps: dict[str, dict[str, object]], target: str
) -> tuple[list[str], dict[str, int]]:
    if target not in label_maps:
        raise ValueError(f"label maps do not contain target {target}")
    spec = label_maps[target]
    classes = [str(label) for label in spec["classes"]]
    mapping = {str(label): int(index) for label, index in dict(spec["label_to_index"]).items()}
    if mapping != {label: index for index, label in enumerate(classes)}:
        raise ValueError(f"{target} label map is not a stable contiguous ordering")
    return classes, mapping


def check_task3_baseline_setup(
    target: str,
    *,
    root: str | Path = ROOT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Validate the complete baseline contract without an optimiser step."""
    root = Path(root)
    config = Task3BaselineConfig(target=target)  # type: ignore[arg-type]
    splits_path = root / SPLITS_CSV.relative_to(ROOT)
    label_maps_path = root / LABEL_MAPS_JSON.relative_to(ROOT)
    splits = load_splits(splits_path)
    label_maps = load_label_maps(label_maps_path)
    classes, mapping = _class_spec(label_maps, target)
    if len(classes) != config.num_classes:
        raise ValueError("model head and fixed label map disagree")
    if set(pd.to_numeric(splits.loc[splits["partition"].eq("development"), "cv_fold"])) != set(
        range(5)
    ):
        raise ValueError("the development data does not contain all five canonical folds")

    fold_counts: dict[str, dict[str, int]] = {}
    required_paths: set[str] = set()
    for fold in range(5):
        training, validation = task3_target_frames(splits, target=target, validation_fold=fold)
        fold_counts[str(fold)] = {"training": len(training), "validation": len(validation)}
        required_paths.update(str(path) for path in validation["path"])
    missing_paths = sorted(path for path in required_paths if not (root / path).is_file())
    if missing_paths:
        preview = missing_paths[:10]
        raise FileNotFoundError(
            f"{len(missing_paths)} development images are missing; first paths: {preview}"
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but Colab has no active GPU runtime")
    device = torch.device(device_name)
    set_reproducible_seed(config.seed)
    model = Task3BaselineCNN(config).to(device)
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, config.image_height, config.image_width, device=device))
    if tuple(output.shape) != (2, config.num_classes):
        raise RuntimeError(f"unexpected baseline output shape: {tuple(output.shape)}")
    environment = runtime_environment(device)
    validate_verified_colab_runtime(environment)
    return {
        "target": target,
        "classes": classes,
        "label_to_index": mapping,
        "fold_counts": fold_counts,
        "checked_image_paths": len(required_paths),
        "parameter_count": baseline_parameter_count(config.target),
        "config_hash": config_digest(config),
        "device": str(device),
        "environment": environment,
        "optimizer_steps": 0,
        "ready": True,
    }


def _pass(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, list[Any]]]:
    training = optimizer is not None
    model.train(training)
    total_loss_numerator = 0.0
    total_loss_denominator = 0.0
    total_rows = 0
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    trace: dict[str, list[Any]] = {
        "id": [],
        "cv_fold": [],
        "product_family_group": [],
        "path": [],
    }
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                optimizer.step()
            rows = len(target)
            total_rows += rows
            denominator_method = getattr(criterion, "loss_denominator", None)
            if callable(denominator_method):
                batch_loss_denominator = float(
                    denominator_method(target).detach().cpu()
                )
            elif isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
                batch_loss_denominator = float(
                    criterion.weight[target].detach().sum().cpu()
                )
            else:
                batch_loss_denominator = float(rows)
            total_loss_numerator += float(loss.detach()) * batch_loss_denominator
            total_loss_denominator += batch_loss_denominator
            labels.append(target.detach().cpu().numpy())
            probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
            if not training:
                trace["id"].extend(batch["id"].tolist())
                trace["cv_fold"].extend(batch["cv_fold"].tolist())
                trace["product_family_group"].extend(batch["product_family_group"])
                trace["path"].extend(batch["path"])
    if total_rows == 0 or total_loss_denominator == 0.0:
        raise ValueError("a data loader produced no rows")
    return (
        total_loss_numerator / total_loss_denominator,
        np.concatenate(labels),
        np.concatenate(probabilities),
        trace,
    )


def _prediction_frame(
    labels: np.ndarray,
    probabilities: np.ndarray,
    trace: dict[str, list[Any]],
    classes: Sequence[str],
    run_id: str,
) -> pd.DataFrame:
    predicted = probabilities.argmax(axis=1)
    frame = pd.DataFrame(trace)
    frame["run_id"] = run_id
    frame["true_index"] = labels
    frame["true_label"] = [classes[index] for index in labels]
    frame["predicted_index"] = predicted
    frame["predicted_label"] = [classes[index] for index in predicted]
    frame["confidence"] = probabilities.max(axis=1)
    for index, class_name in enumerate(classes):
        frame[f"probability_{index}_{class_name}"] = probabilities[:, index]
    return frame


def _failure_index(predictions: pd.DataFrame, class_names: Sequence[str]) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    errors = predictions[predictions["true_index"] != predictions["predicted_index"]]
    correct = predictions[predictions["true_index"] == predictions["predicted_index"]]
    if not errors.empty:
        top_errors = errors.nlargest(40, "confidence").copy()
        top_errors["selection_reason"] = "high_confidence_error"
        selected.append(top_errors)
    if not correct.empty:
        top_correct = correct.nlargest(20, "confidence").copy()
        top_correct["selection_reason"] = "high_confidence_correct"
        selected.append(top_correct)
    for index, class_name in enumerate(class_names):
        class_errors = errors[errors["true_index"].eq(index)].nlargest(10, "confidence").copy()
        if not class_errors.empty:
            class_errors["selection_reason"] = f"class_error:{class_name}"
            selected.append(class_errors)
    if not selected:
        return predictions.head(0).assign(selection_reason=pd.Series(dtype=str))
    result = pd.concat(selected, ignore_index=True)
    return result.drop_duplicates(subset=["id", "selection_reason"])


def _measure_latency(
    model: nn.Module,
    device: torch.device,
    *,
    height: int,
    width: int,
    repetitions: int = 100,
) -> float:
    model.eval()
    sample = torch.zeros(1, 3, height, width, device=device)
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / repetitions


def _task3_model_contract(
    config: Task3BaselineConfig, child_spec: Task3ChildSpec | None
) -> tuple[str, str, int, int | None]:
    model_family = child_spec.model_family if child_spec is not None else config.model_family
    run_model_token = child_spec.run_model_token if child_spec is not None else "smallcnn"
    if model_family == "task3_tinyresnet18_pm":
        return (
            model_family,
            run_model_token,
            tinyresnet18_pm_parameter_count(config.target),
            tinyresnet18_pm_macs(config.target),
        )
    if model_family == "task3_compact_blur_cnn":
        return (
            model_family,
            run_model_token,
            compact_blur_cnn_parameter_count(config.target),
            compact_blur_cnn_macs(config.target),
        )
    if model_family == "task3_tinyhrnet20":
        return (
            model_family,
            run_model_token,
            tinyhrnet20_parameter_count(config.target),
            tinyhrnet20_macs(config.target),
        )
    if model_family == "task3_tinyconvnext18":
        return (
            model_family,
            run_model_token,
            tinyconvnext18_parameter_count(config.target),
            tinyconvnext18_macs(config.target),
        )
    if model_family == "task3_small_cnn_gem_p3":
        return model_family, run_model_token, baseline_parameter_count(config.target), None
    if model_family != "task3_small_cnn":
        raise ValueError(f"unsupported Task 3 model family: {model_family}")
    return model_family, run_model_token, baseline_parameter_count(config.target), None


def _build_task3_model(
    config: Task3BaselineConfig,
    child_spec: Task3ChildSpec | None,
) -> nn.Module:
    model_family, _, _, _ = _task3_model_contract(config, child_spec)
    if model_family == "task3_tinyresnet18_pm":
        return Task3TinyResNet18PM(config)
    if model_family == "task3_compact_blur_cnn":
        return Task3CompactBlurCNN(config)
    if model_family == "task3_tinyhrnet20":
        return Task3TinyHRNet20(config)
    if model_family == "task3_tinyconvnext18":
        return Task3TinyConvNeXt18(config)
    if model_family == "task3_small_cnn_gem_p3":
        return Task3GeM3CNN(config)
    classifier_dropout = child_spec.classifier_dropout if child_spec is not None else 0.0
    return Task3BaselineCNN(config, classifier_dropout=classifier_dropout)


def run_task3_baseline_fold(
    target: str,
    validation_fold: int,
    *,
    output_root: str | Path,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
    child_spec: Task3ChildSpec | None = None,
) -> dict[str, object]:
    """Train one baseline fold or one locked single-factor child fold."""
    root = Path(root)
    output_root = Path(output_root)
    config = Task3BaselineConfig(target=target)  # type: ignore[arg-type]
    if validation_fold not in range(5):
        raise ValueError("validation_fold must be one of 0,1,2,3,4")
    if child_spec is not None and child_spec.target != target:
        raise ValueError("child target and requested target disagree")
    model_family, run_model_token, parameter_count, architecture_macs = (
        _task3_model_contract(config, child_spec)
    )
    if child_spec is not None:
        parent_run_id = child_spec.parent_run_ids[validation_fold]
        parent_dir = output_root / child_spec.parent_artifact_dir / target / parent_run_id
        parent_metrics_path = parent_dir / "metrics.json"
        parent_checkpoint_path = parent_dir / "final_epoch.pt"
        parent_prediction_path = parent_dir / "oof_predictions.csv"
        parent_robustness_path = parent_dir / "robustness.csv"
        if not all(
            path.is_file()
            for path in (
                parent_metrics_path,
                parent_checkpoint_path,
                parent_prediction_path,
                parent_robustness_path,
            )
        ):
            raise FileNotFoundError(
                f"completed parent artifacts are missing for fold {validation_fold}: "
                f"{parent_dir}"
            )
        parent_metrics = json.loads(parent_metrics_path.read_text(encoding="utf-8"))
        if (
            str(parent_metrics.get("run_id")) != parent_run_id
            or int(parent_metrics.get("validation_fold", -1)) != validation_fold
        ):
            raise ValueError(f"parent metadata disagrees for fold {validation_fold}")
    set_reproducible_seed(config.seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    splits_path = root / SPLITS_CSV.relative_to(ROOT)
    label_maps_path = root / LABEL_MAPS_JSON.relative_to(ROOT)
    splits = load_splits(splits_path)
    label_maps = load_label_maps(label_maps_path)
    classes, label_to_index = _class_spec(label_maps, target)
    training, validation = task3_target_frames(
        splits, target=target, validation_fold=validation_fold
    )
    _log(
        f"preparing target={target} fold={validation_fold}: "
        f"train={len(training):,}, validation={len(validation):,}"
    )

    class_counts = np.array(
        [int(training[target].eq(class_name).sum()) for class_name in classes],
        dtype=np.int64,
    )
    class_weights: np.ndarray | None = None
    if child_spec is not None and child_spec.loss_name in {
        "effective_number_cross_entropy",
        "effective_number_label_smoothed_cross_entropy",
        "effective_number_focal_cross_entropy",
    }:
        if child_spec.class_weight_beta is None or child_spec.class_weight_cap is None:
            raise ValueError("class-balanced loss requires beta and cap")
        class_weights = effective_number_class_weights(
            class_counts,
            beta=child_spec.class_weight_beta,
            cap=child_spec.class_weight_cap,
        )

    config_payload = config.to_dict()
    config_payload["effective_model_family"] = model_family
    config_payload["parameter_count"] = parameter_count
    config_payload["architecture_macs"] = architecture_macs
    if child_spec is None:
        digest = config_digest(config)
        experiment_id = BASELINE_EXPERIMENT_ID
        hypothesis_id = "baseline_contract"
        parent_run_ids: list[str] = []
        artifact_dir = "baseline"
        run_prefix = "t3_baseline"
        training_augmentation = "none"
        loss_name = "cross_entropy"
    else:
        experiment_id = child_spec.experiment_id
        hypothesis_id = child_spec.hypothesis_id
        parent_run_ids = [child_spec.parent_run_ids[validation_fold]]
        artifact_dir = child_spec.artifact_dir
        run_prefix = child_spec.run_prefix
        training_augmentation = child_spec.training_augmentation
        loss_name = child_spec.loss_name
        config_payload["child_experiment"] = child_spec.to_dict()
        digest_payload = {
            "baseline_controls": config.to_dict(),
            "child_experiment": child_spec.to_dict(),
        }
        encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
        config_payload["parent_run_id"] = parent_run_ids[0]
        config_payload["class_counts"] = class_counts.tolist()
        config_payload["class_weights"] = (
            class_weights.tolist() if class_weights is not None else None
        )

    execution = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + uuid.uuid4().hex[:6]
    run_id = (
        f"{run_prefix}_{target}_{run_model_token}_f{validation_fold}_s{config.seed}_"
        f"{digest}_{execution}"
    )
    run_dir = output_root / artifact_dir / target / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.json"
    normalization_path = run_dir / "normalization.json"
    history_path = run_dir / "history.csv"
    prediction_path = run_dir / "oof_predictions.csv"
    metrics_path = run_dir / "metrics.json"
    robustness_path = run_dir / "robustness.csv"
    checkpoint_path = run_dir / "final_epoch.pt"

    _json_dump(config_payload, config_path)
    _log(f"fitting fold-training RGB statistics for target={target} fold={validation_fold}")
    stats = fit_fold_rgb_stats(
        training,
        root=root,
        image_size=(config.image_height, config.image_width),
    )
    _json_dump(
        {
            **stats,
            "fit_scope": "fold_training_content_pixels_only",
            "validation_fold": validation_fold,
            "padding_excluded": True,
        },
        normalization_path,
    )
    _log(f"RGB statistics ready for target={target} fold={validation_fold}")

    dataset_kwargs = {
        "target": target,
        "label_to_index": label_to_index,
        "mean": stats["mean"],
        "std": stats["std"],
        "root": root,
        "image_size": (config.image_height, config.image_width),
    }
    train_dataset = Task3ImageDataset(
        training,
        augmentation=training_augmentation,
        **dataset_kwargs,
    )
    validation_dataset = Task3ImageDataset(validation, **dataset_kwargs)
    train_loader = _loader(train_dataset, config=config, shuffle=True, device=device)
    validation_loader = _loader(validation_dataset, config=config, shuffle=False, device=device)

    classifier_dropout = child_spec.classifier_dropout if child_spec is not None else 0.0
    model = _build_task3_model(config, child_spec).to(device)
    class_weight_tensor = (
        torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    if child_spec is not None and child_spec.focal_gamma > 0.0:
        if class_weight_tensor is None:
            raise ValueError("weighted focal loss requires fold-only class weights")
        criterion: nn.Module = WeightedFocalCrossEntropy(
            class_weight_tensor,
            gamma=child_spec.focal_gamma,
        )
    elif child_spec is not None and child_spec.label_smoothing > 0.0:
        if class_weight_tensor is None:
            raise ValueError("weighted label smoothing requires fold-only class weights")
        criterion: nn.Module = WeightedLabelSmoothedCrossEntropy(
            class_weight_tensor,
            epsilon=child_spec.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.minimum_learning_rate
    )
    environment = runtime_environment(device)
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    split_digest = compute_sha256(splits_path)
    label_map_digest = compute_sha256(label_maps_path)
    registry.start(
        {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "parent_run_ids": parent_run_ids,
            "task": "task3",
            "target": target,
            "validation_fold": validation_fold,
            "seed": config.seed,
            "debug": False,
            "scratch": True,
            "submission_eligible": True,
            "config_hash": digest,
            "config_path": config_path,
            "split_digest": split_digest,
            "label_map_digest": label_map_digest,
            "training_product_count": len(training),
            "validation_product_count": len(validation),
            "training_family_count": training["product_family_group"].nunique(),
            "validation_family_count": validation["product_family_group"].nunique(),
            "model_family": model_family,
            "parameter_count": parameter_count,
            "history_path": history_path,
            "environment_json": environment,
            "last_completed_stage": "registered_before_first_optimizer_step",
        }
    )
    _log(f"registered {run_id}; the first optimiser step may now run")

    history: list[dict[str, object]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_stage = "registered_before_first_optimizer_step"
    try:
        for epoch in range(1, config.epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_loss, train_labels, train_probabilities, _ = _pass(
                model, train_loader, criterion, device, optimizer=optimizer
            )
            validation_loss, validation_labels, validation_probabilities, _ = _pass(
                model, validation_loader, criterion, device
            )
            train_metrics = classification_metrics(train_labels, train_probabilities, classes)
            validation_metrics = classification_metrics(
                validation_labels, validation_probabilities, classes
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "train_loss": train_loss,
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_loss": validation_loss,
                    "validation_macro_f1": validation_metrics["macro_f1"],
                }
            )
            pd.DataFrame(history).to_csv(history_path, index=False)
            scheduler.step()
            last_stage = f"epoch_{epoch}_complete"
            registry.update(run_id, {"last_completed_stage": last_stage})
            _log(
                f"target={target} fold={validation_fold} epoch={epoch}/{config.epochs} "
                f"train_loss={train_loss:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} "
                f"validation_loss={validation_loss:.4f} "
                f"validation_macro_f1={validation_metrics['macro_f1']:.4f}"
            )

        training_finished = time.perf_counter()

        torch.save(
            {
                "run_id": run_id,
                "config": config_payload,
                "class_names": classes,
                "normalization": stats,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        clean_loss, labels, probabilities, trace = _pass(
            model, validation_loader, criterion, device
        )
        training_evaluation_dataset = Task3ImageDataset(training, **dataset_kwargs)
        training_evaluation_loader = _loader(
            training_evaluation_dataset,
            config=config,
            shuffle=False,
            device=device,
        )
        final_train_loss, final_train_labels, final_train_probabilities, _ = _pass(
            model, training_evaluation_loader, criterion, device
        )
        final_train_metrics = classification_metrics(
            final_train_labels, final_train_probabilities, classes
        )
        predictions = _prediction_frame(labels, probabilities, trace, classes, run_id)
        predictions.to_csv(prediction_path, index=False)
        metrics = classification_metrics(labels, probabilities, classes)
        metrics["loss"] = clean_loss
        metrics["run_id"] = run_id
        metrics["target"] = target
        metrics["validation_fold"] = validation_fold
        metrics["experiment_id"] = experiment_id
        metrics["hypothesis_id"] = hypothesis_id
        metrics["parent_run_ids"] = parent_run_ids
        metrics["training_augmentation"] = training_augmentation
        metrics["loss_name"] = loss_name
        metrics["class_counts"] = class_counts.tolist()
        metrics["class_weights"] = class_weights.tolist() if class_weights is not None else None
        metrics["classifier_dropout"] = classifier_dropout
        metrics["label_smoothing"] = (
            child_spec.label_smoothing if child_spec is not None else 0.0
        )
        metrics["focal_gamma"] = child_spec.focal_gamma if child_spec is not None else 0.0
        metrics["model_family"] = model_family
        metrics["parameter_count"] = parameter_count
        metrics["architecture_macs"] = architecture_macs
        metrics["final_train_eval_loss"] = final_train_loss
        metrics["final_train_eval_macro_f1"] = final_train_metrics["macro_f1"]
        metrics["final_train_validation_macro_f1_gap"] = float(
            final_train_metrics["macro_f1"] - metrics["macro_f1"]
        )
        if target == "usage":
            without_home = [index for index, name in enumerate(classes) if name != "Home"]
            per_class = metrics["per_class"]
            metrics["macro_f1_without_home"] = float(
                np.mean([per_class[index]["f1"] for index in without_home])
            )
        _json_dump(metrics, metrics_path)

        robustness_rows: list[dict[str, object]] = []
        clean_macro_f1 = float(metrics["macro_f1"])
        for corruption in CORE_CORRUPTIONS:
            corrupted_dataset = Task3ImageDataset(
                validation, corruption=corruption, **dataset_kwargs
            )
            corrupted_loader = _loader(
                corrupted_dataset, config=config, shuffle=False, device=device
            )
            corrupted_loss, corrupted_labels, corrupted_probabilities, _ = _pass(
                model, corrupted_loader, criterion, device
            )
            corrupted_metrics = classification_metrics(
                corrupted_labels, corrupted_probabilities, classes
            )
            robustness_rows.append(
                {
                    "run_id": run_id,
                    "validation_fold": validation_fold,
                    "corruption": corruption,
                    "loss": corrupted_loss,
                    "macro_f1": corrupted_metrics["macro_f1"],
                    "macro_f1_change": float(corrupted_metrics["macro_f1"]) - clean_macro_f1,
                }
            )
        pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)

        train_seconds = training_finished - started
        diagnostic_seconds = time.perf_counter() - training_finished
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        latency_ms = _measure_latency(
            model,
            device,
            height=config.image_height,
            width=config.image_width,
        )
        metrics["latency_ms_batch_1"] = latency_ms
        metrics["train_seconds"] = train_seconds
        metrics["diagnostic_seconds"] = diagnostic_seconds
        metrics["peak_memory_bytes"] = peak_memory
        _json_dump(metrics, metrics_path)
        registry.complete(
            run_id,
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": compute_sha256(checkpoint_path),
                "prediction_path": prediction_path,
                "prediction_sha256": compute_sha256(prediction_path),
                "metrics_json": metrics,
                "train_seconds": train_seconds,
                "peak_memory_bytes": peak_memory,
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "last_completed_stage": "diagnostic_bundle_complete",
            },
        )
        _log(f"completed target={target} fold={validation_fold}: {run_id}")
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "robustness_path": str(robustness_path),
            "metrics": metrics,
        }
    except BaseException as error:
        registry.fail(run_id, error, last_completed_stage=last_stage)
        _log(
            f"failed target={target} fold={validation_fold} after {last_stage}: "
            f"{type(error).__name__}: {error}"
        )
        raise


def _aggregate_target(
    target: str,
    fold_results: Sequence[dict[str, object]],
    *,
    output_root: Path,
    root: Path,
    artifact_dir: str = "baseline",
    experiment_id: str = BASELINE_EXPERIMENT_ID,
    hypothesis_id: str = "baseline_contract",
    model_family: str = "task3_small_cnn",
    parameter_count: int | None = None,
    architecture_macs: int | None = None,
) -> dict[str, object]:
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    classes, _ = _class_spec(label_maps, target)
    prediction_paths = [Path(str(result["prediction_path"])) for result in fold_results]
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    if predictions["id"].duplicated().any():
        raise ValueError("aggregate OOF predictions contain duplicate IDs")
    probability_columns = [f"probability_{index}_{name}" for index, name in enumerate(classes)]
    probabilities = predictions[probability_columns].to_numpy(dtype=np.float64)
    labels = predictions["true_index"].to_numpy(dtype=np.int64)
    metrics = classification_metrics(labels, probabilities, classes)
    metrics["experiment_id"] = experiment_id
    metrics["hypothesis_id"] = hypothesis_id
    metrics["fold_run_ids"] = [str(result["run_id"]) for result in fold_results]
    metrics["model_family"] = model_family
    metrics["parameter_count"] = parameter_count
    metrics["architecture_macs"] = architecture_macs
    if target == "usage":
        metrics["macro_f1_without_home"] = float(
            np.mean([row["f1"] for row in metrics["per_class"] if row["class_name"] != "Home"])
        )
    aggregate_dir = output_root / artifact_dir / target / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = aggregate_dir / "oof_predictions.csv"
    metrics_path = aggregate_dir / "metrics.json"
    class_report_path = aggregate_dir / "per_class.csv"
    confusion_path = aggregate_dir / "confusion_matrix.csv"
    failures_path = aggregate_dir / "failure_index.csv"
    predictions.sort_values("id").to_csv(predictions_path, index=False)
    _json_dump(metrics, metrics_path)
    pd.DataFrame(metrics["per_class"]).to_csv(class_report_path, index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
        confusion_path, index_label="true_label"
    )
    _failure_index(predictions, classes).to_csv(failures_path, index=False)
    return {
        "target": target,
        "fold_run_ids": [str(result["run_id"]) for result in fold_results],
        "prediction_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "class_report_path": str(class_report_path),
        "confusion_path": str(confusion_path),
        "failure_index_path": str(failures_path),
        "metrics": metrics,
    }


def run_task3_baseline_cv(
    target: str,
    *,
    output_root: str | Path,
    folds: Iterable[int] = range(5),
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
    child_spec: Task3ChildSpec | None = None,
) -> dict[str, object]:
    """Run a full five-fold baseline or locked child and create pooled OOF artifacts."""
    fold_list = tuple(int(fold) for fold in folds)
    if fold_list != tuple(range(5)):
        raise ValueError("Task 3 evidence requires folds 0,1,2,3,4 in order")
    if child_spec is not None and child_spec.target != target:
        raise ValueError("child target and requested target disagree")
    root = Path(root)
    output_root = Path(output_root)
    experiment_id = child_spec.experiment_id if child_spec is not None else BASELINE_EXPERIMENT_ID
    hypothesis_id = child_spec.hypothesis_id if child_spec is not None else "baseline_contract"
    artifact_dir = child_spec.artifact_dir if child_spec is not None else "baseline"
    config = Task3BaselineConfig(target=target)
    model_family, _, parameter_count, architecture_macs = _task3_model_contract(
        config, child_spec
    )
    _log(f"starting five-fold experiment={experiment_id} for target={target}")
    results = [
        run_task3_baseline_fold(
            target,
            fold,
            output_root=output_root,
            registry_path=registry_path,
            registry_mirrors=registry_mirrors,
            root=root,
            device_name=device_name,
            child_spec=child_spec,
        )
        for fold in fold_list
    ]
    aggregate = _aggregate_target(
        target,
        results,
        output_root=output_root,
        root=root,
        artifact_dir=artifact_dir,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        model_family=model_family,
        parameter_count=parameter_count,
        architecture_macs=architecture_macs,
    )
    _log(f"completed pooled five-fold OOF experiment={experiment_id} target={target}")
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("gender", "usage"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--registry", type=Path, default=RUNS_CSV)
    parser.add_argument("--registry-mirror", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.check_only:
        print(
            json.dumps(
                check_task3_baseline_setup(
                    args.target,
                    root=args.root,
                    device_name=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_root is None:
        raise ValueError("--output-root is required with --train")
    result = run_task3_baseline_cv(
        args.target,
        output_root=args.output_root,
        registry_path=args.registry,
        registry_mirrors=args.registry_mirror,
        root=args.root,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
