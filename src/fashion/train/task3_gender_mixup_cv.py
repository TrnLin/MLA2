"""Five fresh Gender MixUp folds with the saved 30-epoch screen recipe."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
import uuid
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from fashion.config import ROOT
from fashion.data import get_samples, load_label_maps
from fashion.data.gender_name_truth import (
    build_gender_name_truth_variant,
    load_gender_name_truth_variant,
)
from fashion.data.hashing import compute_sha256
from fashion.train.artifacts import atomic_write_json
from fashion.train.config import Task3BaselineConfig
from fashion.train.data import (
    CORE_CORRUPTIONS,
    Task3ImageDataset,
    fit_fold_rgb_stats,
    task3_target_frames,
)
from fashion.train.metrics import classification_metrics
from fashion.train.mixup import TrainingMixUp
from fashion.train.model import Task3GeM3CNN
from fashion.train.registry import RunRegistry
from fashion.train.task3_baseline import (
    _loader,
    _pass,
    _prediction_frame,
    runtime_environment,
    set_reproducible_seed,
    validate_verified_colab_runtime,
)
from fashion.train.task3_decisions import oof_metrics, validate_oof
from fashion.train.task3_gender_diagnostic import verify_input_images
from fashion.train.task3_gender_ieee import POLICY as IEEE_POLICY
from fashion.train.task3_gender_precision import ieee_precision
from fashion.train.task3_gender_sam25_refit import _digest, _training_precision

EXPERIMENT = "t3_gender_name_truth_mixup_alpha020_cv"
SOURCE_CONFIG = Path("configs/task3/gender_mixup_alpha020_source.json")
SOURCE_SHA256 = "5a8214917e34a489f8301614cb95fa6058e751c894e801ab77b35686ff0b7043"
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
FOLDS = tuple(range(5))
EXPECTED_ROWS = 32773
MEMORY_LIMIT = 3_000_000_000
IMPLEMENTATION_FILES = (
    "train/task3_gender_mixup_cv.py",
    "train/task3_baseline.py",
    "train/task3_gender_sam25_refit.py",
    "train/task3_gender_precision.py",
    "train/task3_gender_ieee.py",
    "train/task3_gender_diagnostic.py",
    "train/config.py",
    "train/model.py",
    "train/data.py",
    "train/augmentation.py",
    "train/mixup.py",
    "train/metrics.py",
    "train/task3_decisions.py",
    "train/registry.py",
    "train/task3_registry.py",
    "train/artifacts.py",
    "data/dataset.py",
    "data/images.py",
    "data/gender_name_truth.py",
    "data/splits.py",
)


def prepare_cv(*, root=ROOT):
    """Verify the historical configuration, labels, folds and development images."""
    root = Path(root)
    source_path = root / SOURCE_CONFIG
    if compute_sha256(source_path) != SOURCE_SHA256:
        raise ValueError("The frozen MixUp source configuration changed")
    source = json.loads(source_path.read_text())
    contract = source["gender_label_variant"]
    if compute_sha256(root / "data/processed/splits.csv") != contract["canonical_split_sha256"]:
        raise ValueError("The canonical split differs from the MixUp screen")
    build_gender_name_truth_variant(root)
    splits = load_gender_name_truth_variant(root)
    if any(value != contract[key] for key, value in splits.attrs["gender_label_variant"].items()):
        raise ValueError("Corrected Gender labels differ from the MixUp screen")
    development = get_samples(splits, partition="development", target="gender")
    protected = splits.loc[~splits.partition.eq("development")]
    if (
        len(development) != EXPECTED_ROWS
        or set(development.cv_fold) != set(FOLDS)
        or development.id.duplicated().any()
        or set(development.id) & set(protected.id)
        or set(development.product_family_group) & set(protected.product_family_group)
    ):
        raise ValueError("MixUp CV requires exactly the five saved development folds")
    if load_label_maps(root / "data/processed/label_maps.json")["gender"]["classes"] != CLASSES:
        raise ValueError("Gender class order changed")
    values = {field.name: source[field.name] for field in fields(Task3BaselineConfig)}
    values["channels"] = tuple(values["channels"])
    config = Task3BaselineConfig(**values)
    child = source["child_experiment"]
    if config.epochs != 30 or "sam_policy" in child or not config.scratch:
        raise ValueError("Require scratch MixUp alone for 30 epochs")
    payload = {
        **config.to_dict(),
        "experiment_id": EXPERIMENT,
        "class_names": CLASSES,
        "effective_model_family": source["effective_model_family"],
        "classifier_dropout": child["classifier_dropout"],
        "training_augmentation": child["training_augmentation"],
        "input_view": source["input_view"],
        "cosine_t_max": config.epochs,
        "parameter_count": source["parameter_count"],
        "mixup_policy": child["mixup_policy"],
        "training_precision_settings": source["training_precision_settings"],
        "comparison_precision": IEEE_POLICY,
        "gender_label_variant": contract,
        "source_config": str(SOURCE_CONFIG),
        "source_config_sha256": SOURCE_SHA256,
        "source_role": "recipe_only_no_checkpoint_loaded",
        "label_map_sha256": compute_sha256(root / "data/processed/label_maps.json"),
        "implementation_sha256": {
            name: compute_sha256(root / "src/fashion" / name) for name in IMPLEMENTATION_FILES
        },
        "holdout_evaluated": False,
        "teacher_test_evaluated": False,
        "independent_blind_test": False,
    }
    for fold in FOLDS:
        training, validation = task3_target_frames(splits, target="gender", validation_fold=fold)
        if set(training.gender) != set(CLASSES) or set(validation.gender) != set(CLASSES):
            raise ValueError("Every training and validation fold must support all five classes")
        mixup = _mixup(training, config, fold)
        if mixup.contract["policy"] != payload["mixup_policy"]:
            raise ValueError("MixUp implementation differs from the saved policy")
    print(f"Checking {len(development):,} development image hashes", flush=True)
    verify_input_images(development, root)
    return config, payload, splits


def _mixup(training, config, fold):
    return TrainingMixUp(
        training,
        validation_fold=fold,
        label_to_index={name: index for index, name in enumerate(CLASSES)},
        seed=config.seed,
        alpha=0.2,
    )


def _cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("Select a fresh Colab L4 GPU runtime for five-fold MixUp training")
    device = torch.device("cuda")
    validate_verified_colab_runtime(runtime_environment(device))
    return device


def _evaluate(model, frame, *, config, kwargs, device, run_id, corruption=None):
    dataset = Task3ImageDataset(frame, corruption=corruption, **kwargs)
    loader = _loader(dataset, config=config, shuffle=False, device=device)
    loss, labels, probabilities, trace = _pass(model, loader, nn.CrossEntropyLoss(), device)
    metrics = classification_metrics(labels, probabilities, CLASSES)
    metrics["loss"] = loss
    predictions = _prediction_frame(labels, probabilities, trace, CLASSES, run_id)
    # Preserve exact FP32 values when pandas writes the CSV, as in the IEEE audit.
    columns = ["confidence", *(f"probability_{i}_{name}" for i, name in enumerate(CLASSES))]
    predictions[columns] = predictions[columns].astype("float64")
    return metrics, predictions


def _fit(config, payload, training, validation, *, root, run_dir, run_id, registry, device):
    """Reuse the tested data and batch mechanics; no SAM or checkpoint loading."""
    stats = fit_fold_rgb_stats(training, root=root)
    atomic_write_json(
        run_dir / "normalization.json",
        {
            **stats,
            "fit_scope": "fold_training_content_pixels_only",
            "padding_excluded": True,
            "validation_fold": payload["validation_fold"],
            "training_rows_sha256": payload["mixup_contract"]["training_rows_sha256"],
        },
    )
    kwargs = dict(
        target="gender",
        root=root,
        label_to_index={name: i for i, name in enumerate(CLASSES)},
        mean=stats["mean"],
        std=stats["std"],
        image_view=payload["input_view"],
        image_size=(config.image_height, config.image_width),
    )
    train_loader = _loader(
        Task3ImageDataset(training, augmentation=payload["training_augmentation"], **kwargs),
        config=config,
        shuffle=True,
        device=device,
    )
    validation_loader = _loader(
        Task3ImageDataset(validation, **kwargs),
        config=config,
        shuffle=False,
        device=device,
    )
    model = Task3GeM3CNN(config, classifier_dropout=payload["classifier_dropout"]).to(device)
    if sum(p.numel() for p in model.parameters()) != payload["parameter_count"]:
        raise ValueError("Architecture differs from the saved 390,181-parameter model")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=payload["cosine_t_max"],
        eta_min=config.minimum_learning_rate,
    )
    mixup = _mixup(training, config, payload["validation_fold"])
    criterion = nn.CrossEntropyLoss()
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        mixup.begin_epoch(epoch)
        train_loss, _, _, _ = _pass(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            mixup=mixup,
        )
        mixup.end_epoch()
        val_loss, labels, probabilities, _ = _pass(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": train_loss,
                "train_macro_f1": float("nan"),
                "train_metric_scope": payload["mixup_policy"]["online_train_f1"],
                "validation_loss": val_loss,
                "validation_macro_f1": classification_metrics(labels, probabilities, CLASSES)[
                    "macro_f1"
                ],
                "validation_precision": "training_runtime_defaults",
                "selected_checkpoint": epoch == config.epochs,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        atomic_write_json(run_dir / "mixup_training.json", mixup.receipt())
        scheduler.step()
        if device.type == "cuda" and torch.cuda.max_memory_allocated(device) >= MEMORY_LIMIT:
            raise RuntimeError("MixUp exceeded the 3 GB allocated GPU memory limit")
        registry.update(run_id, {"last_completed_stage": f"epoch_{epoch}_complete"})
        print(
            f"Fold {payload['validation_fold']} epoch {epoch}/30: "
            f"loss {train_loss:.4f}, validation F1 {history[-1]['validation_macro_f1']:.4f}",
            flush=True,
        )
    train_seconds = time.perf_counter() - training_started
    checkpoint = {
        "run_id": run_id,
        "config": payload,
        "class_names": CLASSES,
        "normalization": stats,
        "checkpoint_policy": "final_epoch",
        "epochs_completed": config.epochs,
        "selected_epoch": config.epochs,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    temporary = run_dir / "final_epoch.pt.partial"
    torch.save(checkpoint, temporary)
    temporary.replace(run_dir / "final_epoch.pt")
    # Final clean and corrupted predictions use the same IEEE FP32 policy as SAM25 CV.
    with ieee_precision(torch):
        clean, predictions = _evaluate(
            model,
            validation,
            config=config,
            kwargs=kwargs,
            device=device,
            run_id=run_id,
        )
        predictions = validate_oof(
            predictions,
            validation,
            target="gender",
            classes=CLASSES,
            run_ids_by_fold={payload["validation_fold"]: run_id},
        )
        predictions.to_csv(run_dir / "oof_predictions.csv", index=False)
        train_metrics, train_predictions = _evaluate(
            model,
            training,
            config=config,
            kwargs=kwargs,
            device=device,
            run_id=run_id,
        )
        train_predictions.to_csv(run_dir / "train_eval_predictions.csv", index=False)
        robustness = []
        for corruption in CORE_CORRUPTIONS:
            scores, corrupted = _evaluate(
                model,
                validation,
                config=config,
                kwargs=kwargs,
                device=device,
                run_id=run_id,
                corruption=corruption,
            )
            corrupted.to_csv(run_dir / f"predictions_{corruption}.csv", index=False)
            robustness.append(
                {
                    "corruption": corruption,
                    "validation_fold": payload["validation_fold"],
                    "macro_f1": scores["macro_f1"],
                    "accuracy": scores["accuracy"],
                    "macro_f1_change": scores["macro_f1"] - clean["macro_f1"],
                }
            )
        pd.DataFrame(robustness).to_csv(run_dir / "robustness.csv", index=False)
    return {
        **clean,
        "run_id": run_id,
        "validation_fold": payload["validation_fold"],
        "comparison_precision": IEEE_POLICY,
        "checkpoint_policy": "final_epoch",
        "epochs_completed": config.epochs,
        "selected_epoch": config.epochs,
        "training_rows": len(training),
        "validation_rows": len(validation),
        "final_train_eval_loss": train_metrics["loss"],
        "final_train_eval_macro_f1": train_metrics["macro_f1"],
        "final_train_validation_macro_f1_gap": train_metrics["macro_f1"] - clean["macro_f1"],
        "mixup_receipt_sha256": compute_sha256(run_dir / "mixup_training.json"),
        "train_seconds": train_seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "holdout_evaluated": False,
        "teacher_test_evaluated": False,
    }


def _completed_fold(destination, fold, payload, registry_path):
    manifests = list(destination.glob(f"fold{fold}_*/manifest.json"))
    if len(manifests) > 1:
        raise ValueError(f"Duplicate completed MixUp fold {fold}; keep one experiment per folder")
    if not manifests:
        return None
    manifest = json.loads(manifests[0].read_text())
    if manifest["status"] != "complete" or manifest["config_hash"] != _digest(payload):
        raise ValueError("Completed MixUp fold has different code/data/settings")
    directory = manifests[0].parent
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or compute_sha256(path) != digest:
            raise ValueError("Completed MixUp artifact changed")
    rows = pd.read_csv(registry_path, keep_default_na=False)
    row = rows.loc[rows.run_id.eq(manifest["run_id"])]
    if (
        len(row) != 1
        or row.iloc[0].status != "complete"
        or row.iloc[0].config_hash != manifest["config_hash"]
        or row.iloc[0].checkpoint_sha256 != manifest["files"]["final_epoch.pt"]
    ):
        raise ValueError("Completed MixUp registry identity changed")
    return directory


def _run_fold(config, payload, training, validation, *, destination, root, registry, device):
    fold = payload["validation_fold"]
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_id = f"{EXPERIMENT}_f{fold}_{stamp}"
    run_dir = destination / f"fold{fold}_{stamp}"
    run_dir.mkdir(exist_ok=False)
    atomic_write_json(run_dir / "config.json", payload)
    columns = ["id", "cv_fold", "product_family_group", "gender", "sha256", "path"]
    training[columns].to_csv(run_dir / "training_rows.csv", index=False)
    validation[columns].to_csv(run_dir / "validation_rows.csv", index=False)
    environment = runtime_environment(device)
    environment["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    registry.start(
        {
            "run_id": run_id,
            "experiment_id": EXPERIMENT,
            "task": "task3",
            "target": "gender",
            "validation_fold": fold,
            "seed": config.seed,
            "scratch": True,
            "debug": False,
            "submission_eligible": True,
            "config_hash": _digest(payload),
            "config_path": run_dir / "config.json",
            "split_digest": payload["gender_label_variant"]["canonical_split_sha256"],
            "label_map_digest": payload["label_map_sha256"],
            "training_product_count": len(training),
            "validation_product_count": len(validation),
            "training_family_count": training.product_family_group.nunique(),
            "validation_family_count": validation.product_family_group.nunique(),
            "model_family": payload["effective_model_family"],
            "parameter_count": payload["parameter_count"],
            "environment_json": environment,
            "history_path": run_dir / "history.csv",
            "last_completed_stage": "registered_before_first_optimizer_step",
        }
    )
    started = time.perf_counter()
    try:
        set_reproducible_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with _training_precision(payload["training_precision_settings"]):
            metrics = _fit(
                config,
                payload,
                training,
                validation,
                root=root,
                run_dir=run_dir,
                run_id=run_id,
                registry=registry,
                device=device,
            )
        metrics["total_seconds"] = time.perf_counter() - started
        atomic_write_json(run_dir / "metrics.json", metrics)
        files = {p.name: compute_sha256(p) for p in run_dir.iterdir() if p.is_file()}
        registry.complete(
            run_id,
            {
                "checkpoint_path": run_dir / "final_epoch.pt",
                "checkpoint_sha256": files["final_epoch.pt"],
                "checkpoint_bytes": (run_dir / "final_epoch.pt").stat().st_size,
                "prediction_path": run_dir / "oof_predictions.csv",
                "prediction_sha256": files["oof_predictions.csv"],
                "metrics_json": metrics,
                "train_seconds": metrics["train_seconds"],
                "peak_memory_bytes": metrics["peak_memory_bytes"],
                "last_completed_stage": "mixup_cv_fold_artifacts_complete",
            },
        )
        atomic_write_json(
            run_dir / "manifest.json",
            {
                "status": "complete",
                "run_id": run_id,
                "validation_fold": fold,
                "config_hash": _digest(payload),
                "files": files,
            },
        )
    except BaseException as exc:
        registry.fail(run_id, exc, last_completed_stage="mixup_cv_fold_failed")
        raise
    return run_dir


def summarize_cv(directories, splits, *, destination):
    """Pool one held-out prediction per development row; never average training predictions."""
    metrics = [json.loads((directory / "metrics.json").read_text()) for directory in directories]
    if sorted(m["validation_fold"] for m in metrics) != list(FOLDS):
        raise ValueError("Summary requires exactly one complete run for each of five folds")
    run_ids = {m["validation_fold"]: m["run_id"] for m in metrics}
    expected = get_samples(splits, partition="development", target="gender")
    predictions = validate_oof(
        pd.concat(
            [pd.read_csv(d / "oof_predictions.csv", keep_default_na=False) for d in directories]
        ),
        expected,
        target="gender",
        classes=CLASSES,
        run_ids_by_fold=run_ids,
    )
    predictions.to_csv(destination / "oof_predictions.csv", index=False)
    scores = oof_metrics(predictions, CLASSES)
    fold_table = pd.DataFrame(
        [
            {
                key: m[key]
                for key in (
                    "run_id",
                    "validation_fold",
                    "macro_f1",
                    "accuracy",
                    "ece_15",
                    "nll",
                    "final_train_eval_macro_f1",
                    "final_train_validation_macro_f1_gap",
                    "train_seconds",
                )
            }
            for m in metrics
        ]
    )
    fold_table.to_csv(destination / "fold_summary.csv", index=False)
    robustness = []
    for corruption in CORE_CORRUPTIONS:
        corrupted = validate_oof(
            pd.concat(
                [
                    pd.read_csv(d / f"predictions_{corruption}.csv", keep_default_na=False)
                    for d in directories
                ]
            ),
            expected,
            target="gender",
            classes=CLASSES,
            run_ids_by_fold=run_ids,
        )
        corrupted.to_csv(destination / f"oof_{corruption}.csv", index=False)
        value = oof_metrics(corrupted, CLASSES)
        robustness.append(
            {
                "corruption": corruption,
                "macro_f1": value["macro_f1"],
                "macro_f1_change": value["macro_f1"] - scores["macro_f1"],
            }
        )
    pd.DataFrame(robustness).to_csv(destination / "robustness.csv", index=False)
    summary = {
        "status": "complete_for_review",
        "experiment_id": EXPERIMENT,
        "folds": list(FOLDS),
        "run_ids_by_fold": run_ids,
        "oof_rows": len(predictions),
        "pooled_oof": scores,
        "comparison_precision": IEEE_POLICY,
        "mean_fold_macro_f1": float(fold_table.macro_f1.mean()),
        "std_fold_macro_f1": float(fold_table.macro_f1.std(ddof=1)),
        "mean_clean_train_validation_gap": float(
            fold_table.final_train_validation_macro_f1_gap.mean()
        ),
        "holdout_evaluated": False,
        "teacher_test_evaluated": False,
        "independent_blind_test": False,
        "final_model_changed": False,
        "interpretation": (
            "Development comparison after earlier screens; no automatic winner or refit."
        ),
        "models": [
            {
                "validation_fold": m["validation_fold"],
                "run_id": m["run_id"],
                "directory": d.name,
                "manifest_sha256": compute_sha256(d / "manifest.json"),
            }
            for d, m in zip(directories, metrics, strict=True)
        ],
    }
    atomic_write_json(destination / "cv_summary.json", summary)
    return dict(summary, summary_path=str(destination / "cv_summary.json"))


def run_gender_mixup_cv(*, output_root, registry_path=None, registry_mirrors=(), root=ROOT):
    """Train all five folds, reusing only intact completed runs of this exact experiment."""
    root = Path(root)
    device = _cuda_device()
    config, payload, splits = prepare_cv(root=root)
    registry_path = Path(registry_path) if registry_path else root / "results/runs.csv"
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    destination = Path(output_root) / "experiments" / EXPERIMENT / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".cv.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another MixUp CV writer is active") from exc
        directories = []
        # Validate every saved fold before spending GPU time on any missing fold.
        plans = []
        for fold in FOLDS:
            training, validation = task3_target_frames(
                splits, target="gender", validation_fold=fold
            )
            fold_payload = dict(
                payload,
                validation_fold=fold,
                mixup_contract=_mixup(training, config, fold).contract,
            )
            directory = _completed_fold(destination, fold, fold_payload, registry_path)
            plans.append((training, validation, fold_payload, directory))
        for training, validation, fold_payload, directory in plans:
            if directory is None:
                directory = _run_fold(
                    config,
                    fold_payload,
                    training,
                    validation,
                    destination=destination,
                    root=root,
                    registry=registry,
                    device=device,
                )
            else:
                print(
                    f"Verified completed fold {fold_payload['validation_fold']}; reusing it",
                    flush=True,
                )
            directories.append(directory)
        return summarize_cv(directories, splits, destination=destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--registry-mirror", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = run_gender_mixup_cv(
        root=args.root,
        output_root=args.output_root,
        registry_path=args.registry_path,
        registry_mirrors=args.registry_mirror,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
