"""One scratch MixUp 0.20 refit on all development rows, for Colab."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from fashion.config import ROOT
from fashion.data import get_samples
from fashion.data.hashing import compute_sha256
from fashion.train import task3_gender_mixup_cv as cv
from fashion.train.artifacts import atomic_write_json
from fashion.train.data import Task3ImageDataset, fit_fold_rgb_stats
from fashion.train.mixup import TrainingMixUp
from fashion.train.model import Task3GeM3CNN
from fashion.train.registry import RunRegistry
from fashion.train.task3_baseline import (
    _loader,
    _pass,
    runtime_environment,
    set_reproducible_seed,
    validate_verified_colab_runtime,
)
from fashion.train.task3_gender_sam25_refit import (
    _digest,
    _training_precision,
    _verify_completed,
)

EXPERIMENT = "t3_gender_name_truth_mixup_alpha020_refit"
CLASSES = cv.CLASSES
MEMORY_LIMIT = cv.MEMORY_LIMIT
SOURCE_CONFIG = cv.SOURCE_CONFIG


def prepare_refit(*, root=ROOT):
    """Reuse the exact CV recipe checks, then bind MixUp to all development rows."""
    config, payload, splits = cv.prepare_cv(root=root)
    training = get_samples(splits, partition="development", target="gender").reset_index(drop=True)
    mixup = TrainingMixUp(
        training,
        validation_fold=None,
        scope="development_refit",
        label_to_index={name: i for i, name in enumerate(CLASSES)},
        seed=config.seed,
        alpha=payload["mixup_policy"]["alpha"],
    )
    if mixup.contract["policy"] != payload["mixup_policy"]:
        raise ValueError("Refit MixUp policy differs from the five-fold recipe")
    payload.pop("comparison_precision")
    payload.update(
        experiment_id=EXPERIMENT,
        training_scope="all_development",
        validation_fold=None,
        validation_used=False,
        mixup_contract=mixup.contract,
        training_rows_sha256=mixup.contract["training_rows_sha256"],
        selection_status="candidate_refit_for_later_review",
    )
    payload["implementation_sha256"]["train/task3_gender_mixup_refit.py"] = compute_sha256(
        Path(root) / "src/fashion/train/task3_gender_mixup_refit.py"
    )
    return config, payload, training


def _cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("Select a fresh Colab L4 GPU runtime for the MixUp refit")
    device = torch.device("cuda")
    validate_verified_colab_runtime(runtime_environment(device))
    return device


def _fit(config, payload, training, *, root, run_dir, run_id, registry, device):
    """Train MixUp alone on every development row, without validation."""
    stats = fit_fold_rgb_stats(training, root=root)
    atomic_write_json(
        run_dir / "normalization.json",
        {
            **stats,
            "fit_scope": "all_development_content_pixels_only",
            "padding_excluded": True,
            "training_rows_sha256": payload["training_rows_sha256"],
        },
    )
    dataset = Task3ImageDataset(
        training,
        target="gender",
        root=root,
        label_to_index={name: i for i, name in enumerate(CLASSES)},
        mean=stats["mean"],
        std=stats["std"],
        augmentation=payload["training_augmentation"],
        image_view=payload["input_view"],
        image_size=(config.image_height, config.image_width),
    )
    loader = _loader(dataset, config=config, shuffle=True, device=device)
    model = Task3GeM3CNN(config, classifier_dropout=payload["classifier_dropout"]).to(device)
    if sum(p.numel() for p in model.parameters()) != payload["parameter_count"]:
        raise ValueError("Refit architecture differs from the frozen 390,181-parameter model")
    criterion = nn.CrossEntropyLoss()
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
    mixup = TrainingMixUp(
        training,
        validation_fold=None,
        scope="development_refit",
        seed=config.seed,
        alpha=payload["mixup_policy"]["alpha"],
        label_to_index={name: i for i, name in enumerate(CLASSES)},
    )
    if mixup.contract != payload["mixup_contract"]:
        raise ValueError("Refit training rows or MixUp settings changed after preflight")
    history = []
    peak = 0
    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        mixup.begin_epoch(epoch)
        loss, _, _, _ = _pass(
            model,
            loader,
            criterion,
            device,
            optimizer=optimizer,
            mixup=mixup,
        )
        mixup.end_epoch()
        step = mixup.epochs[-1]
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            if peak >= MEMORY_LIMIT:
                raise RuntimeError("MixUp refit exceeded the 3 GB allocated GPU memory limit")
        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr,
                "train_mixed_loss": loss,
                "training_rows": step["rows"],
                "optimizer_steps": step["batches"],
                "selected_checkpoint": epoch == config.epochs,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        atomic_write_json(run_dir / "mixup_training.json", mixup.receipt())
        scheduler.step()
        registry.update(run_id, {"last_completed_stage": f"epoch_{epoch}_complete"})
        print(f"Epoch {epoch}/{config.epochs}: mixed training loss {loss:.4f}", flush=True)
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
    return {
        "scope": "development_refit_training_only",
        "epochs_completed": config.epochs,
        "selected_epoch": config.epochs,
        "checkpoint_policy": "final_epoch",
        "training_rows": len(training),
        "validation_rows": 0,
        "validation_used": False,
        "holdout_evaluated": False,
        "teacher_test_evaluated": False,
        "train_mixed_loss": history[-1]["train_mixed_loss"],
        "online_train_f1": "not_applicable_mixed_inputs",
        "peak_memory_bytes": peak,
    }


def run_gender_mixup_refit(*, output_root, registry_path=None, registry_mirrors=(), root=ROOT):
    """Register and save one new refit; verify completed artifacts on a repeat Run All."""
    root = Path(root)
    registry_path = Path(registry_path) if registry_path else root / "results/runs.csv"
    device = _cuda_device()
    config, payload, training = prepare_refit(root=root)
    destination = Path(output_root) / "experiments" / EXPERIMENT / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".refit.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another refit writer is active in this experiment") from exc
        if (destination / "model_manifest.json").exists():
            return _verify_completed(destination, payload, registry_path)
        # Partial attempts restart from scratch; the accepted final artifact is separate.
        run_id = f"{EXPERIMENT}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        run_dir = destination / run_id
        run_dir.mkdir(exist_ok=False)
        atomic_write_json(run_dir / "config.json", payload)
        training[["id", "cv_fold", "product_family_group", "gender", "sha256", "path"]].to_csv(
            run_dir / "training_rows.csv",
            index=False,
        )
        registry = RunRegistry(registry_path, mirrors=registry_mirrors)
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
                "validation_fold": None,
                "seed": config.seed,
                "scratch": True,
                "debug": False,
                "submission_eligible": True,
                "config_hash": _digest(payload),
                "config_path": run_dir / "config.json",
                "split_digest": payload["gender_label_variant"]["canonical_split_sha256"],
                "label_map_digest": payload["label_map_sha256"],
                "training_product_count": len(training),
                "validation_product_count": 0,
                "training_family_count": training.product_family_group.nunique(),
                "validation_family_count": 0,
                "model_family": payload["effective_model_family"],
                "parameter_count": payload["parameter_count"],
                "environment_json": environment,
                "history_path": run_dir / "history.csv",
                "last_completed_stage": "registered",
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
                    root=root,
                    run_dir=run_dir,
                    run_id=run_id,
                    registry=registry,
                    device=device,
                )
            metrics["train_seconds"] = time.perf_counter() - started
            atomic_write_json(run_dir / "metrics.json", metrics)
            files = {
                name: {"path": f"{run_id}/{name}", "sha256": compute_sha256(run_dir / name)}
                for name in (
                    "final_epoch.pt",
                    "config.json",
                    "normalization.json",
                    "history.csv",
                    "metrics.json",
                    "training_rows.csv",
                    "mixup_training.json",
                )
            }
            manifest = {
                "status": "complete",
                "experiment_id": EXPERIMENT,
                "run_id": run_id,
                "training_scope": "all_development",
                "config_hash": _digest(payload),
                "scratch": True,
                "selected_epoch": config.epochs,
                "class_names": CLASSES,
                "files": files,
                "inference": "one model; saved development normalization; softmax then argmax",
                "validation_used": False,
                "holdout_evaluated": False,
                "teacher_test_evaluated": False,
                "metrics": metrics,
                "final_model_changed": False,
                "selection_status": "candidate_refit_for_later_review",
            }
            registry.complete(
                run_id,
                {
                    "checkpoint_path": run_dir / "final_epoch.pt",
                    "checkpoint_sha256": files["final_epoch.pt"]["sha256"],
                    "checkpoint_bytes": (run_dir / "final_epoch.pt").stat().st_size,
                    "metrics_json": metrics,
                    "train_seconds": metrics["train_seconds"],
                    "peak_memory_bytes": metrics["peak_memory_bytes"],
                    "last_completed_stage": "refit_artifacts_complete",
                },
            )
        except BaseException as exc:
            current = pd.read_csv(registry_path, keep_default_na=False)
            stage = current.loc[current.run_id.eq(run_id), "last_completed_stage"].iloc[0]
            registry.fail(run_id, exc, last_completed_stage=stage)
            raise
        manifest_path = destination / "model_manifest.json"
        atomic_write_json(manifest_path, manifest)
        return dict(manifest, manifest_path=str(manifest_path), reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--registry-mirror", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = run_gender_mixup_refit(
        root=args.root,
        output_root=args.output_root,
        registry_path=args.registry_path,
        registry_mirrors=args.registry_mirror,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
