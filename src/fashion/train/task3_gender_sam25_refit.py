"""One scratch SAM25 fit on all development rows, without validation or test access."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
import uuid
from contextlib import contextmanager
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
from fashion.train.data import Task3ImageDataset, fit_fold_rgb_stats
from fashion.train.mixup import TrainingMixUp
from fashion.train.model import Task3GeM3CNN
from fashion.train.registry import RunRegistry
from fashion.train.sam import SAMStep, policy_for_epochs
from fashion.train.task3_baseline import _loader, _pass, runtime_environment, set_reproducible_seed
from fashion.train.task3_gender_diagnostic import verify_input_images
from fashion.train.task3_gender_precision import _backend, precision_settings

EXPERIMENT = "t3_gender_name_truth_mixup_alpha020_sam005_epoch25_refit"
SOURCE_CONFIG = Path("reports/task3/gender_sam25_cv_result_20260906/fold0/config.json")
SOURCE_SHA256 = "18a4f91771d007c15c033c8c394d3f7a8ddd5de1da01a5f8c4dea9243edba75b"
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
EXPECTED_ROWS = 32773
MEMORY_LIMIT = 3_000_000_000
IMPLEMENTATION_FILES = (
    "train/task3_gender_sam25_refit.py",
    "train/task3_baseline.py",
    "train/config.py",
    "train/model.py",
    "train/data.py",
    "train/augmentation.py",
    "train/mixup.py",
    "train/sam.py",
    "train/registry.py",
    "train/task3_registry.py",
    "train/artifacts.py",
    "train/task3_gender_diagnostic.py",
    "train/task3_gender_precision.py",
    "data/dataset.py",
    "data/images.py",
    "data/gender_name_truth.py",
    "data/splits.py",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_refit(*, root=ROOT):
    """Check the selected recipe and every development image before allocating a model."""
    root = Path(root)
    source_path = root / SOURCE_CONFIG
    if compute_sha256(source_path) != SOURCE_SHA256:
        raise ValueError("The frozen SAM25 source configuration changed")
    source = json.loads(source_path.read_text())
    if (
        compute_sha256(root / "data/processed/splits.csv")
        != source["gender_label_variant"]["canonical_split_sha256"]
    ):
        raise ValueError("The canonical split differs from the frozen Gender recipe")
    build_gender_name_truth_variant(root)
    splits = load_gender_name_truth_variant(root)
    contract = splits.attrs["gender_label_variant"]
    if any(value != source["gender_label_variant"][key] for key, value in contract.items()):
        raise ValueError("Corrected labels differ from the frozen Gender recipe")
    training = get_samples(splits, partition="development", target="gender").reset_index(drop=True)
    protected = splits.loc[~splits.partition.eq("development")]
    if (
        len(training) != EXPECTED_ROWS
        or set(training.cv_fold) != set(range(5))
        or training.id.duplicated().any()
        or set(training.id) & set(protected.id)
        or set(training.product_family_group) & set(protected.product_family_group)
    ):
        raise ValueError(
            "Refit must contain exactly all development rows and no protected families"
        )
    classes = load_label_maps(root / "data/processed/label_maps.json")["gender"]["classes"]
    if classes != CLASSES or set(training.gender) != set(CLASSES):
        raise ValueError("Gender class order or training support changed")
    config_values = {field.name: source[field.name] for field in fields(Task3BaselineConfig)}
    config_values["channels"] = tuple(config_values["channels"])
    config = Task3BaselineConfig(**config_values)
    child = source["child_experiment"]
    if config.epochs != 25 or child["sam_policy"] != policy_for_epochs(25):
        raise ValueError("Refit requires the fixed 25-epoch SAM policy")
    mixup = TrainingMixUp(
        training,
        validation_fold=None,
        scope="development_refit",
        label_to_index={name: i for i, name in enumerate(classes)},
        seed=config.seed,
    )
    if mixup.contract["policy"] != child["mixup_policy"]:
        raise ValueError("MixUp differs from the frozen recipe")
    print(f"Checking {len(training):,} development image hashes", flush=True)
    verify_input_images(training, root)
    payload = {
        **config.to_dict(),
        "experiment_id": EXPERIMENT,
        "training_scope": "all_development",
        "validation_fold": None,
        "validation_used": False,
        "holdout_evaluated": False,
        "teacher_test_evaluated": False,
        "effective_model_family": child["model_family"],
        "classifier_dropout": child["classifier_dropout"],
        "training_augmentation": child["training_augmentation"],
        "input_view": source["input_view"],
        "cosine_t_max": source["cosine_t_max"],
        "sam_policy": child["sam_policy"],
        "mixup_contract": mixup.contract,
        "gender_label_variant": contract,
        "training_precision_settings": source["training_precision_settings"],
        "parameter_count": source["parameter_count"],
        "class_names": classes,
        "source_config": str(SOURCE_CONFIG),
        "source_config_sha256": SOURCE_SHA256,
        "source_role": "recipe_only_no_checkpoint_loaded",
        "training_rows_sha256": mixup.contract["training_rows_sha256"],
        "label_map_sha256": compute_sha256(root / "data/processed/label_maps.json"),
        "implementation_sha256": {
            name: compute_sha256(root / "src/fashion" / name) for name in IMPLEMENTATION_FILES
        },
    }
    return config, payload, training


def _cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Colab GPU runtime for the final SAM25 refit")
    return torch.device("cuda")


@contextmanager
def _training_precision(settings):
    previous = precision_settings(torch)
    try:
        for name, value in settings.items():
            _backend(torch, name).fp32_precision = value
        if precision_settings(torch) != settings:
            raise RuntimeError("Could not restore the frozen training precision settings")
        yield
    finally:
        for name, value in previous.items():
            _backend(torch, name).fp32_precision = value


def _verify_completed(destination, payload, registry_path):
    manifest_path = destination / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_hash"] != _digest(payload) or manifest["status"] != "complete":
        raise ValueError("Existing refit has different code/data/settings; it cannot be reused")
    for record in manifest["files"].values():
        path = (destination / record["path"]).resolve()
        if (
            not path.is_relative_to(destination.resolve())
            or compute_sha256(path) != record["sha256"]
        ):
            raise ValueError("Completed refit artifact changed")
    registry = pd.read_csv(registry_path, keep_default_na=False)
    row = registry.loc[registry.run_id.eq(manifest["run_id"])]
    if len(row) != 1 or row.iloc[0].status != "complete":
        raise ValueError("Completed refit has no unique completed registry row")
    if (
        row.iloc[0].config_hash != manifest["config_hash"]
        or row.iloc[0].checkpoint_sha256 != manifest["files"]["final_epoch.pt"]["sha256"]
    ):
        raise ValueError("Completed refit registry identity changed")
    return dict(manifest, manifest_path=str(manifest_path), reused=True)


def _fit(config, payload, training, *, root, run_dir, run_id, registry, device):
    """Run the existing SAM/MixUp batch mechanics with no validation loader."""
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
        label_to_index={name: i for i, name in enumerate(CLASSES)},
    )
    sam = SAMStep(model, optimizer, policy=payload["sam_policy"])
    history = []
    peak = 0
    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        mixup.begin_epoch(epoch)
        sam.begin_epoch(epoch)
        loss, _, _, _ = _pass(
            model,
            loader,
            criterion,
            device,
            optimizer=optimizer,
            mixup=mixup,
            sam=sam,
        )
        mixup.end_epoch()
        step = sam.end_epoch(mixup.epochs[-1])
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            if peak >= MEMORY_LIMIT:
                raise RuntimeError("SAM25 refit exceeded the 3 GB allocated GPU memory limit")
        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr,
                "train_mixed_loss": loss,
                "sam_perturbed_loss": step["second_loss"],
                "training_rows": step["rows"],
                "optimizer_steps": step["optimizer_steps"],
                "selected_checkpoint": epoch == config.epochs,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        atomic_write_json(run_dir / "mixup_training.json", mixup.receipt())
        atomic_write_json(run_dir / "sam_training.json", sam.receipt())
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


def run_gender_sam25_refit(*, output_root, registry_path=None, registry_mirrors=(), root=ROOT):
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
        # Never resume partial weights or overwrite the frozen five-model ensemble.
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
                    "sam_training.json",
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
                "selected_epoch": 25,
                "class_names": CLASSES,
                "files": files,
                "inference": "one model; saved development normalization; softmax then argmax",
                "validation_used": False,
                "holdout_evaluated": False,
                "teacher_test_evaluated": False,
                "metrics": metrics,
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
    result = run_gender_sam25_refit(
        root=args.root,
        output_root=args.output_root,
        registry_path=args.registry_path,
        registry_mirrors=args.registry_mirror,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
