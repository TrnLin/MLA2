"""Refit the frozen teacher-only Usage E1 recipe on all eligible development rows."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from fashion.config import ROOT
from fashion.data import get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.artifacts import atomic_write_json
from fashion.train.config import Task3BaselineConfig
from fashion.train.data import Task3ImageDataset, fit_fold_rgb_stats
from fashion.train.model import Task3BaselineCNN
from fashion.train.registry import RunRegistry
from fashion.train.task3_baseline import _loader, _pass, runtime_environment, set_reproducible_seed

EXPERIMENT = "t3_usage_e1_teacher_all_development_refit"
SOURCE = Path("reports/task3/usage_final_e1_20260907/model_manifest.json")
SOURCE_SHA256 = "79cec64647d7c37f3bb64af1307e3911333c6198dce41bfc5171bf9f06d72a34"
CLASSES = ["Casual", "Ethnic", "Formal", "Home", "NA", "Party", "Smart Casual", "Sports", "Travel"]
EXPECTED_ROWS = 32772
ROW_COLUMNS = ["id", "cv_fold", "product_family_group", "usage", "sha256", "path"]
IMPLEMENTATION_FILES = (
    "train/task3_usage_e1_refit.py",
    "train/task3_baseline.py",
    "train/config.py",
    "train/model.py",
    "train/data.py",
    "train/augmentation.py",
    "train/losses.py",
    "train/registry.py",
    "train/task3_registry.py",
    "train/artifacts.py",
    "data/dataset.py",
    "data/images.py",
    "data/splits.py",
    "data/metadata.py",
    "data/hashing.py",
    "config.py",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_refit(*, root=ROOT):
    """Read-only preflight: verify frozen recipe, canonical population and image bytes."""
    root = Path(root)
    if compute_sha256(root / SOURCE) != SOURCE_SHA256:
        raise ValueError("Frozen E1 manifest changed")
    source = json.loads((root / SOURCE).read_text())
    for name in (
        "data/processed/splits.csv",
        "data/processed/label_maps.json",
        "src/fashion/data/images.py",
        "src/fashion/train/model.py",
        "src/fashion/train/config.py",
    ):
        if compute_sha256(root / name) != source["source_contracts"][name]["sha256"]:
            raise ValueError(f"Frozen E1 contract changed: {name}")
    recipe = source["folds"][0]["config"]
    if (
        source["experiment_id"] != "t3_primary_baseline_smallcnn"
        or source["class_names"] != CLASSES
        or [f["fold"] for f in source["folds"]] != list(range(5))
        or any(f["config"] != recipe for f in source["folds"])
        or recipe != Task3BaselineConfig(target="usage").to_dict()
    ):
        raise ValueError("E1 requires the identical frozen 30-epoch recipe in all five folds")
    values = {f.name: recipe[f.name] for f in fields(Task3BaselineConfig)}
    values["channels"] = tuple(values["channels"])
    config = Task3BaselineConfig(**values)
    splits = load_splits(root / "data/processed/splits.csv")
    training = get_samples(splits, partition="development", target="usage").reset_index(drop=True)
    protected = splits.loc[~splits.partition.eq("development")]
    if (
        len(training) != EXPECTED_ROWS
        or training.id.duplicated().any()
        or set(training.cv_fold) != set(range(5))
        or set(training.id) & set(protected.id)
        or set(training.product_family_group) & set(protected.product_family_group)
        or training.groupby("product_family_group").cv_fold.nunique().max() != 1
        or not training.path.str.startswith("data/raw/teacher/train/images_train/").all()
    ):
        raise ValueError("Usage refit population must be all eligible teacher development rows")
    class_map = load_label_maps(root / "data/processed/label_maps.json")["usage"]
    if (
        class_map["classes"] != CLASSES
        or class_map["label_to_index"] != {name: i for i, name in enumerate(CLASSES)}
        or set(training.usage) != set(CLASSES)
    ):
        raise ValueError("Usage class order or support changed")
    print(f"Checking {len(training):,} development image hashes", flush=True)
    for row in training.itertuples():
        image = (root / row.path).resolve()
        # Canonical paths may use the existing shared image-directory symlink.
        # The pinned split and each file's digest, rather than physical location, define identity.
        if compute_sha256(image) != row.sha256:
            raise ValueError(f"Input image differs from canonical split: id={row.id}")
    payload = {
        **config.to_dict(),
        "experiment_id": EXPERIMENT,
        "training_scope": "all_eligible_teacher_development",
        "validation_fold": None,
        "validation_used": False,
        "source_manifest": str(SOURCE),
        "source_manifest_sha256": SOURCE_SHA256,
        "source_recipe_sha256": _digest(recipe),
        "source_fold_config_sha256": [f["files"]["config.json"]["sha256"] for f in source["folds"]],
        "source_role": "recipe_only_no_trained_weights_loaded",
        "split_sha256": compute_sha256(root / "data/processed/splits.csv"),
        "label_map_sha256": compute_sha256(root / "data/processed/label_maps.json"),
        "training_rows_sha256": hashlib.sha256(
            training[ROW_COLUMNS].to_csv(index=False).encode()
        ).hexdigest(),
        "training_rows": len(training),
        "training_class_counts": training.usage.value_counts().to_dict(),
        "class_names": CLASSES,
        "parameter_count": source["architecture"]["parameters_per_model"],
        "classifier_dropout": 0.0,
        "input_operations": source["input"]["operations"][:4]
        + ["Apply saved full-development RGB mean/std; zero letterbox padding; HWC to CHW"],
        "implementation_sha256": {
            name: compute_sha256(root / "src/fashion" / name) for name in IMPLEMENTATION_FILES
        },
    }
    return config, payload, training


def _cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("Use a GPU runtime for the final E1 refit; --preflight needs no GPU")
    return torch.device("cuda")


def _fit(config, payload, training, *, root, destination, registry, run_id, device):
    stats = fit_fold_rgb_stats(training, root=root)
    atomic_write_json(
        destination / "normalization.json",
        {
            **stats,
            "fit_scope": "all_eligible_teacher_development_content_pixels",
            "padding_excluded": True,
            "training_rows_sha256": payload["training_rows_sha256"],
        },
    )
    dataset = Task3ImageDataset(
        training,
        target="usage",
        root=root,
        label_to_index={name: i for i, name in enumerate(CLASSES)},
        mean=stats["mean"],
        std=stats["std"],
    )
    loader = _loader(dataset, config=config, shuffle=True, device=device)
    model = Task3BaselineCNN(config).to(device)
    if sum(p.numel() for p in model.parameters()) != payload["parameter_count"]:
        raise ValueError("Expected the frozen 391,209-parameter E1 SmallCNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.minimum_learning_rate
    )
    history = []
    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        seen = []

        def batches():
            # The shared training pass records traces only for evaluation.
            # Observe the actual training batches without changing their order or RNG.
            for batch in loader:
                seen.extend(batch["id"].tolist())
                yield batch

        loss, _, _, _ = _pass(model, batches(), criterion, device, optimizer=optimizer)
        if len(seen) != len(training) or set(seen) != set(training.id):
            raise ValueError("Each admitted development image must train exactly once per epoch")
        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": loss,
                "training_rows": len(seen),
                "selected_checkpoint": epoch == config.epochs,
            }
        )
        pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
        scheduler.step()
        registry.update(run_id, {"last_completed_stage": f"epoch_{epoch}_complete"})
        print(f"Epoch {epoch}/{config.epochs}: training loss {loss:.4f}", flush=True)
    checkpoint = {
        "run_id": run_id,
        "config": payload,
        "class_names": CLASSES,
        "normalization": stats,
        "checkpoint_policy": "final_epoch",
        "epochs_completed": config.epochs,
        "selected_epoch": config.epochs,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    temporary = destination / "final_epoch.pt.partial"
    torch.save(checkpoint, temporary)
    temporary.replace(destination / "final_epoch.pt")
    return {
        "training_rows": len(training),
        "epochs_completed": config.epochs,
        "train_loss": history[-1]["train_loss"],
        "validation_rows": 0,
        "validation_used": False,
        "holdout_evaluated": False,
        "teacher_test_evaluated": False,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }


def _verify_completed(destination, payload, registry_paths):
    manifest = json.loads((destination / "model_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["config_hash"] != _digest(payload):
        raise ValueError(
            "Existing refit is incomplete or has different code/data; inspect it first"
        )
    for record in manifest["files"].values():
        path = (destination / record["path"]).resolve()
        if (
            not path.is_relative_to(destination.resolve())
            or compute_sha256(path) != record["sha256"]
        ):
            raise ValueError("Completed refit artifact changed")
    for path in registry_paths:
        rows = pd.read_csv(path, keep_default_na=False)
        selected = rows.loc[rows.run_id.eq(manifest["run_id"])]
        if len(selected) != 1 or selected.iloc[0].status != "complete":
            raise ValueError("Completed refit needs its unique completed registry row and mirrors")
        row = selected.iloc[0]
        if (
            row.config_hash != manifest["config_hash"]
            or row.checkpoint_sha256 != manifest["files"]["final_epoch.pt"]["sha256"]
        ):
            raise ValueError("Completed refit registry identity changed")
    return dict(manifest, manifest_path=str(destination / "model_manifest.json"), reused=True)


def run_usage_e1_refit(*, output_root, registry_path=None, registry_mirrors=(), root=ROOT):
    """Fit once. Reruns verify completed outputs; interrupted outputs stop for inspection."""
    root = Path(root)
    registry_path = Path(registry_path) if registry_path else root / "results/runs.csv"
    config, payload, training = prepare_refit(root=root)
    destination = Path(output_root) / "experiments" / EXPERIMENT / "usage"
    destination.mkdir(parents=True, exist_ok=True)
    # Lock the experiment beside the authoritative registry, also across output directories.
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with (registry_path.parent / f".{EXPERIMENT}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Usage E1 refit is active") from exc
        manifest_path = destination / "model_manifest.json"
        if manifest_path.exists():
            return _verify_completed(destination, payload, [registry_path, *registry_mirrors])
        for path in (registry_path, *map(Path, registry_mirrors)):
            if path.exists():
                rows = pd.read_csv(path, keep_default_na=False)
                if "experiment_id" in rows and rows.experiment_id.eq(EXPERIMENT).any():
                    raise ValueError(
                        "Usage E1 refit already registered; inspect its original output"
                    )
        if any(destination.iterdir()):
            raise ValueError("Refit output is not empty; inspect it before any new fit")
        device = _cuda_device()
        run_id = f"{EXPERIMENT}_{_digest(payload)[:16]}"
        environment = runtime_environment(device)
        environment["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        environment["git_status_short"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        )
        environment["torch_native_extension"] = torch._C.__file__
        atomic_write_json(destination / "environment.json", environment)
        atomic_write_json(destination / "config.json", payload)
        atomic_write_json(
            destination / "class_map.json",
            {"classes": CLASSES, "label_to_index": {name: i for i, name in enumerate(CLASSES)}},
        )
        training[ROW_COLUMNS].to_csv(destination / "training_rows.csv", index=False)
        registry = RunRegistry(registry_path, mirrors=registry_mirrors)
        registry.start(
            {
                "run_id": run_id,
                "experiment_id": EXPERIMENT,
                "task": "task3",
                "target": "usage",
                "validation_fold": None,
                "seed": config.seed,
                "scratch": True,
                "debug": False,
                "submission_eligible": True,
                "config_hash": _digest(payload),
                "config_path": destination / "config.json",
                "split_digest": payload["split_sha256"],
                "label_map_digest": payload["label_map_sha256"],
                "training_product_count": len(training),
                "validation_product_count": 0,
                "training_family_count": training.product_family_group.nunique(),
                "validation_family_count": 0,
                "model_family": config.model_family,
                "parameter_count": payload["parameter_count"],
                "environment_json": environment,
                "history_path": destination / "history.csv",
                "last_completed_stage": "registered",
            }
        )
        manifest = {
            "status": "running",
            "training_completed": False,
            "evaluation_completed": False,
            "holdout_evaluated": False,
            "teacher_test_evaluated": False,
            "new_blind_evaluation": False,
            "experiment_id": EXPERIMENT,
            "run_id": run_id,
            "config_hash": _digest(payload),
            "artifact_type": "single_all_development_scratch_refit",
            "class_names": CLASSES,
            "selected_epoch": config.epochs,
            "validation_used": False,
            "inference": (
                "one model in eval mode; saved development normalization; softmax then argmax"
            ),
            "decision": "docs/task3_usage_e1_refit.md",
            "files": {},
        }
        started = time.perf_counter()
        try:
            atomic_write_json(manifest_path, manifest)
            set_reproducible_seed(config.seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            metrics = _fit(
                config,
                payload,
                training,
                root=root,
                destination=destination,
                registry=registry,
                run_id=run_id,
                device=device,
            )
            metrics["train_seconds"] = time.perf_counter() - started
            atomic_write_json(destination / "metrics.json", metrics)
            files = {
                name: {"path": name, "sha256": compute_sha256(destination / name)}
                for name in (
                    "final_epoch.pt",
                    "config.json",
                    "normalization.json",
                    "history.csv",
                    "training_rows.csv",
                    "class_map.json",
                    "environment.json",
                    "metrics.json",
                )
            }
            registry.complete(
                run_id,
                {
                    "checkpoint_path": destination / "final_epoch.pt",
                    "checkpoint_sha256": files["final_epoch.pt"]["sha256"],
                    "checkpoint_bytes": (destination / "final_epoch.pt").stat().st_size,
                    "metrics_json": metrics,
                    "train_seconds": metrics["train_seconds"],
                    "peak_memory_bytes": metrics["peak_memory_bytes"],
                    "last_completed_stage": "refit_artifacts_complete",
                },
            )
        except BaseException as exc:
            registry.fail(run_id, exc, last_completed_stage="refit_failed_inspect_history")
            atomic_write_json(manifest_path, dict(manifest, status="failed", error=str(exc)))
            raise
        # If publication fails after registry completion, reruns stop instead of training again.
        manifest.update(status="complete", training_completed=True, files=files, metrics=metrics)
        atomic_write_json(manifest_path, manifest)
        return dict(manifest, manifest_path=str(manifest_path), reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--registry-mirror", type=Path, action="append", default=[])
    parser.add_argument("--preflight", action="store_true", help="Check inputs only; write nothing")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(prepare_refit(root=args.root)[1], indent=2))
    else:
        if args.output_root is None:
            parser.error("--output-root is required for training")
        print(
            json.dumps(
                run_usage_e1_refit(
                    root=args.root,
                    output_root=args.output_root,
                    registry_path=args.registry_path,
                    registry_mirrors=args.registry_mirror,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
