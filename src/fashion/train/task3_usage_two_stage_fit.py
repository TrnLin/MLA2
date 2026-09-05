"""Disposable CUDA worker for the fixed Usage two-stage screen.

Imported only inside the worker (or optional CPU unit tests), never by preflight.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn

from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count
from fashion.train.data import Task3ImageDataset, fit_fold_rgb_stats
from fashion.train.model import Task3BaselineCNN
from fashion.train.task3_baseline import (
    _loader,
    _pass,
    _prediction_frame,
    runtime_environment,
    set_reproducible_seed,
    validate_verified_colab_runtime,
)
from fashion.train.task3_usage_two_stage import (
    CLASSES,
    CORE_CORRUPTIONS,
    MEMORY_LIMIT,
    class_weights_for_training,
    recipe,
    save_predictions,
    training_scope,
    write_json,
)


def backbone_digest(model):
    """Hash all feature parameters AND buffers, including BatchNorm counters."""
    digest = hashlib.sha256()
    for name, tensor in model.features.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_and_reset_head(model, *, seed=2753):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    before = backbone_digest(model)
    set_reproducible_seed(seed)
    old = model.classifier
    model.classifier = nn.Linear(old.in_features, old.out_features).to(old.weight.device)
    nn.init.kaiming_uniform_(model.classifier.weight, a=math.sqrt(5))
    nn.init.zeros_(model.classifier.bias)
    return before


def cache_training_features(model, loader, training, device):
    """Cache only the ordered outer-training rows; no validation loader accepted."""
    model.eval()
    features, labels, ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            features.append(model.pool(model.features(batch["image"].to(device))).flatten(1).cpu())
            labels.append(batch["label"].cpu())
            ids.extend(batch["id"].tolist())
    if ids != training.id.astype(int).tolist() or len(ids) != len(set(ids)):
        raise ValueError("Feature cache does not exactly match ordered outer-training IDs")
    # Concatenation outside inference_mode produces ordinary tensors for head autograd.
    return torch.cat(features), torch.cat(labels), ids


def train_head_epoch(
    model, features, labels, criterion, optimizer, *, generator, batch_size, device
):
    """One shuffled visit per row; never switch the frozen backbone to train mode."""
    model.eval()
    model.classifier.train()
    numerator, denominator = 0.0, 0.0
    for indices in torch.randperm(len(labels), generator=generator).split(batch_size):
        target = labels[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model.classifier(features[indices].to(device)), target)
        loss.backward()
        optimizer.step()
        weight = float(criterion.weight[target].sum())
        numerator += float(loss.detach()) * weight
        denominator += weight
    return numerator / denominator


def fit(request):
    root, directory = Path(request["root"]), Path(request["directory"])
    run_id = request["run_id"]
    contract = json.loads((directory / "config.json").read_text())
    if contract["recipe"] != recipe() or run_id != directory.name:
        raise ValueError("Worker recipe or run identity changed")
    for name, digest in contract["code_sha256"].items():
        if compute_sha256(root / "src/fashion/train" / name) != digest:
            raise ValueError(f"Training code changed after registration: {name}")
    for name, key in (("splits.csv", "split_sha256"), ("label_maps.json", "label_map_sha256")):
        if compute_sha256(root / "data/processed" / name) != contract[key]:
            raise ValueError("Canonical data changed after registration")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a fresh Colab L4 GPU runtime for this screen")
    device = torch.device("cuda")
    environment = runtime_environment(device)
    validate_verified_colab_runtime(environment)
    if "L4" not in str(environment["gpu"]):
        raise RuntimeError("The fixed cost screen requires a Colab L4 GPU")
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_LIMIT / environment["vram_bytes"]))
    torch.cuda.reset_peak_memory_stats()
    environment["precision_flags"] = {
        "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
        "autocast": False,
    }
    config = Task3BaselineConfig(target="usage")
    set_reproducible_seed(config.seed)
    fold = contract["fold"]
    training, validation = training_scope(load_splits(root / "data/processed/splits.csv"), fold)

    def progress(stage, epoch=0):
        write_json({"stage": stage, "epoch": epoch}, directory / "progress.json")
        print(f"Usage fold {fold}: {stage} {epoch or ''}", flush=True)

    progress("training_only_normalization")
    for relative in validation.path:
        with Image.open(root / str(relative)) as image:
            image.verify()
    normalization = fit_fold_rgb_stats(training, root=root)
    normalization.update(
        {
            "fit_scope": "outer_training_only",
            "validation_fold": fold,
            "training_ids_sha256": hashlib.sha256(training.id.to_numpy().tobytes()).hexdigest(),
        }
    )
    write_json(normalization, directory / "normalization.json")
    mapping = dict(zip(CLASSES, range(len(CLASSES)), strict=True))

    def loader(frame, *, shuffle=False, corruption=None):
        data = Task3ImageDataset(
            frame,
            target="usage",
            label_to_index=mapping,
            mean=normalization["mean"],
            std=normalization["std"],
            root=root,
            corruption=corruption,
        )
        return _loader(data, config=config, shuffle=shuffle, device=device)

    train_loader = loader(training, shuffle=True)
    clean_train_loader, validation_loader = loader(training), loader(validation)
    model = Task3BaselineCNN(config).to(device)
    ordinary = nn.CrossEntropyLoss()
    history = []

    def optimizer_for(parameters, epochs):
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=config.minimum_learning_rate
        )
        return optimizer, scheduler

    def persist_prediction(data_loader, name):
        loss, labels, probabilities, trace = _pass(model, data_loader, ordinary, device)
        measured = save_predictions(
            _prediction_frame(labels, probabilities, trace, CLASSES, run_id), directory / name
        )
        return {**measured, "cross_entropy": loss}

    def checkpoint(name, stage, epoch):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "two_stage_contract": contract,
                "normalization": normalization,
                "classes": list(CLASSES),
                "run_id": run_id,
                "stage": stage,
                "epoch": epoch,
            },
            directory / name,
        )

    optimizer, scheduler = optimizer_for(model.parameters(), 30)
    started = time.monotonic()
    for epoch in range(1, 31):
        progress("stage_a", epoch)
        rate = optimizer.param_groups[0]["lr"]
        train_loss, _, _, _ = _pass(model, train_loader, ordinary, device, optimizer=optimizer)
        val_loss, _, _, _ = _pass(model, validation_loader, ordinary, device)
        scheduler.step()
        history.append(
            {
                "stage": "A",
                "epoch": epoch,
                "learning_rate": rate,
                "training_loss": train_loss,
                "validation_cross_entropy": val_loss,
            }
        )
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"loss={train_loss:.6f}; validation CE={val_loss:.6f}", flush=True)
    stage_a_seconds = time.monotonic() - started
    checkpoint("stage_a.pt", "A", 30)
    del optimizer, scheduler, train_loader
    progress("stage_a_clean_diagnostics")
    stage_a = {
        "training": persist_prediction(clean_train_loader, "stage_a_train_predictions.csv"),
        "validation": persist_prediction(validation_loader, "stage_a_oof_predictions.csv"),
    }
    write_json(stage_a, directory / "stage_a_metrics.json")
    progress("freeze_and_cache_training_features")
    frozen_hash = freeze_and_reset_head(model)
    features, labels, ids = cache_training_features(model, clean_train_loader, training, device)
    write_json(
        {
            "scope": "outer_training_only",
            "ids": ids,
            "product_family_groups": training.product_family_group.astype(str).tolist(),
            "shape": list(features.shape),
            "backbone_sha256_before": frozen_hash,
            "features_sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest(),
        },
        directory / "feature_cache_manifest.json",
    )
    counts, weights = class_weights_for_training(training)
    weighted = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer, scheduler = optimizer_for(model.classifier.parameters(), 10)
    generator = torch.Generator().manual_seed(2753)
    started = time.monotonic()
    for epoch in range(1, 11):
        progress("stage_b", epoch)
        rate = optimizer.param_groups[0]["lr"]
        train_loss = train_head_epoch(
            model,
            features,
            labels,
            weighted,
            optimizer,
            generator=generator,
            batch_size=config.batch_size,
            device=device,
        )
        val_loss, _, _, _ = _pass(model, validation_loader, ordinary, device)
        if backbone_digest(model) != frozen_hash:
            raise RuntimeError("Stage B changed a frozen feature parameter or BatchNorm buffer")
        scheduler.step()
        history.append(
            {
                "stage": "B",
                "epoch": epoch,
                "learning_rate": rate,
                "training_loss": train_loss,
                "validation_cross_entropy": val_loss,
            }
        )
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"weighted loss={train_loss:.6f}; validation CE={val_loss:.6f}", flush=True)
    stage_b_seconds = time.monotonic() - started
    checkpoint("final_epoch.pt", "B", 10)
    del optimizer, scheduler, features, labels
    progress("stage_b_clean_and_corruption_diagnostics")
    clean_training = persist_prediction(clean_train_loader, "clean_train_predictions.csv")
    metrics = persist_prediction(validation_loader, "oof_predictions.csv")
    (directory / "corruptions").mkdir()
    robust = []
    for name in CORE_CORRUPTIONS:
        progress(f"corruption_{name}")
        corrupt_loader = loader(validation, corruption=name)
        score = persist_prediction(corrupt_loader, f"corruptions/{name}.csv")
        del corrupt_loader
        robust.append(
            {
                "run_id": run_id,
                "validation_fold": fold,
                "corruption": name,
                "macro_f1": score["macro_f1"],
                "macro_f1_change": score["macro_f1"] - metrics["macro_f1"],
            }
        )
    pd.DataFrame(robust).to_csv(directory / "robustness.csv", index=False)
    metrics.update(
        {
            "clean_training": clean_training,
            "stage_a": stage_a,
            "final_train_eval_macro_f1": clean_training["macro_f1"],
            "train_val_macro_f1_gap": clean_training["macro_f1"] - metrics["macro_f1"],
            "train_seconds": stage_a_seconds + stage_b_seconds,
            "stage_a_seconds": stage_a_seconds,
            "stage_b_seconds": stage_b_seconds,
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
            "parameter_count": baseline_parameter_count("usage"),
            "stage_b_trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "backbone_unchanged": backbone_digest(model) == frozen_hash,
            "backbone_sha256_after": backbone_digest(model),
            "selected_stage": "B",
            "selected_epoch": 10,
            "class_counts": counts,
            "class_weights": weights,
            "environment": environment,
        }
    )
    write_json(metrics, directory / "metrics.json")
    progress("stage_b_and_diagnostics_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    fit(json.loads(parser.parse_args().request.read_text()))
