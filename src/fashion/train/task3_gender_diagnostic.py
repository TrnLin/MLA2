"""Inference-only clean-gap and batch/order diagnostics for saved gender models."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_g2_audit import inspect_gender_run


def verify_input_images(frame, root):
    """Require the exact development image bytes recorded in the canonical split."""
    checked = 0
    for row in frame[["id", "path", "sha256"]].itertuples(index=False):
        path = Path(root) / row.path
        if not path.is_file() or compute_sha256(path) != row.sha256:
            raise ValueError(f"Input image differs from canonical split: id={row.id}, {path}")
        checked += 1
    return checked


def verify_checkpoint_metadata(checkpoint, *, run_id, config, classes, normalization):
    """Compare the trainer's checkpoint fields with their audited JSON equivalents."""
    # normalization.json adds scope notes to StreamingStats.to_dict(); the
    # checkpoint contains only the original statistics returned by that method.
    expected = {
        "run_id": run_id,
        "config": config,
        "class_names": classes,
        "normalization": {
            key: normalization[key] for key in ("channels", "mean", "std", "total_pixels")
        },
    }
    for field, value in expected.items():
        if field not in checkpoint or checkpoint[field] != value:
            raise ValueError(
                f"Checkpoint metadata differs from verified source: {run_id}, field={field}"
            )


def run_gender_diagnostic(*, g2_directory, compact_directory, registry_path, output, root):
    """Read four verified checkpoints; never fit, alter weights, or write the registry."""
    import torch
    from torch.utils.data import DataLoader

    from fashion.train.data import Task3ImageDataset
    from fashion.train.metrics import classification_metrics
    from fashion.train.model import Task3CompactBlurCNN, Task3GeM3CNN
    from fashion.train.task3_baseline import _pass, _prediction_frame, set_reproducible_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Use a Colab GPU. This diagnostic does not train models.")
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "training_performed": False, "runs": []}
    status["runtime"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    status["diagnostic_source_sha256"] = compute_sha256(Path(__file__))
    status["split_sha256"] = compute_sha256(root / "data/processed/splits.csv")

    def save_status():
        (output / "diagnostic_status.json").write_text(json.dumps(status, indent=2))

    save_status()
    try:
        registry = pd.read_csv(registry_path, keep_default_na=False)
        splits = load_splits(root / "data/processed/splits.csv")
        classes = list(
            load_label_maps(root / "data/processed/label_maps.json")["gender"]["classes"]
        )
        development = get_samples(splits, partition="development", target="gender").copy()
        print("Verifying development image contents against splits.csv", flush=True)
        status["verified_input_images"] = verify_input_images(development, root)
        save_status()
        sizes = development.groupby("product_family_group").size()
        development["family_size"] = development.product_family_group.map(sizes)
        development["family_size_bin"] = pd.cut(
            development.family_size, [0, 1, 3, np.inf], labels=["1", "2-3", "4+"]
        ).astype(str)
        metadata = development[["id", "gender", "articleType", "family_size", "family_size_bin"]]
        sources = []
        for name, directory, constructor, family in (
            ("G2", g2_directory, Task3GeM3CNN, "task3_small_cnn_gem_p3"),
            ("CompactBlurCNN", compact_directory, Task3CompactBlurCNN, "task3_compact_blur_cnn"),
        ):
            found = set()
            for path in sorted(Path(directory).glob("t3_*")):
                if not (path / "metrics.json").is_file():
                    continue
                fold = json.loads((path / "metrics.json").read_text())["validation_fold"]
                if fold not in (0, 4):
                    continue
                if fold in found:
                    raise ValueError(f"Multiple sources for {name} fold {fold}")
                found.add(fold)
                audit = inspect_gender_run(
                    path, registry=registry, splits=splits, classes=classes, root=root
                )
                if audit["config"]["effective_model_family"] != family:
                    raise ValueError("Unexpected source model architecture")
                sources.append((name, path, constructor, audit))
            if found != {0, 4}:
                raise ValueError(f"Missing {name} folds 0 and 4")

        all_slices, stability, class_scores = [], [], []
        for name, path, constructor, audit in sources:
            print(f"Checking {name}, fold {audit['fold']} (inference only)", flush=True)
            payload = audit["config"]
            values = {f.name: payload[f.name] for f in fields(Task3BaselineConfig)}
            values["channels"] = tuple(values["channels"])
            config = Task3BaselineConfig(**values)
            set_reproducible_seed(config.seed)
            checkpoint = torch.load(path / "final_epoch.pt", map_location="cpu", weights_only=True)
            stats = json.loads((path / "normalization.json").read_text())
            verify_checkpoint_metadata(
                checkpoint,
                run_id=audit["run_id"],
                config=payload,
                classes=classes,
                normalization=stats,
            )
            model = constructor(config)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            device = torch.device("cuda")
            model.to(device).eval()
            model.requires_grad_(False)
            torch.cuda.reset_peak_memory_stats()
            before = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            train, validation = get_cv_split(splits, audit["fold"])
            frames = {
                "train": get_samples(train, target="gender"),
                "validation": get_samples(validation, target="gender"),
            }
            run_output = output / audit["run_id"]
            run_output.mkdir()
            probability_columns = [f"probability_{i}_{c}" for i, c in enumerate(classes)]

            def predict(frame, batch_size, inference_model=model):
                dataset = Task3ImageDataset(
                    frame,
                    target="gender",
                    label_to_index={c: i for i, c in enumerate(classes)},
                    mean=stats["mean"],
                    std=stats["std"],
                    root=root,
                    image_size=(config.image_height, config.image_width),
                    image_view="full",
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
                _, labels, probabilities, trace = _pass(
                    inference_model, loader, torch.nn.CrossEntropyLoss(), device
                )
                return _prediction_frame(labels, probabilities, trace, classes, audit["run_id"])

            checks = []
            for partition, frame in frames.items():
                rows = predict(frame, 128)
                score = classification_metrics(
                    rows.true_index.to_numpy(), rows[probability_columns].to_numpy(), classes
                )
                for class_score in score["per_class"]:
                    class_scores.append(
                        {
                            "model": name,
                            "fold": audit["fold"],
                            "partition": partition,
                            **class_score,
                        }
                    )
                expected = audit["metrics"][
                    "macro_f1" if partition == "validation" else "final_train_eval_macro_f1"
                ]
                checks.append(
                    {
                        "check": f"{partition}_macro_f1",
                        "actual": score["macro_f1"],
                        "expected": expected,
                        "pass": bool(abs(score["macro_f1"] - expected) <= 1e-6),
                    }
                )
                if partition == "validation":
                    saved = audit["predictions"].set_index("id").loc[rows.id]
                    delta = float(
                        np.max(
                            np.abs(
                                saved[probability_columns].to_numpy()
                                - rows[probability_columns].to_numpy()
                            )
                        )
                    )
                    checks.append(
                        {
                            "check": "saved_validation_probabilities",
                            "max_abs_difference": delta,
                            "pass": delta <= 1e-5,
                        }
                    )
                rows = rows.merge(metadata, on="id", validate="one_to_one")
                rows.to_csv(run_output / f"clean_{partition}_predictions.csv", index=False)
                rows["correct"] = rows.true_index.eq(rows.predicted_index)
                for dims in (
                    ("gender",),
                    ("articleType",),
                    ("family_size_bin",),
                    ("gender", "articleType"),
                    ("gender", "family_size_bin"),
                ):
                    for keys, group in rows.groupby(list(dims), dropna=False):
                        keys = keys if isinstance(keys, tuple) else (keys,)
                        all_slices.append(
                            {
                                "model": name,
                                "fold": audit["fold"],
                                "partition": partition,
                                "dimensions": " x ".join(dims),
                                "slice": " | ".join(map(str, keys)),
                                "rows": len(group),
                                "families": group.product_family_group.nunique(),
                                "accuracy": float(group.correct.mean()),
                            }
                        )
                # Fixed class-balanced sample, from each partition, tests batch size and order.
                sample = pd.concat(
                    [
                        g.sort_values("id").sample(n=min(32, len(g)), random_state=2753)
                        for _, g in frame.groupby("gender")
                    ]
                ).reset_index(drop=True)
                reference = predict(sample, 128).set_index("id").sort_index()
                for batch_size in (1, 32, 128):
                    for order in ("forward", "reverse", "shuffle"):
                        ordered = (
                            sample
                            if order == "forward"
                            else (
                                sample.iloc[::-1]
                                if order == "reverse"
                                else sample.sample(frac=1, random_state=2753)
                            )
                        )
                        actual = (
                            predict(ordered.reset_index(drop=True), batch_size)
                            .set_index("id")
                            .sort_index()
                        )
                        delta = float(
                            np.max(
                                np.abs(
                                    reference[probability_columns].to_numpy()
                                    - actual[probability_columns].to_numpy()
                                )
                            )
                        )
                        flips = int((reference.predicted_index != actual.predicted_index).sum())
                        stability.append(
                            {
                                "run_id": audit["run_id"],
                                "partition": partition,
                                "sample_rows": len(sample),
                                "batch_size": batch_size,
                                "order": order,
                                "max_abs_difference": delta,
                                "prediction_flips": flips,
                                "pass": delta <= 1e-5 and flips == 0,
                            }
                        )
                sample[["id"]].to_csv(run_output / f"stability_{partition}_ids.csv", index=False)
            unchanged = all(
                torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()
            )
            checks.append({"check": "weights_and_batchnorm_buffers_unchanged", "pass": unchanged})
            peak = int(torch.cuda.max_memory_allocated())
            checks.append(
                {
                    "check": "gpu_memory_under_3gb",
                    "peak_bytes": peak,
                    "pass": 0 < peak < 3_000_000_000,
                }
            )
            run_stability = [row for row in stability if row["run_id"] == audit["run_id"]]
            status["runs"].append(
                {
                    "model": name,
                    "fold": audit["fold"],
                    "run_id": audit["run_id"],
                    "source_sha256": audit["sha256"],
                    "checks": checks,
                    "pass": all(c["pass"] for c in checks + run_stability),
                }
            )
            pd.DataFrame(all_slices).to_csv(output / "clean_slices.csv", index=False)
            pd.DataFrame(class_scores).to_csv(output / "clean_class_scores.csv", index=False)
            pd.DataFrame(stability).to_csv(output / "batch_order_stability.csv", index=False)
            save_status()
            del predict, model, before
            torch.cuda.empty_cache()
        slices = pd.DataFrame(all_slices)
        gaps = slices.pivot(
            index=["model", "fold", "dimensions", "slice"], columns="partition", values="accuracy"
        )
        gaps["accuracy_gap"] = gaps["train"] - gaps["validation"]
        gaps.to_csv(output / "slice_accuracy_gaps.csv")
        status["status"] = "pass" if all(r["pass"] for r in status["runs"]) else "review_required"
        status["interpretation"] = (
            "Diagnostic checks only; no model acceptance or permission for more training."
        )
        save_status()
        return status
    except Exception as error:
        status.update(status="error", error=f"{type(error).__name__}: {error}")
        save_status()
        raise
