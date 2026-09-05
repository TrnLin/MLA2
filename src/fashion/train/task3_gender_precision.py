"""Compare runtime-default and IEEE FP32 inference on the saved diagnostic samples."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_diagnostic import verify_checkpoint_metadata, verify_input_images

PRECISION_PATHS = (
    "backends",
    "backends.cuda.matmul",
    "backends.cudnn",
    "backends.cudnn.conv",
    "backends.cudnn.rnn",
)


def _backend(torch, path):
    value = torch
    for name in path.split("."):
        value = getattr(value, name)
    return value


def precision_settings(torch):
    return {path: _backend(torch, path).fp32_precision for path in PRECISION_PATHS}


@contextmanager
def ieee_precision(torch):
    """Use only the new PyTorch precision API; restore settings even after failure."""
    previous = precision_settings(torch)
    try:
        for path in PRECISION_PATHS:
            _backend(torch, path).fp32_precision = "ieee"
        if set(precision_settings(torch).values()) != {"ieee"}:
            raise RuntimeError("Could not enable IEEE FP32 for every backend")
        yield
    finally:
        for path, value in previous.items():
            _backend(torch, path).fp32_precision = value


def probability_difference(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or reference.ndim != 2 or not reference.size:
        raise ValueError("Expected matching nonempty probability matrices")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite probabilities")
    delta = float(np.max(np.abs(reference - actual)))
    flips = int(np.count_nonzero(reference.argmax(1) != actual.argmax(1)))
    return {
        "max_abs_difference": delta,
        "prediction_flips": flips,
        "pass": delta <= 1e-5 and flips == 0,
    }


def run_gender_precision(
    *, diagnostic_directory, g2_directory, compact_directory, registry_path, output, root
):
    """Read checkpoints and saved sample IDs; preserve the original diagnostic decision."""
    import torch

    from fashion.train.data import Task3ImageDataset
    from fashion.train.model import Task3CompactBlurCNN, Task3GeM3CNN
    from fashion.train.task3_baseline import set_reproducible_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Use a fresh Colab L4 GPU runtime; no training is needed")
    root, output, prior = Path(root), Path(output), Path(diagnostic_directory)
    old = json.loads((prior / "diagnostic_status.json").read_text())
    runtime = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    if runtime != old["runtime"]:
        raise RuntimeError(f"Match the earlier diagnostic runtime: {old['runtime']}; got {runtime}")
    if old["split_sha256"] != compute_sha256(root / "data/processed/splits.csv"):
        raise ValueError("Canonical split changed")
    expected = {(m, f) for m in ("G2", "CompactBlurCNN") for f in (0, 4)}
    if len(old["runs"]) != 4 or {(r["model"], r["fold"]) for r in old["runs"]} != expected:
        raise ValueError("Require exactly the four saved diagnostic sources")
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "running",
        "training_performed": False,
        "runtime": runtime,
        "source_diagnostic_sha256": compute_sha256(prior / "diagnostic_status.json"),
        "source_stability_sha256": compute_sha256(prior / "batch_order_stability.csv"),
        "module_sha256": compute_sha256(Path(__file__)),
        "split_sha256": old["split_sha256"],
        "runs": [],
        "historical_precision_settings_recorded": False,
        "baseline_note": "Runtime settings are recorded now; earlier settings were not saved.",
    }
    original_settings = precision_settings(torch)
    status["runtime_default_settings"] = original_settings
    rows = []

    def save():
        (output / "precision_status.json").write_text(json.dumps(status, indent=2))
        pd.DataFrame(rows).to_csv(output / "precision_comparisons.csv", index=False)

    save()
    try:
        splits = load_splits(root / "data/processed/splits.csv")
        classes = list(
            load_label_maps(root / "data/processed/label_maps.json")["gender"]["classes"]
        )
        registry = pd.read_csv(registry_path, keep_default_na=False)
        previous_rows = pd.read_csv(prior / "batch_order_stability.csv")
        for source in old["runs"]:
            name, fold, run_id = source["model"], source["fold"], source["run_id"]
            print(f"Precision check: {name}, fold {fold}", flush=True)
            directory = Path(g2_directory if name == "G2" else compact_directory) / run_id
            audit = inspect_gender_run(
                directory, registry=registry, splits=splits, classes=classes, root=root
            )
            if audit["sha256"] != source["source_sha256"]:
                raise ValueError("Source artifacts changed since the diagnostic")
            config_values = {f.name: audit["config"][f.name] for f in fields(Task3BaselineConfig)}
            config_values["channels"] = tuple(config_values["channels"])
            config = Task3BaselineConfig(**config_values)
            set_reproducible_seed(config.seed)
            status["execution_settings"] = {
                "seed": config.seed,
                "input_dtype": "float32",
                "autocast": False,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            }
            checkpoint = torch.load(
                directory / "final_epoch.pt", map_location="cpu", weights_only=True
            )
            stats = json.loads((directory / "normalization.json").read_text())
            verify_checkpoint_metadata(
                checkpoint,
                run_id=run_id,
                config=audit["config"],
                classes=classes,
                normalization=stats,
            )
            model = (Task3GeM3CNN if name == "G2" else Task3CompactBlurCNN)(config)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            model.cuda().eval().requires_grad_(False)
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.cuda.reset_peak_memory_stats()
            run_out = output / run_id
            run_out.mkdir()
            result = {
                "run_id": run_id,
                "model": name,
                "fold": fold,
                "source_sha256": audit["sha256"],
                "samples": [],
            }
            for partition, full_frame in zip(("train", "validation"), get_cv_split(splits, fold)):
                sample_path = prior / run_id / f"stability_{partition}_ids.csv"
                ids = pd.read_csv(sample_path).id
                frame = get_samples(full_frame, target="gender").set_index("id")
                if len(ids) != 160 or not ids.is_unique or not ids.isin(frame.index).all():
                    raise ValueError("Saved sample IDs do not match the canonical partition")
                frame = frame.loc[ids].reset_index()
                verify_input_images(frame, root)
                dataset = Task3ImageDataset(
                    frame,
                    target="gender",
                    label_to_index={c: i for i, c in enumerate(classes)},
                    mean=stats["mean"],
                    std=stats["std"],
                    root=root,
                    image_size=(80, 60),
                    image_view="full",
                )
                # Prepare once on CPU. Every mode/order uses these exact normalized tensors.
                tensors = torch.stack([dataset[i]["image"] for i in range(len(dataset))])
                tensor_hash = hashlib.sha256(tensors.numpy().tobytes()).hexdigest()
                tensors = tensors.cuda()
                orders = {
                    "forward": np.arange(len(ids)),
                    "reverse": np.arange(len(ids))[::-1].copy(),
                    "shuffle": pd.Series(np.arange(len(ids)))
                    .sample(frac=1, random_state=2753)
                    .to_numpy(),
                }
                saved_arrays = {"ids": ids.to_numpy()}

                def predict(batch_size, order, inference_model=model, images=tensors):
                    aligned = np.empty((len(ids), len(classes)), dtype=np.float32)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                        for start in range(0, len(order), batch_size):
                            indices = order[start : start + batch_size]
                            logits = inference_model(
                                images[torch.as_tensor(indices, device="cuda")]
                            )
                            aligned[indices] = torch.softmax(logits, dim=1).cpu().numpy()
                    return aligned

                def measure(mode, infer=predict):
                    reference = infer(128, orders["forward"])
                    saved_arrays[f"{mode}_reference"] = reference
                    for batch_size in (1, 32, 128):
                        for order_name, order in orders.items():
                            actual = infer(batch_size, order)
                            saved_arrays[f"{mode}_b{batch_size}_{order_name}"] = actual
                            comparison = probability_difference(reference, actual)
                            row = {
                                "run_id": run_id,
                                "partition": partition,
                                "mode": mode,
                                "batch_size": batch_size,
                                "order": order_name,
                                **comparison,
                            }
                            if mode == "runtime_default":
                                earlier = previous_rows[
                                    (previous_rows.run_id == run_id)
                                    & (previous_rows.partition == partition)
                                    & (previous_rows.batch_size == batch_size)
                                    & (previous_rows.order == order_name)
                                ]
                                if len(earlier) != 1:
                                    raise ValueError(
                                        "Missing or duplicate earlier stability comparison"
                                    )
                                earlier = earlier.iloc[0]
                                row["earlier_max_abs_difference"] = float(
                                    earlier.max_abs_difference
                                )
                                row["earlier_summary_reproduced"] = bool(
                                    abs(
                                        comparison["max_abs_difference"]
                                        - earlier.max_abs_difference
                                    )
                                    <= 1e-7
                                    and comparison["prediction_flips"] == earlier.prediction_flips
                                )
                            rows.append(row)

                measure("runtime_default")
                with ieee_precision(torch):
                    status["ieee_settings"] = precision_settings(torch)
                    measure("ieee")
                cross_mode = probability_difference(
                    saved_arrays["runtime_default_reference"], saved_arrays["ieee_reference"]
                )
                np.savez_compressed(run_out / f"{partition}_probabilities.npz", **saved_arrays)
                result["samples"].append(
                    {
                        "partition": partition,
                        "rows": len(ids),
                        "ids_sha256": compute_sha256(sample_path),
                        "prepared_tensor_sha256": tensor_hash,
                        "cross_mode_reference": cross_mode,
                    }
                )
                del predict, measure, tensors
            result["weights_and_buffers_unchanged"] = all(
                torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()
            )
            result["peak_allocated_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
            status["runs"].append(result)
            save()
            if not result["weights_and_buffers_unchanged"]:
                raise RuntimeError("Model state changed during inference")
            if not 0 < result["peak_allocated_gpu_bytes"] < 3_000_000_000:
                raise RuntimeError("Inference exceeded the strict 3 GB GPU memory limit")
            del model, before
            torch.cuda.empty_cache()
        status["settings_restored"] = precision_settings(torch) == original_settings
        status["ieee_all_comparisons_pass"] = all(r["pass"] for r in rows if r["mode"] == "ieee")
        status["earlier_summaries_reproduced"] = all(
            r["earlier_summary_reproduced"] for r in rows if r["mode"] == "runtime_default"
        )
        status["status"] = "complete_for_review"
        status["interpretation"] = (
            "Review saved probabilities; no automatic model acceptance, training, "
            "or change to the earlier review_required result."
        )
        save()
        return status
    except Exception as error:
        status.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise
