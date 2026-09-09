"""Saved-checkpoint diagnostics and zero-step GPU prerequisites for G-D1."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
from PIL import ImageEnhance

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import oof_metrics, probability_columns, validate_oof
from fashion.train.task3_gender_repair import MEMORY_LIMIT, NAME, load_sources

BRIGHTNESS_FACTORS = (0.85, 0.90, 0.95, 1.00)
SHIFTS = tuple((x, y) for x in range(-2, 3) for y in range(-2, 3))
DIAGNOSTICS = (
    ("clean",)
    + tuple(f"brightness_{v:.2f}" for v in BRIGHTNESS_FACTORS)
    + tuple(f"shift_{x}_{y}" for x, y in SHIFTS)
)


def _json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _code_hashes(root):
    names = (
        "augmentation.py",
        "data.py",
        "task3_baseline.py",
        "task3_dataset_v2.py",
        "task3_gender_repair.py",
        "task3_gender_repair_preflight.py",
    )
    return {
        f"src/fashion/train/{name}": compute_sha256(root / "src/fashion/train" / name)
        for name in names
    }


def _cuda_environment(device_name):
    import torch

    from fashion.train.task3_baseline import runtime_environment

    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G-D1 requires CUDA: CPU cannot pass the GPU memory prerequisite")
    env = runtime_environment(torch.device(device_name))
    return {key: env[key] for key in ("gpu", "torch", "cuda_runtime", "vram_bytes")}


def _data_hashes(root):
    return {
        name: compute_sha256(root / name)
        for name in ("data/processed/splits.csv", "data/processed/label_maps.json")
    }


def memory_profile_passes(profile):
    """A GPU-only, strict decimal 3 GB cap; speed is recorded without a limit."""
    peak = profile.get("peak_memory_bytes")
    return (
        profile.get("execution_policy") == "gpu_fp32_batch128_memory_under_3gb_no_speed_cap_v2"
        and profile.get("device_type") == "cuda"
        and profile.get("batch_size") == 128
        and profile.get("optimizer_steps") == 0
        and isinstance(peak, (int, float))
        and np.isfinite(peak)
        and 0 < peak < MEMORY_LIMIT
    )


def profile_gpu(images, labels):
    """Probe normal GPU execution with AdamW reserves and zero optimizer steps."""
    import torch

    from fashion.train.config import Task3BaselineConfig
    from fashion.train.model import Task3GeM3CNN

    device = images.device
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = Task3GeM3CNN(Task3BaselineConfig(target="gender")).to(device)
    model.train()
    reserves = [torch.zeros_like(p) for p in model.parameters() for _ in range(3)]
    timings = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(4):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        del logits, loss
    report = {
        "execution_policy": "gpu_fp32_batch128_memory_under_3gb_no_speed_cap_v2",
        "device_type": device.type,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "forward_backward_seconds": float(np.median(timings[1:])),
        "speed_cap": None,
        "memory_limit_bytes": MEMORY_LIMIT,
        "batch_size": len(images),
        "optimizer_steps": 0,
        "adamw_memory_reserve": "three parameter-sized tensors",
    }
    report["passed"] = bool(memory_profile_passes(report))
    del model, reserves
    return report


def _predict_view(model, dataset, classes, run_id, view, device):
    """Deterministic image-only perturbations; original fold RGB statistics."""
    import torch

    from fashion.data.images import transform_image
    from fashion.train.data import _load_corrupted
    from fashion.train.task3_clean_slate import _prediction_frame

    probabilities = []
    for start in range(0, len(dataset.frame), 128):
        tensors = []
        for record in dataset.frame.iloc[start : start + 128].to_dict("records"):
            image = _load_corrupted(dataset.root / record["path"], None)
            if view.startswith("brightness_"):
                image = ImageEnhance.Brightness(image).enhance(float(view.split("_")[1]))
            elif view.startswith("shift_"):
                from PIL import Image

                _, x, y = view.split("_")
                image = image.transform(
                    image.size,
                    Image.Transform.AFFINE,
                    (1, 0, -int(x), 0, 1, -int(y)),
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=(255, 255, 255),
                )
            array = transform_image(
                image, image_size=dataset.image_size, mean=dataset.mean, std=dataset.std
            )
            tensors.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
        with torch.inference_mode():
            probabilities.append(
                torch.softmax(model(torch.stack(tensors).to(device)), dim=1).cpu().numpy()
            )
    return _prediction_frame(
        dataset.frame,
        target="gender",
        classes=classes,
        probabilities=np.concatenate(probabilities),
        run_id=run_id,
    )


def prepare_gender_repair(
    *,
    g2_directory,
    e6_directory,
    registry_path,
    report_directory,
    root=ROOT,
    device_name="cuda",
):
    """Run no training: reproduce clean OOF, audit perturbations, probe GPU memory."""
    import torch

    from fashion.train.config import Task3BaselineConfig
    from fashion.train.data import Task3ImageDataset
    from fashion.train.model import Task3GeM3CNN
    from fashion.train.task3_baseline import set_reproducible_seed

    root, report_directory = Path(root), Path(report_directory)
    environment = _cuda_environment(device_name)
    sources, classes = load_sources(
        g2_directory=g2_directory, e6_directory=e6_directory, registry_path=registry_path, root=root
    )
    # Keep the saved-source runtime for reproducible checkpoint diagnostics.
    for run in sources["G2"].values():
        for key in ("gpu", "torch", "cuda_runtime"):
            if run["environment"][key] != environment[key]:
                raise RuntimeError(f"G-D1 must use the G2 comparison runtime: {key}")
    spec = dataset_v2_spec(NAME, [sources["G2"][f]["run_id"] for f in range(5)])
    destination = report_directory / "prerequisites.json"
    if destination.exists():
        verify_prerequisites(destination, root=root, spec=spec, device_name=device_name)
        return json.loads(destination.read_text())
    report_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    set_reproducible_seed(2753)
    splits = load_splits(root / "data/processed/splits.csv")
    source_hashes, artifact_hashes, clean_checks = {}, {}, []
    # Fail the cheap memory/time check before the full saved-checkpoint sweep.
    stats = json.loads((Path(sources["G2"][0]["directory"]) / "normalization.json").read_text())
    training, _ = get_cv_split(splits, 0)
    training = get_samples(training, target="gender").sort_values("id").head(128)
    probe_dataset = Task3ImageDataset(
        training,
        target="gender",
        label_to_index={c: i for i, c in enumerate(classes)},
        mean=stats["mean"],
        std=stats["std"],
        root=root,
    )
    batch = [probe_dataset[i] for i in range(len(probe_dataset))]
    probe_input = torch.stack([b["image"] for b in batch]).to(device)
    probe_labels = torch.tensor([b["label"] for b in batch], device=device)
    profile = profile_gpu(probe_input, probe_labels)
    _json_write(profile, report_directory / "memory_profile.json")
    del probe_input, probe_labels, probe_dataset, batch
    if not profile["passed"]:
        raise RuntimeError("G-D1 GPU memory prerequisite failed; stop before training")
    summaries = []
    for group in ("E6", "G2"):
        for fold, run in sources[group].items():
            directory = Path(run["directory"])
            for name, digest in run["sha256"].items():
                source_hashes[str(directory / name)] = digest
            checkpoint = torch.load(
                directory / "final_epoch.pt", map_location="cpu", weights_only=True
            )
            if checkpoint["run_id"] != run["run_id"] or checkpoint["config"] != run["config"]:
                raise ValueError("checkpoint identity/config differs from the verified source")
            if checkpoint["class_names"] != classes:
                raise ValueError("checkpoint classes differ from canonical Gender labels")
            stats = json.loads((directory / "normalization.json").read_text())
            if any(stats.get(key) != value for key, value in checkpoint["normalization"].items()):
                raise ValueError("checkpoint normalisation differs from the saved source")
            _, validation = get_cv_split(splits, fold)
            validation = get_samples(validation, target="gender")
            options = dict(
                target="gender",
                label_to_index={c: i for i, c in enumerate(classes)},
                mean=stats["mean"],
                std=stats["std"],
                root=root,
            )
            dataset = Task3ImageDataset(validation, **options)
            model = Task3GeM3CNN(Task3BaselineConfig(target="gender")).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            clean = None
            for view in DIAGNOSTICS:
                path = report_directory / run["run_id"] / (view + ".csv")
                frame = _predict_view(model, dataset, classes, run["run_id"], view, device)
                frame = validate_oof(
                    frame,
                    validation,
                    target="gender",
                    classes=classes,
                    run_ids_by_fold={fold: run["run_id"]},
                )
                if view == "clean":
                    expected = run["predictions"].sort_values("id")
                    matches = np.allclose(
                        frame[probability_columns(classes)],
                        expected[probability_columns(classes)],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                    matches = bool(
                        matches and np.array_equal(frame.predicted_index, expected.predicted_index)
                    )
                    clean_checks.append({"run_id": run["run_id"], "passed": matches})
                    if not matches:
                        raise RuntimeError(
                            "clean checkpoint inference does not reproduce saved OOF"
                        )
                    clean = frame
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".csv.tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(path)
                artifact_hashes[str(path.resolve())] = compute_sha256(path)
                metrics = oof_metrics(frame, classes)
                summaries.append(
                    {
                        "group": group,
                        "fold": fold,
                        "view": view,
                        "macro_f1": metrics["macro_f1"],
                        "nll": metrics["nll"],
                        "prediction_flips": int(
                            frame.predicted_index.ne(clean.predicted_index).sum()
                        ),
                        "per_class": metrics["per_class"],
                    }
                )
                print(
                    f"[gender-repair] {group} fold {fold} {view}: {metrics['macro_f1']:.6f}",
                    flush=True,
                )
            del model, checkpoint, dataset
            gc.collect()
            torch.cuda.empty_cache()
    summary_path = report_directory / "diagnostic_summary.json"
    _json_write(summaries, summary_path)
    artifact_hashes[str(summary_path.resolve())] = compute_sha256(summary_path)
    report = {
        "version": 2,
        "ready": profile["passed"],
        "spec": spec.to_dict(),
        "environment": environment,
        "data_sha256": _data_hashes(root),
        "code_sha256": _code_hashes(root),
        "source_sha256": source_hashes,
        "diagnostic_sha256": artifact_hashes,
        "diagnostic_views": list(DIAGNOSTICS),
        "clean_reproduction": clean_checks,
        "memory_profile": profile,
        "optimizer_steps": 0,
        "independent_test_evidence": False,
    }
    _json_write(report, destination)
    if not report["ready"]:
        raise RuntimeError(f"G-D1 GPU memory prerequisite failed; see {destination}")
    return report


def verify_prerequisites(path, *, root, spec, device_name):
    if path is None or not Path(path).is_file():
        raise ValueError("G-D1 requires completed diagnostic and memory prerequisites")
    report = json.loads(Path(path).read_text())
    if (
        report.get("version") != 2
        or report.get("ready") is not True
        or report.get("optimizer_steps") != 0
    ):
        raise ValueError("G-D1 prerequisite is incomplete or failed")
    if report["spec"] != spec.to_dict() or report["data_sha256"] != _data_hashes(root):
        raise ValueError("G-D1 prerequisite spec or canonical data changed")
    if report["code_sha256"] != _code_hashes(root) or report["environment"] != _cuda_environment(
        device_name
    ):
        raise ValueError("G-D1 prerequisite code or GPU runtime changed")
    profile = report["memory_profile"]
    if not profile.get("passed") or not memory_profile_passes(profile):
        raise ValueError("G-D1 prerequisite GPU memory check failed")
    if (
        report["diagnostic_views"] != list(DIAGNOSTICS)
        or len(report["clean_reproduction"]) != 10
        or not all(c["passed"] for c in report["clean_reproduction"])
    ):
        raise ValueError("G-D1 diagnostic scope or clean reproduction is incomplete")
    if (
        len(report["diagnostic_sha256"]) != 10 * len(DIAGNOSTICS) + 1
        or len(report["source_sha256"]) != 70
    ):
        raise ValueError("G-D1 source/diagnostic manifest is incomplete")
    for manifest in (report["source_sha256"], report["diagnostic_sha256"]):
        for filename, digest in manifest.items():
            if compute_sha256(filename) != digest:
                raise ValueError(f"G-D1 prerequisite artifact changed: {filename}")
    return report
