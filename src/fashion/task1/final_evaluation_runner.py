"""One-shot Task 1 prediction, explicit scoring, official export, and read-only replay."""

from __future__ import annotations

import io
import json
import platform
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from fashion.config import ROOT
from fashion.data.dataset import (
    load_label_maps,
    load_manifest,
    load_splits,
    load_splits_for_final_evaluation,
)
from fashion.data.hashing import compute_sha256
from fashion.task1.evaluation import classification_metrics, validate_task1_label_map
from fashion.task1.final_evaluation import (
    blind_prediction_frame,
    checked_ids,
    grouped_intervals,
    score_tables,
    validate_predictions,
)
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)

EVIDENCE = Path("results/evidence/task1/final_evaluation")
CONFIG = Path("configs/task1/final_evaluation.json")
SPLITS = Path("data/processed/splits.csv")
MAPS = Path("data/processed/label_maps.json")
MODEL = Path("models/task1_article_type.pt")
MANIFEST = Path("models/task1_article_type.manifest.json")
PREDICTION_MANIFEST = Path("data/processed/prediction_manifest.csv")
TEMPLATE = Path("data/raw/teacher/test/styles_prediction.csv")
EXPORT = Path("results/article_type_test_predictions.csv")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record(root, path):
    path = Path(path)
    absolute = path if path.is_absolute() else root / path
    return {"path": absolute.relative_to(root).as_posix(), "sha256": compute_sha256(absolute)}


def _verify(root, records):
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("artifact path leaves the project")
        verify_artifact(path, record["sha256"])


def _load_bundle(root):
    from fashion.task1.final_inference import load_task1_bundle

    return load_task1_bundle(manifest_path=root / MANIFEST, project_root=root, device="cpu")


def _training_record(root, run_id):
    # Read directly: registry.read() acquires a lock file, which replay must not write.
    registry = pd.read_csv(root / "results/runs.csv", dtype=str, keep_default_na=False)
    selected = registry.loc[registry.run_id.eq(run_id)]
    if len(selected) != 1 or selected.iloc[0]["status"] != "completed":
        raise ValueError("final training registry row is missing, duplicated, or incomplete")
    # New columns added by another task do not change this run's nonempty fields.
    return {name: value for name, value in selected.iloc[0].to_dict().items() if value != ""}


@contextmanager
def _stage(root, name, receipt):
    folder = root / EVIDENCE
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".stage.lock"), timeout=0):
        marker = folder / f"{name}_attempt.json"
        if marker.exists() or (folder / receipt).exists():
            raise ValueError(
                f"{name} already completed or has partial output; audit/inspect it before retry"
            )
        atomic_write_json(marker, {"stage": name, "started_at": _now()})
        yield


def _inputs(root):
    config = _read(root / CONFIG)
    if config["candidate_id"] != "task1_cnn_no_aug_unweighted_v1" or config["num_classes"] != 124:
        raise ValueError("final evaluation must use the selected plain 124-class CNN")
    rows = load_splits(root / SPLITS)
    for partition in ("development", "holdout", "quarantine"):
        if int(rows.partition.eq(partition).sum()) != config["expected_rows"][partition]:
            raise ValueError(f"unexpected {partition} row count")
    _, labels = validate_task1_label_map(load_label_maps(root / MAPS)["articleType"])
    holdout = rows.loc[rows.partition.eq("holdout")].copy()
    checked_ids(holdout.id)
    if holdout.product_family_group.astype(str).str.strip().eq("").any():
        raise ValueError("holdout family groups must be present")
    return config, rows, labels


def _image_records(root, rows):
    records = []
    for row in rows.itertuples():
        path = (root / str(row.path)).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("image path leaves project")
        verify_artifact(path, str(row.sha256))
        records.append({"id": int(row.id), **_record(root, path)})
    return records


def _predict_condition(bundle, paths, condition, batch_size):
    if condition["kind"] == "none":
        return bundle.predict_paths(paths, batch_size=batch_size)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="task1-photo-probe-") as directory:
        for start in range(0, len(paths), batch_size):
            transformed = []
            for j, path in enumerate(paths[start : start + batch_size]):
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    if condition["kind"] == "jpeg":
                        buffer = io.BytesIO()
                        image.save(
                            buffer,
                            format="JPEG",
                            quality=condition["quality"],
                            subsampling=condition["subsampling"],
                        )
                        buffer.seek(0)
                        with Image.open(buffer) as decoded:
                            image = decoded.convert("RGB")
                    elif condition["kind"] == "brightness":
                        image = ImageEnhance.Brightness(image).enhance(condition["factor"])
                    elif condition["kind"] == "blur":
                        image = image.filter(ImageFilter.GaussianBlur(condition["radius"]))
                    else:
                        raise ValueError("unknown frozen robustness condition")
                    output = Path(directory) / f"{j}.png"
                    image.save(output)
                    transformed.append(output)
            outputs.append(bundle.predict_paths(transformed, batch_size=batch_size))
    return np.concatenate(outputs)


def _cost(bundle, paths, root, config):
    import torch

    settings = config["cost"]
    for _ in range(settings["warmup"]):
        bundle.predict_paths(paths[:1], batch_size=1)
    timings = []
    for i in range(settings["repeats"]):
        start = time.perf_counter()
        bundle.predict_paths([paths[i % len(paths)]], batch_size=1)
        timings.append((time.perf_counter() - start) * 1000)
    batch = paths[: settings["batch_size"]]
    start = time.perf_counter()
    bundle.predict_paths(batch, batch_size=settings["batch_size"])
    elapsed = time.perf_counter() - start
    return {
        "device": "cpu",
        "machine": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "checkpoint_bytes": (root / MODEL).stat().st_size,
        "batch_one_ms_median": float(np.median(timings)),
        "batch_one_ms_p95": float(np.quantile(timings, 0.95)),
        "latency_samples_ms": timings,
        "batch_rows": len(batch),
        "batch_seconds": elapsed,
        "batch_images_per_second": len(batch) / elapsed,
        **settings,
    }


def predict_holdout(project_root=ROOT):
    root = Path(project_root).resolve()
    # Refuse even loading a model again when a prior attempt exists.
    with _stage(root, "predict_holdout", "prediction_receipt.json"):
        config, rows, labels = _inputs(root)
        bundle = _load_bundle(root)
        if list(bundle.class_names) != labels:
            raise ValueError("model class order differs from canonical label map")
        holdout = rows.loc[rows.partition.eq("holdout")]
        images = _image_records(root, holdout)
        paths = [root / record["path"] for record in images]
        source = sorted((root / "src/fashion/task1").rglob("*.py"))
        for package in ("data", "train"):
            source.extend(sorted((root / f"src/fashion/{package}").rglob("*.py")))
        source.extend(p for p in (root / "src/fashion/config.py",) if p.exists())
        development_evidence = {
            name: _record(root, path)
            for name in ("comparison", "classical_comparison", "oof_metrics")
            if (path := root / f"results/evidence/task1/{name}.csv").exists()
        }
        refit_manifest = _read(root / MANIFEST)
        history = refit_manifest["history"]
        _verify(root, [history])
        inputs = [
            _record(root, p)
            for p in [CONFIG, SPLITS, MAPS, MODEL, MANIFEST, PREDICTION_MANIFEST, TEMPLATE, *source]
        ]
        frozen = {
            "created_at": _now(),
            "inputs": inputs,
            "class_names": labels,
            "images": images,
            "image_ids_sha256": canonical_sha256(holdout.id.astype(int).tolist()),
            "config": config,
            "access_history": config["access_history"],
            "development_evidence": development_evidence,
            "training_history": history,
            "training_record": _training_record(root, refit_manifest["run_id"]),
        }
        atomic_write_json(root / EVIDENCE / "freeze.json", frozen)
        artifacts = []
        for condition in config["robustness"]:
            filename = (
                "holdout_predictions.csv"
                if condition["condition"] == "clean"
                else f"holdout_{condition['condition']}.csv"
            )
            frame = blind_prediction_frame(
                holdout.id,
                _predict_condition(bundle, paths, condition, config["cost"]["batch_size"]),
                labels,
            )
            output = root / EVIDENCE / filename
            if output.exists():
                raise ValueError(f"partial output already exists: {output}")
            atomic_write_csv(output, frame)
            artifacts.append({"condition": condition["condition"], **_record(root, output)})
        cost_path = root / EVIDENCE / "cost.json"
        atomic_write_json(cost_path, _cost(bundle, paths, root, config))
        _verify(root, inputs)
        _verify(root, [history, *development_evidence.values()])
        _verify(root, images)
        receipt = {
            "completed_at": _now(),
            "rows": len(holdout),
            "labels_accessed": False,
            "freeze": _record(root, EVIDENCE / "freeze.json"),
            "predictions": artifacts,
            "cost": _record(root, cost_path),
        }
        atomic_write_json(root / EVIDENCE / "prediction_receipt.json", receipt)
        return receipt


def _verified_prediction(root):
    path = root / EVIDENCE / "prediction_receipt.json"
    if not path.is_file():
        raise FileNotFoundError("Run predict-holdout before score or evaluation replay")
    receipt = _read(path)
    _verify(root, [receipt["freeze"], receipt["cost"], *receipt["predictions"]])
    frozen = _read(root / receipt["freeze"]["path"])
    _verify(root, frozen["inputs"])
    _verify(root, [frozen["training_history"], *frozen["development_evidence"].values()])
    if _training_record(root, frozen["training_record"]["run_id"]) != frozen["training_record"]:
        raise ValueError("final training registry row differs from its freeze")
    declared_conditions = [item["condition"] for item in frozen["config"]["robustness"]]
    if [item["condition"] for item in receipt["predictions"]] != declared_conditions:
        raise ValueError("prediction receipt is missing a frozen robustness condition")
    # Verify the saved image digest list, without reopening images or raw labels.
    expected = [image["id"] for image in frozen["images"]]
    if canonical_sha256(expected) != frozen["image_ids_sha256"]:
        raise ValueError("frozen image IDs disagree")
    for record in receipt["predictions"]:
        frame = pd.read_csv(root / record["path"], keep_default_na=False)
        validate_predictions(frame, expected, frozen["class_names"])
    return receipt, frozen


def score_holdout(project_root=ROOT, *, evaluation_unlocked=False):
    if not evaluation_unlocked:
        raise ValueError("score requires explicit --evaluation-unlocked")
    root = Path(project_root).resolve()
    receipt, frozen = _verified_prediction(root)
    with _stage(root, "score", "evaluation_manifest.json"):
        unlock_path = root / EVIDENCE / "unlock_receipt.json"
        atomic_write_json(
            unlock_path,
            {
                "unlocked_at": _now(),
                "prediction_receipt": _record(root, EVIDENCE / "prediction_receipt.json"),
                "target": "articleType",
                "access": (
                    "shared protected loader joins all target columns; "
                    "scoring uses articleType only"
                ),
                "prior_access": frozen["access_history"],
            },
        )
        rows = load_splits_for_final_evaluation(
            root / SPLITS,
            evaluation_unlocked=True,
            raw_teacher_csv=root / "data/raw/teacher/train/styles_train.csv",
        )
        holdout = rows.loc[rows.partition.eq("holdout")].copy()
        expected = [image["id"] for image in frozen["images"]]
        if holdout.id.tolist() != expected:
            raise ValueError("unlocked rows differ from frozen image IDs")
        labels = frozen["class_names"]
        counts = rows.loc[rows.partition.eq("development"), "articleType"].value_counts().to_dict()
        clean = next(r for r in receipt["predictions"] if r["condition"] == "clean")
        p = validate_predictions(
            pd.read_csv(root / clean["path"], keep_default_na=False), expected, labels
        )
        tables = score_tables(holdout, p, labels, counts)
        tables["scored_rows"] = holdout[
            ["id", "articleType", "product_family_group", "mode"]
        ].copy()
        y = holdout.articleType.map({label: i for i, label in enumerate(labels)}).to_numpy(
            dtype=int
        )
        bootstrap = frozen["config"]["bootstrap"]
        tables["uncertainty"] = grouped_intervals(
            y,
            p,
            holdout.product_family_group,
            replicates=bootstrap["replicates"],
            seed=bootstrap["seed"],
        )
        stress = []
        for record in receipt["predictions"]:
            condition_p = validate_predictions(
                pd.read_csv(root / record["path"], keep_default_na=False), expected, labels
            )
            stress.append(
                {"condition": record["condition"], **classification_metrics(y, condition_p)}
            )
        tables["robustness"] = pd.DataFrame(stress)
        artifacts = {}
        for name, table in tables.items():
            output = root / EVIDENCE / f"{name}.csv"
            atomic_write_csv(output, table)
            artifacts[name] = _record(root, output)
        figures = _figures(root, tables)
        manifest = {
            "completed_at": _now(),
            "target": "articleType",
            "rows": len(holdout),
            "prediction_receipt": _record(root, EVIDENCE / "prediction_receipt.json"),
            "unlock_receipt": _record(root, unlock_path),
            "tables": artifacts,
            "figures": figures,
            "label_source_sha256": compute_sha256(root / "data/raw/teacher/train/styles_train.csv"),
            "limitations": [
                "same-source catalogue holdout",
                "other targets accessed shared holdout earlier",
                "124-class macro-F1 includes absent-class zeros",
                "family bootstrap intervals do not measure distribution shift",
                "photo probes do not establish customer-photo performance",
                "confidence is uncalibrated; no review threshold was selected",
            ],
        }
        atomic_write_json(root / EVIDENCE / "evaluation_manifest.json", manifest)
        return manifest


def _figures(root, tables):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = root / "results/figures/task1/final_evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    output = []
    for name, frame, x, y, title in [
        (
            "robustness",
            tables["robustness"],
            "condition",
            "macro_f1",
            "Fixed photo probes: 124-class macro-F1",
        ),
        (
            "risk_coverage",
            tables["risk_coverage"],
            "coverage",
            "error_rate",
            "Raw confidence: diagnostic only",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(frame[x], frame[y], marker="o")
        ax.set(xlabel=x, ylabel=y, title=title)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output.append(_record(root, path))
    return output


def audit_final_evaluation(project_root=ROOT):
    """Read saved evidence only. Does not import inference, unlock labels, or write files."""
    root = Path(project_root).resolve()
    receipt, frozen = _verified_prediction(root)
    path = root / EVIDENCE / "evaluation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Run score --evaluation-unlocked before evaluation replay or predict-test"
        )
    manifest = _read(path)
    required_tables = {
        "metrics",
        "per_class",
        "errors",
        "confusion_pairs",
        "slices",
        "reliability",
        "risk_coverage",
        "uncertainty",
        "robustness",
        "scored_rows",
    }
    if set(manifest["tables"]) != required_tables:
        raise ValueError("evaluation manifest is missing required score tables")
    _verify(
        root,
        [
            manifest["prediction_receipt"],
            manifest["unlock_receipt"],
            *manifest["tables"].values(),
            *manifest["figures"],
        ],
    )
    result = {
        "prediction_receipt": receipt,
        "freeze": frozen,
        "evaluation_manifest": manifest,
        "tables": {
            name: pd.read_csv(root / record["path"], keep_default_na=False)
            for name, record in manifest["tables"].items()
        },
        "cost": _read(root / receipt["cost"]["path"]),
        "test_prediction_receipt": None,
    }
    tables = result["tables"]
    scored = tables["scored_rows"]
    expected = [image["id"] for image in frozen["images"]]
    if not np.array_equal(checked_ids(scored.id), checked_ids(expected)):
        raise ValueError("scored row IDs/order differ from frozen predictions")
    labels = frozen["class_names"]
    clean = next(record for record in receipt["predictions"] if record["condition"] == "clean")
    p = validate_predictions(
        pd.read_csv(root / clean["path"], keep_default_na=False), expected, labels
    )
    counts = dict(zip(tables["per_class"].class_name, tables["per_class"].development_count))
    recomputed = score_tables(scored, p, labels, counts)
    for name, frame in recomputed.items():
        try:
            pd.testing.assert_frame_equal(
                tables[name].reset_index(drop=True),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError as error:
            raise ValueError(
                f"saved {name} metrics/score table differs from predictions"
            ) from error
    lookup = {label: i for i, label in enumerate(labels)}
    y = scored.articleType.map(lookup).to_numpy(dtype=int)
    robustness = tables["robustness"].set_index("condition")
    if list(robustness.index) != [r["condition"] for r in receipt["predictions"]]:
        raise ValueError("saved robustness metrics conditions differ from predictions")
    for record in receipt["predictions"]:
        probabilities = validate_predictions(
            pd.read_csv(root / record["path"], keep_default_na=False), expected, labels
        )
        for name, value in classification_metrics(y, probabilities).items():
            if not np.isclose(
                robustness.loc[record["condition"], name], value, rtol=1e-10, atol=1e-12
            ):
                raise ValueError("saved robustness metrics differ from predictions")
    test = root / EVIDENCE / "test_prediction_receipt.json"
    if test.exists():
        official = _read(test)
        _verify(
            root,
            [
                official["evaluation_manifest"],
                official["output"],
                official["probabilities"],
                *official["inputs"],
            ],
        )
        template = pd.read_csv(root / TEMPLATE, keep_default_na=False)
        ids = checked_ids(template.id)
        frame = pd.read_csv(root / official["probabilities"]["path"], keep_default_na=False)
        validate_predictions(frame, ids, labels)
        output = pd.read_csv(root / official["output"]["path"], keep_default_na=False)
        if list(output.columns) != ["id", "articleType"] or not np.array_equal(
            checked_ids(output.id), ids
        ):
            raise ValueError("official output schema or ID order differs from template")
        if len(output) != frozen["config"]["expected_rows"]["official_test"] or official[
            "rows"
        ] != len(output):
            raise ValueError("official output has the wrong row count")
        if not np.array_equal(output.articleType, frame.predicted_label):
            raise ValueError("official output labels differ from frozen model probabilities")
        if canonical_sha256(ids.tolist()) != official["image_ids_sha256"]:
            raise ValueError("official output ID digest differs")
        result["test_prediction_receipt"] = official
    return result


def predict_test(project_root=ROOT):
    root = Path(project_root).resolve()
    audited = audit_final_evaluation(root)
    with _stage(root, "predict_test", "test_prediction_receipt.json"):
        if (root / EXPORT).exists():
            raise ValueError("official articleType output already exists; inspect before replacing")
        frozen = audited["freeze"]
        rows = load_manifest(root / PREDICTION_MANIFEST)
        template = pd.read_csv(root / TEMPLATE, keep_default_na=False)
        if list(template.columns) != ["id", "gender", "articleType", "season", "usage"]:
            raise ValueError("official template columns differ from submission contract")
        expected = checked_ids(template.id)
        if len(rows) != frozen["config"]["expected_rows"]["official_test"] or set(
            checked_ids(rows.id)
        ) != set(expected):
            raise ValueError("official image IDs/count differ from template")
        rows = rows.set_index("id").loc[expected].reset_index()
        images = _image_records(root, rows)
        bundle = _load_bundle(root)
        if list(bundle.class_names) != frozen["class_names"]:
            raise ValueError("official class order differs from frozen holdout model")
        frame = blind_prediction_frame(
            expected,
            bundle.predict_paths(
                [root / image["path"] for image in images],
                batch_size=frozen["config"]["cost"]["batch_size"],
            ),
            frozen["class_names"],
        )
        _verify(root, frozen["inputs"])
        _verify(root, images)
        probabilities_path = root / EVIDENCE / "official_test_probabilities.csv"
        atomic_write_csv(probabilities_path, frame)
        atomic_write_csv(
            root / EXPORT,
            frame[["id", "predicted_label"]].rename(columns={"predicted_label": "articleType"}),
        )
        receipt = {
            "completed_at": _now(),
            "rows": len(frame),
            "output": _record(root, EXPORT),
            "probabilities": _record(root, probabilities_path),
            "images": images,
            "image_ids_sha256": canonical_sha256(expected.tolist()),
            "evaluation_manifest": _record(root, EVIDENCE / "evaluation_manifest.json"),
            "inputs": [_record(root, p) for p in [MODEL, MANIFEST, PREDICTION_MANIFEST, TEMPLATE]],
            "combined_submission_status": (
                "pending validated outputs from the other owners; source template unchanged"
            ),
        }
        atomic_write_json(root / EVIDENCE / "test_prediction_receipt.json", receipt)
        return receipt
