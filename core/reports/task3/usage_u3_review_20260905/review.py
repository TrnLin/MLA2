"""Review saved U3 evidence only. No model import, training, or inference."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import hashlib
import io
import json
import pickle
import subprocess
import zipfile
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.task3_decisions import oof_metrics, validate_oof
from fashion.train.task3_usage_two_stage import (
    CLASSES,
    CORE_CORRUPTIONS,
    check_usage_two_stage_sources,
    class_weights_for_training,
    configuration_hash,
    evaluate_usage_two_stage,
    load_two_stage_run,
    read_predictions,
    recipe,
    training_scope,
)

HERE = Path(__file__).resolve().parent
SAVED = HERE / "saved"
E2 = ROOT / "results/evidence/task3/experiments/t3_usage_e2_class_balanced_ce/usage"
FIGURES = ROOT / "results/figures/task3/usage_u3_review"
METRICS = ("macro_f1", "nll", "brier", "ece_15", "accuracy")
RUN_COMMIT = "67e71e5cc762fcc2573f3a215c1f43ffd578b041"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def compare_saved(left, right, path="decision"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        return max((compare_saved(left[k], right[k], f"{path}.{k}") for k in left), default=0)
    if isinstance(left, list):
        assert len(left) == len(right), path
        return max(
            (compare_saved(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right))),
            default=0,
        )
    if isinstance(left, (float, int)) and not isinstance(left, bool):
        error = abs(left - right)
        assert error <= 1e-10, (path, left, right)
        return error
    assert left == right, (path, left, right)
    return 0


def tensor_reference(storage, offset, shape, stride, requires_grad, hooks):
    del requires_grad, hooks
    return {"storage": storage, "offset": offset, "shape": shape, "stride": stride}


class TensorMetadataReader(pickle.Unpickler):
    """Interpret only the four globals observed in these tensor-only archives.

    Never load PyTorch or execute checkpoint-defined code. Storage payloads are
    read as bounded byte arrays; all other global names fail closed.
    """

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): tensor_reference,
            ("torch", "FloatStorage"): "float32",
            ("torch", "LongStorage"): "int64",
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unexpected checkpoint global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, pid):
        kind, dtype, key, location, size = pid
        assert kind == "storage" and dtype in {"float32", "int64"}
        assert str(key).isdigit() and 0 <= size <= 2_000_000
        return {"dtype": dtype, "key": key, "location": location, "size": size}


def inspect_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        meta_path = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = meta_path.removesuffix("data.pkl")
        assert archive.read(prefix + "byteorder") == b"little"
        data = TensorMetadataReader(io.BytesIO(archive.read(meta_path))).load()
        digest = hashlib.sha256()
        tensors = {}
        for name, tensor in data["model_state_dict"].items():
            storage = tensor["storage"]
            shape = tuple(tensor["shape"])
            assert tensor["offset"] == 0
            assert int(np.prod(shape)) == storage["size"]
            raw = archive.read(prefix + "data/" + storage["key"])
            assert len(raw) == storage["size"] * np.dtype(storage["dtype"]).itemsize
            expected_stride = np.empty(shape, dtype=storage["dtype"]).strides
            assert tuple(tensor["stride"]) == tuple(
                n // np.dtype(storage["dtype"]).itemsize for n in expected_stride
            )
            tensors[name] = hashlib.sha256(raw).hexdigest()
            if name.startswith("features."):
                digest.update(name.removeprefix("features.").encode())
                digest.update(f"(torch.{storage['dtype']}, {shape})".encode())
                digest.update(raw)
        return data, tensors, digest.hexdigest()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(HERE / "runs.csv", keep_default_na=False)
    splits = load_splits(ROOT / "data/processed/splits.csv")
    sources, _ = check_usage_two_stage_sources(
        e2_directory=E2, registry_path=HERE / "runs.csv", root=ROOT
    )
    audit = read_json(SAVED / "source_audit.json")
    assert audit["recipe"] == recipe()
    assert audit["split_sha256"] == compute_sha256(ROOT / "data/processed/splits.csv")
    assert audit["label_map_sha256"] == compute_sha256(ROOT / "data/processed/label_maps.json")
    live_source_changes = []
    for name, digest in audit["code_sha256"].items():
        path = (ROOT / "src/fashion/train" / name).resolve().relative_to(ROOT)
        historical = subprocess.check_output(["git", "show", f"{RUN_COMMIT}:{path}"], cwd=ROOT)
        assert hashlib.sha256(historical).hexdigest() == digest
        if compute_sha256(resolve_task3_path(path, root=ROOT)) != digest:
            live_source_changes.append(str(path))
            # This file is not imported by the read-only review path. Other
            # changed dependencies require a fresh check before reproduction.
            assert name == "task3_baseline.py"
    for fold, source in sources.items():
        assert audit["parents"][str(fold)] == {
            "run_id": source["run_id"],
            "sha256": source["sha256"],
        }
    child, stages, stage_a_frames, checks, resources, histories = {}, [], [], [], [], {}
    verified_hashes, maximum_metric_error = 0, 0.0
    registry_findings = []
    for directory in sorted(SAVED.glob("t3_usage*")):
        config, manifest = (
            read_json(directory / "config.json"),
            read_json(directory / "manifest.json"),
        )
        fold = config["fold"]
        assert config == {**audit, "fold": fold}
        digest = configuration_hash(config)
        assert manifest["run_id"] == directory.name and manifest["config_hash"] == digest
        assert f"_{digest[:12]}_" in directory.name
        for name, expected in manifest["files"].items():
            relative = Path(name)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert compute_sha256(directory / relative) == expected, name
            verified_hashes += 1
        train, validation = training_scope(splits, fold)
        metrics = read_json(directory / "metrics.json")
        cache = read_json(directory / "feature_cache_manifest.json")
        assert cache["scope"] == "outer_training_only"
        assert cache["ids"] == train.id.astype(int).tolist()
        assert cache["product_family_groups"] == train.product_family_group.astype(str).tolist()
        assert cache["shape"] == [len(train), 256]
        counts, weights = class_weights_for_training(train)
        assert metrics["class_counts"] == counts
        assert np.allclose(metrics["class_weights"], weights, atol=1e-12, rtol=0)
        norm = read_json(directory / "normalization.json")
        assert norm["fit_scope"] == "outer_training_only" and norm["validation_fold"] == fold
        assert (
            norm["training_ids_sha256"] == hashlib.sha256(train.id.to_numpy().tobytes()).hexdigest()
        )
        parent_norm = read_json(E2 / sources[fold]["run_id"] / "normalization.json")
        assert all(norm[k] == parent_norm[k] for k in ("mean", "std"))
        history = pd.read_csv(directory / "history.csv")
        histories[fold] = history
        assert list(zip(history.stage, history.epoch)) == [
            *(("A", i) for i in range(1, 31)),
            *(("B", i) for i in range(1, 11)),
        ]
        assert (metrics["selected_stage"], metrics["selected_epoch"]) == ("B", 10)
        assert (
            metrics["stage_b_trainable_parameters"] == 2313 and metrics["parameter_count"] == 391209
        )
        checkpoints = [inspect_checkpoint(directory / n) for n in ("stage_a.pt", "final_epoch.pt")]
        feature_names = [n for n in checkpoints[0][1] if n.startswith("features.")]
        assert len(feature_names) == 24
        assert all(checkpoints[0][1][n] == checkpoints[1][1][n] for n in feature_names)
        assert checkpoints[0][1]["classifier.weight"] != checkpoints[1][1]["classifier.weight"]
        for index, (data, _, backbone) in enumerate(checkpoints):
            assert data["two_stage_contract"] == config and data["run_id"] == directory.name
            assert data["config"] == config["recipe"]["baseline"] and data["normalization"] == norm
            assert data["classes"] == list(CLASSES)
            assert (data["stage"], data["epoch"]) == [("A", 30), ("B", 10)][index]
            assert backbone == cache["backbone_sha256_before"] == metrics["backbone_sha256_after"]
        stage_a = read_json(directory / "stage_a_metrics.json")
        assert stage_a == metrics["stage_a"] and metrics["backbone_unchanged"]
        for stage, scope, name, expected, frame in (
            ("A", "training", "stage_a_train_predictions.csv", stage_a["training"], train),
            ("A", "validation", "stage_a_oof_predictions.csv", stage_a["validation"], validation),
            ("B", "training", "clean_train_predictions.csv", metrics["clean_training"], train),
            ("B", "validation", "oof_predictions.csv", metrics, validation),
        ):
            predictions = validate_oof(
                read_predictions(directory / name), frame, target="usage", classes=CLASSES
            )
            assert predictions.run_id.eq(directory.name).all()
            actual = oof_metrics(predictions, CLASSES)
            for key in METRICS:
                error = abs(actual[key] - expected[key])
                assert error <= 1e-10
                maximum_metric_error = max(maximum_metric_error, error)
            assert actual["confusion_matrix"] == expected["confusion_matrix"]
            stages.append(
                {
                    "fold": fold,
                    "stage": stage,
                    "scope": scope,
                    "support": len(frame),
                    **{k: actual[k] for k in METRICS},
                }
            )
            if (stage, scope) == ("A", "validation"):
                stage_a_frames.append(predictions)
        robust = pd.read_csv(directory / "robustness.csv", keep_default_na=False)
        assert set(robust.corruption) == set(CORE_CORRUPTIONS) and len(robust) == 5
        for name in CORE_CORRUPTIONS:
            frame = validate_oof(
                read_predictions(directory / "corruptions" / f"{name}.csv"),
                validation,
                target="usage",
                classes=CLASSES,
                run_ids_by_fold={fold: directory.name},
            )
            assert (
                abs(
                    oof_metrics(frame, CLASSES)["macro_f1"]
                    - robust.loc[robust.corruption.eq(name), "macro_f1"].item()
                )
                <= 1e-10
            )
        rows = registry[registry.run_id.eq(directory.name)]
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row.config_hash == digest and row.split_digest == config["split_sha256"]
        assert row.label_map_digest == config["label_map_sha256"]
        assert row.target == "usage" and str(row.scratch).lower() == "true"
        assert str(row.debug).lower() == "false" and int(row.seed) == 2753
        assert int(row.validation_fold) == fold and int(row.parameter_count) == 391209
        assert json.loads(row.parent_run_ids) == [sources[fold]["run_id"]]
        assert int(row.training_product_count) == len(train) and int(
            row.validation_product_count
        ) == len(validation)
        assert int(row.training_family_count) == train.product_family_group.nunique()
        assert int(row.validation_family_count) == validation.product_family_group.nunique()
        try:
            load_two_stage_run(directory, config, HERE / "runs.csv", splits)
            registry_status = "complete_and_verified"
        except ValueError as error:
            registry_status = str(error)
            assert fold == 4 and row.status == "running"
        registry_findings.append(
            {
                "fold": fold,
                "run_id": directory.name,
                "current_status": row.status,
                "completion_proof": registry_status,
                "checkpoint_sha256": manifest["files"]["final_epoch.pt"],
                "prediction_sha256": manifest["files"]["oof_predictions.csv"],
                "config_hash": digest,
            }
        )
        child[fold] = {
            "run_id": directory.name,
            "metrics": metrics,
            "predictions": read_predictions(directory / "oof_predictions.csv"),
            "robustness": robust,
        }
        resources.append(
            {
                "fold": fold,
                **{
                    k: metrics[k]
                    for k in (
                        "stage_a_seconds",
                        "stage_b_seconds",
                        "train_seconds",
                        "fold_wall_seconds",
                        "peak_memory_bytes",
                        "peak_host_memory_bytes",
                    )
                },
            }
        )
        checks.append(
            {
                "fold": fold,
                "validated_prediction_files": 9,
                "frozen_tensors_verified": len(feature_names),
                "backbone_sha256": checkpoints[0][2],
                "normalization_matches_e2": True,
            }
        )
    result = evaluate_usage_two_stage(child, sources, splits)
    saved_decision = read_json(SAVED / "screen_decision.json")
    decision_error = compare_saved(
        {k: v for k, v in saved_decision.items() if k != "run_ids"}, result
    )
    pooled_oof = (
        pd.concat([r["predictions"] for r in child.values()])
        .sort_values("id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        pooled_oof,
        read_predictions(SAVED / "oof_predictions.csv").sort_values("id").reset_index(drop=True),
        atol=1e-14,
        rtol=0,
    )
    pooled_a = oof_metrics(pd.concat(stage_a_frames), CLASSES)
    summary = [
        {"model": name, **{k: m[k] for k in METRICS}}
        for name, m in [
            ("E2", result["matched_parent_metrics"]),
            ("U3 Stage A", pooled_a),
            ("U3 Stage B", result["candidate_metrics"]),
        ]
    ]
    class_rows = []
    for name, metric in [
        ("E2", result["matched_parent_metrics"]),
        ("U3 Stage A", pooled_a),
        ("U3 Stage B", result["candidate_metrics"]),
    ]:
        class_rows.extend({"model": name, **c} for c in metric["per_class"])
    pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    pd.DataFrame(stages).to_csv(HERE / "stages.csv", index=False)
    pd.DataFrame(resources).to_csv(HERE / "resources.csv", index=False)
    pd.DataFrame(class_rows).to_csv(HERE / "classes.csv", index=False)
    write_json(HERE / "recomputed_decision.json", result)
    write_json(
        HERE / "verification.json",
        {
            "source_commit": RUN_COMMIT,
            "concurrent_live_source_changes_preserved": live_source_changes,
            "source_code_files_verified": len(audit["code_sha256"]),
            "manifest_file_hashes_verified": verified_hashes,
            "maximum_metric_error": maximum_metric_error,
            "pooled_oof_rows": len(pooled_oof),
            "decision_matches_saved": True,
            "maximum_decision_numeric_error": decision_error,
            "checks": checks,
            "registry": registry_findings,
            "registry_sha256": compute_sha256(HERE / "runs.csv"),
            "inference_repeated": False,
            "training_started": False,
            "limits": [
                "Fold 4 completion is missing from the current and inspected prior Drive registry.",
                "Cached feature values were not saved; their hash needs inference to reproduce.",
                "Clean E2 training predictions are missing; the clean-gap route stays closed.",
                "These folds have prior exposure; intervals do not remove selection bias.",
            ],
        },
    )
    draw_figures(
        pd.DataFrame(summary), pd.DataFrame(stages), pd.DataFrame(class_rows), result, histories
    )
    print(pd.DataFrame(summary).to_string(index=False))
    print("Verified", verified_hashes, "artifact hashes; maximum score error", maximum_metric_error)
    print("Decision:", result["status"], "Registry:", registry_findings)


def draw_figures(summary, stages, classes, decision, histories):
    colors = ["#687785", "#e4a23b", "#2a8b8b"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, metric, title in zip(
        axes[0],
        ("macro_f1", "nll"),
        ("Validation macro F1 · higher is better", "Validation NLL · lower is better"),
    ):
        ax.bar(summary.model, summary[metric], color=colors)
        for i, value in enumerate(summary[metric]):
            ax.text(i, value + 0.009, f"{value:.4f}", ha="center")
        ax.set_ylim(0, max(summary[metric]) * 1.22)
        ax.set_title(title)
    ax = axes[1, 0]
    wide = classes.pivot(index="class_name", columns="model", values="f1").reindex(CLASSES)
    delta = wide["U3 Stage B"] - wide["E2"]
    ax.barh(delta.index, delta, color=["#b24742" if v < -0.03 else "#2a8b8b" for v in delta])
    ax.axvline(-0.03, color="#b24742", linestyle="--", label="Loss limit −0.03")
    ax.set_xlim(-0.12, 0.08)
    ax.set_title("Class F1 change versus E2")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1, 1]
    rows = [c for c in decision["checks"] if c["gate"].startswith("robustness.")]
    names = [r["gate"].removeprefix("robustness.") for r in rows]
    values = [r["value"] for r in rows]
    ax.barh(names, values, color=["#b24742" if v < -0.02 else "#2a8b8b" for v in values])
    ax.axvline(-0.02, color="#b24742", linestyle="--", label="Loss limit −0.02")
    ax.set_title("Corruption effect versus E2 · higher is better")
    ax.set_xlim(-0.03, 0.012)
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Usage U3 · folds 0 and 4 · fixed final Stage B checkpoint fails", fontsize=16)
    fig.savefig(FIGURES / "screen_review.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for fold, history in histories.items():
        for stage in ("A", "B"):
            part = history[history.stage.eq(stage)]
            axes[0 if stage == "A" else 1].plot(
                part.epoch, part.validation_cross_entropy, label=f"Fold {fold}"
            )
    for i, title in enumerate(("Stage A: ordinary CE", "Stage B: reset weighted head")):
        axes[i].set(title=title, xlabel="Epoch", ylabel="Validation cross entropy")
        axes[i].legend()
    ax = axes[2]
    means = stages.groupby(["stage", "scope"]).macro_f1.mean()
    x = np.arange(2)
    for offset, scope, color in [(-0.18, "training", "#e4a23b"), (0.18, "validation", "#2a8b8b")]:
        values = [means.loc[stage, scope] for stage in ("A", "B")]
        ax.bar(x + offset, values, width=0.36, color=color, label=scope)
        for pos, value in zip(x + offset, values):
            ax.text(pos, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    ax.set(
        xticks=x,
        xticklabels=["Stage A", "Stage B"],
        ylim=(0, 1),
        title="Mean fold F1: training gain is much larger",
    )
    ax.legend()
    fig.suptitle("Stage B improves rare training labels; the validation gap grows", fontsize=15)
    fig.savefig(FIGURES / "stage_review.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
