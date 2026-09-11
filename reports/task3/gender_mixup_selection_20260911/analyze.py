"""Audit saved Gender runs and compare development predictions; never fit a model."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fashion.config import ROOT
from fashion.data import get_samples, load_splits
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.model import Task3GeM3CNN
from fashion.train.task3_decisions import oof_metrics, paired_family_bootstrap, validate_oof
from fashion.train.task3_gender_mixup_cv import CLASSES
from fashion.train.task3_gender_sam25_refit import _digest

HERE = Path(__file__).resolve().parent
SAM = HERE.parent / "gender_sam25_cv_result_20260906"
CORRUPTIONS = ["jpeg_75", "brightness_085", "brightness_115", "grayscale", "translation_003"]
VERIFIED_FILES = {}


def read(path):
    return json.loads(Path(path).read_text())


def csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_hash(path, expected):
    actual = compute_sha256(path)
    assert actual == expected, f"Changed artifact: {path}"
    VERIFIED_FILES[str(path.relative_to(ROOT))] = actual


def same_metrics(actual, saved):
    for key in ("accuracy", "macro_f1", "nll", "brier", "ece_15"):
        assert np.isclose(actual[key], saved[key], rtol=0, atol=1e-10), key
    assert actual["confusion_matrix"] == saved["confusion_matrix"]


def bootstrap(name, a, b, *, repetitions=10000):
    """Cache only against exact paired inputs, code, seed and draw count."""
    keys = ["id", "cv_fold", "product_family_group", "true_index", "predicted_index"]
    digest = hashlib.sha256()
    for frame in (a, b):
        digest.update(frame[keys].sort_values("id").to_csv(index=False).encode())
    digest.update(str(repetitions).encode())
    digest.update(compute_sha256(ROOT / "src/fashion/train/task3_decisions.py").encode())
    cache = HERE / "bootstrap_cache" / f"{name}_{digest.hexdigest()}.json"
    if cache.exists():
        return read(cache)
    result = paired_family_bootstrap(a, b, classes=CLASSES, repetitions=repetitions, seed=2753)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(result, indent=2) + "\n")
    print(name, result["point"], result["lower_95"], result["upper_95"], flush=True)
    return result


def verify_run(directory, manifest, registry, expected, *, refit=False):
    config = read(directory / "config.json")
    assert manifest["status"] == "complete"
    assert manifest["config_hash"] == _digest(config)
    for name, record in manifest["files"].items():
        if refit:
            assert record["path"] == f"{manifest['run_id']}/{name}"
        check_hash(directory / name, record["sha256"] if refit else record)
    row = registry.loc[manifest["run_id"]]
    assert row.status == "complete" and row.config_hash == manifest["config_hash"]
    assert row.checkpoint_sha256 == compute_sha256(directory / "final_epoch.pt")
    runtime = json.loads(row.environment_json)
    for name, digest in config["implementation_sha256"].items():
        source = subprocess.check_output(
            ["git", "show", f"{runtime['git_commit']}:src/fashion/{name}"], cwd=ROOT
        )
        assert hashlib.sha256(source).hexdigest() == digest, name
    contract = config["gender_label_variant"]
    check_hash(ROOT / "data/processed/splits.csv", contract["canonical_split_sha256"])
    check_hash(
        ROOT / "configs/task3/gender_mixup_alpha020_source.json", config["source_config_sha256"]
    )
    check_hash(ROOT / "data/processed/label_maps.json", config["label_map_sha256"])
    training = csv(directory / "training_rows.csv")
    columns = ["id", "cv_fold", "product_family_group", "gender", "sha256", "path"]
    pd.testing.assert_frame_equal(
        training[columns].sort_values("id").reset_index(drop=True),
        expected[columns].sort_values("id").reset_index(drop=True),
        check_dtype=False,
    )
    history = csv(directory / "history.csv")
    receipt = read(directory / "mixup_training.json")
    assert history.epoch.tolist() == list(range(1, 31))
    assert receipt["contract"] == config["mixup_contract"]
    assert config["mixup_policy"]["alpha"] == 0.2
    assert config["scratch"] and not config["early_stopping"]
    assert "sam_policy" not in config and config["cosine_t_max"] == 30
    assert not config["holdout_evaluated"] and not config["teacher_test_evaluated"]
    assert [r["epoch"] for r in receipt["epochs"]] == list(range(1, 31))
    assert all(r["rows"] == len(training) for r in receipt["epochs"])
    assert all(r["batches"] == int(np.ceil(len(training) / 128)) for r in receipt["epochs"])
    expected_lr = 1e-5 + (1e-3 - 1e-5) * (1 + np.cos(np.pi * np.arange(30) / 30)) / 2
    assert np.allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-14)
    checkpoint = torch.load(directory / "final_epoch.pt", map_location="cpu", weights_only=True)
    assert checkpoint["selected_epoch"] == 30 and checkpoint["epochs_completed"] == 30
    assert checkpoint["class_names"] == CLASSES and checkpoint["run_id"] == manifest["run_id"]
    assert checkpoint["config"] == config
    normalization = read(directory / "normalization.json")
    for key in ("mean", "std", "total_pixels"):
        assert checkpoint["normalization"][key] == normalization[key]
    values = {f.name: config[f.name] for f in fields(Task3BaselineConfig)}
    values["channels"] = tuple(values["channels"])
    model = Task3GeM3CNN(Task3BaselineConfig(**values), classifier_dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 390181
    assert all(torch.isfinite(t).all() for t in checkpoint["model_state_dict"].values())
    metrics = read(directory / "metrics.json")
    assert metrics == json.loads(row.metrics_json)
    return config, metrics, runtime


def main():
    splits = load_gender_name_truth_variant()
    development = get_samples(splits, partition="development", target="gender")
    assert len(development) == 32773
    registry = csv(HERE / "drive_runs.csv").set_index("run_id")
    assert registry.index.is_unique
    cv = read(HERE / "cv/cv_summary.json")
    sam = read(SAM / "cv_summary.json")
    assert cv["status"] == sam["status"] == "complete_for_review"
    frames = {"MixUp": [], "SAM25": []}
    corrupt = {name: {c: [] for c in CORRUPTIONS} for name in frames}
    fold_rows, class_gaps, histories, used_runs = [], [], {}, []
    for entry in cv["models"]:
        fold, run_id = entry["validation_fold"], entry["run_id"]
        directory = HERE / "cv" / entry["directory"]
        check_hash(directory / "manifest.json", entry["manifest_sha256"])
        manifest = read(directory / "manifest.json")
        training = development.loc[development.cv_fold.ne(fold)]
        validation = development.loc[development.cv_fold.eq(fold)]
        config, metrics, runtime = verify_run(directory, manifest, registry, training)
        used_runs.append(run_id)
        assert not set(training.product_family_group) & set(validation.product_family_group)
        assert set(csv(directory / "validation_rows.csv").id) == set(validation.id)
        sam_entry = next(r for r in sam["folds"] if r["fold"] == fold)
        sam_id = sam_entry["run_id"]
        used_runs.append(sam_id)
        sam_dir = HERE / "sam_ieee" / sam_id
        evaluation = read(sam_dir / "evaluation_manifest.json")
        assert evaluation == read(SAM / f"ieee{fold}/evaluation_manifest.json")
        sam_registry = registry.loc[sam_id]
        assert sam_registry.status == "complete"
        assert (
            sam_registry.checkpoint_sha256
            == evaluation["identity"]["source_sha256"]["final_epoch.pt"]
        )
        assert (
            sam_registry.prediction_sha256
            == evaluation["identity"]["source_sha256"]["oof_predictions.csv"]
        )
        for name, digest in evaluation["files"].items():
            check_hash(sam_dir / name, digest)
        for name in ("config.json", "normalization.json", "history.csv", "metrics.json"):
            check_hash(SAM / f"fold{fold}" / name, evaluation["identity"]["source_sha256"][name])
        sam_config = read(SAM / f"fold{fold}/config.json")
        assert config["gender_label_variant"] == sam_config["gender_label_variant"]
        assert config["mixup_contract"] == sam_config["mixup_contract"]
        for key in (
            "channels",
            "batch_size",
            "seed",
            "learning_rate",
            "weight_decay",
            "image_height",
            "image_width",
            "input_view",
            "training_precision_settings",
        ):
            assert config[key] == sam_config[key], key
        assert (
            config["training_augmentation"]
            == sam_config["child_experiment"]["training_augmentation"]
        )
        norm, sam_norm = (
            read(directory / "normalization.json"),
            read(SAM / f"fold{fold}/normalization.json"),
        )
        for key in ("mean", "std", "total_pixels"):
            assert norm[key] == sam_norm[key], key
        for name, src, score, pred_name, train_name in (
            ("MixUp", directory, metrics, "oof_predictions.csv", "train_eval_predictions.csv"),
            (
                "SAM25",
                sam_dir,
                read(sam_dir / "metrics.json"),
                "oof_predictions.csv",
                "clean_train_predictions.csv",
            ),
        ):
            rid = run_id if name == "MixUp" else sam_id
            pred = validate_oof(
                csv(src / pred_name),
                validation,
                target="gender",
                classes=CLASSES,
                run_ids_by_fold={fold: rid},
            )
            actual = oof_metrics(pred, CLASSES)
            same_metrics(actual, score)
            train = csv(src / train_name)
            train = validate_oof(train, training, target="gender", classes=CLASSES)
            assert train.run_id.eq(rid).all()
            clean_train = oof_metrics(train, CLASSES)
            assert np.isclose(clean_train["macro_f1"], score["final_train_eval_macro_f1"])
            frames[name].append(pred)
            histories[name, fold] = csv(
                directory / "history.csv" if name == "MixUp" else SAM / f"fold{fold}/history.csv"
            )
            fold_rows.append(
                dict(
                    model=name,
                    fold=fold,
                    run_id=rid,
                    **{k: actual[k] for k in ("macro_f1", "accuracy", "nll", "ece_15", "brier")},
                    train_f1=clean_train["macro_f1"],
                    gap=clean_train["macro_f1"] - actual["macro_f1"],
                    train_seconds=metrics["train_seconds"]
                    if name == "MixUp"
                    else sam_entry["train_seconds"],
                    gpu=runtime["gpu"] if name == "MixUp" else evaluation["runtime"]["gpu"],
                )
            )
            for a, t in zip(actual["per_class"], clean_train["per_class"], strict=True):
                class_gaps.append(
                    dict(
                        model=name,
                        fold=fold,
                        class_name=a["class_name"],
                        train_f1=t["f1"],
                        validation_f1=a["f1"],
                        gap=t["f1"] - a["f1"],
                    )
                )
            for c in CORRUPTIONS:
                file = f"predictions_{c}.csv" if name == "MixUp" else f"{c}_predictions.csv"
                corrupt[name][c].append(
                    validate_oof(
                        csv(src / file),
                        validation,
                        target="gender",
                        classes=CLASSES,
                        run_ids_by_fold={fold: rid},
                    )
                )
    pooled = {
        name: pd.concat(parts).sort_values("id").reset_index(drop=True)
        for name, parts in frames.items()
    }
    for name in pooled:
        assert len(pooled[name]) == 32773 and not pooled[name].id.duplicated().any()
    metrics = {name: oof_metrics(frame, CLASSES) for name, frame in pooled.items()}
    same_metrics(metrics["MixUp"], cv["pooled_oof"])
    same_metrics(metrics["SAM25"], sam["scopes"]["all_five"]["metrics"])
    for name, path in (
        ("MixUp", HERE / "cv/oof_predictions.csv"),
        ("SAM25", SAM / "aggregate/oof_predictions.csv"),
    ):
        pd.testing.assert_frame_equal(
            pooled[name],
            csv(path).sort_values("id").reset_index(drop=True),
            check_exact=False,
            atol=1e-14,
            rtol=1e-14,
        )
    a, b = pooled["MixUp"], pooled["SAM25"]
    intervals = {"all_five": bootstrap("clean", a, b)}
    intervals["all_five_100000_draw_sensitivity"] = bootstrap(
        "clean_100000", a, b, repetitions=100000
    )
    for name, folds in (("screen_folds", [0, 4]), ("additional_folds", [1, 2, 3])):
        x, y = a[a.cv_fold.isin(folds)], b[b.cv_fold.isin(folds)]
        intervals[name] = bootstrap(name, x, y)
    teacher = load_splits().set_index("id").gender
    originals = {}
    for name, frame in pooled.items():
        original = frame.copy()
        original["true_label"] = original.id.map(teacher)
        original["true_index"] = original.true_label.map(dict(zip(CLASSES, range(5), strict=True)))
        originals[name] = original
    intervals["original_teacher"] = bootstrap(
        "original_teacher", originals["MixUp"], originals["SAM25"]
    )
    original_robustness = []
    for name, frame in originals.items():
        scores = oof_metrics(frame, CLASSES)
        original_robustness.append(
            dict(
                model=name,
                condition="clean",
                **{k: scores[k] for k in ("macro_f1", "accuracy", "nll", "ece_15", "brier")},
            )
        )
    robustness = []
    for c in CORRUPTIONS:
        views = {
            name: pd.concat(corrupt[name][c]).sort_values("id").reset_index(drop=True)
            for name in pooled
        }
        interval = bootstrap(c, views["MixUp"], views["SAM25"])
        scores = {name: oof_metrics(frame, CLASSES) for name, frame in views.items()}
        for name, frame in views.items():
            original = frame.copy()
            original["true_index"] = original.id.map(teacher).map(
                dict(zip(CLASSES, range(5), strict=True))
            )
            original_scores = oof_metrics(original, CLASSES)
            original_robustness.append(
                dict(
                    model=name,
                    condition=c,
                    **{
                        k: original_scores[k]
                        for k in ("macro_f1", "accuracy", "nll", "ece_15", "brier")
                    },
                )
            )
        row = dict(
            corruption=c,
            mixup_f1=scores["MixUp"]["macro_f1"],
            sam25_f1=scores["SAM25"]["macro_f1"],
            difference=interval["point"],
            lower_95=interval["lower_95"],
            upper_95=interval["upper_95"],
        )
        for name, prefix in (("MixUp", "mixup"), ("SAM25", "sam25")):
            row[prefix + "_pooled_drop"] = metrics[name]["macro_f1"] - scores[name]["macro_f1"]
            fold_changes = [
                oof_metrics(views[name][views[name].cv_fold.eq(f)], CLASSES)["macro_f1"]
                - oof_metrics(pooled[name][pooled[name].cv_fold.eq(f)], CLASSES)["macro_f1"]
                for f in range(5)
            ]
            row[prefix + "_mean_fold_drop"] = -float(np.mean(fold_changes))
        robustness.append(row)
        intervals[c] = interval
    refit = read(HERE / "refit/model_manifest.json")
    refit_dir = HERE / "refit" / refit["run_id"]
    _, refit_metrics, refit_runtime = verify_run(
        refit_dir, refit, registry, development, refit=True
    )
    assert refit_metrics["training_rows"] == 32773 and refit_metrics["validation_rows"] == 0
    assert not refit["validation_used"] and refit["class_names"] == CLASSES
    used_runs.append(refit["run_id"])
    registry.loc[used_runs].to_csv(HERE / "verified_registry_rows.csv")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(HERE / "fold_comparison.csv", index=False)
    pd.DataFrame(class_gaps).to_csv(HERE / "class_gaps.csv", index=False)
    robust = pd.DataFrame(robustness)
    robust.to_csv(HERE / "robustness_comparison.csv", index=False)
    pd.DataFrame(original_robustness).to_csv(HERE / "original_label_robustness.csv", index=False)
    class_rows = pd.DataFrame(
        [dict(model=n, **r) for n, m in metrics.items() for r in m["per_class"]]
    )
    class_rows.to_csv(HERE / "class_comparison.csv", index=False)
    changed = (
        development.set_index("id")
        .loc[a.id, ["gender", "articleType", "product_family_group"]]
        .reset_index()
    )
    changed["mixup_correct"] = a.true_index.eq(a.predicted_index).to_numpy()
    changed["sam25_correct"] = b.true_index.eq(b.predicted_index).to_numpy()
    changed["fixed"] = changed.mixup_correct & ~changed.sam25_correct
    changed["new_error"] = ~changed.mixup_correct & changed.sam25_correct
    groups = (
        changed.groupby(["gender", "articleType"])
        .agg(
            rows=("id", "size"),
            fixed=("fixed", "sum"),
            new_errors=("new_error", "sum"),
            mixup_correct=("mixup_correct", "sum"),
            sam25_correct=("sam25_correct", "sum"),
        )
        .reset_index()
    )
    groups["net_correct"] = groups.fixed - groups.new_errors
    groups.to_csv(HERE / "article_error_changes.csv", index=False)
    summary = dict(
        status="saved_artifacts_verified",
        development_rows=len(a),
        development_families=a.product_family_group.nunique(),
        metrics=metrics,
        original_teacher_metrics={n: oof_metrics(f, CLASSES) for n, f in originals.items()},
        bootstrap=intervals,
        fixed_errors=int(changed.fixed.sum()),
        new_errors=int(changed.new_error.sum()),
        refit=dict(
            run_id=refit["run_id"],
            checkpoint_sha256=refit["files"]["final_epoch.pt"]["sha256"],
            metrics=refit_metrics,
            runtime=refit_runtime,
        ),
        mean_clean_training_f1=folds.groupby("model").train_f1.mean().to_dict(),
        mean_gap=folds.groupby("model").gap.mean().to_dict(),
        fold_f1_std=folds.groupby("model").macro_f1.std().to_dict(),
        holdout_used=False,
        teacher_test_used=False,
        new_training=False,
        new_inference=False,
        checkpoints_loaded_for_structure_only=True,
        final_model_changed=False,
    )
    write("summary.json", summary)
    write("bootstrap_intervals.json", intervals)
    historical_comparison()
    write("verified_artifact_hashes.json", VERIFIED_FILES)
    plot(summary, folds, robust, class_rows, histories, csv(refit_dir / "history.csv"))
    print(
        json.dumps({k: summary[k] for k in ("status", "fixed_errors", "new_errors", "mean_gap")}),
        flush=True,
    )


def plot(summary, folds, robust, classes, histories, refit_history):
    colors = {"MixUp": "#087f8c", "SAM25": "#b25d2e"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    for name in colors:
        f = folds[folds.model.eq(name)].sort_values("fold")
        axes[0, 0].plot(f.fold, 100 * f.macro_f1, "o-", label=name, color=colors[name])
        cl = classes[classes.model.eq(name)]
        axes[0, 1].plot(cl.class_name, 100 * cl.f1, "o-", color=colors[name], label=name)
    axes[0, 0].set(
        title="Held-out fold scores",
        ylabel="Macro-F1 (%)",
        xlabel="Saved development fold",
        xticks=range(5),
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="Class F1 from all 32,773 OOF predictions", ylabel="F1 (%)")
    labels = ["JPEG", "Darker", "Brighter", "Grayscale", "Small shift"]
    for name, prefix in (("MixUp", "mixup"), ("SAM25", "sam25")):
        axes[1, 0].plot(labels, 100 * robust[prefix + "_f1"], "o-", color=colors[name], label=name)
    axes[1, 0].set(title="Scores after fixed image damage", ylabel="Pooled macro-F1 (%)")
    bs = summary["bootstrap"]
    keys = [
        "all_five",
        "all_five_100000_draw_sensitivity",
        "screen_folds",
        "additional_folds",
        "original_teacher",
    ]
    labels = [
        "All five · 10,000 draws",
        "All five · 100,000 draws",
        "Folds 0,4 · corrected",
        "Folds 1,2,3 · corrected",
        "All five · original",
    ]
    point = np.array([bs[k]["point"] for k in keys]) * 100
    lo = np.array([bs[k]["lower_95"] for k in keys]) * 100
    hi = np.array([bs[k]["upper_95"] for k in keys]) * 100
    axes[1, 1].errorbar(
        point, range(5), xerr=[point - lo, hi - point], fmt="o", capsize=4, color=colors["MixUp"]
    )
    axes[1, 1].axvline(0, color="#555555", linewidth=1, linestyle="--")
    axes[1, 1].set(
        yticks=range(5),
        yticklabels=labels,
        title="Paired family bootstrap: 95% intervals",
        xlabel="MixUp minus SAM25 (F1 points)",
    )
    axes[1, 1].invert_yaxis()
    fig.suptitle(
        "Gender development comparison: MixUp 0.20 / 30 epochs versus SAM / 25 epochs\n"
        "Same saved development folds and labels; A100 versus L4",
        fontsize=13,
    )
    figure_dir = ROOT / "results/figures/task3"
    fig.savefig(figure_dir / "gender_mixup_selection_20260911.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for fold, ax in zip(range(5), axes.flat, strict=False):
        for name in colors:
            h = histories[name, fold]
            ax.plot(h.epoch, 100 * h.validation_macro_f1, color=colors[name], label=name)
        ax.set(
            title=f"Fold {fold}: logged validation",
            xlabel="Epoch",
            ylabel="Macro-F1 (%)",
            ylim=(45, 85),
        )
    axes[0, 0].legend(frameon=False)
    axes[1, 2].plot(refit_history.epoch, refit_history.train_mixed_loss, color=colors["MixUp"])
    axes[1, 2].set(
        title="Full-development MixUp refit", xlabel="Epoch", ylabel="Mixed training loss"
    )
    fig.suptitle(
        "Fixed final epochs, not the best point on each curve\n"
        "Logged training-runtime validation; final comparisons use separate IEEE evaluation",
        fontsize=13,
    )
    fig.savefig(figure_dir / "gender_mixup_learning_curves_20260911.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for fold, ax in zip(range(5), axes.flat, strict=False):
        h = histories["MixUp", fold]
        ax.plot(h.epoch, h.train_loss, label="Mixed training", color=colors["MixUp"])
        ax.plot(h.epoch, h.validation_loss, label="Clean validation", color=colors["SAM25"])
        ax.set(title=f"MixUp fold {fold}: recorded losses", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend(frameon=False)
    gaps = folds.pivot(index="fold", columns="model", values="gap")[list(colors)].mul(100)
    gaps.plot.bar(ax=axes[1, 2], rot=0, color=list(colors.values()))
    axes[1, 2].set(
        title="Final clean train–validation gap", xlabel="Saved fold", ylabel="F1 points"
    )
    axes[1, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Mixed training loss and clean validation loss have different targets\n"
        "The separate final clean F1 audit measures the fit gap",
        fontsize=13,
    )
    fig.savefig(figure_dir / "gender_mixup_fold_losses_20260911.png", dpi=160)
    plt.close(fig)


def historical_comparison():
    """Keep every registered Gender stage visible, with its own label/fold scope."""
    pack = HERE.parent / "main_report_20260906"
    lock = read(pack / "evidence_lock.json")
    assets = read(pack / "analysis_assets.json")
    for relative, digest in {**lock["files"], **assets}.items():
        if relative.endswith(("results/runs.csv", "results/classical_runs.csv")):
            check_hash(ROOT / relative.replace("reports/task3_", "reports/task3/", 1), digest)
    registry = csv(pack / "results/runs.csv").set_index("run_id")
    classical = csv(pack / "results/classical_runs.csv").set_index("run_id")
    registry = pd.concat([registry, classical.loc[~classical.index.isin(registry.index)]])
    rows = []
    for key, stage in lock["stages"].items():
        if stage["target"] != "gender":
            continue
        runs = registry.loc[stage["run_ids"]]
        assert runs.status.eq("complete").all()
        assert sorted(runs.validation_fold.tolist()) == stage["folds"]
        cm = np.sum([json.loads(value)["confusion_matrix"] for value in runs.metrics_json], axis=0)
        assert cm.sum() == runs.validation_product_count.sum()
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(5), where=denominator > 0)
        rows.append(
            dict(
                stage=key,
                experiment=stage["label"],
                folds=str(stage["folds"]),
                label_basis=stage["labels"],
                rows=int(cm.sum()),
                macro_f1=float(f1.mean()),
                accuracy=float(np.trace(cm) / cm.sum()),
                numerical_basis="historical registered confusion counts; see IEEE audits",
            )
        )
    pd.DataFrame(rows).to_csv(HERE / "all_gender_development_stages.csv", index=False)


if __name__ == "__main__":
    main()
