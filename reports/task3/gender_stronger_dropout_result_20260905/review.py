"""Verify the saved 04ab screen without importing PyTorch or running inference."""

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import CORE_CORRUPTIONS, oof_metrics, validate_oof
from fashion.train.task3_experiments import gender_gem_p3_spec
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_diagnostic import verify_input_images
from fashion.train.task3_gender_dropout_darkening import (
    NAME as PREVIOUS_NAME,
)
from fashion.train.task3_gender_dropout_darkening import (
    PARENT_RUN_IDS as ORIGINAL_DROPOUT_RUN_IDS,
)
from fashion.train.task3_gender_dropout_darkening import (
    _verify_training_evidence,
    compare_with_dropout,
)
from fashion.train.task3_gender_ieee import POLICY, load_ieee_evaluation
from fashion.train.task3_gender_narrow import (
    RUNTIME,
    evaluate_gender_narrow_screen,
    read_precision_prerequisites,
)
from fashion.train.task3_gender_stronger_dropout import NAME, PARENT_RUN_IDS
from fashion.train.task3_gender_weight_decay import _pool, _verify_baseline_controls

HERE = Path(__file__).resolve().parent
SAVED = HERE / "saved"
COMMIT = "1caa1fa55f9b2489484b9d958a3080af1965687f"
PREVIOUS = ROOT / "reports/task3/gender_dropout_darkening_result_20260905"


def read(path):
    return json.loads(path.read_text())


def same(saved, actual, location="root"):
    if isinstance(saved, dict):
        assert set(saved) == set(actual), location
        for key in saved:
            same(saved[key], actual[key], f"{location}.{key}")
    elif isinstance(saved, list):
        assert len(saved) == len(actual), location
        for i, (left, right) in enumerate(zip(saved, actual, strict=True)):
            same(left, right, f"{location}[{i}]")
    elif isinstance(saved, (float, int)) and not isinstance(saved, bool):
        assert np.isclose(saved, actual, rtol=0, atol=1e-10), (location, saved, actual)
    else:
        assert saved == actual, (location, saved, actual)


def verify_metric_block(saved, actual):
    for key, value in actual.items():
        same(saved[key], value, key)


def main():
    registry = pd.read_csv(HERE / "runs.csv", keep_default_na=False)
    splits = load_splits(ROOT / "data/processed/splits.csv")
    classes = load_label_maps(ROOT / "data/processed/label_maps.json")["gender"]["classes"]
    original = read(SAVED / "screen_decision.json")
    audit = read(SAVED / "source_audit.json")["identity"]
    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    same(audit["spec"], spec.to_dict())
    same(audit["baseline_controls"], Task3BaselineConfig(target="gender").to_dict())
    assert audit["split_sha256"] == compute_sha256(ROOT / "data/processed/splits.csv")
    assert audit["folds"] == [0, 4]
    evidence = read_precision_prerequisites(
        ROOT / "reports/task3/gender_precision_result_20260905", root=ROOT
    )
    same(audit["precision_evidence_sha256"], evidence["artifact_sha256"])
    development = get_samples(splits.loc[splits.partition.eq("development")], target="gender")
    image_count = verify_input_images(development, ROOT)
    groups = {name: {} for name in ("G2", "E6", "Drop30 + darkening", "Drop45 + darkening")}
    source_checks, fold_rows, class_rows, robust_rows = [], [], [], []
    prediction_rows = 0
    for directory in sorted((SAVED / "comparison_ieee_v2").iterdir()):
        run_id = directory.name
        if run_id in original["run_ids"].values():
            name, source = "Drop45 + darkening", SAVED / run_id
        elif run_id in PARENT_RUN_IDS:
            name, source = "Drop30 + darkening", PREVIOUS / "saved" / run_id
        elif run_id.startswith("t3_gender_e6_gem_p3_"):
            name, source = "E6", PREVIOUS / "parents" / run_id
        else:
            assert run_id.startswith("t3_gender_v2_g2_translation_")
            name, source = "G2", ROOT / "reports/task3/g2_gate_audit_20260905/drive" / run_id
        run = inspect_gender_run(
            source, registry=registry, splits=splits, classes=classes, root=ROOT
        )
        fold = run["fold"]
        assert fold in (0, 4) and fold not in groups[name]
        _verify_baseline_controls(run["config"], Task3BaselineConfig(target="gender"))
        manifest = read(directory / "evaluation_manifest.json")
        identity = manifest["identity"]
        same(identity["source_sha256"], run["sha256"])
        if name != "Drop45 + darkening":
            same(audit["source_sha256"][run_id], run["sha256"])
        assert identity["run_id"] == run_id and identity["policy"] == POLICY
        same(identity["runtime"], RUNTIME)
        same(manifest["runtime"], RUNTIME)
        assert manifest["verified_input_images"] == image_count == 32773
        assert 0 < manifest["evaluation_peak_memory_bytes"] < 3_000_000_000
        for path, digest in identity["dependencies"].items():
            contents = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)
            assert hashlib.sha256(contents).hexdigest() == digest, path
            assert compute_sha256(resolve_task3_path(path, root=ROOT)) == digest, (
                f"Current evaluation dependency changed: {path}"
            )
        matched = load_ieee_evaluation(
            directory, run=run, splits=splits, classes=classes, identity=identity
        )
        training, validation = (
            get_samples(frame, target="gender") for frame in get_cv_split(splits, fold)
        )
        train = validate_oof(
            pd.read_csv(
                directory / "clean_train_predictions.csv",
                keep_default_na=False,
                float_precision="round_trip",
            ),
            training,
            target="gender",
            classes=classes,
        )
        assert train.run_id.eq(run_id).all()
        train_metrics = oof_metrics(train, classes)
        validation_metrics = oof_metrics(matched["predictions"], classes)
        verify_metric_block(matched["metrics"]["final_train_eval_metrics"], train_metrics)
        verify_metric_block(matched["metrics"], validation_metrics)
        same(matched["metrics"]["final_train_eval_macro_f1"], train_metrics["macro_f1"])
        same(
            matched["metrics"]["final_train_validation_macro_f1_gap"],
            train_metrics["macro_f1"] - validation_metrics["macro_f1"],
        )
        prediction_rows += len(train) + len(validation)
        for corruption in CORE_CORRUPTIONS:
            probabilities = validate_oof(
                pd.read_csv(
                    directory / f"{corruption}_predictions.csv",
                    keep_default_na=False,
                    float_precision="round_trip",
                ),
                validation,
                target="gender",
                classes=classes,
                run_ids_by_fold={fold: run_id},
            )
            calculated = oof_metrics(probabilities, classes)
            row = matched["robustness"].set_index("corruption").loc[corruption]
            same(row.macro_f1, calculated["macro_f1"])
            same(row.macro_f1_change, calculated["macro_f1"] - validation_metrics["macro_f1"])
            prediction_rows += len(probabilities)
            robust_rows.append(
                dict(
                    model=name,
                    fold=fold,
                    corruption=corruption,
                    clean_f1=validation_metrics["macro_f1"],
                    macro_f1=calculated["macro_f1"],
                    induced_change=calculated["macro_f1"] - validation_metrics["macro_f1"],
                )
            )
        fold_rows.append(
            dict(
                model=name,
                fold=fold,
                train_f1=train_metrics["macro_f1"],
                validation_f1=validation_metrics["macro_f1"],
                gap=train_metrics["macro_f1"] - validation_metrics["macro_f1"],
            )
        )
        for t, v in zip(train_metrics["per_class"], validation_metrics["per_class"], strict=True):
            assert t["class_name"] == v["class_name"]
            class_rows.append(
                dict(
                    model=name,
                    fold=fold,
                    class_name=t["class_name"],
                    train_f1=t["f1"],
                    validation_f1=v["f1"],
                    gap=t["f1"] - v["f1"],
                    train_support=t["support"],
                    validation_support=v["support"],
                )
            )
        groups[name][fold] = matched
        source_checks.append(
            dict(
                model=name,
                fold=fold,
                run_id=run_id,
                sha256=run["sha256"],
                config_digest=run["config_digest"],
                checkpoint_unpickled=False,
            )
        )
        print("Verified", name, fold, flush=True)
    assert all(set(group) == {0, 4} for group in groups.values())
    reference_ids = {}
    for label, prefix in (("G2", "t3_gender_v2_g2_translation_"), ("E6", "t3_gender_e6_gem_p3_")):
        ids = [run_id for run_id in audit["source_sha256"] if run_id.startswith(prefix)]
        assert len(ids) == 5
        by_fold = {
            int(registry.loc[registry.run_id.eq(run_id), "validation_fold"].iloc[0]): run_id
            for run_id in ids
        }
        assert set(by_fold) == set(range(5))
        reference_ids[label] = [by_fold[f] for f in range(5)]
    for fold in (0, 4):
        child = groups["Drop45 + darkening"][fold]
        assert child["run_id"] == original["run_ids"][str(fold)]
        assert groups["Drop30 + darkening"][fold]["run_id"] == spec.parent_run_id_for_fold(fold)
        _verify_training_evidence(
            child,
            spec,
            spec.parent_run_id_for_fold(fold),
            evidence,
            audit_sha256=compute_sha256(SAVED / "source_audit.json"),
        )
        _verify_training_evidence(
            groups["Drop30 + darkening"][fold],
            dataset_v2_spec(PREVIOUS_NAME, ORIGINAL_DROPOUT_RUN_IDS),
            ORIGINAL_DROPOUT_RUN_IDS[(0, 4).index(fold)],
            evidence,
            audit_sha256=compute_sha256(PREVIOUS / "saved/source_audit.json"),
        )
        same(
            groups["G2"][fold]["config"]["child_experiment"],
            dataset_v2_spec("gender_v2_translation", reference_ids["E6"]).to_dict(),
        )
        e6_spec = groups["E6"][fold]["config"]["child_experiment"]
        same(e6_spec, gender_gem_p3_spec(e6_spec["parent_run_ids"]).to_dict())
    child = groups["Drop45 + darkening"]
    recomputed = evaluate_gender_narrow_screen(child, groups, classes, experiment_name=NAME)
    for key, value in recomputed.items():
        same(original[key], value, key)
    incremental = compare_with_dropout(child, groups["Drop30 + darkening"], classes)
    incremental["comparison"] = (
        "candidate minus matched IEEE Drop30Dark (dropout 0.30 plus mild darkening); "
        "descriptive, no extra acceptance gates"
    )
    same(original["direct_parent_comparison"], {
        "name": "Drop30Dark",
        "classifier_dropout": 0.30,
        "training_augmentation": spec.training_augmentation,
        "run_ids": {str(f): groups["Drop30 + darkening"][f]["run_id"] for f in (0, 4)},
    })
    same(original["incremental_comparison"], incremental)
    same(read(SAVED / "incremental_comparison.json"), incremental)
    expected = get_samples(development.loc[development.cv_fold.isin([0, 4])], target="gender")
    pooled = validate_oof(
        pd.read_csv(
            SAVED / "ieee_oof_predictions.csv", keep_default_na=False, float_precision="round_trip"
        ),
        expected,
        target="gender",
        classes=classes,
        run_ids_by_fold={f: child[f]["run_id"] for f in (0, 4)},
    )
    verify_metric_block(original["candidate"], oof_metrics(pooled, classes))
    pd.testing.assert_frame_equal(pooled, _pool(child).sort_values("id").reset_index(drop=True))
    folds = pd.DataFrame(fold_rows)
    class_frame = pd.DataFrame(class_rows)
    robust = pd.DataFrame(robust_rows)
    summary = folds.groupby("model")[["train_f1", "validation_f1", "gap"]].mean()
    for name in groups:
        summary.loc[name, "pooled_validation_f1"] = oof_metrics(_pool(groups[name]), classes)[
            "macro_f1"
        ]
    summary.to_csv(HERE / "verified_summary.csv")
    folds.to_csv(HERE / "verified_folds.csv", index=False)
    class_frame.to_csv(HERE / "verified_class_gaps.csv", index=False)
    robust.to_csv(HERE / "verified_robustness.csv", index=False)
    pd.DataFrame(recomputed["checks"]).to_csv(HERE / "verified_gates.csv", index=False)
    verification = dict(
        verified=True,
        training_performed=False,
        inference_performed=False,
        source_bundles_verified=len(source_checks),
        reference_bundles_verified=6,
        reference_bundles_listed_in_source_audit=len(audit["source_sha256"]),
        evaluation_artifact_hashes_verified=72,
        evaluation_prediction_rows_verified=prediction_rows,
        development_image_hashes_verified=image_count,
        pooled_validation_rows_verified=len(pooled),
        bootstrap_draws_per_comparison=10000,
        bootstrap_comparisons_reproduced=2,
        notebook_reported_commit=COMMIT,
        registry_sha256=compute_sha256(HERE / "runs.csv"),
        source_audit_sha256=compute_sha256(SAVED / "source_audit.json"),
        sources=source_checks,
        failed_checks=[row for row in recomputed["checks"] if row["status"] != "pass"],
        limits=[
            "No checkpoint unpickling or fresh inference; model restoration and unchanged state "
            "remain recorded GPU evidence.",
            "Training runtime, timing and peak GPU memory were read from saved records, "
            "not measured locally.",
            "The six G2/E6 reference bundles on folds 1–3 were not independently downloaded "
            "in this review.",
            "Two reused development folds do not remove model-selection bias or supply "
            "independent final-test evidence.",
        ],
    )
    (HERE / "verified_review.json").write_text(json.dumps(verification, indent=2) + "\n")
    (HERE / "recomputed_decision.json").write_text(json.dumps(recomputed, indent=2) + "\n")
    print(summary.to_string(), flush=True)
    print("Failures:", verification["failed_checks"], flush=True)
    plot(summary, class_frame, robust)


def plot(summary, classes, robust):
    names = ["G2", "Drop30 + darkening", "Drop45 + darkening"]
    colors = ["#64748b", "#2563eb", "#c65d20"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    x = np.arange(3)
    for offset, metric, label, color in [
        (-0.18, "train_f1", "Clean training (fold mean)", "#94a3b8"),
        (0.18, "pooled_validation_f1", "Validation (pooled)", "#2563eb"),
    ]:
        bars = axes[0, 0].bar(
            x + offset, summary.loc[names, metric], width=0.34, label=label, color=color
        )
        axes[0, 0].bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    axes[0, 0].set(
        xticks=x,
        xticklabels=[n.replace(' + ', '\n+ ') for n in names],
        ylim=(0, 1.11),
        ylabel="Macro-F1",
        title="Stronger dropout lowers clean fit and validation",
    )
    axes[0, 0].legend(loc="lower left", fontsize=9)
    bars = axes[0, 1].bar(x, summary.loc[names, "gap"], color=colors)
    axes[0, 1].bar_label(bars, fmt="%.3f", padding=3)
    limit = summary.loc["G2", "gap"] - 0.05
    axes[0, 1].axhline(
        limit, color="#b91c1c", linestyle="--", label=f"Required at most {limit:.3f}"
    )
    axes[0, 1].set(
        xticks=x,
        xticklabels=[n.replace(' + ', '\n+ ') for n in names],
        ylim=(0, 0.29),
        ylabel="Train F1 − validation F1",
        title="Mean gap shrinks, but still misses the rule",
    )
    axes[0, 1].legend(loc="lower left", fontsize=9)
    means = robust.groupby(["model", "corruption"]).macro_f1.mean().unstack()
    conditions = ["brightness_085", "grayscale", "translation_003"]
    x = np.arange(len(conditions))
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        bars = axes[1, 0].bar(
            x + (i - 1) * 0.24, means.loc[name, conditions], width=0.23, color=color, label=name
        )
        axes[1, 0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[1, 0].set(
        xticks=x,
        xticklabels=["Darker images", "Grayscale", "Shifted images"],
        ylim=(0, 0.84),
        ylabel="Raw macro-F1 (fold mean)",
        title="Almost no extra raw lighting gain",
    )
    axes[1, 0].legend(loc="lower left", fontsize=9)
    c = (
        classes.loc[classes.model.eq("Drop45 + darkening")]
        .groupby("class_name")[["train_f1", "validation_f1"]]
        .mean()
        .loc[["Boys", "Girls", "Men", "Unisex", "Women"]]
    )
    x = np.arange(5)
    for offset, key, label, color in [
        (-0.17, "train_f1", "Clean training", "#94a3b8"),
        (0.17, "validation_f1", "Validation", "#c65d20"),
    ]:
        axes[1, 1].bar(x + offset, c[key], width=0.32, label=label, color=color)
    axes[1, 1].set(
        xticks=x,
        xticklabels=c.index,
        ylim=(0, 1.08),
        ylabel="Class F1 (fold mean)",
        title="Class gaps remain large",
    )
    axes[1, 1].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Gender stronger dropout (0.45): screen failed\n"
        "Matched IEEE evaluation, canonical folds 0 and 4",
        fontsize=17,
    )
    fig.savefig(HERE / "review.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
