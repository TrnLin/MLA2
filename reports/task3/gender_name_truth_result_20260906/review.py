"""Reproduce the 04ad decision and both label bases from saved probabilities."""

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
from fashion.data.gender_name_truth import VARIANT_RELATIVE_PATH, load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import (
    CORE_CORRUPTIONS,
    oof_metrics,
    paired_family_bootstrap,
    validate_oof,
)
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_diagnostic import verify_input_images
from fashion.train.task3_gender_dropout_darkening import (
    DARKENING_RUN_IDS,
    GRAYSCALE_NAME,
    _verify_training_evidence,
    compare_with_dropout,
)
from fashion.train.task3_gender_ieee import POLICY, load_ieee_evaluation
from fashion.train.task3_gender_name_truth import (
    NAME,
    PARENT_RUN_IDS,
    label_contract,
    name_truth_spec,
)
from fashion.train.task3_gender_narrow import (
    RUNTIME,
    evaluate_gender_narrow_screen,
    read_precision_prerequisites,
)
from fashion.train.task3_gender_weight_decay import _pool, _verify_baseline_controls

HERE = Path(__file__).resolve().parent
SAVED = HERE / "saved"
COMMIT = "854d7703adcae426b584d9161abe963ba80938d3"
GRAY = ROOT / "reports/task3/gender_grayscale_result_20260905/saved"


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


def metric_block(saved, actual):
    for key, value in actual.items():
        same(saved[key], value, key)


def main():
    registry = pd.read_csv(HERE / "runs.csv", keep_default_na=False)
    canonical = load_splits(ROOT / "data/processed/splits.csv")
    splits = load_gender_name_truth_variant(ROOT)
    classes = load_label_maps(ROOT / "data/processed/label_maps.json")["gender"]["classes"]
    contract, spec = label_contract(ROOT), name_truth_spec(ROOT)
    saved = read(SAVED / "screen_decision.json")
    audit = read(SAVED / "source_audit.json")["identity"]
    same(audit["spec"], spec.to_dict())
    same(audit["comparison_label_basis"], contract)
    same(audit["baseline_controls"], Task3BaselineConfig(target="gender").to_dict())
    same(saved["comparison_label_basis"], contract)
    assert audit["folds"] == [0, 4]
    assert audit["split_sha256"] == compute_sha256(ROOT / "data/processed/splits.csv")
    prior_audit = read(GRAY / "source_audit.json")["identity"]
    for run_id, hashes in prior_audit["source_sha256"].items():
        same(audit["source_sha256"][run_id], hashes)
    for file in (resolve_task3_path(VARIANT_RELATIVE_PATH, root=ROOT)).iterdir():
        if file.suffix in {".csv", ".json"}:
            assert compute_sha256(file) == compute_sha256(SAVED / "label_variant" / file.name)
    evidence = read_precision_prerequisites(
        ROOT / "reports/task3/gender_precision_result_20260905",
        root=ROOT,
    )
    same(audit["precision_evidence_sha256"], evidence["artifact_sha256"])
    development = get_samples(splits[splits.partition.eq("development")], target="gender")
    image_count = verify_input_images(development, ROOT)
    diagnostic = read(SAVED / "original_label_diagnostic.json")
    same(diagnostic["name_truth_contract"], contract)
    assert diagnostic["probabilities_and_predictions_unchanged"]
    assert diagnostic["table_sha256"] == compute_sha256(SAVED / "label_basis_comparison.csv")
    assert diagnostic["canonical_split_sha256"] == audit["split_sha256"]
    diagnostic_scores = {(row["run_id"], row["view"]): row for row in diagnostic["scores"]}
    groups = {name: {} for name in ("G2", "E6", "Gray10", "NameTruth")}
    frames, scores, source_checks = {}, {}, []
    rows, class_rows, robust_rows, comparison_rows = [], [], [], []
    prediction_count = 0
    dependencies = {}
    for directory in sorted((SAVED / "comparison_name_truth_ieee").iterdir()):
        run_id = directory.name
        if run_id in saved["run_ids"].values():
            name, source = "NameTruth", SAVED / run_id
        elif run_id in PARENT_RUN_IDS:
            name, source = "Gray10", GRAY / run_id
        elif run_id.startswith("t3_gender_e6_gem_p3_"):
            name = "E6"
            source = (
                ROOT / "reports/task3/gender_dropout_darkening_result_20260905/parents" / run_id
            )
        else:
            assert run_id.startswith("t3_gender_v2_g2_translation_")
            name = "G2"
            source = ROOT / "reports/task3/g2_gate_audit_20260905/drive" / run_id
        run = inspect_gender_run(
            source,
            registry=registry,
            splits=splits if name == "NameTruth" else canonical,
            classes=classes,
            root=ROOT,
        )
        fold = run["fold"]
        assert fold in (0, 4) and fold not in groups[name]
        _verify_baseline_controls(run["config"], Task3BaselineConfig(target="gender"))
        if name == "NameTruth":
            _verify_training_evidence(
                run,
                spec,
                spec.parent_run_id_for_fold(fold),
                evidence,
                audit_sha256=compute_sha256(SAVED / "source_audit.json"),
            )
            same(run["config"]["gender_label_variant"], contract)
            record = registry[registry.run_id.eq(run_id)].iloc[0]
            assert str(record.submission_eligible).lower() == "false"
        else:
            same(audit["source_sha256"][run_id], run["sha256"])
        if name == "Gray10":
            _verify_training_evidence(
                run,
                dataset_v2_spec(GRAYSCALE_NAME, DARKENING_RUN_IDS),
                DARKENING_RUN_IDS[(0, 4).index(fold)],
                evidence,
                audit_sha256=compute_sha256(GRAY / "source_audit.json"),
            )
        manifest = read(directory / "evaluation_manifest.json")
        identity = manifest["identity"]
        same(identity["source_sha256"], run["sha256"])
        same(identity["label_variant"], contract)
        same(identity["runtime"], RUNTIME)
        same(manifest["runtime"], RUNTIME)
        assert identity["policy"] == POLICY and identity["run_id"] == run_id
        assert manifest["verified_input_images"] == image_count == 32773
        assert 0 < manifest["evaluation_peak_memory_bytes"] < 3_000_000_000
        assert diagnostic["source_evaluation_manifest_sha256"][run_id] == compute_sha256(
            directory / "evaluation_manifest.json"
        )
        for path, digest in identity["dependencies"].items():
            if path not in dependencies:
                content = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)
                dependencies[path] = hashlib.sha256(content).hexdigest()
            assert dependencies[path] == digest == compute_sha256(resolve_task3_path(path, root=ROOT)), path
        matched = load_ieee_evaluation(
            directory,
            run=run,
            splits=splits,
            classes=classes,
            identity=identity,
        )
        same(matched["metrics"]["evaluation_label_variant"], contract)
        expected = [get_samples(f, target="gender") for f in get_cv_split(splits, fold)]
        original = [get_samples(f, target="gender") for f in get_cv_split(canonical, fold)]
        for view, filename, part in (
            ("clean_train", "clean_train_predictions.csv", 0),
            ("clean_validation", "oof_predictions.csv", 1),
            *((name, f"{name}_predictions.csv", 1) for name in CORE_CORRUPTIONS),
        ):
            p = validate_oof(
                pd.read_csv(
                    directory / filename, keep_default_na=False, float_precision="round_trip"
                ),
                expected[part],
                target="gender",
                classes=classes,
                run_ids_by_fold={int(f): run_id for f in expected[part].cv_fold.unique()},
            )
            teacher = p.copy()
            teacher["true_label"] = teacher.id.map(original[part].set_index("id").gender)
            teacher["true_index"] = teacher.true_label.map(dict(zip(classes, range(len(classes)))))
            teacher = validate_oof(teacher, original[part], target="gender", classes=classes)
            pd.testing.assert_frame_equal(
                p.drop(columns=["true_label", "true_index"]),
                teacher.drop(columns=["true_label", "true_index"]),
            )
            values = {
                "name_truth": oof_metrics(p, classes),
                "original_teacher": oof_metrics(teacher, classes),
            }
            recorded = diagnostic_scores[run_id, view]
            assert recorded["group"] == name and recorded["fold"] == fold
            same(recorded["scores"], values)
            comparison = dict(group=name, fold=fold, run_id=run_id, view=view)
            for basis, prediction in (("name_truth", p), ("original_teacher", teacher)):
                frames[name, fold, view, basis] = prediction
                scores[name, fold, view, basis] = values[basis]
                for metric in ("macro_f1", "nll", "ece_15"):
                    comparison[f"{basis}_{metric}"] = values[basis][metric]
            comparison_rows.append(comparison)
            prediction_count += len(p)
            if view == "clean_train":
                metric_block(matched["metrics"]["final_train_eval_metrics"], values["name_truth"])
            elif view == "clean_validation":
                metric_block(matched["metrics"], values["name_truth"])
            else:
                row = matched["robustness"].set_index("corruption").loc[view]
                clean = scores[name, fold, "clean_validation", "name_truth"]["macro_f1"]
                same(row.macro_f1, values["name_truth"]["macro_f1"])
                same(row.macro_f1_change, values["name_truth"]["macro_f1"] - clean)
                robust_rows.append(
                    dict(
                        model=name,
                        fold=fold,
                        corruption=view,
                        macro_f1=row.macro_f1,
                        induced_change=row.macro_f1_change,
                    )
                )
        for basis in ("name_truth", "original_teacher"):
            train, val = [
                scores[name, fold, f"clean_{part}", basis] for part in ("train", "validation")
            ]
            gap = train["macro_f1"] - val["macro_f1"]
            if basis == "name_truth":
                same(matched["metrics"]["final_train_eval_macro_f1"], train["macro_f1"])
                same(matched["metrics"]["final_train_validation_macro_f1_gap"], gap)
            rows.append(
                dict(
                    model=name,
                    fold=fold,
                    basis=basis,
                    train_f1=train["macro_f1"],
                    validation_f1=val["macro_f1"],
                    gap=gap,
                )
            )
            for t, v in zip(train["per_class"], val["per_class"], strict=True):
                class_rows.append(
                    dict(
                        model=name,
                        fold=fold,
                        basis=basis,
                        class_name=t["class_name"],
                        train_f1=t["f1"],
                        validation_f1=v["f1"],
                        gap=t["f1"] - v["f1"],
                    )
                )
        groups[name][fold] = matched
        source_checks.append(dict(model=name, fold=fold, run_id=run_id, sha256=run["sha256"]))
        print("Verified", name, fold, flush=True)
    assert all(set(group) == {0, 4} for group in groups.values())
    assert len(diagnostic_scores) == len(comparison_rows) == 56
    actual_table = (
        pd.DataFrame(comparison_rows).sort_values(["run_id", "view"]).reset_index(drop=True)
    )
    original_table = (
        pd.read_csv(SAVED / "label_basis_comparison.csv")
        .sort_values(["run_id", "view"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(original_table, actual_table[original_table.columns], atol=1e-10)
    child = groups["NameTruth"]
    result = evaluate_gender_narrow_screen(child, groups, classes, experiment_name=NAME)
    for key, value in result.items():
        same(saved[key], value, key)
    incremental = compare_with_dropout(child, groups["Gray10"], classes)
    incremental["comparison"] = (
        "candidate minus completed Gray10; both evaluated on name-truth labels"
    )
    same(saved["incremental_comparison"], incremental)
    same(read(SAVED / "incremental_comparison.json"), incremental)
    same(saved["direct_parent_run_ids"], {str(f): groups["Gray10"][f]["run_id"] for f in (0, 4)})
    expected = get_samples(development[development.cv_fold.isin([0, 4])], target="gender")
    pooled = validate_oof(
        pd.read_csv(
            SAVED / "ieee_oof_predictions.csv", keep_default_na=False, float_precision="round_trip"
        ),
        expected,
        target="gender",
        classes=classes,
        run_ids_by_fold={f: child[f]["run_id"] for f in (0, 4)},
    )
    pd.testing.assert_frame_equal(pooled, _pool(child).sort_values("id").reset_index(drop=True))
    folds = pd.DataFrame(rows)
    summaries, pooled_class = [], []
    pooled_frames = {}
    for name in groups:
        for basis in ("name_truth", "original_teacher"):
            pool = pd.concat([frames[name, f, "clean_validation", basis] for f in (0, 4)])
            pool = pool.sort_values("id").reset_index(drop=True)
            pooled_frames[name, basis] = pool
            metrics = oof_metrics(pool, classes)
            means = folds[(folds.model == name) & (folds.basis == basis)][
                ["train_f1", "gap"]
            ].mean()
            summaries.append(
                dict(
                    model=name,
                    basis=basis,
                    pooled_validation_f1=metrics["macro_f1"],
                    mean_train_f1=means.train_f1,
                    mean_gap=means.gap,
                    nll=metrics["nll"],
                    ece=metrics["ece_15"],
                )
            )
            pooled_class.extend(dict(model=name, basis=basis, **r) for r in metrics["per_class"])
    teacher_interval = paired_family_bootstrap(
        pooled_frames["NameTruth", "original_teacher"],
        pooled_frames["Gray10", "original_teacher"],
        classes=classes,
        repetitions=10000,
        seed=2753,
    )
    changes = []
    old, new = pooled_frames["Gray10", "name_truth"], pooled_frames["NameTruth", "name_truth"]
    changed_ids = set(splits.loc[splits.gender.ne(canonical.gender), "id"])
    for label in ("all", "changed_labels", "unchanged_labels", *classes):
        mask = (
            pd.Series(True, index=old.index)
            if label == "all"
            else (
                old.id.isin(changed_ids)
                if label == "changed_labels"
                else ~old.id.isin(changed_ids)
                if label == "unchanged_labels"
                else old.true_label.eq(label)
            )
        )
        before, after = (
            old.predicted_label.eq(old.true_label),
            new.predicted_label.eq(new.true_label),
        )
        changes.append(
            dict(
                group=label,
                rows=int(mask.sum()),
                old_correct=int((mask & before).sum()),
                new_correct=int((mask & after).sum()),
                recovered=int((mask & ~before & after).sum()),
                newly_wrong=int((mask & before & ~after).sum()),
            )
        )
    summary, class_frame, robust = (
        pd.DataFrame(summaries),
        pd.DataFrame(pooled_class),
        pd.DataFrame(robust_rows),
    )
    for file, frame in (
        ("verified_summary.csv", summary),
        ("verified_folds.csv", folds),
        ("verified_class_f1.csv", class_frame),
        ("verified_class_gaps.csv", pd.DataFrame(class_rows)),
        ("verified_robustness.csv", robust),
        ("verified_prediction_changes.csv", pd.DataFrame(changes)),
        ("verified_gates.csv", pd.DataFrame(result["checks"])),
    ):
        frame.to_csv(HERE / file, index=False)
    verification = dict(
        verified=True,
        training_performed=False,
        inference_performed=False,
        source_bundles_verified=len(source_checks),
        evaluation_hashes_verified=80,
        prediction_rows_verified=prediction_count,
        image_hashes_verified=image_count,
        source_audit_listed_runs=len(audit["source_sha256"]),
        code_commit=COMMIT,
        sources=source_checks,
        original_teacher_interval=teacher_interval,
        failed_checks=[r for r in result["checks"] if r["status"] != "pass"],
        limits=[
            "No checkpoint was unpickled and no fresh GPU inference was performed.",
            "Timing, GPU memory, and unchanged model state are recorded GPU evidence.",
            "Eight run bundles used in comparisons were rechecked locally; "
            "other audit-listed bundles were not all downloaded.",
            "These reused development folds are not an independent final test.",
        ],
    )
    (HERE / "verified_review.json").write_text(json.dumps(verification, indent=2) + "\n")
    (HERE / "recomputed_decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("Teacher-label interval:", teacher_interval, flush=True)
    print("Prediction changes:", changes, flush=True)
    print("Failed checks:", verification["failed_checks"], flush=True)
    plot(summary, class_frame, robust)


def plot(summary, classes, robust):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), layout="constrained")
    models, colors = (
        ["G2", "E6", "Gray10", "NameTruth"],
        ["#8998a5", "#b5bcc2", "#e4a04b", "#238878"],
    )
    names = ["G2", "E6", "Old grayscale", "New labels"]
    ax = axes[0, 0]
    for i, basis in enumerate(("original_teacher", "name_truth")):
        values = (
            summary[summary.basis.eq(basis)].set_index("model").loc[models, "pooled_validation_f1"]
        )
        bars = ax.bar(
            np.arange(4) + (i - 0.5) * 0.36,
            values,
            0.36,
            label=basis.replace("_", " "),
            color="#9eb7c4" if i == 0 else "#238878",
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set(
        xticks=np.arange(4),
        xticklabels=names,
        ylim=(0, 0.9),
        title="Validation F1: keep each label basis separate",
    )
    ax.legend(loc="lower left", fontsize=9)
    ax = axes[0, 1]
    s = summary[summary.basis.eq("name_truth")].set_index("model").loc[models]
    ax.plot(names, s.mean_train_f1, "o-", color="#64788a", label="Mean clean training F1")
    ax.plot(names, s.pooled_validation_f1, "o-", color="#238878", label="Pooled validation F1")
    for i, gap in enumerate(s.mean_gap):
        ax.text(i, 0.60, f"Mean gap\n{gap:.3f}", ha="center", fontsize=10)
    ax.set(ylim=(0.55, 1.025), title="New label basis: a train–validation gap remains")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.17), fontsize=9)
    ax = axes[1, 0]
    selected = classes[classes.basis.eq("name_truth")]
    class_names = ["Boys", "Girls", "Men", "Unisex", "Women"]
    for i, model in enumerate(("Gray10", "NameTruth")):
        values = selected[selected.model.eq(model)].set_index("class_name").loc[class_names, "f1"]
        bars = ax.bar(
            np.arange(5) + (i - 0.5) * 0.36, values, 0.36, color=colors[2 + i], label=names[2 + i]
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set(
        xticks=np.arange(5),
        xticklabels=class_names,
        ylim=(0, 1.05),
        title="Clean class F1 on the same new labels",
    )
    ax.legend(loc="lower left", fontsize=9)
    ax = axes[1, 1]
    conditions = list(CORE_CORRUPTIONS)
    for i, model in enumerate(("Gray10", "NameTruth")):
        values = (
            robust[robust.model.eq(model)].groupby("corruption").macro_f1.mean().loc[conditions]
        )
        bars = ax.bar(
            np.arange(5) + (i - 0.5) * 0.36, values, 0.36, color=colors[2 + i], label=names[2 + i]
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    labels = {
        "brightness_085": "Darker",
        "brightness_115": "Brighter",
        "translation_003": "Shifted",
        "jpeg_75": "JPEG",
        "grayscale": "Grayscale",
    }
    ax.set(
        xticks=np.arange(5),
        xticklabels=[labels[c] for c in conditions],
        ylim=(0, 0.95),
        title="Raw corrupted-image F1 (fold mean, new labels)",
    )
    ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Name-based labels help prediction; 17 of 19 screen checks pass", fontsize=18)
    fig.savefig(HERE / "review.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
