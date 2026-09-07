"""Reproduce the 04af decision, MixUp receipts and direct-parent error changes."""

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
from fashion.train.mixup import training_contract
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
from fashion.train.task3_gender_mixup import (
    NAME,
    PARENT_RUN_IDS,
    mixup_spec,
    verify_mixup_evidence,
)
from fashion.train.task3_gender_name_truth import PARENT_RUN_IDS as GRAY_RUN_IDS
from fashion.train.task3_gender_name_truth import (
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
COMMIT = "68fef49ab1d55d671531113a71a3e71400a0e3fc"
GRAY = ROOT / "reports/task3/gender_grayscale_result_20260905/saved"
PARENT = ROOT / "reports/task3/gender_name_truth_result_20260906/saved"


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
    contract, spec = label_contract(ROOT), mixup_spec(ROOT)
    saved = read(SAVED / "screen_decision.json")
    audit = read(SAVED / "source_audit.json")["identity"]
    same(audit["spec"], spec.to_dict())
    same(audit["comparison_label_basis"], contract)
    same(audit["baseline_controls"], Task3BaselineConfig(target="gender").to_dict())
    same(saved["comparison_label_basis"], contract)
    assert audit["folds"] == [0, 4]
    assert audit["split_sha256"] == compute_sha256(ROOT / "data/processed/splits.csv")
    prior_audit = read(PARENT / "source_audit.json")["identity"]
    for run_id, hashes in prior_audit["source_sha256"].items():
        same(audit["source_sha256"][run_id], hashes)
    for name, expected_digest in audit["mixup_implementation_sha256"].items():
        content = subprocess.check_output(
            ["git", "show", f"{COMMIT}:src/fashion/train/{name}"], cwd=ROOT
        )
        assert hashlib.sha256(content).hexdigest() == expected_digest
    for fold in (0, 4):
        train = get_samples(get_cv_split(splits, fold)[0], target="gender")
        same(audit["mixup_contracts"][str(fold)], training_contract(train, validation_fold=fold))
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
    groups = {name: {} for name in ("G2", "E6", "Gray10", "NameTruth", "MixUp")}
    frames, scores, source_checks = {}, {}, []
    rows, class_rows, robust_rows, comparison_rows = [], [], [], []
    prediction_count = 0
    dependencies = {}
    for directory in sorted((SAVED / "comparison_name_truth_ieee").iterdir()):
        run_id = directory.name
        if run_id in saved["run_ids"].values():
            name, source = "MixUp", SAVED / run_id
        elif run_id in PARENT_RUN_IDS:
            name, source = "NameTruth", PARENT / run_id
        elif run_id in GRAY_RUN_IDS:
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
            splits=splits if name in {"NameTruth", "MixUp"} else canonical,
            classes=classes,
            root=ROOT,
        )
        fold = run["fold"]
        assert fold in (0, 4) and fold not in groups[name]
        _verify_baseline_controls(run["config"], Task3BaselineConfig(target="gender"))
        if name == "MixUp":
            verify_mixup_evidence(run, fold=fold, splits=splits, directory=source)
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
        if name == "NameTruth":
            old_spec = name_truth_spec(ROOT)
            _verify_training_evidence(
                run,
                old_spec,
                old_spec.parent_run_id_for_fold(fold),
                evidence,
                audit_sha256=compute_sha256(PARENT / "source_audit.json"),
            )
            same(run["config"]["gender_label_variant"], contract)
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
            assert dependencies[path] == digest, path
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
    assert len(diagnostic_scores) == len(comparison_rows) == 70
    actual_table = (
        pd.DataFrame(comparison_rows).sort_values(["run_id", "view"]).reset_index(drop=True)
    )
    original_table = (
        pd.read_csv(SAVED / "label_basis_comparison.csv")
        .sort_values(["run_id", "view"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(original_table, actual_table[original_table.columns], atol=1e-10)
    child = groups["MixUp"]
    result = evaluate_gender_narrow_screen(child, groups, classes, experiment_name=NAME)
    for key, value in result.items():
        same(saved[key], value, key)
    incremental = compare_with_dropout(child, groups["NameTruth"], classes)
    incremental["comparison"] = (
        "candidate minus completed NameTruth; both evaluated on name-truth labels"
    )
    same(saved["incremental_comparison"], incremental)
    same(read(SAVED / "incremental_comparison.json"), incremental)
    same(saved["direct_parent_run_ids"], {str(f): groups["NameTruth"][f]["run_id"] for f in (0, 4)})
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
        pooled_frames["MixUp", "original_teacher"],
        pooled_frames["NameTruth", "original_teacher"],
        classes=classes,
        repetitions=10000,
        seed=2753,
    )
    changes = []
    old, new = pooled_frames["NameTruth", "name_truth"], pooled_frames["MixUp", "name_truth"]
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
        evaluation_hashes_verified=sum(
            len(read(d / "evaluation_manifest.json")["files"])
            for d in (SAVED / "comparison_name_truth_ieee").iterdir()
        ),
        prediction_rows_verified=prediction_count,
        image_hashes_verified=image_count,
        source_audit_listed_runs=len(audit["source_sha256"]),
        code_commit=COMMIT,
        current_workspace_dependencies_different_from_run=[
            p for p, sha in dependencies.items() if compute_sha256(resolve_task3_path(p, root=ROOT)) != sha
        ],
        sources=source_checks,
        original_teacher_interval=teacher_interval,
        failed_checks=[r for r in result["checks"] if r["status"] != "pass"],
        limits=[
            "No checkpoint was unpickled and no fresh GPU inference was performed.",
            "Timing, GPU memory, and unchanged model state are recorded GPU evidence.",
            "Ten run bundles used in comparisons were rechecked locally; "
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
    analyze_errors(old, new, frames, classes, splits, canonical)
    plot(summary, class_frame, robust)


def analyze_errors(old, new, frames, classes, splits, canonical):
    assert old.id.tolist() == new.id.tolist()
    joined = splits.set_index("id").loc[old.id].reset_index()
    joined["parent_prediction"] = old.predicted_label.to_numpy()
    joined["mixup_prediction"] = new.predicted_label.to_numpy()
    joined["confidence"] = new.confidence.to_numpy()
    joined["parent_correct"] = old.predicted_label.eq(old.true_label).to_numpy()
    joined["mixup_correct"] = new.predicted_label.eq(new.true_label).to_numpy()
    joined["recovered"] = ~joined.parent_correct & joined.mixup_correct
    joined["newly_wrong"] = joined.parent_correct & ~joined.mixup_correct
    joined["persistent_error"] = ~joined.parent_correct & ~joined.mixup_correct
    joined["label_changed"] = joined.id.isin(splits.loc[splits.gender.ne(canonical.gender), "id"])
    groups = (
        joined.groupby(["gender", "articleType"], sort=True)
        .agg(
            rows=("id", "size"),
            parent_correct=("parent_correct", "sum"),
            mixup_correct=("mixup_correct", "sum"),
            recovered=("recovered", "sum"),
            newly_wrong=("newly_wrong", "sum"),
            persistent_error=("persistent_error", "sum"),
        )
        .reset_index()
    )
    groups["net_correct"] = groups.mixup_correct - groups.parent_correct
    groups["parent_errors"] = groups.rows - groups.parent_correct
    groups["mixup_errors"] = groups.rows - groups.mixup_correct
    groups.to_csv(HERE / "gender_article_error_changes.csv", index=False)
    joined.to_csv(HERE / "paired_validation_errors.csv", index=False)
    matrices = {}
    for model, column in (
        ("NameTruth", "parent_prediction"),
        ("MixUp", "mixup_prediction"),
    ):
        matrix = pd.crosstab(joined.gender, joined[column]).reindex(
            index=classes, columns=classes, fill_value=0
        )
        matrix.to_csv(HERE / f"confusion_{model}.csv")
        matrices[model] = matrix
    print(
        "GROUP ERROR CHANGES", groups.sort_values("net_correct").to_string(index=False), flush=True
    )
    high_conf = joined.loc[joined.confidence.ge(0.9)]
    detail = {
        "validation_rows": len(joined),
        "parent_errors": int((~joined.parent_correct).sum()),
        "mixup_errors": int((~joined.mixup_correct).sum()),
        "recovered": int(joined.recovered.sum()),
        "newly_wrong": int(joined.newly_wrong.sum()),
        "persistent_errors": int(joined.persistent_error.sum()),
        "high_confidence_errors": int((~high_conf.mixup_correct).sum()),
        "high_confidence_rows": len(high_conf),
    }
    (HERE / "error_summary.json").write_text(json.dumps(detail, indent=2) + "\n")
    class_robust = []
    for model in ("NameTruth", "MixUp"):
        clean = pd.concat(
            [frames[model, f, "clean_validation", "name_truth"] for f in (0, 4)]
        ).sort_values("id")
        for view in CORE_CORRUPTIONS:
            p = pd.concat([frames[model, f, view, "name_truth"] for f in (0, 4)]).sort_values("id")
            assert p.id.tolist() == clean.id.tolist()
            ok = p.predicted_label.eq(p.true_label).to_numpy()
            was_ok = clean.predicted_label.eq(clean.true_label).to_numpy()
            scores = oof_metrics(p, classes)
            for row in scores["per_class"]:
                mask = p.true_label.eq(row["class_name"]).to_numpy()
                class_robust.append(
                    dict(
                        model=model,
                        view=view,
                        **row,
                        newly_wrong=int((mask & was_ok & ~ok).sum()),
                        recovered=int((mask & ~was_ok & ok).sum()),
                    )
                )
    pd.DataFrame(class_robust).to_csv(HERE / "corruption_class_changes.csv", index=False)
    plot_examples(joined)


def plot_examples(rows):
    from PIL import Image

    cases = []
    used = set()
    for action in ("recovered", "newly_wrong"):
        subset = rows[rows[action]].sort_values(["confidence", "id"], ascending=[False, True])
        for gender in ("Unisex", "Boys", "Girls", "Men", "Women"):
            available = subset[(subset.gender == gender) & ~subset.product_family_group.isin(used)]
            if not available.empty:
                row = available.iloc[0].copy()
                row["action"] = action
                cases.append(row)
                used.add(row.product_family_group)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, cases):
        path = resolve_task3_path(row.path, root=ROOT)
        assert compute_sha256(path) == row.sha256
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"), interpolation="nearest")
        ax.set_title(
            f"{row.action.replace('_', ' ')} · {row.id}\n"
            f"True: {row.gender} · {row.articleType}\n"
            f"{row.parent_prediction} → {row.mixup_prediction}",
            fontsize=9,
        )
    fig.suptitle(
        "Selected changes after MixUp — one high-confidence example per class and direction",
        fontsize=13,
    )
    fig.savefig(HERE / "error_examples.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(cases).to_csv(HERE / "visual_cases.csv", index=False)


def plot(summary, classes, robust):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    pair = ("NameTruth", "MixUp")
    labels = ("Name-label parent", "MixUp alpha 0.2")
    colors = ("#8c9aa5", "#168879")
    selected = classes[classes.basis.eq("name_truth")]
    class_names = ["Boys", "Girls", "Men", "Unisex", "Women"]
    for ax, metric, title in (
        (axes[0, 0], "f1", "Class F1 — most gains are Boys and Girls"),
        (axes[0, 1], "recall", "Class recall — which true labels are recovered"),
    ):
        for index, model in enumerate(pair):
            values = (
                selected[selected.model.eq(model)].set_index("class_name").loc[class_names, metric]
            )
            bars = ax.bar(
                np.arange(5) + (index - 0.5) * 0.36,
                values,
                0.36,
                label=labels[index],
                color=colors[index],
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        ax.set(xticks=np.arange(5), xticklabels=class_names, ylim=(0, 1.06), title=title)
        ax.legend(loc="lower left", fontsize=9)
    ax = axes[1, 0]
    table = summary[summary.basis.eq("name_truth")].set_index("model").loc[list(pair)]
    ax.plot(labels, table.mean_train_f1, "o-", color="#64788a", label="Mean clean training F1")
    ax.plot(labels, table.pooled_validation_f1, "o-", color="#168879", label="Pooled validation F1")
    for index, row in enumerate(table.itertuples()):
        ax.text(index, 0.69, f"Mean clean gap\n{row.mean_gap:.3f}", ha="center")
        ax.annotate(
            f"{row.mean_train_f1:.3f}",
            (index, row.mean_train_f1),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
        ax.annotate(
            f"{row.pooled_validation_f1:.3f}",
            (index, row.pooled_validation_f1),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    ax.set(ylim=(0.65, 1.04), title="Clean gap shrinks; validation F1 rises")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.70), fontsize=9)
    ax = axes[1, 1]
    conditions = list(CORE_CORRUPTIONS)
    for index, model in enumerate(pair):
        values = (
            robust[robust.model.eq(model)].groupby("corruption").macro_f1.mean().loc[conditions]
        )
        bars = ax.bar(
            np.arange(5) + (index - 0.5) * 0.36,
            values,
            0.36,
            label=labels[index],
            color=colors[index],
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set(
        xticks=np.arange(5),
        xticklabels=["JPEG", "Dark", "Bright", "Shift", "Gray"],
        ylim=(0, 1),
        title="Mean corrupted F1 — grayscale is slightly worse",
    )
    ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("04af: MixUp alpha 0.2 on the same name-truth labels and folds 0 + 4", fontsize=15)
    fig.savefig(HERE / "review.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
