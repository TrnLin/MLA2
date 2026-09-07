"""Diagnose remaining 04ad failures without changing data, models or screen rules."""

from fashion.task3_paths import resolve_task3_path

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_label_maps
from fashion.data.gender_name_truth import VARIANT_RELATIVE_PATH, load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.task3_decisions import (
    CORE_CORRUPTIONS,
    oof_metrics,
    paired_family_bootstrap,
    probability_columns,
    validate_oof,
)

HERE = Path(__file__).resolve().parent
REVIEW = ROOT / "reports/task3/gender_name_truth_result_20260906"
SAVED = REVIEW / "saved"


def read(path):
    return json.loads(path.read_text())


def main():
    splits = load_gender_name_truth_variant(ROOT)
    classes = load_label_maps()["gender"]["classes"]
    labels = pd.read_csv(resolve_task3_path(VARIANT_RELATIVE_PATH, root=ROOT) / "labels.csv", keep_default_na=False)
    meta = splits.merge(labels.drop(columns="gender"), on="id", how="left", validate="one_to_one")
    before = compute_sha256(ROOT / "data/processed/splits.csv")
    verified = read(REVIEW / "verified_review.json")
    sources = {(r["model"], r["fold"]): r["run_id"] for r in verified["sources"]}
    frames, metrics, hashes = {}, {}, {}
    for model in ("G2", "Gray10", "NameTruth"):
        for fold in (0, 4):
            run_id = sources[model, fold]
            directory = SAVED / "comparison_name_truth_ieee" / run_id
            manifest = read(directory / "evaluation_manifest.json")
            assert (
                manifest["identity"]["label_variant"]
                == read(SAVED / "source_audit.json")["identity"]["comparison_label_basis"]
            )
            expected = [get_samples(f, target="gender") for f in get_cv_split(splits, fold)]
            views = [
                ("train", "clean_train_predictions.csv", 0),
                ("validation", "oof_predictions.csv", 1),
            ]
            if model in ("Gray10", "NameTruth"):
                views.extend((c, f"{c}_predictions.csv", 1) for c in CORE_CORRUPTIONS)
            for view, filename, part in views:
                path = directory / filename
                digest = compute_sha256(path)
                assert manifest["files"][filename] == digest
                hashes[str(path.relative_to(ROOT))] = digest
                frame = validate_oof(
                    pd.read_csv(path, keep_default_na=False, float_precision="round_trip"),
                    expected[part],
                    target="gender",
                    classes=classes,
                    run_ids_by_fold={int(f): run_id for f in expected[part].cv_fold.unique()},
                )
                metrics[model, fold, view] = oof_metrics(frame, classes)
                frame = frame.merge(
                    meta.drop(columns=["cv_fold", "path", "product_family_group"]),
                    on="id",
                    validate="one_to_one",
                )
                frame["correct"] = frame.true_label.eq(frame.predicted_label)
                frame["changed"] = frame.changed.astype(bool)
                frame["cue_count"] = frame.cue_count.astype(int)
                frame["true_probability"] = frame[probability_columns(classes)].to_numpy()[
                    np.arange(len(frame)), frame.true_index.to_numpy(dtype=int)
                ]
                frame["margin"] = frame.confidence - frame.true_probability
                frame["model_fold"] = fold
                frames[model, fold, view] = frame
    gap_rows, basis_rows = [], []
    for model in ("G2", "Gray10", "NameTruth"):
        for fold in (0, 4):
            train, val = metrics[model, fold, "train"], metrics[model, fold, "validation"]
            for t, v in zip(train["per_class"], val["per_class"], strict=True):
                gap_rows.append(
                    dict(
                        model=model,
                        fold=fold,
                        class_name=t["class_name"],
                        train_f1=t["f1"],
                        validation_f1=v["f1"],
                        gap=t["f1"] - v["f1"],
                        contribution_to_macro_gap=(t["f1"] - v["f1"]) / 5,
                        train_support=t["support"],
                        validation_support=v["support"],
                    )
                )
            for scope in ("all", "unchanged_labels", "explicit_name_cue"):
                rows = []
                for part in ("train", "validation"):
                    f = frames[model, fold, part]
                    selected = (
                        f
                        if scope == "all"
                        else f[~f.changed]
                        if scope == "unchanged_labels"
                        else f[f.cue_count.eq(1)]
                    )
                    rows.append(oof_metrics(selected, classes))
                basis_rows.append(
                    dict(
                        model=model,
                        fold=fold,
                        scope=scope,
                        train_f1=rows[0]["macro_f1"],
                        validation_f1=rows[1]["macro_f1"],
                        gap=rows[0]["macro_f1"] - rows[1]["macro_f1"],
                        train_rows=rows[0]["support"],
                        validation_rows=rows[1]["support"],
                    )
                )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(HERE / "class_gap_contributions.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(HERE / "gap_by_label_scope.csv", index=False)
    pools = {
        model: pd.concat([frames[model, f, "validation"] for f in (0, 4)], ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
        for model in ("G2", "Gray10", "NameTruth")
    }
    current = pools["NameTruth"].copy()
    unchanged_interval = paired_family_bootstrap(
        pools["NameTruth"][~pools["NameTruth"].changed],
        pools["Gray10"][~pools["Gray10"].changed],
        classes=classes,
        repetitions=10000,
        seed=2753,
    )
    (HERE / "unchanged_label_bootstrap.json").write_text(
        json.dumps(unchanged_interval, indent=2) + "\n"
    )
    current["old_correct"] = pools["Gray10"].correct
    current["old_predicted_label"] = pools["Gray10"].predicted_label
    current["g2_correct"] = pools["G2"].correct
    current["error_status"] = np.select(
        [
            current.correct & ~current.old_correct,
            ~current.correct & current.old_correct,
            ~current.correct & ~current.old_correct,
        ],
        ["recovered", "new_error", "persistent_error"],
        default="still_correct",
    )
    current["has_exact_cross_label_conflict"] = current.sha256.isin(
        pd.read_csv(SAVED / "label_variant/same_image_label_conflicts.csv").sha256
    )
    pd.crosstab(current.true_label, current.predicted_label).reindex(
        index=classes, columns=classes, fill_value=0
    ).to_csv(HERE / "confusion_matrix.csv")
    counts = []
    for column in ("true_label", "articleType", "label_source", "baseColour", "error_status"):
        for value, f in current.groupby(column):
            counts.append(
                dict(
                    dimension=column,
                    value=value,
                    rows=len(f),
                    errors=int((~f.correct).sum()),
                    error_rate=float((~f.correct).mean()),
                    families=f.product_family_group.nunique(),
                    high_confidence_errors=int((~f.correct & f.confidence.ge(0.9)).sum()),
                )
            )
    pd.DataFrame(counts).to_csv(HERE / "error_cohorts.csv", index=False)
    article_rows = []
    for (label, article), f in current.groupby(["true_label", "articleType"]):
        train_counts, class_shares, train_errors, families = [], [], [], []
        for fold in (0, 4):
            train = frames["NameTruth", fold, "train"]
            selected = train[train.true_label.eq(label) & train.articleType.eq(article)]
            train_counts.append(len(selected))
            class_shares.append(len(selected) / max(1, train.articleType.eq(article).sum()))
            train_errors.append(int((~selected.correct).sum()))
            families.append(selected.product_family_group.nunique())
        article_rows.append(
            dict(
                true_label=label,
                articleType=article,
                validation_rows=len(f),
                validation_errors=int((~f.correct).sum()),
                error_rate=float((~f.correct).mean()),
                train_rows_f0=train_counts[0],
                train_rows_f4=train_counts[1],
                train_class_share_f0=class_shares[0],
                train_class_share_f4=class_shares[1],
                train_errors_f0=train_errors[0],
                train_errors_f4=train_errors[1],
                train_families_f0=families[0],
                train_families_f4=families[1],
                validation_families=f.product_family_group.nunique(),
            )
        )
    articles = pd.DataFrame(article_rows).sort_values("validation_errors", ascending=False)
    articles.to_csv(HERE / "class_article_errors.csv", index=False)
    corruption_rows = []
    for corruption in CORE_CORRUPTIONS:
        altered = pd.concat([frames["NameTruth", f, corruption] for f in (0, 4)], ignore_index=True)
        altered = altered.sort_values("id").reset_index(drop=True)
        assert altered.id.equals(current.id)
        for label in classes:
            mask = current.true_label.eq(label)
            cp = oof_metrics(altered, classes)["per_class"][classes.index(label)]
            corruption_rows.append(
                dict(
                    corruption=corruption,
                    class_name=label,
                    support=int(mask.sum()),
                    clean_correct=int((mask & current.correct).sum()),
                    corrupted_correct=int((mask & altered.correct).sum()),
                    newly_wrong=int((mask & current.correct & ~altered.correct).sum()),
                    recovered=int((mask & ~current.correct & altered.correct).sum()),
                    corrupted_f1=cp["f1"],
                )
            )
    pd.DataFrame(corruption_rows).to_csv(HERE / "corruption_class_changes.csv", index=False)
    fold_compare = gaps.pivot(index=["fold", "class_name"], columns="model", values="gap")
    fold_compare["reduction_vs_g2"] = fold_compare.G2 - fold_compare.NameTruth
    fold_compare["reduction_vs_gray10"] = fold_compare.Gray10 - fold_compare.NameTruth
    fold_compare.to_csv(HERE / "gap_reduction_by_class.csv")
    curves = []
    for fold in (0, 4):
        history = pd.read_csv(SAVED / sources["NameTruth", fold] / "history.csv")
        # Registered file hashes were checked in the preceding result review; check again here.
        source_hashes = next(
            r["sha256"]
            for r in verified["sources"]
            if r["model"] == "NameTruth" and r["fold"] == fold
        )
        assert (
            compute_sha256(SAVED / sources["NameTruth", fold] / "history.csv")
            == source_hashes["history.csv"]
        )
        history["fold"] = fold
        curves.append(history)
    history = pd.concat(curves, ignore_index=True)
    history.to_csv(HERE / "learning_curves.csv", index=False)
    # A perfect-fix sensitivity is an upper bound, not an achievable model or an estimate.
    ceiling = []
    baseline = oof_metrics(current, classes)["macro_f1"]
    for label in classes:
        p = current.copy()
        mask = p.true_label.eq(label) & ~p.correct
        for col in probability_columns(classes):
            p.loc[mask, col] = 0.0
        p.loc[mask, probability_columns(classes)[classes.index(label)]] = 1.0
        p.loc[mask, "predicted_label"] = label
        p.loc[mask, "predicted_index"] = classes.index(label)
        p.loc[mask, "confidence"] = 1.0
        ceiling.append(
            dict(
                class_name=label,
                corrected_false_negatives=int(mask.sum()),
                hypothetical_macro_f1=oof_metrics(p, classes)["macro_f1"],
                hypothetical_gain=oof_metrics(p, classes)["macro_f1"] - baseline,
            )
        )
    pd.DataFrame(ceiling).to_csv(HERE / "perfect_fix_sensitivity.csv", index=False)
    errors = current[~current.correct].copy()
    columns = [
        "id",
        "model_fold",
        "true_label",
        "predicted_label",
        "confidence",
        "true_probability",
        "old_predicted_label",
        "error_status",
        "g2_correct",
        "articleType",
        "baseColour",
        "productDisplayName",
        "label_source",
        "changed",
        "has_exact_cross_label_conflict",
        "product_family_group",
        "sha256",
        "path",
    ]
    errors[columns].to_csv(HERE / "remaining_errors.csv", index=False)
    # One example per family, ranked by confidence, within a named failure category.
    cases, used = [], set()
    selections = [
        ("Unisex → Men", "Unisex", "Men"),
        ("Unisex → Women", "Unisex", "Women"),
        ("Boys → Men", "Boys", "Men"),
        ("Girls → Women", "Girls", "Women"),
        ("Men → Unisex", "Men", "Unisex"),
        ("Women → Girls", "Women", "Girls"),
    ]
    for title, truth, prediction in selections:
        candidate = errors[errors.true_label.eq(truth) & errors.predicted_label.eq(prediction)]
        candidate = candidate.sort_values(["confidence", "id"], ascending=[False, True])
        selected = 0
        for _, row in candidate.iterrows():
            if row.product_family_group in used:
                continue
            used.add(row.product_family_group)
            cases.append(dict(category=title, **row[columns].to_dict()))
            selected += 1
            if selected == 2:
                break
    pd.DataFrame(cases).to_csv(HERE / "visual_cases.csv", index=False)
    mean_gaps = gaps[gaps.model.eq("NameTruth")].groupby("class_name").gap.mean()
    summary = dict(
        validation_rows=len(current),
        errors=len(errors),
        correct=int(current.correct.sum()),
        high_confidence_errors=int(errors.confidence.ge(0.9).sum()),
        high_confidence_error_rate=float((~current[current.confidence.ge(0.9)].correct).mean()),
        explicit_name_errors=int(errors.cue_count.eq(1).sum()),
        unclear_name_errors=int(errors.cue_count.ne(1).sum()),
        changed_label_errors=int(errors.changed.sum()),
        unchanged_label_errors=int((~errors.changed).sum()),
        exact_conflict_errors=int(errors.has_exact_cross_label_conflict.sum()),
        three_class_gap_share=float(mean_gaps[["Boys", "Girls", "Unisex"]].sum() / mean_gaps.sum()),
        data_sha256=before,
        verified_prediction_hashes=hashes,
        training_performed=False,
        inference_performed=False,
        interpretation="post-hoc diagnosis on two reused development folds; no new acceptance test",
    )
    assert before == compute_sha256(ROOT / "data/processed/splits.csv")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", {k: v for k, v in summary.items() if k != "verified_prediction_hashes"})
    print("MEAN CLASS GAPS\n", mean_gaps.to_string())
    print("CONFUSIONS\n", pd.crosstab(current.true_label, current.predicted_label).to_string())
    print("TOP CLASS/ARTICLE ERRORS\n", articles.head(18).to_string(index=False))
    print(
        "GAP BY LABEL SCOPE\n",
        pd.DataFrame(basis_rows)
        .groupby(["scope", "model"])[["train_f1", "validation_f1", "gap"]]
        .mean()
        .to_string(),
    )
    plot(gaps, current, history, classes)
    montage(cases)


def plot(gaps, current, history, classes):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    plt.rcParams.update({"font.size": 11})
    ax = axes[0, 0]
    mean = (
        gaps[gaps.model.eq("NameTruth")]
        .groupby("class_name")[["train_f1", "validation_f1"]]
        .mean()
        .loc[classes]
    )
    for i, col in enumerate(("train_f1", "validation_f1")):
        bars = ax.bar(
            np.arange(5) + (i - 0.5) * 0.36,
            mean[col],
            0.36,
            color=["#7396a6", "#258774"][i],
            label=col.replace("_", " "),
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set(
        xticks=np.arange(5),
        xticklabels=classes,
        ylim=(0, 1.09),
        title="New model: class F1 on clean images (fold means)",
    )
    ax.legend(loc="lower left")
    ax = axes[0, 1]
    matrix = pd.crosstab(current.true_label, current.predicted_label).reindex(
        index=classes, columns=classes, fill_value=0
    )
    rate = matrix.div(matrix.sum(axis=1), axis=0)
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{matrix.iloc[i, j]}\n{rate.iloc[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if rate.iloc[i, j] > 0.5 else "black",
                fontsize=9,
            )
    ax.set(
        xticks=range(5),
        xticklabels=classes,
        yticks=range(5),
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="Name-truth label",
        title="Remaining errors: Unisex and adult–child confusion",
    )
    for fold, ax in zip((0, 4), axes[1], strict=True):
        f = history[history.fold.eq(fold)]
        ax.plot(f.epoch, f.train_loss, label="Online train loss (augmented)", color="#7396a6")
        ax.plot(f.epoch, f.validation_loss, label="Clean validation loss", color="#258774")
        ax.axvline(20, color="#b1b1b1", ls="--", lw=1)
        ax.set(
            xlabel="Epoch",
            ylabel="Cross-entropy loss",
            title=f"Fold {fold}: validation gains slow late in training",
        )
        ax.legend(fontsize=9)
    fig.suptitle("What still fails after the label change", fontsize=19)
    fig.savefig(HERE / "diagnosis.png", dpi=160)
    plt.close(fig)


def montage(cases):
    import textwrap

    from PIL import Image

    fig, axes = plt.subplots(3, 4, figsize=(15, 12), layout="constrained")
    for ax, row in zip(axes.flat, cases, strict=True):
        path = resolve_task3_path(row["path"], root=ROOT)
        assert compute_sha256(path) == row["sha256"]
        image = Image.open(path).convert("RGB")
        ax.imshow(image, interpolation="nearest")
        ax.set_title(
            f"{row['category']} | ID {row['id']}\nconfidence {row['confidence']:.1%}", fontsize=11
        )
        name = "\n".join(textwrap.wrap(row["productDisplayName"], 34))
        ax.set_xlabel(name + f"\n{row['label_source']} | {row['error_status']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "High-confidence errors: saved catalog images and name-based labels\n"
        "Two examples per error direction; one per family. "
        "These are selected cases, not a random sample.",
        fontsize=15,
    )
    fig.savefig(HERE / "error_examples.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
