"""Explain remaining class gaps using saved predictions and development metadata."""

from fashion.task3_paths import resolve_task3_path

import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import precision_recall_fscore_support

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.task3_decisions import validate_oof

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/task3/gender_grayscale_result_20260905/saved"
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
FOCUS = ["Girls", "Boys", "Unisex"]
HASHES = {}


def read_predictions(run_id, filename, expected):
    folder = SOURCE / "comparison_ieee_v2" / run_id
    path = folder / filename
    manifest = json.loads((folder / "evaluation_manifest.json").read_text())
    digest = compute_sha256(path)
    assert digest == manifest["files"][filename]
    HASHES[str(path.relative_to(ROOT))] = digest
    frame = validate_oof(
        pd.read_csv(path, keep_default_na=False, float_precision="round_trip"),
        expected,
        target="gender",
        classes=CLASSES,
    )
    assert frame.run_id.eq(run_id).all()
    return frame.sort_values("id").reset_index(drop=True)


def named_class(frame):
    names = frame.productDisplayName.str.lower().str.replace("’", "'", regex=False)
    signals = pd.DataFrame(
        {
            label: names.str.contains(pattern, regex=True)
            for label, pattern in {
                "Boys": r"\bboy(?:s|'s)?\b",
                "Girls": r"\bgirl(?:s|'s)?\b",
                "Men": r"\bmen(?:s|'s)?\b",
                "Women": r"\bwomen(?:s|'s)?\b",
                "Unisex": r"\bunisex\b",
            }.items()
        }
    )
    return signals.idxmax(axis=1).where(signals.sum(axis=1).eq(1), "")


def metrics(frame, predicted, model, fold, scope):
    pr, re, f1, support = precision_recall_fscore_support(
        frame.gender,
        frame[predicted],
        labels=CLASSES,
        zero_division=0,
    )
    return [
        dict(
            model=model,
            fold=fold,
            scope=scope,
            label=label,
            precision=pr[i],
            recall=re[i],
            f1=f1[i],
            support=int(support[i]),
            predicted_count=int(frame[predicted].eq(label).sum()),
        )
        for i, label in enumerate(CLASSES)
    ]


def load():
    splits = load_splits()
    metadata = get_samples(splits, partition="development", target="gender").copy()
    rows = metadata[metadata.cv_fold.isin([0, 4])].sort_values("id").reset_index(drop=True)
    assert len(rows) == 13110 and rows.id.is_unique
    decision = json.loads((SOURCE / "screen_decision.json").read_text())
    ids = {
        "gray10": decision["run_ids"],
        "parent": decision["direct_parent_comparison"]["run_ids"],
    }
    ids["G2"] = {
        str(f): next(
            p.name
            for p in (SOURCE / "comparison_ieee_v2").iterdir()
            if p.name.startswith("t3_gender_v2_g2_translation_") and f"_f{f}_" in p.name
        )
        for f in (0, 4)
    }
    training, all_metrics = {}, []
    for model, runs in ids.items():
        for condition, filename in [
            ("clean", "oof_predictions.csv"),
            ("gray", "grayscale_predictions.csv"),
        ]:
            frame = (
                pd.concat(
                    [
                        read_predictions(runs[str(f)], filename, rows[rows.cv_fold.eq(f)])
                        for f in (0, 4)
                    ]
                )
                .sort_values("id")
                .reset_index(drop=True)
            )
            assert frame.id.tolist() == rows.id.tolist()
            tag = f"{model}_{condition}"
            rows[tag] = frame.predicted_label
            rows[f"{tag}_confidence"] = frame.confidence
            rows[f"{tag}_correct"] = frame.predicted_label.eq(frame.true_label)
            probabilities = frame[[f"probability_{i}_{c}" for i, c in enumerate(CLASSES)]]
            rows[f"{tag}_p_true"] = probabilities.to_numpy()[
                np.arange(len(frame)), frame.true_index.to_numpy()
            ]
            for fold, subset in [
                ("pooled", rows),
                ("0", rows[rows.cv_fold.eq(0)]),
                ("4", rows[rows.cv_fold.eq(4)]),
            ]:
                all_metrics += metrics(subset, tag, model, fold, condition)
        if model != "gray10":
            continue
        for fold in (0, 4):
            train, validation = (
                get_samples(x, target="gender") for x in get_cv_split(splits, fold)
            )
            assert not set(train.product_family_group) & set(validation.product_family_group)
            assert not set(train.sha256) & set(validation.sha256)
            frame = read_predictions(runs[str(fold)], "clean_train_predictions.csv", train)
            frame = train.merge(
                frame[["id", "predicted_label", "confidence"]], on="id", validate="one_to_one"
            )
            frame["correct"] = frame.gender.eq(frame.predicted_label)
            training[fold] = frame
            all_metrics += metrics(frame, "predicted_label", model, str(fold), "clean_train")
    rows["name_signal"] = named_class(rows)
    rows["name_conflict"] = rows.name_signal.ne("") & rows.name_signal.ne(rows.gender)
    rows["name_matches_prediction"] = rows.name_signal.eq(rows.gray10_clean)
    metadata["name_signal"] = named_class(metadata)
    metadata["name_conflict"] = metadata.name_signal.ne("") & metadata.name_signal.ne(
        metadata.gender
    )
    metric_frame = pd.DataFrame(all_metrics)
    metric_frame.to_csv(HERE / "class_metrics.csv", index=False)
    gap = metric_frame[
        metric_frame.model.eq("gray10")
        & metric_frame.fold.ne("pooled")
        & metric_frame.scope.isin(["clean", "clean_train"])
    ].pivot(index=["fold", "label"], columns="scope", values="f1")
    gap["gap"] = gap.clean_train - gap.clean
    contributions = gap.groupby("label")[["clean_train", "clean", "gap"]].mean()
    contributions["contribution_to_macro_gap"] = contributions.gap / len(CLASSES)
    contributions["share_of_macro_gap"] = contributions.gap / contributions.gap.sum()
    contributions.to_csv(HERE / "class_gap_contributions.csv")
    rows.to_csv(HERE / "validation_rows.csv", index=False)
    metadata[metadata.name_conflict].to_csv(HERE / "development_name_conflicts.csv", index=False)
    return rows, metadata, training


def tables(rows, metadata, training):
    confusion, articles, standardized, confidence, support, switches = [], [], [], [], [], []
    for fold, frame in [
        ("pooled", rows),
        ("0", rows[rows.cv_fold.eq(0)]),
        ("4", rows[rows.cv_fold.eq(4)]),
    ]:
        for condition in ("clean", "gray"):
            tag = f"gray10_{condition}"
            for (truth, pred), n in frame.groupby(["gender", tag]).size().items():
                confusion.append(
                    dict(fold=fold, condition=condition, truth=truth, prediction=pred, n=int(n))
                )
        for label in FOCUS:
            sub = frame[frame.gender.eq(label)]
            wrong = sub[~sub.gray10_clean_correct]
            false_positive = frame[frame.gray10_clean.eq(label) & frame.gender.ne(label)]
            confidence.append(
                dict(
                    fold=fold,
                    label=label,
                    n=len(sub),
                    errors=len(wrong),
                    high_confidence_errors=int(wrong.gray10_clean_confidence.ge(0.9).sum()),
                    high_confidence_predictions=int(sub.gray10_clean_confidence.ge(0.9).sum()),
                    false_positives=len(false_positive),
                    false_positives_name_matches_prediction=int(
                        (
                            false_positive.name_conflict & false_positive.name_matches_prediction
                        ).sum()
                    ),
                    false_negatives_name_matches_prediction=int(
                        (wrong.name_conflict & wrong.name_matches_prediction).sum()
                    ),
                    stable_wrong_vs_parent=int(
                        (~sub.parent_clean_correct & ~sub.gray10_clean_correct).sum()
                    ),
                    stable_wrong_all_three=int(
                        (
                            ~sub.parent_clean_correct
                            & ~sub.gray10_clean_correct
                            & ~sub.G2_clean_correct
                        ).sum()
                    ),
                )
            )
            for comparison, left, right in [
                ("training_grayscale_clean", "parent_clean", "gray10_clean"),
                ("remaining_color_sensitivity", "gray10_clean", "gray10_gray"),
            ]:
                old, new = sub[f"{left}_correct"], sub[f"{right}_correct"]
                switches.append(
                    dict(
                        fold=fold,
                        label=label,
                        comparison=comparison,
                        harmed=int((old & ~new).sum()),
                        helped=int((~old & new).sum()),
                        both_wrong=int((~old & ~new).sum()),
                    )
                )
            for article, block in sub.groupby("articleType"):
                errors = block[~block.gray10_clean_correct]
                fp = false_positive[false_positive.articleType.eq(article)]
                destinations = errors.gray10_clean.value_counts().to_dict()
                articles.append(
                    dict(
                        fold=fold,
                        label=label,
                        article=article,
                        n=len(block),
                        families=block.product_family_group.nunique(),
                        errors=len(errors),
                        recall=block.gray10_clean_correct.mean(),
                        false_positives=len(fp),
                        parent_recall=block.parent_clean_correct.mean(),
                        gray_recall=block.gray10_gray_correct.mean(),
                        destinations=json.dumps(destinations, sort_keys=True),
                    )
                )
    for fold, train in training.items():
        validation = rows[rows.cv_fold.eq(fold)]
        for label in CLASSES:
            tr, val = train[train.gender.eq(label)], validation[validation.gender.eq(label)]
            support.append(
                dict(
                    fold=fold,
                    label=label,
                    train_n=len(tr),
                    train_families=tr.product_family_group.nunique(),
                    validation_n=len(val),
                    validation_families=val.product_family_group.nunique(),
                )
            )
            if label not in FOCUS:
                continue
            covered, weighted, unseen, covered_correct = 0, 0.0, 0, 0
            for article, block in val.groupby("articleType"):
                train_slice = tr[tr.articleType.eq(article)]
                support.append(
                    dict(
                        fold=fold,
                        label=label,
                        article=article,
                        train_n=len(train_slice),
                        train_families=train_slice.product_family_group.nunique(),
                        validation_n=len(block),
                        validation_families=block.product_family_group.nunique(),
                    )
                )
                if train_slice.empty:
                    unseen += len(block)
                else:
                    covered += len(block)
                    weighted += len(block) * train_slice.correct.mean()
                    covered_correct += int(block.gray10_clean_correct.sum())
            standardized.append(
                dict(
                    fold=fold,
                    label=label,
                    train_recall=tr.correct.mean(),
                    train_recall_at_validation_article_mix=weighted / covered,
                    validation_recall=val.gray10_clean_correct.mean(),
                    validation_recall_on_covered_articles=covered_correct / covered,
                    validation_n=len(val),
                    unseen_article_n=unseen,
                    train_family_mean_recall=tr.groupby("product_family_group")
                    .correct.mean()
                    .mean(),
                    validation_family_mean_recall=val.groupby("product_family_group")
                    .gray10_clean_correct.mean()
                    .mean(),
                )
            )
    for name, values in [
        ("confusions", confusion),
        ("article_errors", articles),
        ("confidence_and_persistence", confidence),
        ("training_support", support),
        ("recall_composition_check", standardized),
        ("paired_changes", switches),
    ]:
        pd.DataFrame(values).to_csv(HERE / f"{name}.csv", index=False)
    distribution = metadata.groupby("gender").agg(
        n=("id", "size"), families=("product_family_group", "nunique")
    )
    distribution.to_csv(HERE / "development_support.csv")
    pd.concat(
        [
            frame.groupby(["articleType", "gender"])
            .size()
            .rename("n")
            .reset_index()
            .assign(fold=fold)
            for fold, frame in training.items()
        ]
    ).to_csv(HERE / "training_article_gender_counts.csv", index=False)
    training_recall = []
    for fold, frame in training.items():
        grouped = (
            frame.groupby(["gender", "articleType"])
            .agg(n=("id", "size"), correct=("correct", "sum"), recall=("correct", "mean"))
            .reset_index()
            .assign(fold=fold)
        )
        training_recall.append(grouped)
    training_recall = pd.concat(training_recall, ignore_index=True)
    training_recall.to_csv(HERE / "training_article_recall.csv", index=False)
    plot_slices(pd.DataFrame(articles), training_recall)
    print(
        "CONFIDENCE/PERSISTENCE\n",
        pd.DataFrame(confidence).query("fold == 'pooled'").to_string(index=False),
    )
    print(
        "LARGEST ARTICLE ERRORS\n",
        pd.DataFrame(articles)
        .query("fold == 'pooled'")
        .sort_values(["label", "errors"], ascending=[True, False])
        .groupby("label")
        .head(7)
        .to_string(index=False),
    )
    print("RECALL COMPOSITION CHECK\n", pd.DataFrame(standardized).to_string(index=False))


def plot_slices(articles, training_recall):
    choices = {
        "Girls": ["Tops", "Tshirts", "Sandals", "Dresses", "Casual Shoes"],
        "Boys": ["Tshirts", "Shorts", "Casual Shoes", "Jeans"],
        "Unisex": ["Backpacks", "Casual Shoes", "Sunglasses", "Watches", "Socks"],
    }
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    for axis, (label, selected) in zip(axes, choices.items(), strict=True):
        val = articles[articles.fold.eq("pooled") & articles.label.eq(label)].set_index("article")
        train = (
            training_recall[training_recall.gender.eq(label)].groupby("articleType").recall.mean()
        )
        y = np.arange(len(selected))
        axis.barh(
            y - 0.18, train.loc[selected], height=0.34, color="#94a3b8", label="Clean training"
        )
        bars = axis.barh(
            y + 0.18, val.loc[selected, "recall"], height=0.34, color="#2563eb", label="Validation"
        )
        axis.bar_label(
            bars,
            labels=[
                f"{int(val.loc[a, 'n'] - val.loc[a, 'errors'])}/{int(val.loc[a, 'n'])}"
                for a in selected
            ],
            padding=4,
            fontsize=10,
        )
        axis.set(
            yticks=y,
            yticklabels=selected,
            xlim=(0, 1.12),
            xlabel="Recall: fraction of this class found",
            title=label,
        )
        axis.invert_yaxis()
        axis.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Same product types, much lower validation recall\n"
        "Training: fold mean. Validation: pooled; labels show correct / total. "
        "Small slices are descriptive.",
        fontsize=14,
    )
    fig.savefig(HERE / "class_slice_recall.png", dpi=150)
    plt.close(fig)


def render_examples(rows):
    font_path = findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 13)
    title_font = ImageFont.truetype(font_path, 22)
    selections = []

    def diverse(frame, n):
        ranked = frame.sort_values(["gray10_clean_confidence", "id"], ascending=[False, True])
        ranked = ranked.drop_duplicates("product_family_group")
        first = ranked.groupby("articleType", sort=False).head(2).head(n)
        rest = ranked[~ranked.id.isin(first.id)].head(n - len(first))
        return pd.concat([first, rest])

    def sheet(frame, name, title):
        width, height, columns = 350, 315, 4
        canvas = Image.new(
            "RGB", (columns * width, 85 + ((len(frame) + 3) // 4) * height), "#f1f5f9"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), title, fill="#111827", font=title_font)
        draw.text(
            (16, 42),
            "Catalog labels; selected examples, not a random sample. Original images enlarged.",
            fill="#475569",
            font=small,
        )
        for index, row in enumerate(frame.itertuples()):
            x, y = index % columns * width, 85 + index // columns * height
            draw.rounded_rectangle(
                (x + 5, y + 4, x + width - 5, y + height - 4), radius=6, fill="white"
            )
            draw.text(
                (x + 12, y + 12),
                f"ID {row.id} | {row.gender} | fold {row.cv_fold}",
                fill="#111827",
                font=font,
            )
            draw.text((x + 12, y + 34), str(row.articleType), fill="#475569", font=small)
            path = resolve_task3_path(row.path, root=ROOT)
            assert compute_sha256(path) == row.sha256
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                picture = ImageOps.contain(rgb, (120, 160), Image.Resampling.NEAREST)
            canvas.paste(picture, (x + 14, y + 58))
            details = [
                f"Now: {row.gray10_clean}",
                f"Confidence {row.gray10_clean_confidence:.2f}",
                f"Before: {row.parent_clean}",
                f"G2: {row.G2_clean}",
                f"Gray: {row.gray10_gray}",
            ]
            draw.multiline_text(
                (x + 145, y + 66), "\n".join(details), font=small, fill="#1f2937", spacing=7
            )
            lines = textwrap.wrap(str(row.productDisplayName), width=43)[:4]
            draw.multiline_text(
                (x + 12, y + 228), "\n".join(lines), font=small, fill="#1f2937", spacing=2
            )
            selections.append(
                dict(
                    sheet=name,
                    id=row.id,
                    fold=row.cv_fold,
                    label=row.gender,
                    prediction=row.gray10_clean,
                    article=row.articleType,
                    family=row.product_family_group,
                    sha256=row.sha256,
                )
            )
        canvas.save(HERE / f"{name}.png")

    for label in FOCUS:
        wrong = rows[rows.gender.eq(label) & ~rows.gray10_clean_correct]
        sheet(
            diverse(wrong, 12),
            f"{label.lower()}_misses",
            f"{label}: confident wrong predictions across article types",
        )
        fp = rows[rows.gender.ne(label) & rows.gray10_clean.eq(label)]
        sheet(
            diverse(fp, 8),
            f"{label.lower()}_false_positives",
            f"Predicted {label}, but the catalog label differs",
        )
    unisex = rows[rows.gender.eq("Unisex") & rows.gray10_clean_correct]
    sheet(
        diverse(unisex, 12),
        "unisex_correct_controls",
        "Unisex: correct controls across article types",
    )
    conflicts = rows[rows.name_conflict & rows.name_matches_prediction]
    sample = pd.concat(
        [diverse(conflicts[conflicts.gray10_clean.eq(label)], 6) for label in ("Boys", "Girls")]
    )
    sheet(sample, "name_label_conflicts", "Predictions follow child wording; catalog labels differ")
    pd.DataFrame(selections).to_csv(HERE / "visual_selection.csv", index=False)


if __name__ == "__main__":
    HERE.mkdir(exist_ok=True)
    rows, metadata, training = load()
    tables(rows, metadata, training)
    render_examples(rows)
    (HERE / "input_hashes.json").write_text(json.dumps(HASHES, indent=2) + "\n")
