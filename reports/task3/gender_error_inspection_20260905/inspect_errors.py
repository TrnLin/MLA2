"""Inspect matched saved gender errors on canonical validation folds 0 and 4."""

from fashion.task3_paths import resolve_task3_path

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.font_manager import findfont
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import precision_recall_fscore_support

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/task3/gender_stronger_dropout_result_20260905/saved"
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
PREFIXES = {
    "D30": "t3_gender_dropout_030_mild_darkening_",
    "D45": "t3_gender_dropout_045_mild_darkening_",
    "G2": "t3_gender_v2_g2_translation_",
    "E6": "t3_gender_e6_gem_p3_",
}


def load():
    metadata = pd.read_csv(ROOT / "data/processed/splits.csv", keep_default_na=False)
    expected = metadata.loc[
        metadata.partition.eq("development")
        & metadata.cv_fold.astype(str).isin(["0", "4"])
        & metadata.has_gender_label
    ].copy()
    expected.cv_fold = expected.cv_fold.astype(int)
    expected = expected.sort_values("id").reset_index(drop=True)
    assert len(expected) == 13110 and expected.id.is_unique
    hashes = {}
    for model, prefix in PREFIXES.items():
        folders = sorted((SOURCE / "comparison_ieee_v2").glob(prefix + "*"))
        assert len(folders) == 2
        for condition, filename in [
            ("clean", "oof_predictions.csv"),
            ("gray", "grayscale_predictions.csv"),
        ]:
            frames = []
            for folder in folders:
                path = folder / filename
                manifest = json.loads((folder / "evaluation_manifest.json").read_text())
                # All artifacts were verified by the preceding result review. Record exact inputs.
                hashes[str(path.relative_to(ROOT))] = compute_sha256(path)
                assert hashes[str(path.relative_to(ROOT))] == manifest["files"][filename]
                assert manifest["identity"]["run_id"] == folder.name
                frames.append(
                    pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
                )
            frame = pd.concat(frames).sort_values("id").reset_index(drop=True)
            assert frame.id.is_unique and frame.id.tolist() == expected.id.tolist()
            assert frame.cv_fold.tolist() == expected.cv_fold.tolist()
            assert frame.true_label.tolist() == expected.gender.tolist()
            assert frame.product_family_group.tolist() == expected.product_family_group.tolist()
            tag = f"{model}_{condition}"
            expected[tag] = frame.predicted_label
            expected[f"{tag}_confidence"] = frame.confidence
            expected[f"{tag}_correct"] = frame.predicted_label.eq(frame.true_label)
            probs = frame[
                [f"probability_{i}_{label}" for i, label in enumerate(CLASSES)]
            ].to_numpy()
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
            assert np.isfinite(probs).all() and (probs >= 0).all()
            expected[f"{tag}_p_true"] = probs[np.arange(len(frame)), frame.true_index]
    for model in PREFIXES:
        expected[f"{model}_gray_harm"] = (
            expected[f"{model}_clean_correct"] & ~expected[f"{model}_gray_correct"]
        )
        expected[f"{model}_gray_help"] = (
            ~expected[f"{model}_clean_correct"] & expected[f"{model}_gray_correct"]
        )
    expected["dropout_harm"] = expected.D30_clean_correct & ~expected.D45_clean_correct
    expected["dropout_help"] = ~expected.D30_clean_correct & expected.D45_clean_correct
    (HERE / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    expected.to_csv(HERE / "matched_validation_rows.csv", index=False)
    return expected, metadata


def tables(rows, metadata):
    metrics, confusion, switches, slices = [], [], [], []
    for fold, frame in [
        ("pooled", rows),
        ("0", rows[rows.cv_fold.eq(0)]),
        ("4", rows[rows.cv_fold.eq(4)]),
    ]:
        for model in PREFIXES:
            for condition in ("clean", "gray"):
                tag = f"{model}_{condition}"
                pr, re, f1, support = precision_recall_fscore_support(
                    frame.gender, frame[tag], labels=CLASSES, zero_division=0
                )
                for i, label in enumerate(CLASSES):
                    metrics.append(
                        dict(
                            fold=fold,
                            model=model,
                            condition=condition,
                            label=label,
                            support=int(support[i]),
                            precision=pr[i],
                            recall=re[i],
                            f1=f1[i],
                            predicted_count=int(frame[tag].eq(label).sum()),
                        )
                    )
                for (truth, prediction), n in frame.groupby(["gender", tag]).size().items():
                    confusion.append(
                        dict(
                            fold=fold,
                            model=model,
                            condition=condition,
                            truth=truth,
                            prediction=prediction,
                            count=n,
                        )
                    )
        comparisons = [
            ("dropout_clean", "D30_clean", "D45_clean"),
            ("dropout_gray", "D30_gray", "D45_gray"),
        ]
        comparisons += [(f"{m}_grayscale", f"{m}_clean", f"{m}_gray") for m in PREFIXES]
        for name, old, new in comparisons:
            for label in CLASSES:
                sub = frame[frame.gender.eq(label)]
                old_ok, new_ok = sub[f"{old}_correct"], sub[f"{new}_correct"]
                switches.append(
                    dict(
                        fold=fold,
                        comparison=name,
                        label=label,
                        support=len(sub),
                        both_correct=int((old_ok & new_ok).sum()),
                        harmed=int((old_ok & ~new_ok).sum()),
                        helped=int((~old_ok & new_ok).sum()),
                        both_wrong=int((~old_ok & ~new_ok).sum()),
                    )
                )
    for keys, sub in rows.groupby(["gender", "articleType"]):
        slices.append(
            dict(
                label=keys[0],
                articleType=keys[1],
                n=len(sub),
                families=sub.product_family_group.nunique(),
                dropout_harm=int(sub.dropout_harm.sum()),
                dropout_help=int(sub.dropout_help.sum()),
                **{
                    f"{m}_{c}_recall": sub[f"{m}_{c}_correct"].mean()
                    for m in PREFIXES
                    for c in ("clean", "gray")
                },
                **{f"{m}_gray_harm": int(sub[f"{m}_gray_harm"].sum()) for m in PREFIXES},
            )
        )
    for name, data in [
        ("class_metrics", metrics),
        ("confusions", confusion),
        ("paired_switches", switches),
        ("article_slices", slices),
    ]:
        pd.DataFrame(data).to_csv(HERE / f"{name}.csv", index=False)
    training = []
    for fold in (0, 4):
        sub = metadata[
            metadata.partition.eq("development")
            & ~metadata.cv_fold.astype(str).eq(str(fold))
            & metadata.has_gender_label
        ]
        for (label, article), block in sub.groupby(["gender", "articleType"]):
            training.append(
                dict(
                    fold=fold,
                    label=label,
                    articleType=article,
                    n=len(block),
                    families=block.product_family_group.nunique(),
                )
            )
    pd.DataFrame(training).to_csv(HERE / "training_support.csv", index=False)
    cm = pd.DataFrame(metrics)
    means = cm[~cm.fold.eq("pooled")].groupby(["model", "label", "condition"]).f1.mean().unstack()
    means["induced_f1_change"] = means.gray - means.clean
    means["induced_difference_vs_e6"] = (
        means.induced_f1_change
        - means.loc["E6", "induced_f1_change"]
        .reindex(means.index.get_level_values("label"))
        .to_numpy()
    )
    means["contribution_to_macro_gate"] = means.induced_difference_vs_e6 / len(CLASSES)
    means.to_csv(HERE / "grayscale_gate_decomposition.csv")
    print("POOLED CLASS SCORES\n", cm[cm.fold.eq("pooled")].to_string(index=False))
    print(
        "POOLED SWITCHES\n", pd.DataFrame(switches).query("fold == 'pooled'").to_string(index=False)
    )
    print(
        "CHILD ARTICLE SLICES\n",
        pd.DataFrame(slices)
        .query("label in ['Boys', 'Girls']")
        .sort_values("n", ascending=False)
        .head(25)
        .to_string(index=False),
    )


def name_signals(frame):
    names = frame.productDisplayName.str.lower().str.replace("’", "'", regex=False)
    boys = names.str.contains(r"\bboy(?:s|'s)?\b", regex=True)
    girls = names.str.contains(r"\bgirl(?:s|'s)?\b", regex=True)
    return pd.Series(
        np.select([boys & ~girls, girls & ~boys], ["Boys", "Girls"], default=""), index=frame.index
    )


def extra(rows, metadata):
    rows["name_child_label"] = name_signals(rows)
    rows["name_label_conflict"] = rows.name_child_label.ne("") & rows.gender.ne(
        rows.name_child_label
    )
    suspect = rows[rows.name_label_conflict].copy()
    suspect.to_csv(HERE / "validation_name_label_conflicts.csv", index=False)
    development = metadata[metadata.partition.eq("development") & metadata.has_gender_label].copy()
    development["name_child_label"] = name_signals(development)
    conflict = development[
        development.name_child_label.ne("") & development.gender.ne(development.name_child_label)
    ]
    conflict.to_csv(HERE / "development_name_label_conflicts.csv", index=False)
    print(
        "DEVELOPMENT NAME/LABEL CONFLICTS",
        len(conflict),
        "families",
        conflict.product_family_group.nunique(),
    )
    print(pd.crosstab(conflict.gender, conflict.name_child_label).to_string())
    print(
        "VALIDATION NAME/LABEL CONFLICTS",
        len(suspect),
        "families",
        suspect.product_family_group.nunique(),
    )
    print(pd.crosstab(suspect.gender, suspect.name_child_label).to_string())
    for model in ["D30", "D45"]:
        print(
            model,
            "title conflicts predicted as name",
            int(suspect[f"{model}_clean"].eq(suspect.name_child_label).sum()),
        )
    selected = []

    def sheet(frame, name, title, gray=False):
        font_path = findfont("DejaVu Sans")
        font = ImageFont.truetype(font_path, 16)
        small = ImageFont.truetype(font_path, 14)
        heading = ImageFont.truetype(font_path, 22)
        width, height = 360, 330 if gray else 320
        canvas = Image.new("RGB", (width * 4, 88 + height * ((len(frame) + 3) // 4)), "#f7f8fa")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), title, fill="#111827", font=heading)
        draw.text(
            (16, 46),
            "Catalog labels. Original 60 x 80 inputs enlarged 2x; no added detail.",
            fill="#374151",
            font=small,
        )
        for i, (_, row) in enumerate(frame.iterrows()):
            x, y = (i % 4) * width, 88 + (i // 4) * height
            draw.rectangle(
                (x + 5, y + 5, x + width - 5, y + height - 5), fill="white", outline="#d1d5db"
            )
            path = resolve_task3_path(row.path, root=ROOT)
            assert compute_sha256(path) == row.sha256
            with Image.open(path) as im:
                original = ImageOps.exif_transpose(im).convert("RGB")
            thumb = original.resize((120, 160), Image.Resampling.NEAREST)
            canvas.paste(thumb, (x + (40 if gray else 120), y + 30))
            if gray:
                thumb_gray = (
                    original.convert("L")
                    .convert("RGB")
                    .resize((120, 160), Image.Resampling.NEAREST)
                )
                canvas.paste(thumb_gray, (x + 200, y + 30))
            draw.text(
                (x + 13, y + 10),
                f"{row.id} | fold {row.cv_fold} | {row.gender}",
                fill="#111827",
                font=font,
            )
            if gray:
                lines = [
                    f"0.30: {row.D30_clean} -> {row.D30_gray}",
                    f"0.45: {row.D45_clean} -> {row.D45_gray}",
                ]
            else:
                lines = [
                    f"0.30: {row.D30_clean} ({row.D30_clean_confidence:.2f})",
                    f"0.45: {row.D45_clean} ({row.D45_clean_confidence:.2f})",
                ]
            lines += textwrap.wrap(str(row.productDisplayName), width=43)[:3]
            draw.multiline_text(
                (x + 13, y + 199), "\n".join(lines), fill="#1f2937", font=small, spacing=3
            )
            selected.append(
                dict(
                    sheet=name,
                    id=row.id,
                    fold=row.cv_fold,
                    label=row.gender,
                    family=row.product_family_group,
                    selection="See README for selection; this is not a random prevalence sample.",
                )
            )
        canvas.save(HERE / f"{name}.png")

    for label in ["Boys", "Girls"]:
        subset = rows[rows.gender.eq(label) & rows.dropout_harm].sort_values(["cv_fold", "id"])
        print(
            label,
            "regressions",
            len(subset),
            "families",
            subset.product_family_group.nunique(),
            "unique images",
            subset.sha256.nunique(),
        )
        # All affected families, one lowest-ID example per family, shown over bounded pages.
        subset = subset.drop_duplicates("product_family_group")
        for page in range(0, len(subset), 16):
            sheet(
                subset.iloc[page : page + 16],
                f"{label.lower()}_regressions_{page // 16 + 1}",
                f"{label}: correct at 0.30, wrong at 0.45 (one example per family)",
            )
    title_errors = suspect[
        (suspect.D45_clean.eq(suspect.name_child_label))
        | (suspect.D30_clean.eq(suspect.name_child_label))
    ]
    title_errors = title_errors.sort_values(["name_child_label", "id"]).drop_duplicates(
        "product_family_group"
    )
    sample = pd.concat([g.head(8) for _, g in title_errors.groupby("name_child_label")])
    sheet(
        sample,
        "name_label_conflicts",
        "Catalog label and child wording disagree; either model follows the name",
    )
    # Stratify by catalog class, then prefer large drops in probability for the true label.
    gray_harm = rows[rows.D30_gray_harm].copy()
    gray_harm["probability_drop"] = gray_harm.D30_clean_p_true - gray_harm.D30_gray_p_true
    sample = pd.concat(
        [
            gray_harm[gray_harm.gender.eq(label)]
            .sort_values(["probability_drop", "id"], ascending=[False, True])
            .drop_duplicates("product_family_group")
            .head(n)
            for label, n in [("Boys", 4), ("Girls", 8), ("Women", 4)]
        ]
    )
    sheet(
        sample,
        "grayscale_failures",
        "Grayscale breaks correct 0.30 predictions (large probability drops)",
        gray=True,
    )
    stable = rows[
        rows.gender.isin(["Boys", "Girls"]) & ~rows.D30_clean_correct & ~rows.D45_clean_correct
    ]
    sample = pd.concat(
        [
            g.sort_values(["D30_clean_confidence", "id"], ascending=[False, True])
            .drop_duplicates("product_family_group")
            .head(8)
            for _, g in stable.groupby("gender")
        ]
    )
    sheet(
        sample,
        "shared_child_errors",
        "Wrong at both dropout rates (confident errors, one example per family)",
    )
    helped = rows[rows.gender.isin(["Boys", "Girls"]) & rows.dropout_help]
    sample = pd.concat(
        [
            g.sort_values("id").drop_duplicates("product_family_group").head(8)
            for _, g in helped.groupby("gender")
        ]
    )
    sheet(sample, "child_recoveries", "Controls: stronger dropout fixes these child-item mistakes")
    pd.DataFrame(selected).to_csv(HERE / "visually_inspected_examples.csv", index=False)


if __name__ == "__main__":
    rows, metadata = load()
    tables(rows, metadata)
    extra(rows, metadata)
