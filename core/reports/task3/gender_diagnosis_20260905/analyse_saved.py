"""Matched saved-OOF diagnosis; clean-training slice inference is a separate step."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.train.task3_g2_audit import inspect_gender_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
registry = pd.read_csv(ROOT / "reports/task3/gender_weight_decay_result_20260905/runs.csv",
                       keep_default_na=False)
splits = load_splits(ROOT / "data/processed/splits.csv")
classes = list(load_label_maps(ROOT / "data/processed/label_maps.json")["gender"]["classes"])
development = get_samples(splits, partition="development", target="gender").copy()
sizes = development.groupby("product_family_group").size()
development["family_size"] = development.product_family_group.map(sizes)
development["family_size_bin"] = pd.cut(
    development.family_size, bins=[0, 1, 3, np.inf], labels=["1", "2–3", "4+"],
).astype(str)
metadata = development[["id", "gender", "articleType", "family_size", "family_size_bin"]]
directories = {
    "G2": ROOT / "reports/task3/g2_gate_audit_20260905/drive",
    "CompactBlurCNN": ROOT / "results/evidence/task3/experiments/t3_gender_e5_compact_blur_cnn/gender",
}
runs, predictions, summary, distribution = {}, [], [], []
for name, directory in directories.items():
    runs[name] = {}
    for path in sorted(directory.glob("t3_*")):
        if not (path / "metrics.json").is_file():
            continue
        m = json.loads((path / "metrics.json").read_text())
        fold = m["validation_fold"]
        if fold not in (0, 4):
            continue
        run = inspect_gender_run(path, registry=registry, splits=splits, classes=classes, root=ROOT)
        assert fold not in runs[name]
        runs[name][fold] = run
        p = run["predictions"].merge(metadata, on="id", validate="one_to_one")
        p["model"] = name
        p["correct"] = p.true_index.eq(p.predicted_index)
        predictions.append(p)
        summary.append(dict(model=name, fold=fold,
                            train_f1=m["final_train_eval_macro_f1"], val_f1=m["macro_f1"],
                            gap=m["final_train_validation_macro_f1_gap"]))
    assert set(runs[name]) == {0, 4}
p = pd.concat(predictions, ignore_index=True)
p.to_csv(OUT / "matched_validation_rows.csv", index=False)
pd.DataFrame(summary).to_csv(OUT / "clean_gap_decomposition.csv", index=False)

slices = []
for model, rows in p.groupby("model"):
    for dimensions in (("gender",), ("articleType",), ("family_size_bin",),
                       ("gender", "articleType"), ("gender", "family_size_bin")):
        for keys, group in rows.groupby(list(dimensions), dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            slices.append(dict(model=model, dimensions=" × ".join(dimensions),
                               slice=" | ".join(map(str, keys)), rows=len(group),
                               families=group.product_family_group.nunique(),
                               errors=int((~group.correct).sum()), accuracy=float(group.correct.mean())))
slices = pd.DataFrame(slices)
slices.to_csv(OUT / "validation_slices.csv", index=False)

# Match each model on identical validation IDs, rather than comparing unrelated subsets.
paired = p[p.model.eq("G2")].merge(
    p[p.model.eq("CompactBlurCNN")][["id", "correct"]], on="id", validate="one_to_one",
    suffixes=("_g2", "_compact"),
)
switches = []
for dimensions in (("gender",), ("gender", "articleType"), ("gender", "family_size_bin")):
    for keys, group in paired.groupby(list(dimensions)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        switches.append(dict(dimensions=" × ".join(dimensions), slice=" | ".join(map(str, keys)),
                             rows=len(group), families=group.product_family_group.nunique(),
                             g2_errors=int((~group.correct_g2).sum()),
                             compact_errors=int((~group.correct_compact).sum()),
                             compact_fixes=int((~group.correct_g2 & group.correct_compact).sum()),
                             compact_new_errors=int((group.correct_g2 & ~group.correct_compact).sum())))
pd.DataFrame(switches).to_csv(OUT / "paired_error_switches.csv", index=False)


def weighted_f1(rows, weights):
    matrix = np.zeros((len(classes), len(classes)))
    np.add.at(matrix, (rows.true_index.to_numpy(), rows.predicted_index.to_numpy()), weights)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.divide(2 * np.diag(matrix), denominator, out=np.zeros(len(classes)),
                           where=denominator > 0).mean())


# Descriptive importance weighting: this reweights validation predictions only.
# It cannot replace missing training predictions or estimate unseen strata.
standardized = []
for fold in (0, 4):
    train, val = get_cv_split(splits, fold)
    train = get_samples(train, target="gender").merge(
        metadata[["id", "family_size_bin"]], on="id", validate="one_to_one",
    )
    val = get_samples(val, target="gender").merge(
        metadata[["id", "family_size_bin"]], on="id", validate="one_to_one",
    )
    for dimensions in (("gender",), ("gender", "articleType"),
                       ("gender", "family_size_bin"), ("gender", "articleType", "family_size_bin")):
        cols = list(dimensions)
        a = train.groupby(cols).size().rename("train_count")
        b = val.groupby(cols).size().rename("val_count")
        counts = pd.concat([a, b], axis=1).fillna(0)
        for key, row in counts.iterrows():
            distribution.append(dict(fold=fold, dimensions=" × ".join(cols), slice=str(key),
                                     train_count=int(row.train_count), val_count=int(row.val_count)))
        train_coverage = float(counts.loc[counts.val_count.gt(0), "train_count"].sum() / len(train))
        val_coverage = float(counts.loc[counts.train_count.gt(0), "val_count"].sum() / len(val))
        counts["weight"] = np.where(counts.val_count.gt(0),
                                    (counts.train_count / len(train)) / (counts.val_count / len(val)), 0)
        for model in directories:
            rows = p[p.model.eq(model) & p.cv_fold.eq(fold)].merge(
                counts[["weight"]].reset_index(), on=cols, validate="many_to_one",
            )
            weights = rows.weight.to_numpy()
            standardized.append(dict(model=model, fold=fold, dimensions=" × ".join(cols),
                                     raw_f1=weighted_f1(rows, np.ones(len(rows))),
                                     standardized_f1=weighted_f1(rows, weights),
                                     train_mass_covered=train_coverage, val_mass_covered=val_coverage,
                                     effective_rows=float(weights.sum() ** 2 / (weights ** 2).sum()),
                                     max_weight=float(weights.max())))
pd.DataFrame(distribution).to_csv(OUT / "train_validation_composition.csv", index=False)
pd.DataFrame(standardized).to_csv(OUT / "composition_standardization.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
for index, dimension in enumerate(("gender", "family_size_bin")):
    frame = slices[slices.dimensions.eq(dimension)].pivot(index="slice", columns="model", values="accuracy")
    frame.plot.bar(ax=axes[index], rot=0)
    axes[index].set(title=f"Matched validation accuracy by {dimension}", ylim=(0, 1), ylabel="Accuracy")
    axes[index].spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "validation_slice_review.png", dpi=150)
report = dict(status="saved_validation_diagnosis_complete_inference_pending", verified_runs=4,
              folds=[0, 4], final_clean_training_aggregate_available=True,
              clean_training_predictions_available=False, batch_order_invariance_tested=False,
              inference_performed=False, holdout_used=False,
              limitations=["No clean training per-image predictions were saved by the trainer.",
                           "Slice-level train/validation gaps and batch/order stability need checkpoint inference.",
                           "Composition standardization is descriptive and does not prove causality."])
(OUT / "diagnostic_status.json").write_text(json.dumps(report, indent=2) + "\n")
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(standardized).to_string(index=False))
