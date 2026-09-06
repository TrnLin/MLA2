"""Verify downloaded inference results against canonical rows and source evidence."""

from fashion.task3_paths import resolve_task3_path
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.metrics import classification_metrics

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
status = json.loads((OUT / "diagnostic_status.json").read_text())
assert status["split_sha256"] == compute_sha256(ROOT / "data/processed/splits.csv")
assert status["diagnostic_source_sha256"] == compute_sha256(ROOT / "src/fashion/train/task3_gender_diagnostic.py")
assert not status["training_performed"]
assert {(r["model"], r["fold"]) for r in status["runs"]} == {
    (name, fold) for name in ("G2", "CompactBlurCNN") for fold in (0, 4)
}
splits = load_splits(ROOT / "data/processed/splits.csv")
development = get_samples(splits, partition="development", target="gender")
assert status["verified_input_images"] == len(development)
classes = list(load_label_maps(ROOT / "data/processed/label_maps.json")["gender"]["classes"])
cols = [f"probability_{i}_{c}" for i, c in enumerate(classes)]
class_scores = pd.read_csv(OUT / "clean_class_scores.csv")
saved_slices = pd.read_csv(OUT / "clean_slices.csv")
stability = pd.read_csv(OUT / "batch_order_stability.csv")
summary = []
for run in status["runs"]:
    source = resolve_task3_path("reports/task3/g2_gate_audit_20260905/drive" if run["model"] == "G2"
                     else "reports/task3/gender_diagnostic_result_20260905/source_compact", root=ROOT) / run["run_id"]
    for filename, digest in run["source_sha256"].items():
        assert compute_sha256(source / filename) == digest
    frames = dict(zip(("train", "validation"), get_cv_split(splits, run["fold"])))
    scores = {}
    for partition, frame in frames.items():
        expected = get_samples(frame, target="gender").set_index("id").sort_index()
        rows = pd.read_csv(OUT / run["run_id"] / f"clean_{partition}_predictions.csv").set_index("id").sort_index()
        assert rows.index.is_unique and rows.index.equals(expected.index)
        for column in ("product_family_group", "gender", "articleType", "path", "cv_fold"):
            assert rows[column].astype(str).equals(expected[column].astype(str)), column
        assert np.array_equal(rows.true_index, expected.gender.map({c:i for i,c in enumerate(classes)}))
        probability = rows[cols].to_numpy()
        assert np.isfinite(probability).all() and (probability >= 0).all()
        assert np.allclose(probability.sum(axis=1), 1, atol=1e-6)
        assert np.array_equal(rows.predicted_index, probability.argmax(axis=1))
        sizes = development.groupby("product_family_group").size()
        assert np.array_equal(rows.family_size, rows.product_family_group.map(sizes))
        bins = pd.cut(rows.family_size, [0,1,3,np.inf], labels=["1","2-3","4+"]).astype(str)
        assert np.array_equal(rows.family_size_bin.astype(str), bins)
        rows["correct"] = rows.true_index.eq(rows.predicted_index)
        subset = saved_slices[(saved_slices.model == run["model"]) &
                              (saved_slices.fold == run["fold"]) &
                              (saved_slices.partition == partition)]
        for dims in (("gender",), ("articleType",), ("family_size_bin",),
                     ("gender","articleType"), ("gender","family_size_bin")):
            groups = subset[subset.dimensions.eq(" x ".join(dims))].set_index("slice")
            for keys, group in rows.groupby(list(dims), dropna=False):
                keys = keys if isinstance(keys,tuple) else (keys,)
                saved = groups.loc[" | ".join(map(str,keys))]
                assert len(group) == saved["rows"]
                assert group.product_family_group.nunique() == saved.families
                assert abs(group.correct.mean() - saved.accuracy) < 1e-12
        calculated = classification_metrics(rows.true_index.to_numpy(), probability, classes)
        scores[partition] = calculated["macro_f1"]
        check = next(c for c in run["checks"] if c["check"] == f"{partition}_macro_f1")
        assert abs(calculated["macro_f1"] - check["actual"]) < 1e-12
        saved_classes = class_scores[(class_scores.model == run["model"]) &
                                    (class_scores.fold == run["fold"]) &
                                    (class_scores.partition == partition)].sort_values("class_index")
        assert np.allclose(saved_classes.f1, [c["f1"] for c in calculated["per_class"]], atol=1e-12)
        sample = pd.read_csv(OUT / run["run_id"] / f"stability_{partition}_ids.csv").id
        assert sample.is_unique and sample.isin(rows.index).all() and len(sample) == 160
        stable = stability[(stability.run_id == run["run_id"]) & (stability.partition == partition)]
        assert len(stable) == 9
        assert set(zip(stable.batch_size, stable.order)) == {(b,o) for b in (1,32,128) for o in ("forward","reverse","shuffle")}
        assert np.array_equal(stable["pass"], (stable.max_abs_difference <= 1e-5) & stable.prediction_flips.eq(0))
        if partition == "validation":
            old = pd.read_csv(source / "oof_predictions.csv").set_index("id").loc[rows.index]
            assert np.max(np.abs(old[cols].to_numpy() - probability)) <= 1e-5
    summary.append({"model":run["model"], "fold":run["fold"], **scores,
                    "gap":scores["train"]-scores["validation"]})
pd.DataFrame(summary).to_csv(OUT / "verified_summary.csv", index=False)
verification = {"verified":True, "runs":4, "predictions_checked":4*len(development),
                "stability_rows":len(stability), "failed_stability_checks":int((~stability["pass"]).sum()),
                "sampled_prediction_flips":int(stability.prediction_flips.sum()),
                "maximum_probability_difference":float(stability.max_abs_difference.max()),
                "limits":"Stability probabilities and frozen weights are attested by the saved GPU run; no local GPU rerun."}
(OUT / "verified_review.json").write_text(json.dumps(verification, indent=2))
print(json.dumps(verification, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, name in zip(axes, ("G2", "CompactBlurCNN")):
    data = class_scores[class_scores.model.eq(name)].groupby(["class_name", "partition"]).f1.mean().unstack().reindex(classes)
    data[["train", "validation"]].plot.bar(ax=ax, color=["#2070a0", "#e89635"], rot=0)
    ax.set(title=name, ylabel="Mean per-class F1 (folds 0 and 4)", xlabel="", ylim=(0, 1.15))
    ax.legend(loc="upper right", ncol=2)
fig.suptitle("Clean training fit stays much stronger than validation")
fig.tight_layout()
fig.savefig(OUT / "class_gap_review.png", dpi=150)
