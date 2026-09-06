"""Rebuild the class-gap, coverage and history tables without fitting anything."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import load_gender_name_truth_variant

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
HERE = Path(__file__).resolve().parent
source_gaps = ROOT / "reports/task3/gender_mixup_result_20260906/verified_class_gaps.csv"
frame = pd.read_csv(source_gaps)
old = frame.loc[frame.model.eq("MixUp") & frame.basis.eq("name_truth")]
old = old.groupby("class_name")[["train_f1", "validation_f1", "gap"]].mean()
old["gap_share"] = old.gap / old.gap.sum()
old.to_csv(HERE / "mixup20_class_gaps.csv")
rows = []
for fold in [0, 4]:
    metrics = json.loads((HERE / f"mixup40_fold{fold}_metrics.json").read_text())
    training = {c["class_name"]: c for c in metrics["final_train_eval_metrics"]["per_class"]}
    for c in metrics["per_class"]:
        name = c["class_name"]
        rows.append(
            {
                "fold": fold,
                "class_name": name,
                "train_f1": training[name]["f1"],
                "validation_f1": c["f1"],
                "gap": training[name]["f1"] - c["f1"],
            }
        )
    assert np.isclose(
        np.mean([r["gap"] for r in rows if r["fold"] == fold]),
        metrics["final_train_validation_macro_f1_gap"],
    )
new = pd.DataFrame(rows)
new.to_csv(HERE / "mixup40_class_gap_folds.csv", index=False)
new = new.groupby("class_name")[["train_f1", "validation_f1", "gap"]].mean()
new["gap_share"] = new.gap / new.gap.sum()
new.to_csv(HERE / "mixup40_class_gaps.csv")
splits = load_gender_name_truth_variant(ROOT)
rows = []
for fold in [0, 4]:
    training, validation = [get_samples(t, target="gender") for t in get_cv_split(splits, fold)]
    assert not set(training.product_family_group) & set(validation.product_family_group)
    for name, group in training.groupby("gender"):
        rows.append(
            {
                "fold": fold,
                "class_name": name,
                "training_rows": len(group),
                "training_families": group.product_family_group.nunique(),
                "validation_rows": int(validation.gender.eq(name).sum()),
                "training_fraction": len(group) / len(training),
                "expected_random_partner_adult_fraction": training.gender.isin(
                    ["Men", "Women"]
                ).mean(),
            }
        )
pd.DataFrame(rows).to_csv(HERE / "training_coverage.csv", index=False)
pattern = re.compile(
    r"fold=(\d) epoch=(\d+)/30 train_loss=([\d.]+) "
    r"train_macro_f1=n/a \(mixed inputs\) validation_loss=([\d.]+) "
    r"validation_macro_f1=([\d.]+)"
)
rows, notebook_hashes = [], {}
for alpha, filename in [
    (0.2, "task3_training/gender_mixup_screen.ipynb"),
    (0.4, "task3_training/gender_stronger_mixup_screen.ipynb"),
]:
    path = ROOT / "notebooks" / filename
    notebook_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    notebook = json.loads(path.read_text())
    text = "\n".join(
        "".join(o.get("text", [])) for c in notebook["cells"] for o in c.get("outputs", [])
    )
    for fold, epoch, train_loss, val_loss, val_f1 in pattern.findall(text):
        rows.append(
            {
                "alpha": alpha,
                "fold": int(fold),
                "epoch": int(epoch),
                "train_mixed_loss": float(train_loss),
                "validation_loss": float(val_loss),
                "validation_macro_f1": float(val_f1),
            }
        )
history = pd.DataFrame(rows).drop_duplicates(["alpha", "fold", "epoch"], keep="last")
assert len(history) == 120
history.to_csv(HERE / "notebook_epoch_history.csv", index=False)
bn = pd.read_csv(HERE / "bn_probe_scores.csv")
means = bn.groupby(["model", "scope"]).macro_f1.mean().unstack()
summary = {
    "notebook_sha256": notebook_hashes,
    "mixup20_class_gap_source_sha256": hashlib.sha256(source_gaps.read_bytes()).hexdigest(),
    "minority_gap_share": {
        "mixup20": float(old.loc[["Boys", "Girls", "Unisex"], "gap_share"].sum()),
        "mixup40": float(new.loc[["Boys", "Girls", "Unisex"], "gap_share"].sum()),
    },
    "probability_lambda_between_025_and_075": {
        str(a): float(beta.cdf(0.75, a, a) - beta.cdf(0.25, a, a)) for a in [0.2, 0.4]
    },
    "normalization_probe_mean_clean_gap": (means.train - means.validation).to_dict(),
    "epoch20_training_ceiling_for_two_point_gap_reduction": float(
        history.loc[history.alpha.eq(0.2) & history.epoch.eq(20), "validation_macro_f1"].mean()
        + old.gap.mean()
        - 0.02
    ),
}
(HERE / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
