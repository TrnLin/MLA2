"""Fixed prior check and influence diagnostics; no optimization or model fit."""

import numpy as np
import pandas as pd
from analyse_evidence import CLASSES, EXPERIMENTS, OUT, PROBS, csv, flatten, save_json, score


def main():
    dev = csv("data/processed/splits.csv")
    dev = dev.loc[
        dev.partition.eq("development") & dev.has_usage_label.astype(str).str.lower().eq("true")
    ].copy()
    dev["cv_fold"] = dev.cv_fold.astype(int)
    tables, perclass, folds, counts = [], [], [], []
    for label in ["E2", "E3", "E8"]:
        frame = csv(EXPERIMENTS[label] + "/oof_predictions.csv")
        probability = frame[PROBS].to_numpy()
        transformed = probability.copy()
        for fold in range(5):
            support = dev.loc[dev.cv_fold.ne(fold)].usage.value_counts()
            weights = np.array(
                [
                    (support["Casual"] / support[name]) ** 0.25 if name != "Home" else 1.0
                    for name in CLASSES
                ]
            )
            mask = frame.cv_fold.eq(fold).to_numpy()
            transformed[mask] *= weights
            for name, weight in zip(CLASSES, weights, strict=True):
                counts.append(
                    {"model": label, "fold": fold, "class_name": name, "multiplier": weight}
                )
        transformed /= transformed.sum(axis=1, keepdims=True)
        for scope, mask in [
            ("all_five_folds", np.ones(len(frame), bool)),
            ("folds_0_4", frame.cv_fold.isin([0, 4]).to_numpy()),
        ]:
            m = score(transformed[mask], frame.true_index.to_numpy()[mask])
            tables.append(flatten(label + " prior^0.25", scope, m))
            perclass.extend({"model": label, "scope": scope, **row} for row in m["per_class"])
        for fold in range(5):
            mask = frame.cv_fold.eq(fold).to_numpy()
            folds.append(
                {
                    "model": label,
                    "fold": fold,
                    **flatten(
                        label,
                        "fixed_prior_diagnostic",
                        score(transformed[mask], frame.true_index.to_numpy()[mask]),
                    ),
                }
            )
    pd.DataFrame(tables).to_csv(OUT / "fixed_prior_diagnostics.csv", index=False)
    pd.DataFrame(perclass).to_csv(OUT / "fixed_prior_classes.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "fixed_prior_folds.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "fixed_prior_multipliers.csv", index=False)
    classes = pd.read_csv(OUT / "class_comparison.csv", keep_default_na=False)
    selected = classes.loc[classes.scope.eq("all_available_folds")].set_index(
        ["model", "class_name"]
    )
    changes = []
    for child in ["E3", "E8", "E9"]:
        for name in CLASSES:
            baseline, new = selected.loc[("E2", name)], selected.loc[(child, name)]
            changes.append(
                {
                    "model": child,
                    "class_name": name,
                    "support": new.support,
                    "class_f1_delta": new.f1 - baseline.f1,
                    "contribution_to_nine_class_delta": (new.f1 - baseline.f1) / 9,
                    "child_predicted_count": new.predicted_count,
                    "child_true_positive": round(float(new.recall) * int(new.support)),
                    "parent_true_positive": round(float(baseline.recall) * int(baseline.support)),
                }
            )
    pd.DataFrame(changes).to_csv(OUT / "macro_f1_gain_decomposition.csv", index=False)
    save_json(
        "decision_diagnostic_contract.json",
        {
            "exponent": 0.25,
            "sources": ["E2", "E3", "E8"],
            "fits": 0,
            "score_scope": "post-hoc sensitivity only; no acceptance gate is applied or changed",
        },
    )
    print(pd.DataFrame(tables).to_string(index=False))


if __name__ == "__main__":
    main()
