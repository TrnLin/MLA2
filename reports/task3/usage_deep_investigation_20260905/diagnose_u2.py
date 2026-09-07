"""Read trusted saved U2 models; separate base ranking from calibration, with no fit.

Only development features are read. Raw-margin decisions are descriptive
counterfactuals, not calibrated candidates or replacements for the frozen U2.
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from analyse_evidence import CLASSES, OUT, PROBS, ROOT, csv, read_json, save_json, sha
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from fashion.train.task3_clean_slate import _expand_probabilities


def main():
    started = time.monotonic()
    base = ROOT / "results/task3/experiments/t3_usage_v2_u2_full_rgb_hog_svm"
    contract_path = next((base / "feature_cache").glob("*.contract.json"))
    contract = read_json(contract_path)
    hashes = {}
    for name, expected in contract["artifact_sha256"].items():
        path = contract_path.parent / name
        actual = sha(path)
        assert actual == expected, path
        hashes[str(path.relative_to(ROOT))] = actual
    ids = csv(contract_path.parent / "full_rgb_hog_ids.csv")
    dev = csv("data/processed/splits.csv")
    dev = dev.loc[dev.partition.eq("development")].set_index("id").loc[ids.id]
    dev["cv_fold"] = dev.cv_fold.astype(int)
    assert len(dev) == contract["rows"] == 32773
    matrix = np.load(
        next((base / "feature_cache").glob("*.npy")), mmap_mode="r", allow_pickle=False
    )
    assert matrix.shape == (32773, 1944)
    valid = dev.has_usage_label.astype(str).str.lower().eq("true").to_numpy()
    frames, calibrators, scores, classes, history = [], [], [], [], []
    outer_details = []
    with threadpool_limits(limits=2):
        for path in sorted((base / "usage").glob("t3*/model.pkl")):
            metrics = read_json(path.parent / "metrics.json")
            registry = csv("results/runs.csv")
            row = registry.loc[registry.run_id.eq(metrics["run_id"])]
            assert len(row) == 1 and row.status.iloc[0] == "complete"
            actual = sha(path)
            assert actual == metrics["checkpoint_sha256"] == row.checkpoint_sha256.iloc[0]
            hashes[str(path.relative_to(ROOT))] = actual
            # A pickle is executable. This one is the locally produced model,
            # hash-verified against its registered run, not an external download.
            with path.open("rb") as stream:
                bundle = pickle.load(stream)
            model = bundle["model"]
            fold = int(metrics["validation_fold"])
            outer_mask = valid & dev.cv_fold.eq(fold).to_numpy()
            outer = dev.loc[outer_mask]
            p = model.predict_proba(matrix[outer_mask])
            saved = csv(path.parent / "oof_predictions.csv").set_index("id").loc[outer.index]
            expanded = _expand_probabilities(p, model.classes_, CLASSES)
            error = float(np.max(np.abs(expanded - saved[PROBS].to_numpy())))
            assert error < 1e-8
            inner_folds = sorted(set(range(5)) - {fold})
            raw_outer, one_calibrated = [], []
            raw_inner_oof = np.full((len(dev), 8), np.nan)
            solver_history = csv(path.parent / "solver_history.csv")
            for inner_fold, calibrated in zip(
                inner_folds, model.calibrated_classifiers_, strict=True
            ):
                estimator = calibrated.estimator
                assert list(estimator.classes_) == list(model.classes_)
                raw = estimator.decision_function(matrix)
                raw_outer.append(raw[outer_mask])
                base_fit = (
                    valid
                    & ~dev.usage.eq("Home").to_numpy()
                    & ~dev.cv_fold.isin([fold, inner_fold]).to_numpy()
                )
                calibration_mask = (
                    valid & ~dev.usage.eq("Home").to_numpy() & dev.cv_fold.eq(inner_fold).to_numpy()
                )
                raw_inner_oof[calibration_mask] = raw[calibration_mask]
                for scope, mask in [
                    ("inner_base_training", base_fit),
                    ("inner_base_heldout", calibration_mask),
                    ("outer_validation", outer_mask),
                ]:
                    truth = dev.loc[mask, "usage"].to_numpy()
                    predicted = model.classes_[raw[mask].argmax(axis=1)]
                    values = precision_recall_fscore_support(
                        truth, predicted, labels=CLASSES, zero_division=0
                    )
                    scores.append(
                        {
                            "outer_fold": fold,
                            "inner_calibration_fold": inner_fold,
                            "scope": scope,
                            "rows": int(mask.sum()),
                            "raw_macro_f1_nine": f1_score(
                                truth, predicted, labels=CLASSES, average="macro", zero_division=0
                            ),
                        }
                    )
                    for i, label in enumerate(CLASSES):
                        classes.append(
                            {
                                "outer_fold": fold,
                                "inner_calibration_fold": inner_fold,
                                "scope": scope,
                                "class_name": label,
                                "precision": values[0][i],
                                "recall": values[1][i],
                                "f1": values[2][i],
                                "support": int(values[3][i]),
                            }
                        )
                cprob = calibrated.predict_proba(matrix[outer_mask])
                one_calibrated.append(cprob)
                for i, label in enumerate(model.classes_):
                    cal = calibrated.calibrators[i]
                    positive = outer.usage.eq(label).to_numpy()
                    calibration_positive = calibration_mask & dev.usage.eq(label).to_numpy()
                    calibrators.append(
                        {
                            "outer_fold": fold,
                            "inner_calibration_fold": inner_fold,
                            "class_name": label,
                            "calibration_positive_rows": int(calibration_positive.sum()),
                            "calibration_positive_families": int(
                                dev.loc[calibration_positive].product_family_group.nunique()
                            ),
                            "slope_a": float(cal.a_),
                            "intercept_b": float(cal.b_),
                            "reversed_slope": bool(cal.a_ > 0),
                            "outer_margin_ap": average_precision_score(
                                positive, raw[outer_mask, i]
                            ),
                            "outer_calibrated_ap": average_precision_score(positive, cprob[:, i]),
                            "outer_true_margin_median": float(
                                np.median(raw[outer_mask, i][positive])
                            ),
                            "outer_true_probability_median": float(np.median(cprob[positive, i])),
                            "fit_class_weight": estimator.class_weights_[
                                list(estimator.weight_classes_).index(label)
                            ],
                        }
                    )
            mean_margin = np.mean(raw_outer, axis=0)
            mean_probability = np.mean(one_calibrated, axis=0)
            assert np.max(np.abs(p - mean_probability)) < 1e-10
            truth = outer.usage.to_numpy()
            raw_pred = model.classes_[mean_margin.argmax(axis=1)]
            frame = pd.DataFrame(
                {
                    "id": outer.index,
                    "cv_fold": fold,
                    "true_label": truth,
                    "raw_margin_prediction": raw_pred,
                    "calibrated_prediction": saved.predicted_label.to_numpy(),
                }
            )
            for i, name in enumerate(model.classes_):
                frame["mean_margin_" + name] = mean_margin[:, i]
            frames.append(frame)
            outer_details.append(
                {
                    "fold": fold,
                    "run_id": metrics["run_id"],
                    "saved_probability_max_abs_error": error,
                    "raw_mean_margin_f1": f1_score(
                        truth, raw_pred, labels=CLASSES, average="macro", zero_division=0
                    ),
                    "calibrated_f1": metrics["macro_f1"],
                    "all_solvers_converged": bool(solver_history.solver_converged.all()),
                    "solver_iterations_min": int(solver_history.solver_iterations.min()),
                    "solver_iterations_max": int(solver_history.solver_iterations.max()),
                }
            )
            # This is cross-fit only for each raw base estimator. The sigmoid is
            # fitted on these rows, so no calibrated inner score is called OOF.
            eligible = np.isfinite(raw_inner_oof).all(axis=1)
            history.append(
                {
                    "outer_fold": fold,
                    "inner_raw_oof_rows": int(eligible.sum()),
                    "inner_raw_oof_f1": f1_score(
                        dev.loc[eligible].usage,
                        model.classes_[raw_inner_oof[eligible].argmax(axis=1)],
                        labels=CLASSES,
                        average="macro",
                        zero_division=0,
                    ),
                }
            )
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(OUT / "u2_raw_margin_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(calibrators).to_csv(OUT / "u2_calibrator_audit.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "u2_base_fit_scores.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "u2_base_fit_classes.csv", index=False)
    summary = {
        "folds": outer_details,
        "inner_raw_crossfit": history,
        "pooled_raw_mean_margin_f1": f1_score(
            output.true_label,
            output.raw_margin_prediction,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "pooled_calibrated_f1": f1_score(
            output.true_label,
            output.calibrated_prediction,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "input_hashes": hashes,
        "seconds": time.monotonic() - started,
        "new_model_fits": 0,
        "new_optimizer_steps": 0,
        "protected_rows": 0,
        "interpretation": "post-hoc paired calibration/margin diagnostic, not a new admitted model",
    }
    save_json("u2_mechanism_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(
        pd.DataFrame(calibrators)
        .groupby("class_name")[
            [
                "calibration_positive_rows",
                "calibration_positive_families",
                "outer_margin_ap",
                "outer_calibrated_ap",
                "reversed_slope",
            ]
        ]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
