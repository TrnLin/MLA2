"""Recompute usage evidence without training, tuning, or touching protected targets.

Run from the repository root with ./.venv/bin/python. Outputs stay beside this
script, except report figures saved in results/figures/task3/usage_investigation.
Fixed blends and oracle analyses are post-hoc diagnostics, not admission results.
"""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from fashion.train.metrics import classification_metrics
from fashion.train.task3_decisions import probability_columns, validate_oof

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
CLASSES = ["Casual", "Ethnic", "Formal", "Home", "NA", "Party", "Smart Casual", "Sports", "Travel"]
PROBS = probability_columns(CLASSES)
EXPERIMENTS = {
    "E2": "results/evidence/task3/experiments/t3_usage_e2_class_balanced_ce/usage/aggregate",
    "E3": "results/evidence/task3/experiments/t3_usage_e3_classifier_dropout/usage/aggregate",
    "E4": "results/evidence/task3/experiments/t3_usage_e4_tinyresnet18_pm/usage/aggregate",
    "E5": "results/evidence/task3/experiments/t3_usage_e5_label_smoothing/usage/aggregate",
    "E6": "results/evidence/task3/experiments/t3_usage_e6_focal_gamma1/usage/aggregate",
    "E7": "results/evidence/task3/experiments/t3_usage_e7_tinyconvnext18/usage/aggregate",
    "E8": "results/evidence/task3/experiments/t3_usage_e8_translation/usage/aggregate",
    "E9": "results/evidence/task3/experiments/t3_usage_e9_exception_balance/usage/aggregate",
    "U1": "results/evidence/task3/experiments/t3_usage_v2_u1_component_weight/usage/aggregate",
    "U2": "results/task3/experiments/t3_usage_v2_u2_full_rgb_hog_svm/usage/aggregate_folds_0_4",
    "S1": "results/task3/experiments/task3_clean_slate_screen_1/usage/aggregate_folds_0_4",
    "S2": "reports/task3/failure_review/evidence/gpu_screen_2/usage/aggregate_folds_0_4",
}
READS: dict[str, str] = {}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> None:
    READS[str(path.relative_to(ROOT))] = sha(path)


def csv(path: str | Path) -> pd.DataFrame:
    path = resolve_task3_path(path, root=ROOT)
    record(path)
    return pd.read_csv(path, keep_default_na=False)


def read_json(path: str | Path):
    path = resolve_task3_path(path, root=ROOT)
    record(path)
    return json.loads(path.read_text())


def save_json(name: str, data) -> None:
    (OUT / name).write_text(
        json.dumps(
            data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)
        )
        + "\n"
    )


def score(probabilities, y):
    return classification_metrics(
        np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float), CLASSES
    )


def flatten(label, scope, metrics):
    return {
        "model": label,
        "scope": scope,
        **{k: metrics[k] for k in ["support", "macro_f1", "accuracy", "nll", "brier", "ece_15"]},
        "common_four_f1": np.mean(
            [
                x["f1"]
                for x in metrics["per_class"]
                if x["class_name"] in {"Casual", "Ethnic", "Formal", "Sports"}
            ]
        ),
        "rare_four_f1": np.mean(
            [
                x["f1"]
                for x in metrics["per_class"]
                if x["class_name"] in {"NA", "Party", "Smart Casual", "Travel"}
            ]
        ),
    }


def main():
    OUT.mkdir(exist_ok=True)
    split = csv("data/processed/splits.csv")
    # No protected target is ever used below. Do not export other partitions.
    dev = split.loc[split.partition.eq("development")].copy()
    del split
    dev["cv_fold"] = dev.cv_fold.astype(int)
    valid = dev.loc[dev.has_usage_label.astype(str).str.lower().eq("true")].copy()
    valid = valid.sort_values("id").reset_index(drop=True)
    assert set(valid.usage) == set(CLASSES)
    families = valid.groupby("product_family_group").cv_fold.nunique()
    assert families.max() == 1
    lookup = dict(zip(CLASSES, range(9)))
    valid["true_index"] = valid.usage.map(lookup)

    # Preserve a full filename inventory, including ignored evidence, but do not
    # open protected test evidence or unrelated image trees.
    inventory = []
    for base in ["results/task3", "results/evidence/task3", "reports", "notebooks"]:
        for path in sorted((resolve_task3_path(base, root=ROOT)).rglob("*")):
            if path.is_file() and OUT not in path.parents:
                inventory.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "protected_content_not_read": "/test/" in str(path),
                    }
                )
    pd.DataFrame(inventory).to_csv(OUT / "file_inventory.csv", index=False)
    notebook_records, notebook_text = [], []
    for path in sorted((ROOT / "notebooks").glob("04*.ipynb")):
        notebook = read_json(path)
        notebook_records.append(
            {
                "notebook": path.name,
                "cells": len(notebook["cells"]),
                "executed_cells": sum(
                    c.get("execution_count") is not None for c in notebook["cells"]
                ),
                "output_count": sum(len(c.get("outputs", [])) for c in notebook["cells"]),
                "sha256": READS[str(path.relative_to(ROOT))],
            }
        )
        notebook_text.append(f"\n=== {path.name} ===\n")
        for i, cell in enumerate(notebook["cells"]):
            notebook_text.append(
                f"\nCELL {i} {cell['cell_type']}\n" + "".join(cell.get("source", []))
            )
            for output in cell.get("outputs", []):
                for value in [
                    output.get("text"),
                    output.get("data", {}).get("text/plain"),
                    output.get("traceback"),
                ]:
                    if value:
                        notebook_text.append("".join(value) if isinstance(value, list) else value)
    (OUT / "notebook_readback.txt").write_text("\n".join(notebook_text))
    pd.DataFrame(notebook_records).to_csv(OUT / "notebook_inventory.csv", index=False)

    registries = {}
    for path in ["results/runs.csv", "results/evidence/task3/results/runs.csv"]:
        r = csv(path)
        registries[path] = r.loc[r.target.eq("usage")].copy()
    registry = pd.concat([r.assign(registry_source=name) for name, r in registries.items()])
    registry.to_csv(OUT / "usage_registry_inventory.csv", index=False)
    pd.DataFrame(
        [
            {
                "registry": name,
                "experiment_id": e,
                "status": s,
                "rows": len(g),
                "folds": ",".join(map(str, sorted(g.validation_fold.astype(int).unique()))),
            }
            for name, r in registries.items()
            for (e, s), g in r.groupby(["experiment_id", "status"])
        ]
    ).to_csv(OUT / "registry_coverage.csv", index=False)

    frames, all_summary, class_rows, fold_rows, checks, configs, robust = {}, [], [], [], [], [], []
    for label, relative in EXPERIMENTS.items():
        path = resolve_task3_path(relative, root=ROOT)
        m = read_json(path / "metrics.json")
        run_ids = m["fold_run_ids"]
        fold_saved = []
        for run_id in run_ids:
            run_path = path.parent / run_id
            fm = read_json(run_path / "metrics.json")
            cfg = read_json(run_path / "config.json")
            fold = int(
                fm.get(
                    "validation_fold",
                    registry.loc[registry.run_id.eq(run_id), "validation_fold"].iloc[0]
                    if (registry.run_id == run_id).any()
                    else 0,
                )
            )
            fold_saved.append(fm)
            configs.append(
                {
                    "model": label,
                    "run_id": run_id,
                    "path": str(run_path.relative_to(ROOT)),
                    "config": cfg,
                }
            )
            matched_registry = registry.loc[registry.run_id.eq(run_id)]
            train_score = fm.get("final_train_eval_macro_f1", fm.get("final_train_macro_f1"))
            train_source = "clean final checkpoint" if train_score is not None else "unavailable"
            hp = run_path / "history.csv"
            if hp.exists():
                hist = csv(hp)
                if train_score is None and "train_macro_f1" in hist:
                    train_score, train_source = (
                        float(hist.train_macro_f1.iloc[-1]),
                        "last online epoch",
                    )
            if label == "U2":
                train_source = "ensemble on mixed base-fit/calibration exposures"
            fold_rows.append(
                {
                    "model": label,
                    "fold": fold,
                    "run_id": run_id,
                    "macro_f1": fm["macro_f1"],
                    "train_macro_f1": train_score,
                    "train_source": train_source,
                    "gap": None if train_score is None else train_score - fm["macro_f1"],
                    "nll": fm["nll"],
                    "brier": fm["brier"],
                    "ece_15": fm["ece_15"],
                    "train_seconds": fm.get("train_seconds"),
                    "fold_wall_seconds": fm.get("fold_wall_seconds"),
                    "peak_memory_bytes": fm.get("peak_memory_bytes"),
                    "parameter_count": fm.get("parameter_count", cfg.get("parameter_count")),
                    "registry_sources": ",".join(matched_registry.registry_source.unique()),
                    "metrics_path": str((run_path / "metrics.json").relative_to(ROOT)),
                }
            )
            if (run_path / "robustness.csv").exists():
                rs = csv(run_path / "robustness.csv")
                for row in rs.to_dict("records"):
                    robust.append({"model": label, **row})
        f = path / "oof_predictions.csv"
        if f.exists():
            raw = csv(f)
            folds = sorted(raw.cv_fold.unique())
            expected = valid.loc[valid.cv_fold.isin(folds)]
            pred = validate_oof(
                raw, expected, target="usage", classes=CLASSES, allow_legacy_na=True
            )
            frames[label] = pred.set_index("id").sort_index()
            measured = score(pred[PROBS], pred.true_index)
            checks.append(
                {
                    "model": label,
                    "rows": len(pred),
                    "folds": str(folds),
                    "legacy_na_repairs": pred.attrs.get("legacy_na_label_repairs", 0),
                    "score_abs_error": abs(measured["macro_f1"] - m["macro_f1"]),
                    "max_probability_metric_error": max(
                        abs(measured[k] - m[k]) for k in ["nll", "brier", "ece_15"]
                    ),
                    "canonical_oof": True,
                }
            )
            assert abs(measured["macro_f1"] - m["macro_f1"]) < 1e-10
        else:
            folds = [int(x["validation_fold"]) for x in fold_saved]
            checks.append(
                {
                    "model": label,
                    "rows": m["support"],
                    "folds": str(folds),
                    "canonical_oof": "unavailable",
                }
            )
        all_summary.append(
            flatten(label, "all_available_folds", m)
            | {"folds": ",".join(map(str, folds)), "metrics_path": relative + "/metrics.json"}
        )
        for row in m["per_class"]:
            class_rows.append({"model": label, "scope": "all_available_folds", **row})
        if label in frames:
            pred = frames[label].loc[frames[label].cv_fold.isin([0, 4])]
            sm = score(pred[PROBS], pred.true_index)
            all_summary.append(flatten(label, "folds_0_4", sm) | {"folds": "0,4"})
            class_rows.extend(
                {"model": label, "scope": "folds_0_4", **row} for row in sm["per_class"]
            )

    # E1 has registered exact fold confusion matrices but no local row-level OOF.
    e1 = registries["results/runs.csv"].loc[
        lambda d: d.experiment_id.eq("t3_primary_baseline_smallcnn") & d.status.eq("complete")
    ]
    assert len(e1) == 5
    e1_metrics = [json.loads(s) for s in e1.metrics_json]
    cm = np.sum([np.asarray(m["confusion_matrix"]) for m in e1_metrics], axis=0)
    support, predicted, true_positive = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
    f1s = np.divide(
        2 * true_positive, support + predicted, out=np.zeros(9), where=(support + predicted) != 0
    )
    em = {
        "support": int(cm.sum()),
        "macro_f1": float(f1s.mean()),
        "accuracy": float(true_positive.sum() / cm.sum()),
        "nll": sum(m["nll"] * m["support"] for m in e1_metrics) / cm.sum(),
        "brier": sum(m["brier"] * m["support"] for m in e1_metrics) / cm.sum(),
        "ece_15": None,
        "per_class": [
            {
                "class_name": c,
                "f1": float(f1s[i]),
                "support": int(support[i]),
                "predicted_count": int(predicted[i]),
            }
            for i, c in enumerate(CLASSES)
        ],
    }
    all_summary.append(
        flatten("E1", "all_available_folds", em)
        | {
            "folds": "0,1,2,3,4",
            "metrics_path": (
                "results/runs.csv (summed fold confusion matrices; pooled ECE unavailable)"
            ),
        }
    )
    class_rows.extend({"model": "E1", "scope": "all_available_folds", **r} for r in em["per_class"])
    for (_, row), m in zip(e1.iterrows(), e1_metrics):
        fold = int(row.validation_fold)
        hist = csv(f"reports/task3/failure_review/evidence/usage_e1_drive/fold_{fold}_history.csv")
        train = float(hist.train_macro_f1.iloc[-1])
        fold_rows.append(
            {
                "model": "E1",
                "fold": fold,
                "run_id": row.run_id,
                "macro_f1": m["macro_f1"],
                "train_macro_f1": train,
                "train_source": "last online epoch",
                "gap": train - m["macro_f1"],
                "nll": m["nll"],
                "brier": m["brier"],
                "ece_15": m["ece_15"],
                "train_seconds": row.train_seconds,
                "peak_memory_bytes": row.peak_memory_bytes,
                "parameter_count": row.parameter_count,
                "registry_sources": "results/runs.csv",
            }
        )
    pd.DataFrame(all_summary).to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "class_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "run_ledger.csv", index=False)
    pd.DataFrame(robust).to_csv(OUT / "robustness_ledger.csv", index=False)
    save_json("run_configurations.json", configs)
    save_json("oof_verification.json", checks)

    # Family counts, fold support, label conflicts, and train-local type maps.
    count_rows, type_rows = [], []
    for label, rows in valid.groupby("usage"):
        counts = rows.groupby("product_family_group").size()
        count_rows.append(
            {
                "class_name": label,
                "images": len(rows),
                "families": len(counts),
                "article_types": rows.articleType.nunique(),
                "family_size_concentration": len(rows) ** 2 / (counts**2).sum(),
                "max_family": int(counts.max()),
            }
        )
    pd.DataFrame(count_rows).to_csv(OUT / "class_support.csv", index=False)
    valid.groupby(["cv_fold", "usage"]).agg(
        images=("id", "size"), families=("product_family_group", "nunique")
    ).to_csv(OUT / "fold_support.csv")
    for fold in range(5):
        train, validation = valid[valid.cv_fold.ne(fold)], valid[valid.cv_fold.eq(fold)]
        crosstab = pd.crosstab(train.articleType, train.usage).reindex(
            columns=CLASSES, fill_value=0
        )
        mode = crosstab.idxmax(axis=1)
        for row in validation.itertuples():
            usual = mode.get(row.articleType, "<no_training_type>")
            type_rows.append(
                {
                    "id": row.id,
                    "usual_usage": usual,
                    "type_status": "unseen"
                    if usual == "<no_training_type>"
                    else "usual"
                    if usual == row.usage
                    else "exception",
                }
            )
    types = pd.DataFrame(type_rows).set_index("id")
    valid = valid.set_index("id").join(types)
    mixed = valid.groupby("product_family_group").usage.nunique().gt(1)
    save_json(
        "family_audit.json",
        {
            "development_rows": len(dev),
            "usage_rows": len(valid),
            "mixed_usage_families": int(mixed.sum()),
            "mixed_usage_rows": int(valid.product_family_group.isin(mixed[mixed].index).sum()),
            "family_fold_crossings": int(families.gt(1).sum()),
            "no_training_type_rows": int(valid.type_status.eq("unseen").sum()),
        },
    )

    ranking, errors, overlaps, oracle_rows = [], [], [], []
    for label, frame in frames.items():
        for scope, current in [
            ("all_available_folds", frame),
            ("folds_0_4", frame[frame.cv_fold.isin([0, 4])]),
        ]:
            y, probs = current.true_index.to_numpy(), current[PROBS].to_numpy()
            ranks = np.argsort(-probs, axis=1, kind="stable")
            for i, name in enumerate(CLASSES):
                positives = y == i
                if not positives.any():
                    continue
                ranking.append(
                    {
                        "model": label,
                        "scope": scope,
                        "class_name": name,
                        "support": int(positives.sum()),
                        "prevalence": positives.mean(),
                        "average_precision": average_precision_score(positives, probs[:, i]),
                        "roc_auc": roc_auc_score(positives, probs[:, i]),
                        "true_top1": (ranks[positives, :1] == i).any(axis=1).mean(),
                        "true_top2": (ranks[positives, :2] == i).any(axis=1).mean(),
                        "true_top3": (ranks[positives, :3] == i).any(axis=1).mean(),
                        "true_probability_median": float(np.median(probs[positives, i])),
                        "largest_probability": probs[:, i].max(),
                    }
                )
        joined = frame.join(valid[["articleType", "type_status", "usual_usage"]])
        for category, group in joined.groupby("type_status"):
            errors.append(
                {
                    "model": label,
                    "scope": "all_available_folds",
                    "slice": category,
                    "rows": len(group),
                    "errors": int(group.true_index.ne(group.predicted_index).sum()),
                    "error_rate": group.true_index.ne(group.predicted_index).mean(),
                }
            )
        if label == "E2":
            continue
        ids = frame.index.intersection(frames["E2"].index)
        a, b = frames["E2"].loc[ids], frame.loc[ids]
        for category in ["all", "usual", "exception", *CLASSES]:
            chosen = (
                np.ones(len(ids), dtype=bool)
                if category == "all"
                else (
                    valid.loc[ids, "type_status"].eq(category)
                    if category in ["usual", "exception"]
                    else a.true_label.eq(category)
                ).to_numpy()
            )
            aa, bb = a.loc[chosen], b.loc[chosen]
            ea, eb = aa.true_index.ne(aa.predicted_index), bb.true_index.ne(bb.predicted_index)
            overlaps.append(
                {
                    "child": label,
                    "slice": category,
                    "rows": len(aa),
                    "both_wrong": int((ea & eb).sum()),
                    "fixed_e2_errors": int((ea & ~eb).sum()),
                    "new_errors": int((~ea & eb).sum()),
                    "both_correct": int((~ea & ~eb).sum()),
                    "prediction_disagreement": int(aa.predicted_index.ne(bb.predicted_index).sum()),
                }
            )
    pd.DataFrame(ranking).to_csv(OUT / "class_ranking_diagnostics.csv", index=False)
    pd.DataFrame(errors).to_csv(OUT / "type_error_slices.csv", index=False)
    pd.DataFrame(overlaps).to_csv(OUT / "paired_error_changes.csv", index=False)

    # These equal-weight combinations are fixed diagnostics. There is no search
    # for blend weights or class thresholds and no promotion on these results.
    blends, blend_classes, blend_folds = [], [], []
    for members in [
        ("E2", "E3"),
        ("E2", "E8"),
        ("E3", "E8"),
        ("E2", "E3", "E8"),
        ("E2", "U2"),
        ("E2", "E9"),
    ]:
        ids = frames[members[0]].index
        for member in members[1:]:
            ids = ids.intersection(frames[member].index)
        pred = frames[members[0]].loc[ids]
        probs = np.mean([frames[member].loc[ids, PROBS].to_numpy() for member in members], axis=0)
        name = "+".join(members)
        m = score(probs, pred.true_index)
        blends.append(
            flatten(name, "posthoc_equal_weight_diagnostic", m)
            | {"folds": ",".join(map(str, sorted(pred.cv_fold.unique())))}
        )
        blend_classes.extend({"model": name, **row} for row in m["per_class"])
        for fold in sorted(pred.cv_fold.unique()):
            mask = pred.cv_fold.eq(fold).to_numpy()
            blend_folds.append(
                {
                    "model": name,
                    "fold": fold,
                    **flatten(
                        name, "posthoc", score(probs[mask], pred.true_index.to_numpy()[mask])
                    ),
                }
            )
        correct = np.stack(
            [
                frames[x].loc[ids].predicted_index.to_numpy() == pred.true_index.to_numpy()
                for x in members
            ]
        )
        for i, label in enumerate(CLASSES):
            mask = pred.true_index.to_numpy() == i
            oracle_rows.append(
                {
                    "members": name,
                    "class_name": label,
                    "support": int(mask.sum()),
                    "none_correct": int((~correct.any(axis=0) & mask).sum()),
                    "at_least_one_correct": int((correct.any(axis=0) & mask).sum()),
                }
            )
    pd.DataFrame(blends).to_csv(OUT / "fixed_blend_diagnostics.csv", index=False)
    pd.DataFrame(blend_classes).to_csv(OUT / "fixed_blend_classes.csv", index=False)
    pd.DataFrame(blend_folds).to_csv(OUT / "fixed_blend_folds.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(OUT / "error_union_oracle.csv", index=False)

    # Probe scores live on a radically different class distribution. Re-score
    # current E2/U2 on the exact old probe IDs as a matched diagnostic only.
    pp = csv(
        "results/evidence/task3/clean_slate_eda/view_probes/usage_full_rgb_hog_oof.csv"
    ).set_index("id")
    probe_readback = []
    for label in ["E2", "U2"]:
        frame = frames[label]
        ids = frame.index.intersection(pp.index)
        sub = frame.loc[ids]
        probe_readback.append(
            flatten(label, "old_probe_ids_intersection", score(sub[PROBS], sub.true_index))
            | {"folds": str(sorted(sub.cv_fold.unique()))}
        )
        pr = pp.loc[ids]
        pcols = [c for c in pr if c.startswith("probability_")]
        if len(pcols) == 9:
            probe_readback.append(
                flatten(
                    "old_balanced_probe",
                    "same_ids_as_" + label,
                    score(pr[pcols], valid.loc[ids].true_index),
                )
            )
        else:
            assert (pr.true_label.to_numpy() == valid.loc[ids].usage.to_numpy()).all()
            assert (pr.cv_fold.to_numpy() == valid.loc[ids].cv_fold.to_numpy()).all()
            probe_readback.append(
                {
                    "model": "old_balanced_probe",
                    "scope": "same_ids_as_" + label,
                    "support": len(pr),
                    "macro_f1": f1_score(
                        pr.true_label,
                        pr.predicted_label,
                        labels=CLASSES,
                        average="macro",
                        zero_division=0,
                    ),
                    "accuracy": pr.true_label.eq(pr.predicted_label).mean(),
                    "probability_metrics": (
                        "unavailable; saved probe has no full probability vector"
                    ),
                }
            )
    pd.DataFrame(probe_readback).to_csv(OUT / "probe_scope_comparison.csv", index=False)
    save_json("input_hashes.json", READS)
    print(
        pd.DataFrame(all_summary)[
            ["model", "scope", "support", "macro_f1", "nll", "ece_15"]
        ].to_string(index=False)
    )
    print("\nFixed equal-weight diagnostic blends (not accepted candidates):")
    print(
        pd.DataFrame(blends)[["model", "macro_f1", "nll", "ece_15", "folds"]].to_string(index=False)
    )
    print("\nSaved", len(READS), "source hashes;", len(fold_rows), "fold run rows.")


if __name__ == "__main__":
    main()
