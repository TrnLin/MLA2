"""Verify the completed v3 screen and compare saved predictions; never fit models."""

from fashion.task3_paths import resolve_task3_path

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train import task3_usage_mixup_sam as prior
from fashion.train import task3_usage_replaced_v3 as current
from fashion.train.registry import RunRegistry
from fashion.train.task3_decisions import oof_metrics, paired_family_bootstrap, validate_oof

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/usage_replaced_v3_mixup_sam_20260907"
BASELINE = ROOT / "results/evidence/task3/usage_mixup_sam_20260906"
CLASSES = current.CLASSES
METRICS = ("macro_f1", "accuracy", "nll", "brier", "ece_15")


def check_scores(measured, recorded):
    for key in METRICS:
        assert np.isclose(measured[key], recorded[key], rtol=0, atol=1e-7), key


def check_scopes(measured, recorded):
    assert set(measured) == set(recorded)
    for name in measured:
        assert measured[name]["rows"] == recorded[name]["rows"]
        if measured[name]["rows"]:
            check_scores(measured[name]["metrics"], recorded[name]["metrics"])


def class_table(frame, model, scope):
    rows = []
    for entry in oof_metrics(frame, CLASSES)["per_class"]:
        label = entry["class_name"]
        rows.append(
            {
                "model": model,
                "scope": scope,
                **entry,
                "correct": int(
                    (frame.true_label.eq(label) & frame.predicted_label.eq(label)).sum()
                ),
                "predicted": int(frame.predicted_label.eq(label).sum()),
            }
        )
    return rows


def main():
    splits, contract = current.validate_dataset(check_images=False)
    old_splits, _ = v2.validate_dataset(check_images=False)
    registry_path = current.usage_registry_path(EVIDENCE)
    rows = RunRegistry(registry_path)._read_rows()
    assert len(rows) == 2 and all(row["status"] == "complete" for row in rows)
    assert sorted(int(row["validation_fold"]) for row in rows) == [0, 4]
    references = current.check_reference(
        directory=BASELINE / prior.ARTIFACT_DIRECTORY / "usage",
        registry_path=prior.usage_registry_path(BASELINE),
    )
    results, frames, fold_rows, robust_rows, hashes, diagnostics = [], [], [], [], [], []
    for fold in current.FOLDS:
        run = current.completed_fold(
            fold=fold,
            output_root=EVIDENCE,
            registry_path=registry_path,
            splits=splits,
            spec=current.screen_spec(),
        )
        assert run is not None
        results.append(run)
        for model, result, data in (("v3", run, splits), ("v2", references[fold], old_splits)):
            directory = Path(result["run_dir"])
            train, validation = current.training_scope(data, fold)
            pred = current.source_predictions(
                v2.read_predictions(result["prediction_path"]), validation
            )
            clean = current.source_predictions(
                v2.read_predictions(directory / "training_predictions.csv"), train
            )
            assert clean.run_id.eq(result["run_id"]).all()
            source_file = json.loads((directory / "source_metrics.json").read_text())
            assert source_file == result["metrics"]["source_metrics"]
            check_scopes(current.source_metrics(pred), source_file["validation"])
            check_scopes(current.source_metrics(clean), source_file["clean_training"])
            a = source_file["clean_training"]["teacher"]["metrics"]["macro_f1"]
            b = source_file["validation"]["teacher"]["metrics"]["macro_f1"]
            fold_rows.append(
                dict(
                    model=model,
                    fold=fold,
                    teacher_train_f1=a,
                    teacher_validation_f1=b,
                    teacher_gap=a - b,
                    combined_validation_f1=result["metrics"]["macro_f1"],
                    fit_minutes=result["metrics"]["train_seconds"] / 60,
                )
            )
            robust = pd.read_csv(directory / "robustness.csv")
            assert len(robust) == 5 and robust.run_id.eq(result["run_id"]).all()
            assert np.allclose(
                robust.macro_f1 - result["metrics"]["macro_f1"], robust.macro_f1_change
            )
            robust_rows.append(robust.assign(model=model, fold=fold))
            diagnostics.append(
                dict(
                    model=model,
                    fold=fold,
                    epochs=json.loads((directory / "clean_epoch_diagnostics.json").read_text()),
                )
            )
            if model == "v3":
                frames.append(pred)
        row = next(row for row in rows if int(row["validation_fold"]) == fold)
        for name, digest in {
            **run["metrics"]["expanded_artifact_sha256"],
            "final_epoch.pt": row["checkpoint_sha256"],
            "oof_predictions.csv": row["prediction_sha256"],
            "mixup_training.json": run["metrics"]["mixup_receipt_sha256"],
            "sam_training.json": run["metrics"]["sam_receipt_sha256"],
            "clean_epoch_diagnostics.json": run["metrics"]["clean_epoch_diagnostics_sha256"],
        }.items():
            assert compute_sha256(Path(run["run_dir"]) / name) == digest
            hashes.append(dict(run_id=run["run_id"], file=name, sha256=digest))
        print(
            f"Verified fold {fold}: saved recipe, training receipts, hashes and predictions",
            flush=True,
        )

    comparison = current.build_comparison(
        results, splits=splits, references=references, contract=contract
    )
    aggregate = EVIDENCE / current.ARTIFACT_DIRECTORY / "usage/aggregate_folds_0_4"
    saved = json.loads((aggregate / "teacher_comparison.json").read_text())
    for key in ("sources", "baseline_sources"):
        check_scopes(comparison[key], saved[key])
    for key in ("fold_comparison", "teacher_per_class"):
        pd.testing.assert_frame_equal(
            pd.DataFrame(comparison[key]).sort_index(axis=1),
            pd.DataFrame(saved[key]).sort_index(axis=1),
            atol=1e-7,
        )
    assert comparison["dataset"] == saved["dataset"]
    assert np.isclose(comparison["teacher_macro_f1_change"], saved["teacher_macro_f1_change"])
    expected = pd.concat([current.training_scope(splits, f)[1] for f in current.FOLDS])
    pooled = validate_oof(
        v2.read_predictions(aggregate / "oof_predictions.csv"),
        expected,
        target="usage",
        classes=CLASSES,
        run_ids_by_fold={r["metrics"]["validation_fold"]: r["run_id"] for r in results},
    )
    pd.testing.assert_frame_equal(
        pooled, pd.concat(frames).sort_values("id").reset_index(drop=True), check_dtype=False
    )
    check_scores(oof_metrics(pooled, CLASSES), json.loads((aggregate / "metrics.json").read_text()))
    old = pd.concat([references[f]["predictions"] for f in current.FOLDS], ignore_index=True)
    teacher = pooled.loc[pooled.source_dataset.eq("teacher")]
    old_teacher = old.loc[old.source_dataset.eq("teacher")]
    print("Measuring uncertainty with 10,000 paired product-family resamples", flush=True)
    bootstrap = paired_family_bootstrap(teacher, old_teacher, classes=CLASSES)
    retained = pooled.loc[pooled[current.COHORT_COLUMN].eq("previous_added")]
    matched_old = old.set_index("id").loc[retained.id].reset_index()
    assert retained.true_index.tolist() == matched_old.true_index.tolist()
    source_class, source_rows = [], []
    scopes = (
        ("teacher", teacher, "v3"),
        ("teacher", old_teacher, "v2"),
        ("same_retained_external", retained, "v3"),
        ("same_retained_external", matched_old, "v2"),
        ("new_replacements", pooled.loc[pooled[current.COHORT_COLUMN].eq("new_added")], "v3"),
    )
    for scope, frame, model in scopes:
        source_class.extend(class_table(frame, model, scope))
        source_rows.append(
            dict(
                model=model,
                scope=scope,
                rows=len(frame),
                correct=int(frame.true_index.eq(frame.predicted_index).sum()),
                **{k: oof_metrics(frame, CLASSES)[k] for k in METRICS},
            )
        )
    changes = teacher.merge(
        old_teacher[["id", "predicted_label"]].rename(columns={"predicted_label": "v2_prediction"}),
        on="id",
        validate="one_to_one",
    )
    changes["v2_correct"] = changes.true_label.eq(changes.v2_prediction)
    changes["v3_correct"] = changes.true_label.eq(changes.predicted_label)
    changes = changes.merge(
        splits.loc[splits.partition.eq("development"), ["id", "articleType"]],
        on="id",
        validate="one_to_one",
    )
    changes.loc[
        changes.true_label.isin(["Home", "Party", "Smart Casual", "Travel", "NA"]),
        ["id", "cv_fold", "true_label", "articleType", "v2_prediction", "predicted_label", "path"],
    ].to_csv(REPORT / "rare_teacher_predictions.csv", index=False)
    transition = changes.groupby(["v2_correct", "v3_correct"]).size().to_dict()
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(REPORT / "fold_comparison.csv", index=False)
    pd.DataFrame(source_class).to_csv(REPORT / "source_class_scores.csv", index=False)
    pd.DataFrame(source_rows).to_csv(REPORT / "source_scores.csv", index=False)
    pd.DataFrame(comparison["teacher_per_class"]).to_csv(
        REPORT / "teacher_per_class.csv", index=False
    )
    for model, frame in (("v3", teacher), ("v2", old_teacher)):
        pd.crosstab(frame.true_label, frame.predicted_label).reindex(
            index=CLASSES, columns=CLASSES, fill_value=0
        ).to_csv(REPORT / f"{model}_teacher_confusion.csv")
    robust = pd.concat(robust_rows)
    robust.to_csv(REPORT / "combined_robustness_by_fold.csv", index=False)
    robust.groupby(["model", "corruption"])[["macro_f1", "macro_f1_change"]].mean().to_csv(
        REPORT / "combined_robustness.csv"
    )
    v2.write_json(diagnostics, REPORT / "clean_diagnostics.json")

    notebook_path = Path("notebooks/task3_training/usage_replaced_v3_mixup_sam.ipynb")
    notebook = json.loads((resolve_task3_path(notebook_path, root=ROOT)).read_text())
    committed = json.loads(
        subprocess.check_output(
            ["git", "show", "1e72921:" + str(notebook_path)], cwd=ROOT, text=True
        )
    )
    assert [(c["cell_type"], c["source"]) for c in notebook["cells"]] == [
        (c["cell_type"], c["source"]) for c in committed["cells"]
    ]
    assert not any(
        o["output_type"] == "error" for c in notebook["cells"] for o in c.get("outputs", [])
    )
    setup = json.loads((EVIDENCE / "code_and_data_setup.json").read_text())
    means = fold_table.groupby("model")[
        ["teacher_train_f1", "teacher_validation_f1", "teacher_gap"]
    ].mean()
    summary = dict(
        decision="Keep v2 over v3 for this comparison; do not extend v3 to remaining folds",
        folds=[0, 4],
        teacher_rows=len(teacher),
        candidate_teacher_f1=comparison["sources"]["teacher"]["metrics"]["macro_f1"],
        baseline_teacher_f1=comparison["baseline_sources"]["teacher"]["metrics"]["macro_f1"],
        teacher_f1_change=comparison["teacher_macro_f1_change"],
        teacher_paired_family_bootstrap=bootstrap,
        mean_teacher_fit=means.to_dict(orient="index"),
        candidate_fit_minutes=sum(r["metrics"]["train_seconds"] for r in results) / 60,
        candidate_peak_memory_mb=max(r["metrics"]["peak_memory_bytes"] for r in results) / 1024**2,
        verified_candidate_hashes=len(hashes),
        source_scores=source_rows,
        teacher_prediction_changes={f"v2_{a}_v3_{b}": int(n) for (a, b), n in transition.items()},
        comparison=comparison,
    )
    v2.write_json(summary, REPORT / "verified_summary.json")
    v2.write_json(
        dict(
            candidate_artifacts=hashes,
            reference_folds_verified=2,
            notebook_code_matches_pushed_version=True,
            notebook_has_no_errors=True,
            registry_sha256=compute_sha256(registry_path),
            data_setup=setup,
            no_training_or_new_holdout_test_evaluation=True,
        ),
        REPORT / "checks.json",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = {"v2": "#39759a", "v3": "#c16641"}
    for model in ("v2", "v3"):
        f = fold_table.loc[fold_table.model.eq(model)]
        x = np.arange(2) + (-0.18 if model == "v2" else 0.18)
        bars = axes[0].bar(x, f.teacher_validation_f1, 0.32, color=colors[model], label=model)
        axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Fold 0", "Fold 4"],
        ylim=(0, 0.46),
        ylabel="Teacher macro-F1",
        title="Both teacher folds decline",
    )
    axes[0].legend(frameon=False)
    for model in ("v2", "v3"):
        axes[1].plot(
            [0, 1],
            [means.loc[model, "teacher_train_f1"], means.loc[model, "teacher_validation_f1"]],
            "o-",
            color=colors[model],
            label=model,
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Clean training", "Validation"],
        ylim=(0, 0.75),
        ylabel="Mean fold teacher macro-F1",
        title="The fitting gap is almost unchanged",
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("130 photo replacements did not improve the teacher task", fontsize=15)
    dest = ROOT / "results/figures/task3/usage_replaced_v3_mixup_sam_review_20260907.png"
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "comparison"}, indent=2))


if __name__ == "__main__":
    main()
