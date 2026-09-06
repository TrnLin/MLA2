"""Review the completed two-fold Usage screen without training or loading model weights."""

from fashion.task3_paths import resolve_task3_path

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train import task3_usage_mixup_sam as screen
from fashion.train.registry import RunRegistry
from fashion.train.task3_decisions import oof_metrics, paired_family_bootstrap, validate_oof

REPORT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/task3/usage_mixup_sam_20260906"
BASELINE = ROOT / "results/evidence/task3/usage_expanded_v2_e8_20260906"
METRICS = ("macro_f1", "accuracy", "nll", "brier", "ece_15")


def check_scores(measured, recorded):
    for key in METRICS:
        assert np.isclose(measured[key], recorded[key], rtol=0, atol=1e-7), key


def check_scopes(measured, recorded):
    assert set(measured) == set(recorded)
    for name, result in measured.items():
        assert result["rows"] == recorded[name]["rows"]
        if result["rows"]:
            check_scores(result["metrics"], recorded[name]["metrics"])


def class_table(predictions, model):
    rows = []
    for source, frame in predictions.groupby(v2.COHORT_COLUMN):
        scores = oof_metrics(frame, v2.CLASSES)
        for item in scores["per_class"]:
            label = item["class_name"]
            rows.append(
                dict(
                    model=model,
                    source=source,
                    class_name=label,
                    support=int(frame.true_label.eq(label).sum()),
                    correct=int(
                        (frame.true_label.eq(label) & frame.predicted_label.eq(label)).sum()
                    ),
                    predicted=int(frame.predicted_label.eq(label).sum()),
                    precision=item["precision"],
                    recall=item["recall"],
                    f1=item["f1"],
                )
            )
    return pd.DataFrame(rows)


def main():
    registry_path = screen.usage_registry_path(EVIDENCE)
    rows = RunRegistry(registry_path)._read_rows()
    assert len(rows) == 2
    assert sorted(int(row["validation_fold"]) for row in rows) == [0, 4]
    assert all(row["status"] == "complete" for row in rows)
    splits, contract = v2.validate_dataset(check_images=False)
    references = screen.check_reference(
        directory=BASELINE, registry_path=BASELINE / "results/runs.csv", splits=splits
    )
    results, prediction_frames, fold_rows, robust_rows, hashes = [], [], [], [], []
    for fold in screen.FOLDS:
        result = screen.completed_fold(
            fold=fold,
            output_root=EVIDENCE,
            registry_path=registry_path,
            splits=splits,
            spec=screen.screen_spec(),
        )
        assert result is not None
        results.append(result)
        directory = Path(result["run_dir"])
        row = next(row for row in rows if int(row["validation_fold"]) == fold)
        training, validation = v2.training_scope(splits, fold)
        prediction = v2.source_predictions(
            v2.read_predictions(result["prediction_path"]), validation
        )
        clean_train = v2.source_predictions(
            v2.read_predictions(directory / "training_predictions.csv"), training
        )
        assert clean_train.run_id.eq(result["run_id"]).all()
        measured = dict(
            validation=v2.source_metrics(prediction), clean_training=v2.source_metrics(clean_train)
        )
        source_file = json.loads((directory / "source_metrics.json").read_text())
        assert source_file == result["metrics"]["source_metrics"]
        for scope in measured:
            check_scopes(measured[scope], source_file[scope])
        prediction_frames.append(prediction)
        for model, run in (("v2_e8", references[fold]), ("mixup_sam", result)):
            scores = run["metrics"]["source_metrics"]
            # Recompute the reference training scores too, on the identical fold-training IDs.
            if model == "v2_e8":
                base_train = v2.source_predictions(
                    v2.read_predictions(Path(run["run_dir"]) / "training_predictions.csv"), training
                )
                check_scopes(v2.source_metrics(base_train), scores["clean_training"])
                check_scopes(v2.source_metrics(run["predictions"]), scores["validation"])
            teacher_train = scores["clean_training"]["teacher"]["metrics"]["macro_f1"]
            teacher_val = scores["validation"]["teacher"]["metrics"]["macro_f1"]
            fold_rows.append(
                dict(
                    model=model,
                    fold=fold,
                    teacher_train_f1=teacher_train,
                    teacher_validation_f1=teacher_val,
                    teacher_gap=teacher_train - teacher_val,
                    combined_validation_f1=run["metrics"]["macro_f1"],
                    fit_minutes=run["metrics"]["train_seconds"] / 60,
                )
            )
            robust = pd.read_csv(Path(run["run_dir"]) / "robustness.csv")
            assert len(robust) == 5 and robust.run_id.eq(run["run_id"]).all()
            assert np.allclose(
                robust.macro_f1 - run["metrics"]["macro_f1"],
                robust.macro_f1_change,
                atol=1e-12,
                rtol=0,
            )
            robust_rows.append(robust.assign(model=model))
        hashes.extend(
            dict(run_id=result["run_id"], file=name, sha256=digest)
            for name, digest in {
                **result["metrics"]["expanded_artifact_sha256"],
                "final_epoch.pt": row["checkpoint_sha256"],
                "oof_predictions.csv": row["prediction_sha256"],
                "mixup_training.json": result["metrics"]["mixup_receipt_sha256"],
                "sam_training.json": result["metrics"]["sam_receipt_sha256"],
                "clean_epoch_diagnostics.json": result["metrics"]["clean_epoch_diagnostics_sha256"],
            }.items()
        )
        print(
            f"Verified fold {fold}: complete recipe, files, two SAM passes and source scores",
            flush=True,
        )

    comparison = screen.build_comparison(
        results, splits=splits, references=references, contract=contract
    )
    aggregate = EVIDENCE / screen.ARTIFACT_DIRECTORY / "usage/aggregate_folds_0_4"
    saved = json.loads((aggregate / "teacher_comparison.json").read_text())
    # NumPy reduction order across runtimes can differ in the final floating-point bit.
    for key in ("sources", "baseline_sources"):
        check_scopes(comparison[key], saved[key])
    assert {k: v for k, v in saved.items() if k not in {"sources", "baseline_sources"}} == {
        k: v for k, v in comparison.items() if k not in {"sources", "baseline_sources"}
    }
    expected = pd.concat([v2.training_scope(splits, fold)[1] for fold in screen.FOLDS])
    pooled = validate_oof(
        v2.read_predictions(aggregate / "oof_predictions.csv"),
        expected,
        target="usage",
        classes=v2.CLASSES,
        run_ids_by_fold={run["metrics"]["validation_fold"]: run["run_id"] for run in results},
    )
    pd.testing.assert_frame_equal(
        pooled,
        pd.concat(prediction_frames).sort_values("id").reset_index(drop=True),
        check_dtype=False,
    )
    check_scores(
        oof_metrics(pooled, v2.CLASSES), json.loads((aggregate / "metrics.json").read_text())
    )
    old = pd.concat([references[fold]["predictions"] for fold in screen.FOLDS], ignore_index=True)
    teacher = pooled.loc[pooled.source_dataset.eq("teacher")]
    old_teacher = old.loc[old.source_dataset.eq("teacher")]
    print("Checking uncertainty with paired product-family resampling", flush=True)
    bootstrap = paired_family_bootstrap(teacher, old_teacher, classes=v2.CLASSES)
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(REPORT / "fold_comparison.csv", index=False)
    classes = pd.concat([class_table(pooled, "mixup_sam"), class_table(old, "v2_e8")])
    classes.to_csv(REPORT / "source_class_scores.csv", index=False)
    pd.DataFrame(comparison["teacher_per_class"]).to_csv(
        REPORT / "teacher_per_class.csv", index=False
    )
    robust = pd.concat(robust_rows)
    robust.to_csv(REPORT / "combined_robustness_by_fold.csv", index=False)
    robust_summary = robust.groupby(["model", "corruption"])[["macro_f1", "macro_f1_change"]].mean()
    robust_summary.to_csv(REPORT / "combined_robustness.csv")
    pd.crosstab(teacher.true_label, teacher.predicted_label).reindex(
        index=v2.CLASSES, columns=v2.CLASSES, fill_value=0
    ).to_csv(REPORT / "teacher_confusion.csv")

    notebook_path = Path("notebooks/task3_training/usage_mixup_sam_screen.ipynb")
    notebook = json.loads((resolve_task3_path(notebook_path, root=ROOT)).read_text())
    committed = json.loads(
        subprocess.check_output(
            ["git", "show", "53cf353:" + str(notebook_path)], cwd=ROOT, text=True
        )
    )
    assert [(c["cell_type"], c["source"]) for c in notebook["cells"]] == [
        (c["cell_type"], c["source"]) for c in committed["cells"]
    ]
    assert not any(
        output["output_type"] == "error"
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
    means = fold_table.groupby("model")[
        ["teacher_train_f1", "teacher_validation_f1", "teacher_gap"]
    ].mean()
    summary = dict(
        decision="Do not expand this recipe to the remaining folds on this evidence",
        folds=[0, 4],
        epochs=30,
        teacher_rows=len(teacher),
        combined_rows=len(pooled),
        candidate_teacher_f1=comparison["sources"]["teacher"]["metrics"]["macro_f1"],
        baseline_teacher_f1=comparison["baseline_sources"]["teacher"]["metrics"]["macro_f1"],
        teacher_f1_change=comparison["teacher_macro_f1_change"],
        combined_candidate_f1=comparison["sources"]["combined"]["metrics"]["macro_f1"],
        combined_baseline_f1=comparison["baseline_sources"]["combined"]["metrics"]["macro_f1"],
        teacher_paired_family_bootstrap=bootstrap,
        mean_teacher_fit=means.to_dict(orient="index"),
        candidate_fit_minutes=sum(r["metrics"]["train_seconds"] for r in results) / 60,
        candidate_peak_memory_mb=max(r["metrics"]["peak_memory_bytes"] for r in results) / 1024**2,
        verified_candidate_hashes=len(hashes),
        comparison=comparison,
        source_correct={
            model: {
                source: dict(
                    rows=len(frame), correct=int(frame.true_index.eq(frame.predicted_index).sum())
                )
                for source, frame in data.groupby(v2.COHORT_COLUMN)
            }
            for model, data in (("mixup_sam", pooled), ("v2_e8", old))
        },
    )
    v2.write_json(summary, REPORT / "verified_summary.json")
    v2.write_json(
        dict(
            candidate_artifacts=hashes,
            reference_folds_verified=2,
            notebook_code_matches_pushed_version=True,
            data_split_sha256=contract["split_sha256"],
            registry_sha256=compute_sha256(registry_path),
            no_new_training_or_holdout_test=True,
        ),
        REPORT / "checks.json",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "comparison"}, indent=2))


if __name__ == "__main__":
    main()
