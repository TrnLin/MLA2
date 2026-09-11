"""Read-only helpers for the Task 3 development report.

Historical locks also contain final-evaluation assets. Reject those paths before
opening files so development replay never needs reserved-image predictions.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

SELECTION = Path("reports/task3/gender_mixup_selection_20260911")


def development_path(root: Path, relative: str) -> Path:
    """Resolve a development asset, rejecting reserved-result locations."""
    relative = relative.replace("reports/task3_", "reports/task3/", 1)
    if re.search(r"holdout|(?:^|[/_])test(?:[/_.]|$)", relative.lower()):
        raise ValueError(f"Final-evaluation asset is outside development scope: {relative}")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def verify_development_files(root: Path, files: dict[str, str]) -> int:
    """Verify the development subset of an older mixed-purpose file lock."""
    checked = 0
    for relative, expected in files.items():
        try:
            path = development_path(root, relative)
        except ValueError:
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
        checked += 1
    return checked


def verify_mixup_selection(root: Path) -> dict:
    """Verify the completed training artifact without opening evaluation data."""
    folder = root / SELECTION / "refit"
    receipt = json.loads((folder / "model_manifest.json").read_text())
    assert receipt["run_id"] == (
        "t3_gender_name_truth_mixup_alpha020_refit_20260911T041436Z_3af0b94e"
    )
    assert receipt["files"]["final_epoch.pt"]["sha256"] == (
        "860f688162cccfcd903e8e4874b368697c0637b6e6a15baae3b4d4a3008f4ef9"
    )
    for asset in receipt["files"].values():
        path = development_path(root, str((folder / asset["path"]).relative_to(root)))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
    assert receipt["status"] == "complete" and receipt["scratch"] is True
    assert receipt["selected_epoch"] == 30 and receipt["validation_used"] is False
    assert receipt["metrics"]["training_rows"] == 32773
    assert receipt["metrics"]["validation_rows"] == 0
    result = dict(receipt)
    for key, filename in [
        ("checkpoint", "final_epoch.pt"),
        ("config", "config.json"),
        ("normalization", "normalization.json"),
        ("history", "history.csv"),
    ]:
        asset = receipt["files"][filename]
        result[key] = {
            "path": str((folder / asset["path"]).relative_to(root)),
            "sha256": asset["sha256"],
        }
    result.update(
        training_rows=32773,
        parameter_count=390181,
        selected_on="2026-09-11",
        selection_basis="development only",
        selection_status="selected_from_development",
    )
    return result


def experiment_inventory(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union saved registries by run ID, retaining incomplete attempts.

    Completion takes precedence over stale running snapshots. Conflicting
    complete metric records are counted for review, never silently pooled.
    The inventory counts runs; it does not select the most favourable repeat.
    """
    sources = {root / "results/runs.csv"}
    for base in [root / "reports/task3", root / "results/evidence/task3"]:
        for pattern in [
            "runs.csv",
            "*registry*.csv",
            "*refit_runs.csv",
            "drive_runs.csv",
            "source_runs.csv",
            "classical_runs.csv",
        ]:
            for path in base.rglob(pattern):
                try:
                    development_path(root, str(path.relative_to(root)))
                except ValueError:
                    continue
                columns = pd.read_csv(path, nrows=0).columns
                if {"run_id", "experiment_id", "target", "status"} <= set(columns):
                    sources.add(path)
    snapshots = []
    for path in sorted(sources):
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
        frame = frame.loc[frame.target.isin(["gender", "usage"])].copy()
        frame["registry_source"] = str(path.relative_to(root))
        snapshots.append(frame)
    all_rows = pd.concat(snapshots, ignore_index=True).fillna("")
    all_rows["status_rank"] = all_rows.status.map({"running": 0, "failed": 1, "complete": 2})
    assert all_rows.status_rank.notna().all()
    rows = all_rows.sort_values("status_rank", kind="stable").drop_duplicates("run_id", keep="last")
    rows = rows.copy()
    rows["role"] = "Development candidate"
    rows.loc[rows.experiment_id.str.contains("eda_view_probe"), "role"] = (
        "EDA probe; sampled population"
    )
    rows.loc[rows.experiment_id.str.contains("clean_bn_probe"), "role"] = (
        "Training-only BN diagnostic"
    )
    rows.loc[rows.experiment_id.str.contains("refit"), "role"] = (
        "Full-development refit; no OOF score"
    )
    rows.loc[rows.status.ne("complete"), "role"] = "Incomplete attempt; no candidate score"
    rows["model_variant"] = rows.experiment_id
    probe = rows.experiment_id.str.contains("eda_view_probe")
    rows.loc[probe, "model_variant"] = rows.loc[probe, "model_family"]
    coverage = (
        rows.groupby(["target", "model_variant", "role", "status"], dropna=False)
        .agg(
            runs=("run_id", "size"),
            folds=("validation_fold", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    complete = all_rows.loc[all_rows.status.eq("complete")]
    conflicts = complete.groupby("run_id").metrics_json.nunique()
    rows["complete_metric_versions"] = rows.run_id.map(conflicts).fillna(0).astype(int)
    return rows.sort_values(["target", "experiment_id", "run_id"]), coverage


def gender_development_comparison(root: Path, selection: dict) -> pd.DataFrame:
    """Keep every recorded stage and add matched five-fold recipe audits.

    Original and corrected scoring labels stay separate. Audit rows are marked
    explicitly so a second score of the same checkpoint is not a new model.
    """
    stages = pd.read_csv(root / SELECTION / "all_gender_development_stages.csv")
    stages["Score source"] = "Registry"
    additions = []
    for label_basis, key in [
        ("Original", "original_teacher_metrics"),
        ("Name-corrected", "metrics"),
    ]:
        for model, metric in selection[key].items():
            additions.append(
                {
                    "experiment": f"{model} · Five-fold audit",
                    "folds": "[0, 1, 2, 3, 4]",
                    "label_basis": label_basis,
                    "rows": metric["support"],
                    "macro_f1": metric["macro_f1"],
                    "accuracy": metric["accuracy"],
                    "Score source": "IEEE audit",
                }
            )
    result = pd.concat([stages, pd.DataFrame(additions)], ignore_index=True)
    result[["macro_f1", "accuracy"]] *= 100
    return result.rename(columns={"macro_f1": "F1 (%)", "accuracy": "Accuracy (%)"})


def selection_tables(root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load checked saved statistics; no training or prediction happens here."""
    folder = root / SELECTION
    lock = json.loads((folder / "notebook_assets.json").read_text())
    for relative, expected in lock.items():
        path = development_path(root, relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["holdout_used"] is False and summary["teacher_test_used"] is False
    metrics = pd.DataFrame(summary["metrics"]).loc[
        ["macro_f1", "accuracy", "nll", "brier", "ece_15"]
    ]
    metrics.loc["mean_clean_training_f1"] = summary["mean_clean_training_f1"]
    metrics.loc["mean_gap"] = summary["mean_gap"]
    metrics.loc["fold_f1_std"] = summary["fold_f1_std"]
    keys = [
        "all_five",
        "all_five_100000_draw_sensitivity",
        "screen_folds",
        "additional_folds",
        "original_teacher",
    ]
    intervals = pd.DataFrame({key: summary["bootstrap"][key] for key in keys}).T
    return (
        summary,
        metrics.astype(float),
        intervals[["point", "lower_95", "upper_95", "repetitions"]].astype(float),
    )
