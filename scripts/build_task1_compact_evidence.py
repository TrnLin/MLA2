"""Build compact Task 1 notebook evidence from local registered run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from fashion.config import ROOT, SPLITS_CSV, TASK1_EVIDENCE_DIR, TASK1_FIGURE_DIR
from fashion.task1.evaluation import validate_oof_predictions
from fashion.task1.plotting import write_task1_learning_curve_figure
from fashion.train.artifacts import atomic_write_csv

SELECTED_CANDIDATE = "task1_cnn_no_aug_unweighted_v1"
HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "macro_f1",
    "weighted_f1",
    "top1_accuracy",
    "top5_accuracy",
    "validation_loss",
)
PREDICTION_COLUMNS = (
    "id",
    "true_index",
    "predicted_index",
    "true_label",
    "predicted_label",
)


def _artifact_path(source_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else source_root / path


def build_compact_evidence(
    *,
    source_root: Path,
    registry_path: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    """Write only histories and selected OOF labels needed by the notebook."""
    fold_metrics = pd.read_csv(evidence_dir / "fold_metrics.csv", keep_default_na=False)
    registry = pd.read_csv(registry_path, keep_default_na=False)
    task1_rows = registry.loc[registry["task"].eq("task1")].copy()
    if task1_rows["run_id"].duplicated().any():
        raise ValueError("Task 1 registry contains duplicate run IDs")
    registry_by_id = task1_rows.set_index("run_id", drop=False)

    expected_candidates = set(fold_metrics["candidate_id"].astype(str))
    fold_counts = fold_metrics.groupby("candidate_id")["fold"].nunique()
    if len(fold_metrics) != 15 or set(fold_counts) != {5}:
        raise ValueError("compact histories require three candidates with five folds each")

    histories: list[pd.DataFrame] = []
    for row in fold_metrics.sort_values(["candidate_id", "fold"]).itertuples(index=False):
        if row.run_id not in registry_by_id.index:
            raise ValueError(f"registry row is missing for {row.run_id}")
        registered = registry_by_id.loc[row.run_id]
        history_path = _artifact_path(source_root, str(registered["history_path"]))
        history = pd.read_csv(history_path, keep_default_na=False)
        missing = set(HISTORY_COLUMNS).difference(history.columns)
        if missing:
            raise ValueError(f"history {row.run_id} is missing columns: {sorted(missing)}")
        history = history.loc[:, HISTORY_COLUMNS].copy()
        history.insert(0, "fold", int(row.fold))
        history.insert(0, "candidate_id", str(row.candidate_id))
        history.insert(0, "run_id", str(row.run_id))
        histories.append(history)
    compact_histories = pd.concat(histories, ignore_index=True)

    selected_folds = fold_metrics.loc[
        fold_metrics["candidate_id"].eq(SELECTED_CANDIDATE)
    ].sort_values("fold")
    if len(selected_folds) != 5 or set(selected_folds["fold"].astype(int)) != set(range(5)):
        raise ValueError("selected OOF evidence requires exactly five folds")
    predictions: list[pd.DataFrame] = []
    for row in selected_folds.itertuples(index=False):
        registered = registry_by_id.loc[row.run_id]
        prediction_path = _artifact_path(source_root, str(registered["prediction_path"]))
        prediction = pd.read_csv(prediction_path, usecols=PREDICTION_COLUMNS)
        prediction.insert(0, "fold", int(row.fold))
        prediction.insert(0, "run_id", str(row.run_id))
        predictions.append(prediction)
    compact_predictions = pd.concat(predictions, ignore_index=True)
    splits = pd.read_csv(SPLITS_CSV, usecols=["id", "partition"])
    development_ids = splits.loc[splits["partition"].eq("development"), "id"].astype(int)
    validate_oof_predictions(compact_predictions, development_ids)

    history_output = evidence_dir / "cnn_learning_histories.csv"
    prediction_output = evidence_dir / "selected_oof_predictions.csv"
    atomic_write_csv(history_output, compact_histories)
    atomic_write_csv(prediction_output, compact_predictions)
    histories_by_candidate: dict[str, list[pd.DataFrame]] = {}
    for (candidate_id, _), history in compact_histories.groupby(
        ["candidate_id", "fold"], sort=True
    ):
        histories_by_candidate.setdefault(str(candidate_id), []).append(
            history.sort_values("epoch", kind="stable")
        )
    figure_output = write_task1_learning_curve_figure(
        histories_by_candidate,
        output=TASK1_FIGURE_DIR / "cnn_learning_curves.png",
        include_validation_loss=False,
    )
    return {
        "candidates": sorted(expected_candidates),
        "history_rows": len(compact_histories),
        "history_path": str(history_output.relative_to(ROOT)),
        "prediction_rows": len(compact_predictions),
        "prediction_path": str(prediction_output.relative_to(ROOT)),
        "figure_path": str(figure_output.relative_to(ROOT)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, default=TASK1_EVIDENCE_DIR)
    args = parser.parse_args(argv)
    summary = build_compact_evidence(
        source_root=args.source_root.resolve(),
        registry_path=args.registry.resolve(),
        evidence_dir=args.evidence_dir.resolve(),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
