"""Orchestration for the rebuildable data preparation stages."""

from __future__ import annotations

from pathlib import Path

from fashion.config import ROOT
from fashion.data.audit import audit_raw_data
from fashion.data.manifests import build_manifests
from fashion.data.splits import make_splits
from fashion.data.statistics import compute_normalization_stats


def prepare_data(root: str | Path = ROOT, workers: int | None = None) -> None:
    """Run audit, manifest, split, and train-only statistics in order."""
    root = Path(root)
    train_dir = root / "data/raw/teacher/train"
    prediction_dir = root / "data/raw/teacher/test"
    processed_dir = root / "data/processed"
    audit_dir = processed_dir / "audit"

    audit_raw_data(
        train_csv=train_dir / "styles_train.csv",
        prediction_csv=prediction_dir / "styles_prediction.csv",
        train_image_dir=train_dir / "images_train",
        prediction_image_dir=prediction_dir / "images_test",
        output_dir=audit_dir,
        root=root,
        workers=workers,
    )
    build_manifests(
        train_csv=train_dir / "styles_train.csv",
        prediction_csv=prediction_dir / "styles_prediction.csv",
        image_audit_csv=audit_dir / "image_audit.csv",
        train_output=processed_dir / "train_manifest.csv",
        prediction_output=processed_dir / "prediction_manifest.csv",
        label_maps_output=processed_dir / "label_maps.json",
    )
    make_splits(
        train_manifest_csv=processed_dir / "train_manifest.csv",
        duplicate_groups_csv=audit_dir / "exact_duplicate_groups.csv",
        output_csv=processed_dir / "splits.csv",
        summary_output=processed_dir / "split_summary.json",
        development_summary_output=processed_dir / "development_class_summary.csv",
    )
    compute_normalization_stats(
        splits_csv=processed_dir / "splits.csv",
        output_path=processed_dir / "normalization.json",
        root=root,
        workers=workers,
    )
