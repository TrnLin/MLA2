"""Generate compact EDA evidence and report figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import (
    AUDIT_DIR,
    EDA_EVIDENCE_DIR,
    EDA_FIGURE_DIR,
    LABEL_MAPS_JSON,
    NORMALIZATION_JSON,
    ROOT,
    SPLITS_CSV,
    TARGET_COLUMNS,
)
from fashion.data.dataset import load_splits
from fashion.data.statistics import train_ids_digest
from fashion.eda.diagnostics import build_validation_diagnostics, summarize_target
from fashion.eda.plots import (
    configure_style,
    plot_article_type_long_tail,
    plot_development_balance,
    plot_image_profile,
    plot_target_distributions,
)
from fashion.eda.provenance import csv_record, file_record, runtime_record
from fashion.eda.reconciliation import build_data_reconciliation
from fashion.eda.scope import (
    bool_mask,
    derive_duplicate_evidence,
    scope_record,
    select_modelling_scope,
)

PLOT_FILENAMES = (
    "article_type_long_tail.png",
    "target_distributions.png",
    "image_profile.png",
    "development_balance.png",
)


def _target_table(targets: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, summary in targets.items():
        counts = summary["class_counts"]
        most_common_label, most_common_count = next(iter(counts.items()))
        rows.append(
            {
                "target": target,
                "source_partition": "train",
                "valid_labels": summary["valid_labels"],
                "missing_labels": summary["missing_labels"],
                "classes": summary["num_classes"],
                "most_common_class": most_common_label,
                "most_common_count": most_common_count,
                "singleton_classes": len(summary["singleton_classes"]),
                "classes_below_10": len(summary["rare_classes_lt10"]),
            }
        )
    return pd.DataFrame(rows)


def _validation_table(diagnostics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, diagnostic in diagnostics.items():
        rows.append(
            {
                "target": target,
                "train_valid_labels": diagnostic["valid_label_denominators"]["train"],
                "validation_valid_labels": diagnostic["valid_label_denominators"]["val"],
                "training_classes": diagnostic["training_class_count"],
                "training_classes_seen_in_validation": diagnostic[
                    "training_classes_observed_in_validation"
                ],
                "training_class_coverage_percent": diagnostic["training_class_coverage_percent"],
                "max_absolute_gap_percentage_points": diagnostic["distribution_gap_summary"][
                    "max_absolute_percentage_point_gap"
                ],
            }
        )
    return pd.DataFrame(rows)


def generate_eda(
    splits_csv: str | Path = SPLITS_CSV,
    image_audit_csv: str | Path = AUDIT_DIR / "image_audit.csv",
    duplicate_groups_csv: str | Path = AUDIT_DIR / "exact_duplicate_groups.csv",
    csv_summary_json: str | Path = AUDIT_DIR / "csv_summary.json",
    label_maps_json: str | Path = LABEL_MAPS_JSON,
    normalization_json: str | Path = NORMALIZATION_JSON,
    figure_dir: str | Path = EDA_FIGURE_DIR,
    evidence_dir: str | Path = EDA_EVIDENCE_DIR,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Generate EDA while keeping holdout, quarantine, and prediction outcomes closed."""
    root = Path(root)
    splits_csv = Path(splits_csv)
    image_audit_csv = Path(image_audit_csv)
    duplicate_groups_csv = Path(duplicate_groups_csv)
    csv_summary_json = Path(csv_summary_json)
    label_maps_json = Path(label_maps_json)
    normalization_json = Path(normalization_json)
    figure_dir = Path(figure_dir)
    evidence_dir = Path(evidence_dir)
    for path in (
        splits_csv,
        image_audit_csv,
        duplicate_groups_csv,
        csv_summary_json,
        label_maps_json,
        normalization_json,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required EDA input not found: {path}")
    figure_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(splits_csv)
    image_audit = pd.read_csv(image_audit_csv)
    duplicate_groups = pd.read_csv(duplicate_groups_csv)
    csv_summary = json.loads(csv_summary_json.read_text(encoding="utf-8"))
    label_maps = json.loads(label_maps_json.read_text(encoding="utf-8"))
    normalization = json.loads(normalization_json.read_text(encoding="utf-8"))
    if normalization.get("source_partition") != "train":
        raise ValueError("normalization statistics are not training-only")

    train, modelling_images = select_modelling_scope(splits, image_audit)
    current_train_ids = train["id"].astype(int).tolist()
    if normalization.get("num_images") != len(current_train_ids):
        raise ValueError(
            "normalization num_images does not match the current training partition "
            f"({normalization.get('num_images')!r} != {len(current_train_ids)})"
        )
    current_train_ids_digest = train_ids_digest(current_train_ids)
    if normalization.get("train_ids_digest") != current_train_ids_digest:
        raise ValueError("normalization train_ids_digest does not match current training IDs")
    diagnostics = build_validation_diagnostics(splits)
    duplicate_evidence = derive_duplicate_evidence(splits, image_audit, duplicate_groups)
    data_reconciliation = build_data_reconciliation(csv_summary, splits, label_maps)
    target_summaries = {target: summarize_target(train, target) for target in TARGET_COLUMNS}

    configure_style()
    plot_paths = [
        plot_article_type_long_tail(train, figure_dir / PLOT_FILENAMES[0]),
        plot_target_distributions(train, figure_dir / PLOT_FILENAMES[1]),
        plot_image_profile(modelling_images, figure_dir / PLOT_FILENAMES[2]),
        plot_development_balance(splits, diagnostics, figure_dir / PLOT_FILENAMES[3]),
    ]

    target_summary_path = evidence_dir / "target_summary.csv"
    validation_summary_path = evidence_dir / "validation_summary.csv"
    reconciliation_path = evidence_dir / "data_reconciliation.json"
    _target_table(target_summaries).to_csv(target_summary_path, index=False)
    _validation_table(diagnostics).to_csv(validation_summary_path, index=False)
    reconciliation_path.write_text(json.dumps(data_reconciliation, indent=2), encoding="utf-8")

    resolution_counts = {
        f"{int(width)}x{int(height)}": int(count)
        for (width, height), count in modelling_images.groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .items()
    }
    file_quantiles = modelling_images["file_size_bytes"].quantile([0.5, 0.95, 0.99])
    candidate_images = image_audit[image_audit["error"].fillna("").ne("non_image_extension")]
    structural_inventory = {
        "labelled_image_backed_rows": len(splits),
        "partition_counts": {
            str(partition): int(count)
            for partition, count in splits["partition"].value_counts().items()
        },
        "source_image_audit": {
            "candidate_train_images": int(candidate_images["role"].eq("train").sum()),
            "candidate_prediction_images": int(candidate_images["role"].eq("prediction").sum()),
            "ignored_non_image_entries": int(
                image_audit["error"].fillna("").eq("non_image_extension").sum()
            ),
            "decode_failures": int((~bool_mask(candidate_images["decode_ok"])).sum()),
        },
    }

    code_paths = [
        root / "src/fashion/data/dataset.py",
        root / "src/fashion/data/splits.py",
        root / "src/fashion/data/statistics.py",
        root / "src/fashion/eda/generate.py",
        root / "src/fashion/eda/scope.py",
        root / "src/fashion/eda/diagnostics.py",
        root / "src/fashion/eda/plots.py",
        root / "src/fashion/eda/provenance.py",
        root / "src/fashion/eda/reconciliation.py",
    ]
    summary: dict[str, Any] = {
        "schema_version": "4.0.0",
        "scope": scope_record(train, modelling_images),
        "structural_inventory": structural_inventory,
        "data_reconciliation": data_reconciliation,
        "modelling_evidence": {
            "targets": target_summaries,
            "images": {
                "source_partition": "train",
                "eligible_images": len(modelling_images),
                "resolution_counts": resolution_counts,
                "modes": {
                    str(label): int(count)
                    for label, count in modelling_images["mode"].value_counts().items()
                },
                "formats": {
                    str(label): int(count)
                    for label, count in modelling_images["format"].value_counts().items()
                },
                "file_size_bytes": {
                    "minimum": int(modelling_images["file_size_bytes"].min()),
                    "median": float(file_quantiles.loc[0.5]),
                    "p95": float(file_quantiles.loc[0.95]),
                    "p99": float(file_quantiles.loc[0.99]),
                    "maximum": int(modelling_images["file_size_bytes"].max()),
                },
                "quality_scope_note": (
                    "Evidence covers decode validity, dimensions, format, file size, and colour "
                    "mode. It does not claim subjective visual quality."
                ),
            },
            "normalization": {
                key: normalization[key]
                for key in (
                    "source_partition",
                    "image_size",
                    "pad_color",
                    "mean",
                    "std",
                    "num_images",
                    "train_ids_digest",
                )
            },
        },
        "validation_diagnostics": {
            "source_partitions": ["train", "val"],
            "class_selection_partition": "train",
            "measure": "within-partition percentage of valid labels",
            "targets": diagnostics,
        },
        "duplicate_and_leakage_control": duplicate_evidence,
        "provenance": {
            "inputs": [
                csv_record(splits_csv, root),
                csv_record(image_audit_csv, root),
                csv_record(duplicate_groups_csv, root),
                file_record(csv_summary_json, root),
                file_record(label_maps_json, root),
                file_record(normalization_json, root),
                file_record(root / "pyproject.toml", root),
            ],
            "code": [file_record(path, root) for path in code_paths],
            "runtime": runtime_record(),
            "outputs": {
                "plots": [file_record(path, root) for path in plot_paths],
                "tables": [
                    csv_record(target_summary_path, root),
                    csv_record(validation_summary_path, root),
                    file_record(reconciliation_path, root),
                ],
            },
            "reproducibility_note": (
                "Counts should match unchanged inputs and code. PNG bytes may vary with fonts "
                "or rendering libraries across systems."
            ),
        },
    }
    summary_path = evidence_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
