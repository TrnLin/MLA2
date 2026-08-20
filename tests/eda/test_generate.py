from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.eda.generate import PLOT_FILENAMES, generate_eda


def _generate_kwargs(prepared_project, tmp_path):
    return {
        "splits_csv": prepared_project.splits,
        "image_audit_csv": prepared_project.audit / "image_audit.csv",
        "duplicate_groups_csv": prepared_project.audit / "exact_duplicate_groups.csv",
        "csv_summary_json": prepared_project.audit / "csv_summary.json",
        "label_maps_json": prepared_project.label_maps,
        "normalization_json": prepared_project.normalization,
        "figure_dir": tmp_path / "figures",
        "evidence_dir": tmp_path / "evidence",
        "root": ROOT,
    }


def test_generate_eda_writes_scoped_evidence_and_figures(prepared_project, tmp_path):
    figures = tmp_path / "figures"
    evidence = tmp_path / "evidence"
    summary = generate_eda(
        splits_csv=prepared_project.splits,
        image_audit_csv=prepared_project.audit / "image_audit.csv",
        duplicate_groups_csv=prepared_project.audit / "exact_duplicate_groups.csv",
        csv_summary_json=prepared_project.audit / "csv_summary.json",
        label_maps_json=prepared_project.label_maps,
        normalization_json=prepared_project.normalization,
        figure_dir=figures,
        evidence_dir=evidence,
        root=ROOT,
    )
    assert summary["schema_version"] == "4.0.0"
    assert summary["scope"]["modelling_partition"] == "train"
    assert summary["scope"]["prediction_images_in_modelling_evidence"] == 0
    assert summary["scope"]["protected_target_partitions"] == ["holdout", "quarantine"]
    inventory = summary["structural_inventory"]["source_image_audit"]
    assert inventory["ignored_non_image_entries"] == 1
    assert inventory["decode_failures"] == 0
    assert summary["modelling_evidence"]["targets"]["usage"]["class_counts"]["NA"] == 1
    reconciliation = summary["data_reconciliation"]
    assert reconciliation["raw_to_usable"] == {
        "raw_metadata_rows": 12,
        "image_backed_rows": 11,
        "excluded_missing_image_rows": 1,
        "missing_valid_image_ids": [11],
        "product_names_repaired": 1,
    }
    article = reconciliation["target_taxonomy"]["articleType"]
    assert article["raw_valid_classes"] == 4
    assert article["active_image_backed_classes"] == 3
    assert article["removed_by_missing_images"] == [
        {"class": "D", "raw_count": 1, "missing_image_ids": [11]}
    ]
    usage = reconciliation["target_taxonomy"]["usage"]
    assert usage["active_image_backed_classes"] == 2
    assert usage["literal_NA_is_valid"] is True
    duplicate_evidence = summary["duplicate_and_leakage_control"]
    assert duplicate_evidence["conflicting_label_exact_duplicate_groups"] == 1
    assert duplicate_evidence["conflicting_label_training_samples"] == 2
    assert duplicate_evidence["quarantined_training_samples"] == 3
    assert {path.name for path in figures.iterdir()} == set(PLOT_FILENAMES)
    assert (evidence / "summary.json").exists()
    assert (evidence / "target_summary.csv").exists()
    assert (evidence / "validation_summary.csv").exists()
    assert (evidence / "data_reconciliation.json").exists()


def test_generated_plot_provenance_matches_bytes(prepared_project, tmp_path):
    figures = tmp_path / "figures"
    evidence = tmp_path / "evidence"
    summary = generate_eda(
        splits_csv=prepared_project.splits,
        image_audit_csv=prepared_project.audit / "image_audit.csv",
        duplicate_groups_csv=prepared_project.audit / "exact_duplicate_groups.csv",
        csv_summary_json=prepared_project.audit / "csv_summary.json",
        label_maps_json=prepared_project.label_maps,
        normalization_json=prepared_project.normalization,
        figure_dir=figures,
        evidence_dir=evidence,
        root=ROOT,
    )
    for record in summary["provenance"]["outputs"]["plots"]:
        path = Path(record["path"])
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_generate_eda_rejects_crossed_exact_duplicate_group(prepared_project, tmp_path):
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    splits.loc[splits["id"].eq(1), "partition"] = "train"
    invalid_path = tmp_path / "invalid-splits.csv"
    splits.to_csv(invalid_path, index=False)
    kwargs = _generate_kwargs(prepared_project, tmp_path)
    kwargs["splits_csv"] = invalid_path

    with pytest.raises(ValueError, match="exact SHA-256 duplicate group crosses partitions"):
        generate_eda(**kwargs)


@pytest.mark.parametrize(
    ("field", "stale_value", "message"),
    [
        ("num_images", -1, "normalization num_images does not match"),
        ("train_ids_digest", "stale-digest", "normalization train_ids_digest does not match"),
    ],
)
def test_generate_eda_rejects_stale_normalization(
    prepared_project, tmp_path, field, stale_value, message
):
    normalization = json.loads(prepared_project.normalization.read_text(encoding="utf-8"))
    normalization[field] = stale_value
    stale_path = tmp_path / f"stale-{field}.json"
    stale_path.write_text(json.dumps(normalization), encoding="utf-8")
    kwargs = _generate_kwargs(prepared_project, tmp_path)
    kwargs["normalization_json"] = stale_path

    with pytest.raises(ValueError, match=message):
        generate_eda(**kwargs)
