from __future__ import annotations

import pandas as pd
import pytest

from fashion.data.dataset import load_splits
from fashion.eda.scope import derive_duplicate_evidence, select_modelling_scope


def _split_rows() -> pd.DataFrame:
    rows = []
    for item_id, partition, article_type in [
        (1, "train", "A"),
        (2, "val", "B"),
        (3, "holdout", "Protected"),
        (4, "quarantine", "ProtectedTwin"),
    ]:
        rows.append(
            {
                "id": item_id,
                "partition": partition,
                "sha256": f"sha-{item_id}",
                "articleType": article_type,
                "season": "Summer",
                "gender": "Unisex",
                "usage": "Casual",
                "has_articleType_label": True,
                "has_season_label": True,
                "has_gender_label": True,
                "has_usage_label": True,
            }
        )
    return pd.DataFrame(rows)


def _audit_rows() -> pd.DataFrame:
    rows = [
        {
            "id": item_id,
            "role": "train",
            "decode_ok": True,
            "sha256": f"sha-{item_id}",
            "width": 60,
            "height": 80,
            "aspect_ratio": 0.75,
            "mode": "RGB",
            "format": "JPEG",
            "file_size_bytes": 1000,
        }
        for item_id in range(1, 5)
    ]
    rows.append(
        {
            "id": 99,
            "role": "prediction",
            "decode_ok": True,
            "sha256": "prediction-only",
            "width": 999,
            "height": 999,
            "aspect_ratio": 1,
            "mode": "L",
            "format": "PNG",
            "file_size_bytes": 999999,
        }
    )
    return pd.DataFrame(rows)


def test_modelling_scope_is_training_only():
    train, images = select_modelling_scope(_split_rows(), _audit_rows())
    assert train["id"].tolist() == [1]
    assert images["id"].tolist() == [1]
    assert 999 not in images["width"].tolist()


def test_modelling_scope_requires_complete_audit_coverage():
    audit = _audit_rows()
    audit = audit[audit["id"].ne(1)]
    with pytest.raises(ValueError, match="coverage mismatch"):
        select_modelling_scope(_split_rows(), audit)


def test_duplicate_evidence_matches_quarantine(prepared_project):
    splits = load_splits(prepared_project.splits)
    audit = pd.read_csv(prepared_project.audit / "image_audit.csv")
    groups = pd.read_csv(prepared_project.audit / "exact_duplicate_groups.csv")
    evidence = derive_duplicate_evidence(splits, audit, groups)
    assert evidence["cross_role_exact_duplicate_groups"] == 1
    assert evidence["cross_role_training_samples"] == 1
    assert evidence["cross_role_prediction_samples"] == 1
    assert evidence["conflicting_label_exact_duplicate_groups"] == 1
    assert evidence["conflicting_label_training_samples"] == 2
    assert evidence["quarantined_training_samples"] == 3
    assert evidence["official_prediction_ids_in_labelled_splits"] == 0


def test_duplicate_evidence_rejects_cross_role_row_outside_quarantine(prepared_project):
    splits = load_splits(prepared_project.splits)
    splits.loc[splits["id"].eq(10), "partition"] = "train"
    audit = pd.read_csv(prepared_project.audit / "image_audit.csv")
    groups = pd.read_csv(prepared_project.audit / "exact_duplicate_groups.csv")
    with pytest.raises(ValueError, match="outside quarantine"):
        derive_duplicate_evidence(splits, audit, groups)


def test_duplicate_evidence_rejects_crossed_exact_duplicate_group(prepared_project):
    splits = load_splits(prepared_project.splits)
    splits.loc[splits["id"].eq(1), "partition"] = "train"
    audit = pd.read_csv(prepared_project.audit / "image_audit.csv")
    groups = pd.read_csv(prepared_project.audit / "exact_duplicate_groups.csv")

    with pytest.raises(ValueError, match="exact SHA-256 duplicate group crosses partitions"):
        derive_duplicate_evidence(splits, audit, groups)
