from __future__ import annotations

import pandas as pd
import pytest

from fashion.data.families import build_product_families, normalize_product_name
from fashion.data.metadata import (
    build_label_maps,
    encode_target_labels,
    has_valid_label,
    is_missing_product_name,
    repair_product_name,
)


def test_repair_product_name_keeps_segment_order():
    assert repair_product_name(["One", None, " Two ", float("nan")]) == "One, Two"


def test_product_name_na_is_missing_but_usage_na_is_a_valid_label():
    assert is_missing_product_name("NA")
    assert not is_missing_product_name("N/A")
    assert repair_product_name(["NA", "repaired spill"]) == "repaired spill"
    assert normalize_product_name(" NA ") == ""
    assert has_valid_label("NA")


def test_missing_na_names_do_not_create_one_product_family(tmp_path):
    targets = ("articleType", "season", "gender", "usage")
    manifest = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "sha256": ["a" * 64, "b" * 64, "c" * 64, "d" * 64],
            "productDisplayName": ["NA", "NA", "Real name", " real  name "],
            "articleType": ["Clutches", "Perfume", "Bag", "Bag"],
            "season": ["Summer"] * 4,
            "gender": ["Women"] * 4,
            "usage": ["Casual", "NA", "Casual", "Casual"],
            **{f"has_{target}_label": [True] * 4 for target in targets},
        }
    )
    prediction = pd.DataFrame({"id": [101], "sha256": ["e" * 64]})
    candidates = pd.DataFrame(columns=["accepted_near_duplicate"])
    manifest_path = tmp_path / "manifest.csv"
    prediction_path = tmp_path / "prediction.csv"
    candidates_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "families.csv.gz"
    manifest.to_csv(manifest_path, index=False)
    prediction.to_csv(prediction_path, index=False)
    candidates.to_csv(candidates_path, index=False)

    families = build_product_families(
        train_manifest_csv=manifest_path,
        prediction_manifest_csv=prediction_path,
        candidates_csv=candidates_path,
        output_csv=output_path,
        summary_output=tmp_path / "summary.json",
        targets=targets,
    ).set_index("id")

    assert families.loc[1, "product_name_key"] == ""
    assert families.loc[2, "product_name_key"] == ""
    assert families.loc[1, "product_family_group"] != families.loc[2, "product_family_group"]
    assert families.loc[3, "product_family_group"] == families.loc[4, "product_family_group"]


def test_label_validity_does_not_treat_na_as_missing():
    assert has_valid_label("NA")
    assert not has_valid_label("")
    assert not has_valid_label(None)


def test_label_maps_are_sorted_and_bijective():
    manifest = pd.DataFrame(
        {
            "articleType": ["B", "A", ""],
            "has_articleType_label": [True, True, False],
        }
    )
    mapping = build_label_maps(manifest, ("articleType",))["articleType"]
    assert mapping["classes"] == ["A", "B"]
    assert mapping["label_to_index"] == {"A": 0, "B": 1}
    assert mapping["index_to_label"] == {"0": "A", "1": "B"}


def test_label_maps_fit_development_and_report_protected_only_classes():
    frame = pd.DataFrame(
        {
            "partition": ["development", "development", "holdout", "quarantine"],
            "usage": ["Casual", "NA", "Future", "Hidden"],
            "has_usage_label": [True, True, True, True],
        }
    )
    mapping = build_label_maps(frame, ("usage",))["usage"]
    values, known, unknown = encode_target_labels(frame, "usage", mapping)

    assert mapping["source_scope"] == "development"
    assert mapping["classes"] == ["Casual", "NA"]
    assert values.tolist() == [0, 1, -1, -1]
    assert known.tolist() == [True, True, False, False]
    assert unknown == ["Future", "Hidden"]
    assert mapping["unknown_policy"] == "report_without_expanding_during_development"
    assert mapping["unknown_index"] == -1
    assert values[1] == pytest.approx(1)
