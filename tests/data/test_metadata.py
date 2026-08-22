from __future__ import annotations

import pandas as pd
import pytest

from fashion.data.metadata import (
    build_label_maps,
    encode_target_labels,
    has_valid_label,
    repair_product_name,
)


def test_repair_product_name_keeps_segment_order():
    assert repair_product_name(["One", None, " Two ", float("nan")]) == "One, Two"


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


def test_label_maps_fit_train_and_mask_holdout_or_quarantine_only_classes():
    frame = pd.DataFrame(
        {
            "partition": ["train", "train", "holdout", "quarantine"],
            "usage_deployed": ["Casual", "NA", "Future", "Hidden"],
            "has_usage_deployed_label": [True, True, True, True],
        }
    )
    mapping = build_label_maps(frame, ("usage",))["usage"]
    values, known, unknown = encode_target_labels(frame, "usage", mapping)

    assert mapping["source_partition"] == "train"
    assert mapping["classes"] == ["Casual", "NA"]
    assert values.tolist() == [0, 1, -1, -1]
    assert known.tolist() == [True, True, False, False]
    assert unknown == ["Future", "Hidden"]
    assert mapping["unknown_policy"] == "mask_and_report"
    assert mapping["unknown_index"] == -1
    assert values[1] == pytest.approx(1)
