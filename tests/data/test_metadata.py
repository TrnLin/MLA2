from __future__ import annotations

import pandas as pd

from fashion.data.metadata import build_label_maps, has_valid_label, repair_product_name


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
