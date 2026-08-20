from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.eda import (
    EdaPaths,
    association_matrix,
    build_population,
    cramers_v,
    deterministic_id_bins,
    distribution_table,
    drift_table,
    product_name_audit,
    row_normalized_cooccurrence,
    skew_table,
    support_band_table,
)


HEADER = (
    "id,gender,masterCategory,subCategory,articleType,baseColour,"
    "season,year,usage,productDisplayName"
)


def _write_metadata(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join([HEADER, *rows]), encoding="utf-8")


def _save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), "blue").save(path)


@pytest.fixture
def population_paths(tmp_path: Path) -> EdaPaths:
    """A selected product must have a teacher label row and two readable image views."""
    teacher_csv = tmp_path / "teacher.csv"
    original_csv = tmp_path / "original.csv"
    test_csv = tmp_path / "test.csv"
    original_images = tmp_path / "original-images"
    lowres_images = tmp_path / "lowres-images"
    _write_metadata(
        teacher_csv,
        [
            "1,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,Casual,Teacher one",
            "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,NA,Teacher two",
            "3,Men,Apparel,Bottomwear,Jeans,Black,Spring,2014,,Teacher three",
            "4,Women,Apparel,Topwear,Tshirts,White,Summer,2015,Casual,Teacher four",
        ],
    )
    _write_metadata(
        original_csv,
        [
            "1,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,Casual,Original one",
            "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,NA,Original two",
            "3,Men,Apparel,Bottomwear,Jeans,Black,Spring,2014,,Original three",
            "4,Women,Apparel,Topwear,Tshirts,White,Summer,2015,Casual,Original four",
            "90,Women,Accessories,Bags,Bags,Green,Fall,2016,Casual,Quarantined test",
            "91,Men,Footwear,Shoes,Casual Shoes,Grey,Summer,2016,Casual,Quarantined too",
        ],
    )
    test_csv.write_text(
        "id,gender,articleType,season,usage\n90,Women,Bags,Fall,Casual\n91,Men,Shoes,Summer,Casual\n",
        encoding="utf-8",
    )
    for product_id in (1, 2, 3):
        _save_image(original_images / f"{product_id}.jpg")
        _save_image(lowres_images / f"{product_id}.jpg")
    _save_image(original_images / "90.jpg")
    _save_image(lowres_images / "91.jpg")
    return EdaPaths(
        teacher_train_csv=teacher_csv,
        original_csv=original_csv,
        test_csv=test_csv,
        original_image_dir=original_images,
        lowres_image_dir=lowres_images,
        expected_test_count=2,
        expected_train_count=4,
        expected_original_count=6,
    )


def test_build_population_uses_teacher_labels_and_requires_both_image_views(
    population_paths: EdaPaths,
) -> None:
    """Using original labels or either image view alone would leak or inflate products."""
    frame, audit = build_population(population_paths)

    assert audit.source_train_ids == 4
    assert audit.usable_products == 3
    assert audit.missing_original_image_ids == (4,)
    assert audit.missing_lowres_image_ids == (4,)
    assert audit.quarantined_test_ids == (90, 91)
    assert frame["id"].tolist() == [1, 2, 3]
    assert pd.api.types.is_integer_dtype(frame["id"])
    assert not set(frame["id"]).intersection({90, 91})
    assert {"original_image_path", "lowres_image_path"}.issubset(frame.columns)
    required_metadata = {
        "id",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "year",
        "usage",
        "productDisplayName",
    }
    assert required_metadata.issubset(frame.columns)
    assert frame.loc[0, "productDisplayName"] == "Teacher one"
    assert frame.loc[1, "usage"] == "NA"
    assert frame.loc[2, "usage"] == ""


def test_build_population_rejects_teacher_and_test_id_overlap(
    population_paths: EdaPaths,
) -> None:
    """An ID in both official partitions makes selected-population safety unknowable."""
    population_paths.test_csv.write_text(
        "id,gender,articleType,season,usage\n2,Women,Tshirts,Winter,NA\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Official train/test ID overlap"):
        build_population(
            EdaPaths(
                teacher_train_csv=population_paths.teacher_train_csv,
                original_csv=population_paths.original_csv,
                test_csv=population_paths.test_csv,
                original_image_dir=population_paths.original_image_dir,
                lowres_image_dir=population_paths.lowres_image_dir,
                expected_test_count=1,
                expected_train_count=4,
                expected_original_count=6,
            )
        )


def test_build_population_preserves_complete_teacher_csv_audit(
    population_paths: EdaPaths,
) -> None:
    """Dropping parser evidence would hide malformed rows after population filtering."""
    population_paths.teacher_train_csv.write_text(
        "\n".join(
            [
                f"{HEADER},,",
                "1,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,Casual,Teacher one,,",
                "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,NA,Teacher two,,",
                "3,Men,Apparel,Bottomwear,Jeans,Black,Spring,2014,,Teacher three,,",
                "4,Women,Apparel,Topwear,Tshirts,White,Summer,2015,NA,Teacher four,,",
            ]
        ),
        encoding="utf-8",
    )

    _, audit = build_population(population_paths)

    assert audit.teacher_csv_audit.row_count == 4
    assert audit.teacher_csv_audit.phantom_columns == ("Unnamed: 10", "Unnamed: 11")
    assert audit.teacher_csv_audit.duplicate_ids == ()
    assert audit.teacher_csv_audit.blank_counts["usage"] == 1
    assert audit.teacher_csv_audit.literal_na_usage_count == 2


def test_distribution_keeps_blank_and_literal_na_as_distinct_categories() -> None:
    """Collapsing NA into blank corrupts the documented usage label distribution."""
    frame = pd.DataFrame({"usage": ["NA", "", "NA", "Casual"]})

    distribution = distribution_table(frame, "usage")

    assert distribution["count"].sum() == 4
    assert dict(zip(distribution["label"], distribution["count"])) == {
        "NA": 2,
        "": 1,
        "Casual": 1,
    }


def test_skew_table_reports_hand_calculated_entropy_and_gini() -> None:
    """Wrong class-probability calculations misstate target imbalance."""
    frame = pd.DataFrame(
        {
            "balanced": ["a", "a", "b", "b"],
            "constant": ["only", "only", "only", "only"],
        }
    )

    skew = skew_table(frame, columns=("balanced", "constant"))

    assert skew.loc["balanced", "normalized_entropy"] == pytest.approx(1.0)
    assert skew.loc["balanced", "gini_impurity"] == pytest.approx(0.5)
    assert skew.loc["constant", "gini_impurity"] == pytest.approx(0.0)
    assert skew.loc["constant", "normalized_entropy"] == pytest.approx(0.0)


def test_learned_class_metrics_exclude_blanks_but_keep_literal_na() -> None:
    """Treating blanks as classes inflates learned support and changes probabilities."""
    frame = pd.DataFrame({"usage": ["", "", "NA", "NA", "Casual"]})

    skew = skew_table(frame, columns=("usage",))
    support = support_band_table(frame, column="usage")

    assert skew.loc["usage", "total"] == 3
    assert skew.loc["usage", "classes"] == 2
    assert skew.loc["usage", "blank_count"] == 2
    assert skew.loc["usage", "literal_na_count"] == 2
    assert skew.loc["usage", "majority_share"] == pytest.approx(2 / 3)
    assert support["class_count"].tolist() == [1, 1, 0, 0, 0]
    assert support["product_count"].tolist() == [1, 2, 0, 0, 0]
    assert support["product_share"].sum() == pytest.approx(1.0)


def test_all_blank_labels_have_zero_finite_learned_class_metrics() -> None:
    """An all-blank target has no learnable classes, never undefined metrics."""
    frame = pd.DataFrame({"usage": ["", "", ""]})

    skew = skew_table(frame, columns=("usage",))
    support = support_band_table(frame, column="usage")

    assert skew.loc["usage", "total"] == 0
    assert skew.loc["usage", "classes"] == 0
    assert skew.loc["usage", "effective_class_count"] == 0.0
    assert skew.loc["usage", "blank_count"] == 3
    assert np.isfinite(skew.loc["usage"].to_numpy(dtype=float)).all()
    assert support["class_count"].sum() == 0
    assert support["product_count"].sum() == 0
    assert support["product_share"].sum() == 0.0


def test_support_bands_have_stable_report_order() -> None:
    """Sorting bands lexically would put the long-tail report in the wrong order."""
    frame = pd.DataFrame(
        {
            "articleType": (
                ["singleton"]
                + ["double"] * 2
                + ["three"] * 3
                + ["five"] * 5
                + ["ten"] * 10
            )
        }
    )

    support = support_band_table(frame)

    assert list(support["band"]) == ["1", "2", "3–4", "5–9", "10+"]
    assert support["class_count"].tolist() == [1, 1, 1, 1, 1]


def test_bias_corrected_cramers_v_is_one_for_a_large_perfect_mapping() -> None:
    """A wrong contingency calculation hides a deterministic categorical relationship."""
    left = pd.Series(["a"] * 10 + ["b"] * 10)
    right = pd.Series(["x"] * 10 + ["y"] * 10)

    assert cramers_v(left, right) == pytest.approx(1.0)


def test_cramers_v_returns_zero_for_constant_and_sparse_inputs() -> None:
    """Degenerate contingency tables must return finite zero association."""
    assert cramers_v(pd.Series(["same"] * 4), pd.Series(["x", "y", "x", "y"])) == 0.0
    assert cramers_v(pd.Series(["a"]), pd.Series(["x"])) == 0.0


def test_cramers_v_pairs_series_on_shared_indexes() -> None:
    """Outer-index pairing invents blank categories and weakens true association."""
    left = pd.Series(
        ["a"] * 20 + ["b"] * 10,
        index=range(0, 30),
    )
    right = pd.Series(
        ["x"] * 10 + ["y"] * 10 + ["ignored"] * 10,
        index=range(10, 40),
    )

    assert cramers_v(left, right) == pytest.approx(1.0)


def test_empty_metrics_return_zero_or_empty_tables() -> None:
    """Empty filtered populations must not leak NaN values into report outputs."""
    frame = pd.DataFrame({"usage": pd.Series(dtype="string")})

    assert distribution_table(frame, "usage").empty
    assert skew_table(frame, columns=("usage",)).loc["usage", "gini_impurity"] == 0.0
    assert cramers_v(pd.Series(dtype="string"), pd.Series(dtype="string")) == 0.0
    assert association_matrix(frame, ("usage",)).loc["usage", "usage"] == 0.0


def test_association_and_cooccurrence_are_symmetric_and_row_normalized() -> None:
    """Categorical summaries must not treat labels as numeric or misstate conditional share."""
    frame = pd.DataFrame(
        {
            "gender": ["Men", "Men", "Women"],
            "usage": ["Casual", "Casual", "Formal"],
        }
    )

    matrix = association_matrix(frame, ("gender", "usage"))
    cooccurrence = row_normalized_cooccurrence(frame, "gender", "usage")

    assert matrix.loc["gender", "usage"] == matrix.loc["usage", "gender"]
    assert cooccurrence.loc["Men", "Casual"] == pytest.approx(1.0)
    assert cooccurrence.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_drift_uses_total_variation_and_deterministic_id_ranges() -> None:
    """A drift summary must compare each group with the full-population distribution."""
    frame = pd.DataFrame(
        {
            "id": [1, 2, 101, 102],
            "year": ["2012", "2012", "2013", "2013"],
            "usage": ["Casual", "Casual", "Formal", "Formal"],
        }
    )

    drift = drift_table(frame, "year", "usage")
    binned = deterministic_id_bins(frame, bins=2)

    assert drift["total_variation"].tolist() == pytest.approx([0.5, 0.5])
    assert binned["id_bin"].tolist() == ["1–51", "1–51", "52–102", "52–102"]


def test_product_name_audit_reports_missing_words_and_gender_candidates() -> None:
    """Name evidence must be deterministic and flag only review candidates."""
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "gender": ["Women", "Men", "Women"],
            "productDisplayName": ["Men blue shirt", "", "Blue blue dress"],
        }
    )

    audit = product_name_audit(frame)

    assert audit["missing_name_count"] == 1
    assert audit["common_words"][0] == {"word": "blue", "count": 3}
    assert audit["gender_contradiction_candidates"] == [
        {"id": 1, "gender": "Women", "productDisplayName": "Men blue shirt"}
    ]


def test_product_name_gender_tokens_map_to_adult_and_child_labels() -> None:
    """Treating boy/girl as Men/Women creates false review candidates for child labels."""
    frame = pd.DataFrame(
        {
            "id": list(range(1, 9)),
            "gender": ["Men", "Women", "Boys", "Girls", "Women", "Men", "Girls", "Boys"],
            "productDisplayName": [
                "men shirt",
                "female dress",
                "boys shoes",
                "girl jacket",
                "male jacket",
                "women trousers",
                "boy cap",
                "girls sandals",
            ],
        }
    )

    audit = product_name_audit(frame)

    assert audit["gender_contradiction_candidates"] == [
        {"id": 5, "gender": "Women", "productDisplayName": "male jacket"},
        {"id": 6, "gender": "Men", "productDisplayName": "women trousers"},
        {"id": 7, "gender": "Girls", "productDisplayName": "boy cap"},
        {"id": 8, "gender": "Boys", "productDisplayName": "girls sandals"},
    ]
