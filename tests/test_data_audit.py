from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from fashion.data_audit import (
    SchemaAudit,
    audit_csv,
    dhash,
    hamming_distance,
    hierarchy_conflicts,
)


def test_audit_csv_preserves_literal_na_and_repairs_spilled_product_names(
    tmp_path: Path,
) -> None:
    """A parser that turns NA into missing or drops spills corrupts source metadata."""
    path = tmp_path / "styles.csv"
    path.write_text(
        "\n".join(
            [
                "id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName,,",
                "1,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,NA,Alpha, Beta, Gamma",
                "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,,Simple name,,",
            ]
        ),
        encoding="utf-8",
    )

    frame, audit = audit_csv(path)

    assert frame.loc[0, "usage"] == "NA"
    assert frame.loc[1, "usage"] == ""
    assert frame.loc[0, "productDisplayName"] == "Alpha, Beta, Gamma"
    assert isinstance(audit, SchemaAudit)
    assert audit.literal_na_usage_count == 1
    assert audit.blank_counts["usage"] == 1


def test_audit_csv_normalizes_valid_ids_to_integers(tmp_path: Path) -> None:
    """String IDs would break numeric joins and ID-boundary checks."""
    path = tmp_path / "styles.csv"
    path.write_text(
        "\n".join(
            [
                "id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName",
                "1,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,Casual,First",
                "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,Casual,Second",
            ]
        ),
        encoding="utf-8",
    )

    frame, _ = audit_csv(path)

    assert pd.api.types.is_integer_dtype(frame["id"])
    assert frame["id"].tolist() == [1, 2]


def test_audit_csv_rejects_empty_files_with_clear_error(tmp_path: Path) -> None:
    """An empty metadata file must report its bad input instead of leaking StopIteration."""
    path = tmp_path / "empty.csv"
    path.touch()

    with pytest.raises(ValueError, match="empty"):
        audit_csv(path)


def test_audit_csv_lists_missing_required_columns(tmp_path: Path) -> None:
    """A truncated metadata schema must name the missing fields for repair."""
    path = tmp_path / "missing-columns.csv"
    path.write_text("id,gender\n1,Men\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"articleType.*usage"):
        audit_csv(path)


def test_audit_csv_rejects_non_integer_ids_with_clear_error(tmp_path: Path) -> None:
    """A non-integer ID would make ID reconciliation unreliable."""
    path = tmp_path / "invalid-id.csv"
    path.write_text(
        "\n".join(
            [
                "id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName",
                "not-an-id,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,Casual,First",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer ID"):
        audit_csv(path)


def test_hierarchy_conflicts_keep_conflicting_product_ids() -> None:
    """Dropping IDs would make an inconsistent hierarchy impossible to review."""
    frame = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "articleType": ["Tshirts", "Tshirts", "Jeans"],
            "masterCategory": ["Apparel", "Apparel", "Apparel"],
            "subCategory": ["Topwear", "Bottomwear", "Bottomwear"],
        }
    )

    conflicts = hierarchy_conflicts(frame)

    assert conflicts["articleType"].tolist() == ["Tshirts"]
    assert conflicts.loc[0, "ids"] == ("101", "102")


def test_dhash_returns_fixed_width_hex_and_exact_hamming_distance() -> None:
    """Changing the dHash representation would silently break comparison distances."""
    blank = Image.new("L", (8, 8), 0)
    one_bit = blank.copy()
    one_bit.putpixel((0, 0), 255)

    blank_hash = dhash(blank)
    one_bit_hash = dhash(one_bit)

    assert blank_hash == "0000000000000000"
    assert one_bit_hash == "2000000000000000"
    assert hamming_distance(blank_hash, one_bit_hash) == 1
