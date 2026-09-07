"""External exchanges preserve identities, targets, and saved teacher boundaries."""

import pandas as pd
import pytest

from fashion.data.usage_replacement import replace_usage_dataset

pytest_plugins = ["test_usage_extension"]


def test_retired_highest_id_is_never_recycled(extension_inputs, tmp_path):
    previous, new, maps, _ = extension_inputs
    new = new.iloc[:1].copy()
    links = new[["external_id", "external_group_id"]].copy()
    retirements = pd.DataFrame([{"id": "1000000000", "reason": "Replace redundant source style"}])
    combined = replace_usage_dataset(previous, retirements, new, maps, links, root=tmp_path)
    assert len(combined) == len(previous)
    assert "1000000000" not in set(combined.id)
    added = combined.loc[combined.source_dataset.eq("new_shop")]
    assert added.id.tolist() == ["1000000001"]
    teacher = previous.loc[previous.source_dataset.eq("teacher")]
    pd.testing.assert_frame_equal(
        combined.loc[combined.source_dataset.eq("teacher"), previous.columns],
        teacher,
        check_dtype=False,
    )
    assert added.usage.eq("Home").all()
    assert added.has_usage_label.all() and not added.has_gender_label.any()
    assert combined.loc[combined.id.eq("1"), "usage"].item() == "NA"
    assert combined.loc[combined.id.eq("6"), "partition"].item() == "holdout"


@pytest.mark.parametrize("change", ["teacher", "missing_reason", "count", "identity", "image"])
def test_replacement_rejects_boundary_and_identity_violations(extension_inputs, tmp_path, change):
    previous, new, maps, _ = extension_inputs
    new = new.iloc[:1].copy()
    retirements = pd.DataFrame([{"id": "1000000000", "reason": "Reviewed replacement"}])
    if change == "teacher":
        retirements.loc[0, "id"] = "1"
    elif change == "missing_reason":
        retirements.loc[0, "reason"] = ""
    elif change == "count":
        new.loc[:, "usage"] = "Party"
    elif change == "identity":
        new.loc[:, "external_id"] = "old_shop:previous"
    else:
        new.loc[:, "sha256"] = "old_hash_6"
    links = new[["external_id", "external_group_id"]].copy()
    with pytest.raises(ValueError):
        replace_usage_dataset(previous, retirements, new, maps, links, root=tmp_path)


def test_wholly_retired_family_cannot_lose_its_fold_anchor(extension_inputs, tmp_path):
    previous, new, maps, _ = extension_inputs
    new = new.iloc[:1].copy()
    old_family = previous.loc[previous.id.eq("1000000000"), "source_group_id"].item()
    new.loc[:, "external_group_id"] = old_family
    links = new[["external_id", "external_group_id"]].copy()
    retirements = pd.DataFrame([{"id": "1000000000", "reason": "Reviewed replacement"}])
    with pytest.raises(ValueError, match="wholly retired family"):
        replace_usage_dataset(previous, retirements, new, maps, links, root=tmp_path)
