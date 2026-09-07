"""Join prior and new external families without changing their source records."""

from fashion.task3_paths import resolve_task3_path

from pathlib import Path

import pandas as pd

from fashion.data.external_usage import file_sha256
from fashion.data.external_usage_audit import link_external_families

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent


def main():
    def read(path):
        return pd.read_csv(resolve_task3_path(path, root=ROOT), keep_default_na=False, dtype=str)

    previous = read("data/processed/teacher_plus_rare_usage_20260906/splits.csv")
    old = read("data/external/rare_usage_20260906/splits.csv")
    new = read("data/external/rare_usage_expansion_20260906/manifest.csv")
    assert set(old.external_id) == set(
        previous.loc[previous.source_dataset.ne("teacher"), "external_id"]
    )
    for frame in (old, new):
        for row in frame.itertuples():
            assert file_sha256(resolve_task3_path(row.original_path, root=ROOT)) == row.file_sha256
            assert file_sha256(resolve_task3_path(row.path, root=ROOT)) == row.sha256
        frame["source_group_id"] = frame.external_group_id
    new["source_label"] = new.usage
    frame = pd.concat([old, new], ignore_index=True).fillna("")
    assert not frame.external_id.duplicated().any()
    linked, edges = link_external_families(
        frame,
        root=ROOT,
        cache=ROOT / "data/external/rare_usage_expansion_20260906/comparison_views/v2_families",
    )
    folds = previous.loc[previous.source_dataset.ne("teacher")].set_index("external_id").cv_fold
    linked["prior_fold"] = linked.external_id.map(folds).fillna("")
    linked["intake_batch"] = linked.external_id.map(
        lambda key: "previous_120" if key in folds else "new_567"
    )
    linked.to_csv(OUT / "linked_external_families.csv", index=False)
    edges.to_csv(OUT / "family_edges.csv", index=False)
    conflict = (
        linked.loc[linked.prior_fold.ne("")].groupby("external_group_id").prior_fold.nunique()
    )
    if conflict.gt(1).any():
        raise ValueError(
            "Linked old families occupy different existing folds: "
            + str(conflict[conflict.gt(1)].to_dict())
        )
    mixed = linked.groupby("external_group_id").intake_batch.nunique()
    print(
        {
            "external_images": len(linked),
            "families": linked.external_group_id.nunique(),
            "cross_intake_families": int(mixed.gt(1).sum()),
            "existing_fold_conflicts": 0,
        }
    )


if __name__ == "__main__":
    main()
