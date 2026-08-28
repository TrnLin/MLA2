from __future__ import annotations

import pandas as pd
import torch

from fashion.data.dataset import load_label_maps, load_splits
from fashion.data.multitask import AUXILIARY_IGNORE_INDEX, build_multitask_loaders


def _build(prepared_project, *, seed: int = 2753):
    return build_multitask_loaders(
        validation_fold=0,
        image_size=(80, 60),
        batch_size=2,
        seed=seed,
        num_workers=0,
        pin_memory=False,
        root=prepared_project.root,
        splits_path=prepared_project.splits,
        label_map_path=prepared_project.label_maps,
    )


def test_multitask_loaders_preserve_season_coverage_and_article_order(
    prepared_project,
) -> None:
    loaders = _build(prepared_project)
    splits = load_splits(prepared_project.splits)
    mappings = load_label_maps(prepared_project.label_maps)
    valid_season = splits.loc[
        splits["partition"].eq("development") & splits["has_season_label"]
    ]
    batch = next(iter(loaders.validation))

    assert set(loaders.training_ids) | set(loaders.validation_ids) == set(valid_season["id"])
    assert set(loaders.training_ids).isdisjoint(loaders.validation_ids)
    assert loaders.labels == tuple(mappings["season"]["classes"])
    assert loaders.auxiliary_labels == tuple(mappings["articleType"]["classes"])
    assert batch["target"].dtype == torch.int64
    assert batch["auxiliary_target"].dtype == torch.int64
    assert batch["auxiliary_mask"].dtype == torch.bool
    assert batch["image"].shape[1:] == (3, 80, 60)


def test_missing_auxiliary_label_is_masked_without_dropping_season_row(
    prepared_project,
) -> None:
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    folds = pd.to_numeric(splits["cv_fold"], errors="coerce")
    candidate_index = splits.index[
        splits["partition"].eq("development")
        & splits["season"].ne("")
        & folds.ne(0)
    ][0]
    candidate_id = int(splits.loc[candidate_index, "id"])
    splits.loc[candidate_index, "articleType"] = ""
    splits.loc[candidate_index, "has_articleType_label"] = False
    splits.to_csv(prepared_project.splits, index=False)

    loaders = _build(prepared_project)
    samples = {
        int(sample["id"]): sample
        for sample in loaders.train.dataset
    }

    assert candidate_id in loaders.training_ids
    assert candidate_id in samples
    assert int(samples[candidate_id]["auxiliary_target"]) == AUXILIARY_IGNORE_INDEX
    assert not samples[candidate_id]["auxiliary_mask"]
    assert loaders.auxiliary_training_count == len(loaders.training_ids) - 1


def test_multitask_training_shuffle_is_repeatable(prepared_project) -> None:
    first = _build(prepared_project, seed=2753)
    second = _build(prepared_project, seed=2753)

    first_ids = torch.cat([batch["id"] for batch in first.train]).tolist()
    second_ids = torch.cat([batch["id"] for batch in second.train]).tolist()

    assert first_ids == second_ids
    assert len(first.auxiliary_training_id_sha256) == 64


def test_repository_multitask_contract_has_no_current_auxiliary_gaps() -> None:
    splits = load_splits()
    valid_season = splits.loc[
        splits["partition"].eq("development") & splits["has_season_label"]
    ]

    assert len(valid_season) == 32_753
    assert int(valid_season["has_articleType_label"].sum()) == 32_753
    assert load_label_maps()["articleType"]["num_classes"] == 124
