from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fashion.data.dataset import load_label_maps, load_splits
from fashion.data.torch import build_task_loaders


def _build(prepared_project, *, validation_fold: int = 0, seed: int = 2753):
    return build_task_loaders(
        validation_fold=validation_fold,
        image_size=(80, 60),
        batch_size=2,
        seed=seed,
        num_workers=0,
        pin_memory=False,
        root=prepared_project.root,
        splits_path=prepared_project.splits,
        label_map_path=prepared_project.label_maps,
    )


def test_build_task_loaders_uses_only_valid_canonical_fold_rows(prepared_project) -> None:
    loaders = _build(prepared_project, validation_fold=0)
    splits = load_splits(prepared_project.splits)
    valid = splits.loc[
        splits["partition"].eq("development") & splits["has_season_label"]
    ]

    assert set(loaders.training_ids).isdisjoint(loaders.validation_ids)
    assert set(loaders.training_ids) | set(loaders.validation_ids) == set(valid["id"])
    assert set(splits.loc[splits["id"].isin(loaders.validation_ids), "cv_fold"].astype(int)) == {0}
    assert set(splits.loc[splits["id"].isin(loaders.training_ids), "cv_fold"].astype(int)) <= {
        1,
        2,
        3,
        4,
    }
    protected_ids = set(splits.loc[splits["partition"].ne("development"), "id"])
    assert not ((set(loaders.training_ids) | set(loaders.validation_ids)) & protected_ids)
    assert loaders.stats.image_count == len(loaders.training_ids)
    assert loaders.audit()["id_overlap"] == 0


def test_loader_targets_follow_canonical_label_order(prepared_project) -> None:
    loaders = _build(prepared_project)
    mapping = load_label_maps(prepared_project.label_maps)["season"]
    batch = next(iter(loaders.validation))

    assert loaders.labels == tuple(mapping["classes"])
    assert loaders.label_to_index == mapping["label_to_index"]
    assert batch["image"].shape[1:] == (3, 80, 60)
    assert batch["target"].dtype == torch.int64
    assert set(batch["target"].tolist()) <= set(range(len(loaders.labels)))


def test_train_shuffle_is_repeatable_for_same_seed(prepared_project) -> None:
    first = _build(prepared_project, seed=2753)
    second = _build(prepared_project, seed=2753)

    first_ids = torch.cat([batch["id"] for batch in first.train]).tolist()
    second_ids = torch.cat([batch["id"] for batch in second.train]).tolist()

    assert first_ids == second_ids


def test_loader_rejects_stats_from_another_fold(prepared_project) -> None:
    loaders = _build(prepared_project, validation_fold=0)
    wrong_stats = replace(loaders.stats, validation_fold=1)

    with pytest.raises(ValueError, match="different fold"):
        build_task_loaders(
            validation_fold=0,
            image_size=(80, 60),
            batch_size=2,
            root=prepared_project.root,
            splits_path=prepared_project.splits,
            label_map_path=prepared_project.label_maps,
            stats=wrong_stats,
        )


def test_repository_season_contract_has_expected_32753_rows_and_order() -> None:
    splits = load_splits()
    mapping = load_label_maps()["season"]
    valid = splits.loc[
        splits["partition"].eq("development") & splits["has_season_label"]
    ]

    assert len(valid) == 32_753
    assert mapping["classes"] == ["Fall", "Spring", "Summer", "Winter"]
    assert mapping["label_to_index"] == {
        "Fall": 0,
        "Spring": 1,
        "Summer": 2,
        "Winter": 3,
    }
