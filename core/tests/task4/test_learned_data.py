from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fashion.config import ROOT
from fashion.data.splits import cv_assignment_digest, id_set_digest
from fashion.task4.augmentation import DEFAULT_GEOMETRY_POLICY
from fashion.task4.cache import (
    DevelopmentImageCache,
    ensure_development_image_cache,
    fit_cached_fold_rgb_statistics,
)
from fashion.task4.learned_data import (
    CrossSourcePairDataset,
    FamilyBatchSampler,
    TrainingPairsProvenance,
    build_training_pairs,
)
from fashion.task4.preprocessing import PreprocessingContract

CONTRACT = PreprocessingContract(width=240, height=320)


def _variant_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [5, 2, 4, 3, 1],
            "partition": [
                "development",
                "holdout",
                "quarantine",
                "development",
                "development",
            ],
            "cv_fold": [0, "", "", 1, 2],
            "teacher_path": [f"teacher/{value}.jpg" for value in [5, 2, 4, 3, 1]],
            "external_path": [f"v1/{value}.jpg" for value in [5, 2, 4, 3, 1]],
            "external_sha256": [f"v1-{value}" for value in [5, 2, 4, 3, 1]],
            "duplicate_group": [f"duplicate-{value}" for value in [5, 2, 4, 3, 1]],
            "product_family_group": [f"family-{value}" for value in [5, 2, 4, 3, 1]],
            "teacher_width": [60] * 5,
            "teacher_height": [80] * 5,
            "teacher_file_size_bytes": [100] * 5,
            "external_width": [240] * 5,
            "external_height": [320] * 5,
            "external_aspect_ratio": [0.75] * 5,
            "external_mode": ["RGB"] * 5,
            "external_format": ["JPEG"] * 5,
            "external_file_size_bytes": [1000] * 5,
        }
    )


def _canonical_splits() -> pd.DataFrame:
    ids = [5, 2, 4, 3, 1]
    partitions = ["development", "holdout", "quarantine", "development", "development"]
    return pd.DataFrame(
        {
            "id": ids,
            "path": [f"teacher/{value}.jpg" for value in ids],
            "sha256": [f"teacher-{value}" for value in ids],
            "duplicate_group": [f"duplicate-{value}" for value in ids],
            "product_name_key": [f"name-{value}" for value in ids],
            "product_family_group": [f"family-{value}" for value in ids],
            "partition": partitions,
            "cv_fold": [0, "", "", 1, 2],
            "is_cross_role_exact_duplicate": [False] * 5,
            "is_cross_role_near_duplicate": [False] * 5,
            "has_conflicting_target_labels": [False] * 5,
            "conflicting_targets": [""] * 5,
            "quarantine_reason": ["", "", "decode_failure", "", ""],
        }
    )


def test_build_training_pairs_keeps_only_sorted_unique_training_ids() -> None:
    result = build_training_pairs(
        _variant_rows(),
        validation_fold=1,
        canonical_splits=_canonical_splits(),
    )

    assert result["id"].tolist() == [1, 5]
    assert result["cv_fold"].tolist() == [2, 0]
    assert result["partition"].tolist() == ["development", "development"]
    assert {
        "teacher_path",
        "external_path",
        "teacher_sha256",
        "external_sha256",
        "sha256",
        "duplicate_group",
        "product_family_group",
    }.issubset(result.columns)
    provenance = result.attrs.get("task4_training_provenance")
    assert getattr(provenance, "validation_fold", None) == 1
    assert getattr(provenance, "split_fingerprint", None) == cv_assignment_digest(
        _canonical_splits()
    )


def test_build_training_pairs_requires_canonical_splits() -> None:
    rows = _variant_rows()
    rows["sha256"] = rows["id"].map(lambda value: f"teacher-{value}")
    with pytest.raises(TypeError, match="canonical_splits"):
        build_training_pairs(rows, validation_fold=1)


def test_build_training_pairs_joins_teacher_hash_from_canonical_splits() -> None:
    result = build_training_pairs(
        _variant_rows(),
        validation_fold=1,
        canonical_splits=_canonical_splits(),
    )

    assert result["sha256"].tolist() == ["teacher-1", "teacher-5"]
    assert result["teacher_sha256"].tolist() == ["teacher-1", "teacher-5"]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("partition", "teacher_test", "partition"),
        ("cv_fold", "bad", "fold"),
    ],
)
def test_build_training_pairs_rejects_malformed_roles_and_folds_before_pixel_access(
    column: str,
    value: object,
    message: str,
) -> None:
    rows = _variant_rows()
    rows.loc[rows["id"].eq(5), column] = value
    rows.loc[rows["id"].eq(5), "teacher_path"] = "does-not-exist.jpg"

    with pytest.raises(ValueError, match=message):
        build_training_pairs(
            rows,
            validation_fold=1,
            canonical_splits=_canonical_splits(),
        )


def test_build_training_pairs_rejects_protected_rows_with_folds() -> None:
    rows = _variant_rows()
    rows.loc[rows["partition"].eq("holdout"), "cv_fold"] = 0

    with pytest.raises(ValueError, match="holdout|protected"):
        build_training_pairs(
            rows,
            validation_fold=1,
            canonical_splits=_canonical_splits(),
        )


def test_build_training_pairs_rejects_duplicate_variant_ids() -> None:
    duplicate = pd.concat([_variant_rows(), _variant_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_training_pairs(
            duplicate,
            validation_fold=1,
            canonical_splits=_canonical_splits(),
        )


@pytest.mark.parametrize("path_column", ["teacher_path", "external_path"])
def test_build_training_pairs_rejects_real_path_stem_id_mismatch(
    path_column: str,
) -> None:
    mismatch = _variant_rows()
    mismatch.loc[mismatch["id"].eq(5), path_column] = "changed/999.jpg"

    with pytest.raises(ValueError, match="same product ID"):
        build_training_pairs(
            mismatch,
            validation_fold=1,
            canonical_splits=_canonical_splits(),
        )


def test_build_training_pairs_rejects_variant_and_canonical_group_disagreement() -> None:
    rows = _variant_rows()
    rows.loc[rows["id"].eq(5), "duplicate_group"] = "wrong"

    with pytest.raises(ValueError, match="duplicate_group"):
        build_training_pairs(
            rows,
            validation_fold=1,
            canonical_splits=_canonical_splits(),
        )


def _id_digest(ids: list[int]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode("ascii")).hexdigest()


def _cache(
    source: str,
    ids: list[int],
    *,
    fill: tuple[int, int, int],
) -> DevelopmentImageCache:
    images = np.full((len(ids), 320, 240, 3), 255, dtype=np.uint8)
    bounds = np.tile(np.array([40, 40, 280, 200], dtype=np.int32), (len(ids), 1))
    for index in range(len(ids)):
        images[index, 40:280, 40:200] = fill
        images[index, 80:240, 80:160, 0] = (index + 1) * 20 % 256
    return DevelopmentImageCache(
        cache_dir=Path("."),
        ids=np.asarray(ids, dtype=np.int64),
        images=images,
        content_bounds=bounds,
        manifest={
            "scope": "development",
            "source": source,
            "rows": len(ids),
            "id_sha256": _id_digest(ids),
            "source_fingerprint": f"{source}-fingerprint",
            "contract": CONTRACT.to_dict(),
        },
    )


def _pairs() -> pd.DataFrame:
    pairs = pd.DataFrame(
        {
            "id": [1],
            "partition": ["development"],
            "cv_fold": [0],
            "teacher_path": ["teacher/1.jpg"],
            "external_path": ["v1/1.jpg"],
            "teacher_sha256": ["teacher-1"],
            "external_sha256": ["v1-1"],
            "sha256": ["teacher-1"],
            "duplicate_group": ["duplicate-1"],
            "product_family_group": ["family-1"],
        }
    )
    pairs.attrs["task4_training_provenance"] = TrainingPairsProvenance(
        validation_fold=1,
        split_fingerprint=cv_assignment_digest(_canonical_splits()),
    )
    return pairs


def _statistics(source: str) -> dict[str, object]:
    return {
        "validation_fold": 1,
        "split_fingerprint": cv_assignment_digest(_canonical_splits()),
        "training_rows": 1,
        "training_id_sha256": id_set_digest([1]),
        "mean": [0.25, 0.5, 0.75],
        "std": [0.25, 0.25, 0.25],
        "source": source,
        "source_fingerprint": f"{source}-fingerprint",
        "contract": CONTRACT.to_dict(),
    }


def _dataset(
    *,
    teacher_cache: DevelopmentImageCache | None = None,
    v1_cache: DevelopmentImageCache | None = None,
    teacher_statistics: dict[str, object] | None = None,
    v1_statistics: dict[str, object] | None = None,
    geometry: bool = False,
) -> CrossSourcePairDataset:
    return CrossSourcePairDataset(
        _pairs(),
        teacher_cache=teacher_cache or _cache("teacher", [1, 2], fill=(64, 96, 128)),
        v1_cache=v1_cache or _cache("v1", [1, 2], fill=(32, 64, 96)),
        teacher_statistics=teacher_statistics or _statistics("teacher"),
        v1_statistics=v1_statistics or _statistics("v1"),
        validation_fold=1,
        split_fingerprint=cv_assignment_digest(_canonical_splits()),
        geometry_policy=DEFAULT_GEOMETRY_POLICY if geometry else None,
    )


def test_dataset_retains_canonical_fold_and_split_provenance() -> None:
    dataset = _dataset()

    assert dataset.validation_fold == 1
    assert dataset.split_fingerprint == cv_assignment_digest(_canonical_splits())


def test_official_statistics_helper_output_constructs_the_pair_dataset(
    tmp_path: Path,
) -> None:
    splits = _canonical_splits()
    fingerprint = cv_assignment_digest(splits)
    colours = {5: (0, 0, 0), 3: (255, 0, 0), 1: (255, 255, 255)}
    for source in ("teacher", "v1"):
        (tmp_path / "images" / source).mkdir(parents=True)
        for product_id, colour in colours.items():
            Image.new("RGB", (240, 320), colour).save(
                tmp_path / "images" / source / f"{product_id}.png"
            )
    development = pd.DataFrame(
        {
            "id": [5, 3, 1],
            "partition": ["development"] * 3,
            "cv_fold": [0, 1, 2],
            "teacher_path": [f"images/teacher/{value}.png" for value in (5, 3, 1)],
            "teacher_sha256": [f"teacher-{value}" for value in (5, 3, 1)],
            "external_path": [f"images/v1/{value}.png" for value in (5, 3, 1)],
            "external_sha256": [f"v1-{value}" for value in (5, 3, 1)],
        }
    )
    caches: dict[str, DevelopmentImageCache] = {}
    statistics: dict[str, dict[str, object]] = {}
    for source, path_column, sha_column in (
        ("teacher", "teacher_path", "teacher_sha256"),
        ("v1", "external_path", "external_sha256"),
    ):
        caches[source] = ensure_development_image_cache(
            development,
            path_column=path_column,
            sha_column=sha_column,
            source=source,
            contract=CONTRACT,
            cache_root=tmp_path / "cache",
            root=tmp_path,
        )
        statistics[source] = fit_cached_fold_rgb_statistics(
            caches[source],
            development,
            validation_fold=1,
            canonical_splits=splits,
        )

    pairs = build_training_pairs(
        _variant_rows(),
        validation_fold=1,
        canonical_splits=splits,
    )
    dataset = CrossSourcePairDataset(
        pairs,
        teacher_cache=caches["teacher"],
        v1_cache=caches["v1"],
        teacher_statistics=statistics["teacher"],
        v1_statistics=statistics["v1"],
        validation_fold=1,
        split_fingerprint=fingerprint,
    )

    assert statistics["teacher"]["split_fingerprint"] == fingerprint
    assert statistics["v1"]["split_fingerprint"] == fingerprint
    assert dataset.split_fingerprint == fingerprint
    assert dataset.pairs["id"].tolist() == [1, 5]
    assert dataset[0]["teacher"].shape == (3, 320, 240)
    assert dataset[0]["v1_content_mask"].all()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("pair_fold", "pair.*validation fold"),
        ("pair_split", "pair.*split fingerprint"),
        ("statistics_split", "statistics split fingerprint"),
    ],
)
def test_dataset_rejects_forged_pair_and_statistics_provenance(
    change: str,
    message: str,
) -> None:
    pairs = _pairs()
    teacher_statistics = _statistics("teacher")
    v1_statistics = _statistics("v1")
    if change == "pair_fold":
        pairs.attrs["task4_training_provenance"] = TrainingPairsProvenance(
            validation_fold=2,
            split_fingerprint=cv_assignment_digest(_canonical_splits()),
        )
    elif change == "pair_split":
        pairs.attrs["task4_training_provenance"] = TrainingPairsProvenance(
            validation_fold=1,
            split_fingerprint="c" * 64,
        )
    else:
        teacher_statistics["split_fingerprint"] = "c" * 64

    with pytest.raises(ValueError, match=message):
        CrossSourcePairDataset(
            pairs,
            teacher_cache=_cache("teacher", [1, 2], fill=(64, 96, 128)),
            v1_cache=_cache("v1", [1, 2], fill=(32, 64, 96)),
            teacher_statistics=teacher_statistics,
            v1_statistics=v1_statistics,
            validation_fold=1,
            split_fingerprint=cv_assignment_digest(_canonical_splits()),
        )


def test_dataset_returns_identity_and_exact_source_normalization() -> None:
    item = _dataset()[0]

    assert item["id"] == 1
    assert item["product_family_group"] == "family-1"
    assert item["duplicate_group"] == "duplicate-1"
    assert item["sha256"] == "teacher-1"
    assert item["teacher"].shape == (3, 320, 240)
    assert item["v1"].shape == (3, 320, 240)
    assert item["teacher_content_mask"].shape == (320, 240)
    assert item["v1_content_mask"].shape == (320, 240)
    assert item["teacher_content_mask"].dtype == torch.bool
    assert item["v1_content_mask"].dtype == torch.bool
    expected_mask = torch.zeros((320, 240), dtype=torch.bool)
    expected_mask[40:280, 40:200] = True
    assert torch.equal(item["teacher_content_mask"], expected_mask)
    assert torch.equal(item["v1_content_mask"], expected_mask)
    assert item["teacher"].dtype == torch.float32
    assert item["v1"].dtype == torch.float32
    assert item["teacher"][:, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert item["v1"][:, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert item["teacher"][:, 40, 40].tolist() == pytest.approx(
        [
            (64 / 255 - 0.25) / 0.25,
            (96 / 255 - 0.5) / 0.25,
            (128 / 255 - 0.75) / 0.25,
        ],
        abs=1e-6,
    )
    assert item["v1"][:, 40, 40].tolist() == pytest.approx(
        [
            (32 / 255 - 0.25) / 0.25,
            (64 / 255 - 0.5) / 0.25,
            (96 / 255 - 0.75) / 0.25,
        ],
        abs=1e-6,
    )


@pytest.mark.parametrize(
    ("geometry", "expected_source"),
    [(False, "none"), (True, None)],
)
def test_dataset_items_collate_in_real_dataloader(
    geometry: bool,
    expected_source: str | None,
) -> None:
    loader = DataLoader(_dataset(geometry=geometry), batch_size=1)

    batch = next(iter(loader))

    assert batch["id"].tolist() == [1]
    assert batch["teacher"].shape == (1, 3, 320, 240)
    assert batch["v1"].shape == (1, 3, 320, 240)
    assert batch["teacher_content_mask"].shape == (1, 320, 240)
    assert batch["v1_content_mask"].shape == (1, 320, 240)
    assert batch["teacher_content_mask"].dtype == torch.bool
    assert batch["v1_content_mask"].dtype == torch.bool
    if expected_source is None:
        assert batch["augmented_source"][0] in {"teacher", "v1"}
    else:
        assert batch["augmented_source"] == [expected_source]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("cache_ids", "cache IDs"),
        ("source", "source"),
        ("contract", "contract"),
        ("statistics_fold", "validation fold"),
        ("training_digest", "training-ID digest"),
        ("source_fingerprint", "fingerprint"),
    ],
)
def test_dataset_rejects_cache_and_statistics_mismatches(
    change: str,
    message: str,
) -> None:
    teacher = _cache("teacher", [1, 2], fill=(64, 96, 128))
    v1 = _cache("v1", [1, 2], fill=(32, 64, 96))
    teacher_stats = _statistics("teacher")
    v1_stats = _statistics("v1")
    if change == "cache_ids":
        v1 = _cache("v1", [1, 3], fill=(32, 64, 96))
    elif change == "source":
        v1.manifest["source"] = "teacher"
    elif change == "contract":
        v1.manifest["contract"] = PreprocessingContract(width=120, height=160).to_dict()
    elif change == "statistics_fold":
        teacher_stats["validation_fold"] = 2
    elif change == "training_digest":
        teacher_stats["training_id_sha256"] = id_set_digest([2])
    else:
        v1_stats["source_fingerprint"] = "wrong"

    with pytest.raises(ValueError, match=message):
        _dataset(
            teacher_cache=teacher,
            v1_cache=v1,
            teacher_statistics=teacher_stats,
            v1_statistics=v1_stats,
        )


def test_dataset_geometry_changes_exactly_one_source_and_is_epoch_deterministic() -> None:
    clean = _dataset()
    augmented = _dataset(geometry=True)

    augmented.set_epoch(4)
    first = augmented[0]
    second = augmented[0]
    assert torch.equal(first["teacher"], second["teacher"])
    assert torch.equal(first["v1"], second["v1"])
    assert first["augmented_source"] in {"teacher", "v1"}
    clean_item = clean[0]
    changed = [
        source
        for source in ("teacher", "v1")
        if not torch.equal(first[source], clean_item[source])
    ]
    assert changed == [first["augmented_source"]]
    changed_masks = [
        source
        for source in ("teacher", "v1")
        if not torch.equal(
            first[f"{source}_content_mask"],
            clean_item[f"{source}_content_mask"],
        )
    ]
    assert changed_masks == [first["augmented_source"]]
    clean_source = "v1" if first["augmented_source"] == "teacher" else "teacher"
    assert torch.equal(
        first[f"{clean_source}_content_mask"],
        clean_item[f"{clean_source}_content_mask"],
    )

    augmented.set_epoch(5)
    next_epoch = augmented[0]
    assert not (
        torch.equal(first["teacher"], next_epoch["teacher"])
        and torch.equal(first["v1"], next_epoch["v1"])
    )


def test_dataset_epoch_updates_reach_persistent_dataloader_workers() -> None:
    dataset = _dataset(geometry=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    try:
        dataset.set_epoch(0)
        epoch_zero = next(iter(loader))
        dataset.set_epoch(1)
        epoch_one = next(iter(loader))
        dataset.set_epoch(1)
        repeated = next(iter(loader))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()

    assert not (
        torch.equal(epoch_zero["teacher"], epoch_one["teacher"])
        and torch.equal(epoch_zero["v1"], epoch_one["v1"])
    )
    assert torch.equal(epoch_one["teacher"], repeated["teacher"])
    assert torch.equal(epoch_one["v1"], repeated["v1"])
    assert epoch_one["augmented_source"] == repeated["augmented_source"]


def _sampler_rows(*, extra_ids: int = 32) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    product_id = 1
    for family_index in range(16):
        for member in range(2):
            records.append(
                {
                    "id": product_id,
                    "product_family_group": f"valid-{family_index}",
                    "sha256": f"sha-{product_id}",
                    "duplicate_group": f"duplicate-{product_id}",
                }
            )
            product_id += 1
    for singleton in range(extra_ids):
        records.append(
            {
                "id": product_id,
                "product_family_group": f"singleton-{singleton}",
                "sha256": f"sha-{product_id}",
                "duplicate_group": f"duplicate-{product_id}",
            }
        )
        product_id += 1
    records.extend(
        [
            {
                "id": product_id,
                "product_family_group": "excluded",
                "sha256": "same-sha",
                "duplicate_group": "same-duplicate",
            },
            {
                "id": product_id + 1,
                "product_family_group": "excluded",
                "sha256": "same-sha",
                "duplicate_group": "same-duplicate",
            },
        ]
    )
    return pd.DataFrame(records)


def test_family_sampler_builds_sixteen_valid_pairs_plus_thirty_two_other_ids() -> None:
    rows = _sampler_rows(extra_ids=40)
    sampler = FamilyBatchSampler(rows)
    sampler.set_epoch(3)

    batch = next(iter(sampler))
    batch_rows = rows.iloc[batch].reset_index(drop=True)

    assert len(batch) == 64
    assert batch_rows["id"].nunique() == 64
    for offset in range(0, 32, 2):
        left = batch_rows.iloc[offset]
        right = batch_rows.iloc[offset + 1]
        assert left["id"] != right["id"]
        assert left["product_family_group"] == right["product_family_group"]
        assert left["sha256"] != right["sha256"]
        assert left["duplicate_group"] != right["duplicate_group"]
        assert left["product_family_group"] != "excluded"
    assert len(set(batch[32:])) == 32


def test_family_sampler_is_repeatable_per_epoch_and_changes_between_epochs() -> None:
    rows = _sampler_rows(extra_ids=96)
    sampler = FamilyBatchSampler(rows)
    sampler.set_epoch(7)
    first = list(sampler)
    sampler.set_epoch(7)
    repeated = list(sampler)
    sampler.set_epoch(8)
    changed = list(sampler)

    assert repeated == first
    assert changed != first


def _scheduled_ids(rows: pd.DataFrame, sampler: FamilyBatchSampler) -> set[int]:
    return {
        int(sampler.pairs.iloc[index]["id"])
        for batch in sampler
        for index in batch
    }


def _eligible_family_ids(rows: pd.DataFrame) -> set[int]:
    eligible: set[int] = set()
    for _, group in rows.groupby("product_family_group"):
        records = group.to_dict("records")
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                if (
                    left["sha256"] != right["sha256"]
                    and left["duplicate_group"] != right["duplicate_group"]
                ):
                    eligible.update((int(left["id"]), int(right["id"])))
    return eligible


def test_family_sampler_maximizes_epoch_wide_coverage_and_rotates_omissions() -> None:
    rows = _sampler_rows(extra_ids=96)
    sampler = FamilyBatchSampler(rows)
    family_slots = len(sampler) * 32
    other_slots = len(sampler) * 32
    maximum = min(
        len(rows),
        other_slots + min(family_slots, len(_eligible_family_ids(rows))),
    )

    sampler.set_epoch(0)
    epoch_zero = _scheduled_ids(rows, sampler)
    sampler.set_epoch(1)
    epoch_one = _scheduled_ids(rows, sampler)

    assert len(epoch_zero) == maximum
    assert len(epoch_one) == maximum
    assert set(rows["id"]) - epoch_zero != set(rows["id"]) - epoch_one


def test_family_sampler_maximizes_coverage_on_canonical_training_rows() -> None:
    variant = pd.read_csv(
        ROOT / "data/processed/task4/external_variant_index.csv.gz",
        keep_default_na=False,
    )
    splits = pd.read_csv(ROOT / "data/processed/splits.csv", keep_default_na=False)
    rows = build_training_pairs(
        variant,
        validation_fold=1,
        canonical_splits=splits,
    )
    sampler = FamilyBatchSampler(rows)
    maximum = min(
        len(rows),
        len(sampler) * 32
        + min(len(sampler) * 32, len(_eligible_family_ids(rows))),
    )

    assert len(_scheduled_ids(rows, sampler)) == maximum


def _dataset_for_sampler_rows(rows: pd.DataFrame) -> CrossSourcePairDataset:
    pairs = rows.copy()
    pairs["partition"] = "development"
    pairs["cv_fold"] = 0
    pairs["teacher_path"] = pairs["id"].map(lambda value: f"teacher/{value}.jpg")
    pairs["external_path"] = pairs["id"].map(lambda value: f"v1/{value}.jpg")
    pairs["teacher_sha256"] = pairs["sha256"]
    pairs["external_sha256"] = pairs["id"].map(lambda value: f"v1-{value}")
    pairs.attrs["task4_training_provenance"] = TrainingPairsProvenance(
        validation_fold=1,
        split_fingerprint="b" * 64,
    )
    ids = sorted(pairs["id"].astype(int))
    statistics = {
        "validation_fold": 1,
        "split_fingerprint": "b" * 64,
        "training_rows": len(ids),
        "training_id_sha256": id_set_digest(ids),
        "mean": [0.25, 0.5, 0.75],
        "std": [0.25, 0.25, 0.25],
        "contract": CONTRACT.to_dict(),
    }
    teacher_statistics = {
        **statistics,
        "source": "teacher",
        "source_fingerprint": "teacher-fingerprint",
    }
    v1_statistics = {
        **statistics,
        "source": "v1",
        "source_fingerprint": "v1-fingerprint",
    }
    return CrossSourcePairDataset(
        pairs,
        teacher_cache=_cache("teacher", ids, fill=(64, 96, 128)),
        v1_cache=_cache("v1", ids, fill=(32, 64, 96)),
        teacher_statistics=teacher_statistics,
        v1_statistics=v1_statistics,
        validation_fold=1,
        split_fingerprint="b" * 64,
    )


def test_unsorted_sampler_positions_match_the_dataset_canonical_order() -> None:
    rows = _sampler_rows(extra_ids=40).sample(frac=1.0, random_state=91)
    dataset = _dataset_for_sampler_rows(rows)
    sampler = FamilyBatchSampler(rows)
    sampler.set_epoch(2)
    expected_batches = [
        [int(sampler.pairs.iloc[index]["id"]) for index in batch]
        for batch in sampler
    ]
    loader = DataLoader(dataset, batch_sampler=sampler)

    observed_batches = [batch["id"].tolist() for batch in loader]

    assert observed_batches == expected_batches


def test_family_sampler_ignores_targets_and_handles_singletons_as_other_ids() -> None:
    rows = _sampler_rows(extra_ids=32)
    without_targets = copy.deepcopy(rows)

    batch = next(iter(FamilyBatchSampler(without_targets)))

    assert len(batch) == 64
    assert any(
        str(rows.iloc[index]["product_family_group"]).startswith("singleton-")
        for index in batch
    )


def test_family_sampler_fails_clearly_without_sixteen_valid_families() -> None:
    rows = _sampler_rows(extra_ids=32)
    rows.loc[rows["product_family_group"].eq("valid-15"), "sha256"] = "same"

    with pytest.raises(ValueError, match="16 valid families"):
        FamilyBatchSampler(rows)
