"""Fold-safe paired data and family-aware batches for Task 4 learned models."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from fashion.config import RANDOM_SEED
from fashion.data.splits import (
    cv_assignment_digest,
    id_set_digest,
    validate_split_structure,
)
from fashion.task4.augmentation import (
    GeometryPolicy,
    apply_geometry,
    sample_geometry,
)
from fashion.task4.cache import DevelopmentImageCache
from fashion.task4.preprocessing import (
    PreprocessedImage,
    PreprocessingContract,
    normalize_for_model,
)

_FIXED_CONTRACT = PreprocessingContract(width=240, height=320)
_PAIR_COLUMNS = {
    "id",
    "partition",
    "cv_fold",
    "teacher_path",
    "external_path",
    "external_sha256",
    "duplicate_group",
    "product_family_group",
}

__all__ = (
    "CrossSourcePairDataset",
    "FamilyBatchSampler",
    "TrainingPairsProvenance",
    "build_training_pairs",
)


def _validate_validation_fold(validation_fold: int) -> int:
    if (
        isinstance(validation_fold, bool)
        or not isinstance(validation_fold, Integral)
        or int(validation_fold) not in range(5)
    ):
        raise ValueError("validation_fold must be an integer in range(5)")
    return int(validation_fold)


@dataclass(frozen=True, slots=True)
class TrainingPairsProvenance:
    """Canonical split identity carried by a training-pair frame."""

    validation_fold: int
    split_fingerprint: str

    def __post_init__(self) -> None:
        _validate_validation_fold(self.validation_fold)
        if (
            len(self.split_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.split_fingerprint)
        ):
            raise ValueError("split fingerprint must be a lowercase SHA-256 digest")


def _nonblank(frame: pd.DataFrame, columns: set[str]) -> None:
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"paired data column {column!r} must not be blank")


def _validated_variant_index(variant_index: pd.DataFrame) -> pd.DataFrame:
    if missing := _PAIR_COLUMNS.difference(variant_index.columns):
        raise ValueError(f"variant index is missing columns: {sorted(missing)}")
    if "sha256" not in variant_index and "teacher_sha256" not in variant_index:
        raise ValueError("variant index is missing the teacher SHA-256 column")
    if variant_index.empty:
        raise ValueError("variant index must not be empty")
    working = variant_index.copy()
    numeric_ids = pd.to_numeric(working["id"], errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.mod(1).eq(0).all():
        raise ValueError("paired product IDs must be integer-compatible")
    working["id"] = numeric_ids.astype(np.int64)
    if working["id"].duplicated().any():
        raise ValueError("paired product IDs must be unique")

    allowed_partitions = {"development", "holdout", "quarantine"}
    if unknown := set(working["partition"]) - allowed_partitions:
        raise ValueError(f"variant index has unknown partition roles: {sorted(unknown)}")
    development = working["partition"].eq("development")
    protected = ~development
    folds = pd.to_numeric(
        working["cv_fold"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )
    if folds.loc[development].isna().any():
        raise ValueError("every development row requires a valid fold")
    if (
        not folds.loc[development].mod(1).eq(0).all()
        or not folds.loc[development].between(0, 4).all()
    ):
        raise ValueError("development fold values must be integers in range(5)")
    if folds.loc[protected].notna().any():
        raise ValueError("holdout and quarantine protected rows must not have folds")
    working["cv_fold"] = pd.Series(pd.NA, index=working.index, dtype="Int64")
    working.loc[development, "cv_fold"] = folds.loc[development].astype(int)

    for path_column in ("teacher_path", "external_path"):
        stems = working[path_column].astype(str).map(lambda value: Path(value).stem)
        if not stems.str.fullmatch(r"\d+").all():
            raise ValueError("teacher and V1 views must use the same product ID")
        path_ids = stems.astype(np.int64)
        if not path_ids.equals(working["id"]):
            raise ValueError("teacher and V1 views must use the same product ID")

    if "sha256" not in working:
        working["sha256"] = working["teacher_sha256"].astype(str)
    if "teacher_sha256" not in working:
        working["teacher_sha256"] = working["sha256"].astype(str)
    _nonblank(
        working,
        {
            "teacher_path",
            "external_path",
            "external_sha256",
            "sha256",
            "duplicate_group",
            "product_family_group",
            "teacher_sha256",
        },
    )
    if not working["teacher_sha256"].astype(str).equals(working["sha256"].astype(str)):
        raise ValueError("teacher SHA-256 aliases must match")
    return working


def _join_canonical_provenance(
    variant_index: pd.DataFrame,
    canonical_splits: pd.DataFrame,
) -> pd.DataFrame:
    validate_split_structure(canonical_splits)
    canonical = canonical_splits[
        [
            "id",
            "path",
            "sha256",
            "duplicate_group",
            "product_family_group",
            "partition",
            "cv_fold",
        ]
    ].rename(
        columns={
            "path": "_canonical_teacher_path",
            "sha256": "_canonical_sha256",
            "duplicate_group": "_canonical_duplicate_group",
            "product_family_group": "_canonical_product_family_group",
            "partition": "_canonical_partition",
            "cv_fold": "_canonical_cv_fold",
        }
    )
    if variant_index["id"].duplicated().any():
        raise ValueError("paired product IDs must be unique")
    working = variant_index.merge(canonical, on="id", how="left", validate="one_to_one")
    if working["_canonical_sha256"].isna().any():
        raise ValueError("canonical splits do not cover every variant ID")
    if "sha256" not in working:
        working["sha256"] = working["_canonical_sha256"]
    if "teacher_sha256" not in working:
        working["teacher_sha256"] = working["_canonical_sha256"]
    for column in ("duplicate_group", "product_family_group"):
        if column not in working:
            working[column] = working[f"_canonical_{column}"]
    return working


def _validate_canonical_agreement(working: pd.DataFrame) -> None:
    if "_canonical_sha256" not in working:
        return
    comparisons = (
        ("teacher_path", "_canonical_teacher_path"),
        ("sha256", "_canonical_sha256"),
        ("teacher_sha256", "_canonical_sha256"),
        ("duplicate_group", "_canonical_duplicate_group"),
        ("product_family_group", "_canonical_product_family_group"),
        ("partition", "_canonical_partition"),
    )
    for variant_column, canonical_column in comparisons:
        if not working[variant_column].astype(str).equals(
            working[canonical_column].astype(str)
        ):
            raise ValueError(f"variant {variant_column} disagrees with canonical splits")
    variant_folds = pd.to_numeric(working["cv_fold"], errors="coerce")
    canonical_folds = pd.to_numeric(working["_canonical_cv_fold"], errors="coerce")
    if not np.array_equal(
        variant_folds.fillna(-1).astype(int).to_numpy(),
        canonical_folds.fillna(-1).astype(int).to_numpy(),
    ):
        raise ValueError("variant cv_fold disagrees with canonical splits")


def build_training_pairs(
    variant_index: pd.DataFrame,
    *,
    validation_fold: int,
    canonical_splits: pd.DataFrame,
) -> pd.DataFrame:
    """Return same-ID teacher/V1 rows allowed in the current training round."""
    fold = _validate_validation_fold(validation_fold)
    joined = _join_canonical_provenance(variant_index, canonical_splits)
    working = _validated_variant_index(joined)
    _validate_canonical_agreement(working)
    training = working.loc[
        working["partition"].eq("development") & working["cv_fold"].ne(fold)
    ].copy()
    if training.empty:
        raise ValueError("paired data has no training rows outside the validation fold")
    training["cv_fold"] = training["cv_fold"].astype(int)
    result = training.sort_values("id").reset_index(drop=True)
    result.attrs["task4_training_provenance"] = TrainingPairsProvenance(
        validation_fold=fold,
        split_fingerprint=cv_assignment_digest(canonical_splits),
    )
    return result


def _validate_cache(
    cache: DevelopmentImageCache,
    *,
    source: str,
) -> None:
    manifest = cache.manifest
    if manifest.get("scope") != "development":
        raise ValueError(f"{source} cache scope must be development")
    if manifest.get("source") != source:
        raise ValueError(f"{source} cache source does not match")
    if manifest.get("contract") != _FIXED_CONTRACT.to_dict():
        raise ValueError(f"{source} cache contract must be the frozen 240x320 contract")
    ids = np.asarray(cache.ids)
    if (
        ids.dtype != np.int64
        or ids.ndim != 1
        or len(np.unique(ids)) != len(ids)
        or not np.array_equal(ids, np.sort(ids))
        or int(manifest.get("rows", -1)) != len(ids)
        or manifest.get("id_sha256") != id_set_digest(ids.tolist())
    ):
        raise ValueError(f"{source} cache IDs do not exactly match its manifest")
    if (
        cache.images.dtype != np.uint8
        or cache.images.shape != (len(ids), 320, 240, 3)
        or cache.content_bounds.dtype != np.int32
        or cache.content_bounds.shape != (len(ids), 4)
    ):
        raise ValueError(f"{source} cache arrays do not match the frozen contract")


def _validate_statistics(
    statistics: dict[str, object],
    *,
    source: str,
    cache: DevelopmentImageCache,
    validation_fold: int,
    split_fingerprint: str,
    training_ids: list[int],
) -> tuple[list[float], list[float]]:
    if statistics.get("source") != source:
        raise ValueError(f"{source} statistics source does not match")
    if statistics.get("contract") != _FIXED_CONTRACT.to_dict():
        raise ValueError(f"{source} statistics contract does not match")
    if statistics.get("validation_fold") != validation_fold:
        raise ValueError(f"{source} statistics validation fold does not match")
    if statistics.get("split_fingerprint") != split_fingerprint:
        raise ValueError(f"{source} statistics split fingerprint does not match")
    if statistics.get("training_id_sha256") != id_set_digest(training_ids):
        raise ValueError(f"{source} statistics training-ID digest does not match")
    if statistics.get("training_rows") != len(training_ids):
        raise ValueError(f"{source} statistics training row count does not match")
    if statistics.get("source_fingerprint") != cache.manifest.get("source_fingerprint"):
        raise ValueError(f"{source} statistics source fingerprint does not match its cache")
    mean = np.asarray(statistics.get("mean"), dtype=np.float64)
    std = np.asarray(statistics.get("std"), dtype=np.float64)
    if (
        mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError(f"{source} statistics mean/std are invalid")
    return mean.tolist(), std.tolist()


def _cached_image(cache: DevelopmentImageCache, index: int) -> PreprocessedImage:
    bounds = tuple(int(value) for value in cache.content_bounds[index])
    top, left, bottom, right = bounds
    if not (0 <= top < bottom <= 320 and 0 <= left < right <= 240):
        raise ValueError("cached content bounds must lie inside the frozen image")
    mask = np.zeros((320, 240), dtype=bool)
    mask[top:bottom, left:right] = True
    return PreprocessedImage(
        pixels=np.asarray(cache.images[index]),
        content_mask=mask,
        content_bounds=bounds,
    )


class CrossSourcePairDataset(Dataset[dict[str, object]]):
    """Return normalized teacher/V1 pairs from validated development caches."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        *,
        teacher_cache: DevelopmentImageCache,
        v1_cache: DevelopmentImageCache,
        teacher_statistics: dict[str, object],
        v1_statistics: dict[str, object],
        validation_fold: int,
        split_fingerprint: str,
        geometry_policy: GeometryPolicy | None = None,
    ) -> None:
        fold = _validate_validation_fold(validation_fold)
        provenance = pairs.attrs.get("task4_training_provenance")
        if not isinstance(provenance, TrainingPairsProvenance):
            raise ValueError("pair data requires canonical training provenance")
        if provenance.validation_fold != fold:
            raise ValueError("pair provenance validation fold does not match")
        if provenance.split_fingerprint != split_fingerprint:
            raise ValueError("pair provenance split fingerprint does not match")
        working = _validated_variant_index(pairs)
        if not working["partition"].eq("development").all():
            raise ValueError("pair dataset accepts development rows only")
        if working["cv_fold"].eq(fold).any():
            raise ValueError("pair dataset cannot contain validation-fold rows")
        working = working.sort_values("id").reset_index(drop=True)
        _validate_cache(teacher_cache, source="teacher")
        _validate_cache(v1_cache, source="v1")
        if not np.array_equal(teacher_cache.ids, v1_cache.ids):
            raise ValueError("teacher and V1 cache IDs must match exactly")
        training_ids = working["id"].astype(int).tolist()
        cache_ids = set(int(value) for value in teacher_cache.ids)
        if not set(training_ids).issubset(cache_ids):
            raise ValueError("training pair IDs must all exist in both caches")
        self._teacher_mean, self._teacher_std = _validate_statistics(
            teacher_statistics,
            source="teacher",
            cache=teacher_cache,
            validation_fold=fold,
            split_fingerprint=split_fingerprint,
            training_ids=training_ids,
        )
        self._v1_mean, self._v1_std = _validate_statistics(
            v1_statistics,
            source="v1",
            cache=v1_cache,
            validation_fold=fold,
            split_fingerprint=split_fingerprint,
            training_ids=training_ids,
        )
        self.pairs = working
        self.teacher_cache = teacher_cache
        self.v1_cache = v1_cache
        self.validation_fold = fold
        self.split_fingerprint = split_fingerprint
        self.geometry_policy = geometry_policy
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._cache_positions = {
            int(product_id): index for index, product_id in enumerate(teacher_cache.ids)
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch used by geometry augmentation."""
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._shared_epoch.fill_(int(epoch))

    @staticmethod
    def _tensor(
        image: PreprocessedImage,
        *,
        mean: list[float],
        std: list[float],
    ) -> torch.Tensor:
        normalized = normalize_for_model(image, mean=mean, std=std)
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))

    @staticmethod
    def _content_mask(image: PreprocessedImage) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(image.content_mask, dtype=bool))

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.pairs.iloc[index]
        product_id = int(row["id"])
        cache_index = self._cache_positions[product_id]
        teacher = _cached_image(self.teacher_cache, cache_index)
        v1 = _cached_image(self.v1_cache, cache_index)
        augmented_source = "none"
        if self.geometry_policy is not None:
            geometry = sample_geometry(
                product_id=product_id,
                epoch=int(self._shared_epoch.item()),
                policy=self.geometry_policy,
            )
            augmented_source = geometry.augmented_source
            if augmented_source == "teacher":
                teacher = apply_geometry(
                    teacher.pixels,
                    teacher.content_bounds,
                    sample=geometry,
                    policy=self.geometry_policy,
                )
            else:
                v1 = apply_geometry(
                    v1.pixels,
                    v1.content_bounds,
                    sample=geometry,
                    policy=self.geometry_policy,
                )
        return {
            "id": product_id,
            "product_family_group": str(row["product_family_group"]),
            "duplicate_group": str(row["duplicate_group"]),
            "sha256": str(row["sha256"]),
            "teacher": self._tensor(
                teacher,
                mean=self._teacher_mean,
                std=self._teacher_std,
            ),
            "v1": self._tensor(v1, mean=self._v1_mean, std=self._v1_std),
            "teacher_content_mask": self._content_mask(teacher),
            "v1_content_mask": self._content_mask(v1),
            "augmented_source": augmented_source,
        }


class FamilyBatchSampler(Sampler[list[int]]):
    """Build deterministic 64-ID batches with sixteen valid family pairs."""

    def __init__(self, pairs: pd.DataFrame, *, seed: int = RANDOM_SEED) -> None:
        required = {"id", "product_family_group", "sha256", "duplicate_group"}
        if missing := required.difference(pairs.columns):
            raise ValueError(f"family sampler data is missing columns: {sorted(missing)}")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("family sampler seed must be an integer")
        if len(pairs) < 64:
            raise ValueError("family sampler requires at least 64 unique IDs")
        frame = pairs.copy()
        ids = pd.to_numeric(frame["id"], errors="coerce")
        if (
            ids.isna().any()
            or not ids.mod(1).eq(0).all()
            or ids.astype(np.int64).duplicated().any()
        ):
            raise ValueError("family sampler IDs must be unique integers")
        frame["id"] = ids.astype(np.int64)
        frame = frame.sort_values("id").reset_index(drop=True)
        _nonblank(frame, required - {"id"})
        valid_pairs: dict[str, list[tuple[int, int]]] = {}
        for family, group in frame.groupby("product_family_group", sort=True):
            candidates = [
                (left, right)
                for left, right in itertools.combinations(group.index.tolist(), 2)
                if frame.at[left, "sha256"] != frame.at[right, "sha256"]
                and frame.at[left, "duplicate_group"] != frame.at[right, "duplicate_group"]
            ]
            if candidates:
                valid_pairs[str(family)] = candidates
        if len(valid_pairs) < 16:
            raise ValueError("family sampler requires at least 16 valid families")
        self.pairs = frame
        self.seed = int(seed)
        self._valid_pairs = valid_pairs
        self._epoch = 0
        self._cached_epoch: int | None = None
        self._cached_schedule: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic sampling epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.pairs) / 64)

    def _random(self) -> np.random.Generator:
        payload = f"task4-family-batches-v2\0{self.seed}\0{self._epoch}".encode("ascii")
        digest = hashlib.sha256(payload).digest()
        return np.random.default_rng(int.from_bytes(digest[:16], "big"))

    @staticmethod
    def _coverage_sequence(
        candidates: list[tuple[int, int]],
        random: np.random.Generator,
    ) -> list[tuple[int, int]]:
        candidate_order = random.permutation(len(candidates)).astype(int).tolist()
        candidate_rank = {
            candidate_index: rank
            for rank, candidate_index in enumerate(candidate_order)
        }
        adjacency: dict[int, list[int]] = defaultdict(list)
        for candidate_index, pair in enumerate(candidates):
            adjacency[pair[0]].append(candidate_index)
            adjacency[pair[1]].append(candidate_index)
        id_order = random.permutation(sorted(adjacency)).astype(int).tolist()
        unseen = set(adjacency)
        sequence: list[tuple[int, int]] = []
        while unseen:
            product_index = next(index for index in id_order if index in unseen)
            candidate_index = min(
                adjacency[product_index],
                key=lambda index: (
                    -sum(member in unseen for member in candidates[index]),
                    candidate_rank[index],
                ),
            )
            pair = candidates[candidate_index]
            sequence.append(pair)
            unseen.difference_update(pair)
        return sequence

    def _family_schedule(self, random: np.random.Generator) -> list[list[int]]:
        families = sorted(self._valid_pairs)
        family_order = random.permutation(len(families)).astype(int).tolist()
        family_rank = {families[index]: rank for rank, index in enumerate(family_order)}
        coverage = {
            family: self._coverage_sequence(self._valid_pairs[family], random)
            for family in families
        }
        coverage_positions = {family: 0 for family in families}
        family_uses = {family: 0 for family in families}
        batches: list[list[int]] = []
        for _ in range(len(self)):
            def family_key(family: str) -> tuple[int, int, int, int]:
                position = coverage_positions[family]
                sequence = coverage[family]
                if position < len(sequence):
                    pair = sequence[position]
                    unseen_gain = 2 if position == 0 else 1
                    if position > 0:
                        prior = {
                            member
                            for earlier in sequence[:position]
                            for member in earlier
                        }
                        unseen_gain = sum(member not in prior for member in pair)
                    return (0, -unseen_gain, family_uses[family], family_rank[family])
                return (1, 0, family_uses[family], family_rank[family])

            selected = sorted(families, key=family_key)[:16]
            batch: list[int] = []
            for family in selected:
                position = coverage_positions[family]
                if position < len(coverage[family]):
                    pair = coverage[family][position]
                    coverage_positions[family] += 1
                else:
                    candidates = self._valid_pairs[family]
                    pair = candidates[
                        (family_uses[family] + family_rank[family]) % len(candidates)
                    ]
                if int(random.integers(0, 2)):
                    pair = pair[::-1]
                batch.extend(pair)
                family_uses[family] += 1
            batches.append(batch)
        return batches

    def _other_schedule(
        self,
        family_batches: list[list[int]],
        random: np.random.Generator,
    ) -> list[list[int]]:
        family_ids = {index for batch in family_batches for index in batch}
        permutation = random.permutation(len(self.pairs)).astype(int).tolist()
        priority = [
            *[index for index in permutation if index not in family_ids],
            *[index for index in permutation if index in family_ids],
        ]
        other_batches: list[list[int]] = [[] for _ in family_batches]
        batch_rank_order = random.permutation(len(family_batches)).astype(int).tolist()
        batch_rank = {batch: rank for rank, batch in enumerate(batch_rank_order)}
        for index in priority:
            candidates = [
                batch_index
                for batch_index, family_batch in enumerate(family_batches)
                if len(other_batches[batch_index]) < 32 and index not in family_batch
            ]
            if not candidates:
                continue
            chosen = min(
                candidates,
                key=lambda batch_index: (
                    len(other_batches[batch_index]),
                    batch_rank[batch_index],
                ),
            )
            other_batches[chosen].append(index)
            if all(len(batch) == 32 for batch in other_batches):
                break
        for batch_index, family_batch in enumerate(family_batches):
            if len(other_batches[batch_index]) == 32:
                continue
            reusable = [
                index
                for index in permutation
                if index not in family_batch and index not in other_batches[batch_index]
            ]
            needed = 32 - len(other_batches[batch_index])
            other_batches[batch_index].extend(reusable[:needed])
        if not all(len(batch) == 32 for batch in other_batches):
            raise RuntimeError("could not construct complete family-aware epoch schedule")
        return other_batches

    def _build_schedule(self) -> list[list[int]]:
        random = self._random()
        family_batches = self._family_schedule(random)
        other_batches = self._other_schedule(family_batches, random)
        return [
            [*family_batch, *other_batch]
            for family_batch, other_batch in zip(
                family_batches,
                other_batches,
                strict=True,
            )
        ]

    def __iter__(self) -> Iterator[list[int]]:
        if self._cached_epoch != self._epoch or self._cached_schedule is None:
            self._cached_schedule = self._build_schedule()
            self._cached_epoch = self._epoch
        yield from (batch.copy() for batch in self._cached_schedule)
