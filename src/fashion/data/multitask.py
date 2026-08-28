"""Leakage-safe loaders for a main target and a masked auxiliary target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fashion.config import LABEL_MAPS_JSON, RANDOM_SEED, ROOT, SPLITS_CSV
from fashion.data.dataset import (
    FashionDataset,
    get_cv_split,
    get_samples,
    load_label_maps,
    load_splits,
)
from fashion.data.torch import (
    AugmentationPolicy,
    FoldImageStats,
    TaskLoaders,
    TensorImageTransform,
    build_image_transform,
    build_task_loaders,
)
from fashion.train.artifacts import canonical_sha256
from fashion.train.reproducibility import make_torch_generator, seed_worker

AUXILIARY_IGNORE_INDEX = -100


def _canonical_labels(
    target: str,
    *,
    label_map_path: str | Path,
) -> tuple[tuple[str, ...], dict[str, int]]:
    mappings = load_label_maps(label_map_path)
    if target not in mappings:
        raise KeyError(f"target is absent from canonical label maps: {target}")
    mapping = mappings[target]
    labels = tuple(str(label) for label in mapping["classes"])
    label_to_index = {
        str(label): int(index) for label, index in dict(mapping["label_to_index"]).items()
    }
    expected = {label: index for index, label in enumerate(labels)}
    if label_to_index != expected or int(mapping["num_classes"]) != len(labels):
        raise ValueError(f"canonical {target} label order and indices disagree")
    return labels, label_to_index


class MaskedAuxiliaryDataset(Dataset[dict[str, Any]]):
    """Keep every main-target row and mask missing auxiliary labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        transform: TensorImageTransform,
        main_target: str,
        main_label_to_index: dict[str, int],
        auxiliary_target: str,
        auxiliary_label_to_index: dict[str, int],
        root: str | Path = ROOT,
    ) -> None:
        auxiliary_mask_column = f"has_{auxiliary_target}_label"
        required = {"id", main_target, auxiliary_target, auxiliary_mask_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"multi-task frame is missing columns: {missing}")

        unknown_main = sorted(set(frame[main_target].astype(str)) - set(main_label_to_index))
        if unknown_main:
            raise ValueError(
                f"{main_target} contains labels absent from the canonical map: {unknown_main}"
            )
        auxiliary_mask = frame[auxiliary_mask_column].astype(bool)
        observed_auxiliary = set(frame.loc[auxiliary_mask, auxiliary_target].astype(str))
        unknown_auxiliary = sorted(observed_auxiliary - set(auxiliary_label_to_index))
        if unknown_auxiliary:
            raise ValueError(
                f"{auxiliary_target} contains labels absent from the canonical map: "
                f"{unknown_auxiliary}"
            )

        self.base = FashionDataset(
            frame,
            transform=transform,
            root=root,
            targets=(main_target, auxiliary_target),
        )
        self.main_target = main_target
        self.main_label_to_index = dict(main_label_to_index)
        self.auxiliary_target = auxiliary_target
        self.auxiliary_label_to_index = dict(auxiliary_label_to_index)
        self.auxiliary_mask_column = auxiliary_mask_column

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        auxiliary_mask = bool(sample[self.auxiliary_mask_column])
        auxiliary_index = AUXILIARY_IGNORE_INDEX
        if auxiliary_mask:
            auxiliary_index = self.auxiliary_label_to_index[
                str(sample[self.auxiliary_target])
            ]
        return {
            "id": sample["id"],
            "image": sample["image"],
            "target": self.main_label_to_index[str(sample[self.main_target])],
            "auxiliary_target": auxiliary_index,
            "auxiliary_mask": auxiliary_mask,
        }


@dataclass(frozen=True)
class MultiTaskLoaders:
    """One canonical fold with a masked training-only auxiliary label."""

    train: DataLoader[Any]
    validation: DataLoader[Any]
    stats: FoldImageStats
    labels: tuple[str, ...]
    label_to_index: dict[str, int]
    auxiliary_labels: tuple[str, ...]
    auxiliary_label_to_index: dict[str, int]
    training_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    train_transform_id: str
    validation_transform_id: str
    auxiliary_target: str
    auxiliary_training_count: int
    auxiliary_validation_count: int
    auxiliary_training_id_sha256: str

    def audit(self) -> dict[str, Any]:
        """Return the fold and auxiliary-mask evidence used by I2."""
        return {
            "validation_fold": self.stats.validation_fold,
            "training_products": len(self.training_ids),
            "validation_products": len(self.validation_ids),
            "id_overlap": len(set(self.training_ids) & set(self.validation_ids)),
            "labels": list(self.labels),
            "auxiliary_target": self.auxiliary_target,
            "auxiliary_labels": list(self.auxiliary_labels),
            "auxiliary_training_products": self.auxiliary_training_count,
            "auxiliary_validation_products": self.auxiliary_validation_count,
            "auxiliary_training_id_sha256": self.auxiliary_training_id_sha256,
            "train_transform_id": self.train_transform_id,
            "validation_transform_id": self.validation_transform_id,
            "stats": self.stats.to_dict(),
        }


def _build_loader(
    dataset: MaskedAuxiliaryDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    worker_options: dict[str, Any] = {}
    if num_workers:
        worker_options = {"persistent_workers": True, "prefetch_factor": 2}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
        drop_last=False,
        **worker_options,
    )


def build_multitask_loaders(
    *,
    validation_fold: int,
    image_size: int | tuple[int, int],
    batch_size: int,
    main_target: str = "season",
    auxiliary_target: str = "articleType",
    augmentation: AugmentationPolicy = "a0",
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
    validation_batch_size: int | None = None,
    pin_memory: bool | None = None,
    root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    stats: FoldImageStats | None = None,
    stats_cache_directory: str | Path | None = None,
) -> MultiTaskLoaders:
    """Build I2 loaders without filtering rows that lack the auxiliary label."""
    base: TaskLoaders = build_task_loaders(
        validation_fold=validation_fold,
        image_size=image_size,
        batch_size=batch_size,
        target=main_target,
        augmentation=augmentation,
        seed=seed,
        num_workers=num_workers,
        validation_batch_size=validation_batch_size,
        pin_memory=pin_memory,
        root=root,
        splits_path=splits_path,
        label_map_path=label_map_path,
        stats=stats,
        stats_cache_directory=stats_cache_directory,
    )
    resolved_validation_batch_size = validation_batch_size or batch_size * 2
    splits = load_splits(splits_path)
    training_frame, validation_frame = get_cv_split(splits, validation_fold)
    training_frame = get_samples(training_frame, target=main_target).reset_index(drop=True)
    validation_frame = get_samples(validation_frame, target=main_target).reset_index(drop=True)
    training_ids = tuple(int(value) for value in training_frame["id"])
    validation_ids = tuple(int(value) for value in validation_frame["id"])
    if training_ids != base.training_ids or validation_ids != base.validation_ids:
        raise ValueError("multi-task rows drifted from the canonical main-target loaders")

    auxiliary_labels, auxiliary_label_to_index = _canonical_labels(
        auxiliary_target,
        label_map_path=label_map_path,
    )
    train_transform = build_image_transform(
        base.stats,
        training=True,
        augmentation=augmentation,
    )
    validation_transform = build_image_transform(base.stats, training=False)
    train_dataset = MaskedAuxiliaryDataset(
        training_frame,
        transform=train_transform,
        main_target=main_target,
        main_label_to_index=base.label_to_index,
        auxiliary_target=auxiliary_target,
        auxiliary_label_to_index=auxiliary_label_to_index,
        root=root,
    )
    validation_dataset = MaskedAuxiliaryDataset(
        validation_frame,
        transform=validation_transform,
        main_target=main_target,
        main_label_to_index=base.label_to_index,
        auxiliary_target=auxiliary_target,
        auxiliary_label_to_index=auxiliary_label_to_index,
        root=root,
    )
    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    auxiliary_mask_column = f"has_{auxiliary_target}_label"
    training_auxiliary_ids = sorted(
        int(value)
        for value in training_frame.loc[
            training_frame[auxiliary_mask_column].astype(bool), "id"
        ]
    )
    validation_auxiliary_count = int(
        validation_frame[auxiliary_mask_column].astype(bool).sum()
    )
    return MultiTaskLoaders(
        train=_build_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
        validation=_build_loader(
            validation_dataset,
            batch_size=resolved_validation_batch_size,
            shuffle=False,
            seed=seed + 1,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
        stats=base.stats,
        labels=base.labels,
        label_to_index=base.label_to_index,
        auxiliary_labels=auxiliary_labels,
        auxiliary_label_to_index=auxiliary_label_to_index,
        training_ids=training_ids,
        validation_ids=validation_ids,
        train_transform_id=train_transform.spec.transform_id,
        validation_transform_id=validation_transform.spec.transform_id,
        auxiliary_target=auxiliary_target,
        auxiliary_training_count=len(training_auxiliary_ids),
        auxiliary_validation_count=validation_auxiliary_count,
        auxiliary_training_id_sha256=canonical_sha256(training_auxiliary_ids),
    )
