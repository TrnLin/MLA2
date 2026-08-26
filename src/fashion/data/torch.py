"""Fold-fitted image statistics and picklable PyTorch tensor transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fashion.config import LABEL_MAPS_JSON, RANDOM_SEED, ROOT, SPLITS_CSV
from fashion.data.dataset import (
    FashionDataset,
    get_cv_split,
    get_samples,
    load_label_maps,
    load_splits,
)
from fashion.data.images import StreamingStats, resolve_image_size, transform_image_with_mask
from fashion.train.artifacts import canonical_sha256
from fashion.train.reproducibility import make_torch_generator, seed_worker

AugmentationPolicy = Literal["none", "a0", "a1"]


@dataclass(frozen=True)
class FoldImageStats:
    """Normalization fitted only on the training side of one canonical fold."""

    validation_fold: int
    image_size: tuple[int, int]
    image_count: int
    content_pixel_count: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    training_id_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageTransformSpec:
    """Exact geometry and mild augmentation settings for a registered run."""

    image_size: tuple[int, int]
    augmentation: AugmentationPolicy
    horizontal_flip_probability: float = 0.5
    affine_degrees: float = 8.0
    affine_translate: tuple[float, float] = (0.05, 0.05)
    affine_scale: tuple[float, float] = (0.95, 1.05)
    brightness: float = 0.10
    contrast: float = 0.10
    saturation: float = 0.08
    hue: float = 0.02

    @property
    def transform_id(self) -> str:
        return f"{self.augmentation}-{canonical_sha256(asdict(self))[:16]}"


class TensorImageTransform:
    """Load one image, apply optional training augmentation, and return CHW float32."""

    def __init__(
        self,
        *,
        stats: FoldImageStats,
        spec: ImageTransformSpec,
    ) -> None:
        if spec.image_size != stats.image_size:
            raise ValueError("transform and fitted-stat image sizes must match")
        if any(value <= 0 for value in stats.std):
            raise ValueError("fitted channel std values must be positive")
        self.stats = stats
        self.spec = spec
        neutral_fill = tuple(round(value * 255) for value in stats.mean)
        operations: list[Any] = []
        if spec.augmentation in {"a0", "a1"}:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(spec.horizontal_flip_probability),
                    transforms.RandomAffine(
                        degrees=spec.affine_degrees,
                        translate=spec.affine_translate,
                        scale=spec.affine_scale,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=neutral_fill,
                    ),
                ]
            )
        if spec.augmentation == "a1":
            operations.append(
                transforms.ColorJitter(
                    brightness=spec.brightness,
                    contrast=spec.contrast,
                    saturation=spec.saturation,
                    hue=spec.hue,
                )
            )
        self.augmentation = transforms.Compose(operations)

    def __call__(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = self.augmentation(image)
            array, _ = transform_image_with_mask(
                image,
                image_size=self.stats.image_size,
                normalize_range=True,
                mean=self.stats.mean,
                std=self.stats.std,
            )
        return torch.from_numpy(np.moveaxis(array, -1, 0).copy())


class EncodedClassificationDataset(Dataset[dict[str, Any]]):
    """Return the minimal image, integer target, and stable ID training contract."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        transform: TensorImageTransform,
        target: str,
        label_to_index: dict[str, int],
        root: str | Path = ROOT,
    ) -> None:
        unknown = sorted(set(frame[target].astype(str)) - set(label_to_index))
        if unknown:
            raise ValueError(f"{target} contains labels absent from the canonical map: {unknown}")
        self.base = FashionDataset(
            frame,
            transform=transform,
            root=root,
            targets=(target,),
        )
        self.target = target
        self.label_to_index = dict(label_to_index)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        label = str(sample[self.target])
        return {
            "id": sample["id"],
            "image": sample["image"],
            "target": self.label_to_index[label],
        }


@dataclass(frozen=True)
class TaskLoaders:
    """One canonical CV round and the fitted preprocessing used by both sides."""

    train: DataLoader[Any]
    validation: DataLoader[Any]
    stats: FoldImageStats
    labels: tuple[str, ...]
    label_to_index: dict[str, int]
    training_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    train_transform_id: str
    validation_transform_id: str

    def audit(self) -> dict[str, Any]:
        """Return compact structural evidence without reading target outcomes."""
        return {
            "validation_fold": self.stats.validation_fold,
            "training_products": len(self.training_ids),
            "validation_products": len(self.validation_ids),
            "id_overlap": len(set(self.training_ids) & set(self.validation_ids)),
            "labels": list(self.labels),
            "train_transform_id": self.train_transform_id,
            "validation_transform_id": self.validation_transform_id,
            "stats": self.stats.to_dict(),
        }


def _safe_image_path(root: Path, relative_path: Any) -> Path:
    raw_path = Path(str(relative_path))
    if raw_path.is_absolute():
        raise ValueError(f"manifest image paths must be relative: {raw_path}")
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest image path escapes project root: {raw_path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest image does not exist: {resolved}")
    return resolved


def fit_fold_stats(
    training_frame: pd.DataFrame,
    *,
    validation_fold: int,
    image_size: int | tuple[int, int],
    root: str | Path = ROOT,
) -> FoldImageStats:
    """Fit RGB mean/std on training-fold content pixels, never letterbox padding."""
    required = {"id", "path", "partition", "cv_fold"}
    missing = sorted(required - set(training_frame.columns))
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    if training_frame.empty:
        raise ValueError("training frame must not be empty")
    if training_frame["id"].duplicated().any():
        raise ValueError("training frame IDs must be unique")
    if set(training_frame["partition"].astype(str)) != {"development"}:
        raise ValueError("fold statistics may use development rows only")
    folds = pd.to_numeric(training_frame["cv_fold"], errors="raise").astype(int)
    if folds.eq(validation_fold).any():
        raise ValueError("validation-fold rows cannot fit image statistics")

    resolved_size = resolve_image_size(image_size)
    project_root = Path(root).resolve()
    streaming = StreamingStats(channels=3)
    ordered = training_frame.sort_values("id", kind="stable")
    for row in ordered.itertuples(index=False):
        path = _safe_image_path(project_root, getattr(row, "path"))
        try:
            with Image.open(path) as image:
                array, content_mask = transform_image_with_mask(
                    image,
                    image_size=resolved_size,
                    normalize_range=True,
                )
        except Exception as error:
            raise RuntimeError(f"failed to fit image statistics for ID {row.id}: {path}") from error
        streaming.update(array, content_mask=content_mask)

    mean = tuple(float(value) for value in streaming.mean)
    std = tuple(float(value) for value in streaming.std)
    if any(value <= 0 for value in std):
        raise ValueError("training-fold image statistics contain a non-positive channel std")
    training_ids = sorted(int(value) for value in ordered["id"])
    return FoldImageStats(
        validation_fold=validation_fold,
        image_size=resolved_size,
        image_count=len(ordered),
        content_pixel_count=streaming.total_pixels,
        mean=mean,
        std=std,
        training_id_sha256=canonical_sha256(training_ids),
    )


def build_image_transform(
    stats: FoldImageStats,
    *,
    training: bool,
    augmentation: AugmentationPolicy | None = None,
) -> TensorImageTransform:
    """Build an evaluation transform or one of the predeclared A0/A1 training policies."""
    policy: AugmentationPolicy = augmentation or ("a0" if training else "none")
    if policy not in {"none", "a0", "a1"}:
        raise ValueError(f"unknown augmentation policy: {policy}")
    if not training and policy != "none":
        raise ValueError("validation and inference transforms cannot use random augmentation")
    return TensorImageTransform(
        stats=stats,
        spec=ImageTransformSpec(image_size=stats.image_size, augmentation=policy),
    )


def _canonical_label_contract(
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


def build_task_loaders(
    *,
    validation_fold: int,
    image_size: int | tuple[int, int],
    batch_size: int,
    target: str = "season",
    augmentation: AugmentationPolicy = "a0",
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
    validation_batch_size: int | None = None,
    pin_memory: bool | None = None,
    root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    stats: FoldImageStats | None = None,
) -> TaskLoaders:
    """Build one leakage-safe fold exclusively from the canonical split and label map."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if validation_batch_size is None:
        validation_batch_size = batch_size * 2
    if validation_batch_size < 1:
        raise ValueError("validation_batch_size must be positive")

    splits = load_splits(splits_path)
    training_frame, validation_frame = get_cv_split(splits, validation_fold)
    training_frame = get_samples(training_frame, target=target).reset_index(drop=True)
    validation_frame = get_samples(validation_frame, target=target).reset_index(drop=True)
    if training_frame.empty or validation_frame.empty:
        raise ValueError(
            f"fold {validation_fold} has no valid {target} training or validation rows"
        )
    if not training_frame["partition"].eq("development").all():
        raise ValueError("training loader received a protected partition")
    if not validation_frame["partition"].eq("development").all():
        raise ValueError("validation loader received a protected partition")
    training_ids = tuple(int(value) for value in training_frame["id"])
    validation_ids = tuple(int(value) for value in validation_frame["id"])
    if set(training_ids) & set(validation_ids):
        raise ValueError("training and validation IDs overlap")

    labels, label_to_index = _canonical_label_contract(
        target,
        label_map_path=label_map_path,
    )
    unknown = (
        set(training_frame[target].astype(str))
        | set(validation_frame[target].astype(str))
    ) - set(labels)
    if unknown:
        raise ValueError(f"fold contains {target} labels absent from the canonical map: {unknown}")

    resolved_size = resolve_image_size(image_size)
    fitted_stats = stats or fit_fold_stats(
        training_frame,
        validation_fold=validation_fold,
        image_size=resolved_size,
        root=root,
    )
    if fitted_stats.validation_fold != validation_fold:
        raise ValueError("supplied normalization stats were fitted for a different fold")
    if fitted_stats.image_size != resolved_size:
        raise ValueError("supplied normalization stats were fitted for a different image size")
    expected_training_hash = canonical_sha256(sorted(training_ids))
    if fitted_stats.training_id_sha256 != expected_training_hash:
        raise ValueError("supplied normalization stats were fitted on different training IDs")

    train_transform = build_image_transform(
        fitted_stats,
        training=True,
        augmentation=augmentation,
    )
    validation_transform = build_image_transform(fitted_stats, training=False)
    train_dataset = EncodedClassificationDataset(
        training_frame,
        transform=train_transform,
        target=target,
        label_to_index=label_to_index,
        root=root,
    )
    validation_dataset = EncodedClassificationDataset(
        validation_frame,
        transform=validation_transform,
        target=target,
        label_to_index=label_to_index,
        root=root,
    )
    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    worker_options: dict[str, Any] = {}
    if num_workers:
        worker_options = {"persistent_workers": True, "prefetch_factor": 2}
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
        drop_last=False,
        **worker_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 1),
        drop_last=False,
        **worker_options,
    )
    return TaskLoaders(
        train=train_loader,
        validation=validation_loader,
        stats=fitted_stats,
        labels=labels,
        label_to_index=label_to_index,
        training_ids=training_ids,
        validation_ids=validation_ids,
        train_transform_id=train_transform.spec.transform_id,
        validation_transform_id=validation_transform.spec.transform_id,
    )
