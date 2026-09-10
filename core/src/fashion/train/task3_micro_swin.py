"""Scratch micro-Swin screen for Task 3 gender and usage.

This is a clean-slate GPU comparison, not a child of the E1--E10 CNN chain.
It uses linear patch projection and shifted-window self-attention only; no
pretrained weights or convolution tokenizer are used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from fashion.config import LABEL_MAPS_JSON, RANDOM_SEED, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data import load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.images import StreamingStats, transform_image, transform_image_with_mask
from fashion.data.task3_clean_slate_eda import foreground_views
from fashion.train.augmentation import apply_training_augmentation
from fashion.train.data import (
    CORE_CORRUPTIONS,
    Task3ImageDataset,
    _load_corrupted,
    task3_target_frames,
)
from fashion.train.metrics import classification_metrics
from fashion.train.registry import RunRegistry
from fashion.train.task3_baseline import (
    _class_spec,
    _measure_latency,
    _prediction_frame,
    _worker_seed,
    runtime_environment,
    set_reproducible_seed,
    validate_verified_colab_runtime,
)
from fashion.train.task3_clean_slate import _aggregate_screen
from fashion.train.task3_experiments import effective_number_class_weights

SCREEN_FOLDS = (0, 4)
ARTIFACT_ROOT = "experiments/task3_clean_slate_micro_swin_screen_2"
MODEL_FAMILY = "scratch_pure_patch_micro_swin"
RUN_PREFIX = "t3_cs2_micro_swin"
EXPERIMENT_IDS = {
    "gender": "t3_clean_slate_s2_gender_micro_swin",
    "usage": "t3_clean_slate_s2_usage_micro_swin",
}
HYPOTHESIS_IDS = {
    "gender": "patch_attention_reduces_gender_shortcut_memorisation",
    "usage": "patch_attention_captures_usage_parts_and_context",
}
INPUT_VIEWS = {"gender": "foreground_masked", "usage": "full"}
GPU_MEMORY_LIMIT_BYTES = 7 * 1024**3
L4_SECONDS_PER_FOLD_LIMIT = 60 * 60


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _log(message: str) -> None:
    print(f"[task3-micro-swin] {message}", flush=True)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MicroSwinConfig:
    """Frozen two-fold micro-Swin screening recipe."""

    target: Literal["gender", "usage"]
    image_height: int = 80
    image_width: int = 60
    patch_size: int = 4
    dimensions: tuple[int, int, int] = (64, 128, 256)
    depths: tuple[int, int, int] = (2, 2, 2)
    heads: tuple[int, int, int] = (2, 4, 8)
    windows: tuple[tuple[int, int], ...] = ((5, 5), (5, 4), (5, 4))
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    drop_path: float = 0.10
    batch_size: int = 128
    epochs: int = 60
    learning_rate: float = 0.0003
    minimum_learning_rate: float = 0.00001
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0
    num_workers: int = 2
    seed: int = RANDOM_SEED
    mixed_precision: bool = True
    augmentation: str = "none"
    scratch: bool = True
    submission_eligible: bool = True

    def __post_init__(self) -> None:
        if self.target not in ("gender", "usage"):
            raise ValueError(f"unsupported Task 3 target: {self.target}")
        if (self.image_height, self.image_width, self.patch_size) != (80, 60, 4):
            raise ValueError("micro-Swin uses the fixed 80x60 teacher image and 4x4 patches")
        if not (
            len(self.dimensions) == len(self.depths) == len(self.heads) == len(self.windows) == 3
        ):
            raise ValueError("micro-Swin requires exactly three stages")
        if any(dim % heads for dim, heads in zip(self.dimensions, self.heads, strict=True)):
            raise ValueError("every micro-Swin width must be divisible by its head count")
        if self.epochs <= self.warmup_epochs or self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("invalid micro-Swin training schedule")
        if not self.scratch or not self.submission_eligible:
            raise ValueError("the micro-Swin screen must remain scratch-trained and eligible")

    @property
    def num_classes(self) -> int:
        return 5 if self.target == "gender" else 9

    @property
    def input_view(self) -> str:
        return INPUT_VIEWS[self.target]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = list(self.dimensions)
        payload["depths"] = list(self.depths)
        payload["heads"] = list(self.heads)
        payload["windows"] = [list(window) for window in self.windows]
        payload["num_classes"] = self.num_classes
        payload["input_view"] = self.input_view
        payload["model_family"] = MODEL_FAMILY
        payload["pretrained_weights"] = False
        return payload


class _DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return values
        keep = 1.0 - self.probability
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        mask = values.new_empty(shape).bernoulli_(keep)
        return values * mask / keep


def _partition_windows(values: torch.Tensor, window: tuple[int, int]) -> torch.Tensor:
    batch, height, width, channels = values.shape
    window_height, window_width = window
    if height % window_height or width % window_width:
        raise ValueError("feature grid is not divisible by the declared attention window")
    return (
        values.view(
            batch,
            height // window_height,
            window_height,
            width // window_width,
            window_width,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_height * window_width, channels)
    )


def _reverse_windows(
    windows: torch.Tensor,
    window: tuple[int, int],
    resolution: tuple[int, int],
    batch: int,
) -> torch.Tensor:
    height, width = resolution
    window_height, window_width = window
    return (
        windows.view(
            batch,
            height // window_height,
            width // window_width,
            window_height,
            window_width,
            -1,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, -1)
    )


def _shift_attention_mask(
    resolution: tuple[int, int],
    window: tuple[int, int],
    shift: tuple[int, int],
) -> torch.Tensor | None:
    if shift == (0, 0):
        return None
    height, width = resolution
    window_height, window_width = window
    shift_height, shift_width = shift
    region = torch.zeros((1, height, width, 1))
    height_slices = (
        slice(0, -window_height),
        slice(-window_height, -shift_height),
        slice(-shift_height, None),
    )
    width_slices = (
        slice(0, -window_width),
        slice(-window_width, -shift_width),
        slice(-shift_width, None),
    )
    count = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            region[:, height_slice, width_slice, :] = count
            count += 1
    windows = _partition_windows(region, window).squeeze(-1)
    difference = windows.unsqueeze(1) - windows.unsqueeze(2)
    return difference.masked_fill(difference != 0, -100.0).masked_fill(difference == 0, 0.0)


class _WindowAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, window: tuple[int, int]) -> None:
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.window = window
        self.head_dimension = dimension // heads
        self.scale = self.head_dimension**-0.5
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.projection = nn.Linear(dimension, dimension)
        window_height, window_width = window
        table_rows = (2 * window_height - 1) * (2 * window_width - 1)
        self.relative_position_bias = nn.Parameter(torch.zeros(table_rows, heads))

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_height),
                torch.arange(window_width),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_height - 1
        relative[:, :, 1] += window_width - 1
        relative[:, :, 0] *= 2 * window_width - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)

    def forward(self, values: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        batch_windows, tokens, channels = values.shape
        qkv = (
            self.qkv(values)
            .reshape(batch_windows, tokens, 3, self.heads, self.head_dimension)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attention = ((query * self.scale) @ key.transpose(-2, -1)).float()
        relative_bias = self.relative_position_bias[
            self.relative_position_index.reshape(-1)
        ].reshape(tokens, tokens, self.heads)
        attention = attention + relative_bias.float().permute(2, 0, 1).unsqueeze(0)
        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            attention = attention.view(
                batch_windows // window_count,
                window_count,
                self.heads,
                tokens,
                tokens,
            )
            attention = attention + attention_mask.float().unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.heads, tokens, tokens)
        attention = attention.softmax(dim=-1).to(value.dtype)
        output = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection(output)


class _MicroSwinBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        resolution: tuple[int, int],
        heads: int,
        window: tuple[int, int],
        shift: tuple[int, int],
        *,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        if any(size % window_size for size, window_size in zip(resolution, window, strict=True)):
            raise ValueError("micro-Swin stage resolution must divide into whole windows")
        if any(value < 0 or value >= size for value, size in zip(shift, window, strict=True)):
            raise ValueError("invalid shifted-window offset")
        self.dimension = dimension
        self.resolution = resolution
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = _WindowAttention(dimension, heads, window)
        self.drop_path = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dimension)
        hidden = round(dimension * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "attention_mask",
            _shift_attention_mask(resolution, window, shift),
            persistent=False,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = values.shape
        height, width = self.resolution
        if tokens != height * width or channels != self.dimension:
            raise ValueError("micro-Swin block received an unexpected token grid")
        shortcut = values
        grid = self.norm1(values).view(batch, height, width, channels)
        if self.shift != (0, 0):
            grid = torch.roll(grid, shifts=(-self.shift[0], -self.shift[1]), dims=(1, 2))
        windows = _partition_windows(grid, self.window)
        windows = self.attention(windows, self.attention_mask)
        grid = _reverse_windows(windows, self.window, self.resolution, batch)
        if self.shift != (0, 0):
            grid = torch.roll(grid, shifts=self.shift, dims=(1, 2))
        values = shortcut + self.drop_path(grid.reshape(batch, tokens, channels))
        return values + self.drop_path(self.mlp(self.norm2(values)))


class _PatchMerging(nn.Module):
    def __init__(self, dimension: int, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.dimension = dimension
        self.resolution = resolution
        self.output_resolution = ((resolution[0] + 1) // 2, (resolution[1] + 1) // 2)
        self.norm = nn.LayerNorm(4 * dimension)
        self.reduction = nn.Linear(4 * dimension, 2 * dimension, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = values.shape
        height, width = self.resolution
        if tokens != height * width or channels != self.dimension:
            raise ValueError("patch merging received an unexpected token grid")
        grid = values.view(batch, height, width, channels)
        if height % 2 or width % 2:
            grid = F.pad(grid, (0, 0, 0, width % 2, 0, height % 2))
        merged = torch.cat(
            (grid[:, 0::2, 0::2], grid[:, 1::2, 0::2], grid[:, 0::2, 1::2], grid[:, 1::2, 1::2]),
            dim=-1,
        )
        return self.reduction(self.norm(merged.reshape(batch, -1, 4 * channels)))


class Task3MicroSwin(nn.Module):
    """A small pure-patch shifted-window transformer trained from random weights."""

    def __init__(self, config: MicroSwinConfig) -> None:
        super().__init__()
        self.config = config
        patch_dimension = 3 * config.patch_size**2
        first_resolution = (
            config.image_height // config.patch_size,
            config.image_width // config.patch_size,
        )
        self.patchifier = nn.Unfold(kernel_size=config.patch_size, stride=config.patch_size)
        self.patch_projection = nn.Linear(patch_dimension, config.dimensions[0])
        self.position = nn.Parameter(
            torch.zeros(1, first_resolution[0] * first_resolution[1], config.dimensions[0])
        )
        resolutions = (first_resolution, (10, 8), (5, 4))
        self.stage1 = self._stage(config, 0, resolutions[0])
        self.merge1 = _PatchMerging(config.dimensions[0], resolutions[0])
        self.stage2 = self._stage(config, 1, resolutions[1])
        self.merge2 = _PatchMerging(config.dimensions[1], resolutions[1])
        self.stage3 = self._stage(config, 2, resolutions[2])
        self.norm = nn.LayerNorm(config.dimensions[-1])
        self.classifier = nn.Linear(config.dimensions[-1], config.num_classes)
        self.apply(self._initialise)
        nn.init.trunc_normal_(self.position, std=0.02)
        for module in self.modules():
            if isinstance(module, _WindowAttention):
                nn.init.trunc_normal_(module.relative_position_bias, std=0.02)

    @staticmethod
    def _stage(config: MicroSwinConfig, stage: int, resolution: tuple[int, int]) -> nn.Sequential:
        blocks: list[nn.Module] = []
        depth_total = sum(config.depths)
        depth_before = sum(config.depths[:stage])
        window = config.windows[stage]
        window_count = (resolution[0] // window[0]) * (resolution[1] // window[1])
        for block in range(config.depths[stage]):
            shifted = block % 2 == 1 and window_count > 1
            shift = (window[0] // 2, window[1] // 2) if shifted else (0, 0)
            probability = config.drop_path * (depth_before + block) / max(1, depth_total - 1)
            blocks.append(
                _MicroSwinBlock(
                    config.dimensions[stage],
                    resolution,
                    config.heads[stage],
                    window,
                    shift,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_path=probability,
                )
            )
        return nn.Sequential(*blocks)

    @staticmethod
    def _initialise(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(images).transpose(1, 2)
        tokens = self.patch_projection(tokens) + self.position
        tokens = self.stage1(tokens)
        tokens = self.merge1(tokens)
        tokens = self.stage2(tokens)
        tokens = self.merge2(tokens)
        tokens = self.stage3(tokens)
        return self.classifier(self.norm(tokens).mean(dim=1))


class _Task3ViewDataset(Task3ImageDataset):
    def __init__(self, *args: Any, image_view: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if image_view not in ("full", "foreground_masked"):
            raise ValueError(f"unsupported micro-Swin image view: {image_view}")
        self.image_view = image_view

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.image_view == "full":
            return super().__getitem__(index)
        row = self.frame.iloc[index]
        image = _load_corrupted(self.root / str(row["path"]), self.corruption)
        image = Image.fromarray(foreground_views(image)[self.image_view])
        image = apply_training_augmentation(image, self.augmentation)
        array = transform_image(
            image,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )
        return {
            "image": torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()),
            "label": int(self.label_to_index[str(row[self.target])]),
            "id": int(row["id"]),
            "cv_fold": int(row["cv_fold"]),
            "product_family_group": str(row["product_family_group"]),
            "path": str(row["path"]),
        }


def _fit_view_rgb_stats(
    training: pd.DataFrame,
    *,
    root: Path,
    image_view: str,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    stats = StreamingStats(channels=3)
    total = len(training)
    for position, relative_path in enumerate(training["path"], start=1):
        image = _load_corrupted(root / str(relative_path), None)
        if image_view != "full":
            image = Image.fromarray(foreground_views(image)[image_view])
        array, mask = transform_image_with_mask(image, image_size=image_size)
        stats.update(array, content_mask=mask)
        if position == 1 or position % 5_000 == 0 or position == total:
            _log(f"normalisation view={image_view}: {position:,}/{total:,} images")
    result = stats.to_dict()
    if any(float(value) <= 0 for value in result["std"]):
        raise ValueError("fold-training RGB statistics contain a non-positive standard deviation")
    return result


def _loader(
    dataset: Task3ImageDataset,
    *,
    config: MicroSwinConfig,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def _pass(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    gradient_clip_norm: float = 1.0,
    mixed_precision: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, list[Any]]]:
    training = optimizer is not None
    model.train(training)
    loss_numerator = 0.0
    loss_denominator = 0.0
    row_count = 0
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    trace: dict[str, list[Any]] = {"id": [], "cv_fold": [], "product_family_group": [], "path": []}
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, target)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
            rows = len(target)
            if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
                batch_denominator = float(criterion.weight[target].detach().sum().cpu())
            else:
                batch_denominator = float(rows)
            loss_numerator += float(loss.detach()) * batch_denominator
            loss_denominator += batch_denominator
            row_count += rows
            labels.append(target.detach().cpu().numpy())
            probabilities.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
            if not training:
                trace["id"].extend(batch["id"].tolist())
                trace["cv_fold"].extend(batch["cv_fold"].tolist())
                trace["product_family_group"].extend(batch["product_family_group"])
                trace["path"].extend(batch["path"])
    if row_count == 0 or loss_denominator == 0.0:
        raise ValueError("a micro-Swin data loader produced no rows")
    return (
        loss_numerator / loss_denominator,
        np.concatenate(labels),
        np.concatenate(probabilities),
        trace,
    )


def _learning_rate_multiplier(config: MicroSwinConfig, epoch_index: int) -> float:
    minimum_ratio = config.minimum_learning_rate / config.learning_rate
    if epoch_index < config.warmup_epochs:
        return max(minimum_ratio, (epoch_index + 1) / config.warmup_epochs)
    progress = (epoch_index - config.warmup_epochs) / max(
        1, config.epochs - config.warmup_epochs - 1
    )
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _completed_fold(
    target_dir: Path,
    *,
    fold: int,
    config_hash: str,
    registry_path: Path,
) -> dict[str, Any] | None:
    if not registry_path.is_file():
        return None
    registry = pd.read_csv(registry_path, keep_default_na=False)
    complete_ids = set(registry.loc[registry["status"].eq("complete"), "run_id"].astype(str))
    for metrics_path in sorted(target_dir.glob("*/metrics.json"), reverse=True):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(metrics.get("run_id", ""))
        run_dir = metrics_path.parent
        checkpoint = run_dir / "final_epoch.pt"
        predictions = run_dir / "oof_predictions.csv"
        if (
            run_id in complete_ids
            and int(metrics.get("validation_fold", -1)) == fold
            and metrics.get("config_hash") == config_hash
            and checkpoint.is_file()
            and predictions.is_file()
            and metrics.get("checkpoint_sha256") == compute_sha256(checkpoint)
            and metrics.get("prediction_sha256") == compute_sha256(predictions)
        ):
            return {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "prediction_path": str(predictions),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
    return None


def run_micro_swin_fold(
    target: str,
    validation_fold: int,
    *,
    output_root: str | Path,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
    reuse_completed: bool = True,
) -> dict[str, Any]:
    """Train or reuse one canonical micro-Swin screen fold."""

    if target not in ("gender", "usage"):
        raise ValueError(f"unsupported Task 3 target: {target}")
    if validation_fold not in SCREEN_FOLDS:
        raise ValueError("the micro-Swin screen is frozen to folds 0 and 4")
    root = Path(root)
    output_root = Path(output_root)
    registry_path = Path(registry_path)
    config = MicroSwinConfig(target=target)  # type: ignore[arg-type]
    config_payload = {
        **config.to_dict(),
        "validation_fold": validation_fold,
        "screen_folds": list(SCREEN_FOLDS),
        "source_sha256": compute_sha256(Path(__file__)),
        "checkpoint_policy": "fixed_final_epoch",
        "normalisation_scope": "outer_fold_training_rows_only",
    }
    config_hash = _digest(config_payload)
    target_dir = output_root / ARTIFACT_ROOT / target
    if reuse_completed and (
        completed := _completed_fold(
            target_dir,
            fold=validation_fold,
            config_hash=config_hash,
            registry_path=registry_path,
        )
    ):
        _log(f"reusing completed target={target} fold={validation_fold}: {completed['run_id']}")
        return completed

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but Colab has no active GPU runtime")
    device = torch.device(device_name)
    set_reproducible_seed(config.seed)
    splits_path = root / SPLITS_CSV.relative_to(ROOT)
    label_maps_path = root / LABEL_MAPS_JSON.relative_to(ROOT)
    splits = load_splits(splits_path)
    label_maps = load_label_maps(label_maps_path)
    classes, label_to_index = _class_spec(label_maps, target)
    training, validation = task3_target_frames(
        splits, target=target, validation_fold=validation_fold
    )
    class_counts = np.asarray(
        [int(training[target].eq(class_name).sum()) for class_name in classes], dtype=np.int64
    )
    class_weights = (
        effective_number_class_weights(class_counts, beta=0.999, cap=5.0)
        if target == "usage"
        else None
    )

    execution = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + uuid.uuid4().hex[:6]
    run_id = (
        f"{RUN_PREFIX}_{target}_f{validation_fold}_s{config.seed}_{config_hash[:12]}_{execution}"
    )
    run_dir = target_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "config": run_dir / "config.json",
        "normalization": run_dir / "normalization.json",
        "history": run_dir / "history.csv",
        "checkpoint": run_dir / "final_epoch.pt",
        "predictions": run_dir / "oof_predictions.csv",
        "metrics": run_dir / "metrics.json",
        "robustness": run_dir / "robustness.csv",
    }
    _json_dump(config_payload, paths["config"])
    _log(f"target={target} fold={validation_fold}: fitting {config.input_view} RGB statistics")
    stats = _fit_view_rgb_stats(
        training,
        root=root,
        image_view=config.input_view,
        image_size=(config.image_height, config.image_width),
    )
    _json_dump(
        {
            **stats,
            "fit_scope": "outer_fold_training_rows_only",
            "input_view": config.input_view,
            "validation_fold": validation_fold,
            "padding_excluded": True,
        },
        paths["normalization"],
    )

    dataset_kwargs = {
        "target": target,
        "label_to_index": label_to_index,
        "mean": stats["mean"],
        "std": stats["std"],
        "root": root,
        "image_size": (config.image_height, config.image_width),
        "image_view": config.input_view,
    }
    train_dataset = _Task3ViewDataset(training, augmentation=config.augmentation, **dataset_kwargs)
    validation_dataset = _Task3ViewDataset(validation, **dataset_kwargs)
    train_loader = _loader(train_dataset, config=config, shuffle=True, device=device)
    validation_loader = _loader(validation_dataset, config=config, shuffle=False, device=device)
    model = Task3MicroSwin(config).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    weight_tensor = (
        torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: _learning_rate_multiplier(config, epoch)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    environment = runtime_environment(device)
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    registry.start(
        {
            "run_id": run_id,
            "experiment_id": EXPERIMENT_IDS[target],
            "hypothesis_id": HYPOTHESIS_IDS[target],
            "parent_run_ids": [],
            "task": "task3",
            "target": target,
            "validation_fold": validation_fold,
            "seed": config.seed,
            "debug": False,
            "scratch": True,
            "submission_eligible": True,
            "config_hash": config_hash,
            "config_path": paths["config"],
            "split_digest": compute_sha256(splits_path),
            "label_map_digest": compute_sha256(label_maps_path),
            "training_product_count": len(training),
            "validation_product_count": len(validation),
            "training_family_count": training["product_family_group"].nunique(),
            "validation_family_count": validation["product_family_group"].nunique(),
            "model_family": MODEL_FAMILY,
            "parameter_count": parameter_count,
            "history_path": paths["history"],
            "environment_json": environment,
            "last_completed_stage": "registered_before_first_optimizer_step",
        }
    )
    _log(f"registered {run_id}; first optimiser step may now run on {environment['gpu']}")

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_stage = "registered_before_first_optimizer_step"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, config.epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_loss, train_labels, train_probabilities, _ = _pass(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                gradient_clip_norm=config.gradient_clip_norm,
                mixed_precision=config.mixed_precision,
            )
            validation_loss, validation_labels, validation_probabilities, _ = _pass(
                model,
                validation_loader,
                criterion,
                device,
                mixed_precision=config.mixed_precision,
            )
            train_metrics = classification_metrics(train_labels, train_probabilities, classes)
            validation_metrics = classification_metrics(
                validation_labels, validation_probabilities, classes
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "train_loss": train_loss,
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_loss": validation_loss,
                    "validation_macro_f1": validation_metrics["macro_f1"],
                }
            )
            pd.DataFrame(history).to_csv(paths["history"], index=False)
            scheduler.step()
            last_stage = f"epoch_{epoch}_complete"
            registry.update(run_id, {"last_completed_stage": last_stage})
            _log(
                f"target={target} fold={validation_fold} epoch={epoch}/{config.epochs} "
                f"train_macro_f1={train_metrics['macro_f1']:.4f} "
                f"validation_macro_f1={validation_metrics['macro_f1']:.4f}"
            )

        training_finished = time.perf_counter()
        torch.save(
            {
                "run_id": run_id,
                "config": config_payload,
                "class_names": classes,
                "normalization": stats,
                "model_state_dict": model.state_dict(),
            },
            paths["checkpoint"],
        )
        clean_loss, labels, probabilities, trace = _pass(
            model,
            validation_loader,
            criterion,
            device,
            mixed_precision=config.mixed_precision,
        )
        train_evaluation_loader = _loader(
            _Task3ViewDataset(training, **dataset_kwargs),
            config=config,
            shuffle=False,
            device=device,
        )
        final_train_loss, final_train_labels, final_train_probabilities, _ = _pass(
            model,
            train_evaluation_loader,
            criterion,
            device,
            mixed_precision=config.mixed_precision,
        )
        predictions = _prediction_frame(labels, probabilities, trace, classes, run_id)
        predictions.to_csv(paths["predictions"], index=False)
        metrics = classification_metrics(labels, probabilities, classes)
        train_metrics = classification_metrics(
            final_train_labels, final_train_probabilities, classes
        )
        if target == "usage":
            metrics["macro_f1_without_home"] = float(
                np.mean([row["f1"] for row in metrics["per_class"] if row["class_name"] != "Home"])
            )
        train_seconds = training_finished - started
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        metrics.update(
            {
                "run_id": run_id,
                "target": target,
                "validation_fold": validation_fold,
                "experiment_id": EXPERIMENT_IDS[target],
                "hypothesis_id": HYPOTHESIS_IDS[target],
                "parent_run_ids": [],
                "model_family": MODEL_FAMILY,
                "config_hash": config_hash,
                "input_view": config.input_view,
                "scratch": True,
                "submission_eligible": True,
                "parameter_count": parameter_count,
                "epochs_completed": config.epochs,
                "loss": clean_loss,
                "final_train_eval_loss": final_train_loss,
                "final_train_eval_macro_f1": train_metrics["macro_f1"],
                "final_train_validation_macro_f1_gap": float(
                    train_metrics["macro_f1"] - metrics["macro_f1"]
                ),
                "class_counts": class_counts.tolist(),
                "class_weights": class_weights.tolist() if class_weights is not None else None,
                "train_seconds": train_seconds,
                "peak_memory_bytes": peak_memory,
                "screen_scope": "canonical_outer_folds_0_and_4",
            }
        )

        robustness_rows: list[dict[str, Any]] = []
        for corruption in CORE_CORRUPTIONS:
            corrupted_loader = _loader(
                _Task3ViewDataset(validation, corruption=corruption, **dataset_kwargs),
                config=config,
                shuffle=False,
                device=device,
            )
            corruption_loss, corruption_labels, corruption_probabilities, _ = _pass(
                model,
                corrupted_loader,
                criterion,
                device,
                mixed_precision=config.mixed_precision,
            )
            corruption_metrics = classification_metrics(
                corruption_labels, corruption_probabilities, classes
            )
            robustness_rows.append(
                {
                    "run_id": run_id,
                    "validation_fold": validation_fold,
                    "corruption": corruption,
                    "loss": corruption_loss,
                    "macro_f1": corruption_metrics["macro_f1"],
                    "macro_f1_change": float(corruption_metrics["macro_f1"] - metrics["macro_f1"]),
                }
            )
        pd.DataFrame(robustness_rows).to_csv(paths["robustness"], index=False)
        metrics["diagnostic_seconds"] = time.perf_counter() - training_finished
        metrics["latency_ms_batch_1"] = _measure_latency(
            model,
            device,
            height=config.image_height,
            width=config.image_width,
        )
        metrics["prediction_sha256"] = compute_sha256(paths["predictions"])
        metrics["checkpoint_sha256"] = compute_sha256(paths["checkpoint"])
        _json_dump(metrics, paths["metrics"])
        registry.complete(
            run_id,
            {
                "checkpoint_path": paths["checkpoint"],
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "prediction_path": paths["predictions"],
                "prediction_sha256": metrics["prediction_sha256"],
                "metrics_json": metrics,
                "train_seconds": train_seconds,
                "peak_memory_bytes": peak_memory,
                "checkpoint_bytes": paths["checkpoint"].stat().st_size,
                "last_completed_stage": "screen_fold_complete",
            },
        )
        _log(f"complete target={target} fold={validation_fold}: macro_f1={metrics['macro_f1']:.4f}")
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "prediction_path": str(paths["predictions"]),
            "metrics_path": str(paths["metrics"]),
            "metrics": metrics,
        }
    except BaseException as error:
        registry.fail(run_id, error, last_completed_stage=last_stage)
        raise


def run_micro_swin_screen(
    target: str,
    *,
    output_root: str | Path,
    folds: Iterable[int] = SCREEN_FOLDS,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
    anchor_prediction_path: str | Path | None = None,
    reuse_completed: bool = True,
) -> dict[str, Any]:
    """Run the frozen two-fold transformer screen and create matched evidence."""

    fold_list = tuple(int(fold) for fold in folds)
    if fold_list != SCREEN_FOLDS:
        raise ValueError("the micro-Swin screen requires folds 0 and 4 in order")
    results = [
        run_micro_swin_fold(
            target,
            fold,
            output_root=output_root,
            registry_path=registry_path,
            registry_mirrors=registry_mirrors,
            root=root,
            device_name=device_name,
            reuse_completed=reuse_completed,
        )
        for fold in fold_list
    ]
    return _aggregate_screen(
        target,
        results,
        output_root=Path(output_root),
        root=Path(root),
        model_family=MODEL_FAMILY,
        experiment_id=EXPERIMENT_IDS[target],
        hypothesis_id=HYPOTHESIS_IDS[target],
        anchor_prediction_path=anchor_prediction_path,
        artifact_root=ARTIFACT_ROOT,
        seconds_per_fold_limit=L4_SECONDS_PER_FOLD_LIMIT,
        memory_limit_bytes=GPU_MEMORY_LIMIT_BYTES,
    )


def check_micro_swin_screen_setup(
    *,
    root: str | Path = ROOT,
    device_name: str = "cuda",
    folds: Iterable[int] = SCREEN_FOLDS,
) -> dict[str, Any]:
    """Validate data, split, GPU, and both models without an optimizer step."""

    fold_list = tuple(int(fold) for fold in folds)
    if fold_list != SCREEN_FOLDS:
        raise ValueError("the micro-Swin screen requires folds 0 and 4 in order")
    root = Path(root)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but Colab has no active GPU runtime")
    device = torch.device(device_name)
    environment = runtime_environment(device)
    if device.type == "cuda":
        validate_verified_colab_runtime(environment)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    required_paths = {
        str(path) for path in splits.loc[splits["partition"].eq("development"), "path"].astype(str)
    }
    missing = sorted(path for path in required_paths if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} development images are missing; first: {missing[:5]}"
        )

    targets: dict[str, Any] = {}
    for target in ("gender", "usage"):
        config = MicroSwinConfig(target=target)  # type: ignore[arg-type]
        classes, _ = _class_spec(label_maps, target)
        if len(classes) != config.num_classes:
            raise ValueError("micro-Swin head and fixed class order disagree")
        fold_rows = []
        for fold in fold_list:
            training, validation = task3_target_frames(splits, target=target, validation_fold=fold)
            fold_rows.append(
                {"fold": fold, "training_rows": len(training), "validation_rows": len(validation)}
            )
        set_reproducible_seed(config.seed)
        model = Task3MicroSwin(config).to(device)
        with torch.inference_mode():
            output = model(
                torch.zeros(2, 3, config.image_height, config.image_width, device=device)
            )
        if tuple(output.shape) != (2, config.num_classes):
            raise RuntimeError(f"unexpected {target} output shape: {tuple(output.shape)}")
        if any(isinstance(module, nn.Conv2d) for module in model.modules()):
            raise RuntimeError("micro-Swin must not contain a convolution tokenizer or block")
        targets[target] = {
            "classes": classes,
            "input_view": config.input_view,
            "fold_rows": fold_rows,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "config": config.to_dict(),
        }
        del model
    return {
        "ready": True,
        "device": str(device),
        "environment": environment,
        "screen_folds": list(SCREEN_FOLDS),
        "model_family": MODEL_FAMILY,
        "targets": targets,
        "gpu_memory_limit_bytes": GPU_MEMORY_LIMIT_BYTES,
        "optimizer_steps": 0,
        "pretrained_weights_loaded": False,
    }
