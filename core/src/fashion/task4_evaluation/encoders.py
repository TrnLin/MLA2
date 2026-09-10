"""One encoder interface over the four embedding methods the holdout run compares.

Each direction of the evaluation normalizes its images with that direction's own saved
training statistics, so a built encoder is bound to one direction and never re-reads it.
The fifth compared method, the random floor, produces rankings from a seeded permutation
instead of embeddings and therefore has its own function here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn

from fashion.task4.hog_fusion import HOG_FUSION_DESCRIPTOR_DIM, extract_hog_fusion
from fashion.task4.models import EMBEDDING_DIM
from fashion.task4.portable_model import load_r5_inference_package
from fashion.task4.preprocessing import PreprocessedImage, normalize_for_model
from fashion.task4.probe import (
    COLOUR_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    extract_spatial_probe,
)
from fashion.task4.training import load_checkpoint
from fashion.task4_evaluation.spec import HoldoutEvaluationSpec

R5_METHOD = "r5_scratch_autoencoder"
PROBE_METHOD = "spatial_hsv_edge_probe"
HOG_FUSION_METHOD = "hog_hsv_edge_fusion"
B1_METHOD = "b1_pretrained_resnet18"
RANDOM_FLOOR_METHOD = "random_floor"

PROBE_DESCRIPTOR_DIM = COLOUR_FEATURE_DIM + EDGE_FEATURE_DIM
RANDOM_FLOOR_COLUMNS = ("query_id", "candidate_id", "distance", "rank")

B1_EVIDENCE_MANIFEST_RELATIVE_PATH = (
    "results/evidence/task4/learned/task4-benchmark-b1-task9-preexec/manifest.json"
)
# The B1 evidence manifest records no weight origin; the registry row for that run does,
# and load_checkpoint cross-checks this value against the checkpoint's own provenance.
B1_WEIGHT_ORIGIN = "ResNet18_Weights.IMAGENET1K_V1"
# results/runs.csv leaves parent_run_id empty for B1, which the checkpoint stores as None.
B1_PARENT_RUN_ID: str | None = None

__all__ = [
    "B1_EVIDENCE_MANIFEST_RELATIVE_PATH",
    "B1_METHOD",
    "B1_PARENT_RUN_ID",
    "B1_WEIGHT_ORIGIN",
    "HOG_FUSION_METHOD",
    "PROBE_DESCRIPTOR_DIM",
    "PROBE_METHOD",
    "R5_METHOD",
    "RANDOM_FLOOR_COLUMNS",
    "RANDOM_FLOOR_METHOD",
    "B1Encoder",
    "HogFusionEncoder",
    "MethodEncoder",
    "ProbeEncoder",
    "R5Encoder",
    "build_method_encoders",
    "build_random_floor_rankings",
    "load_b1_encoder",
    "load_r5_encoder",
]


class MethodEncoder(Protocol):
    """One compared method reduced to image in, unit-norm embedding out."""

    name: str
    dimension: int
    pretrained: bool

    def encode(self, image: PreprocessedImage) -> np.ndarray: ...


def _unit_norm(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("embedding must be finite")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("embedding must have positive norm")
    return (values / np.float32(norm)).astype(np.float32, copy=False)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _required(payload: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in payload:
        raise ValueError(f"{label} is missing {key}")
    return payload[key]


def _required_mapping(
    payload: Mapping[str, Any], key: str, *, label: str
) -> Mapping[str, Any]:
    value = _required(payload, key, label=label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} {key} must be one JSON object")
    return value


def _statistics_triple(
    payload: Mapping[str, Any], *, field_name: str, label: str
) -> tuple[float, float, float]:
    raw = payload.get(field_name)
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{label} {field_name} must contain three values")
    try:
        first, second, third = (float(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} {field_name} must contain three numbers") from error
    return first, second, third


def _encode_with_model(
    model: nn.Module,
    image: PreprocessedImage,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    device: torch.device,
) -> np.ndarray:
    normalized = normalize_for_model(image, mean=mean, std=std)
    batch = (
        torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        embedding = model.encode(batch)
    return embedding.detach().to("cpu").numpy()[0]


@dataclass(frozen=True, slots=True)
class ProbeEncoder:
    """The untrained spatial HSV and edge descriptor, which reads no statistics."""

    name: str = PROBE_METHOD
    dimension: int = PROBE_DESCRIPTOR_DIM
    pretrained: bool = False

    def encode(self, image: PreprocessedImage) -> np.ndarray:
        return _unit_norm(extract_spatial_probe(image.pixels, image.content_mask))


@dataclass(frozen=True, slots=True)
class HogFusionEncoder:
    """The equal-block HOG and probe fusion, which also reads no statistics."""

    name: str = HOG_FUSION_METHOD
    dimension: int = HOG_FUSION_DESCRIPTOR_DIM
    pretrained: bool = False

    def encode(self, image: PreprocessedImage) -> np.ndarray:
        return _unit_norm(extract_hog_fusion(image.pixels, image.content_mask))


@dataclass(frozen=True, slots=True)
class R5Encoder:
    """The selected scratch autoencoder, bound to one direction's statistics."""

    model: nn.Module
    direction: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    device: torch.device
    name: str = R5_METHOD
    dimension: int = EMBEDDING_DIM
    pretrained: bool = False

    def encode(self, image: PreprocessedImage) -> np.ndarray:
        return _unit_norm(
            _encode_with_model(
                self.model,
                image,
                mean=self.mean,
                std=self.std,
                device=self.device,
            )
        )


@dataclass(frozen=True, slots=True)
class B1Encoder:
    """The pretrained ResNet-18 benchmark ceiling, bound to one direction."""

    model: nn.Module
    direction: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    device: torch.device
    name: str = B1_METHOD
    dimension: int = EMBEDDING_DIM
    pretrained: bool = True

    def encode(self, image: PreprocessedImage) -> np.ndarray:
        return _unit_norm(
            _encode_with_model(
                self.model,
                image,
                mean=self.mean,
                std=self.std,
                device=self.device,
            )
        )


def load_r5_encoder(
    package_directory: str | Path,
    *,
    direction: str,
    device: torch.device | str = "cpu",
) -> R5Encoder:
    """Load the portable R5 package and bind one direction's saved statistics."""

    package = Path(package_directory)
    manifest = _read_json(package / "manifest.json", label="portable R5 manifest")
    normalization = manifest.get("normalization")
    statistics = (
        normalization.get(direction) if isinstance(normalization, Mapping) else None
    )
    if not isinstance(statistics, Mapping):
        raise ValueError(
            f"portable R5 manifest has no {direction} normalization statistics"
        )
    label = f"portable R5 {direction} normalization"
    dimension = _required(manifest, "embedding_dim", label="portable R5 manifest")
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise ValueError("portable R5 manifest embedding_dim must be an integer")
    parsed_device = torch.device(device)
    model = load_r5_inference_package(package, device=parsed_device)
    return R5Encoder(
        model=model,
        direction=direction,
        mean=_statistics_triple(statistics, field_name="mean", label=label),
        std=_statistics_triple(statistics, field_name="std", label=label),
        device=parsed_device,
        dimension=dimension,
    )


def load_b1_encoder(
    project_root: str | Path,
    *,
    direction: str,
    split_fingerprint: str,
    device: torch.device | str = "cpu",
) -> B1Encoder:
    """Load the validated B1 benchmark checkpoint and its direction's statistics."""

    root = Path(project_root)
    evidence_label = "B1 evidence manifest"
    evidence = _read_json(
        root / B1_EVIDENCE_MANIFEST_RELATIVE_PATH, label=evidence_label
    )
    sources = evidence.get("source_artifacts")
    source = sources.get(direction) if isinstance(sources, Mapping) else None
    statistics_record = (
        source.get("statistics") if isinstance(source, Mapping) else None
    )
    if not isinstance(statistics_record, Mapping):
        raise ValueError(f"{evidence_label} has no {direction} statistics artifact")
    statistics_label = f"B1 {direction} statistics"
    statistics_path = str(
        _required(statistics_record, "path", label=f"{statistics_label} artifact")
    )
    statistics = _read_json(root / statistics_path, label=statistics_label)
    if statistics.get("source") != direction:
        raise ValueError(f"B1 statistics source is not {direction}")
    if statistics.get("split_fingerprint") != split_fingerprint:
        raise ValueError("B1 statistics split fingerprint does not match the spec")

    checkpoint = _required_mapping(evidence, "checkpoint", label=evidence_label)
    checkpoint_label = f"{evidence_label} checkpoint"
    checkpoint_path = str(_required(checkpoint, "path", label=checkpoint_label))
    checkpoint_sha256 = str(_required(checkpoint, "sha256", label=checkpoint_label))
    config_hash = str(_required(evidence, "config_hash", label=evidence_label))
    run_id = str(_required(evidence, "run_id", label=evidence_label))
    run_kind = str(_required(evidence, "run_kind", label=evidence_label))
    parsed_device = torch.device(device)
    loaded = load_checkpoint(
        root / checkpoint_path,
        expected_sha256=checkpoint_sha256,
        expected_config_hash=config_hash,
        expected_split_fingerprint=split_fingerprint,
        expected_weight_origin=B1_WEIGHT_ORIGIN,
        expected_parent_run_id=B1_PARENT_RUN_ID,
        expected_run_id=run_id,
        expected_run_kind=run_kind,
        map_location=parsed_device,
    )
    return B1Encoder(
        model=loaded.model.to(parsed_device).eval(),
        direction=direction,
        mean=_statistics_triple(statistics, field_name="mean", label=statistics_label),
        std=_statistics_triple(statistics, field_name="std", label=statistics_label),
        device=parsed_device,
    )


def build_method_encoders(
    spec: HoldoutEvaluationSpec,
    *,
    project_root: Path,
    direction: str,
    device: str = "cpu",
) -> dict[str, MethodEncoder]:
    """Build every embedding method the spec compares, bound to one direction."""

    if direction not in spec.directions:
        raise ValueError(
            f"direction {direction!r} is not one of the declared directions "
            f"{list(spec.directions)}"
        )
    known = {R5_METHOD, PROBE_METHOD, HOG_FUSION_METHOD, B1_METHOD, RANDOM_FLOOR_METHOD}
    if set(spec.methods) != known:
        raise ValueError(
            f"spec methods {list(spec.methods)} do not match the implemented methods "
            f"{sorted(known)}"
        )
    root = Path(project_root)
    encoders: dict[str, MethodEncoder] = {
        PROBE_METHOD: ProbeEncoder(),
        HOG_FUSION_METHOD: HogFusionEncoder(),
        R5_METHOD: load_r5_encoder(
            root / spec.model_package_directory, direction=direction, device=device
        ),
        B1_METHOD: load_b1_encoder(
            root,
            direction=direction,
            split_fingerprint=spec.split_fingerprint,
            device=device,
        ),
    }
    return {method: encoders[method] for method in spec.methods if method in encoders}


def _unique_int64(values: Iterable[int], *, label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{label} must contain at least one ID")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{label} must be unique")
    return array


def build_random_floor_rankings(
    query_ids: Iterable[int],
    gallery_ids: Iterable[int],
    *,
    seed: int,
    max_k: int,
) -> pd.DataFrame:
    """Rank a seeded random sample of the gallery for every query."""

    queries = np.sort(_unique_int64(query_ids, label="query IDs"))
    gallery = np.sort(_unique_int64(gallery_ids, label="gallery IDs"))
    if max_k < 1:
        raise ValueError("max_k must be a positive integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    records: list[dict[str, float | int]] = []
    for query_id in queries:
        candidates = gallery[gallery != query_id]
        if len(candidates) < max_k:
            raise ValueError(
                f"query {int(query_id)} has {len(candidates)} candidates, fewer than "
                f"max_k {max_k}"
            )
        chosen = generator.choice(candidates, size=max_k, replace=False)
        records.extend(
            {
                "query_id": int(query_id),
                "candidate_id": int(candidate_id),
                "distance": float(rank),
                "rank": int(rank),
            }
            for rank, candidate_id in enumerate(chosen, start=1)
        )
    frame = pd.DataFrame.from_records(records)
    return frame.loc[:, list(RANDOM_FLOOR_COLUMNS)]
