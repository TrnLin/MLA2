"""Fixed, untrained spatial colour-and-edge probe for Task 4."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from fashion.task4.protocol import (
    RetrievalViews,
    family_candidate_mask,
    prepare_rankings,
)

GRID_SIZE = 4
HSV_BINS = (8, 4, 4)
EDGE_BINS = 9
PROBE_VERSION = "spatial-hsv-edge-4x4-v2"
COLOUR_FEATURE_DIM = GRID_SIZE * GRID_SIZE * sum(HSV_BINS)
EDGE_FEATURE_DIM = GRID_SIZE * GRID_SIZE * EDGE_BINS

__all__ = (
    "COLOUR_FEATURE_DIM",
    "EDGE_BINS",
    "EDGE_FEATURE_DIM",
    "GRID_SIZE",
    "HSV_BINS",
    "PROBE_VERSION",
    "extract_spatial_probe",
    "rank_embeddings",
    "rank_probe_embeddings",
)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 0

    red_max = nonzero & (maximum == rgb[..., 0])
    green_max = nonzero & (maximum == rgb[..., 1])
    blue_max = nonzero & (maximum == rgb[..., 2])
    red_hue = np.divide(
        rgb[..., 1] - rgb[..., 2],
        delta,
        out=np.zeros_like(delta),
        where=nonzero,
    )
    green_hue = np.divide(
        rgb[..., 2] - rgb[..., 0],
        delta,
        out=np.zeros_like(delta),
        where=nonzero,
    )
    blue_hue = np.divide(
        rgb[..., 0] - rgb[..., 1],
        delta,
        out=np.zeros_like(delta),
        where=nonzero,
    )
    hue[red_max] = red_hue[red_max] % 6
    hue[green_max] = green_hue[green_max] + 2
    hue[blue_max] = blue_hue[blue_max] + 4
    hue /= 6

    saturation = np.zeros_like(maximum)
    positive = maximum > 0
    saturation[positive] = delta[positive] / maximum[positive]
    return np.stack([hue, saturation, maximum], axis=-1)


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0 else values


def extract_spatial_probe(pixels: np.ndarray, content_mask: np.ndarray) -> np.ndarray:
    """Extract the frozen 4×4 HSV and gradient-orientation descriptor."""
    if (
        pixels.ndim != 3
        or pixels.shape[-1] != 3
        or pixels.dtype != np.uint8
        or content_mask.shape != pixels.shape[:2]
    ):
        raise ValueError("probe expects uint8 HWC RGB pixels and a matching mask")
    if content_mask.dtype != bool:
        raise ValueError("content mask must be boolean")
    if not content_mask.any():
        raise ValueError("probe requires at least one content pixel")

    rgb = pixels.astype(np.float32) / 255.0
    hsv = _rgb_to_hsv(rgb)
    luminance = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    )
    luminance[~content_mask] = 0.0
    gradient_x = (
        np.gradient(luminance, axis=1).astype(np.float32)
        if luminance.shape[1] > 1
        else np.zeros_like(luminance)
    )
    gradient_y = (
        np.gradient(luminance, axis=0).astype(np.float32)
        if luminance.shape[0] > 1
        else np.zeros_like(luminance)
    )
    valid_x = np.zeros_like(content_mask)
    if content_mask.shape[1] > 2:
        valid_x[:, 1:-1] = (
            content_mask[:, :-2]
            & content_mask[:, 1:-1]
            & content_mask[:, 2:]
        )
    valid_y = np.zeros_like(content_mask)
    if content_mask.shape[0] > 2:
        valid_y[1:-1] = (
            content_mask[:-2]
            & content_mask[1:-1]
            & content_mask[2:]
        )
    gradient_x[~valid_x] = 0.0
    gradient_y[~valid_y] = 0.0
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)

    row_edges = np.linspace(0, pixels.shape[0], GRID_SIZE + 1, dtype=int)
    column_edges = np.linspace(0, pixels.shape[1], GRID_SIZE + 1, dtype=int)
    colour_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            row_slice = slice(row_edges[row], row_edges[row + 1])
            column_slice = slice(column_edges[column], column_edges[column + 1])
            cell_mask = content_mask[row_slice, column_slice]
            cell_hsv = hsv[row_slice, column_slice][cell_mask]
            for channel, bins in enumerate(HSV_BINS):
                histogram, _ = np.histogram(
                    cell_hsv[:, channel] if len(cell_hsv) else np.empty(0),
                    bins=bins,
                    range=(0.0, 1.0),
                )
                colour_parts.append(histogram.astype(np.float32))

            cell_orientation = orientation[row_slice, column_slice][cell_mask]
            cell_magnitude = magnitude[row_slice, column_slice][cell_mask]
            edge_histogram, _ = np.histogram(
                cell_orientation,
                bins=EDGE_BINS,
                range=(0.0, np.pi),
                weights=cell_magnitude,
            )
            edge_parts.append(edge_histogram.astype(np.float32))

    colour = _unit(np.concatenate(colour_parts))
    edges = _unit(np.concatenate(edge_parts))
    feature = np.concatenate([colour, edges]) / np.sqrt(2.0)
    return _unit(feature).astype(np.float32, copy=False)


def _numeric_ids(values: np.ndarray, label: str) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = series.notna() & np.isfinite(series) & series.mod(1).eq(0)
    if not valid.all():
        raise ValueError(f"{label} must contain integer-compatible IDs")
    numeric = series.to_numpy(dtype=np.int64)
    if len(np.unique(numeric)) != len(numeric):
        raise ValueError(f"{label} must contain unique IDs")
    return numeric


def _validate_features(
    ids: np.ndarray,
    features: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_ids = _numeric_ids(np.asarray(ids), f"{label} IDs")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or len(numeric_ids) != matrix.shape[0]:
        raise ValueError(f"{label} feature rows must align with IDs")
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{label} features must be finite and non-empty")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"{label} features must have unit norm")
    return numeric_ids, matrix


def _view_by_numeric_id(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "id" not in frame:
        raise ValueError(f"{label} view is missing id")
    result = frame.copy()
    result["_numeric_id"] = _numeric_ids(result["id"].to_numpy(), f"{label} view")
    return result.set_index("_numeric_id", drop=False)


def rank_embeddings(
    *,
    query_ids: np.ndarray,
    query_features: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_features: np.ndarray,
    views: RetrievalViews,
    protocol: Literal["primary", "family"],
    max_k: int = 20,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Rank exact cosine neighbours with protocol exclusions before Top-K."""
    if protocol not in {"primary", "family"}:
        raise ValueError("protocol must be 'primary' or 'family'")
    if isinstance(max_k, bool) or not isinstance(max_k, int) or max_k <= 0:
        raise ValueError("max_k must be a positive integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    numeric_queries, query_matrix = _validate_features(
        query_ids, query_features, "query"
    )
    numeric_gallery, gallery_matrix = _validate_features(
        gallery_ids, gallery_features, "gallery"
    )
    if query_matrix.shape[1] != gallery_matrix.shape[1]:
        raise ValueError("query and gallery feature dimensions must match")

    queries_by_id = _view_by_numeric_id(views.queries, "query")
    gallery_by_id = _view_by_numeric_id(views.gallery, "gallery")
    if set(numeric_queries) != set(queries_by_id.index):
        raise ValueError("query feature IDs must match the retrieval view")
    if set(numeric_gallery) != set(gallery_by_id.index):
        raise ValueError("gallery feature IDs must match the retrieval view")
    aligned_gallery = gallery_by_id.loc[numeric_gallery]

    records: list[dict[str, object]] = []
    for start in range(0, len(numeric_queries), chunk_size):
        stop = min(start + chunk_size, len(numeric_queries))
        distances = 1.0 - query_matrix[start:stop] @ gallery_matrix.T
        distances = np.clip(distances, 0.0, 2.0)
        for offset, query_id in enumerate(numeric_queries[start:stop]):
            row_distances = distances[offset]
            eligible = np.ones(len(numeric_gallery), dtype=bool)
            if protocol == "family":
                eligible = family_candidate_mask(
                    queries_by_id.loc[query_id],
                    aligned_gallery,
                ).to_numpy()
            eligible_indices = np.flatnonzero(eligible)
            if len(eligible_indices) < max_k:
                raise ValueError(
                    f"query {int(query_id)} has fewer than max_k eligible candidates"
                )

            eligible_distances = row_distances[eligible_indices]
            initial = np.argpartition(eligible_distances, max_k - 1)[:max_k]
            threshold = float(eligible_distances[initial].max())
            boundary = eligible_indices[eligible_distances <= threshold]
            ordered = boundary[
                np.lexsort((numeric_gallery[boundary], row_distances[boundary]))
            ][:max_k]
            records.extend(
                {
                    "query_id": int(query_id),
                    "candidate_id": int(numeric_gallery[index]),
                    "distance": float(row_distances[index]),
                }
                for index in ordered
            )

    return prepare_rankings(
        pd.DataFrame.from_records(records),
        views,
        protocol=protocol,
        max_k=max_k,
    )


def rank_probe_embeddings(
    *,
    query_ids: np.ndarray,
    query_features: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_features: np.ndarray,
    views: RetrievalViews,
    protocol: Literal["primary", "family"],
    max_k: int = 20,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Compatibility wrapper for the method-agnostic embedding ranker."""
    return rank_embeddings(
        query_ids=query_ids,
        query_features=query_features,
        gallery_ids=gallery_ids,
        gallery_features=gallery_features,
        views=views,
        protocol=protocol,
        max_k=max_k,
        chunk_size=chunk_size,
    )
