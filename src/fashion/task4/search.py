"""Safe shared query preparation and one-image Task 4 search."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn

from fashion.config import ROOT
from fashion.data.dataset import load_splits
from fashion.task4.gallery_artifact import (
    TeacherGallery,
    load_teacher_gallery_artifact,
)
from fashion.task4.image_safety import (
    reject_protected_image_path,
    reject_protected_image_sha256,
    reject_sealed_image_rows,
)
from fashion.task4.portable_model import load_r5_inference_package
from fashion.task4.preprocessing import (
    PreprocessedImage,
    PreprocessingContract,
    normalize_for_model,
    preprocess_image,
)
from fashion.task4.probe import rank_single_embedding
from fashion.task4.protocol import (
    FIXED_VALIDATION_FOLD,
    build_development_views,
    primary_relevance,
)

SEARCH_RECORD_SCHEMA_VERSION = "1.1.0"
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 20
OUTSIDE_RATINGS = ("good", "mixed", "bad")
_EXPECTED_CONTRACT = PreprocessingContract(240, 320)
_REDACTED_SPLIT_COLUMNS = ("id", "sha256", "partition", "cv_fold")
_PROJECT_HOLDOUT_STATUS = "opened_once_for_final_evaluation_and_now_closed"
_PROJECT_HOLDOUT_RECEIPT = (
    "results/evidence/task4/final_evaluation/unlock_receipt.json"
)
_PROJECT_HOLDOUT_MANIFEST = (
    "results/evidence/task4/final_evaluation/evaluation_manifest.json"
)


@dataclass(frozen=True, slots=True)
class CropBox:
    """Integer crop bounds in Pillow's left, top, right, bottom order."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        coordinates = (self.left, self.top, self.right, self.bottom)
        if (
            any(isinstance(value, bool) or not isinstance(value, Integral) for value in coordinates)
            or self.left < 0
            or self.top < 0
            or self.right <= self.left
            or self.bottom <= self.top
        ):
            raise ValueError(
                "crop coordinates must be non-negative ordered integers"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "right": int(self.right),
            "bottom": int(self.bottom),
        }


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """One checked, oriented, optionally cropped, and preprocessed query."""

    kind: Literal["known", "outside"]
    known_id: int | None
    source_path: Path
    source_sha256: str
    source_dimensions: tuple[int, int]
    effective_dimensions: tuple[int, int]
    crop: CropBox | None
    preprocessed: PreprocessedImage
    aspect_ratio: float
    content_fraction: float
    warnings: tuple[str, ...]
    known_row: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked gallery result."""

    rank: int
    candidate_id: int
    distance: float
    article_type: str
    base_colour: str
    product_display_name: str
    grade: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "candidate_id": int(self.candidate_id),
            "distance": float(self.distance),
            "articleType": self.article_type,
            "baseColour": self.base_colour,
            "productDisplayName": self.product_display_name,
            "grade": None if self.grade is None else int(self.grade),
        }


@dataclass(frozen=True, slots=True)
class SearchRecord:
    """JSON-safe search evidence plus its in-memory prepared query."""

    schema_version: str
    query_key: str
    query: PreparedQuery
    top_k: int
    preprocessing: Mapping[str, object]
    model_identity: Mapping[str, object]
    gallery_identity: Mapping[str, object]
    results: tuple[SearchHit, ...]
    rating: str | None
    note: str | None
    safety: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        query = self.query
        query_payload = {
            "kind": query.kind,
            "known_id": query.known_id,
            "source_sha256": query.source_sha256,
            "source_dimensions": list(query.source_dimensions),
            "effective_dimensions": list(query.effective_dimensions),
            "crop": None if query.crop is None else query.crop.to_dict(),
            "aspect_ratio": float(query.aspect_ratio),
            "content_fraction": float(query.content_fraction),
            "warnings": list(query.warnings),
        }
        return {
            "schema_version": self.schema_version,
            "query_key": self.query_key,
            "query": query_payload,
            "top_k": int(self.top_k),
            "preprocessing": dict(self.preprocessing),
            "model_identity": dict(self.model_identity),
            "gallery_identity": dict(self.gallery_identity),
            "results": [hit.to_dict() for hit in self.results],
            "rating": self.rating,
            "note": self.note,
            "safety": dict(self.safety),
        }


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """In-memory query evidence and ordered gallery display rows."""

    record: SearchRecord
    result_metadata: pd.DataFrame


@dataclass(frozen=True, slots=True)
class SearchBundle:
    """Validated model, gallery, preprocessing, and fixed-fold query state."""

    model: nn.Module
    device: torch.device
    contract: PreprocessingContract
    teacher_mean: tuple[float, float, float]
    teacher_std: tuple[float, float, float]
    model_manifest: Mapping[str, object]
    model_manifest_sha256: str
    gallery: TeacherGallery
    query_catalogue: pd.DataFrame
    splits: pd.DataFrame


def _temporary_sibling(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    return Path(name)


def _query_figure_title(record: SearchRecord) -> str:
    query = record.query
    title = (
        f"KNOWN QUERY {query.known_id}"
        if query.kind == "known"
        else "OUTSIDE QUERY"
    )
    if query.warnings:
        title += f"\nWarnings: {', '.join(query.warnings)}"
    return title


def write_search_outputs(
    response: SearchResponse,
    *,
    figure_directory: str | Path,
    evidence_directory: str | Path,
) -> tuple[Path, Path]:
    """Atomically publish one deterministic PNG and strict JSON evidence pair."""

    from fashion.task4.report_figures import render_search_grid

    payload = (
        json.dumps(
            response.record.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    figure_target = Path(figure_directory)
    evidence_target = Path(evidence_directory)
    figure_target.mkdir(parents=True, exist_ok=True)
    evidence_target.mkdir(parents=True, exist_ok=True)
    stem = response.record.query_key
    final_png = figure_target / f"{stem}.png"
    final_json = evidence_target / f"{stem}.json"
    temporary_png = _temporary_sibling(final_png)
    temporary_json = _temporary_sibling(final_json)
    backup_png: Path | None = None
    first_replaced = False

    try:
        render_search_grid(
            query_pixels=response.record.query.preprocessed.pixels,
            query_title=_query_figure_title(response.record),
            results=response.record.results,
            catalogue=response.result_metadata,
            destination=temporary_png,
        )
        temporary_json.write_bytes(payload)
        if not temporary_png.is_file() or not temporary_json.is_file():
            raise OSError("search output staging did not produce both files")
        json.loads(temporary_json.read_text(encoding="utf-8"))

        if final_png.exists():
            backup_png = _temporary_sibling(final_png)
            shutil.copyfile(final_png, backup_png)
        os.replace(temporary_png, final_png)
        first_replaced = True
        try:
            os.replace(temporary_json, final_json)
        except Exception:
            if backup_png is None:
                final_png.unlink(missing_ok=True)
            else:
                os.replace(backup_png, final_png)
            first_replaced = False
            raise
    finally:
        temporary_png.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
        if backup_png is not None:
            backup_png.unlink(missing_ok=True)

    if not first_replaced:
        raise RuntimeError("search output pair was not published")
    return final_png, final_json


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"portable R5 manifest cannot be read: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("portable R5 manifest must contain one JSON object")
    return payload, raw


def _device(value: str | torch.device) -> torch.device:
    try:
        result = torch.device(value)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid search device: {value}") from error
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("search device must be CPU or CUDA")
    return result


def _teacher_statistics(
    manifest: Mapping[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normalization = manifest.get("normalization")
    teacher = (
        normalization.get("teacher")
        if isinstance(normalization, Mapping)
        else None
    )
    if not isinstance(teacher, Mapping):
        raise ValueError("portable R5 manifest has no teacher normalization")
    values: list[tuple[float, float, float]] = []
    for field in ("mean", "std"):
        raw = teacher.get(field)
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or (field == "std" and float(item) <= 0.0)
                for item in raw
            )
        ):
            raise ValueError(f"teacher normalization {field} is invalid")
        values.append(tuple(float(item) for item in raw))
    return values[0], values[1]


def _checkpoint_sha256(manifest: Mapping[str, object], *, label: str) -> str:
    checkpoint = manifest.get("source_checkpoint")
    if label == "gallery":
        checkpoint = manifest.get("r5_checkpoint")
    digest = checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} checkpoint identity is invalid")
    return digest


def _load_model_snapshot(
    package: Path,
    *,
    manifest: Mapping[str, object],
    manifest_bytes: bytes,
    device: torch.device,
) -> nn.Module:
    weights = manifest.get("weights")
    weights_path = weights.get("path") if isinstance(weights, Mapping) else None
    if weights_path != "weights.pt":
        raise ValueError("portable R5 weight path must be exactly weights.pt")

    with tempfile.TemporaryDirectory(prefix="task4-r5-snapshot-") as temporary:
        snapshot = Path(temporary)
        (snapshot / "manifest.json").write_bytes(manifest_bytes)
        try:
            shutil.copyfile(package / "weights.pt", snapshot / "weights.pt")
        except OSError as error:
            raise ValueError(f"portable R5 weights cannot be copied: {error}") from error
        return load_r5_inference_package(snapshot, device=device)


def load_search_bundle(
    *,
    model_package: str | Path,
    gallery_directory: str | Path,
    splits_path: str | Path,
    device: str | torch.device = "cpu",
) -> SearchBundle:
    """Load and cross-check every artifact needed for repeatable one-image search."""

    selected_device = _device(device)
    package = Path(model_package)
    manifest_path = package / "manifest.json"
    model_manifest, manifest_bytes = _read_manifest(manifest_path)
    model = _load_model_snapshot(
        package,
        manifest=model_manifest,
        manifest_bytes=manifest_bytes,
        device=selected_device,
    )

    expected_contract = _EXPECTED_CONTRACT.to_dict()
    if model_manifest.get("input_contract") != expected_contract:
        raise ValueError("portable R5 input contract must be RGB 240x320 letterbox")
    teacher_mean, teacher_std = _teacher_statistics(model_manifest)

    gallery = load_teacher_gallery_artifact(gallery_directory)
    if _checkpoint_sha256(
        gallery.manifest, label="gallery"
    ) != _checkpoint_sha256(model_manifest, label="model"):
        raise ValueError("gallery checkpoint does not match the portable R5 checkpoint")

    splits = load_splits(splits_path)
    primary, _ = build_development_views(
        splits,
        validation_fold=FIXED_VALIDATION_FOLD,
    )
    expected_gallery_ids = np.sort(
        pd.to_numeric(primary.gallery["id"], errors="raise").to_numpy(dtype=np.int64)
    )
    if not np.array_equal(np.asarray(gallery.ids), expected_gallery_ids):
        raise ValueError("gallery IDs do not match the fixed fold-1 primary gallery")
    reject_sealed_image_rows(gallery.metadata, require_development=True)
    reject_sealed_image_rows(primary.queries, require_development=True)

    missing_redacted = set(_REDACTED_SPLIT_COLUMNS).difference(splits.columns)
    if missing_redacted:
        raise ValueError(
            f"canonical splits are missing safety columns: {sorted(missing_redacted)}"
        )
    redacted_splits = splits.loc[:, _REDACTED_SPLIT_COLUMNS].copy()
    model = model.to(selected_device).eval()
    return SearchBundle(
        model=model,
        device=selected_device,
        contract=_EXPECTED_CONTRACT,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        model_manifest=dict(model_manifest),
        model_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        gallery=gallery,
        query_catalogue=primary.queries.copy(),
        splits=redacted_splits,
    )


def _validate_crop(crop: CropBox | None, dimensions: tuple[int, int]) -> None:
    if crop is None:
        return
    if not isinstance(crop, CropBox):
        raise ValueError("crop must be a CropBox")
    width, height = dimensions
    if crop.right > width or crop.bottom > height:
        raise ValueError("crop must fit inside oriented source bounds")


def _sha256_open_file(source: BinaryIO) -> str:
    source.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def _open_prepared_image(
    source: BinaryIO,
    *,
    crop: CropBox | None,
    contract: PreprocessingContract,
) -> tuple[tuple[int, int], tuple[int, int], PreprocessedImage]:
    try:
        with Image.open(source) as image:
            oriented = ImageOps.exif_transpose(image)
            source_dimensions = (int(oriented.width), int(oriented.height))
            if source_dimensions[0] <= 0 or source_dimensions[1] <= 0:
                raise ValueError("source image must have positive dimensions")
            _validate_crop(crop, source_dimensions)
            effective = (
                oriented
                if crop is None
                else oriented.crop(
                    (crop.left, crop.top, crop.right, crop.bottom)
                )
            )
            effective.load()
            effective_dimensions = (int(effective.width), int(effective.height))
            transformed = preprocess_image(effective, contract)
    except OSError as error:
        raise ValueError(f"query image cannot be decoded: {error}") from error
    return source_dimensions, effective_dimensions, transformed


def _known_query_row(bundle: SearchBundle, query_id: int) -> dict[str, object]:
    if isinstance(query_id, bool) or not isinstance(query_id, Integral):
        raise ValueError("known query ID must be an integer")
    numeric_id = int(query_id)
    catalogue_ids = pd.to_numeric(
        bundle.query_catalogue["id"], errors="coerce"
    )
    matches = bundle.query_catalogue.loc[catalogue_ids.eq(numeric_id)]
    if len(matches) == 1:
        return dict(matches.iloc[0])
    if len(matches) > 1:
        raise ValueError("fixed fold-1 query catalogue contains duplicate IDs")

    split_ids = pd.to_numeric(bundle.splits["id"], errors="coerce")
    split_matches = bundle.splits.loc[split_ids.eq(numeric_id)]
    if not split_matches.empty:
        raise ValueError(f"known query ID {numeric_id} is not in fixed fold 1")
    raise ValueError(f"unknown known query ID: {numeric_id}")


def _resolved_canonical_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _warnings(
    *,
    aspect_ratio: float,
    content_fraction: float,
) -> tuple[str, ...]:
    return tuple(
        name
        for name, active in (
            (
                "unusual_aspect_ratio",
                not math.isclose(
                    aspect_ratio,
                    0.75,
                    rel_tol=0.0,
                    abs_tol=0.01,
                ),
            ),
            ("heavy_letterbox_padding", content_fraction < 0.75),
        )
        if active
    )


def _prepare_query(
    bundle: SearchBundle,
    *,
    image_path: str | Path | None,
    query_id: int | None,
    crop: CropBox | None,
) -> PreparedQuery:
    known_row: dict[str, object] | None = None
    if image_path is not None:
        source_path = reject_protected_image_path(image_path)
        kind: Literal["known", "outside"] = "outside"
        known_id = None
    else:
        if query_id is None:
            raise ValueError("known query ID is required")
        known_row = _known_query_row(bundle, query_id)
        reject_sealed_image_rows(
            pd.DataFrame([known_row]),
            require_development=True,
        )
        source_path = reject_protected_image_path(
            _resolved_canonical_path(known_row["path"])
        )
        kind = "known"
        known_id = int(query_id)

    try:
        with source_path.open("rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError(f"{kind} query path must be a regular file")
            source_sha256 = _sha256_open_file(source)
            if known_row is not None:
                expected_sha256 = str(known_row["sha256"]).lower()
                if source_sha256 != expected_sha256:
                    raise ValueError(
                        "known query image SHA-256 does not match canonical splits"
                    )
            reject_protected_image_sha256(source_sha256, bundle.splits)
            source_dimensions, effective_dimensions, transformed = (
                _open_prepared_image(
                    source,
                    crop=crop,
                    contract=bundle.contract,
                )
            )
    except OSError as error:
        raise ValueError(f"{kind} query path must be a regular readable file") from error
    effective_width, effective_height = effective_dimensions
    aspect_ratio = effective_width / effective_height
    content_fraction = float(transformed.content_mask.mean())
    return PreparedQuery(
        kind=kind,
        known_id=known_id,
        source_path=source_path,
        source_sha256=source_sha256,
        source_dimensions=source_dimensions,
        effective_dimensions=effective_dimensions,
        crop=crop,
        preprocessed=transformed,
        aspect_ratio=aspect_ratio,
        content_fraction=content_fraction,
        warnings=_warnings(
            aspect_ratio=aspect_ratio,
            content_fraction=content_fraction,
        ),
        known_row=known_row,
    )


def _validate_run_arguments(
    *,
    image_path: str | Path | None,
    query_id: int | None,
    top_k: int,
    rating: str | None,
    note: str | None,
) -> None:
    if (image_path is None) == (query_id is None):
        raise ValueError("exactly one of image_path and query_id is required")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, Integral)
        or not MIN_TOP_K <= int(top_k) <= MAX_TOP_K
    ):
        raise ValueError("top_k must be an integer from 1 through 20")

    if query_id is not None:
        if rating is not None or note is not None:
            raise ValueError("rating and note are not allowed for known queries")
        return
    if (rating is None) != (note is None):
        raise ValueError("rating and note must both be absent or both be present")
    if rating is not None and rating not in OUTSIDE_RATINGS:
        raise ValueError("outside rating must be good, mixed, or bad")
    if note is not None and (
        not isinstance(note, str) or not note.strip()
    ):
        raise ValueError("outside note cannot be blank")


def _encode_query(bundle: SearchBundle, query: PreparedQuery) -> np.ndarray:
    normalized = normalize_for_model(
        query.preprocessed,
        mean=bundle.teacher_mean,
        std=bundle.teacher_std,
    )
    tensor = torch.from_numpy(
        np.ascontiguousarray(normalized.transpose(2, 0, 1))
    ).unsqueeze(0).to(bundle.device)
    with torch.inference_mode():
        encoded = bundle.model.encode(tensor)
    if not isinstance(encoded, torch.Tensor):
        raise ValueError("model encode must return a tensor")
    embedding = encoded.detach().cpu().numpy()
    if embedding.shape != (1, 128):
        raise ValueError("model must return one 128-value embedding")
    query_embedding = np.asarray(embedding[0], dtype=np.float32)
    if not np.isfinite(query_embedding).all():
        raise ValueError("model embedding must be finite")
    if not np.isclose(
        np.linalg.norm(query_embedding),
        1.0,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("model embedding must have unit norm")
    return query_embedding


def _ranked_metadata(
    bundle: SearchBundle,
    embedding: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    rankings = rank_single_embedding(
        query_feature=embedding,
        gallery_ids=bundle.gallery.ids,
        gallery_features=bundle.gallery.features,
        max_k=top_k,
    )
    metadata = bundle.gallery.metadata.copy()
    metadata["id"] = pd.to_numeric(metadata["id"], errors="raise").astype(np.int64)
    result = rankings.merge(
        metadata,
        left_on="candidate_id",
        right_on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if result["id"].isna().any():
        raise ValueError("ranked candidate metadata is missing")
    return result.sort_values("rank", kind="stable").reset_index(drop=True)


def _query_key(query: PreparedQuery, top_k: int) -> str:
    identity = {
        "kind": query.kind,
        "known_id": query.known_id,
        "source_sha256": None if query.kind == "known" else query.source_sha256,
        "crop": None if query.crop is None else query.crop.to_dict(),
        "top_k": top_k,
    }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()[:12]
    if query.kind == "known":
        return f"known-{query.known_id}-{digest}"
    return f"outside-{query.source_sha256[:12]}-{digest}"


def _model_identity(bundle: SearchBundle) -> dict[str, object]:
    manifest = bundle.model_manifest
    weights = manifest.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("portable R5 weight identity is missing")
    return {
        "method": manifest.get("method"),
        "architecture": manifest.get("architecture"),
        "manifest_sha256": bundle.model_manifest_sha256,
        "source_checkpoint": dict(manifest.get("source_checkpoint", {})),
        "weights": dict(weights),
    }


def _gallery_identity(bundle: SearchBundle) -> dict[str, object]:
    manifest = bundle.gallery.manifest
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        not isinstance(record, Mapping) for record in files.values()
    ):
        raise ValueError("gallery payload file identities are missing")
    return {
        "artifact_identity_sha256": bundle.gallery.identity_sha256,
        "r5_checkpoint": dict(manifest.get("r5_checkpoint", {})),
        "fold": manifest.get("fold"),
        "rows": len(bundle.gallery.ids),
        "files": {
            str(filename): dict(record)
            for filename, record in files.items()
        },
    }


def run_search(
    bundle: SearchBundle,
    *,
    image_path: str | Path | None = None,
    query_id: int | None = None,
    crop: CropBox | None = None,
    top_k: int = DEFAULT_TOP_K,
    rating: str | None = None,
    note: str | None = None,
) -> SearchResponse:
    """Prepare, encode, and rank one safe outside or fixed fold-1 query."""

    _validate_run_arguments(
        image_path=image_path,
        query_id=query_id,
        top_k=top_k,
        rating=rating,
        note=note,
    )
    query = _prepare_query(
        bundle,
        image_path=image_path,
        query_id=query_id,
        crop=crop,
    )
    embedding = _encode_query(bundle, query)
    result_metadata = _ranked_metadata(bundle, embedding, int(top_k))

    grades: list[int | None]
    if query.known_row is None:
        grades = [None] * len(result_metadata)
    else:
        grades = [
            int(value)
            for value in primary_relevance(
                pd.Series(query.known_row),
                result_metadata,
            )
        ]
    hits = tuple(
        SearchHit(
            rank=int(row.rank),
            candidate_id=int(row.candidate_id),
            distance=float(row.distance),
            article_type=str(row.articleType),
            base_colour=str(row.baseColour),
            product_display_name=str(row.productDisplayName),
            grade=grades[index],
        )
        for index, row in enumerate(result_metadata.itertuples(index=False))
    )
    record = SearchRecord(
        schema_version=SEARCH_RECORD_SCHEMA_VERSION,
        query_key=_query_key(query, int(top_k)),
        query=query,
        top_k=int(top_k),
        preprocessing={
            "contract": bundle.contract.to_dict(),
            "normalization": {
                "source": "teacher",
                "mean": list(bundle.teacher_mean),
                "std": list(bundle.teacher_std),
            },
            "neutral_mask_padding": True,
        },
        model_identity=_model_identity(bundle),
        gallery_identity=_gallery_identity(bundle),
        results=hits,
        rating=rating,
        note=note,
        safety={
            "path_checked_before_decode": True,
            "hash_checked_before_decode": True,
            "known_queries_fixed_fold_1": True,
            "demo_accessed_protected_images": False,
            "demo_accessed_holdout": False,
            "demo_used_holdout_gallery": False,
            "demo_accessed_quarantine": False,
            "demo_accessed_official_teacher_test": False,
            "project_holdout_status": _PROJECT_HOLDOUT_STATUS,
            "project_holdout_receipt": _PROJECT_HOLDOUT_RECEIPT,
            "project_holdout_manifest": _PROJECT_HOLDOUT_MANIFEST,
        },
    )
    return SearchResponse(record=record, result_metadata=result_metadata)


__all__ = (
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "MIN_TOP_K",
    "OUTSIDE_RATINGS",
    "SEARCH_RECORD_SCHEMA_VERSION",
    "CropBox",
    "PreparedQuery",
    "SearchBundle",
    "SearchHit",
    "SearchRecord",
    "SearchResponse",
    "load_search_bundle",
    "run_search",
    "write_search_outputs",
)
