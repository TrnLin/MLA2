"""Deterministic development-only figures cited by the Task 4 comparison report.

Every value plotted here is read from the strict frozen comparison bundle or from a
hash-checked tracked evidence artifact. Nothing is recomputed from model weights and
no sealed partition is opened.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import AxesImage
from PIL import Image

from fashion.config import ROOT
from fashion.data.dataset import load_splits
from fashion.data.splits import PROTECTED_PARTITIONS
from fashion.task4.final_freeze import (
    FINAL_FREEZE_RELATIVE_PATH,
    validate_final_comparison_bundle,
)
from fashion.task4.preprocessing import PreprocessingContract, preprocess_image
from fashion.task4.preprocessing_experiment import build_odd_aspect_canvas
from fashion.task4.protocol import primary_relevance

__all__ = (
    "FIGURE_CAPTIONS",
    "FIGURE_DIR_RELATIVE",
    "PANEL_COUNT",
    "REPORT_FIGURE_NAMES",
    "RETRIEVAL_TOP_K",
    "RetrievalPanel",
    "RetrievalResult",
    "WinnerRetrievalEvidence",
    "build_chart_figures",
    "build_report_figures",
    "build_retrieval_figures",
    "development_catalogue",
    "graded_relevance",
    "load_winner_retrieval_evidence",
    "resolve_image_rows",
    "same_family",
    "select_retrieval_panels",
    "slice_retrieval_panels",
)

FIGURE_DIR_RELATIVE = Path("results/figures/task4/final")
VARIANT_INDEX_RELATIVE = Path("data/processed/task4/external_variant_index.csv.gz")
RETRIEVAL_TOP_K = 10
PANEL_COUNT = 5
SLICE_TOP_K = 5

CONTRACT = PreprocessingContract(width=240, height=320)
_TEACHER_TEST_MARKERS = ("teacher/test", "images_test", "styles_prediction")
_SLICE_ORDER = (
    "normal_success",
    "grayscale",
    "unusual_geometry",
    "canvas_failure",
    "weak_family",
    "family_unavailable",
    "rare_article_type",
    "rare_type_colour",
)
_PRIMARY_SLICES = (
    "grayscale",
    "rare_article_type",
    "rare_type_colour",
    "unusual_geometry",
)
_FAMILY_SLICES = ("weak_family", "family_unavailable")
_CANVAS_VARIANTS = ("clean", "wide", "tall")
_GALLERY_LABELS = {
    "teacher": "Teacher-only",
    "v1": "V1-only",
    "two_view": "Two-view collapse",
}

_INK = "#16181d"
_MUTED = "#6b7280"
_BAR = "#4b5563"
_BAR_STAR = "#1f2937"
_BAR_BASE = "#9aa2ae"
_BAR_BLOCKED = "#8a8f99"
_GRADE_COLOURS = {2: "#1f5132", 1: "#b3801f", 0: "#7d2727"}
_GRADE_MARKS = {2: "hit", 1: "part", 0: "miss"}
_GRADE_LABELS = {
    2: "'hit' — grade 2, same article type and same base colour",
    1: "'part' — grade 1, same article type, different base colour",
    0: "'miss' — grade 0, different article type (incorrect)",
}
_HEAT = LinearSegmentedColormap.from_list(
    "task4_heat",
    ["#f7e3e3", "#e9c9a8", "#cfd8bd", "#8fb99f", "#3f7a5a"],
)
_HEAT_VMAX = 0.6
_HEAT_TEXT_SWITCH = 0.50

REPORT_FIGURE_NAMES: tuple[str, ...] = (
    "method_quality_comparison.png",
    "stability_folds.png",
    "gallery_policy_comparison.png",
    "cost_quality_tradeoff.png",
    "failure_slice_profile.png",
    "canvas_robustness.png",
    "r5_retrieval_success.png",
    "r5_retrieval_failure.png",
    "r5_retrieval_slices.png",
)
CHART_FIGURE_NAMES: tuple[str, ...] = REPORT_FIGURE_NAMES[:6]
RETRIEVAL_FIGURE_NAMES: tuple[str, ...] = REPORT_FIGURE_NAMES[6:]

FIGURE_CAPTIONS: dict[str, str] = {
    "method_quality_comparison.png": (
        "Same-source bars are the equal mean of teacher-to-teacher and V1-to-V1 "
        "query-mean linear nDCG@10; cross-source bars are the equal mean of the two "
        "swapped directions. Source: strict final freeze bundle for R1-R5 and B1, "
        "hash-checked HOG and HOG+HSV-edge fusion manifests, and the frozen "
        "spatial HSV-edge and random-floor baseline summary. Development fold 1 only."
    ),
    "stability_folds.png": (
        "One point per fresh from-scratch run; the line is the five-fold mean and the "
        "band is plus or minus one sample SD. Source: strict final freeze bundle "
        "stability summaries. Development validation folds 0-4 only."
    ),
    "gallery_policy_comparison.png": (
        "Quality is the equal mean of teacher and V1 query nDCG@10 at K=10, latency is "
        "the CPU batch-one p95 end-to-end second, storage is decimal MB; the quality "
        "axis is zoomed because the three policies differ by well under one thousandth. "
        "Source: strict final freeze bundle three-policy gallery study on the selected "
        "R5 checkpoint."
    ),
    "cost_quality_tradeoff.png": (
        "Left panel plots same-source mean nDCG@10 against the worst of the four measured "
        "CPU batch-one p95 end-to-end seconds, with marker area proportional to "
        "single-source index MB; the two right panels give the same index MB and the "
        "parameter count exactly. Source: strict final freeze bundle costs plus the "
        "hash-checked HOG and fusion cost records."
    ),
    "failure_slice_profile.png": (
        "Left cells are Protocol A query-mean nDCG@10 per slice; right cells are "
        "Protocol B query-mean Recall@10; cross-hatched cells are undefined at zero "
        "coverage and coverage is printed only when it is below 100 percent. Source: "
        "strict final freeze bundle failure slices plus hash-checked HOG and fusion "
        "slice artifacts, teacher-to-teacher direction only."
    ),
    "canvas_robustness.png": (
        "Left bars are V1-to-V1 query-mean nDCG@10 per canvas variant; right bars are "
        "the mean Top-10 overlap with the clean ranking, where the clean canvas is its "
        "own reference at exactly 1 and is therefore not drawn. Source: strict final "
        "freeze bundle canvas evidence plus hash-checked HOG and fusion canvas "
        "summaries."
    ),
    "r5_retrieval_success.png": (
        "The five highest query-mean nDCG@10 V1-to-V1 development queries with a distinct "
        "article type, each with its exact-cosine Top-10. Border colour is the frozen "
        "Protocol A graded relevance. Source: hash-checked R5 rankings and per-query "
        "metrics artifacts; development partition images only."
    ),
    "r5_retrieval_failure.png": (
        "The five lowest scorable query-mean nDCG@10 V1-to-V1 development queries with a "
        "distinct article type, each with its exact-cosine Top-10. Border colour is the "
        "frozen Protocol A graded relevance. Source: hash-checked R5 rankings and "
        "per-query metrics artifacts; development partition images only."
    ),
    "r5_retrieval_slices.png": (
        "The eight pre-registered slice examples, one query and its Top-5 each, including "
        "the wide-canvas failure query drawn on the canvas the model actually received. "
        "Border colour is the frozen Protocol A graded relevance and the mean per-query "
        "value is printed per row. Source: hash-checked R5 examples artifact; development "
        "partition images only."
    ),
}


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One returned product with its frozen graded relevance."""

    rank: int
    candidate_id: int
    distance: float
    grade: int
    same_family: bool


@dataclass(frozen=True, slots=True)
class RetrievalPanel:
    """One query row of a retrieval-prediction figure."""

    label: str
    query_id: int
    query_variant: str
    article_type: str
    base_colour: str
    metric: str
    value: float | None
    results: tuple[RetrievalResult, ...]


@dataclass(frozen=True, slots=True)
class WinnerRetrievalEvidence:
    """Hash-checked ranking evidence for the selected deployment method."""

    method: str
    run_id: str
    rankings: pd.DataFrame
    query_metrics: pd.DataFrame
    examples: pd.DataFrame
    catalogue: pd.DataFrame


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def _bundle(root: Path) -> dict[str, Any]:
    return validate_final_comparison_bundle(
        root / FINAL_FREEZE_RELATIVE_PATH,
        root=root,
    )


def _hash_checked_artifact(
    manifest_path: Path,
    manifest: Mapping[str, object],
    name: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        records = [
            {"name": key, **value}
            for key, value in artifacts.items()
            if isinstance(value, Mapping)
        ]
    elif isinstance(artifacts, list):
        records = [item for item in artifacts if isinstance(item, Mapping)]
    else:
        raise ValueError(f"{manifest_path.name} artifact records are malformed")
    matches = [item for item in records if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"{manifest_path.name} lacks exactly one {name} artifact")
    record = matches[0]
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{manifest_path.name} {name} path must stay relative")
    candidate = manifest_path.parent / relative
    if not candidate.is_file():
        candidate = ROOT / relative
    if not candidate.is_file():
        raise ValueError(f"{manifest_path.name} {name} artifact is missing")
    if _sha256_file(candidate) != str(record["sha256"]):
        raise ValueError(f"{manifest_path.name} {name} artifact hash disagrees")
    return candidate


def _finite_bounded(value: object, *, label: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} is outside the valid 0-1 range")
    return number


def development_catalogue(*, root: Path = ROOT) -> pd.DataFrame:
    """Return the development-only product catalogue used to draw images."""

    splits = load_splits(root / "data/processed/splits.csv")
    variants = pd.read_csv(
        root / VARIANT_INDEX_RELATIVE,
        usecols=["id", "partition", "teacher_path", "external_path"],
        keep_default_na=False,
    )
    labels = splits.loc[
        :,
        ["id", "partition", "articleType", "baseColour", "product_family_group"],
    ].copy()
    labels["id"] = labels["id"].astype(int)
    variants["id"] = variants["id"].astype(int)
    merged = labels.merge(
        variants.drop(columns="partition"),
        on="id",
        how="inner",
        validate="one_to_one",
    )
    catalogue = merged.loc[merged["partition"].eq("development")].reset_index(drop=True)
    if catalogue.empty:
        raise ValueError("development catalogue must not be empty")
    _reject_sealed(catalogue)
    return catalogue


def _reject_sealed(frame: pd.DataFrame) -> None:
    if "partition" not in frame:
        raise ValueError("image rows must carry canonical partition values")
    protected = frame["partition"].isin(list(PROTECTED_PARTITIONS))
    if protected.any():
        ids = frame.loc[protected, "id"].astype(str).head(5).tolist()
        raise ValueError(f"sealed holdout/quarantine rows reached image access: {ids}")
    if not frame["partition"].eq("development").all():
        raise ValueError("sealed or unknown partition rows reached image access")
    for column in (name for name in frame.columns if str(name).endswith("_path")):
        lowered = frame[column].astype(str).str.replace("\\", "/", regex=False).str.lower()
        if lowered.map(
            lambda value: any(marker in value for marker in _TEACHER_TEST_MARKERS)
        ).any():
            raise ValueError("official teacher-test path reached image access")


def resolve_image_rows(
    catalogue: pd.DataFrame,
    ids: Iterable[int],
) -> pd.DataFrame:
    """Return catalogue rows for ``ids`` after refusing sealed or unknown rows."""

    wanted = [int(value) for value in ids]
    lookup = catalogue.set_index(catalogue["id"].astype(int), drop=False)
    if missing := sorted(set(wanted).difference(lookup.index)):
        raise ValueError(f"image rows are missing IDs: {missing}")
    rows = lookup.loc[wanted].reset_index(drop=True)
    _reject_sealed(rows)
    return rows


def graded_relevance(query: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    """Return the frozen Protocol A graded relevance for ``candidates``."""

    return primary_relevance(query, candidates)


def same_family(query: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    """Mark candidates sharing the query's product family group."""

    return (
        candidates["product_family_group"]
        .eq(query["product_family_group"])
        .to_numpy(dtype=bool)
    )


def load_winner_retrieval_evidence(*, root: Path = ROOT) -> WinnerRetrievalEvidence:
    """Open the selected method's hash-checked ranking evidence."""

    bundle = _bundle(root)
    method = str(bundle["decision"]["method"])
    entry = next(
        item for item in bundle["methods"] if str(item["method"]) == method
    )
    run_id = str(entry["run_id"])
    manifest_path = (
        root / "results/evidence/task4/learned" / run_id / "manifest.json"
    )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("run_id") != run_id
        or manifest.get("method") != method
        or manifest.get("scope") != "development"
        or manifest.get("split_fingerprint") != bundle["split"]["fingerprint"]
    ):
        raise ValueError("winner manifest identity does not match the strict freeze")
    if manifest.get("selected_metrics") != entry["selected_metrics"]:
        raise ValueError("winner manifest metrics do not match the strict freeze")

    frames = {
        name: pd.read_csv(_hash_checked_artifact(manifest_path, manifest, name))
        for name in ("rankings", "query_metrics", "examples")
    }
    frozen_first = {
        (str(row["slice"]), int(row["candidate_id"]))
        for row in entry["examples"]
    }
    examples = frames["examples"]
    observed_first = {
        (str(row.slice), int(row.candidate_id))
        for row in examples.loc[examples["rank"].eq(1)].itertuples(index=False)
    }
    if frozen_first != observed_first:
        raise ValueError("winner examples artifact disagrees with the strict freeze")
    return WinnerRetrievalEvidence(
        method=method,
        run_id=run_id,
        rankings=frames["rankings"],
        query_metrics=frames["query_metrics"],
        examples=examples,
        catalogue=development_catalogue(root=root),
    )


def _panel_from_rows(
    *,
    label: str,
    query_id: int,
    query_variant: str,
    metric: str,
    value: float | None,
    ranked: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> RetrievalPanel:
    ordered = ranked.sort_values("rank", kind="mergesort")
    query_row = resolve_image_rows(catalogue, [query_id]).iloc[0]
    candidates = resolve_image_rows(
        catalogue,
        ordered["candidate_id"].astype(int).tolist(),
    )
    grades = graded_relevance(query_row, candidates)
    families = same_family(query_row, candidates)
    results = tuple(
        RetrievalResult(
            rank=int(row.rank),
            candidate_id=int(row.candidate_id),
            distance=float(row.distance),
            grade=int(grade),
            same_family=bool(family),
        )
        for row, grade, family in zip(
            ordered.itertuples(index=False),
            grades,
            families,
            strict=True,
        )
    )
    return RetrievalPanel(
        label=label,
        query_id=query_id,
        query_variant=query_variant,
        article_type=str(query_row["articleType"]),
        base_colour=str(query_row["baseColour"]),
        metric=metric,
        value=value,
        results=results,
    )


def select_retrieval_panels(
    evidence: WinnerRetrievalEvidence,
    *,
    kind: str,
    top_k: int = RETRIEVAL_TOP_K,
    limit: int = PANEL_COUNT,
) -> tuple[RetrievalPanel, ...]:
    """Deterministically pick the best or worst scorable V1-to-V1 queries."""

    if kind not in {"success", "failure"}:
        raise ValueError("kind must be 'success' or 'failure'")
    metrics = evidence.query_metrics
    scorable = metrics.loc[
        metrics["query_source"].eq("v1")
        & metrics["gallery_source"].eq("v1")
        & metrics["protocol"].eq("primary")
        & metrics["ndcg_at_10"].notna(),
        ["query_id", "articleType", "ndcg_at_10"],
    ].copy()
    if scorable.empty:
        raise ValueError("winner evidence has no scorable V1-to-V1 primary queries")
    scorable["query_id"] = scorable["query_id"].astype(int)
    ordered = scorable.sort_values(
        ["ndcg_at_10", "query_id"],
        ascending=[kind == "failure", True],
        kind="mergesort",
    )
    rankings = evidence.rankings
    ranked = rankings.loc[
        rankings["query_source"].eq("v1")
        & rankings["gallery_source"].eq("v1")
        & rankings["protocol"].eq("primary")
        & rankings["rank"].le(top_k)
    ]
    grouped = dict(tuple(ranked.groupby("query_id", sort=False)))

    panels: list[RetrievalPanel] = []
    seen_types: set[str] = set()
    for row in ordered.itertuples(index=False):
        article_type = str(row.articleType)
        if article_type in seen_types:
            continue
        candidate_rows = grouped.get(int(row.query_id))
        if candidate_rows is None or len(candidate_rows) != top_k:
            continue
        seen_types.add(article_type)
        panels.append(
            _panel_from_rows(
                label=kind,
                query_id=int(row.query_id),
                query_variant="clean",
                metric="ndcg_at_10",
                value=float(row.ndcg_at_10),
                ranked=candidate_rows,
                catalogue=evidence.catalogue,
            )
        )
        if len(panels) == limit:
            break
    if len(panels) != limit:
        raise ValueError(f"could not select {limit} distinct {kind} retrieval panels")
    return tuple(panels)


def slice_retrieval_panels(
    evidence: WinnerRetrievalEvidence,
) -> tuple[RetrievalPanel, ...]:
    """Return the pre-registered slice examples as retrieval panels."""

    examples = evidence.examples
    observed = set(examples["slice"].astype(str))
    if observed != set(_SLICE_ORDER):
        raise ValueError("winner examples do not cover the pre-registered slices")
    panels: list[RetrievalPanel] = []
    for name in _SLICE_ORDER:
        rows = examples.loc[examples["slice"].astype(str).eq(name)]
        if len(rows) != SLICE_TOP_K:
            raise ValueError(f"slice {name} must persist exactly Top-{SLICE_TOP_K}")
        first = rows.sort_values("rank", kind="mergesort").iloc[0]
        raw = first["value"]
        panels.append(
            _panel_from_rows(
                label=name,
                query_id=int(first["query_id"]),
                query_variant=str(first["query_variant"]),
                metric=str(first["metric"]),
                value=None if pd.isna(raw) else float(raw),
                ranked=rows,
                catalogue=evidence.catalogue,
            )
        )
    return tuple(panels)


def _untrained_rows(root: Path) -> list[dict[str, Any]]:
    definitions = (
        (
            "HOG + HSV-edge fusion",
            "HOG+HSV",
            "hog_fusion",
            "hog-plus-spatial-hsv-edge-equal-v1",
        ),
        ("HOG", "HOG", "hog", "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1"),
    )
    rows: list[dict[str, Any]] = []
    for label, short, folder, method in definitions:
        manifest_path = root / "results/evidence/task4" / folder / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("method") != method or manifest.get("scope") != "development":
            raise ValueError(f"{folder} manifest identity is invalid")
        if manifest.get("holdout_opened") or manifest.get("quarantine_opened"):
            raise ValueError(f"{folder} manifest is not a sealed development artifact")
        quality = pd.read_csv(
            _hash_checked_artifact(manifest_path, manifest, "quality_summary")
        )
        canvas = pd.read_csv(
            _hash_checked_artifact(manifest_path, manifest, "canvas_summary")
        )
        slices = pd.read_csv(
            _hash_checked_artifact(manifest_path, manifest, "failure_slices")
        )
        cost = _read_json(_hash_checked_artifact(manifest_path, manifest, "cost"))
        rows.append(
            {
                "label": label,
                "short": short,
                "note": "untrained baseline",
                "method": method,
                "eligibility": "untrained baseline",
                "same_source": _direction_mean(quality, method, same=True),
                "cross_source": _direction_mean(quality, method, same=False),
                "worst_p95": _worst_end_to_end_p95(cost["timing_summary"]),
                "index_mb": max(
                    float(item["index_bytes"])
                    for item in cost["per_source_index_cost"].values()
                )
                / 1_000_000.0,
                "parameters": int(cost["parameters"]),
                "canvas": _canvas_rows(canvas),
                "slices": slices,
            }
        )
    return rows


def _direction_mean(quality: pd.DataFrame, method: str, *, same: bool) -> float:
    directions = (
        (("teacher", "teacher"), ("v1", "v1"))
        if same
        else (("teacher", "v1"), ("v1", "teacher"))
    )
    values = []
    for query_source, gallery_source in directions:
        selected = quality.loc[
            quality["method"].eq(method)
            & quality["query_source"].eq(query_source)
            & quality["gallery_source"].eq(gallery_source)
            & quality["protocol"].eq("primary")
            & quality["metric"].eq("ndcg")
            & quality["k"].eq(10)
            & quality["aggregation"].eq("query_mean"),
            "value",
        ]
        if len(selected) != 1:
            raise ValueError(f"{method} lacks one {query_source}->{gallery_source} nDCG@10")
        values.append(_finite_bounded(selected.iloc[0], label=f"{method} nDCG@10"))
    return float(np.mean(values))


def _worst_end_to_end_p95(timing_summary: Sequence[Mapping[str, object]]) -> float:
    values = [
        float(row["value_seconds"])
        for row in timing_summary
        if row.get("metric") == "end_to_end" and row.get("percentile") == "p95"
    ]
    if len(values) != 4:
        raise ValueError("cost record needs four end-to-end p95 directions")
    return max(values)


def _canvas_rows(canvas: pd.DataFrame) -> dict[str, dict[str, float]]:
    selected = canvas.loc[
        canvas["query_source"].eq("v1") & canvas["gallery_source"].eq("v1")
    ]
    rows = {
        str(row.query_variant): {
            "ndcg_at_10": _finite_bounded(row.ndcg_at_10, label="canvas nDCG@10"),
            "mean_top10_overlap": _finite_bounded(
                row.mean_top10_overlap,
                label="canvas overlap",
            ),
        }
        for row in selected.itertuples(index=False)
    }
    if set(rows) != set(_CANVAS_VARIANTS):
        raise ValueError("canvas summary must cover clean, wide, and tall")
    return rows


def _frozen_baseline_rows(root: Path) -> tuple[dict[str, Any], float]:
    summary = pd.read_csv(root / "results/evidence/task4/baseline_summary.csv")
    selected = summary.loc[
        summary["protocol"].eq("primary")
        & summary["metric"].eq("ndcg")
        & summary["k"].eq(10)
        & summary["aggregation"].eq("query_mean")
    ]
    hsv = {
        "label": "Spatial HSV-edge",
        "short": "HSV-edge",
        "note": "untrained baseline",
        "method": "spatial-hsv-edge-4x4-v2",
        "eligibility": "untrained baseline",
        "same_source": _direction_mean(selected, "spatial-hsv-edge-4x4-v2", same=True),
        "cross_source": _direction_mean(selected, "spatial-hsv-edge-4x4-v2", same=False),
    }
    floor = selected.loc[
        selected["method"].eq("random-seed-2753")
        & selected["query_source"].eq("random")
        & selected["gallery_source"].eq("random"),
        "value",
    ]
    if len(floor) != 1:
        raise ValueError("frozen baseline summary lacks one random-floor row")
    return hsv, _finite_bounded(floor.iloc[0], label="random floor")


_LEARNED_DESCRIPTIONS = {
    "R1": "scratch ResNet-18 + VICReg",
    "R2": "scratch ResNet-34 + VICReg",
    "R3": "R1 + geometry/canvas augmentation",
    "R4": "R3 + family triplet loss",
    "R5": "scratch conv. autoencoder",
    "B1": "pretrained ResNet-18 + VICReg",
}


def _learned_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in bundle["methods"]:
        metrics = item["selected_metrics"]
        cost = item["cost"]
        rows.append(
            {
                "label": str(item["method"]),
                "short": str(item["method"]),
                "note": _LEARNED_DESCRIPTIONS[str(item["method"])],
                "method": str(item["method"]),
                "eligibility": (
                    "pretrained comparison only"
                    if item["pretrained"]
                    else "scratch candidate"
                ),
                "same_source": _finite_bounded(
                    metrics["development_winner_score"],
                    label="development winner score",
                ),
                "cross_source": _finite_bounded(
                    metrics["cross_source_score"],
                    label="cross-source score",
                ),
                "worst_p95": _worst_end_to_end_p95(cost["timing_summary"]),
                "index_mb": max(cost["index_bytes"].values()) / 1_000_000.0,
                "parameters": int(cost["parameters"]),
                "canvas": {
                    str(row["query_variant"]): {
                        "ndcg_at_10": _finite_bounded(
                            row["ndcg_at_10"],
                            label="canvas nDCG@10",
                        ),
                        "mean_top10_overlap": _finite_bounded(
                            row["mean_top10_overlap"],
                            label="canvas overlap",
                        ),
                    }
                    for row in item["canvas"]
                },
                "slices": item["failure_slices"],
            }
        )
    return rows


def _style(axes: plt.Axes) -> None:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(_MUTED)
    axes.tick_params(colors=_INK, labelsize=8)
    axes.grid(True, axis="both", color="#e8eaee", linewidth=0.7)
    axes.set_axisbelow(True)


def _save(figure: plt.Figure, destination: Path, name: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / name
    figure.savefig(output, dpi=170, metadata={"Software": "MLA2"})
    plt.close(figure)
    return output


def _method_quality_figure(
    rows: Sequence[Mapping[str, Any]],
    floor: float,
    destination: Path,
) -> Path:
    ordered = sorted(rows, key=lambda row: row["same_source"])
    positions = np.arange(len(ordered), dtype=float)
    figure, axes = plt.subplots(figsize=(9.6, 6.4), constrained_layout=True)
    _style(axes)
    height = 0.38
    for offset, key, colour, label in (
        (height / 2, "same_source", _BAR, "Same-source mean (teacher-to-teacher, V1-to-V1)"),
        (
            -height / 2,
            "cross_source",
            _BAR_BASE,
            "Cross-source mean (teacher-to-V1, V1-to-teacher)",
        ),
    ):
        values = [row[key] for row in ordered]
        bars = axes.barh(
            positions + offset,
            values,
            height=height,
            color=colour,
            edgecolor=_INK,
            linewidth=0.4,
            label=label,
        )
        for bar, row in zip(bars, ordered, strict=True):
            if row["eligibility"] == "pretrained comparison only":
                bar.set_hatch("///")
            axes.text(
                bar.get_width() + 0.006,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.5f}",
                va="center",
                fontsize=7,
                color=_INK,
            )
    axes.axvline(
        floor,
        color=_BAR_BLOCKED,
        linestyle="--",
        linewidth=1.3,
        label=f"Random ordering floor ({floor:.5f})",
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{row['label']}\n{row['note']}" for row in ordered],
        fontsize=8,
    )
    axes.set_xlim(0.0, 0.60)
    axes.set_xlabel(
        "Query-mean linear nDCG@10 at K=10 (unitless score, 0 to 1)",
        fontsize=9,
    )
    axes.set_ylabel("Retrieval method (hatched bars are pretrained)", fontsize=9)
    axes.set_title(
        "Task 4 development fold 1 — retrieval quality of every compared method",
        fontsize=11,
        color=_INK,
    )
    axes.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    return _save(figure, destination, "method_quality_comparison.png")


def _stability_figure(bundle: Mapping[str, Any], destination: Path) -> Path:
    summaries = {str(item["method"]): item for item in bundle["stability"]}
    figure, axes = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    _style(axes)
    palette = {"R5": (_BAR_STAR, "o"), "R3": ("#8a5a1f", "s")}
    for method in ("R5", "R3"):
        summary = summaries[method]
        folds = sorted(summary["folds"], key=lambda row: int(row["fold"]))
        x = np.array([int(row["fold"]) for row in folds], dtype=float)
        y = np.array([float(row["score"]) for row in folds], dtype=float)
        mean = float(summary["mean"])
        deviation = float(summary["sample_standard_deviation"])
        colour, marker = palette[method]
        axes.plot(
            x,
            y,
            marker=marker,
            markersize=8,
            linestyle="none",
            color=colour,
            label=(
                f"{method} per-fold run — mean {mean:.5f}, "
                f"sample SD {deviation:.5f}"
            ),
        )
        axes.axhline(mean, color=colour, linewidth=1.3)
        axes.fill_between(
            [-0.4, 4.4],
            mean - deviation,
            mean + deviation,
            color=colour,
            alpha=0.16,
            linewidth=0,
            label=f"{method} mean ± 1 sample SD band",
        )
        for fold, score in zip(x, y, strict=True):
            axes.annotate(
                f"{score:.5f}",
                (fold, score),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                color=colour,
            )
    decision = bundle["decision"]
    axes.set_xlim(-0.4, 4.4)
    axes.set_ylim(0.275, 0.535)
    axes.set_xticks([0, 1, 2, 3, 4])
    axes.set_xlabel("Validation fold (fold number)", fontsize=9)
    axes.set_ylabel("Mean linear nDCG@10 (unitless score)", fontsize=9)
    axes.set_title(
        "Five-fold from-scratch stability — R5 beats R3 on every fold",
        fontsize=11,
        color=_INK,
    )
    axes.legend(fontsize=7.5, loc="center", framealpha=0.95)
    axes.annotate(
        f"Mean gap {float(decision['mean_gap']):.5f} is larger than the pooled spread "
        f"{float(decision['pooled_spread']):.5f}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=8,
        color=_INK,
    )
    return _save(figure, destination, "stability_folds.png")


def _gallery_figure(bundle: Mapping[str, Any], destination: Path) -> Path:
    gallery = bundle["gallery"]
    selected = str(gallery["final_policy"]["policy"])
    order = {name: index for index, name in enumerate(_GALLERY_LABELS)}
    policies = sorted(gallery["policies"], key=lambda row: order[str(row["policy"])])
    labels = [_GALLERY_LABELS[str(row["policy"])] for row in policies]
    colours = [
        _BAR_STAR if str(row["policy"]) == selected else _BAR_BASE for row in policies
    ]
    panels = (
        (
            "Search quality (axis zoomed)",
            [float(row["quality_at_10"]) for row in policies],
            "Equal mean of teacher and V1 nDCG@10 (unitless)",
            "{:.7f}",
            True,
        ),
        (
            "Search latency",
            [float(row["p95_end_to_end_seconds"]) for row in policies],
            "CPU batch-one p95 end-to-end (seconds)",
            "{:.5f}",
            False,
        ),
        (
            "Index storage",
            [float(row["index_bytes"]) / 1_000_000.0 for row in policies],
            "Searchable index size (decimal MB)",
            "{:.2f}",
            False,
        ),
    )
    figure, axes_row = plt.subplots(
        1,
        3,
        figsize=(11.6, 4.6),
        constrained_layout=True,
    )
    for axes, (title, values, ylabel, fmt, zoom) in zip(axes_row, panels, strict=True):
        _style(axes)
        bottom = 0.0
        if zoom:
            span = max(values) - min(values)
            bottom = min(values) - span * 0.6
            axes.set_ylim(bottom, max(values) + span * 0.9)
        else:
            axes.set_ylim(0.0, max(values) * 1.24)
        bars = axes.bar(
            labels,
            [value - bottom for value in values],
            bottom=bottom,
            color=colours,
            edgecolor=_INK,
            linewidth=0.4,
        )
        for bar, value in zip(bars, values, strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=_INK,
            )
        axes.set_title(title, fontsize=10, color=_INK)
        axes.set_ylabel(ylabel, fontsize=8.5)
        axes.set_xlabel("Gallery policy", fontsize=8.5)
        axes.tick_params(axis="x", labelrotation=16)
    axes_row[0].annotate(
        "The three policies differ by only "
        f"{max(panels[0][1]) - min(panels[0][1]):.7f} nDCG@10",
        xy=(0.03, 0.94),
        xycoords="axes fraction",
        fontsize=7.5,
        color=_INK,
    )
    figure.suptitle(
        "Three-policy gallery study on the selected R5 checkpoint — "
        f"{_GALLERY_LABELS[selected]} wins on all three axes",
        fontsize=11,
        color=_INK,
    )
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=_BAR_STAR,
            edgecolor=_INK,
            label="Selected final policy",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=_BAR_BASE,
            edgecolor=_INK,
            label="Rejected policy",
        ),
    ]
    figure.legend(
        handles=handles,
        fontsize=8,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    return _save(figure, destination, "gallery_policy_comparison.png")


def _row_colour(row: Mapping[str, Any]) -> str:
    if row["label"] == "R5":
        return _BAR_STAR
    if row["eligibility"] == "pretrained comparison only":
        return _BAR_BLOCKED
    if row["eligibility"] == "untrained baseline":
        return _BAR_BASE
    return _BAR


def _tradeoff_figure(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    figure, axes_row = plt.subplots(
        1,
        3,
        figsize=(13.4, 5.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]},
    )
    scatter = axes_row[0]
    _style(scatter)
    ordered = sorted(rows, key=lambda item: item["same_source"])
    previous = -1.0
    flip = False
    for row in ordered:
        colour = _row_colour(row)
        scatter.scatter(
            row["worst_p95"],
            row["same_source"],
            s=40.0 + row["index_mb"] * 4.0,
            color=colour,
            edgecolor=_INK,
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        flip = not flip if row["same_source"] - previous < 0.02 else False
        previous = row["same_source"]
        scatter.annotate(
            row["short"],
            (row["worst_p95"], row["same_source"]),
            textcoords="offset points",
            xytext=(11, -12 if flip else 5),
            fontsize=8,
            color=_INK,
        )
    scatter.axvline(1.0, color=_GRADE_COLOURS[0], linestyle="--", linewidth=1.2)
    scatter.set_xlim(0.1, 1.35)
    scatter.set_ylim(0.15, 0.58)
    scatter.set_xlabel(
        "Worst measured CPU batch-one p95 end-to-end time (seconds)",
        fontsize=9,
    )
    scatter.set_ylabel("Same-source mean linear nDCG@10 (unitless score)", fontsize=9)
    scatter.set_title(
        "Quality against latency — marker area is proportional to index MB",
        fontsize=10,
        color=_INK,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=colour,
            markeredgecolor=_INK,
            label=label,
        )
        for colour, label in (
            (_BAR_STAR, "Selected scratch R5"),
            (_BAR, "Other scratch candidate"),
            (_BAR_BLOCKED, "Pretrained comparison only"),
            (_BAR_BASE, "Untrained baseline"),
        )
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color=_GRADE_COLOURS[0],
            linestyle="--",
            label="Pre-registered 1 second CPU p95 gate",
        )
    )
    scatter.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.95)

    bar_panels = (
        (
            axes_row[1],
            [row["index_mb"] for row in ordered],
            "Single-source searchable index (decimal MB)",
            "Storage cost",
            "{:.1f}",
        ),
        (
            axes_row[2],
            [row["parameters"] / 1_000_000.0 for row in ordered],
            "Trained parameters (millions)",
            "Model size",
            "{:.2f}",
        ),
    )
    positions = np.arange(len(ordered), dtype=float)
    for axes, values, xlabel, title, fmt in bar_panels:
        _style(axes)
        axes.barh(
            positions,
            values,
            height=0.62,
            color=[_row_colour(row) for row in ordered],
            edgecolor=_INK,
            linewidth=0.4,
        )
        for position, value in zip(positions, values, strict=True):
            axes.text(
                value + max(values) * 0.02,
                position,
                fmt.format(value),
                va="center",
                fontsize=7.5,
                color=_INK,
            )
        axes.set_yticks(positions)
        axes.set_yticklabels([row["short"] for row in ordered], fontsize=8)
        axes.set_xlim(0.0, max(values) * 1.28)
        axes.set_xlabel(xlabel, fontsize=8.5)
        axes.set_ylabel("Method", fontsize=8.5)
        axes.set_title(title, fontsize=10, color=_INK)
    figure.suptitle(
        "What each method costs to run — latency, index storage, and parameter count",
        fontsize=11,
        color=_INK,
    )
    return _save(figure, destination, "cost_quality_tradeoff.png")


def _slice_lookup(rows: object) -> dict[str, tuple[float | None, float | None]]:
    if isinstance(rows, pd.DataFrame):
        records = [
            {
                "query_source": row.query_source,
                "gallery_source": row.gallery_source,
                "slice": row.slice,
                "value": row.value,
                "coverage": row.coverage,
            }
            for row in rows.itertuples(index=False)
        ]
    else:
        records = list(rows)  # type: ignore[arg-type]
    lookup: dict[str, tuple[float | None, float | None]] = {}
    for record in records:
        if record["query_source"] != "teacher" or record["gallery_source"] != "teacher":
            continue
        value = record["value"]
        coverage = record["coverage"]
        lookup[str(record["slice"])] = (
            None if value is None or pd.isna(value) else float(value),
            None if coverage is None or pd.isna(coverage) else float(coverage),
        )
    return lookup


def _slice_panel(
    axes: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    slices: Sequence[str],
    *,
    title: str,
    ylabel: str,
) -> AxesImage:
    matrix = np.full((len(rows), len(slices)), np.nan)
    coverage = np.full((len(rows), len(slices)), np.nan)
    for row_index, row in enumerate(rows):
        lookup = _slice_lookup(row["slices"])
        for column_index, name in enumerate(slices):
            if name not in lookup:
                raise ValueError(f"{row['label']} is missing the {name} slice")
            value, share = lookup[name]
            matrix[row_index, column_index] = np.nan if value is None else value
            coverage[row_index, column_index] = np.nan if share is None else share
    mesh = axes.imshow(
        np.ma.masked_invalid(matrix),
        cmap=_HEAT,
        vmin=0.0,
        vmax=_HEAT_VMAX,
        aspect="auto",
    )
    for row_index in range(len(rows)):
        for column_index in range(len(slices)):
            value = matrix[row_index, column_index]
            share = coverage[row_index, column_index]
            if np.isnan(value):
                axes.add_patch(
                    plt.Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor="#e8eaee",
                        hatch="xxx",
                        edgecolor="#ffffff",
                        linewidth=0.8,
                    )
                )
                text = "undefined\n0% coverage"
                colour = _INK
            else:
                text = (
                    f"{value:.4f}"
                    if share is not None and abs(share - 1.0) < 1e-9
                    else f"{value:.4f}\n{share * 100:.1f}% cov."
                )
                colour = "#ffffff" if value >= _HEAT_TEXT_SWITCH else _INK
            axes.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7.0,
                color=colour,
            )
    axes.set_xticks(range(len(slices)))
    axes.set_xticklabels(
        [name.replace("_", "\n") for name in slices],
        fontsize=7.5,
    )
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([row["short"] for row in rows], fontsize=8)
    axes.set_xlabel("Development query slice", fontsize=8.5)
    axes.set_ylabel(ylabel, fontsize=8.5)
    axes.set_title(title, fontsize=10, color=_INK)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    return mesh


def _failure_slice_figure(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
) -> Path:
    ordered = sorted(rows, key=lambda row: -row["same_source"])
    figure, axes_row = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.0, 1.05]},
    )
    mesh = _slice_panel(
        axes_row[0],
        ordered,
        _PRIMARY_SLICES,
        title="Protocol A — query-mean nDCG@10",
        ylabel="Method (best same-source quality first)",
    )
    _slice_panel(
        axes_row[1],
        ordered,
        _FAMILY_SLICES,
        title="Protocol B — query-mean Recall@10",
        ylabel="",
    )
    bar = figure.colorbar(
        mesh,
        ax=axes_row,
        location="right",
        fraction=0.028,
        pad=0.015,
        ticks=[0.0, 0.15, 0.30, 0.45, _HEAT_VMAX],
    )
    bar.set_label(
        f"Slice score (unitless, 0 to {_HEAT_VMAX:g}); darker green is better",
        fontsize=8.5,
    )
    bar.ax.tick_params(labelsize=8, length=0)
    bar.outline.set_visible(False)
    figure.suptitle(
        "Where every method actually fails — teacher-to-teacher development slices; "
        "cross-hatched cells are undefined",
        fontsize=11,
        color=_INK,
    )
    return _save(figure, destination, "failure_slice_profile.png")


def _canvas_figure(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    ordered = [row for row in rows if row.get("canvas")]
    ordered.sort(key=lambda row: -row["canvas"]["clean"]["ndcg_at_10"])
    positions = np.arange(len(ordered), dtype=float)
    width = 0.26
    figure, axes_row = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.2),
        constrained_layout=True,
    )
    palette = {"clean": _BAR_STAR, "tall": _BAR, "wide": _BAR_BASE}
    panels = (
        (
            axes_row[0],
            "ndcg_at_10",
            "V1-to-V1 query-mean linear nDCG@10 (unitless)",
            "Retrieval quality collapses on odd canvases",
            _CANVAS_VARIANTS,
        ),
        (
            axes_row[1],
            "mean_top10_overlap",
            "Mean Top-10 overlap with the clean ranking (share, 0 to 1)",
            "Only R3 keeps a recognisable result list",
            ("wide", "tall"),
        ),
    )
    for axes, key, ylabel, title, variants in panels:
        _style(axes)
        span = width * len(variants)
        offsets = [
            -span / 2.0 + width / 2.0 + index * width
            for index in range(len(variants))
        ]
        highest = 0.0
        for offset, variant in zip(offsets, variants, strict=True):
            values = [row["canvas"][variant][key] for row in ordered]
            highest = max(highest, *values)
            axes.bar(
                positions + offset,
                values,
                width=width,
                color=palette[variant],
                edgecolor=_INK,
                linewidth=0.4,
                label=f"{variant} canvas",
            )
            for position, value in zip(positions + offset, values, strict=True):
                axes.annotate(
                    f"{value:.3f}",
                    (position, value),
                    textcoords="offset points",
                    xytext=(0, 2.5),
                    ha="center",
                    fontsize=6.2,
                    color=_MUTED,
                    rotation=90,
                )
        axes.set_xticks(positions)
        axes.set_xticklabels([row["short"] for row in ordered], fontsize=8)
        axes.set_xlabel("Method", fontsize=8.5)
        axes.set_ylabel(ylabel, fontsize=8.5)
        axes.set_ylim(0.0, highest * 1.42)
        axes.set_title(title, fontsize=10, color=_INK)
        axes.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    axes_row[1].annotate(
        "The clean canvas is its own reference, so its\noverlap is 1.000 by definition "
        "and is not drawn.",
        (0.03, 0.62),
        xycoords="axes fraction",
        fontsize=7.2,
        color=_MUTED,
        va="top",
    )
    figure.suptitle(
        "Wide and tall white-canvas stress test — V1 queries against the V1 gallery",
        fontsize=11,
        color=_INK,
    )
    return _save(figure, destination, "canvas_robustness.png")


def build_chart_figures(
    *,
    root: Path = ROOT,
    destination: Path | None = None,
) -> dict[str, Path]:
    """Render every non-image chart the report cites."""

    target = Path(destination) if destination is not None else root / FIGURE_DIR_RELATIVE
    bundle = _bundle(root)
    learned = _learned_rows(bundle)
    untrained = _untrained_rows(root)
    hsv, floor = _frozen_baseline_rows(root)
    quality_rows = [*learned, *untrained, hsv]
    cost_rows = [*learned, *untrained]

    outputs = {
        "method_quality_comparison.png": _method_quality_figure(
            quality_rows,
            floor,
            target,
        ),
        "stability_folds.png": _stability_figure(bundle, target),
        "gallery_policy_comparison.png": _gallery_figure(bundle, target),
        "cost_quality_tradeoff.png": _tradeoff_figure(cost_rows, target),
        "failure_slice_profile.png": _failure_slice_figure(cost_rows, target),
        "canvas_robustness.png": _canvas_figure(cost_rows, target),
    }
    return outputs


def _display_pixels(row: pd.Series, variant: str) -> np.ndarray:
    path = Path(str(row["external_path"]))
    resolved = path if path.is_absolute() else ROOT / path
    with Image.open(resolved) as image:
        prepared = (
            build_odd_aspect_canvas(image, variant)
            if variant in {"wide", "tall"}
            else image
        )
        return preprocess_image(prepared, CONTRACT).pixels


def _draw_panels(
    panels: Sequence[RetrievalPanel],
    catalogue: pd.DataFrame,
    *,
    title: str,
    top_k: int,
    destination: Path,
    name: str,
) -> Path:
    columns = top_k + 1
    figure, axes = plt.subplots(
        len(panels),
        columns,
        figsize=(1.30 * columns + 2.2, 2.05 * len(panels) + 0.9),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, panel in enumerate(panels):
        query_row = resolve_image_rows(catalogue, [panel.query_id]).iloc[0]
        query_axes = axes[row_index][0]
        query_axes.imshow(_display_pixels(query_row, panel.query_variant))
        query_axes.set_title(
            f"QUERY {panel.query_id}\n{panel.article_type} · {panel.base_colour}",
            fontsize=7.5,
            color=_INK,
        )
        score = "undefined" if panel.value is None else f"{panel.value:.4f}"
        query_axes.set_ylabel(
            f"{panel.label.replace('_', ' ')}\n{panel.metric} = {score}\n"
            f"{panel.query_variant} canvas",
            fontsize=7.5,
            color=_INK,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )
        for spine in query_axes.spines.values():
            spine.set_edgecolor(_INK)
            spine.set_linewidth(2.2)
        result_rows = resolve_image_rows(
            catalogue,
            [result.candidate_id for result in panel.results],
        )
        for column_index, (result, candidate) in enumerate(
            zip(panel.results, result_rows.itertuples(index=False), strict=True),
            start=1,
        ):
            cell = axes[row_index][column_index]
            cell.imshow(_display_pixels(pd.Series(candidate._asdict()), "clean"))
            family = " ·F" if result.same_family else ""
            cell.set_title(
                f"#{result.rank} {_GRADE_MARKS[result.grade]}{family}\n"
                f"{result.candidate_id} · d={result.distance:.3f}",
                fontsize=7,
                color=_GRADE_COLOURS[result.grade],
            )
            for spine in cell.spines.values():
                spine.set_edgecolor(_GRADE_COLOURS[result.grade])
                spine.set_linewidth(3.0)
        for cell in axes[row_index]:
            cell.set_xticks([])
            cell.set_yticks([])
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=_GRADE_COLOURS[grade],
            edgecolor=_INK,
            label=_GRADE_LABELS[grade],
        )
        for grade in (2, 1, 0)
    ]
    handles.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#ffffff",
            edgecolor=_INK,
            label="'F' marks a result inside the query's product family",
        )
    )
    figure.legend(
        handles=handles,
        fontsize=7.5,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(title, fontsize=11, color=_INK)
    return _save(figure, destination, name)


def build_retrieval_figures(
    *,
    root: Path = ROOT,
    destination: Path | None = None,
    evidence: WinnerRetrievalEvidence | None = None,
) -> dict[str, Path]:
    """Render the query plus Top-K prediction pictures for the selected method."""

    target = Path(destination) if destination is not None else root / FIGURE_DIR_RELATIVE
    winner = evidence if evidence is not None else load_winner_retrieval_evidence(root=root)
    successes = select_retrieval_panels(winner, kind="success")
    failures = select_retrieval_panels(winner, kind="failure")
    slices = slice_retrieval_panels(winner)
    return {
        "r5_retrieval_success.png": _draw_panels(
            successes,
            winner.catalogue,
            title=(
                f"{winner.method} best development retrievals — V1 query and exact-cosine "
                "Top-10, one row per article type"
            ),
            top_k=RETRIEVAL_TOP_K,
            destination=target,
            name="r5_retrieval_success.png",
        ),
        "r5_retrieval_failure.png": _draw_panels(
            failures,
            winner.catalogue,
            title=(
                f"{winner.method} worst development retrievals — V1 query and exact-cosine "
                "Top-10, one row per article type"
            ),
            top_k=RETRIEVAL_TOP_K,
            destination=target,
            name="r5_retrieval_failure.png",
        ),
        "r5_retrieval_slices.png": _draw_panels(
            slices,
            winner.catalogue,
            title=(
                f"{winner.method} pre-registered slice examples — V1 query and Top-5, "
                "including the wide-canvas failure"
            ),
            top_k=SLICE_TOP_K,
            destination=target,
            name="r5_retrieval_slices.png",
        ),
    }


def build_report_figures(
    *,
    root: Path = ROOT,
    destination: Path | None = None,
) -> dict[str, Path]:
    """Render every figure cited by the final Task 4 comparison report."""

    outputs = {
        **build_chart_figures(root=root, destination=destination),
        **build_retrieval_figures(root=root, destination=destination),
    }
    if sorted(outputs) != sorted(REPORT_FIGURE_NAMES):
        raise ValueError("report figure set does not match the cited figure names")
    return outputs
