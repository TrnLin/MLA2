"""Traceable Task 2 file-impact evidence and an HTML-safe Matplotlib flow diagram."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.hashing import compute_sha256
from fashion.train.artifacts import atomic_write_bytes, atomic_write_csv, atomic_write_json

FILE_IMPACT_COLUMNS = ("producer", "artifact", "consumer", "effect", "phase")
FROZEN_BUNDLE = "Frozen Season bundle"
DEPLOYMENT_CONSUMERS = frozenset(
    {"Notebook 06 holdout", "Prediction CLI", "Streamlit app"}
)
TRAINING_NODES = frozenset(
    {"fashion.data.torch", "fashion.models.season", "fashion.train.engine", "Task 2 runner"}
)


def build_file_impact_edges() -> pd.DataFrame:
    """Return the declared producer-artifact-consumer contract in execution order."""
    rows = [
        (
            "data/processed/splits.csv",
            "canonical partitions and cv_fold",
            "fashion.data.torch",
            "Changing membership or folds invalidates every comparable run.",
            "inputs",
        ),
        (
            "data/processed/label_maps.json",
            "Fall, Spring, Summer, Winter class order",
            "fashion.data.torch",
            "Changing class indices invalidates models, metrics, and predictions.",
            "inputs",
        ),
        (
            "fashion.data.torch",
            "fold-fitted tensors and loaders",
            "Task 2 runner",
            "Transform changes create a new implementation and transform hash.",
            "training",
        ),
        (
            "configs/task2/*.json",
            "immutable experiment choices",
            "Task 2 runner",
            "A changed scientific choice creates a new config hash and run.",
            "training",
        ),
        (
            "fashion.models.season",
            "scratch logits and feature maps",
            "Task 2 runner",
            "Architecture changes invalidate checkpoints and OOF evidence.",
            "training",
        ),
        (
            "fashion.train.engine",
            "best checkpoint, OOF predictions, and history",
            "Task 2 runner",
            "Optimisation changes invalidate the implementation hash.",
            "training",
        ),
        (
            "Task 2 runner",
            "run lifecycle and provenance row",
            "results/runs.csv",
            "Every physical run is retained as completed, failed, or interrupted.",
            "artifacts",
        ),
        (
            "Task 2 runner",
            "checkpoint, OOF, and history files",
            "tmp/task2 artifacts",
            "Bytes are reused only when their declared SHA-256 values match.",
            "artifacts",
        ),
        (
            "results/runs.csv",
            "run IDs, hashes, metrics, and status",
            "results/evidence/task2",
            "Registry changes rebuild evidence but do not alter trained weights.",
            "evidence",
        ),
        (
            "tmp/task2 artifacts",
            "verified OOF and histories",
            "results/evidence/task2",
            "Only hash-verified outputs may support a measured claim.",
            "evidence",
        ),
        (
            "results/evidence/task2",
            "tables, figures, run IDs, and limitations",
            "Notebook 03",
            "Notebook claims must trace to measured run IDs.",
            "narrative",
        ),
        (
            "results/evidence/task2",
            "selected tables, figures, and limitations",
            "Task 2 report",
            "Report text changes interpretation only, never model bytes.",
            "narrative",
        ),
        (
            "Notebook 03",
            "predeclared scorecard decision",
            "selection_freeze.json",
            "The selected run and rules become immutable before holdout access.",
            "freeze",
        ),
        (
            "selection_freeze.json",
            "winner config, hashes, and median epoch rule",
            "Development refit",
            "Refit must reproduce the frozen method on all valid development rows.",
            "freeze",
        ),
        (
            "Development refit",
            "task2_season.pt and manifest",
            FROZEN_BUNDLE,
            "The manifest binds model bytes, labels, transform, and config by SHA-256.",
            "bundle",
        ),
        (
            FROZEN_BUNDLE,
            "immutable model plus manifest",
            "Notebook 06 holdout",
            "Holdout evaluates the bundle once and cannot tune it.",
            "deployment",
        ),
        (
            FROZEN_BUNDLE,
            "immutable model plus manifest",
            "Prediction CLI",
            "Official Season predictions must come from the frozen bytes.",
            "deployment",
        ),
        (
            FROZEN_BUNDLE,
            "immutable model plus manifest",
            "Streamlit app",
            "The app reports the same model/config hash as offline inference.",
            "deployment",
        ),
    ]
    frame = pd.DataFrame(rows, columns=FILE_IMPACT_COLUMNS)
    validate_file_impact_edges(frame)
    return frame


def validate_file_impact_edges(frame: pd.DataFrame) -> None:
    """Fail closed on missing nodes, unsafe feedback, or alternative deployment inputs."""
    if tuple(frame.columns) != FILE_IMPACT_COLUMNS:
        raise ValueError(f"file-impact columns must be {list(FILE_IMPACT_COLUMNS)}")
    if frame.empty or frame.isna().any().any():
        raise ValueError("file-impact edges must be non-empty and complete")
    if frame.duplicated(["producer", "artifact", "consumer"]).any():
        raise ValueError("file-impact edges must be unique")
    nodes = set(frame["producer"]) | set(frame["consumer"])
    required = {
        "data/processed/splits.csv",
        "data/processed/label_maps.json",
        "fashion.data.torch",
        "configs/task2/*.json",
        "fashion.models.season",
        "fashion.train.engine",
        "Task 2 runner",
        "results/runs.csv",
        "tmp/task2 artifacts",
        "results/evidence/task2",
        "Notebook 03",
        "Task 2 report",
        "selection_freeze.json",
        "Development refit",
        FROZEN_BUNDLE,
        *DEPLOYMENT_CONSUMERS,
    }
    missing = sorted(required - nodes)
    if missing:
        raise ValueError(f"file-impact flow is missing nodes: {missing}")
    unsafe_holdout = frame.loc[
        frame["producer"].astype(str).str.contains("holdout", case=False)
        & frame["consumer"].isin(TRAINING_NODES)
    ]
    if not unsafe_holdout.empty:
        raise ValueError("holdout information cannot flow into training")
    for consumer in DEPLOYMENT_CONSUMERS:
        producers = set(frame.loc[frame["consumer"].eq(consumer), "producer"])
        if producers != {FROZEN_BUNDLE}:
            raise ValueError(f"{consumer} must consume only the frozen bundle")
    evidence_feedback = frame.loc[
        frame["producer"].eq("results/evidence/task2")
        & frame["consumer"].isin(TRAINING_NODES | {"Development refit"})
    ]
    if not evidence_feedback.empty:
        raise ValueError("descriptive evidence cannot modify training or refit")


def _flow_positions() -> dict[str, tuple[float, float]]:
    return {
        "data/processed/splits.csv": (0.7, 8.8),
        "data/processed/label_maps.json": (0.7, 7.1),
        "configs/task2/*.json": (0.7, 5.4),
        "fashion.data.torch": (3.3, 8.8),
        "fashion.models.season": (3.3, 7.1),
        "fashion.train.engine": (3.3, 5.4),
        "Task 2 runner": (6.0, 7.1),
        "results/runs.csv": (8.7, 8.3),
        "tmp/task2 artifacts": (8.7, 6.0),
        "results/evidence/task2": (11.4, 7.1),
        "Notebook 03": (14.0, 8.3),
        "Task 2 report": (14.0, 6.0),
        "selection_freeze.json": (16.6, 8.3),
        "Development refit": (16.6, 6.0),
        FROZEN_BUNDLE: (16.6, 3.7),
        "Notebook 06 holdout": (14.0, 1.2),
        "Prediction CLI": (16.6, 1.2),
        "Streamlit app": (19.2, 1.2),
    }


def _node_colour(node: str) -> str:
    if node in DEPLOYMENT_CONSUMERS:
        return "#FDE68A"
    if node == FROZEN_BUNDLE:
        return "#86EFAC"
    if node in {"selection_freeze.json", "Development refit"}:
        return "#C4B5FD"
    if node in {"Notebook 03", "Task 2 report", "results/evidence/task2"}:
        return "#BAE6FD"
    if node in {"results/runs.csv", "tmp/task2 artifacts"}:
        return "#FED7AA"
    return "#E2E8F0"


def _portable_artifact_path(path: Path, *, fallback_root: Path) -> str:
    resolved = path.resolve()
    for anchor in (ROOT.resolve(), fallback_root.resolve()):
        try:
            return resolved.relative_to(anchor).as_posix()
        except ValueError:
            continue
    return path.name


def plot_file_impact_flow(
    edges: pd.DataFrame | None = None,
    *,
    output_path: str | Path = TASK2_FIGURE_DIR / "file_impact_flow.png",
) -> tuple[Figure, Axes]:
    """Render and atomically save the notebook-safe file-impact flow as PNG."""
    frame = build_file_impact_edges() if edges is None else edges.copy()
    validate_file_impact_edges(frame)
    positions = _flow_positions()
    unknown = (set(frame["producer"]) | set(frame["consumer"])) - set(positions)
    if unknown:
        raise ValueError(f"file-impact layout is missing positions: {sorted(unknown)}")

    figure = Figure(figsize=(20, 10.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.set_xlim(-0.7, 20.7)
    axis.set_ylim(-0.2, 10.2)
    axis.axis("off")
    axis.set_title(
        "Task 2 file impact: data and code create runs; evidence explains frozen bytes",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    for row in frame.itertuples(index=False):
        start = positions[row.producer]
        end = positions[row.consumer]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#64748B",
            connectionstyle="arc3,rad=0.04",
            shrinkA=46,
            shrinkB=46,
            zorder=1,
        )
        axis.add_patch(arrow)
    for node, (x, y) in positions.items():
        width = 2.15 if node not in {FROZEN_BUNDLE, "results/evidence/task2"} else 2.35
        box = FancyBboxPatch(
            (x - width / 2, y - 0.42),
            width,
            0.84,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=_node_colour(node),
            edgecolor="#334155",
            linewidth=1.2,
            zorder=2,
        )
        axis.add_patch(box)
        axis.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="semibold",
            wrap=True,
            zorder=3,
        )
    axis.text(
        0.0,
        0.05,
        "Rule: evidence and narrative may explain a model, but cannot flow back into training.",
        fontsize=10,
        color="#475569",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(output_path, buffer.getvalue())
    return figure, axis


def build_task2_evidence(
    *,
    evidence_directory: str | Path = TASK2_EVIDENCE_DIR,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
) -> dict[str, Any]:
    """Build the static traceability evidence used before measured experiments exist."""
    evidence_root = Path(evidence_directory)
    figure_root = Path(figure_directory)
    edges_path = evidence_root / "file_impact_edges.csv"
    figure_path = figure_root / "file_impact_flow.png"
    manifest_path = evidence_root / "file_impact_manifest.json"
    common_root = Path(
        os.path.commonpath([evidence_root.resolve(), figure_root.resolve()])
    )
    edges = build_file_impact_edges()
    atomic_write_csv(edges_path, edges)
    figure, _ = plot_file_impact_flow(edges, output_path=figure_path)
    figure.clear()
    manifest = {
        "schema_version": "1.0.0",
        "scope": "task2_file_impact",
        "edge_count": len(edges),
        "node_count": len(set(edges["producer"]) | set(edges["consumer"])),
        "edges_path": _portable_artifact_path(edges_path, fallback_root=common_root),
        "edges_sha256": compute_sha256(edges_path),
        "figure_path": _portable_artifact_path(figure_path, fallback_root=common_root),
        "figure_sha256": compute_sha256(figure_path),
        "holdout_to_training_edges": 0,
        "deployment_input": FROZEN_BUNDLE,
    }
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest
