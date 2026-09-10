"""Generate the Task 4 Cursor Canvas from the strict final freeze."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from fashion.config import ROOT
from fashion.task4.final_freeze import (
    FINAL_FREEZE_RELATIVE_PATH,
    validate_final_comparison_bundle,
)

CANVAS_FILENAME = "task4-model-comparison.canvas.tsx"
TEMPLATE_RELATIVE_PATH = Path(
    "scripts/task4/templates/task4-model-comparison.canvas.tsx.template"
)


def managed_canvas_path(*, root: Path = ROOT, home: Path | None = None) -> Path:
    """Return this checkout's Cursor-managed Canvas path."""

    workspace_key = "-".join(root.resolve().parts[1:])
    user_home = Path.home() if home is None else home
    return (
        user_home
        / ".cursor"
        / "projects"
        / workspace_key
        / "canvases"
        / CANVAS_FILENAME
    )


def _object_literal(item: Mapping[str, object]) -> str:
    fields = ", ".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in item.items()
    )
    return f"{{ {fields} }}"


def _object_array(items: list[Mapping[str, object]]) -> str:
    return "[\n  " + ",\n  ".join(_object_literal(item) for item in items) + ",\n]"


def _number_array(values: list[object]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _row_array(items: list[list[str]]) -> str:
    return "[\n  " + ",\n  ".join(json.dumps(item) for item in items) + ",\n]"


def _title(value: str) -> str:
    return {
        "teacher": "Teacher-only",
        "v1": "V1-only",
        "two_view": "Two-view",
    }[value]


def _number_word(value: int) -> str:
    return (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    )[value]


def _canvas_values(payload: Mapping[str, Any]) -> dict[str, str]:
    methods = {item["method"]: item for item in payload["methods"]}
    candidates = [
        {
            "method": item["method"],
            "kind": (
                "pretrained benchmark"
                if item["pretrained"]
                else "scratch candidate"
            ),
            "score": item["selected_metrics"]["development_winner_score"],
            "cross": item["selected_metrics"]["cross_source_score"],
            "ratio": item["selected_metrics"]["source_robustness_ratio"],
        }
        for item in payload["methods"]
    ]

    summaries = {item["method"]: item for item in payload["stability"]}
    r5 = summaries["R5"]
    r3 = summaries["R3"]
    folds = sorted(row["fold"] for row in r5["folds"])
    r5_scores = [
        next(row["score"] for row in r5["folds"] if row["fold"] == fold)
        for fold in folds
    ]
    r3_scores = [
        next(row["score"] for row in r3["folds"] if row["fold"] == fold)
        for fold in folds
    ]

    variant_order = ("clean", "tall", "wide")

    def canvas_scores(method: str) -> list[object]:
        rows = {
            row["query_variant"]: row["ndcg_at_10"]
            for row in methods[method]["canvas"]
        }
        return [rows[variant] for variant in variant_order]

    gallery = [
        {
            "policy": _title(item["policy"]),
            "normalization": item["query_normalization_source"],
            "quality": item["quality_at_10"],
            "p95": item["p95_end_to_end_seconds"],
            "bytes": item["index_bytes"],
            "decision": (
                "Selected"
                if item["policy"] == payload["decision"]["gallery_policy"]
                else "Rejected"
            ),
        }
        for item in payload["gallery"]["policies"]
    ]
    attempts: list[list[str]] = []
    for item in payload["attempts"]:
        stale = item["error_type"] == "StaleRunningAttempt"
        attempts.append(
            [
                item["run_id"],
                item["method"],
                (
                    "Abandoned"
                    if stale
                    else "Completed"
                    if item["status"] == "completed"
                    else "Failed"
                ),
                (
                    "Bounded retry"
                    if item["status"] == "completed"
                    else item["error_type"]
                ),
            ]
        )

    query_counts = {
        row["queries"]
        for method in ("R5", "R3")
        for row in methods[method]["canvas"]
    }
    if len(query_counts) != 1:
        raise ValueError("final comparison Canvas query counts disagree")
    query_count = query_counts.pop()

    winner = payload["decision"]["method"]
    gallery_policy = payload["decision"]["gallery_policy"]
    method_count = len(candidates)
    stability_run_count = len(r5_scores) + len(r3_scores)
    gallery_policy_count = len(gallery)
    replacements = {
        "FREEZE_SHA256": payload["bundle_sha256"],
        "CANDIDATES": _object_array(candidates),
        "FOLD_LABELS": json.dumps([f"Fold {fold}" for fold in folds]),
        "R5_FOLDS": _number_array(r5_scores),
        "R3_FOLDS": _number_array(r3_scores),
        "R5_MEAN": json.dumps(r5["mean"]),
        "R5_SD": json.dumps(r5["sample_standard_deviation"]),
        "R3_MEAN": json.dumps(r3["mean"]),
        "R3_SD": json.dumps(r3["sample_standard_deviation"]),
        "MEAN_GAP": json.dumps(payload["decision"]["mean_gap"]),
        "POOLED_SPREAD": json.dumps(payload["decision"]["pooled_spread"]),
        "CANVAS_CATEGORIES": json.dumps(
            ["Clean", "Tall white canvas", "Wide white canvas"]
        ),
        "R5_CANVAS": _number_array(canvas_scores("R5")),
        "R3_CANVAS": _number_array(canvas_scores("R3")),
        "GALLERY": _object_array(gallery),
        "ATTEMPTS": _row_array(attempts),
        "METHOD_COUNT": str(method_count),
        "METHOD_COUNT_WORD": _number_word(method_count),
        "METHOD_COUNT_WORD_TITLE": _number_word(method_count).title(),
        "STABILITY_RUN_COUNT": str(stability_run_count),
        "STABILITY_RUN_COUNT_WORD": _number_word(stability_run_count),
        "GALLERY_POLICY_COUNT": str(gallery_policy_count),
        "GALLERY_POLICY_COUNT_WORD": _number_word(gallery_policy_count),
        "WINNER": str(winner),
        "GALLERY_POLICY": _title(str(gallery_policy)),
        "GALLERY_POLICY_LOWER": _title(str(gallery_policy)).lower(),
        "GALLERY_POLICY_SHORT": str(gallery_policy).title(),
        "QUERY_COUNT": f"{query_count:,}",
        "SPLIT_PATH": str(payload["split"]["path"]),
        "SCRATCH_GATE": "PASS" if not methods[winner]["pretrained"] else "FAIL",
        "SPEED_GATE": (
            "PASS"
            if max(
                item["p95_end_to_end_seconds"]
                for item in payload["gallery"]["policies"]
            )
            < 1.0
            else "FAIL"
        ),
        "INDEX_GATE": (
            "PASS"
            if max(item["index_bytes"] for item in payload["gallery"]["policies"])
            < 1024**3
            else "FAIL"
        ),
    }
    return replacements


def render_canvas(*, root: Path = ROOT) -> str:
    """Render inline Canvas data only after strict freeze validation."""

    payload = validate_final_comparison_bundle(
        root / FINAL_FREEZE_RELATIVE_PATH,
        root=root,
    )
    rendered = (root / TEMPLATE_RELATIVE_PATH).read_text(encoding="utf-8")
    for name, value in _canvas_values(payload).items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", rendered):
        raise ValueError("Canvas template contains an unresolved placeholder")
    return rendered


def write_canvas(
    destination: Path | None = None,
    *,
    root: Path = ROOT,
    home: Path | None = None,
) -> Path:
    """Write one self-contained Canvas to an explicit or managed path."""

    output = (
        managed_canvas_path(root=root, home=home)
        if destination is None
        else Path(destination)
    )
    output.write_text(render_canvas(root=root), encoding="utf-8")
    return output


def main() -> None:
    print(write_canvas())


if __name__ == "__main__":
    main()
