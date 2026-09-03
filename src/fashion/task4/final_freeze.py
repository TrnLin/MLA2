"""Strict, compact Task 4 final-comparison freeze."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from fashion.config import ROOT
from fashion.task4.experiments import (
    validate_final_gallery_decision_artifact,
    validate_post_stability_deployment_artifact,
)

FINAL_FREEZE_RELATIVE_PATH = Path(
    "results/evidence/task4/final/task4-final-comparison.json"
)
EXPECTED_FINAL_FREEZE_SHA256 = (
    "83ca0730c413572880577cd7757e00ea473633aa6ffee42c46eb950726ca185a"
)
_PROTECTED_PARTS = {"holdout", "quarantine", "test"}
_METHODS = {"R1", "R2", "R3", "R4", "R5", "B1"}
_FACTOR_CHANGES = {
    "R1": "Scratch ResNet-18 plus VICReg; base learned reference.",
    "R2": "R1 with a larger scratch ResNet-34 encoder.",
    "R3": "R1 plus geometry and wide/tall canvas augmentation.",
    "R4": "R3 plus batch-hard product-family triplet loss.",
    "R5": "Independent scratch convolutional autoencoder with a 128-value bottleneck.",
    "B1": "R1 recipe with ImageNet pretrained weights; comparison-only.",
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_type",
    "producer",
    "producer_validation",
    "safety",
    "split",
    "methods",
    "stability",
    "gallery",
    "attempts",
    "decision",
    "source_artifacts",
    "bundle_sha256",
}
_METHOD_FIELDS = {
    "method",
    "run_id",
    "run_kind",
    "factor_change",
    "pretrained",
    "selected_metrics",
    "protocol_a",
    "protocol_b",
    "failure_slices",
    "examples",
    "canvas",
    "cost",
}


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_sha256(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("bundle_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"final comparison artifact cannot be read: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("final comparison artifact must be a JSON object")
    return value


def _repository_path(path: str | Path, *, root: Path) -> tuple[Path, Path]:
    repository = root.resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else repository / candidate).resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError as error:
        raise ValueError("final comparison path must stay inside the repository") from error
    lowered = {part.lower() for part in relative.parts}
    if "quarantine" in lowered or (
        "data" in lowered
        and "raw" in lowered
        and "teacher" in lowered
        and "test" in lowered
    ):
        raise ValueError("final comparison path enters a protected data location")
    return resolved, relative


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"final comparison {label} schema is invalid")
    return value


def _finite(value: object, *, label: str, bounded: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"final comparison {label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (bounded and not 0.0 <= number <= 1.0):
        raise ValueError(f"final comparison {label} is outside its valid range")
    return number


def _validate_method(method: object) -> None:
    item = _require_exact_fields(method, _METHOD_FIELDS, label="method")
    name = item["method"]
    if name not in _METHODS or item["factor_change"] != _FACTOR_CHANGES[name]:
        raise ValueError("final comparison method identity is invalid")
    if (
        type(item["pretrained"]) is not bool
        or item["pretrained"] != (name == "B1")
    ):
        raise ValueError("final comparison pretrained boundary is invalid")
    selected = _require_exact_fields(
        item["selected_metrics"],
        {
            "development_winner_score",
            "cross_source_score",
            "source_robustness_ratio",
        },
        label="selected metrics",
    )
    for key, value in selected.items():
        _finite(value, label=str(key), bounded=True)
    for key in ("protocol_a", "protocol_b", "failure_slices", "examples", "canvas"):
        rows = item[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"final comparison {key} must be non-empty")
    protocols = {
        row.get("protocol")
        for row in item["protocol_a"] + item["protocol_b"]  # type: ignore[operator]
        if isinstance(row, Mapping)
    }
    if protocols != {"primary", "family"}:
        raise ValueError("final comparison Protocol A/B coverage is incomplete")
    canvas_labels = {
        row.get("query_variant")
        for row in item["canvas"]  # type: ignore[union-attr]
        if isinstance(row, Mapping)
    }
    if canvas_labels != {"clean", "wide", "tall"}:
        raise ValueError("final comparison canvas coverage is incomplete")
    cost = _require_exact_fields(
        item["cost"],
        {
            "parameters",
            "checkpoint_bytes",
            "embedding_bytes",
            "index_bytes",
            "per_source_index_cost",
            "timing_summary",
            "measurement_route",
        },
        label="cost",
    )
    for key in ("parameters", "checkpoint_bytes"):
        if isinstance(cost[key], bool) or not isinstance(cost[key], int) or cost[key] <= 0:
            raise ValueError(f"final comparison {key} is invalid")
    for key in ("embedding_bytes", "index_bytes", "per_source_index_cost"):
        values = cost[key]
        if not isinstance(values, Mapping) or set(values) != {"teacher", "v1"}:
            raise ValueError(f"final comparison {key} source coverage is invalid")
    timings = cost["timing_summary"]
    if not isinstance(timings, list) or {
        (row.get("metric"), row.get("percentile"))
        for row in timings
        if isinstance(row, Mapping)
    } != {
        ("encoding", "p50"),
        ("encoding", "p95"),
        ("search", "p50"),
        ("search", "p95"),
        ("end_to_end", "p50"),
        ("end_to_end", "p95"),
    }:
        raise ValueError("final comparison timing metric coverage is incomplete")


def validate_final_comparison_bundle(
    path: str | Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate exact bundle schema, safety, identity, paths, and SHA-256."""

    resolved, relative = _repository_path(path, root=root)
    if relative != FINAL_FREEZE_RELATIVE_PATH:
        raise ValueError(
            "final comparison bundle must use its canonical repository path"
        )
    payload = _read_json(resolved)
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("final comparison top-level schema is invalid")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "task4_final_comparison_freeze"
        or payload["producer"]
        != "fashion.task4.final_freeze.write_final_comparison_bundle"
    ):
        raise ValueError("final comparison artifact identity is invalid")
    safety = _require_exact_fields(
        payload["safety"],
        {
            "development_only",
            "holdout_opened",
            "quarantine_opened",
            "official_teacher_test_opened",
        },
        label="safety",
    )
    if safety != {
        "development_only": True,
        "holdout_opened": False,
        "quarantine_opened": False,
        "official_teacher_test_opened": False,
    }:
        raise ValueError("final comparison is not a sealed development-only artifact")
    split = _require_exact_fields(
        payload["split"],
        {"path", "fingerprint"},
        label="split",
    )
    if split["path"] != "data/processed/splits.csv":
        raise ValueError("final comparison must use only the canonical development split")
    methods = payload["methods"]
    if (
        not isinstance(methods, list)
        or len(methods) != 6
        or {item.get("method") for item in methods if isinstance(item, Mapping)}
        != _METHODS
    ):
        raise ValueError("final comparison method set is invalid")
    for method in methods:
        _validate_method(method)
    stability = payload["stability"]
    if (
        not isinstance(stability, list)
        or {item.get("method") for item in stability if isinstance(item, Mapping)}
        != {"R5", "R3"}
    ):
        raise ValueError("final comparison stability method set is invalid")
    for summary in stability:
        if not isinstance(summary, Mapping):
            raise ValueError("final comparison stability row is invalid")
        folds = summary.get("folds")
        if (
            not isinstance(folds, list)
            or {row.get("fold") for row in folds if isinstance(row, Mapping)}
            != set(range(5))
        ):
            raise ValueError("final comparison requires all five stability folds")
        _finite(summary.get("mean"), label="stability mean", bounded=True)
        _finite(summary.get("sample_standard_deviation"), label="stability SD")
    gallery = payload["gallery"]
    if (
        not isinstance(gallery, Mapping)
        or {item.get("policy") for item in gallery.get("policies", [])}
        != {"teacher", "v1", "two_view"}
    ):
        raise ValueError("final comparison gallery policy set is invalid")
    decision = _require_exact_fields(
        payload["decision"],
        {"method", "gallery_policy", "mean_gap", "pooled_spread"},
        label="decision",
    )
    if decision["method"] != "R5" or decision["gallery_policy"] != "teacher":
        raise ValueError("final comparison decision identity is invalid")
    _finite(decision["mean_gap"], label="mean gap")
    _finite(decision["pooled_spread"], label="pooled spread")
    sources = payload["source_artifacts"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("final comparison source artifacts are missing")
    for source in sources:
        record = _require_exact_fields(
            source,
            {"path", "sha256"},
            label="source artifact",
        )
        source_path = Path(str(record["path"]))
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("final comparison source paths must be repository-relative")
        lowered = {part.lower() for part in source_path.parts}
        if lowered & _PROTECTED_PARTS:
            raise ValueError("final comparison source path is protected")
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("final comparison source SHA-256 is malformed")
    recorded = payload["bundle_sha256"]
    calculated = _bundle_sha256(payload)
    if recorded != calculated:
        raise ValueError("final comparison bundle SHA-256 does not match its payload")
    if EXPECTED_FINAL_FREEZE_SHA256 and calculated != EXPECTED_FINAL_FREEZE_SHA256:
        raise ValueError("final comparison bundle hash is not the frozen repository hash")
    return payload


def _artifact_record(manifest: Mapping[str, object], name: str) -> Mapping[str, object]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("validated learned manifest artifacts are malformed")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"validated learned manifest lacks one {name} artifact")
    return matches[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_quality_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fields = (
        "query_source",
        "gallery_source",
        "protocol",
        "metric",
        "aggregation",
        "value",
        "query_count",
        "class_count",
    )
    selected = [
        row
        for row in rows
        if row["k"] == "10"
        and row["aggregation"] in {"query_mean", "article_type_macro"}
        and (
            row["protocol"] == "primary"
            and row["metric"] in {"ndcg", "precision_any", "precision_strict"}
            or row["protocol"] == "family"
            and row["metric"] in {"coverage", "recall", "hit_rate", "precision"}
        )
    ]
    normalized = [
        {
            key: (
                float(row[key])
                if key == "value"
                else int(row[key])
                if key in {"query_count", "class_count"} and row[key]
                else None
                if key in {"query_count", "class_count"}
                else row[key]
            )
            for key in fields
        }
        for row in selected
    ]
    return (
        [row for row in normalized if row["protocol"] == "primary"],
        [row for row in normalized if row["protocol"] == "family"],
    )


def _selected_failure_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fields = (
        "query_source",
        "gallery_source",
        "protocol",
        "slice",
        "metric",
        "aggregation",
        "value",
        "total_queries",
        "scored_queries",
        "excluded_queries",
        "coverage",
        "caveat",
    )
    return [
        {
            key: (
                float(row[key])
                if key in {"value", "coverage"} and row[key]
                else int(row[key])
                if key in {"total_queries", "scored_queries", "excluded_queries"}
                else None
                if key in {"value", "coverage"}
                else row[key]
            )
            for key in fields
        }
        for row in rows
    ]


def _selected_examples(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fields = (
        "query_source",
        "gallery_source",
        "slice",
        "query_variant",
        "query_id",
        "metric",
        "value",
        "rank",
        "candidate_id",
        "distance",
    )
    return [
        {
            key: (
                int(row[key])
                if key in {"query_id", "rank", "candidate_id"}
                else float(row[key])
                if key in {"value", "distance"} and row[key]
                else None
                if key == "value"
                else row[key]
            )
            for key in fields
        }
        for row in rows
        if row["rank"] == "1"
    ]


def _selected_canvas(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fields = (
        "query_source",
        "gallery_source",
        "query_variant",
        "queries",
        "ndcg_at_10",
        "ndcg_change_from_clean",
        "mean_top10_overlap",
        "caveat",
    )
    return [
        {
            key: (
                int(row[key])
                if key == "queries"
                else float(row[key])
                if key in {
                    "ndcg_at_10",
                    "ndcg_change_from_clean",
                    "mean_top10_overlap",
                }
                else row[key]
            )
            for key in fields
        }
        for row in rows
    ]


def _cost_summary(cost: Mapping[str, object]) -> dict[str, object]:
    timings = cost["timing_summary"]
    if not isinstance(timings, list):
        raise ValueError("validated learned cost timing summary is malformed")
    return {
        "parameters": int(cost["parameters"]),
        "checkpoint_bytes": int(cost["checkpoint_bytes"]),
        "embedding_bytes": dict(cost["feature_bytes"]),  # type: ignore[arg-type]
        "index_bytes": dict(cost["index_bytes"]),  # type: ignore[arg-type]
        "per_source_index_cost": {
            source: {
                "build_seconds": float(values["build_seconds"]),
                "peak_rss_bytes": int(values["peak_rss_bytes"]),
            }
            for source, values in cost["per_source_index_cost"].items()  # type: ignore[union-attr]
        },
        "timing_summary": [
            {
                "query_source": row["query_source"],
                "gallery_source": row["gallery_source"],
                "metric": row["metric"],
                "percentile": row["percentile"],
                "value_seconds": float(row["value_seconds"]),
            }
            for row in timings
        ],
        "measurement_route": cost["measurement_route"],
    }


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _strict_source_inputs(root: Path) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, str]],
]:
    if root.resolve() != ROOT.resolve():
        raise ValueError("final comparison producer only accepts the canonical repository")
    from scripts.task4 import run_model_comparisons as runner

    splits = runner.load_canonical_splits()
    artifacts = runner.load_config_artifacts(root / "results/evidence/task4")
    with (root / "results/runs.csv").open(encoding="utf-8", newline="") as handle:
        registry = list(csv.DictReader(handle))
    completed = {
        row["run_id"]: row
        for row in registry
        if row.get("task") == "task4" and row.get("status") == "completed"
    }
    learned_paths = sorted(
        root.glob("results/evidence/task4/learned/*/manifest.json")
    )
    validated_methods: list[dict[str, object]] = []
    with _working_directory(root):
        for manifest_path in learned_paths:
            manifest = _read_json(manifest_path)
            run_id = str(manifest.get("run_id") or "")
            registry_row = completed.get(run_id)
            if registry_row is None:
                raise ValueError("learned manifest lacks one completed registry row")
            spec = {
                "run_id": run_id,
                "config_artifact_path": manifest_path.with_name(
                    "experiment_config.json"
                ),
                "checkpoint_path": registry_row["checkpoint_path"],
                "manifest_path": manifest_path,
            }
            runner._task6_input_from_request(  # noqa: SLF001
                spec,
                artifacts=artifacts,
                splits=splits,
                registry_row=registry_row,
            ).validated()
            records = {
                name: _artifact_record(manifest, name)
                for name in (
                    "quality_summary",
                    "failure_slices",
                    "examples",
                    "canvas_summary",
                    "cost",
                )
            }
            frames = {
                name: _read_csv(manifest_path.parent / str(record["path"]))
                for name, record in records.items()
                if name != "cost"
            }
            cost = _read_json(manifest_path.parent / str(records["cost"]["path"]))
            protocol_a, protocol_b = _selected_quality_rows(
                frames["quality_summary"]
            )
            validated_methods.append(
                {
                    "method": manifest["method"],
                    "run_id": run_id,
                    "run_kind": manifest["run_kind"],
                    "factor_change": _FACTOR_CHANGES[str(manifest["method"])],
                    "pretrained": manifest["method"] == "B1",
                    "selected_metrics": manifest["selected_metrics"],
                    "protocol_a": protocol_a,
                    "protocol_b": protocol_b,
                    "failure_slices": _selected_failure_rows(
                        frames["failure_slices"]
                    ),
                    "examples": _selected_examples(frames["examples"]),
                    "canvas": _selected_canvas(frames["canvas_summary"]),
                    "cost": _cost_summary(cost),
                }
            )

        request_path = (
            root
            / "results/evidence/task4/phase_requests/task9-gallery-request.json"
        )
        request = runner.load_gallery_phase_request(request_path)
        deployment_inputs = runner._deployment_inputs_from_request(  # noqa: SLF001
            request,
            artifacts=artifacts,
            splits=splits,
        )
        deployment_path = (
            root
            / "results/evidence/task4/phase_results/"
            "task9-post-stability-deployment.json"
        )
        deployment = validate_post_stability_deployment_artifact(
            deployment_path,
            deployment_inputs,
        )
        selected = next(
            item
            for item in deployment_inputs
            if item.finalist.method == deployment.winner.method
        )
        gallery_path = (
            root
            / "results/evidence/task4/phase_results/"
            "task9-final-gallery-decision.json"
        )
        timing_path = (
            root
            / "results/evidence/task4/phase_results/gallery_policy_timing.json"
        )
        validate_final_gallery_decision_artifact(
            gallery_path,
            deployment=deployment,
            evidence_manifest=selected.candidate_evidence_manifest,
            timing_artifact_path=timing_path,
        )
    return (
        validated_methods,
        _read_json(deployment_path),
        _read_json(gallery_path),
        registry,
    )


def write_final_comparison_bundle(
    destination: str | Path = ROOT / FINAL_FREEZE_RELATIVE_PATH,
    *,
    root: Path = ROOT,
) -> Path:
    """Strictly validate full evidence, then write its compact final view."""

    resolved, relative = _repository_path(destination, root=root)
    if relative != FINAL_FREEZE_RELATIVE_PATH:
        raise ValueError(
            "final comparison bundle must use its canonical repository path"
        )
    methods, deployment, gallery, registry = _strict_source_inputs(root)
    source_paths = [
        root / "results/runs.csv",
        *sorted(root.glob("results/evidence/task4/learned/*/manifest.json")),
        *sorted(
            root.glob(
                "results/evidence/task4/stability/*/stability_evidence.json"
            )
        ),
        root
        / "results/evidence/task4/phase_results/"
        "task9-post-stability-deployment.json",
        root
        / "results/evidence/task4/phase_results/"
        "task9-final-gallery-decision.json",
        root
        / "results/evidence/task4/phase_results/task9-gallery-result.json",
        root
        / "results/evidence/task4/phase_results/gallery_policy_timing.json",
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "task4_final_comparison_freeze",
        "producer": "fashion.task4.final_freeze.write_final_comparison_bundle",
        "producer_validation": [
            "Task6ManifestInput.validated",
            "StabilityEvidenceInput.validated",
            "validate_post_stability_deployment_artifact",
            "validate_final_gallery_decision_artifact",
        ],
        "safety": {
            "development_only": True,
            "holdout_opened": False,
            "quarantine_opened": False,
            "official_teacher_test_opened": False,
        },
        "split": {
            "path": deployment["split_path"],
            "fingerprint": deployment["split_fingerprint"],
        },
        "methods": sorted(methods, key=lambda item: str(item["method"])),
        "stability": deployment["stability_summaries"],
        "gallery": {
            "candidate_cost_policy": gallery["candidate_cost_policy"],
            "final_policy": gallery["final_policy"],
            "policies": gallery["policies"],
        },
        "attempts": [
            {
                "run_id": row.get("run_id", ""),
                "method": row.get("method", ""),
                "status": row.get("status", ""),
                "error_type": row.get("error_type", ""),
                "error_message": row.get("error_message", ""),
            }
            for row in registry
            if row.get("task") == "task4"
            and (
                row.get("status") in {"failed", "abandoned"}
                or "retry" in row.get("run_id", "")
            )
        ],
        "decision": {
            "method": deployment["selected_model"]["method"],
            "gallery_policy": gallery["final_policy"]["policy"],
            "mean_gap": deployment["mean_gap"],
            "pooled_spread": deployment["pooled_spread"],
        },
        "source_artifacts": [
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in source_paths
        ],
    }
    payload["bundle_sha256"] = _bundle_sha256(payload)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_final_comparison_bundle(resolved, root=root)
    return resolved


def main() -> None:
    path = write_final_comparison_bundle()
    print(f"{path} {_sha256_file(path)}")


if __name__ == "__main__":
    main()
