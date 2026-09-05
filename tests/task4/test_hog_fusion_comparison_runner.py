from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = (
    ("teacher", "teacher"),
    ("v1", "v1"),
    ("teacher", "v1"),
    ("v1", "teacher"),
)


def _runner() -> dict[str, object]:
    path = ROOT / "scripts/task4/run_hog_fusion_comparison.py"
    assert path.is_file(), "Task 4 HOG fusion comparison runner is missing"
    return runpy.run_path(path, run_name="task4_hog_fusion_test")


def _cost(*, p95: float, index_bytes: int) -> dict[str, object]:
    return {
        "scope": "development",
        "fold": 1,
        "parameters": 0,
        "checkpoint_bytes": 0,
        "timing_summary": [
            {
                "query_source": query_source,
                "gallery_source": gallery_source,
                "metric": "end_to_end",
                "percentile": "p95",
                "value_seconds": p95,
                "timed_queries": 2,
            }
            for query_source, gallery_source in DIRECTIONS
        ],
        "per_source_index_cost": {
            source: {"source": source, "index_bytes": index_bytes} for source in ("teacher", "v1")
        },
    }


def _summary(method: str, fingerprint: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": 1,
                "method": method,
                "descriptor_fingerprint": fingerprint,
                "query_source": query_source,
                "gallery_source": gallery_source,
            }
            for query_source, gallery_source in DIRECTIONS
        ]
    )


def _timings(query_ids: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": 1,
                "query_source": query_source,
                "gallery_source": gallery_source,
                "query_id": query_id,
            }
            for query_source, gallery_source in DIRECTIONS
            for query_id in query_ids
        ]
    )


def test_fusion_runner_is_cpu_only_and_forces_single_thread_timing() -> None:
    path = ROOT / "scripts/task4/run_hog_fusion_comparison.py"
    assert path.is_file(), "Task 4 HOG fusion comparison runner is missing"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "OMP_NUM_THREADS": "9",
            "OPENBLAS_NUM_THREADS": "9",
            "MKL_NUM_THREADS": "9",
            "NUMEXPR_NUM_THREADS": "9",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, runpy; "
                "runpy.run_path('scripts/task4/run_hog_fusion_comparison.py', "
                "run_name='task4_hog_fusion_env_test'); "
                "print(os.environ['CUDA_VISIBLE_DEVICES'] + '|' + "
                "'|'.join(os.environ[name] for name in "
                "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',"
                "'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')))"
            ),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "|1|1|1|1"


def test_source_selection_filters_sealed_rows_and_requires_exact_development_ids() -> None:
    runner = _runner()
    splits = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "partition": ["development", "holdout", "quarantine"],
            "sha256": ["a", "b", "c"],
        }
    )
    variants = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "partition": ["development", "holdout", "quarantine"],
            "teacher_path": ["dev-teacher", "sealed-teacher", "quarantine-teacher"],
            "external_path": ["dev-v1", "sealed-v1", "quarantine-v1"],
            "external_sha256": ["d", "e", "f"],
        }
    )

    development = runner["_select_development_sources"](splits, variants)

    assert development["id"].tolist() == [1]
    assert set(development["partition"]) == {"development"}
    assert "sealed-teacher" not in development.to_string()
    assert "quarantine-v1" not in development.to_string()
    with pytest.raises(ValueError, match="exactly match"):
        runner["_select_development_sources"](
            splits,
            variants.loc[variants["id"].ne(1)],
        )


def test_registry_guard_detects_any_byte_change(tmp_path: Path) -> None:
    runner = _runner()
    registry = tmp_path / "runs.csv"
    registry.write_bytes(b"run_id,status\nexisting,completed\n")

    with runner["_unchanged_file"](registry):
        pass

    with pytest.raises(RuntimeError, match="must not modify"):
        with runner["_unchanged_file"](registry):
            registry.write_bytes(b"changed")


def test_practical_gates_are_strict_at_boundaries() -> None:
    runner = _runner()

    assert runner["_gate_verdict"](_cost(p95=0.999999, index_bytes=2**30 - 1)) == {
        "p95_end_to_end_under_one_second": True,
        "index_under_one_gibibyte": True,
    }
    assert runner["_gate_verdict"](_cost(p95=1.0, index_bytes=2**30)) == {
        "p95_end_to_end_under_one_second": False,
        "index_under_one_gibibyte": False,
    }


def test_semantic_guards_require_four_directions_and_exact_query_coverage() -> None:
    runner = _runner()
    method = "hog-plus-spatial-hsv-edge-equal-v1"
    fingerprint = "a" * 64
    summary = _summary(method, fingerprint)
    timings = _timings()

    runner["_validate_semantic_outputs"](
        summary=summary,
        timings=timings,
        method=method,
        descriptor_fingerprint=fingerprint,
        expected_query_ids=(1, 2),
    )

    with pytest.raises(ValueError, match="four directions"):
        runner["_validate_semantic_outputs"](
            summary=summary.iloc[:-1],
            timings=timings,
            method=method,
            descriptor_fingerprint=fingerprint,
            expected_query_ids=(1, 2),
        )
    with pytest.raises(ValueError, match="query coverage"):
        runner["_validate_semantic_outputs"](
            summary=summary,
            timings=_timings((1, 3)),
            method=method,
            descriptor_fingerprint=fingerprint,
            expected_query_ids=(1, 2),
        )


def test_manifest_hashes_artifacts_and_records_both_parent_identities(
    tmp_path: Path,
) -> None:
    runner = _runner()
    artifact = tmp_path / "quality.csv"
    artifact.write_bytes(b"scope\n development\n")
    parent_manifests = {
        source: {
            "hog": {
                "path": f"hog/{source}/manifest.json",
                "sha256": "1" * 64,
                "manifest": {
                    "method": "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1",
                    "descriptor_fingerprint": (
                        "c4604537474ff5ed8889a9e480053b225064fb20027e147582960bb3afd3a848"
                    ),
                    "source": source,
                    "source_fingerprint": f"{source}-source",
                },
            },
            "spatial_hsv_edge": {
                "path": f"spatial/{source}/manifest.json",
                "sha256": "2" * 64,
                "manifest": {
                    "probe": "spatial-hsv-edge-4x4-v2",
                    "source": source,
                    "source_fingerprint": f"{source}-source",
                },
            },
        }
        for source in ("teacher", "v1")
    }

    manifest = runner["_build_manifest"](
        split_fingerprint="3" * 64,
        fused_cache_manifests={
            source: {
                "path": f"fusion/{source}/manifest.json",
                "sha256": "4" * 64,
            }
            for source in ("teacher", "v1")
        },
        parent_manifests=parent_manifests,
        artifacts={"quality_summary": artifact},
        cost=_cost(p95=0.2, index_bytes=1_000),
    )

    assert manifest["scope"] == "development"
    assert manifest["fold"] == 1
    assert manifest["method"] == "hog-plus-spatial-hsv-edge-equal-v1"
    assert manifest["checkpoint_sha256"] is None
    assert manifest["config_fingerprint"] is None
    assert manifest["registry_appended"] is False
    assert manifest["holdout_opened"] is False
    assert manifest["quarantine_opened"] is False
    assert manifest["parents"]["teacher"]["hog"]["sha256"] == "1" * 64
    assert (
        manifest["parents"]["v1"]["spatial_hsv_edge"]["manifest"]["probe"]
        == "spatial-hsv-edge-4x4-v2"
    )
    assert (
        manifest["artifacts"]["quality_summary"]["sha256"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )


def test_atomic_publication_never_leaves_a_trusted_partial_manifest(
    tmp_path: Path,
) -> None:
    runner = _runner()
    destination = tmp_path / "published"
    destination.mkdir()
    old_manifest = destination / "manifest.json"
    old_manifest.write_text(json.dumps({"trusted": "old"}) + "\n", encoding="utf-8")

    def invalid_staging(staging: Path) -> None:
        (staging / "artifact.csv").write_text("broken\n", encoding="utf-8")
        raise ValueError("semantic validation failed")

    with pytest.raises(ValueError, match="semantic validation failed"):
        runner["_publish_package_atomically"](destination, invalid_staging)

    assert json.loads(old_manifest.read_text(encoding="utf-8")) == {"trusted": "old"}
    assert sorted(path.name for path in destination.iterdir()) == ["manifest.json"]
