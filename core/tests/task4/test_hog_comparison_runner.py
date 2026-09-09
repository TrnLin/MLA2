from __future__ import annotations

import hashlib
import os
import runpy
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = (
    ("teacher", "teacher"),
    ("v1", "v1"),
    ("teacher", "v1"),
    ("v1", "teacher"),
)


def _runner() -> dict[str, object]:
    path = ROOT / "scripts/task4/run_hog_comparison.py"
    assert path.is_file(), "Task 4 HOG comparison runner is missing"
    return runpy.run_path(path, run_name="task4_hog_test")


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
            source: {"source": source, "index_bytes": index_bytes}
            for source in ("teacher", "v1")
        },
    }


def test_hog_gates_are_strict_at_one_second_and_one_gibibyte() -> None:
    runner = _runner()

    below = runner["_gate_verdict"](
        _cost(p95=0.999999, index_bytes=2**30 - 1)
    )
    boundary = runner["_gate_verdict"](_cost(p95=1.0, index_bytes=2**30))

    assert below == {
        "p95_end_to_end_under_one_second": True,
        "index_under_one_gibibyte": True,
    }
    assert boundary == {
        "p95_end_to_end_under_one_second": False,
        "index_under_one_gibibyte": False,
    }


def test_hog_source_selection_returns_only_canonical_development_rows() -> None:
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


def test_hog_semantic_validation_requires_four_directions_and_output_identity() -> None:
    runner = _runner()
    method = "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1"
    fingerprint = "a" * 64
    summary = pd.DataFrame.from_records(
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
    timings = summary.assign(query_id=1)

    runner["_validate_semantic_outputs"](
        summary=summary,
        timings=timings,
        method=method,
        descriptor_fingerprint=fingerprint,
        query_count=1,
    )

    with pytest.raises(ValueError, match="four directions"):
        runner["_validate_semantic_outputs"](
            summary=summary.iloc[:-1],
            timings=timings,
            method=method,
            descriptor_fingerprint=fingerprint,
            query_count=1,
        )
    with pytest.raises(ValueError, match="descriptor identity"):
        runner["_validate_semantic_outputs"](
            summary=summary.assign(descriptor_fingerprint="b" * 64),
            timings=timings,
            method=method,
            descriptor_fingerprint=fingerprint,
            query_count=1,
        )


def test_hog_registry_guard_detects_any_registry_row_or_byte_change(
    tmp_path: Path,
) -> None:
    runner = _runner()
    registry = tmp_path / "runs.csv"
    registry.write_text("run_id,status\nexisting,completed\n", encoding="utf-8")

    with runner["_unchanged_file"](registry):
        pass

    with pytest.raises(RuntimeError, match="must not modify"):
        with runner["_unchanged_file"](registry):
            registry.write_text(
                "run_id,status\nexisting,completed\nhog,completed\n",
                encoding="utf-8",
            )


def test_hog_feature_indexes_build_both_sources_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    barrier = threading.Barrier(2)
    thread_ids: set[int] = set()
    worker_counts: list[int] = []

    def fake_ensure(cache, *, cache_root, workers):
        del cache_root
        thread_ids.add(threading.get_ident())
        worker_counts.append(workers)
        barrier.wait(timeout=2)
        return SimpleNamespace(index=cache)

    monkeypatch.setattr(runner["hog_module"], "ensure_hog_feature_index", fake_ensure)
    caches = {"teacher": object(), "v1": object()}

    indexes = runner["_open_hog_indexes"](caches, workers=16, cache_root=ROOT)

    assert indexes == caches
    assert len(thread_ids) == 2
    assert sorted(worker_counts) == [8, 8]


def test_hog_manifest_has_descriptor_not_checkpoint_identity(tmp_path: Path) -> None:
    runner = _runner()
    artifact = tmp_path / "quality.csv"
    artifact.write_text("scope\n development\n", encoding="utf-8")
    manifest = runner["_build_manifest"](
        split_fingerprint="1" * 64,
        source_manifests={
            "teacher": {
                "source_fingerprint": "2" * 64,
                "contract": {"width": 240, "height": 320},
            },
            "v1": {
                "source_fingerprint": "3" * 64,
                "contract": {"width": 240, "height": 320},
            },
        },
        artifacts={"quality_summary": artifact},
        cost=_cost(p95=0.2, index_bytes=1_000),
    )

    assert manifest["scope"] == "development"
    assert manifest["fold"] == 1
    assert manifest["method"] == "hog-luma-g5-u8-c32-b2-s1-l2hys02-v1"
    assert manifest["checkpoint_sha256"] is None
    assert len(manifest["descriptor_fingerprint"]) == 64
    assert manifest["artifacts"]["quality_summary"]["sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()


def test_hog_runner_is_cpu_only_and_forces_single_thread_timing() -> None:
    path = ROOT / "scripts/task4/run_hog_comparison.py"
    assert path.is_file(), "Task 4 HOG comparison runner is missing"
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
                "runpy.run_path('scripts/task4/run_hog_comparison.py', "
                "run_name='task4_hog_env_test'); "
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


def test_hog_runner_can_parallelize_untimed_work_but_restores_one_thread() -> None:
    runner = _runner()

    with runner["_thread_limit"](4):
        active = [
            int(pool["num_threads"])
            for pool in threadpool_info()
            if pool.get("user_api") == "blas"
        ]
        assert active and set(active) == {4}

    with runner["_thread_limit"](1):
        restored = [
            int(pool["num_threads"])
            for pool in threadpool_info()
            if pool.get("user_api") == "blas"
        ]
        assert restored and set(restored) == {1}
