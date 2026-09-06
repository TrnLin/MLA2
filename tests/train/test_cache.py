from __future__ import annotations

from pathlib import Path

from fashion.data.hashing import compute_sha256
from fashion.train.cache import (
    build_run_cache_key,
    find_cached_run,
    implementation_sha256,
)
from fashion.train.registry import RunRecord, RunRegistry


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    split = tmp_path / "splits.csv"
    labels = tmp_path / "label_maps.json"
    source = tmp_path / "src" / "fashion"
    source.mkdir(parents=True)
    split.write_text("id,cv_fold\n1,0\n", encoding="utf-8")
    labels.write_text('{"season":["Fall","Winter"]}\n', encoding="utf-8")
    (source / "model.py").write_text("WIDTH = 32\n", encoding="utf-8")
    return split, labels, source


def _key(tmp_path: Path):
    split, labels, source = _inputs(tmp_path)
    return build_run_cache_key(
        {"model": "smallcnn", "learning_rate": 3e-4},
        fold=0,
        seed=2753,
        implementation_paths=(source,),
        split_path=split,
        label_map_path=labels,
        root=tmp_path,
    )


def _completed_record(key, checkpoint: Path, run_id: str = "cache-hit") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        experiment_id="c1-screen",
        fold=key.fold,
        seed=key.seed,
        config_sha256=key.config_sha256,
        split_sha256=key.split_sha256,
        label_map_sha256=key.label_map_sha256,
        implementation_sha256=key.implementation_sha256,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=compute_sha256(checkpoint),
    )


def _finalize(registry: RunRegistry, record: RunRecord, status: str = "completed") -> None:
    registry.append(record)
    record.status = status
    record.finished_at_utc = "2026-08-26T00:00:00Z"
    registry.finalize(record)


def test_implementation_hash_changes_for_code_but_not_discovered_docs(tmp_path: Path) -> None:
    _, _, source = _inputs(tmp_path)
    docs = source / "notes.md"
    docs.write_text("first explanation\n", encoding="utf-8")
    original = implementation_sha256(source, root=tmp_path)

    docs.write_text("rewritten explanation\n", encoding="utf-8")
    after_docs = implementation_sha256(source, root=tmp_path)
    (source / "model.py").write_text("WIDTH = 64\n", encoding="utf-8")
    after_code = implementation_sha256(source, root=tmp_path)

    assert after_docs == original
    assert after_code != original


def test_cache_key_is_stable_across_config_mapping_order(tmp_path: Path) -> None:
    split, labels, source = _inputs(tmp_path)
    common = {
        "fold": 0,
        "seed": 2753,
        "implementation_paths": (source,),
        "split_path": split,
        "label_map_path": labels,
        "root": tmp_path,
    }

    first = build_run_cache_key({"b": 2, "a": 1}, **common)
    second = build_run_cache_key({"a": 1, "b": 2}, **common)

    assert first == second
    assert len(first.digest) == 64


def test_cache_hit_requires_completed_row_and_valid_artifact(tmp_path: Path) -> None:
    key = _key(tmp_path)
    checkpoint = tmp_path / "fold-0.pt"
    checkpoint.write_bytes(b"scratch weights")
    registry = RunRegistry(tmp_path / "runs.csv")
    _finalize(registry, _completed_record(key, checkpoint))

    cached = find_cached_run(registry, key, artifact_root=tmp_path)

    assert cached is not None
    assert cached.run_id == "cache-hit"
    assert cached.verified_artifacts["checkpoint"] == compute_sha256(checkpoint)


def test_cache_misses_when_artifact_changed_or_run_failed(tmp_path: Path) -> None:
    key = _key(tmp_path)
    checkpoint = tmp_path / "fold-0.pt"
    checkpoint.write_bytes(b"original")
    registry = RunRegistry(tmp_path / "runs.csv")
    _finalize(registry, _completed_record(key, checkpoint, "changed"))
    checkpoint.write_bytes(b"corrupted")

    assert find_cached_run(registry, key, artifact_root=tmp_path) is None

    failed_checkpoint = tmp_path / "failed.pt"
    failed_checkpoint.write_bytes(b"partial")
    failed_registry = RunRegistry(tmp_path / "failed-runs.csv")
    _finalize(
        failed_registry,
        _completed_record(key, failed_checkpoint, "failed"),
        status="failed",
    )
    assert find_cached_run(failed_registry, key, artifact_root=tmp_path) is None


def test_cache_can_require_prediction_evidence(tmp_path: Path) -> None:
    key = _key(tmp_path)
    checkpoint = tmp_path / "fold-0.pt"
    checkpoint.write_bytes(b"scratch weights")
    registry = RunRegistry(tmp_path / "runs.csv")
    _finalize(registry, _completed_record(key, checkpoint))

    assert (
        find_cached_run(
            registry,
            key,
            required_artifacts=("checkpoint", "prediction"),
            artifact_root=tmp_path,
        )
        is None
    )
