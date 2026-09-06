from __future__ import annotations

import hashlib
import inspect
import io
import json
import multiprocessing
import pickle
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import fashion.task4 as task4
import fashion.task4.learned_evidence as learned
import fashion.task4.training as training
from fashion.config import SPLITS_CSV
from fashion.data.splits import cv_assignment_digest
from fashion.task4.cache import DevelopmentImageCache
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import FeatureIndex
from fashion.task4.protocol import RetrievalViews, family_candidate_mask
from fashion.task4.training import (
    AugmentationPolicy,
    CandidateConfig,
    CheckpointRecord,
    SourcePolicy,
    TrainingHyperparameters,
    TrainingResult,
    TrainingSessionConfig,
)
from fashion.train.registry import (
    TASK4_RUN_COLUMNS as RUN_COLUMNS,
)
from fashion.train.registry import (
    Task4RunRegistry as RunRegistry,
)

CONTRACT = PreprocessingContract(width=240, height=320)
SPLIT_SHA = "b" * 64


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        assert not self.training
        assert torch.is_inference_mode_enabled()
        self.calls += 1
        values = torch.zeros((len(images), 128), dtype=torch.float32, device=images.device)
        values[:, :3] = images.mean(dim=(2, 3))
        values[:, 3] = self.anchor
        return torch.nn.functional.normalize(values, dim=1)


def _session(
    *,
    run_id: str = "run-1",
    fold: int = 1,
    checkpoints: tuple[int, ...] = (2,),
    split_fingerprint: str = SPLIT_SHA,
) -> TrainingSessionConfig:
    return TrainingSessionConfig(
        run_id=run_id,
        run_kind="candidate",
        candidate=CandidateConfig("R1", "resnet18"),
        hyperparameters=TrainingHyperparameters(
            warmup_epochs=1,
            planned_epochs=max(checkpoints),
            checkpoint_epochs=checkpoints,
        ),
        objective="vicreg",
        source_policy=SourcePolicy.TEACHER_V1_PAIRS,
        augmentation_policy=AugmentationPolicy.NONE,
        validation_fold=fold,
        split_fingerprint=split_fingerprint,
    )


def _checkpoint(
    tmp_path: Path,
    session: TrainingSessionConfig,
    *,
    epoch: int | None = None,
    score: float = 0.4,
) -> CheckpointRecord:
    selected_epoch = session.hyperparameters.checkpoint_epochs[-1] if epoch is None else epoch
    path = tmp_path / f"{session.run_id}-{selected_epoch}.pt"
    path.write_bytes(f"{session.run_id}:{selected_epoch}".encode())
    return CheckpointRecord(
        epoch=selected_epoch,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config_hash=session.config_hash,
        score=score,
        split_fingerprint=session.split_fingerprint,
        weight_origin=session.model_metadata.weight_origin,
        parent_run_id=session.parent_run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
    )


def _cache(tmp_path: Path, source: str = "teacher") -> DevelopmentImageCache:
    cache_dir = tmp_path / source
    cache_dir.mkdir(exist_ok=True)
    ids = np.array([3, 8], dtype=np.int64)
    images = np.zeros((2, 320, 240, 3), dtype=np.uint8)
    images[0, 1:319, 1:239] = (64, 128, 192)
    images[1, 2:318, 2:238] = (192, 128, 64)
    bounds = np.array([[1, 1, 319, 239], [2, 2, 318, 238]], dtype=np.int32)
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "scope": "development",
        "source": source,
        "rows": 2,
        "array_shape": list(images.shape),
        "array_dtype": "uint8",
        "bounds_shape": list(bounds.shape),
        "bounds_dtype": "int32",
        "id_dtype": "int64",
        "source_fingerprint": fingerprint,
        "contract": CONTRACT.to_dict(),
    }
    return DevelopmentImageCache(cache_dir, ids, images, bounds, manifest)


def _write_cache_files(cache: DevelopmentImageCache) -> None:
    id_digest = hashlib.sha256()
    for product_id in cache.ids:
        id_digest.update(str(int(product_id)).encode())
        id_digest.update(b"\n")
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache.cache_dir / "ids.npy", cache.ids, allow_pickle=False)
    np.save(cache.cache_dir / "images.npy", cache.images, allow_pickle=False)
    np.save(
        cache.cache_dir / "content_bounds.npy",
        cache.content_bounds,
        allow_pickle=False,
    )
    cache.manifest.update(
        {
            "files": {
                "ids": "ids.npy",
                "images": "images.npy",
                "content_bounds": "content_bounds.npy",
            },
            "rows": len(cache.ids),
            "array_shape": list(cache.images.shape),
            "array_dtype": str(cache.images.dtype),
            "bounds_shape": list(cache.content_bounds.shape),
            "bounds_dtype": str(cache.content_bounds.dtype),
            "id_dtype": str(cache.ids.dtype),
            "id_sha256": id_digest.hexdigest(),
        }
    )
    (cache.cache_dir / "manifest.json").write_bytes(
        learned._canonical_json(cache.manifest)
    )


def _statistics(cache: DevelopmentImageCache, fold: int = 1) -> dict[str, object]:
    record = {
        "validation_fold": fold,
        "split_fingerprint": SPLIT_SHA,
        "training_rows": 10,
        "training_id_sha256": "a" * 64,
        "content_pixels": 100,
        "mean": [0.25, 0.5, 0.75],
        "std": [0.25, 0.25, 0.25],
        "source": cache.manifest["source"],
        "source_fingerprint": cache.manifest["source_fingerprint"],
        "contract": CONTRACT.to_dict(),
    }
    return record


def _index(source: str, ids: np.ndarray, features: np.ndarray, fold: int = 1) -> FeatureIndex:
    return FeatureIndex(
        source=source,
        contract=CONTRACT,
        ids=ids,
        features=features.astype(np.float32),
        transform_seconds=0.1,
        source_bytes=100,
        method="R1",
        fold=fold,
        checkpoint_fingerprint="c" * 64,
        config_fingerprint="d" * 64,
        provenance=_exact_provenance(source, fold=fold),
    )


def _timing_manifest(index: FeatureIndex) -> dict[str, object]:
    provenance = index.provenance
    assert isinstance(provenance, learned.LearnedProvenance)
    manifest = provenance.to_dict()
    manifest["source_statistics_sha256"] = manifest.pop("statistics_sha256")
    manifest["feature_method"] = index.method
    return manifest


def _bound_timing_inputs(
    tmp_path: Path,
    indexes: dict[str, FeatureIndex],
) -> tuple[dict[str, FeatureIndex], dict[str, Any]]:
    bound_indexes: dict[str, FeatureIndex] = {}
    encoders: dict[str, Any] = {}
    for source, index in indexes.items():
        session = _session(fold=index.fold)
        checkpoint = _checkpoint(tmp_path, session)
        cache = _cache(tmp_path, source)
        statistics = _statistics(cache, fold=index.fold)
        statistics["split_fingerprint"] = session.split_fingerprint
        provenance = replace(
            index.provenance,
            checkpoint_sha256=checkpoint.sha256,
            config_hash=session.config_hash,
            split_fingerprint=session.split_fingerprint,
            source_fingerprint=str(cache.manifest["source_fingerprint"]),
            statistics_sha256=hashlib.sha256(
                learned._canonical_json(statistics)
            ).hexdigest(),
        )
        bound = replace(
            index,
            checkpoint_fingerprint=checkpoint.sha256,
            config_fingerprint=session.config_hash,
            provenance=provenance,
        )
        bound_indexes[source] = bound
        encoders[source] = learned.LearnedTimingEncoder(
            extractor=learned.LazyCPUQueryExtractor(
                source=source,
                checkpoint=checkpoint,
                session=session,
                statistics=statistics,
                path_column=f"{source}_path",
            ),
            feature_manifest=_timing_manifest(bound),
        )
    return bound_indexes, encoders


def test_task4_exports_stable_learned_evidence_interfaces() -> None:
    assert task4.encode_development_cache is learned.encode_development_cache
    assert task4.ensure_learned_feature_index is learned.ensure_learned_feature_index
    assert task4.evaluate_learned_quality is learned.evaluate_learned_quality
    assert task4.rank_two_view_gallery is learned.rank_two_view_gallery
    assert task4.write_learned_artifacts is learned.write_learned_artifacts
    assert task4.complete_learned_evidence is learned.complete_learned_evidence
    assert task4.reconstruct_training_result is learned.reconstruct_training_result


def test_cache_encoding_uses_exact_source_normalization_sorted_ids_and_128_unit_output(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    model = TinyEncoder()

    index = learned.encode_development_cache(
        cache,
        model=model,
        statistics=_statistics(cache),
        session=_session(),
        checkpoint=_checkpoint(tmp_path, _session()),
        batch_size=1,
    )

    assert index.ids.tolist() == [3, 8]
    assert index.features.shape == (2, 128)
    assert index.features.dtype == np.float32
    assert np.isfinite(index.features).all()
    assert np.linalg.norm(index.features, axis=1) == pytest.approx([1.0, 1.0])
    assert model.calls == 2
    assert not model.training
    assert index.features[0, 0] == pytest.approx(0.00386424, abs=1e-6)
    assert index.features[0, 1] == pytest.approx(0.00772848, abs=1e-6)
    assert index.features[0, 2] == pytest.approx(0.01159271, abs=1e-6)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cache, stats: cache.manifest.update(scope="holdout"), "development"),
        (lambda cache, stats: cache.manifest.update(source="v1"), "source"),
        (lambda cache, stats: stats.update(source_fingerprint="0" * 64), "fingerprint"),
        (lambda cache, stats: stats.update(validation_fold=2), "fold"),
        (lambda cache, stats: stats.update(split_fingerprint="0" * 64), "split"),
        (lambda cache, stats: stats.update(contract={"width": 1}), "contract"),
        (lambda cache, stats: stats.update(std=[1.0, 0.0, 1.0]), "standard deviation"),
    ],
)
def test_cache_encoding_rejects_mismatched_or_sealed_provenance_before_model_use(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    cache = _cache(tmp_path)
    statistics = _statistics(cache)
    mutation(cache, statistics)
    model = TinyEncoder()
    session = _session()

    with pytest.raises(ValueError, match=message):
        learned.encode_development_cache(
            cache,
            model=model,
            statistics=statistics,
            session=session,
            checkpoint=_checkpoint(tmp_path, session),
        )
    assert model.calls == 0


def test_learned_feature_cache_reuses_exact_identity_and_rebuilds_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache(tmp_path)
    session = _session()
    checkpoint = _checkpoint(tmp_path, session)
    models: list[TinyEncoder] = []

    def load_checkpoint(*args: Any, **kwargs: Any) -> Any:
        model = TinyEncoder()
        models.append(model)
        return SimpleNamespace(model=model)

    monkeypatch.setattr(learned, "load_checkpoint", load_checkpoint)
    kwargs = {
        "cache": cache,
        "statistics": _statistics(cache),
        "session": session,
        "checkpoint": checkpoint,
        "cache_root": tmp_path / "features",
    }

    first = learned.ensure_learned_feature_index(**kwargs)
    second = learned.ensure_learned_feature_index(**kwargs)
    assert first.cache_dir == second.cache_dir
    assert len(models) == 1

    (first.cache_dir / "features.npy").write_bytes(b"corrupt")
    rebuilt = learned.ensure_learned_feature_index(**kwargs)
    assert len(models) == 2
    assert rebuilt.index.features.shape == (2, 128)

    other = replace(checkpoint, sha256="e" * 64)
    third = learned.ensure_learned_feature_index(**{**kwargs, "checkpoint": other})
    assert third.cache_dir != rebuilt.cache_dir
    assert len(models) == 3


def _split_row(product_id: int, fold: int) -> dict[str, object]:
    family = f"family-fold-{fold}"
    record = {
        "id": product_id,
        "sha256": f"sha-{product_id}",
        "duplicate_group": f"duplicate-{product_id}",
        "product_name_key": family,
        "product_family_group": family,
        "partition": "development",
        "cv_fold": fold,
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "quarantine_reason": "",
        "articleType": "Shirts",
        "baseColour": "Blue",
        "season": "Summer",
        "gender": "Unisex",
        "usage": "Casual",
        "has_articleType_label": True,
        "has_season_label": True,
        "has_gender_label": True,
        "has_usage_label": True,
        "mode": "RGB",
        "aspect_ratio": 0.75,
    }
    return record


def _splits(fold: int = 1) -> pd.DataFrame:
    rows = [_split_row(100 + offset, fold) for offset in range(11)]
    other_folds = [value for value in range(5) if value != fold]
    rows += [_split_row(200 + offset, other_folds[offset % 4]) for offset in range(20)]
    return pd.DataFrame(rows)


def _quality_indexes(fold: int = 1) -> dict[str, FeatureIndex]:
    ids = np.array([*range(100, 111), *range(200, 220)], dtype=np.int64)
    features = np.zeros((len(ids), 128), dtype=np.float32)
    features[:, 0] = 1.0
    return {source: _index(source, ids, features, fold=fold) for source in ("teacher", "v1")}


def test_learned_quality_has_complete_four_direction_query_groups_for_any_fold() -> None:
    result = learned.evaluate_learned_quality(_splits(3), _quality_indexes(3), fold=3)

    assert set(result.pair_evaluations) == set(learned.source_directions())
    assert set(
        result.query_metrics[["query_source", "gallery_source"]].itertuples(index=False, name=None)
    ) == set(learned.source_directions())
    for _, rows in result.query_metrics.groupby(
        ["query_source", "gallery_source", "protocol"], sort=False
    ):
        assert rows["query_id"].tolist() == sorted(rows["query_id"])
        assert rows["query_id"].nunique() == 11


def _summary_values(values: dict[tuple[str, str], float]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": 1,
                "protocol": "primary",
                "metric": "ndcg",
                "k": 10,
                "aggregation": "query_mean",
                "query_source": query,
                "gallery_source": gallery,
                "value": value,
            }
            for (query, gallery), value in values.items()
        ]
    )


def test_milestone_scorer_and_source_scores_use_equal_direction_weighting() -> None:
    summary = _summary_values(
        {
            ("teacher", "teacher"): 0.8,
            ("v1", "v1"): 0.4,
            ("teacher", "v1"): 0.3,
            ("v1", "teacher"): 0.5,
        }
    )
    scorer = learned.make_milestone_scorer(lambda _model: summary, fold=1)

    assert scorer(TinyEncoder(), 20) == pytest.approx(0.6)
    assert learned.summarize_learned_scores(summary) == {
        "development_winner_score": pytest.approx(0.6),
        "cross_source_score": pytest.approx(0.4),
        "source_robustness_ratio": pytest.approx(2.0 / 3.0),
    }

    zero = summary.copy()
    zero.loc[zero["query_source"].eq(zero["gallery_source"]), "value"] = 0.0
    assert learned.summarize_learned_scores(zero)["source_robustness_ratio"] == 0.0

    wrong_fold = summary.assign(fold=2)
    with pytest.raises(ValueError, match="fold"):
        learned.make_milestone_scorer(lambda _model: wrong_fold, fold=1)(TinyEncoder(), 20)


def test_failure_and_canvas_outputs_keep_teacher_slice_caveat() -> None:
    ordinary = pd.DataFrame({"slice": ["grayscale"], "value": [0.2]})
    canvas_summary = pd.DataFrame({"query_variant": ["wide"], "ndcg_at_10": [0.3]})
    canvas_per_query = pd.DataFrame({"query_variant": ["wide"], "query_id": [4]})

    failure, summary, per_query = learned.attach_teacher_slice_caveat(
        ordinary, canvas_summary, canvas_per_query
    )

    for frame in (failure, summary, per_query):
        assert frame["caveat"].unique().tolist() == [learned.TEACHER_SLICE_CAVEAT]


def test_two_view_gallery_collapses_minimum_distance_before_numeric_ties() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame({"id": [9]}),
        gallery=pd.DataFrame({"id": [10, 20, 30]}),
    )
    query_ids = np.array([9], dtype=np.int64)
    query_features = np.zeros((1, 128), dtype=np.float32)
    query_features[:, 0] = 1.0
    teacher_features = np.zeros((3, 128), dtype=np.float32)
    teacher_features[:, :2] = np.array([[0.8, 0.6], [0.5, np.sqrt(0.75)], [0.8, 0.6]])
    v1_features = np.zeros((3, 128), dtype=np.float32)
    v1_features[:, :2] = np.array([[0.0, 1.0], [0.8, 0.6], [0.0, 1.0]])
    teacher = _index(
        "teacher",
        np.array([10, 20, 30]),
        teacher_features,
    )
    v1 = _index(
        "v1",
        np.array([10, 20, 30]),
        v1_features,
    )

    rankings = learned.rank_two_view_gallery(
        query_index=_index("teacher", query_ids, query_features),
        gallery_indexes={"teacher": teacher, "v1": v1},
        views=views,
        protocol="primary",
        max_k=3,
    )

    assert rankings["candidate_id"].tolist() == [10, 20, 30]
    assert rankings["distance"].tolist() == pytest.approx([0.2, 0.2, 0.2])
    assert rankings["rank"].tolist() == [1, 2, 3]


def test_lazy_cpu_extractor_is_picklable_and_rejects_sealed_row_before_file_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    checkpoint = _checkpoint(tmp_path, session)
    extractor = learned.LazyCPUQueryExtractor(
        source="teacher",
        checkpoint=checkpoint,
        session=session,
        statistics=_statistics(_cache(tmp_path)),
        path_column="path",
        root=tmp_path,
    )
    pickle.loads(pickle.dumps(extractor))
    monkeypatch.setattr(
        learned,
        "load_preprocessed_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("file opened")),
    )

    with pytest.raises(ValueError, match="development"):
        extractor(pd.Series({"id": 1, "partition": "holdout", "path": "sealed.png"}))


def test_timing_routes_every_query_through_all_four_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame({"id": [3, 1, 2], "partition": ["development"] * 3})
    indexes = _quality_indexes()
    views = RetrievalViews(
        queries=rows.loc[:, ["id"]],
        gallery=pd.DataFrame({"id": list(range(200, 220))}),
    )
    indexes, encoders = _bound_timing_inputs(tmp_path, indexes)
    monkeypatch.setattr(
        learned.LazyCPUQueryExtractor,
        "__call__",
        lambda self, row: np.r_[1.0, np.zeros(127)].astype(np.float32),
    )
    samples, summary = learned.benchmark_learned_directions(
        rows,
        views=views,
        indexes=indexes,
        encoders=encoders,
        policy=learned.TimingPolicy(warmup_queries=0),
        clock_ns=iter(range(36)).__next__,
    )

    assert len(samples) == 12
    assert samples.groupby(["query_source", "gallery_source"])["query_id"].nunique().eq(3).all()
    assert len(summary) == 24


def _complete_timing_summary(value: float = 0.1, timed_queries: int = 3) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "query_source": query,
                "gallery_source": gallery,
                "metric": metric,
                "percentile": percentile,
                "value_seconds": value,
                "timed_queries": timed_queries,
            }
            for query, gallery in learned.source_directions()
            for metric in ("encoding", "search", "end_to_end")
            for percentile in ("p50", "p95")
        ]
    )


def test_learned_cost_records_model_storage_build_rss_and_strict_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in learned.THREAD_VARIABLES:
        monkeypatch.setenv(variable, "1")
    costs = {
        source: learned.IndexCost(
            source=source,
            contract=CONTRACT,
            rows=10,
            dimension=128,
            payload_bytes=5120,
            index_bytes=5200 if source == "teacher" else 2**30,
            build_seconds=1.5,
            peak_rss_bytes=50_000,
        )
        for source in ("teacher", "v1")
    }

    session = _session(fold=3)
    checkpoint = _checkpoint(tmp_path, session)
    caches = {source: _cache(tmp_path, source) for source in ("teacher", "v1")}
    statistics = {source: _statistics(caches[source], fold=3) for source in caches}
    monkeypatch.setattr(learned, "_load_evidence_model", lambda *args: TinyEncoder())
    monkeypatch.setattr(
        learned,
        "measure_learned_index_build",
        lambda cache, **kwargs: costs[str(cache.manifest["source"])],
    )

    record = learned.build_learned_cost_record(
        _complete_timing_summary(),
        caches=caches,
        statistics=statistics,
        session=session,
        checkpoint=checkpoint,
        policy=learned.TimingPolicy(),
        fold=3,
        selected_gallery_policy="two_view",
    )

    assert record["fold"] == 3
    assert record["parameters"] == 1
    assert record["checkpoint_bytes"] == checkpoint.path.stat().st_size
    assert record["p95_end_to_end_under_one_second"] is True
    assert record["index_under_one_gibibyte"] is False
    assert record["selected_policy_total_index_bytes"] == 5200 + 2**30
    assert record["per_source_index_cost"]["teacher"]["peak_rss_bytes"] == 50_000


def test_spawned_learned_index_measurement_loads_real_task5_checkpoint(
    tmp_path: Path,
) -> None:
    session = _session()
    model = session.build_cpu_model()
    optimizer = training.build_optimizer(model, session.hyperparameters)
    scheduler = training.WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=1,
        config=session.hyperparameters,
    )
    checkpoint = training.save_checkpoint(
        tmp_path / "real.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=training.make_grad_scaler("cpu"),
        epoch=2,
        session=session,
        score=0.5,
    )
    cache = _cache(tmp_path, "teacher")

    measured = learned.measure_learned_index_build(
        cache,
        statistics=_statistics(cache),
        session=session,
        checkpoint=checkpoint,
    )

    assert measured.source == "teacher"
    assert measured.rows == 2
    assert measured.dimension == 128
    assert measured.payload_bytes == 2 * 128 * 4
    assert measured.index_bytes == measured.payload_bytes + 2 * 8
    assert measured.build_seconds >= 0
    assert measured.peak_rss_bytes > 0


def _artifact_frames(
    query_ids: list[int],
    *,
    checkpoint_sha256: str = "c" * 64,
    config_hash: str = "d" * 64,
) -> dict[str, pd.DataFrame]:
    directions = learned.source_directions()
    primary_grid = [
        (metric, k, aggregation)
        for metric in ("ndcg", "precision_any", "precision_strict", "tie_rate")
        for k in (5, 10, 20)
        for aggregation in ("query_mean", "article_type_macro")
    ]
    family_grid = [
        (metric, 10, "query_mean")
        for metric in ("recall", "hit_rate", "precision", "coverage", "tie_rate")
    ]
    quality_summary = pd.DataFrame.from_records(
        [
            {
                "method": "R1",
                "fold": 1,
                "checkpoint_fingerprint": checkpoint_sha256,
                "config_fingerprint": config_hash,
                "size": "240x320",
                "width": 240,
                "height": 320,
                "query_source": query,
                "gallery_source": gallery,
                "protocol": protocol,
                "query_transform_seconds": 0.1,
                "gallery_transform_seconds": 0.1,
                "query_source_bytes": 100,
                "gallery_source_bytes": 100,
                "feature_bytes_per_image": 512,
                "uint8_tensor_bytes_per_image": 230400,
                "float32_tensor_bytes_per_image": 921600,
                "metric": metric,
                "k": k,
                "aggregation": aggregation,
                "value": 0.5,
                "query_count": len(query_ids),
                "class_count": 1,
            }
            for query, gallery in directions
            for protocol, grid in (("primary", primary_grid), ("family", family_grid))
            for metric, k, aggregation in grid
        ]
    )
    metric_defaults = {
        "ndcg_at_5": 0.5,
        "precision_any_at_5": 0.5,
        "precision_strict_at_5": 0.5,
        "tie_rate_at_5": 0.0,
        "ndcg_at_10": 0.5,
        "precision_any_at_10": 0.5,
        "precision_strict_at_10": 0.5,
        "tie_rate_at_10": 0.0,
        "ndcg_at_20": 0.5,
        "precision_any_at_20": 0.5,
        "precision_strict_at_20": 0.5,
        "tie_rate_at_20": 0.0,
        "recall_at_10": 0.5,
        "hit_rate_at_10": 0.5,
        "precision_at_10": 0.5,
    }
    query_metrics = pd.DataFrame.from_records(
        [
            {
                "method": "R1",
                "fold": 1,
                "size": "240x320",
                "checkpoint_fingerprint": checkpoint_sha256,
                "config_fingerprint": config_hash,
                "scope": "development",
                "query_source": query,
                "gallery_source": gallery,
                "protocol": protocol,
                "query_id": query_id,
                "articleType": "Shirts",
                **metric_defaults,
            }
            for query, gallery in directions
            for protocol in ("primary", "family")
            for query_id in query_ids
        ]
    )
    rankings = pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": 1,
                "query_source": query,
                "gallery_source": gallery,
                "protocol": protocol,
                "query_id": query_id,
                "candidate_id": 100 + rank,
                "distance": rank / 10,
                "rank": rank,
            }
            for query, gallery in directions
            for protocol in ("primary", "family")
            for query_id in query_ids
            for rank in (1, 2)
        ]
    )
    real_quality = learned.evaluate_learned_quality(_splits(), _quality_indexes(), fold=1)
    quality_summary = real_quality.summary.assign(
        checkpoint_fingerprint=checkpoint_sha256,
        config_fingerprint=config_hash,
    )
    query_metrics = real_quality.query_metrics.assign(
        checkpoint_fingerprint=checkpoint_sha256,
        config_fingerprint=config_hash,
    )
    rankings = learned.assemble_learned_rankings(real_quality)
    primary_views, family_views = learned.build_development_views(
        _splits(),
        validation_fold=1,
    )
    real_analysis = learned.evaluate_learned_analysis(
        real_quality,
        primary_views=primary_views,
        family_views=family_views,
        canvas_indexes={
            "wide": _quality_indexes()["v1"],
            "tall": _quality_indexes()["v1"],
        },
        gallery_index=_quality_indexes()["v1"],
        fold=1,
    )
    real_gallery = learned.evaluate_gallery_sources(_splits(), _quality_indexes(), fold=1)
    timing = pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": 1,
                "query_id": query_id,
                "query_source": query,
                "gallery_source": gallery,
                "encoding_seconds": 0.1,
                "search_seconds": 0.1,
                "end_to_end_seconds": 0.2,
            }
            for query, gallery in directions
            for query_id in query_ids
        ]
    )
    return {
        "quality_summary": quality_summary,
        "query_metrics": query_metrics,
        "rankings": rankings,
        "failure_slices": real_analysis.failure_slices,
        "canvas_summary": real_analysis.canvas_summary,
        "canvas_per_query": real_analysis.canvas_per_query,
        "canvas_rankings": learned.assemble_canvas_rankings(real_analysis),
        "examples": learned.assemble_learned_examples(real_analysis, real_quality),
        "timing_samples": timing,
        "timing_summary": pd.concat(
            [
                learned.summarize_timings(rows)
                for _, rows in timing.groupby(
                    ["query_source", "gallery_source"],
                    sort=False,
                )
            ],
            ignore_index=True,
        ),
        "gallery_comparison": real_gallery.comparison,
        "gallery_rankings": learned.assemble_gallery_rankings(real_gallery),
    }


def _source_provenance(
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> dict[str, Any]:
    return {
        source: _exact_provenance(
            source,
            fold=session.validation_fold,
            checkpoint_sha256=checkpoint.sha256,
            config_hash=session.config_hash,
            split_fingerprint=session.split_fingerprint,
        )
        for source in ("teacher", "v1")
    }


def _source_artifact_fixture(
    tmp_path: Path,
    splits: pd.DataFrame,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = np.sort(
        splits.loc[splits["partition"].eq("development"), "id"].to_numpy(dtype=np.int64)
    )
    artifacts: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for source in ("teacher", "v1"):
        directory = tmp_path / "lineage" / source
        feature_dir = directory / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        images = np.zeros((len(ids), 320, 240, 3), dtype=np.uint8)
        bounds = np.tile(np.array([[1, 1, 319, 239]], dtype=np.int32), (len(ids), 1))
        id_digest = hashlib.sha256()
        for product_id in ids:
            id_digest.update(str(int(product_id)).encode())
            id_digest.update(b"\n")
        image_manifest = {
            "schema_version": "1.0.0",
            "scope": "development",
            "source": source,
            "path_column": f"{source}_path",
            "sha_column": f"{source}_sha256",
            "rows": len(ids),
            "array_shape": [len(ids), 320, 240, 3],
            "array_dtype": "uint8",
            "bounds_shape": [len(ids), 4],
            "bounds_dtype": "int32",
            "id_dtype": "int64",
            "id_sha256": id_digest.hexdigest(),
            "source_fingerprint": hashlib.sha256(source.encode()).hexdigest(),
            "contract": CONTRACT.to_dict(),
            "files": {
                "ids": "ids.npy",
                "images": "images.npy",
                "content_bounds": "content_bounds.npy",
            },
        }
        np.save(directory / "ids.npy", ids, allow_pickle=False)
        np.save(directory / "images.npy", images, allow_pickle=False)
        np.save(directory / "content_bounds.npy", bounds, allow_pickle=False)
        image_path = directory / "manifest.json"
        image_path.write_bytes(learned._canonical_json(image_manifest))
        statistics = {
            "source": source,
            "source_fingerprint": image_manifest["source_fingerprint"],
            "validation_fold": session.validation_fold,
            "split_fingerprint": session.split_fingerprint,
            "contract": CONTRACT.to_dict(),
            "mean": [0.25, 0.5, 0.75],
            "std": [0.25, 0.25, 0.25],
        }
        statistics_path = directory / "statistics.json"
        statistics_path.write_bytes(learned._canonical_json(statistics))
        features = np.zeros((len(ids), 128), dtype=np.float32)
        features[:, 0] = 1.0
        np.save(feature_dir / "ids.npy", ids, allow_pickle=False)
        np.save(feature_dir / "features.npy", features, allow_pickle=False)
        identity = {
            "schema_version": learned.LEARNED_EVIDENCE_SCHEMA_VERSION,
            "scope": "development",
            "run_id": session.run_id,
            "run_kind": session.run_kind,
            "method": session.expected_registry_identity.method,
            "checkpoint_sha256": checkpoint.sha256,
            "config_hash": session.config_hash,
            "split_fingerprint": session.split_fingerprint,
            "source": source,
            "image_cache_manifest_sha256": hashlib.sha256(
                learned._canonical_json(image_manifest)
            ).hexdigest(),
            "source_fingerprint": image_manifest["source_fingerprint"],
            "source_statistics_sha256": hashlib.sha256(
                learned._canonical_json(statistics)
            ).hexdigest(),
            "fold": session.validation_fold,
            "contract": CONTRACT.to_dict(),
            "rows": len(ids),
            "dimension": 128,
        }
        identity["feature_cache_identity_sha256"] = hashlib.sha256(
            learned._canonical_json(identity)
        ).hexdigest()
        feature_manifest = {
            **identity,
            "feature_method": session.expected_registry_identity.method,
            "transform_seconds": 0.1,
            "source_bytes": len(ids) * 320 * 240 * 3,
            "ids_sha256": hashlib.sha256((feature_dir / "ids.npy").read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(
                (feature_dir / "features.npy").read_bytes()
            ).hexdigest(),
        }
        feature_path = feature_dir / "manifest.json"
        feature_path.write_bytes(learned._canonical_json(feature_manifest))
        artifacts[source] = learned.LearnedSourceArtifacts(
            image_cache_manifest=image_path,
            statistics=statistics_path,
            feature_cache_manifest=feature_path,
        )
        provenance[source] = learned._provenance_from_identity(identity)
    return artifacts, provenance


def _session_artifact_frames(
    query_ids: list[int],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> dict[str, pd.DataFrame]:
    return _artifact_frames(
        query_ids,
        checkpoint_sha256=checkpoint.sha256,
        config_hash=session.config_hash,
    )


def _artifact_cost(
    checkpoint: CheckpointRecord,
    session: TrainingSessionConfig,
    query_count: int,
    provenance: dict[str, Any] | None = None,
) -> dict[str, object]:
    timing = _complete_timing_summary(timed_queries=query_count)
    timing.loc[timing["metric"].eq("end_to_end"), "value_seconds"] = 0.2
    per_source = {
        source: {
            "source": source,
            "contract": CONTRACT.to_dict(),
            "rows": 31,
            "dimension": 128,
            "payload_bytes": 31 * 128 * 4,
            "index_bytes": 31 * (128 * 4 + 8),
            "build_seconds": 0.1,
            "peak_rss_bytes": 1000,
        }
        for source in ("teacher", "v1")
    }
    index_bytes = {source: int(record["index_bytes"]) for source, record in per_source.items()}
    feature_bytes = {source: int(record["payload_bytes"]) for source, record in per_source.items()}
    total = sum(index_bytes.values())
    record = {
        "schema_version": 1,
        "scope": "development",
        "fold": session.validation_fold,
        "contract": CONTRACT.to_dict(),
        "hardware": {
            "cpu": "test",
            "logical_cores": 1,
            "operating_system": "test",
            "python_version": "3.12",
            "numpy_version": np.__version__,
            "thread_count": 1,
            "thread_environment": {name: "1" for name in learned.THREAD_VARIABLES},
            "native_thread_pools": [
                {
                    "user_api": "blas",
                    "internal_api": "openblas",
                    "num_threads": 1,
                    "prefix": "libopenblas",
                }
            ],
        },
        "warmup_queries": 100,
        "timed_queries": query_count,
        "timing_summary": timing.to_dict("records"),
        "per_source_index_cost": per_source,
        "p95_end_to_end_under_one_second": True,
        "index_under_one_gibibyte": True,
        "method": "R1",
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "checkpoint_sha256": checkpoint.sha256,
        "config_hash": session.config_hash,
        "split_fingerprint": session.split_fingerprint,
        "source_provenance": {
            source: value.to_dict()
            for source, value in (
                provenance or _source_provenance(session, checkpoint)
            ).items()
        },
        "parameters": 1,
        "checkpoint_bytes": checkpoint.path.stat().st_size,
        "feature_bytes": feature_bytes,
        "index_bytes": index_bytes,
        "selected_gallery_policy": "two_view",
        "selected_policy_total_index_bytes": total,
    }
    record["measurement_route"] = learned._MEASUREMENT_ROUTE
    record["measurement_sha256"] = hashlib.sha256(learned._canonical_json(record)).hexdigest()
    return record


def _patch_artifact_cost_builder(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: CheckpointRecord,
    session: TrainingSessionConfig,
    query_count: int,
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def build(
        timing_summary: pd.DataFrame,
        *,
        caches: dict[str, DevelopmentImageCache],
        statistics: dict[str, dict[str, object]],
        session: TrainingSessionConfig,
        checkpoint: CheckpointRecord,
        policy: learned.TimingPolicy,
        fold: int,
        selected_gallery_policy: str,
    ) -> dict[str, object]:
        calls.append(
            {
                "timing_summary": timing_summary,
                "caches": caches,
                "statistics": statistics,
                "session": session,
                "checkpoint": checkpoint,
                "policy": policy,
                "fold": fold,
                "selected_gallery_policy": selected_gallery_policy,
            }
        )
        return _artifact_cost(checkpoint, session, query_count, provenance)

    monkeypatch.setattr(learned, "build_learned_cost_record", build)
    return calls


def test_artifact_writer_is_deterministic_validates_coverage_and_hashes_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    frames = _session_artifact_frames(query_ids, session, checkpoint)
    cost_calls = _patch_artifact_cost_builder(
        monkeypatch, checkpoint, session, len(query_ids), provenance
    )
    writes: list[str] = []
    real_atomic = learned._atomic_write_bytes

    def observe(path: Path, content: bytes) -> None:
        writes.append(path.name)
        real_atomic(path, content)

    monkeypatch.setattr(learned, "_atomic_write_bytes", observe)
    destination = tmp_path / "results/evidence/task4/learned" / session.run_id
    manifest_path = learned.write_learned_artifacts(
        evidence_dir=destination,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=frames,
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
        caveat=learned.TEACHER_SLICE_CAVEAT,
    )
    assert cost_calls[-1]["selected_gallery_policy"] == "two_view"

    assert writes[-1] == "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == session.run_id
    assert manifest["query_count"] == len(query_ids)
    assert manifest["coverage"]["query_metrics_complete"] is True
    assert manifest["caveat"] == learned.TEACHER_SLICE_CAVEAT
    for artifact in manifest["artifacts"]:
        path = destination / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    swaps: list[Path] = []
    real_replace = learned._replace_directory

    def observe_replace(temporary: Path, target: Path) -> None:
        if target.exists():
            assert {path.name: path.read_bytes() for path in target.iterdir()} == before
        swaps.append(target)
        real_replace(temporary, target)

    monkeypatch.setattr(learned, "_replace_directory", observe_replace)
    learned.write_learned_artifacts(
        evidence_dir=destination,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=frames,
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
        caveat=learned.TEACHER_SLICE_CAVEAT,
    )
    after = {path.name: path.read_bytes() for path in destination.iterdir()}
    assert before == after
    assert swaps[-1] == destination

    incomplete = _session_artifact_frames(query_ids, session, checkpoint)
    incomplete["query_metrics"] = incomplete["query_metrics"].iloc[1:]
    with pytest.raises(ValueError, match="exactly once"):
        learned.write_learned_artifacts(
            evidence_dir=tmp_path / "bad",
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=incomplete,
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
            caveat=learned.TEACHER_SLICE_CAVEAT,
        )
    assert not (tmp_path / "bad").exists()

    missing_protocol = _session_artifact_frames(query_ids, session, checkpoint)
    missing_protocol["query_metrics"] = missing_protocol["query_metrics"].loc[
        missing_protocol["query_metrics"]["protocol"].eq("primary")
    ]
    with pytest.raises(ValueError, match="Protocol A/B"):
        learned.write_learned_artifacts(
            evidence_dir=tmp_path / "missing-protocol",
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=missing_protocol,
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
        )


def test_learned_boundary_drops_null_descriptor_fingerprint_and_rejects_a_real_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    _patch_artifact_cost_builder(monkeypatch, checkpoint, session, len(query_ids), provenance)

    frames = _session_artifact_frames(query_ids, session, checkpoint)
    assert "descriptor_fingerprint" in frames["quality_summary"]
    assert frames["quality_summary"]["descriptor_fingerprint"].isna().all()

    destination = tmp_path / "results/evidence/task4/learned" / session.run_id
    manifest_path = learned.write_learned_artifacts(
        evidence_dir=destination,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=frames,
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
        caveat=learned.TEACHER_SLICE_CAVEAT,
    )
    frozen = learned.LEARNED_ARTIFACT_SCHEMAS["quality_summary"]
    written = pd.read_csv(destination / "quality_summary.csv")
    assert tuple(written.columns) == frozen
    manifest = json.loads(manifest_path.read_text())
    recorded = {
        artifact["name"]: artifact.get("columns") for artifact in manifest["artifacts"]
    }
    assert recorded["quality_summary"] == list(frozen)

    for blank_value in (" ", ""):
        blank = _session_artifact_frames(query_ids, session, checkpoint)
        blank["quality_summary"] = blank["quality_summary"].assign(
            descriptor_fingerprint=blank_value
        )
        learned.write_learned_artifacts(
            evidence_dir=destination,
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=blank,
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
            caveat=learned.TEACHER_SLICE_CAVEAT,
        )
        assert tuple(pd.read_csv(destination / "quality_summary.csv").columns) == frozen

    forged = _session_artifact_frames(query_ids, session, checkpoint)
    forged["quality_summary"] = forged["quality_summary"].assign(
        descriptor_fingerprint="f" * 40
    )
    with pytest.raises(ValueError, match="descriptor provenance"):
        learned.write_learned_artifacts(
            evidence_dir=tmp_path / "forged",
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=forged,
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
            caveat=learned.TEACHER_SLICE_CAVEAT,
        )
    assert not (tmp_path / "forged").exists()


def _running_row(session: TrainingSessionConfig) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in RUN_COLUMNS}
    row.update(
        {
            "schema_version": "1",
            "started_at_utc": "2026-08-29T01:02:03Z",
            "status": "running",
            "git_commit": "c" * 40,
            "dirty_tree": False,
            **session.expected_registry_identity.as_dict(),
        }
    )
    return row


def _minimal_valid_manifest(
    tmp_path: Path,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    canonical_splits: pd.DataFrame,
) -> Path:
    evidence_dir = (
        tmp_path
        / "results/evidence/task4/learned"
        / session.run_id
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "quality_summary",
        "query_metrics",
        "rankings",
        "failure_slices",
        "canvas_summary",
        "canvas_per_query",
        "canvas_rankings",
        "examples",
        "timing_samples",
        "timing_summary",
        "gallery_comparison",
        "gallery_rankings",
    )
    artifacts = []
    for name in artifact_names:
        artifact = evidence_dir / f"{name}.csv"
        artifact.write_text("value\n0.5\n", encoding="utf-8")
        artifacts.append(
            {
                "name": name,
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "rows": 1,
                "columns": ["value"],
            }
        )
    cost = evidence_dir / "cost.json"
    cost.write_text("{}\n", encoding="utf-8")
    artifacts.append(
        {
            "name": "cost",
            "path": cost.name,
            "sha256": hashlib.sha256(cost.read_bytes()).hexdigest(),
            "rows": None,
        }
    )
    source_artifacts, source_provenance = _source_artifact_fixture(
        tmp_path, canonical_splits, session, checkpoint
    )
    manifest = {
        "schema_version": learned.LEARNED_EVIDENCE_SCHEMA_VERSION,
        "scope": "development",
        "run_id": session.run_id,
        "run_kind": session.run_kind,
        "method": session.expected_registry_identity.method,
        "fold": session.validation_fold,
        "checkpoint": {
            "epoch": checkpoint.epoch,
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
        },
        "config_hash": session.config_hash,
        "split_fingerprint": session.split_fingerprint,
        "contract": CONTRACT.to_dict(),
        "source_provenance": {
            source: value.to_dict()
            for source, value in source_provenance.items()
        },
        "source_artifacts": {
            source: {
                name: {
                    "path": str(getattr(source_artifacts[source], name)),
                    "sha256": hashlib.sha256(
                        getattr(source_artifacts[source], name).read_bytes()
                    ).hexdigest(),
                }
                for name in (
                    "image_cache_manifest",
                    "statistics",
                    "feature_cache_manifest",
                )
            }
            for source in ("teacher", "v1")
        },
        "query_ids": sorted(
            canonical_splits.loc[
                canonical_splits["cv_fold"].eq(session.validation_fold), "id"
            ].astype(int)
        ),
        "query_count": int(canonical_splits["cv_fold"].eq(session.validation_fold).sum()),
        "coverage": {
            "query_metrics_complete": True,
            "rankings_complete": True,
            "timing_complete": True,
        },
        "caveat": learned.TEACHER_SLICE_CAVEAT,
        "gates": {
            "p95_end_to_end_under_one_second": True,
            "index_under_one_gibibyte": True,
        },
        "selected_metrics": {
            "development_winner_score": 0.6,
            "cross_source_score": 0.5,
            "source_robustness_ratio": 5 / 6,
        },
        "artifacts": artifacts,
    }
    path = evidence_dir / "manifest.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return path


def test_manifest_validation_rejects_identity_and_artifact_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    _patch_artifact_cost_builder(monkeypatch, checkpoint, session, len(query_ids), provenance)
    manifest = learned.write_learned_artifacts(
        evidence_dir=tmp_path / "results/evidence/task4/learned" / session.run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=_session_artifact_frames(query_ids, session, checkpoint),
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
    )

    learned.validate_learned_manifest(
        manifest,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )

    artifact = manifest.parent / "quality_summary.csv"
    original_artifact = artifact.read_bytes()
    original_manifest = json.loads(manifest.read_text())
    artifact.write_text("value\nchanged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        learned.validate_learned_manifest(
            manifest,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )
    artifact.write_bytes(original_artifact)

    wrong_metrics = json.loads(json.dumps(original_manifest))
    wrong_metrics["selected_metrics"]["development_winner_score"] = 0.1
    manifest.write_text(json.dumps(wrong_metrics) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selected metrics"):
        learned.validate_learned_manifest(
            manifest,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )

    query_path = manifest.parent / "query_metrics.csv"
    query_frame = pd.read_csv(query_path)
    query_frame["run_id"] = "other-run"
    query_frame.to_csv(query_path, index=False)
    wrong_provenance = json.loads(json.dumps(original_manifest))
    query_record = next(
        artifact
        for artifact in wrong_provenance["artifacts"]
        if artifact["name"] == "query_metrics"
    )
    query_record["sha256"] = hashlib.sha256(query_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(wrong_provenance) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        learned.validate_learned_manifest(
            manifest,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )


def test_discovery_reconstructs_full_result_and_completion_links_only_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(
        checkpoints=(1, 2),
        split_fingerprint=cv_assignment_digest(splits),
    )
    first = _checkpoint(tmp_path, session, epoch=1, score=0.4)
    best = _checkpoint(tmp_path, session, epoch=2, score=0.7)

    def load(path: Path, **kwargs: Any) -> Any:
        record = first if Path(path) == first.path else best
        assert kwargs["expected_run_id"] == session.run_id
        return SimpleNamespace(
            epoch=record.epoch,
            score=record.score,
            config_hash=session.config_hash,
            split_fingerprint=session.split_fingerprint,
            weight_origin=session.model_metadata.weight_origin,
            parent_run_id=None,
            sha256=record.sha256,
        )

    monkeypatch.setattr(learned, "load_checkpoint", load)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
    registry.append(_running_row(session))
    result = learned.reconstruct_training_result(
        [best.path, first.path],
        session=session,
        registry_row=registry.read()[0],
    )
    assert result.best_checkpoint.epoch == 2
    assert [record.epoch for record in result.checkpoints] == [1, 2]
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, best
    )
    _patch_artifact_cost_builder(monkeypatch, best, session, len(query_ids), provenance)

    manifest = learned.write_learned_artifacts(
        evidence_dir=tmp_path / "results/evidence/task4/learned" / session.run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=best,
        session=session,
        canonical_splits=splits,
        frames=_session_artifact_frames(query_ids, session, best),
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
    )
    completed = learned.complete_learned_evidence(
        registry,
        result=result,
        session=session,
        manifest_path=manifest,
        canonical_splits=splits,
        completed_at="2026-08-29T02:00:00Z",
    )
    assert completed["status"] == "completed"
    assert completed["selected_epoch"] == "2"
    assert completed["checkpoint_path"] == str(best.path)
    assert completed["checkpoint_sha256"] == best.sha256
    assert completed["evidence_manifest_path"] == str(manifest)


def test_completion_rejects_cross_run_checkpoint_and_failure_helper_preserves_original(
    tmp_path: Path,
) -> None:
    session = _session()
    checkpoint = _checkpoint(tmp_path, session)
    result = TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=(replace(checkpoint, run_id="other"),),
        best_checkpoint=replace(checkpoint, run_id="other"),
    )
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.append(_running_row(session))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")

    with pytest.raises(ValueError, match="run ID"):
        learned.complete_learned_evidence(
            registry,
            result=result,
            session=session,
            manifest_path=manifest,
            canonical_splits=_splits(),
            completed_at="2026-08-29T02:00:00Z",
        )
    assert registry.read()[0]["status"] == "running"

    original = RuntimeError("evidence broke")
    learned.record_evidence_failure(
        registry,
        session.run_id,
        original,
        completed_at="2026-08-29T02:00:00Z",
    )
    failed = registry.read()[0]
    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "evidence broke"


def _lock_worker(lock_parent: str, entered: multiprocessing.synchronize.Event, order: Any) -> None:
    with learned._identity_lock(Path(lock_parent), "shared"):
        order.append("entered")
        entered.set()
        time.sleep(0.15)
        order.append("left")


def _feature_build_worker(
    queue: Any,
    cache: DevelopmentImageCache,
    statistics: dict[str, object],
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
    cache_root: str,
) -> None:
    try:
        result = learned.ensure_learned_feature_index(
            cache=cache,
            statistics=statistics,
            session=session,
            checkpoint=checkpoint,
            cache_root=cache_root,
            batch_size=1,
        )
        queue.put(("ok", str(result.cache_dir), result.index.features.shape))
    except BaseException as error:
        queue.put(("error", error.__class__.__name__, str(error)))


def test_feature_cache_identity_lock_serializes_spawned_publishers(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    manager = context.Manager()
    order = manager.list()
    first_entered = context.Event()
    second_entered = context.Event()
    first = context.Process(
        target=_lock_worker,
        args=(str(tmp_path), first_entered, order),
    )
    second = context.Process(
        target=_lock_worker,
        args=(str(tmp_path), second_entered, order),
    )

    first.start()
    assert first_entered.wait(5)
    second.start()
    time.sleep(0.05)
    assert not second_entered.is_set()
    first.join(5)
    second.join(5)

    assert first.exitcode == second.exitcode == 0
    assert list(order) == ["entered", "left", "entered", "left"]


def test_two_spawned_feature_cache_builders_publish_one_valid_identity(
    tmp_path: Path,
) -> None:
    session = _session()
    model = session.build_cpu_model()
    optimizer = training.build_optimizer(model, session.hyperparameters)
    scheduler = training.WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=1,
        config=session.hyperparameters,
    )
    checkpoint = training.save_checkpoint(
        tmp_path / "concurrent.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=training.make_grad_scaler("cpu"),
        epoch=2,
        session=session,
    )
    cache = _cache(tmp_path, "teacher")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    args = (
        queue,
        cache,
        _statistics(cache),
        session,
        checkpoint,
        str(tmp_path / "features"),
    )
    processes = [context.Process(target=_feature_build_worker, args=args) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)

    results = [queue.get(timeout=2) for _ in processes]
    assert [process.exitcode for process in processes] == [0, 0]
    assert all(result[0] == "ok" for result in results)
    assert results[0][1:] == results[1][1:]
    opened = learned.ensure_learned_feature_index(
        cache=cache,
        statistics=_statistics(cache),
        session=session,
        checkpoint=checkpoint,
        cache_root=tmp_path / "features",
    )
    assert opened.index.features.shape == (2, 128)


def test_two_view_requires_128_float32_features_clips_and_applies_family_exclusions() -> None:
    query = np.zeros((1, 128), dtype=np.float32)
    query[0, 0] = 1.0
    gallery = np.zeros((5, 128), dtype=np.float32)
    gallery[:, 0] = 1.0
    ids = np.array([9, 10, 11, 12, 13], dtype=np.int64)
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [9],
                "sha256": ["same-sha"],
                "duplicate_group": ["same-duplicate"],
                "product_family_group": ["family-a"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": ids,
                "sha256": ["query", "same-sha", "other", "valid", "other-family"],
                "duplicate_group": [
                    "query",
                    "other",
                    "same-duplicate",
                    "valid",
                    "other-family",
                ],
                "product_family_group": [
                    "family-a",
                    "family-a",
                    "family-a",
                    "family-a",
                    "family-b",
                ],
            }
        ),
    )
    indexes = {source: _index(source, ids, gallery.copy()) for source in ("teacher", "v1")}

    ranked = learned.rank_two_view_gallery(
        query_index=_index("teacher", np.array([9]), query),
        gallery_indexes=indexes,
        views=views,
        protocol="family",
        max_k=1,
    )

    assert ranked["candidate_id"].tolist() == [12]
    assert ranked["distance"].tolist() == [0.0]
    with pytest.raises(ValueError, match="128"):
        learned.rank_two_view_gallery(
            query_index=_index(
                "teacher",
                np.array([9]),
                np.array([[1.0, 0.0]], dtype=np.float32),
            ),
            gallery_indexes=indexes,
            views=views,
            protocol="family",
            max_k=1,
        )
    with pytest.raises(ValueError, match="float32"):
        learned.rank_two_view_gallery(
            query_index=replace(
                _index("teacher", np.array([9]), query),
                features=query.astype(np.float64),
            ),
            gallery_indexes=indexes,
            views=views,
            protocol="family",
            max_k=1,
        )


def test_timing_binds_source_provenance_threads_and_canonical_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame({"id": [1, 2], "partition": ["development"] * 2})
    views = RetrievalViews(
        queries=pd.DataFrame({"id": [1, 2]}),
        gallery=pd.DataFrame({"id": list(range(200, 220))}),
    )
    indexes = _quality_indexes()
    indexes, encoders = _bound_timing_inputs(tmp_path, indexes)
    monkeypatch.setattr(
        learned.LazyCPUQueryExtractor,
        "__call__",
        lambda self, row: np.r_[1.0, np.zeros(127)].astype(np.float32),
    )
    for variable in learned.THREAD_VARIABLES:
        monkeypatch.setenv(variable, "4")

    samples, _ = learned.benchmark_learned_directions(
        rows,
        views=views,
        indexes=indexes,
        encoders=encoders,
        policy=learned.TimingPolicy(warmup_queries=0),
        clock_ns=iter(range(24)).__next__,
    )

    assert len(samples) == 8
    assert torch.get_num_threads() == 1
    assert all(__import__("os").environ[name] == "1" for name in learned.THREAD_VARIABLES)
    with pytest.raises(ValueError, match="canonical"):
        learned.benchmark_learned_directions(
            rows.iloc[:1],
            views=views,
            indexes=indexes,
            encoders=encoders,
            policy=learned.TimingPolicy(warmup_queries=0),
        )
    wrong_manifest = {
        **_timing_manifest(indexes["teacher"]),
        "source_statistics_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="extractor.*feature-cache provenance"):
        learned.LearnedTimingEncoder(
            extractor=encoders["teacher"].extractor,
            feature_manifest=wrong_manifest,
        )

    teacher_cache = _cache(tmp_path, "teacher")
    with pytest.raises(ValueError, match="source"):
        learned.LazyCPUQueryExtractor(
            source="v1",
            checkpoint=_checkpoint(tmp_path, _session()),
            session=_session(),
            statistics=_statistics(teacher_cache),
            path_column="path",
        )


def test_analysis_rejects_canvas_or_gallery_from_other_checkpoint() -> None:
    quality = learned.evaluate_learned_quality(_splits(), _quality_indexes(), fold=1)
    primary, family = learned.build_development_views(_splits(), validation_fold=1)
    wrong = replace(
        _quality_indexes()["v1"],
        checkpoint_fingerprint="e" * 64,
    )

    with pytest.raises(ValueError, match="provenance"):
        learned.evaluate_learned_analysis(
            quality,
            primary_views=primary,
            family_views=family,
            canvas_indexes={"wide": wrong, "tall": wrong},
            gallery_index=_quality_indexes()["v1"],
            fold=1,
        )


def test_learned_canvas_builder_and_analysis_assembly_use_shared_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    checkpoint = _checkpoint(tmp_path, session)
    cache = _cache(tmp_path, "v1")
    monkeypatch.setattr(
        learned,
        "_load_evidence_model",
        lambda *args, **kwargs: TinyEncoder(),
    )

    canvases = learned.build_learned_canvas_indexes(
        cache,
        statistics=_statistics(cache),
        session=session,
        checkpoint=checkpoint,
        query_ids=[8, 3],
    )

    assert set(canvases) == {"wide", "tall"}
    for index in canvases.values():
        assert index.ids.tolist() == [3, 8]
        assert index.features.shape == (2, 128)
        assert index.features.dtype == np.float32
        assert np.linalg.norm(index.features, axis=1) == pytest.approx([1.0, 1.0])
        assert index.checkpoint_fingerprint == checkpoint.sha256
        assert index.config_fingerprint == session.config_hash

    splits = _splits()
    indexes = _quality_indexes()
    quality = learned.evaluate_learned_quality(splits, indexes, fold=1)
    gallery = learned.evaluate_gallery_sources(splits, indexes, fold=1)
    primary, family = learned.build_development_views(splits, validation_fold=1)
    analysis = learned.evaluate_learned_analysis(
        quality,
        primary_views=primary,
        family_views=family,
        canvas_indexes={"wide": indexes["v1"], "tall": indexes["v1"]},
        gallery_index=indexes["v1"],
        fold=1,
    )
    rankings = learned.assemble_learned_rankings(quality)
    examples = learned.assemble_learned_examples(analysis, quality)

    assert set(
        rankings[["query_source", "gallery_source", "protocol"]].itertuples(
            index=False,
            name=None,
        )
    ) == {
        (*direction, protocol)
        for direction in learned.source_directions()
        for protocol in ("primary", "family")
    }
    assert set(analysis.canvas_rankings) == {"clean", "wide", "tall"}
    assert set(gallery.evaluations) == {
        (query_source, policy)
        for policy in ("teacher", "v1", "two_view")
        for query_source in ("teacher", "v1")
    }
    assert len(gallery.comparison) == 9
    assert not examples.empty
    assert examples["query_source"].eq("v1").all()


def test_evidence_model_loader_moves_each_stage_model_to_requested_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    checkpoint = _checkpoint(tmp_path, session)
    requested = torch.device("cuda:0")
    map_locations: list[torch.device | str] = []
    moved_to: list[torch.device | str] = []

    class DeviceSpyModel(nn.Module):
        def to(self, device: torch.device | str) -> "DeviceSpyModel":  # type: ignore[override]
            moved_to.append(device)
            return self

    def load_checkpoint(*args: object, **kwargs: object) -> object:
        map_locations.append(kwargs["map_location"])  # type: ignore[index]
        return SimpleNamespace(model=DeviceSpyModel())

    monkeypatch.setattr(learned, "load_checkpoint", load_checkpoint)

    feature_model = learned._load_evidence_model(checkpoint, session, requested)
    canvas_model = learned._load_evidence_model(checkpoint, session, requested)

    assert isinstance(feature_model, DeviceSpyModel)
    assert isinstance(canvas_model, DeviceSpyModel)
    assert map_locations == [requested, requested]
    assert moved_to == [requested, requested]


def test_example_assembly_uses_the_selected_queries_worst_canvas_variant() -> None:
    splits = _splits()
    quality = learned.evaluate_learned_quality(splits, _quality_indexes(), fold=1)
    query_id = int(quality.pair_evaluations[("v1", "v1")].primary_rankings["query_id"].iloc[0])
    clean = quality.pair_evaluations[("v1", "v1")].primary_rankings
    analysis = learned.LearnedAnalysis(
        membership=pd.DataFrame({"query_id": [query_id]}),
        failure_slices=pd.DataFrame({"value": [0.5]}),
        canvas_summary=pd.DataFrame({"value": [0.5]}),
        canvas_per_query=pd.DataFrame(
            {
                "query_id": [query_id, query_id],
                "query_variant": ["wide", "tall"],
                "ndcg_change_from_clean": [-0.1, -0.5],
                    "canvas_ndcg_at_10": [0.4, 0.0],
            }
        ),
        example_ids={"canvas_failure": query_id},
        canvas_rankings={"clean": clean, "wide": clean, "tall": clean},
        provenance=dict(quality.provenance),
    )

    examples = learned.assemble_learned_examples(analysis, quality)

    assert examples["query_variant"].unique().tolist() == ["tall"]


def test_manifest_reopens_semantic_artifacts_and_completion_rejects_metric_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    placeholder = _minimal_valid_manifest(tmp_path, session, checkpoint, splits)

    with pytest.raises(ValueError, match="columns"):
        learned.validate_learned_manifest(
            placeholder,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )


def test_artifact_destination_is_run_scoped_and_training_result_name_is_truthful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert learned.reconstruct_training_result.__name__ == "reconstruct_training_result"
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    _patch_artifact_cost_builder(monkeypatch, checkpoint, session, len(query_ids), provenance)
    with pytest.raises(ValueError, match="learned.*run"):
        learned.write_learned_artifacts(
            evidence_dir=tmp_path / "wrong",
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=_session_artifact_frames(query_ids, session, checkpoint),
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
        )


def test_artifact_validation_rejects_incomplete_failure_slice_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    _patch_artifact_cost_builder(monkeypatch, checkpoint, session, len(query_ids), provenance)
    frames = _session_artifact_frames(query_ids, session, checkpoint)
    frames["failure_slices"] = frames["failure_slices"].iloc[:1]

    with pytest.raises(ValueError, match="failure slice.*grid"):
        learned.write_learned_artifacts(
            evidence_dir=tmp_path / "learned" / session.run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=frames,
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
        )


def test_high_level_builder_uses_canonical_inputs_and_completes_registry_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    result = TrainingResult(
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoints=(checkpoint,),
        best_checkpoint=checkpoint,
    )
    registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
    registry.append(_running_row(session))
    development_ids = np.sort(
        splits.loc[splits["partition"].eq("development"), "id"].to_numpy(dtype=np.int64)
    )
    caches: dict[str, DevelopmentImageCache] = {}
    statistics: dict[str, dict[str, object]] = {}
    for source in ("teacher", "v1"):
        images = np.zeros((len(development_ids), 320, 240, 3), dtype=np.uint8)
        bounds = np.tile(np.array([[1, 1, 319, 239]], dtype=np.int32), (len(images), 1))
        manifest = {
            "scope": "development",
            "source": source,
            "source_fingerprint": hashlib.sha256(source.encode()).hexdigest(),
            "contract": CONTRACT.to_dict(),
            "array_shape": list(images.shape),
        }
        cache = DevelopmentImageCache(
            tmp_path / source,
            development_ids,
            images,
            bounds,
            manifest,
        )
        caches[source] = cache
        statistics[source] = _statistics(cache)
        statistics[source]["split_fingerprint"] = session.split_fingerprint
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    query_rows = pd.DataFrame(
        {
            "id": query_ids,
            "partition": ["development"] * len(query_ids),
            "teacher_path": ["teacher.png"] * len(query_ids),
            "v1_path": ["v1.png"] * len(query_ids),
        }
    )
    indexes = {
        source: replace(
            _quality_indexes()[source],
            checkpoint_fingerprint=checkpoint.sha256,
            config_fingerprint=session.config_hash,
            provenance=replace(
                _source_provenance(session, checkpoint)[source],
                source_fingerprint=str(caches[source].manifest["source_fingerprint"]),
                statistics_sha256=hashlib.sha256(
                    learned._canonical_json(statistics[source])
                ).hexdigest(),
            ),
        )
        for source in ("teacher", "v1")
    }
    cached = {
        source: SimpleNamespace(
            index=indexes[source],
            manifest=_timing_manifest(indexes[source]),
            cache_dir=tmp_path / "features" / source,
        )
        for source in ("teacher", "v1")
    }
    quality = learned.LearnedQualityEvaluation(
        summary=pd.DataFrame({"value": [0.5]}),
        query_metrics=pd.DataFrame({"query_id": query_ids}),
        pair_evaluations={},
        selected_metrics={},
    )
    analysis = learned.LearnedAnalysis(
        membership=pd.DataFrame({"query_id": query_ids}),
        failure_slices=pd.DataFrame({"value": [0.5]}),
        canvas_summary=pd.DataFrame({"value": [0.5]}),
        canvas_per_query=pd.DataFrame({"query_id": query_ids}),
        example_ids={"normal_success": query_ids[0]},
    )
    gallery = learned.GallerySourceEvaluation(
        comparison=pd.DataFrame({"value": [0.5]}),
        evaluations={},
    )
    monkeypatch.setattr(
        learned,
        "_utc_z",
        lambda: "2026-08-30T06:13:24Z",
    )
    events: list[str] = []

    monkeypatch.setattr(
        learned,
        "load_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            epoch=checkpoint.epoch,
            score=checkpoint.score,
        ),
    )
    monkeypatch.setattr(
        learned,
        "ensure_learned_feature_index",
        lambda **kwargs: cached[str(kwargs["cache"].manifest["source"])],
    )
    monkeypatch.setattr(learned, "evaluate_learned_quality", lambda *args, **kwargs: quality)
    monkeypatch.setattr(
        learned,
        "build_learned_canvas_indexes",
        lambda *args, **kwargs: {"wide": indexes["v1"], "tall": indexes["v1"]},
    )
    monkeypatch.setattr(learned, "evaluate_learned_analysis", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(learned, "evaluate_gallery_sources", lambda *args, **kwargs: gallery)
    monkeypatch.setattr(
        learned,
        "benchmark_learned_directions",
        lambda *args, **kwargs: (pd.DataFrame({"value": [0.1]}), pd.DataFrame({"value": [0.1]})),
    )
    monkeypatch.setattr(
        learned,
        "build_learned_cost_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pre-write cost handoff")
        ),
    )
    monkeypatch.setattr(
        learned,
        "assemble_learned_rankings",
        lambda value: pd.DataFrame({"value": [0.5]}),
    )
    monkeypatch.setattr(
        learned,
        "assemble_canvas_rankings",
        lambda value: pd.DataFrame({"value": [0.5]}),
    )
    monkeypatch.setattr(
        learned,
        "assemble_gallery_rankings",
        lambda value: pd.DataFrame({"value": [0.5]}),
    )
    monkeypatch.setattr(
        learned,
        "assemble_learned_examples",
        lambda *args: pd.DataFrame({"value": [0.5]}),
    )

    def write(**kwargs: Any) -> Path:
        events.append("write")
        assert kwargs["canonical_splits"] is splits
        assert "cost" not in kwargs
        assert kwargs["selected_gallery_policy"] == "two_view"
        assert kwargs["timing_policy"] == learned.TimingPolicy()
        assert set(kwargs["source_artifacts"]) == {"teacher", "v1"}
        path = Path(kwargs["evidence_dir"]) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return path

    def complete(registry_arg: RunRegistry, **kwargs: Any) -> dict[str, str]:
        events.append("complete")
        assert events == ["write", "complete"]
        assert kwargs["canonical_splits"] is splits
        assert kwargs["completed_at"] == "2026-08-30T06:13:24Z"
        assert registry_arg.read()[0]["status"] == "running"
        return {"status": "completed"}

    monkeypatch.setattr(learned, "write_learned_artifacts", write)
    monkeypatch.setattr(learned, "complete_learned_evidence", complete)

    built = learned.build_learned_evidence(
        registry,
        result=result,
        session=session,
        splits=splits,
        caches=caches,
        statistics=statistics,
        statistics_paths={
            source: tmp_path / f"{source}-statistics.json"
            for source in ("teacher", "v1")
        },
        query_rows=query_rows,
        path_columns={"teacher": "teacher_path", "v1": "v1_path"},
        feature_cache_root=tmp_path / "features",
        evidence_root=tmp_path / "results/evidence/task4",
        selected_gallery_policy="two_view",
        completed_at=None,
    )

    assert built.registry_row["status"] == "completed"
    assert events == ["write", "complete"]

    for stage_name, target_name in (
        ("canvas", "build_learned_canvas_indexes"),
        ("gallery", "evaluate_gallery_sources"),
        ("timing", "benchmark_learned_directions"),
        ("cost", "write_learned_artifacts"),
        ("write", "write_learned_artifacts"),
        ("manifest", "complete_learned_evidence"),
    ):
        with monkeypatch.context() as context:
            context.setattr(
                learned,
                target_name,
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(stage_name)),
            )
            if target_name != "complete_learned_evidence":
                context.setattr(
                    learned,
                    "complete_learned_evidence",
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        AssertionError("registry completed before durable valid evidence")
                    ),
                )
            with pytest.raises(RuntimeError, match=stage_name):
                learned.build_learned_evidence(
                    registry,
                    result=result,
                    session=session,
                    splits=splits,
                    caches=caches,
                    statistics=statistics,
                    statistics_paths={
                        source: tmp_path / f"{source}-statistics.json"
                        for source in ("teacher", "v1")
                    },
                    query_rows=query_rows,
                    path_columns={"teacher": "teacher_path", "v1": "v1_path"},
                    feature_cache_root=tmp_path / "features",
                    evidence_root=tmp_path / "results/evidence/task4",
                    selected_gallery_policy="two_view",
                    completed_at=None,
                )
            assert registry.read()[0]["status"] == "running"

    sealed_rows = query_rows.copy()
    sealed_rows.loc[sealed_rows.index[0], "partition"] = "holdout"
    monkeypatch.setattr(
        learned,
        "ensure_learned_feature_index",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache opened")),
    )
    with pytest.raises(ValueError, match="development"):
        learned.build_learned_evidence(
            registry,
            result=result,
            session=session,
            splits=splits,
            caches=caches,
            statistics=statistics,
            statistics_paths={
                source: tmp_path / f"{source}-statistics.json"
                for source in ("teacher", "v1")
            },
            query_rows=sealed_rows,
            path_columns={"teacher": "teacher_path", "v1": "v1_path"},
            feature_cache_root=tmp_path / "features",
            evidence_root=tmp_path / "results/evidence/task4",
            selected_gallery_policy="two_view",
            completed_at="2026-08-29T02:00:00Z",
        )


def _exact_provenance(
    source: str,
    *,
    statistics: str | None = None,
    fold: int = 1,
    checkpoint_sha256: str = "c" * 64,
    config_hash: str = "d" * 64,
    split_fingerprint: str = SPLIT_SHA,
) -> Any:
    return learned.LearnedProvenance(
        schema_version=learned.LEARNED_EVIDENCE_SCHEMA_VERSION,
        run_id="run-1",
        run_kind="candidate",
        method="R1",
        fold=fold,
        checkpoint_sha256=checkpoint_sha256,
        config_hash=config_hash,
        split_fingerprint=split_fingerprint,
        source=source,
        image_cache_manifest_sha256=("1" if source == "teacher" else "2") * 64,
        source_fingerprint=("3" if source == "teacher" else "4") * 64,
        statistics_sha256=statistics
        or (("5" if source == "teacher" else "6") * 64),
        feature_cache_identity_sha256=("7" if source == "teacher" else "8") * 64,
    )


def test_artifacts_enforce_frozen_topk_cross_file_summaries_and_numeric_contracts(
    tmp_path: Path,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)

    short_rankings = _session_artifact_frames(query_ids, session, checkpoint)
    short_rankings["rankings"] = short_rankings["rankings"].loc[
        short_rankings["rankings"]["rank"].le(2)
    ]
    with pytest.raises(ValueError, match="Top-20.*Top-10"):
        learned._validate_artifact_inputs(
            short_rankings,
            expected_query_ids=query_ids,
            session=session,
            checkpoint=checkpoint,
            primary_views=learned.build_development_views(splits, validation_fold=1)[0],
            family_views=learned.build_development_views(splits, validation_fold=1)[1],
        )

    inconsistent_timing = _session_artifact_frames(query_ids, session, checkpoint)
    inconsistent_timing["timing_summary"].loc[
        inconsistent_timing["timing_summary"]["metric"].eq("encoding"),
        "value_seconds",
    ] = 0.9
    with pytest.raises(ValueError, match="timing summary.*samples"):
        learned._validate_artifact_inputs(
            inconsistent_timing,
            expected_query_ids=query_ids,
            session=session,
            checkpoint=checkpoint,
            primary_views=learned.build_development_views(splits, validation_fold=1)[0],
            family_views=learned.build_development_views(splits, validation_fold=1)[1],
        )

    invalid_numbers = _session_artifact_frames(query_ids, session, checkpoint)
    invalid_numbers["quality_summary"]["query_count"] = invalid_numbers[
        "quality_summary"
    ]["query_count"].astype(float)
    invalid_numbers["quality_summary"].loc[0, "query_count"] = 1.5
    invalid_numbers["failure_slices"].loc[0, "value"] = 2.0
    invalid_numbers["canvas_summary"].loc[1, "ndcg_change_from_clean"] = 2.0
    invalid_numbers["examples"].loc[0, "value"] = -1.0
    with pytest.raises(ValueError, match="numeric"):
        learned._validate_artifact_inputs(
            invalid_numbers,
            expected_query_ids=query_ids,
            session=session,
            checkpoint=checkpoint,
            primary_views=learned.build_development_views(splits, validation_fold=1)[0],
            family_views=learned.build_development_views(splits, validation_fold=1)[1],
        )


def test_manifest_schema_is_exact_and_evidence_root_is_repository_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    frames = _session_artifact_frames(query_ids, session, checkpoint)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    _patch_artifact_cost_builder(monkeypatch, checkpoint, session, len(query_ids), provenance)

    with pytest.raises(ValueError, match="results/evidence/task4/learned"):
        learned.write_learned_artifacts(
            evidence_dir=tmp_path / "arbitrary" / "learned" / session.run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=frames,
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
        )

    monkeypatch.setattr(learned, "ROOT", tmp_path)
    manifest_path = learned.write_learned_artifacts(
        evidence_dir=tmp_path / "results/evidence/task4/learned" / session.run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=frames,
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest fields"):
        learned.validate_learned_manifest(
            manifest_path,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )


def test_timing_controls_live_native_thread_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str | None]] = []

    class Limits:
        def __init__(self, *, limits: int, user_api: str | None = None) -> None:
            calls.append((limits, user_api))

        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(learned, "threadpool_limits", Limits, raising=False)
    monkeypatch.setattr(
        learned,
        "threadpool_info",
        lambda: [
            {"user_api": "blas", "num_threads": 1},
            {"user_api": "openmp", "num_threads": 1},
        ],
        raising=False,
    )
    rows = pd.DataFrame({"id": [1, 2], "partition": ["development"] * 2})
    views = RetrievalViews(
        queries=pd.DataFrame({"id": [1, 2]}),
        gallery=pd.DataFrame({"id": list(range(200, 220))}),
    )
    indexes = _quality_indexes()
    indexes, encoders = _bound_timing_inputs(tmp_path, indexes)
    monkeypatch.setattr(
        learned.LazyCPUQueryExtractor,
        "__call__",
        lambda self, row: np.r_[1.0, np.zeros(127)].astype(np.float32),
    )
    learned.benchmark_learned_directions(
        rows,
        views=views,
        indexes=indexes,
        encoders=encoders,
        policy=learned.TimingPolicy(warmup_queries=0),
        clock_ns=iter(range(24)).__next__,
    )

    assert calls == [(1, None)]


def test_two_view_query_requires_the_same_exact_learned_provenance() -> None:
    query = np.zeros((1, 128), dtype=np.float32)
    query[0, 0] = 1.0
    gallery = np.zeros((21, 128), dtype=np.float32)
    gallery[:, 0] = 1.0
    ids = np.arange(100, 121, dtype=np.int64)
    views = RetrievalViews(
        queries=pd.DataFrame({"id": [100], "sha256": ["q"], "duplicate_group": ["q"]}),
        gallery=pd.DataFrame(
            {
                "id": ids,
                "sha256": [f"sha-{value}" for value in ids],
                "duplicate_group": [f"dup-{value}" for value in ids],
            }
        ),
    )
    teacher = replace(_index("teacher", ids, gallery), provenance=_exact_provenance("teacher"))
    v1 = replace(_index("v1", ids, gallery), provenance=_exact_provenance("v1"))
    matching_query = replace(
        _index("teacher", np.array([100]), query),
        provenance=_exact_provenance("teacher"),
    )

    ranked = learned.rank_two_view_gallery(
        query_index=matching_query,
        gallery_indexes={"teacher": teacher, "v1": v1},
        views=views,
        protocol="primary",
        max_k=20,
    )
    assert len(ranked) == 20

    wrong_query = replace(
        matching_query,
        provenance=_exact_provenance("teacher", statistics="f" * 64),
    )
    with pytest.raises(ValueError, match="query.*provenance"):
        learned.rank_two_view_gallery(
            query_index=wrong_query,
            gallery_indexes={"teacher": teacher, "v1": v1},
            views=views,
            protocol="primary",
            max_k=20,
        )


def test_cost_rows_bind_to_canonical_development_count(tmp_path: Path) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    cost = _artifact_cost(checkpoint, session, len(splits.loc[splits["cv_fold"].eq(1)]))
    cost["per_source_index_cost"]["teacher"]["rows"] -= 1
    timing_summary = _complete_timing_summary(
        timed_queries=len(splits.loc[splits["cv_fold"].eq(1)])
    )
    timing_summary.loc[timing_summary["metric"].eq("end_to_end"), "value_seconds"] = 0.2

    with pytest.raises(ValueError, match="canonical development rows"):
        learned._validate_learned_cost(
            cost,
            session=session,
            checkpoint=checkpoint,
            timing_summary=timing_summary,
            canonical_development_rows=len(splits),
        )


def test_real_components_assemble_a_semantically_valid_evidence_package(
    tmp_path: Path,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    frames = _session_artifact_frames(query_ids, session, checkpoint)
    primary, family = learned.build_development_views(splits, validation_fold=1)

    validated = learned._validate_artifact_inputs(
        frames,
        expected_query_ids=query_ids,
        session=session,
        checkpoint=checkpoint,
        primary_views=primary,
        family_views=family,
    )

    counts = validated["rankings"].groupby(
        ["query_source", "gallery_source", "protocol", "query_id"]
    ).size()
    assert set(counts.loc[counts.index.get_level_values("protocol") == "primary"]) == {20}
    assert set(counts.loc[counts.index.get_level_values("protocol") == "family"]) == {10}
    assert len(validated["failure_slices"]) == 24
    assert len(validated["canvas_per_query"]) == 3 * len(query_ids)


def _validate_frames(
    frames: dict[str, pd.DataFrame],
    splits: pd.DataFrame,
    session: TrainingSessionConfig,
    checkpoint: CheckpointRecord,
) -> dict[str, pd.DataFrame]:
    primary, family = learned.build_development_views(
        splits,
        validation_fold=session.validation_fold,
    )
    return learned._validate_artifact_inputs(
        frames,
        expected_query_ids=primary.queries["id"].astype(int).tolist(),
        session=session,
        checkpoint=checkpoint,
        primary_views=primary,
        family_views=family,
    )


def test_semantic_validation_rejects_rank_distance_and_numeric_id_disagreement(
    tmp_path: Path,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    frames = _session_artifact_frames(
        splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist(),
        session,
        checkpoint,
    )
    group = frames["rankings"].loc[
        frames["rankings"]["query_source"].eq("teacher")
        & frames["rankings"]["gallery_source"].eq("teacher")
        & frames["rankings"]["protocol"].eq("primary")
        & frames["rankings"]["query_id"].eq(100)
    ].sort_values("rank")
    first, second = group.index[:2]
    frames["rankings"].loc[first, "distance"] = 1.0
    frames["rankings"].loc[second, "distance"] = 0.0

    with pytest.raises(ValueError, match="distance.*candidate ID"):
        _validate_frames(frames, splits, session, checkpoint)


def test_semantic_validation_recomputes_canvas_summary_from_query_rows(
    tmp_path: Path,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    frames = _session_artifact_frames(
        splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist(),
        session,
        checkpoint,
    )
    frames["canvas_summary"].loc[
        frames["canvas_summary"]["query_variant"].eq("wide"), "ndcg_at_10"
    ] = 0.25
    clean = float(
        frames["canvas_summary"].loc[
            frames["canvas_summary"]["query_variant"].eq("clean"), "ndcg_at_10"
        ].iloc[0]
    )
    frames["canvas_summary"].loc[
        frames["canvas_summary"]["query_variant"].eq("wide"), "ndcg_change_from_clean"
    ] = 0.25 - clean

    with pytest.raises(ValueError, match="canvas summary.*per-query"):
        _validate_frames(frames, splits, session, checkpoint)


def test_semantic_validation_recomputes_canvas_query_metrics_and_overlap_from_rankings(
    tmp_path: Path,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    frames = _session_artifact_frames(query_ids, session, checkpoint)
    target_query = query_ids[0]
    forged = (
        frames["canvas_per_query"]["query_variant"].eq("wide")
        & frames["canvas_per_query"]["query_id"].eq(target_query)
    )
    frames["canvas_per_query"].loc[forged, "top10_overlap"] = 0.0
    wide = frames["canvas_per_query"]["query_variant"].eq("wide")
    frames["canvas_summary"].loc[
        frames["canvas_summary"]["query_variant"].eq("wide"),
        "mean_top10_overlap",
    ] = frames["canvas_per_query"].loc[wide, "top10_overlap"].mean()

    with pytest.raises(ValueError, match="canvas.*rankings"):
        _validate_frames(frames, splits, session, checkpoint)


def test_semantic_validation_requires_exact_recomputed_gallery_rows(
    tmp_path: Path,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    frames = _session_artifact_frames(
        splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist(),
        session,
        checkpoint,
    )
    selected = (
        frames["gallery_comparison"]["query_source"].eq("equal_teacher_v1_mean")
        & frames["gallery_comparison"]["gallery_policy"].eq("two_view")
    )
    frames["gallery_comparison"].loc[selected, "value"] = 0.25

    with pytest.raises(ValueError, match="gallery comparison.*source rows"):
        _validate_frames(frames, splits, session, checkpoint)


def test_semantic_validation_ties_examples_to_deterministic_ids_metrics_and_rankings(
    tmp_path: Path,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    frames = _session_artifact_frames(
        splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist(),
        session,
        checkpoint,
    )
    frames["examples"].loc[0, "query_id"] = 110

    with pytest.raises(ValueError, match="examples.*deterministic"):
        _validate_frames(frames, splits, session, checkpoint)


def _undefined_query_frames(
    tmp_path: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    TrainingSessionConfig,
    CheckpointRecord,
]:
    splits = _splits()
    query = splits["id"].eq(100)
    splits.loc[query, ["articleType", "baseColour"]] = ["UndefinedType", "Orange"]
    splits.loc[query, "product_family_group"] = "singleton-family"
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    indexes = _quality_indexes()
    indexes = {
        source: replace(
            index,
            provenance=_exact_provenance(
                source,
                checkpoint_sha256="c" * 64,
                config_hash="d" * 64,
                split_fingerprint=session.split_fingerprint,
            ),
        )
        for source, index in indexes.items()
    }
    quality = learned.evaluate_learned_quality(splits, indexes, fold=1)
    primary, family = learned.build_development_views(splits, validation_fold=1)
    analysis = learned.evaluate_learned_analysis(
        quality,
        primary_views=primary,
        family_views=family,
        canvas_indexes={"wide": indexes["v1"], "tall": indexes["v1"]},
        gallery_index=indexes["v1"],
        fold=1,
    )
    gallery = learned.evaluate_gallery_sources(splits, indexes, fold=1)
    query_ids = primary.queries["id"].astype(int).tolist()
    frames = _session_artifact_frames(query_ids, session, checkpoint)
    frames.update(
        {
            "quality_summary": quality.summary.assign(
                checkpoint_fingerprint=checkpoint.sha256,
                config_fingerprint=session.config_hash,
            ),
            "query_metrics": quality.query_metrics.assign(
                checkpoint_fingerprint=checkpoint.sha256,
                config_fingerprint=session.config_hash,
            ),
            "rankings": learned.assemble_learned_rankings(quality),
            "failure_slices": analysis.failure_slices,
            "canvas_summary": analysis.canvas_summary,
            "canvas_per_query": analysis.canvas_per_query,
            "canvas_rankings": learned.assemble_canvas_rankings(analysis),
            "examples": learned.assemble_learned_examples(analysis, quality),
            "gallery_comparison": gallery.comparison,
            "gallery_rankings": learned.assemble_gallery_rankings(gallery),
        }
    )
    return frames, splits, session, checkpoint


def test_valid_undefined_queries_keep_scorable_counts_and_null_canvas_values(
    tmp_path: Path,
) -> None:
    frames, splits, session, checkpoint = _undefined_query_frames(tmp_path)

    validated = _validate_frames(frames, splits, session, checkpoint)

    primary_ndcg = validated["quality_summary"].loc[
        validated["quality_summary"]["protocol"].eq("primary")
        & validated["quality_summary"]["metric"].eq("ndcg")
        & validated["quality_summary"]["aggregation"].eq("query_mean")
    ]
    family_recall = validated["quality_summary"].loc[
        validated["quality_summary"]["protocol"].eq("family")
        & validated["quality_summary"]["metric"].eq("recall")
    ]
    assert set(primary_ndcg["query_count"]) == {10}
    assert set(family_recall["query_count"]) == {10}
    undefined_canvas = validated["canvas_per_query"].loc[
        validated["canvas_per_query"]["query_id"].eq(100)
    ]
    assert undefined_canvas["canvas_ndcg_at_10"].isna().all()
    assert undefined_canvas.loc[
        undefined_canvas["query_variant"].eq("clean"), "ndcg_change_from_clean"
    ].eq(0.0).all()
    assert undefined_canvas.loc[
        ~undefined_canvas["query_variant"].eq("clean"), "ndcg_change_from_clean"
    ].isna().all()


def test_repository_fold_one_contains_the_reviewed_undefined_query_counts() -> None:
    splits = pd.read_csv(SPLITS_CSV)
    development = splits.loc[splits["partition"].eq("development")].copy()
    queries = development.loc[pd.to_numeric(development["cv_fold"]).eq(1)].copy()
    gallery = development.loc[pd.to_numeric(development["cv_fold"]).ne(1)].copy()
    primary_missing = sum(
        not gallery["articleType"].eq(row["articleType"]).any()
        for _, row in queries.iterrows()
    )
    family_missing = sum(
        not queries.loc[
            family_candidate_mask(row, queries),
            "product_family_group",
        ]
        .eq(row["product_family_group"])
        .any()
        for _, row in queries.iterrows()
    )

    assert len(queries) == 6556
    assert primary_missing == 4
    assert family_missing == 3937


def test_csv_serialization_preserves_float32_ranking_order() -> None:
    distances = np.array(
        [
            0.009999999776482582,
            0.010000000707805157,
            0.010000001639127731,
            0.010000002570450306,
        ],
        dtype=np.float32,
    )
    frame = pd.DataFrame(
        {
            "candidate_id": [40, 30, 20, 10],
            "distance": distances,
            "rank": [1, 2, 3, 4],
        }
    )

    reopened = pd.read_csv(io.BytesIO(learned._csv_bytes(frame)))

    assert reopened.sort_values("distance", kind="mergesort")["rank"].tolist() == [
        1,
        2,
        3,
        4,
    ]
    assert reopened["distance"].nunique() == len(distances)


def test_timing_encoder_cannot_attach_claimed_provenance_to_an_arbitrary_callable() -> None:
    provenance = _exact_provenance("teacher")

    with pytest.raises(TypeError):
        learned.LearnedTimingEncoder(
            provenance=provenance,
            encode=lambda _row: np.r_[1.0, np.zeros(127)].astype(np.float32),
        )


def test_writer_rejects_caller_authored_cost_even_when_values_and_hashes_look_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'cost'"):
        learned.write_learned_artifacts(
            evidence_dir=tmp_path / "results/evidence/task4/learned" / session.run_id,
            run_id=session.run_id,
            run_kind=session.run_kind,
            checkpoint=checkpoint,
            session=session,
            canonical_splits=splits,
            frames=_session_artifact_frames(query_ids, session, checkpoint),
            cost=_artifact_cost(checkpoint, session, len(query_ids), provenance),
            source_artifacts=source_artifacts,
            selected_gallery_policy="two_view",
        )


def test_public_writer_api_has_no_authored_cost_handoff() -> None:
    assert "cost" not in inspect.signature(learned.write_learned_artifacts).parameters


def test_writer_measures_cost_from_reopened_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    calls: list[dict[str, object]] = []

    def build(
        timing_summary: pd.DataFrame,
        *,
        caches: dict[str, DevelopmentImageCache],
        statistics: dict[str, dict[str, object]],
        session: TrainingSessionConfig,
        checkpoint: CheckpointRecord,
        policy: learned.TimingPolicy,
        fold: int,
        selected_gallery_policy: str,
    ) -> dict[str, object]:
        calls.append(
            {
                "caches": caches,
                "statistics": statistics,
                "fold": fold,
                "policy": policy,
                "selected_gallery_policy": selected_gallery_policy,
            }
        )
        return _artifact_cost(checkpoint, session, len(query_ids), provenance)

    monkeypatch.setattr(learned, "build_learned_cost_record", build)

    learned.write_learned_artifacts(
        evidence_dir=tmp_path / "results/evidence/task4/learned" / session.run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=_session_artifact_frames(query_ids, session, checkpoint),
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
        timing_policy=learned.TimingPolicy(warmup_queries=0),
    )

    assert len(calls) == 1
    assert calls[0]["selected_gallery_policy"] == "two_view"
    assert calls[0]["fold"] == session.validation_fold
    assert calls[0]["policy"] == learned.TimingPolicy(warmup_queries=0)
    reopened_caches = calls[0]["caches"]
    reopened_statistics = calls[0]["statistics"]
    assert all(
        reopened_caches[source].cache_dir == source_artifacts[source].image_cache_manifest.parent
        for source in ("teacher", "v1")
    )
    assert {
        source: learned._build_learned_provenance(
            reopened_caches[source],
            reopened_statistics[source],
            session,
            checkpoint,
        )
        for source in ("teacher", "v1")
    } == provenance


def test_cost_validation_reloads_selected_checkpoint_model_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "_load_evidence_model", lambda *args: TinyEncoder())
    cost = _artifact_cost(
        checkpoint,
        session,
        len(splits.loc[splits["cv_fold"].eq(1)]),
    )
    cost["parameters"] = 2
    cost["measurement_sha256"] = hashlib.sha256(
        learned._canonical_json(
            {key: value for key, value in cost.items() if key != "measurement_sha256"}
        )
    ).hexdigest()

    timing_summary = _complete_timing_summary(
        timed_queries=len(splits.loc[splits["cv_fold"].eq(1)])
    )
    timing_summary.loc[
        timing_summary["metric"].eq("end_to_end"),
        "value_seconds",
    ] = 0.2

    with pytest.raises(ValueError, match="parameter"):
        learned._validate_learned_cost(
            cost,
            session=session,
            checkpoint=checkpoint,
            timing_summary=timing_summary,
            canonical_development_rows=len(
                splits.loc[splits["partition"].eq("development")]
            ),
        )


def test_public_provenance_boundary_reopens_cache_statistics_and_feature_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "_load_evidence_model", lambda *args: TinyEncoder())
    artifacts: dict[str, Any] = {}
    for source in ("teacher", "v1"):
        cache = _cache(tmp_path, source)
        _write_cache_files(cache)
        statistics = _statistics(cache)
        statistics_path = tmp_path / f"{source}-statistics.json"
        statistics_path.write_bytes(learned._canonical_json(statistics))
        cached = learned.ensure_learned_feature_index(
            cache=cache,
            statistics=statistics,
            session=session,
            checkpoint=checkpoint,
            cache_root=tmp_path / "features",
        )
        artifacts[source] = learned.LearnedSourceArtifacts(
            image_cache_manifest=cache.cache_dir / "manifest.json",
            statistics=statistics_path,
            feature_cache_manifest=cached.cache_dir / "manifest.json",
        )

    changed = json.loads(Path(artifacts["teacher"].statistics).read_text())
    changed["mean"][0] = 0.75
    Path(artifacts["teacher"].statistics).write_bytes(learned._canonical_json(changed))

    with pytest.raises(ValueError, match="statistics.*feature-cache"):
        learned._reopen_source_artifacts(
            artifacts,
            session=session,
            checkpoint=checkpoint,
        )


def test_source_artifact_reopening_requires_canonical_image_cache_arrays(
    tmp_path: Path,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    source_artifacts, _ = _source_artifact_fixture(tmp_path, splits, session, checkpoint)
    (source_artifacts["teacher"].image_cache_manifest.parent / "ids.npy").unlink()

    with pytest.raises(ValueError, match="image cache"):
        learned._reopen_source_artifacts(
            source_artifacts,
            session=session,
            checkpoint=checkpoint,
        )


def test_source_artifact_reopening_recomputes_feature_cache_identity_digest(
    tmp_path: Path,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    source_artifacts, _ = _source_artifact_fixture(tmp_path, splits, session, checkpoint)
    manifest_path = source_artifacts["v1"].feature_cache_manifest
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_cache_identity_sha256"] = "f" * 64
    manifest_path.write_bytes(learned._canonical_json(manifest))

    with pytest.raises(ValueError, match="feature-cache identity"):
        learned._reopen_source_artifacts(
            source_artifacts,
            session=session,
            checkpoint=checkpoint,
        )


def test_manifest_reopening_revalidates_external_provenance_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    checkpoint = _checkpoint(tmp_path, session)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    source_artifacts, provenance = _source_artifact_fixture(
        tmp_path, splits, session, checkpoint
    )
    _patch_artifact_cost_builder(monkeypatch, checkpoint, session, len(query_ids), provenance)
    manifest = learned.write_learned_artifacts(
        evidence_dir=tmp_path / "results/evidence/task4/learned" / session.run_id,
        run_id=session.run_id,
        run_kind=session.run_kind,
        checkpoint=checkpoint,
        session=session,
        canonical_splits=splits,
        frames=_session_artifact_frames(query_ids, session, checkpoint),
        source_artifacts=source_artifacts,
        selected_gallery_policy="two_view",
    )
    changed = json.loads(source_artifacts["v1"].statistics.read_text())
    changed["std"][0] = 0.5
    source_artifacts["v1"].statistics.write_bytes(learned._canonical_json(changed))

    with pytest.raises(ValueError, match="provenance"):
        learned.validate_learned_manifest(
            manifest,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )


def test_real_restart_build_write_reopen_and_complete_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _splits()
    session = _session(split_fingerprint=cv_assignment_digest(splits))
    model = session.build_cpu_model()
    optimizer = training.build_optimizer(model, session.hyperparameters)
    scheduler = training.WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=1,
        config=session.hyperparameters,
    )
    checkpoint = training.save_checkpoint(
        tmp_path / "restart.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=training.make_grad_scaler("cpu"),
        epoch=2,
        session=session,
        score=0.5,
    )
    del model, optimizer, scheduler
    registry_path = tmp_path / "runs.csv"
    RunRegistry(registry_path, project_root=tmp_path).append(_running_row(session))
    restarted_registry = RunRegistry(registry_path, project_root=tmp_path)
    result = learned.reconstruct_training_result(
        [checkpoint.path],
        session=session,
        registry_row=restarted_registry.read()[0],
    )

    development_ids = np.sort(
        splits.loc[splits["partition"].eq("development"), "id"].to_numpy(dtype=np.int64)
    )
    caches: dict[str, DevelopmentImageCache] = {}
    statistics: dict[str, dict[str, object]] = {}
    statistics_paths: dict[str, Path] = {}
    for source, colour in (("teacher", 64), ("v1", 192)):
        images = np.full((len(development_ids), 320, 240, 3), colour, dtype=np.uint8)
        bounds = np.tile(np.array([[1, 1, 319, 239]], dtype=np.int32), (len(images), 1))
        cache_dir = tmp_path / "cache" / source
        cache_dir.mkdir(parents=True)
        manifest = {
            "scope": "development",
            "source": source,
            "source_fingerprint": hashlib.sha256(source.encode()).hexdigest(),
            "contract": CONTRACT.to_dict(),
            "array_shape": list(images.shape),
        }
        cache = DevelopmentImageCache(
            cache_dir,
            development_ids,
            images,
            bounds,
            manifest,
        )
        _write_cache_files(cache)
        caches[source] = cache
        statistics[source] = _statistics(cache)
        statistics[source]["split_fingerprint"] = session.split_fingerprint
        statistics_path = tmp_path / f"{source}-statistics.json"
        statistics_path.write_bytes(learned._canonical_json(statistics[source]))
        statistics_paths[source] = statistics_path
        Image.fromarray(np.full((80, 60, 3), colour, dtype=np.uint8)).save(
            tmp_path / f"{source}.png"
        )
    query_ids = splits.loc[splits["cv_fold"].eq(1), "id"].astype(int).tolist()
    query_rows = pd.DataFrame(
        {
            "id": query_ids,
            "partition": "development",
            "teacher_path": str(tmp_path / "teacher.png"),
            "v1_path": str(tmp_path / "v1.png"),
        }
    )

    monkeypatch.setattr(learned, "ROOT", tmp_path)
    real_load_evidence_model = learned._load_evidence_model
    loaded_checkpoint_paths: list[Path] = []

    def load_selected_checkpoint_model(*args: Any, **kwargs: Any) -> nn.Module:
        loaded_checkpoint_paths.append(Path(args[0].path))
        return real_load_evidence_model(*args, **kwargs)

    monkeypatch.setattr(learned, "_load_evidence_model", load_selected_checkpoint_model)

    def measure_without_process(
        cache: DevelopmentImageCache,
        *,
        statistics: dict[str, object],
        session: TrainingSessionConfig,
        checkpoint: CheckpointRecord,
    ) -> Any:
        model = learned._load_evidence_model(checkpoint, session, "cpu")
        index = learned.encode_development_cache(
            cache,
            model=model,
            statistics=statistics,
            session=session,
            checkpoint=checkpoint,
            batch_size=8,
        )
        return learned.IndexCost(
            source=index.source,
            contract=index.contract,
            rows=len(index.ids),
            dimension=128,
            payload_bytes=index.features.nbytes,
            index_bytes=index.features.nbytes + index.ids.nbytes,
            build_seconds=index.transform_seconds,
            peak_rss_bytes=1,
        )

    monkeypatch.setattr(learned, "measure_learned_index_build", measure_without_process)
    expected_parameters = sum(
        parameter.numel()
        for parameter in real_load_evidence_model(checkpoint, session, "cpu").parameters()
    )
    built = learned.build_learned_evidence(
        restarted_registry,
        result=result,
        session=session,
        splits=splits,
        caches=caches,
        statistics=statistics,
        statistics_paths=statistics_paths,
        query_rows=query_rows,
        path_columns={"teacher": "teacher_path", "v1": "v1_path"},
        feature_cache_root=tmp_path / "features",
        evidence_root=tmp_path / "results/evidence/task4",
        selected_gallery_policy="two_view",
        completed_at="2026-08-29T03:00:00Z",
        timing_policy=learned.TimingPolicy(warmup_queries=0),
    )

    reopened = learned.validate_learned_manifest(
        built.manifest_path,
        session=session,
        checkpoint=result.best_checkpoint,
        canonical_splits=splits,
    )
    assert reopened["run_id"] == session.run_id
    assert built.registry_row["status"] == "completed"
    assert int(built.registry_row["parameter_count"]) == expected_parameters
    assert restarted_registry.read()[0]["status"] == "completed"
    assert set(loaded_checkpoint_paths) == {checkpoint.path}
