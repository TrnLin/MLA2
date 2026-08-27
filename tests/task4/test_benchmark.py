from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import fashion.task4 as task4
from fashion.task4.benchmark import (
    IndexCost,
    TimingPolicy,
    benchmark_source_direction,
    build_cost_record,
    build_protocol_a_search,
    build_query_encoder,
    measure_index_build,
    summarize_timings,
)
from fashion.task4.preprocessing import PreprocessedImage, PreprocessingContract
from fashion.task4.preprocessing_experiment import FeatureIndex
from fashion.task4.protocol import RetrievalViews


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values: Iterator[int] = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def test_benchmark_api_is_exported_from_task4_package() -> None:
    assert task4.TimingPolicy is TimingPolicy
    assert task4.IndexCost is IndexCost
    assert task4.benchmark_source_direction is benchmark_source_direction
    assert task4.summarize_timings is summarize_timings
    assert task4.measure_index_build is measure_index_build
    assert task4.build_cost_record is build_cost_record


def test_timing_keeps_every_query_and_reports_linear_percentiles() -> None:
    samples = pd.DataFrame(
        {
            "query_source": ["teacher"] * 4,
            "gallery_source": ["v1"] * 4,
            "encoding_seconds": [0.1, 0.2, 0.3, 10.0],
            "search_seconds": [0.01, 0.02, 0.03, 1.0],
            "end_to_end_seconds": [0.11, 0.22, 0.33, 11.0],
        }
    )

    summary = summarize_timings(samples)

    p95 = summary.query("metric == 'end_to_end' and percentile == 'p95'")
    assert p95["value_seconds"].item() == pytest.approx(
        np.quantile(samples["end_to_end_seconds"], 0.95)
    )
    assert summary["timed_queries"].unique().tolist() == [4]
    assert summary["query_source"].unique().tolist() == ["teacher"]
    assert summary["gallery_source"].unique().tolist() == ["v1"]
    assert summary["percentile"].tolist() == ["p50", "p95"] * 3


def test_timing_policy_rejects_non_one_thread_count() -> None:
    with pytest.raises(ValueError, match="one thread"):
        TimingPolicy(thread_count=2)


def test_benchmark_warms_first_100_then_times_every_sorted_id_once() -> None:
    rows = pd.DataFrame({"id": list(range(202, 99, -1))})
    encoded_ids: list[int] = []
    searched_ids: list[int] = []

    def encode(row: pd.Series) -> np.ndarray:
        encoded_ids.append(int(row["id"]))
        return np.array([1.0], dtype=np.float32)

    def search(query_id: int, feature: np.ndarray) -> pd.DataFrame:
        searched_ids.append(query_id)
        assert feature.tolist() == [1.0]
        return pd.DataFrame()

    clock_values: list[int] = []
    for query_id in range(100, 203):
        start = query_id * 1_000_000_000
        clock_values.extend([start, start + 100_000_000, start + 150_000_000])

    samples = benchmark_source_direction(
        rows,
        query_source="teacher",
        gallery_source="v1",
        encode=encode,
        search=search,
        clock_ns=FakeClock(clock_values),
    )

    assert encoded_ids[:100] == list(range(100, 200))
    assert searched_ids[:100] == list(range(100, 200))
    assert encoded_ids[100:] == list(range(100, 203))
    assert searched_ids[100:] == list(range(100, 203))
    assert samples["query_id"].tolist() == list(range(100, 203))
    assert samples["encoding_seconds"].tolist() == pytest.approx([0.1] * 103)
    assert samples["search_seconds"].tolist() == pytest.approx([0.05] * 103)
    assert samples["end_to_end_seconds"].tolist() == pytest.approx([0.15] * 103)


def test_benchmark_keeps_a_slow_sample() -> None:
    rows = pd.DataFrame({"id": [3, 1, 2]})
    clock = FakeClock(
        [
            0,
            100_000_000,
            200_000_000,
            1_000_000_000,
            1_100_000_000,
            11_000_000_000,
            12_000_000_000,
            12_100_000_000,
            12_200_000_000,
        ]
    )

    samples = benchmark_source_direction(
        rows,
        query_source="teacher",
        gallery_source="teacher",
        encode=lambda _row: np.array([1.0], dtype=np.float32),
        search=lambda _query_id, _feature: pd.DataFrame(),
        clock_ns=clock,
    )

    assert samples["query_id"].tolist() == [1, 2, 3]
    assert samples["end_to_end_seconds"].tolist() == pytest.approx([0.2, 10.0, 0.2])


def test_query_encoder_reads_the_row_path_then_runs_the_frozen_probe(tmp_path: Path) -> None:
    contract = PreprocessingContract(width=240, height=320)
    transformed = PreprocessedImage(
        pixels=np.ones((1, 1, 3), dtype=np.uint8),
        content_mask=np.ones((1, 1), dtype=bool),
        content_bounds=(0, 0, 1, 1),
    )
    loaded: list[tuple[Path, PreprocessingContract]] = []

    def load_image(path: str | Path, used_contract: PreprocessingContract) -> PreprocessedImage:
        loaded.append((Path(path), used_contract))
        return transformed

    def extract(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert pixels is transformed.pixels
        assert mask is transformed.content_mask
        return np.array([1.0], dtype=np.float32)

    encode = build_query_encoder(
        path_column="teacher_path",
        contract=contract,
        root=tmp_path,
        load_image=load_image,
        extract=extract,
    )

    result = encode(pd.Series({"teacher_path": "images/10.jpg"}))

    assert result.tolist() == [1.0]
    assert loaded == [(tmp_path / "images/10.jpg", contract)]


def test_protocol_a_search_adapter_runs_complete_top_20_batch_one_path() -> None:
    gallery_ids = np.arange(20, 41, dtype=np.int64)
    gallery_features = np.zeros((21, 2), dtype=np.float32)
    gallery_features[:, 0] = 1.0
    gallery_features[-1] = (0.0, 1.0)
    views = RetrievalViews(
        queries=pd.DataFrame({"id": [7]}),
        gallery=pd.DataFrame({"id": gallery_ids}),
    )
    index = FeatureIndex(
        source="teacher",
        contract=PreprocessingContract(width=240, height=320),
        ids=gallery_ids,
        features=gallery_features,
        transform_seconds=0.0,
        source_bytes=0,
    )
    search = build_protocol_a_search(views=views, gallery_index=index)

    result = search(7, np.array([1.0, 0.0], dtype=np.float32))

    assert result["query_id"].unique().tolist() == [7]
    assert result["candidate_id"].tolist() == list(range(20, 40))
    assert result["rank"].tolist() == list(range(1, 21))


def test_measure_index_build_reports_payload_index_and_spawn_peak_rss(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for product_id, colour in ((2, (255, 0, 0)), (1, (0, 0, 255))):
        Image.new("RGB", (10, 8), colour).save(image_dir / f"{product_id}.png")
    rows = pd.DataFrame(
        {
            "id": [2, 1],
            "partition": ["development", "development"],
            "path": ["images/2.png", "images/1.png"],
        }
    )

    cost = measure_index_build(
        rows,
        source="teacher",
        path_column="path",
        contract=PreprocessingContract(width=240, height=320),
        root=tmp_path,
    )

    assert cost.source == "teacher"
    assert cost.contract == PreprocessingContract(width=240, height=320)
    assert cost.rows == 2
    assert cost.dimension == 400
    assert cost.payload_bytes == 2 * 400 * np.dtype(np.float32).itemsize
    assert cost.index_bytes == cost.payload_bytes + 2 * np.dtype(np.int64).itemsize
    assert cost.build_seconds >= 0.0
    assert cost.peak_rss_bytes > cost.index_bytes


def test_measure_index_build_rejects_non_frozen_contract() -> None:
    with pytest.raises(ValueError, match="frozen 240x320"):
        measure_index_build(
            pd.DataFrame(),
            source="teacher",
            path_column="path",
            contract=PreprocessingContract(width=8, height=8),
        )


_DIRECTIONS = (
    ("teacher", "teacher"),
    ("v1", "v1"),
    ("teacher", "v1"),
    ("v1", "teacher"),
)
_METRICS = ("encoding", "search", "end_to_end")
_PERCENTILES = ("p50", "p95")
_FROZEN_CONTRACT = PreprocessingContract(width=240, height=320)
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _complete_timing_summary() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "query_source": query_source,
                "gallery_source": gallery_source,
                "metric": metric,
                "percentile": percentile,
                "value_seconds": (
                    1.25
                    if (query_source, gallery_source, metric, percentile)
                    == ("teacher", "teacher", "end_to_end", "p95")
                    else 0.25
                ),
                "timed_queries": 4,
            }
            for query_source, gallery_source in _DIRECTIONS
            for metric in _METRICS
            for percentile in _PERCENTILES
        ]
    )


def _index_costs() -> dict[str, IndexCost]:
    return {
        "teacher": IndexCost(
            source="teacher",
            contract=_FROZEN_CONTRACT,
            rows=10,
            dimension=400,
            payload_bytes=16_000,
            index_bytes=20_000,
            build_seconds=2.5,
            peak_rss_bytes=40_000,
        ),
        "v1": IndexCost(
            source="v1",
            contract=_FROZEN_CONTRACT,
            rows=10,
            dimension=400,
            payload_bytes=16_000,
            index_bytes=2**30 + 1,
            build_seconds=3.5,
            peak_rss_bytes=50_000,
        ),
    }


def _set_single_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in _THREAD_VARIABLES:
        monkeypatch.setenv(variable, "1")


def test_cost_record_is_json_safe_and_keeps_failed_thresholds_as_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_single_thread(monkeypatch)
    timing_summary = _complete_timing_summary()
    index_costs = _index_costs()

    record = build_cost_record(timing_summary, index_costs, policy=TimingPolicy())

    assert record["scope"] == "development"
    assert record["fold"] == 1
    assert record["parameters"] == 0
    assert record["checkpoint_bytes"] == 0
    assert record["warmup_queries"] == 100
    assert record["timed_queries"] == 4
    assert record["p95_end_to_end_under_one_second"] is False
    assert record["index_under_one_gibibyte"] is False
    assert set(record["per_source_index_cost"]) == {"teacher", "v1"}
    assert record["hardware"]["thread_environment"].keys() == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    json.dumps(record, allow_nan=False)


def test_cost_record_rejects_index_with_non_frozen_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_single_thread(monkeypatch)
    costs = _index_costs()
    costs["v1"] = replace(
        costs["v1"],
        contract=PreprocessingContract(width=8, height=8),
    )

    with pytest.raises(ValueError, match="frozen 240x320"):
        build_cost_record(_complete_timing_summary(), costs, policy=TimingPolicy())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_cost_record_rejects_every_non_finite_timing_value(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    _set_single_thread(monkeypatch)
    summary = _complete_timing_summary()
    summary.loc[0, "value_seconds"] = bad_value

    with pytest.raises(ValueError, match="finite"):
        build_cost_record(summary, _index_costs(), policy=TimingPolicy())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_cost_record_rejects_non_finite_index_build_time(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    _set_single_thread(monkeypatch)
    costs = _index_costs()
    costs["teacher"] = replace(costs["teacher"], build_seconds=bad_value)

    with pytest.raises(ValueError, match="finite"):
        build_cost_record(_complete_timing_summary(), costs, policy=TimingPolicy())


def test_cost_record_rejects_missing_timing_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_single_thread(monkeypatch)
    summary = _complete_timing_summary().iloc[1:].reset_index(drop=True)

    with pytest.raises(ValueError, match="complete"):
        build_cost_record(summary, _index_costs(), policy=TimingPolicy())


def test_cost_record_rejects_duplicate_timing_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_single_thread(monkeypatch)
    summary = _complete_timing_summary()
    summary = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="exactly one"):
        build_cost_record(summary, _index_costs(), policy=TimingPolicy())


def test_cost_record_rejects_inconsistent_timed_query_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_single_thread(monkeypatch)
    summary = _complete_timing_summary()
    summary.loc[0, "timed_queries"] = 3

    with pytest.raises(ValueError, match="timed-query count"):
        build_cost_record(summary, _index_costs(), policy=TimingPolicy())


def test_cost_record_rejects_thread_environment_not_fixed_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_single_thread(monkeypatch)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")

    with pytest.raises(ValueError, match="thread environment"):
        build_cost_record(
            _complete_timing_summary(),
            _index_costs(),
            policy=TimingPolicy(),
        )
