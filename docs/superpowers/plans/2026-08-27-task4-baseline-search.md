# Task 4 Baseline Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate Task 4 Python code and deliver a frozen, measured HSV-and-edge retrieval baseline with auditable quality, cost, timing, failure, and notebook evidence.

**Architecture:** Real Task 4 code lives in `fashion.task4`; `fashion.retrieval` remains a thin compatibility layer. The baseline reuses the frozen preprocessing, probe, ranking, and protocol modules, while focused baseline-analysis and benchmark modules produce tracked evidence consumed by the narrative notebook.

**Tech Stack:** Python 3.12, NumPy, pandas, Pillow, Matplotlib, pytest, Ruff, nbformat

**Spec:** `docs/superpowers/specs/2026-08-27-task4-baseline-search-design.md`

## Global Constraints

- Use `data/processed/splits.csv` as the only split.
- Keep holdout, quarantine, and official teacher-test pixels and labels sealed.
- Use fixed validation fold `1`; do not add five-fold baseline selection.
- Use the frozen `240×320` RGB/LANCZOS letterbox contract from ADR 0021.
- Keep `spatial-hsv-edge-4x4-v2` unchanged unless a test proves a correctness bug.
- Evaluate teacher→teacher, V1→V1, teacher→V1, and V1→teacher.
- Keep Protocol A query-mean linear nDCG@10 primary and Protocol B supporting.
- Do not tune descriptor bins, grid, weights, colour space, or distance.
- Do not create a `results/runs.csv` row because the baseline is untrained.
- Real code belongs in `src/fashion/task4/`; compatibility wrappers contain no logic.
- Use `./.venv/bin/python`.
- Do not open the holdout.
- Commit steps below are optional checkpoints and may run only when the user explicitly authorizes commits.

## File Structure

### Move real Task 4 code

- `src/fashion/retrieval/external.py` → `src/fashion/task4/external.py`
- `src/fashion/retrieval/preprocessing.py` → `src/fashion/task4/preprocessing.py`
- `src/fashion/retrieval/cache.py` → `src/fashion/task4/cache.py`
- `src/fashion/retrieval/protocol.py` → `src/fashion/task4/protocol.py`
- `src/fashion/retrieval/probe.py` → `src/fashion/task4/probe.py`
- `src/fashion/retrieval/preprocessing_experiment.py` → `src/fashion/task4/preprocessing_experiment.py`
- `scripts/run_task4_preprocessing.py` → `scripts/task4/run_preprocessing.py`
- Task-4-only tests → `tests/task4/`

### Create

- `src/fashion/task4/__init__.py` — public Task 4 API
- `src/fashion/task4/baseline.py` — four-direction quality run, random floor, headline checks
- `src/fashion/task4/analysis.py` — support counts, failure slices, canvas stress, example IDs
- `src/fashion/task4/benchmark.py` — batch-one latency, index cost, and peak-memory policy
- `src/fashion/task4/baseline_evidence.py` — deterministic evidence tables and example figure
- `scripts/task4/run_baseline.py` — thin Milestone 4 runner
- `tests/task4/test_import_compatibility.py`
- `tests/task4/test_baseline.py`
- `tests/task4/test_analysis.py`
- `tests/task4/test_benchmark.py`
- `tests/task4/test_baseline_evidence.py`
- `docs/decisions/0022-task4-baseline-search.md`

### Keep as compatibility modules

- `src/fashion/retrieval/__init__.py`
- `src/fashion/retrieval/external.py`
- `src/fashion/retrieval/preprocessing.py`
- `src/fashion/retrieval/cache.py`
- `src/fashion/retrieval/protocol.py`
- `src/fashion/retrieval/probe.py`
- `src/fashion/retrieval/preprocessing_experiment.py`

---

### Task 1: Move Task 4 code and preserve old imports

**Files:**
- Create: `src/fashion/task4/__init__.py`
- Move: the six real modules listed under “Move real Task 4 code”
- Rewrite: all seven `src/fashion/retrieval/*.py` files as compatibility exports
- Move: `scripts/run_task4_preprocessing.py` to `scripts/task4/run_preprocessing.py`
- Move: five Task-4-only test files to `tests/task4/`
- Create: `tests/task4/test_import_compatibility.py`
- Modify: imports in moved modules, moved tests, and both Task 4 notebooks

**Interfaces:**
- Produces: canonical imports such as `fashion.task4.protocol.build_development_views`
- Preserves: old imports such as `fashion.retrieval.protocol.build_development_views`
- Constraint: old and new imports resolve to the same function or class objects

- [ ] **Step 1: Write the failing compatibility test**

Create `tests/task4/test_import_compatibility.py`:

```python
from fashion.retrieval import preprocessing as old_preprocessing
from fashion.retrieval import probe as old_probe
from fashion.retrieval import protocol as old_protocol
from fashion.task4 import preprocessing, probe, protocol


def test_old_retrieval_imports_reexport_task4_objects() -> None:
    assert old_preprocessing.PreprocessingContract is preprocessing.PreprocessingContract
    assert old_probe.extract_spatial_probe is probe.extract_spatial_probe
    assert old_protocol.build_development_views is protocol.build_development_views
```

- [ ] **Step 2: Run the test and confirm the new package is missing**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_import_compatibility.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'fashion.task4'`.

- [ ] **Step 3: Move the real modules, script, and Task-4-only tests**

Use these destination names:

```text
tests/task4/test_external.py
tests/task4/test_preprocessing.py
tests/task4/test_protocol.py
tests/task4/test_probe.py
tests/task4/test_preprocessing_experiment.py
scripts/task4/run_preprocessing.py
```

Change all internal imports in real modules to `fashion.task4.*`. Build
`fashion.task4.__init__` from the current retrieval public API, using relative
imports so it never passes through compatibility wrappers.

- [ ] **Step 4: Replace old modules with thin exports**

Each old module contains only a docstring and explicit re-exports. For example:

```python
"""Compatibility import for Task 4 retrieval protocol helpers."""

from fashion.task4.protocol import *  # noqa: F403
from fashion.task4.protocol import __all__
```

Add an explicit `__all__` tuple to every real Task 4 module before using this
pattern. `src/fashion/retrieval/__init__.py` re-exports only from
`fashion.task4`, never from its sibling wrappers.

- [ ] **Step 5: Update active imports and the preprocessing runner test**

Replace active `fashion.retrieval` imports with `fashion.task4` in:

```text
scripts/task4/run_preprocessing.py
tests/task4/
notebooks/task-4/01_v1_eda.ipynb
notebooks/task-4/05_task4_visual_search.ipynb
```

Update the help-entrypoint test to execute:

```python
subprocess.run(
    [sys.executable, "scripts/task4/run_preprocessing.py", "--help"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
```

- [ ] **Step 6: Run focused tests and lint**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_import_compatibility.py tests/task4/test_external.py tests/task4/test_preprocessing.py tests/task4/test_protocol.py tests/task4/test_probe.py tests/task4/test_preprocessing_experiment.py -q
./.venv/bin/python -m ruff check src/fashion/task4 src/fashion/retrieval scripts/task4 tests/task4
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 7: Optional commit checkpoint**

Only if commits were explicitly authorized:

```bash
git add src/fashion/task4 src/fashion/retrieval scripts/task4 tests/task4 notebooks/task-4
git commit -m "refactor(task4): isolate retrieval implementation"
```

---

### Task 2: Freeze baseline quality and deterministic sanity-floor evaluation

**Files:**
- Create: `src/fashion/task4/baseline.py`
- Create: `tests/task4/test_baseline.py`
- Modify: `src/fashion/task4/__init__.py`

**Interfaces:**
- Consumes: `FeatureIndex`, `PairEvaluation`, `build_development_views`, `evaluate_source_pair`
- Produces:
  - `Direction = tuple[SourceName, SourceName]`
  - `BaselineEvaluation`
  - `build_random_primary_rankings(views, *, seed=2753, max_k=20)`
  - `build_query_metrics(pair_evaluations)`
  - `build_baseline_summary(pair_evaluations, random_rankings, primary_views)`
  - `build_headline_summary(...)`
  - `evaluate_baseline(splits, indexes, *, fold=1)`
  - `verify_preprocessing_reproduction(summary, comparison, *, atol=1e-5)`

- [ ] **Step 1: Write failing tests for the random floor and four directions**

Use small synthetic `RetrievalViews` and unit-normal feature indexes:

```python
def test_random_floor_is_seeded_and_independent_of_gallery_row_order() -> None:
    first = build_random_primary_rankings(views, seed=2753, max_k=2)
    reversed_views = RetrievalViews(
        queries=views.queries,
        gallery=views.gallery.iloc[::-1].reset_index(drop=True),
    )
    second = build_random_primary_rankings(reversed_views, seed=2753, max_k=2)
    pd.testing.assert_frame_equal(first, second)


def test_baseline_runs_all_four_source_directions() -> None:
    result = evaluate_baseline(splits, {"teacher": teacher_index, "v1": v1_index})
    assert set(result.pair_evaluations) == {
        ("teacher", "teacher"),
        ("v1", "v1"),
        ("teacher", "v1"),
        ("v1", "teacher"),
    }
```

- [ ] **Step 2: Run the tests and confirm the API is absent**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_baseline.py -q
```

Expected: import fails because `fashion.task4.baseline` does not exist.

- [ ] **Step 3: Implement the baseline result type and random floor**

Use these public shapes:

```python
Direction = tuple[SourceName, SourceName]


@dataclass(frozen=True)
class BaselineEvaluation:
    summary: pd.DataFrame
    query_metrics: pd.DataFrame
    pair_evaluations: dict[Direction, PairEvaluation]
    random_rankings: pd.DataFrame


def build_random_primary_rankings(
    views: RetrievalViews,
    *,
    seed: int = 2753,
    max_k: int = 20,
) -> pd.DataFrame:
    gallery_ids = (
        pd.to_numeric(views.gallery["id"], errors="raise")
        .astype(np.int64)
        .sort_values()
        .to_numpy()
    )
    ordered_gallery = np.random.default_rng(seed).permutation(gallery_ids)
    selected = ordered_gallery[:max_k]
    records = [
        {
            "query_id": int(query_id),
            "candidate_id": int(candidate_id),
            "distance": float(position),
        }
        for query_id in sorted(pd.to_numeric(views.queries["id"], errors="raise"))
        for position, candidate_id in enumerate(selected, start=1)
    ]
    return prepare_rankings(
        pd.DataFrame.from_records(records),
        views,
        protocol="primary",
        max_k=max_k,
    )
```

Sort numeric gallery IDs before calling `np.random.default_rng(seed).permutation`.
Use the same shuffled order for every query, assign increasing synthetic
distances, and pass the records through `prepare_rankings`. Implement this for
Protocol A only.

- [ ] **Step 4: Implement four-direction evaluation and query-table labelling**

```python
def evaluate_baseline(
    splits: pd.DataFrame,
    indexes: Mapping[SourceName, FeatureIndex],
    *,
    fold: int = 1,
    k_values: tuple[int, ...] = (5, 10, 20),
    family_k: int = 10,
) -> BaselineEvaluation:
    primary, family = build_development_views(splits, validation_fold=fold)
    pairs = {
        direction: evaluate_source_pair(
            indexes[direction[0]],
            indexes[direction[1]],
            primary_views=primary,
            family_views=family,
            fold=fold,
            k_values=k_values,
            family_k=family_k,
        )
        for direction in source_directions()
    }
    random_rankings = build_random_primary_rankings(
        primary,
        seed=2753,
        max_k=max(k_values),
    )
    return BaselineEvaluation(
        summary=build_baseline_summary(pairs, random_rankings, primary),
        query_metrics=build_query_metrics(pairs),
        pair_evaluations=pairs,
        random_rankings=random_rankings,
    )
```

Requirements:

- require exactly teacher and V1 indexes;
- require both indexes to use `PreprocessingContract(width=240, height=320)`;
- call `evaluate_source_pair` once per `source_directions()` entry;
- prefix summary and per-query rows with method, fold, size, query source,
  gallery source, and protocol;
- append the random Protocol A summary with method `random-seed-2753`;
- add headline rows for same-source mean, cross-source mean, random-floor pass,
  and cross-source ratio pass;
- record failed hypotheses as `passed=False`; do not abort a correct weak run.

- [ ] **Step 5: Test headline calculations and reproduction failure**

```python
def test_headline_uses_equal_source_weight_and_records_claim_failures() -> None:
    summary = build_headline_summary(
        teacher_ndcg=0.50,
        v1_ndcg=0.40,
        teacher_to_v1_ndcg=0.30,
        v1_to_teacher_ndcg=0.30,
        random_ndcg=0.10,
    )
    assert summary["same_source_mean"] == pytest.approx(0.45)
    assert summary["cross_source_mean"] == pytest.approx(0.30)
    assert summary["beats_random"] is True
    assert summary["cross_source_within_95_percent"] is False


def test_reproduction_check_rejects_changed_selected_size_score() -> None:
    with pytest.raises(ValueError, match="preprocessing probe"):
        verify_preprocessing_reproduction(summary, comparison, atol=1e-5)
```

The verification function compares all four baseline Protocol A nDCG@10 values
with fold-1 `240x320` rows in `preprocessing_comparison.csv`. The fresh
one-thread quality run uses `chunk_size=256` and the named `1e-5` absolute
tolerance because forcing one BLAS thread moved a few near-tie rankings by at
most `9.709e-6`; default-thread V1→teacher exactly matched stored evidence.

- [ ] **Step 6: Run focused tests and lint**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_baseline.py tests/task4/test_protocol.py tests/task4/test_probe.py -q
./.venv/bin/python -m ruff check src/fashion/task4/baseline.py tests/task4/test_baseline.py
```

Expected: all tests pass.

- [ ] **Step 7: Optional commit checkpoint**

Only if commits were explicitly authorized:

```bash
git add src/fashion/task4/baseline.py src/fashion/task4/__init__.py tests/task4/test_baseline.py
git commit -m "feat(task4): freeze baseline quality evaluation"
```

---

### Task 3: Add failure slices, canvas stress, and deterministic examples

**Files:**
- Create: `src/fashion/task4/analysis.py`
- Create: `tests/task4/test_analysis.py`
- Modify: `src/fashion/task4/preprocessing_experiment.py`
- Modify: `tests/task4/test_preprocessing_experiment.py`
- Modify: `src/fashion/task4/__init__.py`

**Interfaces:**
- Consumes: `BaselineEvaluation`, `RetrievalViews`, `FeatureIndex`, `PairEvaluation`
- Produces:
  - `build_query_support(primary_views, family_views)`
  - `mark_failure_slices(support)`
  - `summarize_failure_slices(query_metrics, membership)`
  - `extract_canvas_feature_index`
  - `evaluate_canvas_stress(clean, canvas_indexes, gallery_index, primary_views, *, fold=1)`
  - `select_example_ids(query_metrics, membership, canvas_per_query)`

- [ ] **Step 1: Write boundary tests for every declared slice**

Create synthetic query/gallery frames where support counts hit each boundary:

```python
def test_failure_slice_boundaries_are_frozen() -> None:
    marked = mark_failure_slices(
        pd.DataFrame(
            {
                "query_id": [1, 2, 3, 4],
                "mode": ["L", "RGB", "RGB", "RGB"],
                "aspect_ratio": [0.75, 0.75, 1.0, 0.75],
                "primary_positive_count": [9, 10, 10, 10],
                "primary_strict_count": [9, 10, 0, 10],
                "family_positive_count": [4, 5, 0, 1],
            }
        )
    )
    assert marked.loc[0, "grayscale"]
    assert marked.loc[0, "rare_article_type"]
    assert not marked.loc[1, "rare_article_type"]
    assert marked.loc[2, "rare_type_colour"]
    assert marked.loc[2, "unusual_geometry"]
    assert marked.loc[2, "family_unavailable"]
    assert marked.loc[3, "weak_family"]
```

- [ ] **Step 2: Run the test and confirm the module is absent**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_analysis.py -q
```

Expected: import fails because `fashion.task4.analysis` does not exist.

- [ ] **Step 3: Implement support counts and slice flags**

```python
def build_query_support(
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
) -> pd.DataFrame:
    queries = primary_views.queries.sort_values("id").reset_index(drop=True)
    type_counts = primary_views.gallery.groupby("articleType", dropna=False).size()
    strict_counts = primary_views.gallery.groupby(
        ["articleType", "baseColour"],
        dropna=False,
    ).size()
    strict_keys = pd.Series(
        list(queries[["articleType", "baseColour"]].itertuples(index=False, name=None))
    )
    family_queries = family_views.queries
    family_size = family_queries.groupby("product_family_group")["id"].transform("size")
    duplicate_size = family_queries.groupby(
        ["product_family_group", "duplicate_group"],
        dropna=False,
    )["id"].transform("size")
    sha_size = family_queries.groupby(
        ["product_family_group", "sha256"],
        dropna=False,
    )["id"].transform("size")
    intersection_size = family_queries.groupby(
        ["product_family_group", "duplicate_group", "sha256"],
        dropna=False,
    )["id"].transform("size")
    family_counts = pd.Series(
        (family_size - duplicate_size - sha_size + intersection_size).to_numpy(),
        index=family_queries["id"].astype(int),
    )
    return pd.DataFrame(
        {
            "query_id": queries["id"].astype(int),
            "mode": queries["mode"],
            "aspect_ratio": queries["aspect_ratio"].astype(float),
            "primary_positive_count": queries["articleType"]
            .map(type_counts)
            .fillna(0)
            .astype(int),
            "primary_strict_count": strict_keys.map(strict_counts).fillna(0).astype(int),
            "family_positive_count": queries["id"].astype(int).map(family_counts).astype(int),
        }
    )


def mark_failure_slices(support: pd.DataFrame) -> pd.DataFrame:
    marked = support.copy()
    marked["grayscale"] = marked["mode"].eq("L")
    marked["rare_article_type"] = marked["primary_positive_count"].between(1, 9)
    marked["rare_type_colour"] = marked["primary_strict_count"].lt(10)
    marked["unusual_geometry"] = marked["aspect_ratio"].ne(0.75)
    marked["family_unavailable"] = marked["family_positive_count"].eq(0)
    marked["weak_family"] = marked["family_positive_count"].between(1, 4)
    return marked
```

Calculate support with the exact same groupings and inclusion-exclusion used by
`compute_relevance_coverage`. Return one row per query with:

```text
query_id, mode, aspect_ratio,
primary_positive_count, primary_strict_count, family_positive_count,
grayscale, rare_article_type, rare_type_colour, unusual_geometry,
family_unavailable, weak_family
```

Use `1 <= primary_positive_count < 10`, `primary_strict_count < 10`,
`aspect_ratio != 0.75`, `family_positive_count == 0`, and
`1 <= family_positive_count < 5`.

- [ ] **Step 4: Implement long-format slice summaries**

```python
def summarize_failure_slices(
    query_metrics: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    joined = query_metrics.merge(membership, on="query_id", validate="many_to_one")
    slice_metrics = (
        ("grayscale", "primary", "ndcg_at_10", "ndcg"),
        ("rare_article_type", "primary", "ndcg_at_10", "ndcg"),
        ("rare_type_colour", "primary", "ndcg_at_10", "ndcg"),
        ("unusual_geometry", "primary", "ndcg_at_10", "ndcg"),
        ("family_unavailable", "family", "recall_at_10", "recall"),
        ("weak_family", "family", "recall_at_10", "recall"),
    )
    rows: list[dict[str, object]] = []
    context = ["scope", "fold", "size", "query_source", "gallery_source"]
    for context_values, direction_rows in joined.groupby(context, sort=True):
        common = dict(zip(context, context_values, strict=True))
        for slice_name, protocol, value_column, metric in slice_metrics:
            selected = direction_rows.loc[
                direction_rows["protocol"].eq(protocol) & direction_rows[slice_name]
            ]
            values = selected[value_column]
            rows.append(
                {
                    **common,
                    "protocol": protocol,
                    "slice": slice_name,
                    "metric": metric,
                    "k": 10,
                    "aggregation": "query_mean",
                    "value": float(values.mean()) if values.notna().any() else np.nan,
                    "total_queries": len(selected),
                    "scored_queries": int(values.notna().sum()),
                    "excluded_queries": int(values.isna().sum()),
                    "coverage": float(values.notna().mean()) if len(selected) else 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)
```

Emit deterministic rows with:

```text
scope, fold, size, query_source, gallery_source, protocol,
slice, metric, k, aggregation, value,
total_queries, scored_queries, excluded_queries, coverage
```

Use Protocol A nDCG@10 for grayscale, rare type, rare pair, and observed
geometry. Use Protocol B Recall@10 for unavailable and weak-family rows.
Unavailable-family score remains null while its count and zero coverage remain
visible.

- [ ] **Step 5: Move reusable canvas extraction out of the old script**

Add to `preprocessing_experiment.py`:

```python
def extract_canvas_feature_index(
    query_rows: pd.DataFrame,
    *,
    source: SourceName,
    path_column: str,
    orientation: Literal["wide", "tall"],
    contract: PreprocessingContract,
    root: str | Path = ROOT,
    workers: int = 1,
) -> FeatureIndex:
    """Extract sorted probe features after deterministic wide/tall canvassing."""
```

Move the complete current `_odd_query_index` body from
`scripts/task4/run_preprocessing.py` into this function, replace its hard-coded
`ROOT` with `Path(root)`, sort `query_rows` by numeric ID before creating
records, and reject rows whose partition is not `development`. Update the
preprocessing runner to call the public function.

- [ ] **Step 6: Implement per-query canvas stress and example selection**

```python
@dataclass(frozen=True)
class CanvasStressEvaluation:
    summary: pd.DataFrame
    per_query: pd.DataFrame
    rankings: dict[str, pd.DataFrame]
```

For each query and canvas orientation, save clean nDCG@10, canvas nDCG@10,
change, and Top-10 overlap. Choose the normal V1→V1 success outside all slices,
lowest applicable failures, and largest canvas drop. Break every tie by
ascending numeric ID and allow repeated IDs.

- [ ] **Step 7: Run focused tests and lint**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_analysis.py tests/task4/test_preprocessing_experiment.py -q
./.venv/bin/python -m ruff check src/fashion/task4/analysis.py src/fashion/task4/preprocessing_experiment.py tests/task4/test_analysis.py
```

Expected: all tests pass.

- [ ] **Step 8: Optional commit checkpoint**

Only if commits were explicitly authorized:

```bash
git add src/fashion/task4/analysis.py src/fashion/task4/preprocessing_experiment.py src/fashion/task4/__init__.py tests/task4/test_analysis.py tests/task4/test_preprocessing_experiment.py scripts/task4/run_preprocessing.py
git commit -m "feat(task4): add baseline failure analysis"
```

---

### Task 4: Implement batch-one timing and memory evidence

**Files:**
- Create: `src/fashion/task4/benchmark.py`
- Create: `tests/task4/test_benchmark.py`
- Modify: `src/fashion/task4/__init__.py`

**Interfaces:**
- Produces:
  - `TimingPolicy`
  - `benchmark_source_direction(query_rows, *, query_source, gallery_source, encode, search, policy, clock_ns)`
  - `summarize_timings(samples)`
  - `measure_index_build(gallery_rows, *, source, path_column, contract, root)`
  - `build_cost_record(timing_summary, index_costs, *, policy)`
- Requires injectable clock and encode/search callables for deterministic tests

- [ ] **Step 1: Write failing timing-policy tests with a fake clock**

```python
def test_timing_keeps_every_query_and_reports_linear_percentiles() -> None:
    samples = pd.DataFrame(
        {
            "encoding_seconds": [0.1, 0.2, 0.3, 10.0],
            "search_seconds": [0.01, 0.02, 0.03, 1.0],
            "end_to_end_seconds": [0.11, 0.22, 0.33, 11.0],
        }
    )
    summary = summarize_timings(samples)
    assert summary.query("metric == 'end_to_end' and percentile == 'p95'")[
        "value_seconds"
    ].item() == pytest.approx(np.quantile(samples["end_to_end_seconds"], 0.95))
    assert summary["timed_queries"].unique().tolist() == [4]
```

Add tests proving the policy rejects non-one thread counts, does exactly 100
warm-ups when enough rows exist, sorts timed IDs, and retains the slow sample.

- [ ] **Step 2: Run the tests and confirm the module is absent**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_benchmark.py -q
```

Expected: import fails because `fashion.task4.benchmark` does not exist.

- [ ] **Step 3: Implement the timing policy and injected benchmark loop**

```python
@dataclass(frozen=True)
class TimingPolicy:
    warmup_queries: int = 100
    thread_count: int = 1


def benchmark_source_direction(
    query_rows: pd.DataFrame,
    *,
    query_source: SourceName,
    gallery_source: SourceName,
    encode: Callable[[pd.Series], np.ndarray],
    search: Callable[[int, np.ndarray], pd.DataFrame],
    policy: TimingPolicy = TimingPolicy(),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> pd.DataFrame:
    ordered = query_rows.assign(
        _numeric_id=pd.to_numeric(query_rows["id"], errors="raise").astype(np.int64)
    ).sort_values("_numeric_id")
    for _, row in ordered.iloc[: policy.warmup_queries].iterrows():
        search(int(row["_numeric_id"]), encode(row))
    records: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        started = clock_ns()
        feature = encode(row)
        encoded = clock_ns()
        search(int(row["_numeric_id"]), feature)
        searched = clock_ns()
        records.append(
            {
                "query_id": int(row["_numeric_id"]),
                "query_source": query_source,
                "gallery_source": gallery_source,
                "encoding_seconds": (encoded - started) / 1e9,
                "search_seconds": (searched - encoded) / 1e9,
                "end_to_end_seconds": (searched - started) / 1e9,
            }
        )
    return pd.DataFrame.from_records(records)
```

Sort by numeric query ID. Run the first 100 IDs without saving measurements,
then time all IDs once. Measure encoding and search as adjacent intervals;
end-to-end is the outer interval. Record one row per direction/query and do not
trim values.

- [ ] **Step 4: Implement the real search adapter**

The real adapter:

- starts encoding from the local image path;
- calls `load_preprocessed_image` and `extract_spatial_probe`;
- builds a one-query `RetrievalViews` object against the complete Protocol A
  gallery;
- calls `rank_probe_embeddings` with `max_k=20` and `chunk_size=1`;
- includes exclusions, fusion, sorting, and tie handling inside search time.

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to `1` at the top of the fresh runner process before
NumPy imports.

- [ ] **Step 5: Implement index build and child-process peak memory**

```python
@dataclass(frozen=True)
class IndexCost:
    source: SourceName
    rows: int
    dimension: int
    payload_bytes: int
    index_bytes: int
    build_seconds: float
    peak_rss_bytes: int


def measure_index_build(
    gallery_rows: pd.DataFrame,
    *,
    source: SourceName,
    path_column: str,
    contract: PreprocessingContract,
    root: str | Path = ROOT,
) -> IndexCost:
    """Build one gallery index in a spawned child and return its measured cost."""
```

Run gallery reading, preprocessing, descriptor extraction, and matrix stacking
with one worker in a clean `multiprocessing.get_context("spawn")` child.
Return Linux `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024`.
Calculate payload bytes from `features.nbytes`; calculate searchable index
bytes from `features.nbytes + ids.nbytes`.

- [ ] **Step 6: Build cost and hardware records**

`build_cost_record` returns JSON-safe values with:

```text
schema_version, scope, fold, contract, probe_version,
parameters=0, checkpoint_bytes=0,
hardware, warmup_queries, timed_queries,
timing_summary,
per_source_index_cost,
p95_end_to_end_under_one_second,
index_under_one_gibibyte
```

Record CPU text from `/proc/cpuinfo`, logical cores, OS, Python, NumPy, and
thread environment. A failed practical threshold is evidence, not an exception.

- [ ] **Step 7: Run focused tests and lint**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_benchmark.py -q
./.venv/bin/python -m ruff check src/fashion/task4/benchmark.py tests/task4/test_benchmark.py
```

Expected: all tests pass.

- [ ] **Step 8: Optional commit checkpoint**

Only if commits were explicitly authorized:

```bash
git add src/fashion/task4/benchmark.py src/fashion/task4/__init__.py tests/task4/test_benchmark.py
git commit -m "feat(task4): measure baseline retrieval cost"
```

---

### Task 5: Write evidence and add the thin baseline runner

**Files:**
- Create: `src/fashion/task4/baseline_evidence.py`
- Create: `scripts/task4/run_baseline.py`
- Create: `tests/task4/test_baseline_evidence.py`
- Modify: `src/fashion/task4/__init__.py`
- Modify: `.gitignore` only if local benchmark scratch paths are not already ignored

**Interfaces:**
- Consumes: baseline, slice, canvas, timing, and cost result objects
- Produces: the six tracked evidence files and `baseline_examples.png`
- Produces: `render_baseline_examples(...)`
- Produces: `run_baseline_evidence(*, workers: int) -> None`
- Runner returns exit code `0` only after reproduction and artifact validation

- [ ] **Step 1: Write failing artifact-schema tests**

```python
def test_writer_creates_complete_deterministic_artifact_set(tmp_path: Path) -> None:
    write_baseline_artifacts(
        baseline=baseline_result,
        slice_summary=slice_summary,
        timings=timings,
        cost=cost,
        examples=examples,
        evidence_dir=tmp_path / "evidence",
        figure_dir=tmp_path / "figures",
        image_rows=image_rows,
        example_rankings=example_rankings,
    )
    assert {path.name for path in (tmp_path / "evidence").iterdir()} == {
        "baseline_summary.csv",
        "baseline_query_metrics.csv",
        "baseline_failure_slices.csv",
        "baseline_timing.csv",
        "baseline_cost.json",
        "baseline_examples.csv",
    }
    assert (tmp_path / "figures/baseline_examples.png").is_file()
```

Add assertions for sorted CSV rows, newline termination, JSON
`schema_version`, development scope, no protected partitions, zero parameter
and checkpoint bytes, and Top-5 example ranks.

- [ ] **Step 2: Run the tests and confirm the writer is absent**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_baseline_evidence.py -q
```

Expected: import fails because `fashion.task4.baseline_evidence` does not exist.

- [ ] **Step 3: Implement deterministic evidence writers**

```python
def write_baseline_artifacts(
    *,
    baseline: BaselineEvaluation,
    slice_summary: pd.DataFrame,
    timings: pd.DataFrame,
    cost: Mapping[str, object],
    examples: pd.DataFrame,
    evidence_dir: str | Path,
    figure_dir: str | Path,
    image_rows: pd.DataFrame,
    example_rankings: Mapping[str, pd.DataFrame],
) -> None:
    evidence_path = Path(evidence_dir)
    figure_path = Path(figure_dir)
    evidence_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(
        baseline.summary,
        evidence_path / "baseline_summary.csv",
        index=False,
        float_format="%.8f",
    )
    write_deterministic_csv(
        baseline.query_metrics,
        evidence_path / "baseline_query_metrics.csv",
        index=False,
        float_format="%.8f",
    )
    write_deterministic_csv(
        slice_summary,
        evidence_path / "baseline_failure_slices.csv",
        index=False,
        float_format="%.8f",
    )
    write_deterministic_csv(
        timings,
        evidence_path / "baseline_timing.csv",
        index=False,
        float_format="%.8f",
    )
    write_deterministic_csv(
        examples,
        evidence_path / "baseline_examples.csv",
        index=False,
        float_format="%.8f",
    )
    (evidence_path / "baseline_cost.json").write_text(
        json.dumps(cost, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_baseline_examples(
        examples,
        image_rows,
        example_rankings,
        figure_path / "baseline_examples.png",
    )
```

Use `write_deterministic_csv` with explicit sort columns and `%.8f` for quality
values. Write JSON with `indent=2`, `sort_keys=True`, and a final newline.
Validate exact output columns before writing. Never persist full Top-20
rankings; persist per-query metrics and only the selected Top-5 example rows.

- [ ] **Step 4: Implement the V1→V1 example figure**

Create one labelled row per selected example:

- query image;
- Top-5 V1 gallery images;
- query ID, slice, score, and result IDs;
- synthetic wide/tall query image for the canvas row.

Use fixed figure dimensions, fonts, row order, and `dpi=180`. Do not select IDs
inside the renderer; it only renders `baseline_examples.csv`.

- [ ] **Step 5: Implement the runner**

At process start, set single-thread environment variables. Then:

1. load safe splits with `load_splits`;
2. load and validate the development-only external variant index;
3. open or build selected-size feature caches;
4. run four-direction quality and random-floor evaluation;
5. verify the four selected-size probe values against preprocessing evidence;
6. run V1 canvas stress and failure-slice aggregation;
7. benchmark all four Protocol A source directions;
8. measure teacher and V1 gallery index cost in child processes;
9. select examples and write artifacts;
10. reopen every artifact and validate scope, columns, counts, and finite values.

The public CLI is:

```python
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild development-only Task 4 baseline evidence."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="parallel cache workers; timed work remains single-threaded",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    run_baseline_evidence(workers=args.workers)
    return 0
```

It supports `--workers` for non-timed cache construction only. Timing and index
cost remain fixed at one thread.

- [ ] **Step 6: Add a help-entrypoint test and run the full Task 4 unit set**

Run:

```bash
./.venv/bin/python scripts/task4/run_baseline.py --help
./.venv/bin/python -m pytest tests/task4 -q
./.venv/bin/python -m ruff check src/fashion/task4 src/fashion/retrieval scripts/task4 tests/task4
```

Expected: help exits zero, all Task 4 tests pass, and Ruff reports no errors.

- [ ] **Step 7: Optional commit checkpoint**

Only if commits were explicitly authorized:

```bash
git add src/fashion/task4/baseline_evidence.py src/fashion/task4/__init__.py scripts/task4/run_baseline.py tests/task4/test_baseline_evidence.py .gitignore
git commit -m "feat(task4): generate baseline evidence"
```

---

### Task 6: Run the real development-only baseline

**Files:**
- Create:
  - `results/evidence/task4/baseline_summary.csv`
  - `results/evidence/task4/baseline_query_metrics.csv`
  - `results/evidence/task4/baseline_failure_slices.csv`
  - `results/evidence/task4/baseline_timing.csv`
  - `results/evidence/task4/baseline_cost.json`
  - `results/evidence/task4/baseline_examples.csv`
  - `results/figures/task4/baseline_examples.png`

**Interfaces:**
- Consumes: only development rows and selected-size local caches
- Produces: tracked Milestone 4 evidence; no model checkpoint and no run row

- [ ] **Step 1: Confirm the holdout guard before the expensive run**

Run:

```bash
./.venv/bin/python -m pytest tests/task4/test_baseline.py tests/task4/test_analysis.py tests/task4/test_benchmark.py tests/task4/test_baseline_evidence.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the baseline evidence command**

Run:

```bash
./.venv/bin/python scripts/task4/run_baseline.py
```

Expected: exit code `0`; it reports four source directions, 6,556 fold-1
queries, selected size `240x320`, probe version
`spatial-hsv-edge-4x4-v2`, and the six evidence paths.

- [ ] **Step 3: Check required evidence facts**

Use a short Python validation command to assert:

```python
assert set(summary["query_source"]) >= {"teacher", "v1"}
assert set(summary["gallery_source"]) >= {"teacher", "v1"}
assert cost["parameters"] == 0
assert cost["checkpoint_bytes"] == 0
assert cost["scope"] == "development"
assert slices.query("slice == 'grayscale'")["total_queries"].max() == 44
assert slices.query("slice == 'rare_article_type'")["total_queries"].max() == 40
assert slices.query("slice == 'rare_type_colour'")["total_queries"].max() == 833
assert slices.query("slice == 'family_unavailable'")["total_queries"].max() == 3937
assert slices.query("slice == 'weak_family'")["total_queries"].max() == 1659
```

Also verify that the one-thread, `chunk_size=256` baseline values reproduce the
selected-size preprocessing probe rows within the named absolute tolerance
`1e-5`. Default-thread V1→teacher exactly matched stored evidence; the forced
one-thread near-tie drift was at most `9.709e-6`. Hypothesis failures remain
visible and do not fail the run.

- [ ] **Step 4: Inspect the generated figure**

Open `results/figures/task4/baseline_examples.png` and confirm:

- every row has one query plus five results;
- labels and IDs are readable;
- grayscale, geometry/canvas, large-background, and weak-family failures appear;
- repeated IDs are labelled rather than silently hidden;
- no holdout or official teacher-test image appears.

- [ ] **Step 5: Optional commit checkpoint**

Only if commits were explicitly authorized:

```bash
git add results/evidence/task4/baseline_*.csv results/evidence/task4/baseline_cost.json results/figures/task4/baseline_examples.png
git commit -m "data(task4): record baseline retrieval evidence"
```

---

### Task 7: Freeze the ADR and update the narrative notebook

**Files:**
- Create: `docs/decisions/0022-task4-baseline-search.md`
- Modify:
  - `docs/decisions/README.md`
  - `docs/decisions/0009-task4-retrieval-isolation.md`
  - `docs/decisions/0020-task4-image-preprocessing.md`
  - `docs/reviews/open_decisions.md`
  - `src/fashion/README.md`
  - `scripts/README.md`
  - `results/evidence/task4/README.md`
  - `results/figures/task4/README.md`
  - `notebooks/task-4/05_task4_visual_search.ipynb`
  - `notebooks/task-4/PROGRESS.md`
  - `notebooks/README.md`
  - `README.md`
  - `tests/test_documentation.py`
  - `tests/test_notebook_scaffolds.py`

**Interfaces:**
- Consumes: generated evidence from Task 6
- Produces: one evidence-backed baseline decision and a concise notebook table/figure

- [ ] **Step 1: Write failing documentation tests**

Add tests that require:

```python
assert "0022-task4-baseline-search.md" in decision_index
assert "fashion.task4" in package_readme
assert "scripts/task4/run_preprocessing.py" in evidence_readme
assert "scripts/task4/run_baseline.py" in evidence_readme
assert "baseline_summary.csv" in evidence_readme
assert "baseline_examples.png" in figure_readme
```

Extend the runtime protected-data scan to include both
`src/fashion/task4/**/*.py` and compatibility wrappers. Add a notebook test
requiring the baseline section to contain the frozen probe, both hypotheses,
the four-direction table, timing verdicts, and no baseline `TODO(owner)`.

- [ ] **Step 2: Run the tests and confirm documentation is incomplete**

Run:

```bash
./.venv/bin/python -m pytest tests/test_documentation.py tests/test_notebook_scaffolds.py -q
```

Expected: failures name missing ADR 0022, new script paths, baseline artifacts,
and unresolved section-7 baseline text.

- [ ] **Step 3: Write ADR 0022 and supersession notes**

ADR 0022 records:

- exact probe version and `240×320` input;
- no tuning and no training-registry row;
- four source directions and same-source headline mean;
- random floor and 95% cross-source hypothesis;
- minimum-distance product fusion;
- timing, memory, and practical thresholds;
- failure-slice boundaries;
- measured quality, cost, and known canvas failure;
- holdout still sealed.

In ADR 0009, point stale query-count and retired protocol wording to ADR 0019
and ADR 0022. In ADR 0020, change the stale “all future candidates use
`96×128`” consequence into a clearly historical statement and point to ADR
0021. Do not rewrite either decision’s original evidence.

- [ ] **Step 4: Update package, runner, evidence, and project indexes**

Document:

- real ownership under `src/fashion/task4/`;
- compatibility-only ownership under `src/fashion/retrieval/`;
- the two Task 4 runner commands;
- every baseline artifact and its scope;
- no `results/runs.csv` row;
- timing values are machine-dependent.

Keep shared README edits narrow to reduce merge conflicts.

- [ ] **Step 5: Replace only the Milestone 4 notebook placeholders**

Use notebook code cells only to load tracked CSV/JSON files, format one compact
four-direction K=10 table, and display `baseline_examples.png`. Reusable
calculation stays in `fashion.task4`.

Replace section 7 with:

- the frozen baseline definition;
- hypothesis 1 and its measured pass/fail result;
- hypothesis 2 and its measured pass/fail result;
- explicit rejection language when a hypothesis fails.

Update sections 12–14 with slice counts, failure observations, p50/p95,
index-build time, index bytes, peak RSS, and practical verdicts. Leave learned
model and final-winner placeholders open.

- [ ] **Step 6: Execute and validate the notebook**

Run the repository’s existing notebook execution route if documented;
otherwise execute with:

```bash
./.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/task-4/05_task4_visual_search.ipynb
```

Then validate:

```bash
./.venv/bin/python - <<'PY'
import nbformat
for path in (
    "notebooks/task-4/01_v1_eda.ipynb",
    "notebooks/task-4/05_task4_visual_search.ipynb",
):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
PY
```

Expected: both notebooks validate; the main notebook has saved outputs and no
baseline calculation drift.

- [ ] **Step 7: Rerun documentation tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_documentation.py tests/test_notebook_scaffolds.py -q
./.venv/bin/python -m ruff check tests/test_documentation.py tests/test_notebook_scaffolds.py
```

Expected: all tests pass.

- [ ] **Step 8: Mark Milestone 4 complete**

Only after Tasks 6 and 7 pass, change the two Milestone 4 checklist items in
`notebooks/task-4/PROGRESS.md` to `[x]`. Keep Milestone 5 as the next milestone
and state that the holdout remains sealed.

- [ ] **Step 9: Optional commit checkpoint**

Only if commits were explicitly authorized:

```bash
git add docs/decisions/0009-task4-retrieval-isolation.md docs/decisions/0020-task4-image-preprocessing.md docs/decisions/0022-task4-baseline-search.md docs/decisions/README.md docs/reviews/open_decisions.md README.md notebooks/README.md notebooks/task-4/05_task4_visual_search.ipynb notebooks/task-4/PROGRESS.md src/fashion/README.md scripts/README.md results/evidence/task4/README.md results/figures/task4/README.md tests/test_documentation.py tests/test_notebook_scaffolds.py
git commit -m "docs(task4): freeze baseline search decision"
```

---

### Task 8: Final verification

**Files:**
- Verify all files changed in Tasks 1–7
- Do not stage, delete, reset, or commit unrelated working-tree files

**Interfaces:**
- Produces: evidence that Milestone 4 is complete without holdout access

- [ ] **Step 1: Run the complete Task 4 suite**

```bash
./.venv/bin/python -m pytest tests/task4 tests/test_documentation.py tests/test_notebook_scaffolds.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the full repository suite**

```bash
./.venv/bin/python -m pytest -q
```

Expected: all tests pass. If another branch has introduced unrelated failures,
record their exact test names and prove the focused Task 4 suite still passes;
do not hide or repair unrelated work.

- [ ] **Step 3: Run full lint and diff checks**

```bash
./.venv/bin/python -m ruff check src scripts tests
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 4: Verify no active Task 4 import uses the old owner**

Search active code and notebooks for `fashion.retrieval`. Expected matches are
limited to compatibility tests and compatibility modules. Historical design
records may retain old paths only with clear supersession text.

- [ ] **Step 5: Verify protected data stayed sealed**

Confirm:

- every tracked baseline table has `scope == "development"`;
- no baseline artifact contains holdout IDs;
- no Task 4 runtime module imports or calls
  `load_splits_for_final_evaluation`;
- no `results/runs.csv` baseline row was created;
- no model checkpoint was created.

- [ ] **Step 6: Compare the final working tree with the plan**

Run `git status --short`. Check that every intended file is present and that no
unrelated uncommitted file was modified, staged, deleted, or reset by this
implementation.

Do not make another aggregate commit or push unless the user explicitly
requests it after reviewing `git status --short`.
