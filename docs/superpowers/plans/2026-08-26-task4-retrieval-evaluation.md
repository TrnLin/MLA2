# Task 4 Retrieval Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and document the frozen, fold-safe Task 4 retrieval evaluation framework.

**Architecture:** Add one pure pandas/NumPy retrieval protocol module. It receives
already-loaded labelled frames and ranked product candidates, constructs the two approved
query/gallery protocols, enforces leakage rules, computes coverage, and scores rankings.
The main Task 4 notebook calls this module to generate auditable coverage evidence without
training a model or unlocking protected labels.

**Tech Stack:** Python 3.12, pandas 3.0.5, NumPy 2.5.2, pytest 9.1.1, nbformat/Jupyter.

**Spec:** `docs/superpowers/specs/2026-08-26-task4-retrieval-evaluation-design.md`

## Global Constraints

- `data/processed/splits.csv` is the only split.
- The fixed candidate-comparison validation fold is `1`.
- Holdout stays sealed; retrieval runtime code must not call the final-evaluation unlock.
- Quarantine and official teacher-test rows never enter development queries or galleries.
- V1 variants collapse to one product ID before Top-K scoring.
- nDCG uses linear relevance gains `0`, `1`, and `2`.
- Protocol A mean per-query nDCG@10 is the winner-selection score.
- Protocol B is supporting evidence only and removes self, same-SHA, and same-duplicate
  candidates before Top-K.
- Do not add dependencies.
- Do not create a git commit unless the user explicitly asks.

---

### Task 1: Fold-safe views and relevance coverage

**Files:**
- Create: `src/fashion/retrieval/protocol.py`
- Create: `tests/test_retrieval_protocol.py`
- Modify: `src/fashion/retrieval/__init__.py`

**Interfaces:**
- Consumes: a protected-safe split frame returned by `fashion.data.dataset.load_splits`.
- Produces:
  - `FIXED_VALIDATION_FOLD: int = 1`
  - `K_VALUES: tuple[int, ...] = (5, 10, 20)`
  - `RetrievalViews(queries: pd.DataFrame, gallery: pd.DataFrame)`
  - `build_development_views(splits, validation_fold=1) -> tuple[RetrievalViews, RetrievalViews]`
  - `primary_relevance(query, candidates) -> np.ndarray`
  - `family_relevance(query, candidates) -> np.ndarray`
  - `family_candidate_mask(query, candidates) -> pd.Series`
  - `compute_relevance_coverage(primary, family, k_values=K_VALUES) -> pd.DataFrame`

- [ ] **Step 1: Write the split-frame test helper and failing view tests**

Create a local `_split_frame()` helper in `tests/test_retrieval_protocol.py`. Give each
development fold at least one family so `validate_splits` accepts all five folds. Include
the full structural and protected-target contract required by
`fashion.data.splits.validate_splits`.

```python
def _row(
    product_id: int,
    fold: int,
    *,
    article_type: str,
    colour: str,
    family: str,
    duplicate: str,
    sha256: str,
) -> dict[str, object]:
    return {
        "id": product_id,
        "sha256": sha256,
        "duplicate_group": duplicate,
        "product_name_key": family,
        "product_family_group": family,
        "partition": "development",
        "cv_fold": fold,
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "quarantine_reason": "",
        "articleType": article_type,
        "baseColour": colour,
        "season": "Summer",
        "gender": "Unisex",
        "usage": "Casual",
        "has_articleType_label": True,
        "has_season_label": True,
        "has_gender_label": True,
        "has_usage_label": True,
    }
```

Test that fold `1` becomes Protocol A queries, the other folds become its gallery, and
Protocol B uses fold `1` for both query and raw gallery views.

```python
def test_development_views_freeze_fold_one() -> None:
    primary, family = build_development_views(_split_frame())

    assert set(primary.queries["cv_fold"]) == {1}
    assert set(primary.gallery["cv_fold"]) == {0, 2, 3, 4}
    assert set(family.queries["cv_fold"]) == {1}
    assert set(family.gallery["cv_fold"]) == {1}
```

- [ ] **Step 2: Run the view test and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py::test_development_views_freeze_fold_one -v
```

Expected: FAIL because `fashion.retrieval.protocol` does not exist.

- [ ] **Step 3: Implement constants, `RetrievalViews`, and view construction**

Use `get_cv_split` so the canonical split validation remains the source of truth.

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fashion.data.dataset import get_cv_split

FIXED_VALIDATION_FOLD = 1
K_VALUES = (5, 10, 20)


@dataclass(frozen=True)
class RetrievalViews:
    queries: pd.DataFrame
    gallery: pd.DataFrame


def build_development_views(
    splits: pd.DataFrame,
    validation_fold: int = FIXED_VALIDATION_FOLD,
) -> tuple[RetrievalViews, RetrievalViews]:
    gallery, queries = get_cv_split(splits, validation_fold)
    _require_columns(
        queries,
        {
            "id",
            "sha256",
            "duplicate_group",
            "product_family_group",
            "articleType",
            "baseColour",
        },
    )
    _assert_primary_isolation(queries, gallery)
    return RetrievalViews(queries.copy(), gallery.copy()), RetrievalViews(
        queries.copy(), queries.copy()
    )
```

`_assert_primary_isolation` must raise `ValueError` naming the offending key when any
non-empty `id`, `sha256`, `duplicate_group`, or `product_family_group` value appears in
both Protocol A sides.

- [ ] **Step 4: Run the view test and verify GREEN**

Run the test from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing relevance and exact-duplicate tests**

```python
def test_primary_relevance_uses_linear_two_one_zero_grades() -> None:
    query = pd.Series({"articleType": "Tshirts", "baseColour": "Blue"})
    candidates = pd.DataFrame(
        {
            "articleType": ["Tshirts", "Tshirts", "Jeans"],
            "baseColour": ["Blue", "Red", "Blue"],
        }
    )

    assert primary_relevance(query, candidates).tolist() == [2, 1, 0]


def test_family_mask_removes_self_sha_and_duplicate_group() -> None:
    query = pd.Series(
        {"id": 10, "sha256": "same", "duplicate_group": "dup", "product_family_group": "f"}
    )
    candidates = pd.DataFrame(
        {
            "id": [10, 11, 12, 13, 14],
            "sha256": ["same", "same", "other-12", "other-13", "other-14"],
            "duplicate_group": ["dup", "other-11", "dup", "other-13", "other-14"],
            "product_family_group": ["f", "f", "f", "f", "other-family"],
        }
    )

    assert family_candidate_mask(query, candidates).tolist() == [False, False, False, True, True]
    assert family_relevance(query, candidates.loc[[3, 4]]).tolist() == [1, 0]
```

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py -k "relevance or family_mask" -v
```

Expected: FAIL because the relevance and family-mask functions are missing.

- [ ] **Step 7: Implement relevance and family exclusion**

`primary_relevance` returns an integer NumPy array. `family_candidate_mask` allows all
non-excluded candidates, including irrelevant families, because retrieval must rank
against the full unseen-fold gallery.

```python
def primary_relevance(query: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    same_type = candidates["articleType"].eq(query["articleType"]).to_numpy()
    same_colour = candidates["baseColour"].eq(query["baseColour"]).to_numpy()
    return same_type.astype(np.int8) + (same_type & same_colour).astype(np.int8)


def family_candidate_mask(query: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    return (
        candidates["id"].ne(query["id"])
        & candidates["sha256"].ne(query["sha256"])
        & candidates["duplicate_group"].ne(query["duplicate_group"])
    )


def family_relevance(query: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    return candidates["product_family_group"].eq(query["product_family_group"]).to_numpy(
        dtype=np.int8
    )
```

- [ ] **Step 8: Run relevance tests and verify GREEN**

Run the command from Step 6. Expected: PASS.

- [ ] **Step 9: Write a failing coverage test**

The fixture must include a fold-1 query with only a broad grade-1 positive, a query with
no Protocol A positive, a family recovered after exclusions, and a family made
unscorable by exact-duplicate removal.

```python
def test_coverage_reports_undefined_and_strict_positive_counts() -> None:
    primary, family = build_development_views(_coverage_split_frame())

    coverage = compute_relevance_coverage(primary, family, k_values=(1, 2))

    primary_k1 = coverage.query("protocol == 'primary' and k == 1").iloc[0]
    family_k1 = coverage.query("protocol == 'family' and k == 1").iloc[0]
    assert primary_k1["total_queries"] == 4
    assert primary_k1["excluded_queries"] == 1
    assert primary_k1["zero_strict_queries"] == 2
    assert family_k1["excluded_queries"] == 2
```

- [ ] **Step 10: Run the coverage test and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py::test_coverage_reports_undefined_and_strict_positive_counts -v
```

Expected: FAIL because `compute_relevance_coverage` is missing.

- [ ] **Step 11: Implement vectorised coverage**

For Protocol A, map gallery counts grouped by `articleType` and by
`(articleType, baseColour)` onto each query. For Protocol B, compute eligible family
positives without an O(query × gallery) join:

```python
family_size = queries.groupby("product_family_group")["id"].transform("size")
duplicate_size = queries.groupby(
    ["product_family_group", "duplicate_group"], dropna=False
)["id"].transform("size")
sha_size = queries.groupby(["product_family_group", "sha256"], dropna=False)[
    "id"
].transform("size")
intersection_size = queries.groupby(
    ["product_family_group", "duplicate_group", "sha256"], dropna=False
)["id"].transform("size")
family_positive_count = family_size - duplicate_size - sha_size + intersection_size
```

Return one row per protocol and K with these stable columns:

```python
COVERAGE_COLUMNS = (
    "protocol",
    "k",
    "total_queries",
    "scored_queries",
    "excluded_queries",
    "zero_strict_queries",
    "fewer_than_k_strict_queries",
)
```

- [ ] **Step 12: Run Task 1 tests and export the public API**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py -v
```

Expected: PASS.

Add the Task 1 constants, dataclass, and functions to
`src/fashion/retrieval/__init__.py`.

---

### Task 2: Deterministic ranking preparation and metrics

**Files:**
- Modify: `src/fashion/retrieval/protocol.py`
- Modify: `src/fashion/retrieval/__init__.py`
- Modify: `tests/test_retrieval_protocol.py`

**Interfaces:**
- Consumes: a long frame with `query_id`, `candidate_id`, and `distance`.
- Produces:
  - `prepare_rankings(rankings, views, protocol, max_k=20) -> pd.DataFrame`
  - `evaluate_primary_rankings(rankings, views, k_values=K_VALUES) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `evaluate_family_rankings(rankings, views, k=10) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write failing ranking-normalisation tests**

```python
def test_prepare_rankings_collapses_variants_and_breaks_ties_by_id() -> None:
    views = _small_primary_views()
    raw = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "candidate_id": [4, 3, 3, 2],
            "distance": [0.2, 0.1, 0.3, 0.1],
        }
    )

    ranked = prepare_rankings(raw, views, protocol="primary", max_k=3)

    assert ranked["candidate_id"].tolist() == [2, 3, 4]
    assert ranked["rank"].tolist() == [1, 2, 3]
```

Also test that unknown query/candidate IDs, missing required columns, duplicate product
IDs after preparation, and fewer than `max_k` eligible results raise clear `ValueError`s.

- [ ] **Step 2: Run ranking tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py -k prepare_rankings -v
```

Expected: FAIL because `prepare_rankings` is missing.

- [ ] **Step 3: Implement deterministic ranking preparation**

Implementation order:

1. Validate columns and protocol name.
2. Validate query and candidate IDs against the supplied views.
3. Stable-sort by `query_id`, ascending `distance`, then numeric `candidate_id`.
4. Drop duplicate `(query_id, candidate_id)` rows, retaining the nearest variant.
5. For Protocol B, merge query/candidate SHA and duplicate metadata and apply
   `family_candidate_mask` per query.
6. Assign one-based rank within each query.
7. Keep ranks through `max_k`.
8. Require exactly `max_k` eligible results for every query that is supplied.

Compute `tie_rate` later from the prepared Top-K rows as the fraction of rows whose
distance is duplicated within that query.

- [ ] **Step 4: Run ranking tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing primary-metric tests with hand-calculated values**

Use a three-result ranking with grades `[2, 1, 0]` and a reversed ranking
`[0, 1, 2]`. Assert the perfect order is `1.0` and calculate the reversed expected value
directly in the test:

```python
expected = (1 / np.log2(3) + 2 / np.log2(4)) / (
    2 / np.log2(2) + 1 / np.log2(3)
)
assert result.loc[result["query_id"].eq(2), "ndcg_at_3"].item() == pytest.approx(expected)
```

Assert strict and broad precision separately. Add an undefined query and assert it is
absent from metric means but present in the coverage counts.

- [ ] **Step 6: Run primary metric tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py -k primary_metric -v
```

Expected: FAIL because `evaluate_primary_rankings` is missing.

- [ ] **Step 7: Implement Protocol A metrics and aggregates**

For every query and K:

```python
discounts = np.log2(np.arange(2, k + 2))
dcg = float((observed_grades[:k] / discounts).sum())
ideal_grades = np.array([2] * strict_count + [1] * (broad_count - strict_count))
idcg = float((ideal_grades[:k] / discounts[: len(ideal_grades[:k])]).sum())
ndcg = dcg / idcg if idcg > 0 else np.nan
```

The per-query frame contains `query_id`, `articleType`, each `ndcg_at_K`,
`precision_any_at_K`, and `precision_strict_at_K`. The summary frame uses stable columns:
`metric`, `k`, `aggregation`, `value`, `query_count`, and `class_count`.

Emit `aggregation="query_mean"` and `aggregation="article_type_macro"`. Compute the
macro by averaging queries within each present class, then averaging those class means.

- [ ] **Step 8: Run primary metric tests and verify GREEN**

Run the command from Step 6. Expected: PASS.

- [ ] **Step 9: Write failing family-metric tests**

Create two scorable family queries. One retrieves one of two eligible relatives; the
other retrieves none. At K=2, assert:

```python
assert per_query["recall_at_2"].tolist() == [0.5, 0.0]
assert per_query["hit_rate_at_2"].tolist() == [1.0, 0.0]
assert per_query["precision_at_2"].tolist() == [0.5, 0.0]
```

Add a query whose only relative has the same SHA and verify that it is excluded from the
metric average and counted as undefined.

- [ ] **Step 10: Run family metric tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py -k family_metric -v
```

Expected: FAIL because `evaluate_family_rankings` is missing.

- [ ] **Step 11: Implement Protocol B metrics and summaries**

Score only queries with at least one eligible family positive. Recall is relevant
retrieved divided by all eligible family positives. Hit Rate is one when any relevant
candidate appears. Precision uses K as its denominator.

The summary frame uses `query_mean` only and includes coverage, K, tie rate, and the
three family metrics without combining them with Protocol A.

- [ ] **Step 12: Run all protocol tests and export Task 2 APIs**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py -v
```

Expected: PASS.

Export the three Task 2 functions from `fashion.retrieval`.

---

### Task 3: Freeze the decision record, notebook, and evidence

**Files:**
- Create: `docs/decisions/0019-task4-retrieval-evaluation.md`
- Modify: `docs/decisions/README.md`
- Modify: `notebooks/task-4/05_task4_visual_search.ipynb`
- Modify: `notebooks/task-4/PROGRESS.md`
- Create: `results/evidence/task4/retrieval_protocol_coverage.csv`
- Modify: `results/evidence/task4/README.md`
- Modify: `tests/test_notebook_scaffolds.py`

**Interfaces:**
- Consumes: `load_splits`, `build_development_views`, and
  `compute_relevance_coverage`.
- Produces: an executed notebook section and tracked coverage CSV proving the frozen
  protocol's real query counts.

- [ ] **Step 1: Change scaffold tests first and verify RED**

Refactor `test_task_scaffolds_leave_owner_decisions_open` so it continues to enforce
open owner decisions for Tasks 1–3, but no longer requires Task 4's evaluation decisions
to remain open or every Task 4 cell to be markdown.

Add a focused test:

```python
def test_task4_evaluation_protocol_is_frozen_and_executed() -> None:
    notebook = nbformat.read(_task_path("05_task4_visual_search.ipynb"), as_version=4)
    source = _source(notebook)
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()
    ]

    for required in (
        "fold `1`",
        "nDCG@10",
        "Recall@10",
        "same `articleType` and `baseColour`",
        "results/evidence/task4/retrieval_protocol_coverage.csv",
    ):
        assert required in source
    assert "Primary ranking-quality metric: TODO(owner)" not in source
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert all(not cell.get("outputs") or not any(
        output.get("output_type") == "error" for output in cell["outputs"]
    ) for cell in code_cells)
    assert "load_splits_for_final_evaluation" not in source
```

Update `test_task_metric_placeholders_are_explicit` to assert Task 4's exact frozen
metric text while retaining placeholders for Tasks 1–3.

Run:

```bash
./.venv/bin/python -m pytest tests/test_notebook_scaffolds.py -v
```

Expected: FAIL because the Task 4 notebook is still an unexecuted planning scaffold.

- [ ] **Step 2: Add accepted decision 0019**

Record:

- fixed fold `1` and the seed draw;
- Protocol A and Protocol B;
- linear nDCG@10, supporting metrics, K values, query averaging, class macro;
- undefined-query handling and deterministic tie-break;
- duplicate/family isolation facts;
- top-two five-fold stability reruns;
- one-shot sealed holdout rule;
- limitations of metadata relevance and V1 independence.

Add `0019-task4-retrieval-evaluation.md` to the accepted decisions list. State that it
supersedes decision 0009's stale fixed query counts and retired validation split details
while preserving its self-match principle.

- [ ] **Step 3: Update the notebook narrative and add one evidence code cell**

Use `EditNotebook`; do not edit raw notebook JSON.

Replace only the now-settled owner fields in cells 0, 1, 2, 13, 14, 19, 22, 23, 24,
and 25. Keep preprocessing, model, run registry, and final winner fields open.

Use one code cell:

```python
from fashion.data.dataset import load_splits
from fashion.retrieval.protocol import (
    build_development_views,
    compute_relevance_coverage,
)

splits = load_splits()
primary_views, family_views = build_development_views(splits)
protocol_coverage = compute_relevance_coverage(primary_views, family_views)
coverage_path = "../../results/evidence/task4/retrieval_protocol_coverage.csv"
protocol_coverage.to_csv(coverage_path, index=False)
protocol_coverage
```

Explain directly above it that this uses development folds only and does not train a
model or unlock holdout.

- [ ] **Step 4: Execute the notebook and verify evidence**

Run from `notebooks/task-4` so the relative evidence path resolves:

```bash
../../.venv/bin/jupyter nbconvert --to notebook --execute 05_task4_visual_search.ipynb --output 05_task4_visual_search.ipynb --ExecutePreprocessor.timeout=300
```

Expected: all code cells have execution counts, no error outputs, and
`results/evidence/task4/retrieval_protocol_coverage.csv` exists.

- [ ] **Step 5: Update progress and evidence README**

Mark all Milestone 2 items complete and set Milestone 3 as current. Document the coverage
CSV as generated by the main Task 4 notebook and define its columns.

- [ ] **Step 6: Run documentation tests and verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_notebook_scaffolds.py tests/test_documentation.py -v
```

Expected: PASS.

---

### Task 4: Final verification

**Files:**
- Verify all files changed in Tasks 1–3.

**Interfaces:**
- Produces: evidence that the protocol implementation, notebook, documentation, and
  repository conventions agree.

- [ ] **Step 1: Run focused tests**

```bash
./.venv/bin/python -m pytest tests/test_retrieval_protocol.py tests/test_notebook_scaffolds.py tests/test_documentation.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

```bash
./.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run Ruff**

```bash
./.venv/bin/python -m ruff check src tests
```

Expected: no errors.

- [ ] **Step 4: Validate the executed notebook**

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path

import nbformat

path = Path("notebooks/task-4/05_task4_visual_search.ipynb")
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
code = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
errors = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
assert code
assert all(cell.execution_count is not None for cell in code)
assert errors == []
print(f"{len(code)} code cells executed; zero error outputs")
PY
```

Expected: at least one executed code cell and zero error outputs.

- [ ] **Step 5: Check the final diff without committing**

```bash
git status --short
git diff --check
```

Confirm that unrelated `.agents/`, `skills-lock.json`, and earlier data-preparation
changes were not edited or removed by this work. Do not commit unless the user asks.

