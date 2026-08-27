# Task 4 Image Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and evidence one fold-safe image preprocessing contract for
Task 4 teacher, V1, and arbitrary user images.

**Architecture:** A focused preprocessing module owns deterministic RGB
letterboxing, fold-only statistics, and lossless development caches. A probe
module owns the fixed untrained HSV/edge descriptor and exact cosine ranking.
An experiment module composes those pieces with the already-frozen retrieval
protocol, while the notebook remains a short narrative that generates and
displays tracked evidence.

**Tech Stack:** Python 3.12, NumPy, pandas, Pillow, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-task4-image-preprocessing-design.md`

## Global Constraints

- Use only `data/processed/splits.csv`; never create another split.
- Open pixels only for `partition == "development"` during this milestone.
- Never call `load_splits_for_final_evaluation`.
- Never read official teacher-test products.
- Treat V1 as an ID-keyed variant, not independent data.
- Candidate sizes are `(width, height)`: `60×80`, `96×128`, `240×320`.
- Code array shapes use `(height, width)`.
- Validation fold `1` is the ordinary comparison; the top two sizes receive
  all-five-fold stability evidence.
- Protocol A query-mean linear nDCG@10 chooses the preprocessing size through
  the equal teacher→teacher and V1→V1 source mean.
- Cross-source directions and Protocol B remain supporting evidence.
- Use no pretrained or learned model in this milestone.
- Do not stage, commit, revert, or clean any existing working-tree changes.

---

### Task 1: Deterministic Task 4 Image Contract

**Files:**
- Create: `src/fashion/retrieval/preprocessing.py`
- Create: `tests/test_retrieval_preprocessing.py`
- Modify: `src/fashion/retrieval/__init__.py`

**Interfaces:**
- Produces:
  - `PreprocessingContract(width: int, height: int, pad_color=(255, 255, 255))`
  - `PreprocessedImage(pixels, content_mask, content_bounds)`
  - `preprocess_image(image, contract) -> PreprocessedImage`
  - `load_preprocessed_image(path, contract) -> PreprocessedImage`
  - `fit_fold_rgb_statistics(frame, path_column, contract, validation_fold, root) -> dict`

- [ ] **Step 1: Write failing image-contract tests**

Cover:

```python
def test_preprocess_composites_transparency_on_white() -> None: ...
def test_preprocess_corrects_exif_before_letterboxing() -> None: ...
def test_preprocess_converts_grayscale_to_rgb_uint8() -> None: ...
def test_preprocess_preserves_wide_and_tall_geometry_with_masks() -> None: ...
def test_fold_statistics_reject_non_development_rows() -> None: ...
def test_fold_statistics_use_only_non_validation_content_pixels() -> None: ...
```

Expected literals must assert output shape, exact white transparent pixels,
mask bounds, source IDs used, and hand-computed channel statistics.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_retrieval_preprocessing.py -q
```

Expected: collection fails because `fashion.retrieval.preprocessing` does not
exist.

- [ ] **Step 3: Implement the minimal contract**

Use `ImageOps.exif_transpose`, white alpha compositing, RGB conversion, and
`fashion.data.images.transform_image_with_mask` with
`normalize_range=False`. Cast the exact result to `uint8`. Validate positive
sizes and non-empty source geometry. Compute statistics through
`StreamingStats` after scaling content pixels to `[0, 1]`.

`fit_fold_rgb_statistics` must reject any frame containing a non-development
row, remove `cv_fold == validation_fold`, reject an empty fit, and return:

```python
{
    "validation_fold": 1,
    "training_rows": 26217,
    "training_id_sha256": "...",
    "mean": [float, float, float],
    "std": [float, float, float],
}
```

- [ ] **Step 4: Run focused and existing image tests**

```bash
./.venv/bin/python -m pytest \
  tests/test_retrieval_preprocessing.py tests/data/test_images.py -q
```

Expected: PASS.

---

### Task 2: Versioned Development-Only Lossless Cache

**Files:**
- Create: `src/fashion/retrieval/cache.py`
- Extend: `tests/test_retrieval_preprocessing.py`
- Modify: `src/fashion/retrieval/__init__.py`

**Interfaces:**
- Consumes: `PreprocessingContract`, `load_preprocessed_image`
- Produces:
  - `CacheManifest`
  - `ensure_development_image_cache(frame, path_column, source, contract, cache_root, root)`
  - `load_development_image_cache(cache_dir)`

- [ ] **Step 1: Write failing cache tests**

Cover:

```python
def test_cache_rejects_holdout_before_opening_pixels() -> None: ...
def test_cache_is_sorted_by_numeric_product_id() -> None: ...
def test_cache_round_trips_uint8_pixels_and_bounds() -> None: ...
def test_cache_reuses_only_an_exact_manifest_match() -> None: ...
def test_cache_rebuilds_when_source_fingerprint_changes() -> None: ...
```

The holdout test uses a missing path and must fail on partition before any file
open. The manifest-match test changes one source SHA/path/size field and
expects a rebuild.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/test_retrieval_preprocessing.py -q
```

Expected: import failure for `fashion.retrieval.cache`.

- [ ] **Step 3: Implement memory-mappable cache files**

Write atomically into a temporary sibling directory, then rename:

```text
<cache_dir>/
  ids.npy
  images.npy
  content_bounds.npy
  manifest.json
```

Store `images.npy` as `(N, H, W, 3)` `uint8`, bounds as `(N, 4)` integers, and
IDs as sorted `int64`. Manifest fields include schema version, source,
contract, rows, array shape, ID digest, source fingerprint, and file names.
Never include targets.

- [ ] **Step 4: Run focused tests**

```bash
./.venv/bin/python -m pytest tests/test_retrieval_preprocessing.py -q
```

Expected: PASS.

---

### Task 3: Fixed Spatial HSV-and-Edge Probe

**Files:**
- Create: `src/fashion/retrieval/probe.py`
- Create: `tests/test_retrieval_probe.py`
- Modify: `src/fashion/retrieval/__init__.py`

**Interfaces:**
- Produces:
  - `extract_spatial_probe(pixels, content_mask) -> np.ndarray`
  - `rank_probe_embeddings(query_ids, query_features, gallery_ids, gallery_features, views, protocol, max_k, chunk_size) -> pd.DataFrame`

- [ ] **Step 1: Write failing descriptor tests**

Cover:

```python
def test_probe_excludes_padding_from_colour_and_edges() -> None: ...
def test_probe_keeps_spatial_colour_layout() -> None: ...
def test_probe_blocks_have_equal_unit_weight() -> None: ...
def test_probe_rejects_empty_content() -> None: ...
```

Use hand-built solid-colour and two-region arrays. Assert identical descriptors
when only masked padding changes, different descriptors when top and bottom
colours swap, finite values, and final unit norm.

- [ ] **Step 2: Verify descriptor tests fail**

```bash
./.venv/bin/python -m pytest tests/test_retrieval_probe.py -q
```

Expected: import failure for `fashion.retrieval.probe`.

- [ ] **Step 3: Implement the fixed descriptor**

Use a `4×4` spatial grid. Use fixed per-channel HSV bins `(8, 4, 4)` and nine
unsigned gradient-orientation bins. Weight gradient votes by magnitude.
L2-normalize the full colour block and full edge block separately, multiply
each by `1 / sqrt(2)`, concatenate, and L2-normalize once more. Add no fit or
tuning API.

- [ ] **Step 4: Write and verify failing exact-ranking tests**

Cover:

```python
def test_ranker_uses_cosine_distance_and_numeric_id_ties() -> None: ...
def test_ranker_filters_family_candidates_before_top_k() -> None: ...
def test_ranker_rejects_misaligned_ids_and_features() -> None: ...
def test_ranker_returns_every_query_with_max_k_rows() -> None: ...
```

Use literal vectors and IDs including `2` and `10`.

- [ ] **Step 5: Implement chunked exact ranking**

Validate unit-finite feature matrices and one unique integer-compatible ID per
row. Compute `1 - query @ gallery.T` in bounded chunks. For Protocol B, apply
the existing self/SHA/duplicate exclusions before selection. Include all
distance ties at the K boundary before sorting by `(distance, numeric ID)`.
Pass output through `prepare_rankings` as a final contract check.

- [ ] **Step 6: Run probe and protocol tests**

```bash
./.venv/bin/python -m pytest \
  tests/test_retrieval_probe.py tests/test_retrieval_protocol.py -q
```

Expected: PASS.

---

### Task 4: Development-Only Preprocessing Experiment

**Files:**
- Create: `src/fashion/retrieval/preprocessing_experiment.py`
- Create: `tests/test_preprocessing_experiment.py`
- Modify: `src/fashion/retrieval/__init__.py`

**Interfaces:**
- Consumes: preprocessing, probe, cache, and protocol APIs
- Produces:
  - `FeatureIndex(ids, features, transform_seconds, source_bytes)`
  - `extract_feature_index(frame, path_column, contract, root, workers)`
  - `evaluate_source_pair(query_index, gallery_index, primary_views, family_views)`
  - `build_size_selection(results) -> pd.DataFrame`
  - `select_top_sizes(selection, count=2) -> tuple[str, ...]`
  - `build_odd_aspect_queries(image, orientation) -> Image`
  - `run_preprocessing_experiment(...) -> PreprocessingExperiment`

- [ ] **Step 1: Write failing experiment tests**

Cover:

```python
def test_feature_extraction_rejects_non_development_rows() -> None: ...
def test_source_matrix_contains_four_directions_per_size() -> None: ...
def test_size_selection_uses_only_equal_same_source_ndcg_mean() -> None: ...
def test_top_size_ties_break_toward_fewer_pixels() -> None: ...
def test_stability_uses_each_fold_once_for_only_top_two_sizes() -> None: ...
def test_wide_and_tall_canvases_do_not_crop_content() -> None: ...
```

The selection fixture must make cross-source scores extreme and prove they do
not affect the winner.

- [ ] **Step 2: Verify experiment tests fail**

```bash
./.venv/bin/python -m pytest tests/test_preprocessing_experiment.py -q
```

Expected: import failure for `preprocessing_experiment`.

- [ ] **Step 3: Implement extraction and source-pair evaluation**

Extract each source/size feature index once. Merge by numeric ID, never row
position. Evaluate both frozen protocols with existing metric functions.
Record fold, size, query source, gallery source, protocol, metric, K,
aggregation, value, coverage, ties, transform throughput, source bytes, feature
bytes, and tensor bytes.

- [ ] **Step 4: Implement selection and stability**

For each size, select the Protocol A `ndcg`, K=10, `query_mean` rows for
teacher→teacher and V1→V1, validate both exist exactly once, and take their
equal mean. Sort descending by score, then ascending pixel count. Reuse
extracted features for folds `0..4` for only the top two sizes and report the
unweighted mean and sample standard deviation.

- [ ] **Step 5: Implement odd-aspect evidence**

Create deterministic `2:1` wide and `1:2` tall white canvases around clean
queries without resizing, cropping, or changing content. At the selected size,
report clean versus odd-aspect Protocol A nDCG@10 and mean Top-10 product-ID
overlap.

- [ ] **Step 6: Run experiment tests and focused suite**

```bash
./.venv/bin/python -m pytest \
  tests/test_retrieval_preprocessing.py \
  tests/test_retrieval_probe.py \
  tests/test_preprocessing_experiment.py \
  tests/test_retrieval_protocol.py -q
```

Expected: PASS.

---

### Task 5: Generate Milestone 3 Evidence and Local Caches

**Files:**
- Create locally: `data/processed/task4/preprocessing/**`
- Create: `results/evidence/task4/preprocessing_comparison.csv`
- Create: `results/evidence/task4/preprocessing_size_selection.csv`
- Create: `results/evidence/task4/preprocessing_stability.csv`
- Create: `results/evidence/task4/preprocessing_robustness.csv`
- Create: `results/evidence/task4/preprocessing_contract.json`
- Create: `results/evidence/task4/preprocessing_normalization_fold1.json`
- Create: `results/figures/task4/preprocessing_comparison.png`

**Interfaces:**
- Consumes the approved source matrix and all reusable modules.
- Produces deterministic tracked evidence and reusable ignored caches.

- [ ] **Step 1: Run the full development experiment**

Load splits through `load_splits`, load
`data/processed/task4/external_variant_index.csv.gz`, restrict both to the
32,773 development IDs, and execute the three-size source matrix on fold `1`.
Run the five-fold check for the selected top two sizes.

- [ ] **Step 2: Fit fold-1 source statistics at the selected size**

Fit teacher and V1 statistics separately from folds `0`, `2`, `3`, and `4`.
Write means, standard deviations, training row counts, and training-ID digests
to one deterministic JSON file. Do not open fold `1`, holdout, or quarantine
pixels for the fit.

- [ ] **Step 3: Build both selected-size development caches**

Build and immediately reopen teacher and V1 lossless caches. Assert exact ID
order, shape, dtype, row count, and metadata match. Do not add cache files to
Git.

- [ ] **Step 4: Write tracked evidence and one figure**

Write deterministic CSV/JSON files. The figure has:

1. same-source nDCG@10 against pixel count with teacher and V1 lines;
2. cross-source nDCG@10 at each size;
3. transform throughput and tensor-size cost for each size.

Do not repeat the existing geometry or retrieval-protocol figures.

- [ ] **Step 5: Sanity-check artifact scope**

Assert every artifact says `scope=development`, contains no holdout ID list,
contains all three candidate sizes, all four source directions, and all five
stability folds for exactly two sizes.

---

### Task 6: Notebook, ADR, Progress, and Documentation

**Files:**
- Modify: `notebooks/task-4/05_task4_visual_search.ipynb`
- Create: `docs/decisions/0020-task4-image-preprocessing.md`
- Modify: `docs/decisions/README.md`
- Modify: `notebooks/task-4/PROGRESS.md`
- Modify: `results/evidence/task4/README.md`
- Modify: `results/figures/task4/README.md`
- Modify: `tests/test_notebook_scaffolds.py`
- Modify: `tests/test_documentation.py`

**Interfaces:**
- Consumes all Task 5 evidence.
- Produces the visible milestone narrative and frozen decision.

- [ ] **Step 1: Write failing notebook and documentation tests**

Require the notebook to contain:

- the selected contract and size;
- all three candidate sizes;
- all four source directions;
- fold-fit normalization and no-refit wording;
- the quality/cost table;
- the preprocessing figure;
- a limitation saying the fixed probe does not prove learned-model quality;
- executed non-empty code cells with no saved errors or cache warnings.

Require ADR 0020 in the accepted-decision index and Milestone 3 checked complete
with its evidence paths.

- [ ] **Step 2: Run tests and verify RED**

```bash
./.venv/bin/python -m pytest \
  tests/test_notebook_scaffolds.py tests/test_documentation.py -q
```

Expected: FAIL because the milestone narrative and ADR do not yet exist.

- [ ] **Step 3: Replace notebook sections 5 and 6 placeholders**

Keep code concise: load generated evidence, display the compact table, render
or display the single figure, and show fixed examples. Cross-reference the
existing V1 geometry and protocol figures instead of repeating them. Do not
fill later model, training, error-analysis, or final-evaluation sections.

- [ ] **Step 4: Write ADR 0020 and update progress/readmes**

Record the winning size from generated evidence, exact contract, probe limits,
source-selection deferral, cache scope, and fold-only normalization. Mark
Milestone 3 complete only after all verification passes.

- [ ] **Step 5: Execute the notebook from a clean kernel**

Use the repository virtual environment and temporary Matplotlib/XDG cache
directories. Save outputs. Confirm no error output and no Fontconfig or
Matplotlib cache warning.

- [ ] **Step 6: Run final verification**

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m ruff check src tests
```

Also validate:

- every non-empty Task 4 code cell has an execution count;
- no saved error output exists;
- all tracked evidence and the figure exist;
- local caches reopen and match their manifests;
- `git status --short` shows no accidental cache files and no unrelated files
  were staged, reverted, or deleted.

Expected: all checks pass.
