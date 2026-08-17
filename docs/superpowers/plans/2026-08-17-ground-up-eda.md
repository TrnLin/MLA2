# Ground-Up Exploratory Data Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the outdated EDA with a tested, reproducible analysis of the 38,612 usable official training products and generate report-ready evidence.

**Architecture:** Shared raw-data primitives move to `fashion.data_audit`, preserving dataset-comparison behaviour. A small public interface in `fashion.eda` coordinates pure metadata metrics, deterministic image analysis, and plotting helpers. The notebook remains narrative and consumes cached outputs rather than owning analysis logic.

**Tech Stack:** Python 3.11+, pandas, NumPy, Matplotlib, Pillow, Jupyter/nbformat, pytest

## Global Constraints

- Run Python through `./.venv/bin/python`.
- Use only the 38,617 official teacher-training IDs; report and remove the five image-less IDs to obtain 38,612 usable products.
- Never expose labels for the 5,829 official test IDs.
- Read CSVs with `keep_default_na=False`; literal `usage == "NA"` is not missing.
- Count paired original and 60×80 images as two views of one product, never as two products.
- Never call `train_test_split` or create/modify `data/processed/splits.csv`.
- Use `RANDOM_SEED = 2753`.
- Keep analysis logic out of notebook cells.
- Treat perceptual-hash matches as review candidates only.
- Use descriptive measures and effect sizes, not hypothesis tests or p-values.
- Save generated EDA evidence under `results/figures/eda/`.
- Make no rare-class, cleaning, splitting, or final modelling decision.
- Do not create Git commits unless the user separately asks for them.

## Planned file structure

- Create `src/fashion/data_audit.py` — shared CSV repair, hierarchy, and image-hash primitives.
- Replace `src/fashion/eda.py` — selected-population construction, metadata metrics, drift, and orchestration.
- Create `src/fashion/eda_images.py` — deterministic sampling, image measurements, and duplicate candidates.
- Create `src/fashion/eda_plots.py` — detailed plots, review grids, and the four-panel report figure.
- Create `tests/test_data_audit.py` — shared primitive contracts.
- Create `tests/test_eda.py` — population and metadata-analysis contracts.
- Create `tests/test_eda_images.py` — image-analysis contracts.
- Create `tests/test_eda_plots.py` — output and plotting contracts.
- Create `scripts/run_eda.py` — command-line execution.
- Create `scripts/verify_eda.py` — post-run reconciliation and safety checks.
- Create `notebooks/00_eda.ipynb` — narrative notebook built from an empty file.
- Modify `src/fashion/dataset_comparison.py` — import shared primitives from `data_audit`.
- Modify `src/fashion/config.py` — add EDA output paths and remove stale compatibility wording.
- Modify `pyproject.toml` — declare pytest as a development dependency.
- Modify `README.md` — document the new single EDA and execution commands.
- Delete `docs/superpowers/specs/2026-08-16-eda-design.md`.
- Delete `docs/superpowers/plans/2026-08-16-eda.md`.
- Delete `results/figures/dataset_overview.png`.

---

### Task 1: Protect shared audit behaviour and remove the old implementation

**Files:**
- Create: `tests/test_data_audit.py`
- Create: `src/fashion/data_audit.py`
- Modify: `src/fashion/dataset_comparison.py:30`
- Modify: `pyproject.toml`
- Replace: `src/fashion/eda.py`
- Delete: `docs/superpowers/specs/2026-08-16-eda-design.md`
- Delete: `docs/superpowers/plans/2026-08-16-eda.md`
- Delete: `results/figures/dataset_overview.png`

**Interfaces:**
- Produces: `audit_csv(path: Path) -> tuple[pd.DataFrame, CsvAudit]`
- Produces: `hierarchy_conflicts(frame: pd.DataFrame) -> pd.DataFrame`
- Produces: `dhash(image: Image.Image, hash_size: int = 8) -> str`
- Produces: `hamming_distance(left: str, right: str) -> int`
- Preserves temporary compatibility alias `_dhash = dhash` for dataset comparison.

- [ ] **Step 1: Write failing CSV-semantics tests**

Create a temporary CSV containing a literal `"NA"`, a blank usage, and a product name
spilled over two trailing columns. Assert:

```python
frame, audit = audit_csv(path)
assert frame.loc[0, "usage"] == "NA"
assert frame.loc[1, "usage"] == ""
assert frame.loc[0, "productDisplayName"] == "Alpha, Beta, Gamma"
assert audit.literal_na_usage_count == 1
assert audit.blank_counts["usage"] == 1
```

- [ ] **Step 2: Write failing hierarchy and hash tests**

Use a frame where one `articleType` maps to two `subCategory` values and two 8×8 images
that differ by one hash bit. Assert conflict IDs are retained and Hamming distance is exact.

- [ ] **Step 3: Run tests and confirm RED**

First add `dev = ["pytest"]` under `[project.optional-dependencies]` and install it:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
```

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_data_audit.py -q
```

Expected: collection fails because `fashion.data_audit` does not exist.

- [ ] **Step 4: Implement the shared deep module**

Implement an immutable `CsvAudit` carrying physical columns, phantom columns, row count,
duplicate IDs, blank counts, and literal-NA count. Repair comma-spilled product names,
normalize IDs to integer, and fail with a clear `ValueError` when required columns are
missing. Keep the four notebook-independent primitives in this module.

- [ ] **Step 5: Move the dataset-comparison import**

Replace:

```python
from fashion.eda import _dhash, audit_csv, hierarchy_conflicts, hamming_distance
```

with:

```python
from fashion.data_audit import _dhash, audit_csv, hierarchy_conflicts, hamming_distance
```

- [ ] **Step 6: Run focused and comparison verification**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_data_audit.py -q
PYTHONPATH=src ./.venv/bin/python scripts/verify_dataset_comparison.py
```

Expected: all focused tests pass and the existing comparison summary verifies.

- [ ] **Step 7: Remove obsolete EDA artifacts**

Delete the old EDA plan, old EDA design, and `results/figures/dataset_overview.png`.
Replace `src/fashion/eda.py` with an empty module docstring only after the shared imports
are verified.

---

### Task 2: Build the selected population and metadata analysis

**Files:**
- Create: `tests/test_eda.py`
- Replace: `src/fashion/eda.py`
- Modify: `src/fashion/config.py`

**Interfaces:**
- Produces: `EdaPaths` and `PopulationAudit` dataclasses.
- Produces: `build_population(paths: EdaPaths) -> tuple[pd.DataFrame, PopulationAudit]`
- Produces: `distribution_table(frame, column) -> pd.DataFrame`
- Produces: `skew_table(frame, columns=TARGET_COLUMNS) -> pd.DataFrame`
- Produces: `support_band_table(frame, column="articleType") -> pd.DataFrame`
- Produces: `cramers_v(left: pd.Series, right: pd.Series) -> float`
- Produces: `association_matrix(frame, columns) -> pd.DataFrame`
- Produces: `drift_table(frame, group_column, category_column) -> pd.DataFrame`
- Produces: `product_name_audit(frame) -> dict[str, object]`

- [ ] **Step 1: Write failing population tests**

Build synthetic teacher, original, test, and image directories. Assert that
`build_population`:

```python
assert audit.source_train_ids == 4
assert audit.usable_products == 3
assert audit.missing_original_image_ids == (4,)
assert set(frame["id"]) == {1, 2, 3}
assert not set(frame["id"]).intersection({90, 91})
assert {"original_image_path", "lowres_image_path"}.issubset(frame.columns)
```

Also assert a test ID appearing in a training input raises `ValueError`.

- [ ] **Step 2: Write failing metric tests**

Use small hand-calculated series to assert:

```python
assert distribution_table(frame, "usage")["count"].sum() == len(frame)
assert skew.loc["balanced", "normalized_entropy"] == pytest.approx(1.0)
assert skew.loc["constant", "gini_impurity"] == pytest.approx(0.0)
assert list(support["band"]) == ["1", "2", "3–4", "5–9", "10+"]
assert cramers_v(perfect_left, perfect_right) == pytest.approx(1.0)
```

Assert empty inputs return documented zeros or empty tables instead of NaNs.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_eda.py -q
```

Expected: failures for the missing interfaces.

- [ ] **Step 4: Implement selected-population construction**

Read teacher metadata first. Use only its IDs to filter original metadata before copying
label columns. Join one original and one low-resolution path per ID. Audit every mismatch,
drop only products lacking either required image view, and sort by ID.

- [ ] **Step 5: Implement pure metadata measures**

Implement counts/shares/blanks, imbalance ratio, normalized entropy, effective classes,
Gini impurity, top-1/top-5 shares, support bands, Cramér's V, row-normalized crosstabs,
year/ID-bin total-variation drift, hierarchy conflicts, and the basic product-name audit.

- [ ] **Step 6: Add EDA output configuration**

Add:

```python
EDA_OUTPUT_DIR = FIGURE_DIR / "eda"
EDA_SAMPLE_SIZE = 2048
```

Remove the stale “existing official-subset EDA” compatibility comment while keeping aliases
needed by other current callers.

- [ ] **Step 7: Run focused tests**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_eda.py -q
```

Expected: all tests pass.

---

### Task 3: Add deterministic image and duplicate analysis

**Files:**
- Create: `tests/test_eda_images.py`
- Create: `src/fashion/eda_images.py`

**Interfaces:**
- Produces: `stratified_sample(frame, column, limit, seed) -> pd.DataFrame`
- Produces: `measure_image(path: Path) -> dict[str, object]`
- Produces: `measure_images(frame, path_column) -> pd.DataFrame`
- Produces: `exact_duplicate_groups(measurements) -> pd.DataFrame`
- Produces: `near_duplicate_candidates(measurements, max_distance=6) -> pd.DataFrame`
- Produces: `paired_image_comparison(low, high) -> pd.DataFrame`

- [ ] **Step 1: Write failing sampling tests**

Use an imbalanced frame with rare classes. Assert the same seed returns the same sorted IDs,
the limit is respected, and each observed class appears when the limit permits.

- [ ] **Step 2: Write failing image-measurement tests**

Create RGB, grayscale, flat, and edge-heavy temporary images. Assert exact dimensions,
mode, SHA-256, finite brightness/contrast/colourfulness/saturation/edge-sharpness, and a
larger sharpness score for the edge-heavy image.

- [ ] **Step 3: Write failing duplicate tests**

Create two byte-identical images, one visually similar image, and one unrelated image.
Assert exact groups contain only identical content and near candidates are deterministically
sorted by distance then IDs.

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_eda_images.py -q
```

Expected: collection fails because `fashion.eda_images` does not exist.

- [ ] **Step 5: Implement deterministic image analysis**

Use Pillow and NumPy only. Convert to RGB for colour measures and grayscale for brightness,
contrast, and adjacent-edge variance. Catch unreadable files into an `error` column. Use
full hashes for exact groups and sampled dHash integers for near candidates.

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_eda_images.py -q
```

Expected: all tests pass.

---

### Task 4: Build evidence outputs and plots

**Files:**
- Create: `tests/test_eda_plots.py`
- Create: `src/fashion/eda_plots.py`
- Modify: `src/fashion/eda.py`
- Create: `scripts/run_eda.py`
- Create: `scripts/verify_eda.py`

**Interfaces:**
- Produces: `run_eda(paths=None, output_dir=EDA_OUTPUT_DIR, refresh=False) -> EdaResult`
- Produces: `load_eda_summary(output_dir=EDA_OUTPUT_DIR) -> dict[str, object]`
- Writes: `summary.json`, detailed CSVs, image-measurement caches, review grids, detailed
  plots, and `eda-report-summary.png`.

- [ ] **Step 1: Write failing plotting contract tests**

Using synthetic analysis tables, call each plot function with a temporary directory. Assert
the combined report figure is non-empty, has four axes, and all plot functions return their
`Figure` without calling `plt.show()`.

- [ ] **Step 2: Write failing orchestration tests**

Use the synthetic population fixture and tiny images. Assert `run_eda` writes the expected
JSON/CSV/PNG names, JSON contains provenance and population totals, and rerunning with
unchanged provenance reuses image caches.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg ./.venv/bin/python -m pytest tests/test_eda_plots.py -q
```

Expected: failures for missing plotting and orchestration interfaces.

- [ ] **Step 4: Implement detailed plots and review grids**

Create target-distribution, full long-tail/log/cumulative, support-band, categorical
association, row-normalized relationship, drift, image-property, and duplicate-summary
plots. Generate deterministic grids for common, rare, unusual, grayscale, exact-duplicate,
and near-duplicate examples.

- [ ] **Step 5: Implement the four-panel report figure**

The axes are, in order:

1. normalized target skew;
2. full `articleType` support on log scale;
3. Cramér's V categorical association matrix; and
4. low/high image-quality comparison.

- [ ] **Step 6: Implement orchestration, caching, and serialization**

Cache image measurements with a provenance object containing input path, file count,
latest modification time, seed, sample limit, and metric version. Reject stale caches.
Serialize NumPy/pandas values safely to JSON and keep all detailed tables in CSV.

- [ ] **Step 7: Add execution and verification scripts**

`run_eda.py` accepts `--refresh`. `verify_eda.py` asserts source/usable totals, no test-ID
overlap, required output files, and the unchanged existence/content hash of
`data/processed/splits.csv` when that file exists.

- [ ] **Step 8: Run focused and full unit tests**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg ./.venv/bin/python -m pytest tests -q
```

Expected: all tests pass.

---

### Task 5: Create the narrative notebook and documentation

**Files:**
- Create: `notebooks/00_eda.ipynb`
- Modify: `README.md`

**Interfaces:**
- Notebook consumes only public functions from `fashion.eda`.
- Notebook has sections for population, metadata, skew, relationships, drift, images,
  duplicates, visual review, findings, likely model effects, and open decisions.

- [ ] **Step 1: Create a new empty notebook**

Create `notebooks/00_eda.ipynb` from scratch with a Python kernel and no copied cells from
the deleted notebooks.

- [ ] **Step 2: Add short narrative and execution cells**

The notebook calls `run_eda(refresh=bool(int(os.getenv("MLA2_RECOMPUTE_EDA", "0"))))`,
loads generated tables, displays detailed plots, and states that literal `"NA"` is distinct
from blank. All numbers come from returned results or generated files.

- [ ] **Step 3: End with evidence, not policy**

The final cells list factual findings, likely model effects, and open questions. They do not
choose rare-class treatment, cleaning actions, split construction, or model architecture.

- [ ] **Step 4: Update README**

List only the new `00_eda.ipynb` and the separate dataset-comparison notebook. Document:

```bash
PYTHONPATH=src MPLBACKEND=Agg ./.venv/bin/python scripts/run_eda.py
PYTHONPATH=src MPLBACKEND=Agg ./.venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/00_eda.ipynb
PYTHONPATH=src ./.venv/bin/python scripts/verify_eda.py
```

- [ ] **Step 5: Execute the real EDA**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg ./.venv/bin/python scripts/run_eda.py --refresh
```

Expected: 38,617 source IDs, five excluded IDs, 38,612 usable products, and generated
evidence under `results/figures/eda/`.

- [ ] **Step 6: Execute the notebook from a clean kernel**

Run:

```bash
PYTHONPATH=src MPLBACKEND=Agg ./.venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/00_eda.ipynb
```

Expected: successful execution with no manual state.

- [ ] **Step 7: Run final verification**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/verify_eda.py
PYTHONPATH=src ./.venv/bin/python scripts/verify_dataset_comparison.py
PYTHONPATH=src MPLBACKEND=Agg ./.venv/bin/python -m pytest tests -q
git diff --check
```

Expected: every command succeeds, comparison remains intact, and no processed split is
created or modified.
