# Phase 1 Trusted Training Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable 38,612-product training manifest, reviewed label-action table, and the project's only leakage-safe split without exposing official test labels.

**Architecture:** A quarantine module establishes the allowed IDs before target labels load. Focused population, review, manifest, split, and validation modules then produce stable CSV contracts. One command-line pipeline and one narrative notebook use these modules; every major task ends at a user approval gate.

**Tech Stack:** Python 3.11+, pandas, NumPy, Pillow, scikit-learn, Jupyter, pytest

## Global Constraints

- Work on branch `feat/phase-1-trusted-training-data`.
- Run Python through `./.venv/bin/python`.
- Read the official test CSV with `usecols=["id"]` before reading any training target.
- Never load target columns from `data/raw/original/styles.csv`.
- Use labels only from `data/raw/teacher/train/styles_train.csv`.
- Preserve literal `usage == "NA"` until the review policy masks it.
- Treat sharp and blurry files as two views of one product.
- Keep all raw files unchanged.
- Use seed `2753`.
- Reserve IDs `46_919` through `51_999` for the catalogue holdout.
- Add IDs `41_236`, `46_732`, and `46_833` to that holdout because their exact-copy groups cross its boundary.
- Keep every exact-copy group and every confirmed near-copy group in one partition.
- Preserve natural class shares in validation and catalogue holdout data.
- Do not call `train_test_split`; `fashion.data.splits` is the only code allowed to assign partitions.
- `data/processed/splits.csv` is the only partition source used by later work.
- Keep reusable logic out of notebooks.
- Write stable CSVs with fixed columns, integer-ID order, UTF-8, `\n`, and no runtime timestamps.
- Stop after each task and wait for user approval.
- Do not create Git commits unless the user separately asks for one.

## Planned File Structure

- Create `src/fashion/data/__init__.py` — public Phase 1 data interfaces.
- Create `src/fashion/data/quarantine.py` — ID-only official population boundary.
- Create `src/fashion/data/population.py` — allowed labels, paired images, and reconciliation.
- Create `src/fashion/data/review.py` — candidate generation and reviewed-action application.
- Create `src/fashion/data/manifest.py` — one-row-per-product training contract.
- Create `src/fashion/data/splits.py` — duplicate grouping and deterministic partition assignment.
- Create `src/fashion/data/validation.py` — cross-artifact safety and rebuild checks.
- Create `src/fashion/data/pipeline.py` — six-stage orchestration.
- Create `tests/data_helpers.py` — miniature raw-data fixture builders.
- Create `tests/test_quarantine.py` — no-label quarantine tests.
- Create `tests/test_population.py` — population and image reconciliation tests.
- Create `tests/test_review.py` — review schema, corrections, masks, and exclusions.
- Create `tests/test_manifest.py` — manifest schema and stable-byte tests.
- Create `tests/test_splits.py` — holdout, grouping, balance, and determinism.
- Create `tests/test_phase1_pipeline.py` — end-to-end miniature build and verification.
- Create `scripts/build_phase1_data.py` — staged and full artifact builds.
- Create `scripts/verify_phase1_data.py` — real-data safety verification.
- Create `notebooks/01_preprocessing.ipynb` — short narrative over generated outputs.
- Modify `src/fashion/config.py` — Phase 1 paths and holdout constants.
- Modify `src/fashion/eda.py` — delegate population construction to the safe shared loader.
- Modify `tests/test_eda.py` — preserve EDA behaviour through the shared loader.
- Modify `notebooks/00_eda.ipynb` — drop the removed original-CSV audit bullet from the raw-data audit cell.
- Regenerate `results/figures/eda/*` — one clean EDA run over teacher-training labels.
- Modify `pyproject.toml` — add scikit-learn.
- Modify `.gitignore` — track only `label_review.csv` and `splits.csv` under processed data.
- Modify `README.md` — document Phase 1 build and verification commands.
- Modify `docs/eda-problem-review.md` — mark actions complete only after Task 6 passes.
- Modify `docs/assignment-roadmap.md` — mark Phase 1 complete only after Task 6 approval.

---

### Task 1: Quarantine Official Test IDs First

**Files:**
- Create: `src/fashion/data/__init__.py`
- Create: `src/fashion/data/quarantine.py`
- Create: `tests/data_helpers.py`
- Create: `tests/test_quarantine.py`
- Modify: `src/fashion/config.py:1-30`
- Modify: `.gitignore:10-12`

**Interfaces:**
- Produces: `QuarantinePaths`
- Produces: `QuarantineAudit`
- Produces: `read_id_column(path: Path, source: str) -> pd.Index`
- Produces: `establish_quarantine(paths: QuarantinePaths = QuarantinePaths()) -> QuarantineAudit`
- `QuarantineAudit` exposes sorted immutable `test_ids`, `train_ids`, and `original_ids`.
- `QuarantinePaths` includes `expected_test_count`, `expected_train_count`, and
  `expected_original_count`; tests set fixture counts while defaults are 5,829, 38,617,
  and 44,446.

- [ ] **Step 1: Add Phase 1 path and boundary constants**

Add these constants to `fashion.config`:

```python
LABEL_REVIEW_CSV = PROCESSED_DATA_DIR / "label_review.csv"
TRAIN_MANIFEST_CSV = PROCESSED_DATA_DIR / "train_manifest.csv"
SPLITS_CSV = PROCESSED_DATA_DIR / "splits.csv"
PHASE1_OUTPUT_DIR = FIGURE_DIR / "phase1"
CATALOGUE_HOLDOUT_MIN_ID = 46_919
CATALOGUE_HOLDOUT_MAX_ID = 51_999
CATALOGUE_HOLDOUT_EXTRA_IDS = (41_236, 46_732, 46_833)
```

- [ ] **Step 2: Set the root-only data ignore rule**

Change `.gitignore` from `data/` to `/data/` immediately. This keeps root raw data ignored
while allowing `src/fashion/data/` source files to be tracked. Task 5 still adds the later
root-data allow-list for `label_review.csv` and `splits.csv`.

- [ ] **Step 3: Write failing ID-only reader tests**

Create miniature teacher-test, teacher-train, and original CSVs. Patch
`fashion.data.quarantine.pd.read_csv` to record every call and assert:

```python
audit = establish_quarantine(paths)

assert audit.test_ids == (90, 91)
assert audit.train_ids == (1, 2, 3)
assert audit.original_ids == (1, 2, 3, 90, 91)
assert recorded_calls[0].path == paths.teacher_test_csv
assert recorded_calls[0].usecols == ["id"]
assert all(call.usecols == ["id"] for call in recorded_calls)
```

Also assert duplicate IDs, non-integer IDs, a train/test overlap, a count mismatch, and an
original-population mismatch raise clear `ValueError` messages.

- [ ] **Step 4: Run the focused test and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_quarantine.py -q
```

Expected: collection fails because `fashion.data.quarantine` does not exist.

- [ ] **Step 5: Implement the ID-only quarantine**

Use this ordering inside `establish_quarantine`:

```python
test_ids = read_id_column(paths.teacher_test_csv, "Official test metadata")
train_ids = read_id_column(paths.teacher_train_csv, "Teacher training metadata")
original_ids = read_id_column(paths.original_csv, "Original metadata")

overlap = test_ids.intersection(train_ids)
if not overlap.empty:
    raise ValueError(f"Official train/test ID overlap: {overlap.tolist()[:10]}")
if len(test_ids) != paths.expected_test_count:
    raise ValueError(
        f"Expected {paths.expected_test_count} official test IDs, found {len(test_ids)}"
    )
if train_ids.union(test_ids).difference(original_ids).size:
    raise ValueError("Official IDs are missing from original metadata")
```

`read_id_column` must use `keep_default_na=False`, string input, explicit integer
validation, duplicate rejection, and sorted integer output. It must never accept a target
column argument.

- [ ] **Step 6: Export only the small public interface**

In `src/fashion/data/__init__.py`, export:

```python
from fashion.data.quarantine import (
    QuarantineAudit,
    QuarantinePaths,
    establish_quarantine,
)
```

- [ ] **Step 7: Run focused tests and the real quarantine check**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_quarantine.py -q
PYTHONPATH=src ./.venv/bin/python -c \
  "from fashion.data.quarantine import establish_quarantine; a=establish_quarantine(); print(len(a.test_ids), len(a.train_ids), len(a.original_ids))"
```

Expected final line:

```text
5829 38617 44446
```

- [ ] **Step 8: Stop for Checkpoint 1**

Report the three counts and prove that every metadata read requested only `id`. Wait for
user approval before Task 2.

---

### Task 2: Reconcile the 38,612 Allowed Image Products

**Files:**
- Create: `src/fashion/data/population.py`
- Create: `tests/test_population.py`
- Modify: `src/fashion/data/__init__.py`
- Modify: `src/fashion/eda.py:55-211`
- Modify: `tests/test_eda.py:41-233`
- Modify: `tests/test_eda_plots.py` — fixture quarantine counts and ID-only original metadata
- Modify: `notebooks/00_eda.ipynb` — raw-data audit cell only
- Regenerate: `results/figures/eda/*` — one clean EDA run

**Interfaces:**
- Consumes: `establish_quarantine(...) -> QuarantineAudit`
- Produces: `PopulationPaths`
- Produces: `PopulationAudit`
- Produces: `inventory_images(directory: Path) -> ImageInventory`
- Produces: `build_allowed_population(paths: PopulationPaths = PopulationPaths()) -> tuple[pd.DataFrame, PopulationAudit]`
- Preserves: `fashion.eda.EdaPaths`, `fashion.eda.PopulationAudit`, and
  `fashion.eda.build_population` as aliases to the shared implementation.

- [ ] **Step 1: Write failing population tests**

Use the fixture helpers to create four allowed metadata rows, two quarantined rows, and
three complete image pairs. Assert:

```python
population, audit = build_allowed_population(paths)

assert population["id"].tolist() == [1, 2, 3]
assert audit.source_train_ids == 4
assert audit.usable_products == 3
assert audit.missing_both_image_ids == (4,)
assert not set(population["id"]).intersection({90, 91})
assert population.loc[1, "usage"] == "NA"
assert population.loc[2, "usage"] == ""
assert {"original_image_path", "lowres_image_path"}.issubset(population)
```

Patch `audit_csv` and assert it is not called until `establish_quarantine` has succeeded.
Patch `pd.read_csv` and assert original metadata is read only through the ID-only
quarantine path.

- [ ] **Step 2: Write failing image-reconciliation tests**

Cover invalid filenames, two files with the same numeric stem, a missing sharp image, a
missing blurry image, and an unreadable image. Assert each problem is retained in
`PopulationAudit`, and unreadable products do not enter the usable population.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_population.py -q
```

Expected: collection fails because `fashion.data.population` does not exist.

- [ ] **Step 4: Implement allowed-label loading and image inventory**

The implementation must establish quarantine first, then call:

```python
teacher, teacher_csv_audit = audit_csv(paths.teacher_train_csv)
teacher = teacher.loc[teacher["id"].isin(quarantine.train_ids)].copy()
```

Do not call `audit_csv` on original metadata. Build sharp and blurry inventories from
filenames, load each candidate image with Pillow, and record decode errors. Add paths only
after all ID checks pass.

- [ ] **Step 5: Build and return the sorted usable population**

Use one row per ID:

```python
population = (
    teacher.drop_duplicates("id", keep=False)
    .assign(
        original_image_path=lambda frame: frame["id"].map(original_inventory.paths),
        lowres_image_path=lambda frame: frame["id"].map(lowres_inventory.paths),
    )
    .dropna(subset=["original_image_path", "lowres_image_path"])
    .loc[lambda frame: ~frame["id"].isin(unreadable_ids)]
    .sort_values("id", ignore_index=True)
)
```

Fail instead of silently choosing a row when teacher IDs or image stems are duplicated.

- [ ] **Step 6: Move EDA population loading to the shared safe module**

Replace the old EDA population classes and loader with imports:

```python
from fashion.data.population import (
    PopulationAudit,
    PopulationPaths as EdaPaths,
    build_allowed_population as build_population,
)
```

Update EDA tests so fixture labels come from teacher-training metadata. Keep all EDA metric,
plot, and evidence interfaces unchanged.

- [ ] **Step 7: Restore notebook compatibility and regenerate the evidence**

The shared loader never audits original metadata, so `data-audit.json` no longer contains an
`original_csv_audit` key. Update only the raw-data audit cell of `notebooks/00_eda.ipynb` so
it reads `teacher_csv_audit` and the population, image, hierarchy, and product-name facts,
and delete the original-CSV bullet. Every other cell, and
`notebooks/01_preprocessing.ipynb` in Task 6, stay unchanged.

Notebook edits must keep a valid document. Check the saved file before executing it:

```bash
./.venv/bin/python -c \
  "import nbformat; nbformat.validate(nbformat.read('notebooks/00_eda.ipynb', as_version=4))"
```

Repair only missing generated-output schema fields, never analysis content. Then run the
whole notebook so its source and saved outputs agree, and regenerate the evidence bundle
from teacher-training labels:

```bash
MPLBACKEND=Agg ./.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/00_eda.ipynb
```

Report every changed evidence value, including the one `subCategory` cell where teacher and
original metadata disagree. Expected evidence differences are documented, never suppressed
by changing a policy decision.

- [ ] **Step 8: Run focused tests and all EDA regressions**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest \
  tests/test_population.py tests/test_eda.py tests/test_eda_images.py tests/test_eda_plots.py -q
PYTHONPATH=src ./.venv/bin/python scripts/verify_eda.py
```

Expected: all tests pass and the regenerated EDA evidence verifies.

- [ ] **Step 9: Run the real population count check**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -c \
  "from fashion.data.population import build_allowed_population; f,a=build_allowed_population(); print(a.source_train_ids, a.missing_both_image_ids, a.usable_products)"
```

Expected:

```text
38617 (12347, 39401, 39403, 39410, 39425) 38612
```

- [ ] **Step 10: Stop for Checkpoint 2**

Report source, missing-image, readable-pair, invalid-filename, and duplicate-stem counts,
plus the regenerated evidence differences. Wait for user approval before Task 3.

---

### Task 3: Create and Complete the Review Table

**Files:**
- Create: `src/fashion/data/review.py`
- Create: `tests/test_review.py`
- Create: `scripts/build_phase1_data.py`
- Modify: `src/fashion/data/__init__.py`
- Generate: `results/figures/phase1/review-candidates.csv`
- Generate: `results/figures/phase1/review/index.html`
- Generate and review: `data/processed/label_review.csv`

**Interfaces:**
- Consumes: `build_allowed_population(...)`
- Produces: `REVIEW_COLUMNS`
- Produces: `build_review_candidates(population: pd.DataFrame, output_dir: Path) -> pd.DataFrame`
- Produces: `seed_policy_actions(population: pd.DataFrame) -> pd.DataFrame`
- Produces: `validate_review_table(review: pd.DataFrame, population: pd.DataFrame, candidates: pd.DataFrame) -> None`
- Produces: `apply_review_table(population: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write failing review-schema tests**

Require this exact final schema:

```python
REVIEW_COLUMNS = (
    "id",
    "field",
    "action",
    "old_value",
    "new_value",
    "reason",
    "evidence",
    "review_status",
)
```

Assert only `keep`, `correct`, `mask`, and `exclude` actions are accepted. Reject unknown
IDs, wrong old values, duplicate target actions for the same ID/field, corrections without
a new value, masks with a new value, exclusions whose field is not `product`, and any
`review_status` other than `approved`. Multiple `duplicate_group` rows are allowed only
when each row has a distinct, valid `near_pair:<left>|<right>;dhash:<distance>` evidence
value.

- [ ] **Step 2: Write failing policy-action tests**

Use a fixture with blank season, blank usage, literal `"NA"` usage, ID `45824`, and ID
`38223`. Assert policy rows include:

```python
assert action_for(actions, 45824, "articleType") == ("correct", "Ipad", "Flats")
assert action_for(actions, 38223, "articleType") == (
    "correct",
    "Ties and Cufflinks",
    "Sunglasses",
)
assert action_for(actions, blank_season_id, "season").action == "mask"
assert action_for(actions, blank_usage_id, "usage").action == "mask"
assert action_for(actions, literal_na_id, "usage").action == "mask"
```

Also require the matching hierarchy corrections for ID `45824`:
`masterCategory -> Footwear` and `subCategory -> Shoes`.

- [ ] **Step 3: Write failing application tests**

Apply correction, mask, keep, and exclusion rows. Assert corrections replace only the named
field, a target mask changes only that target's validity flag, and product exclusion changes
only `is_in_scope`.

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_review.py -q
```

Expected: collection fails because `fashion.data.review` does not exist.

- [ ] **Step 5: Implement deterministic candidate generation**

Generate candidates for:

- all product-name gender contradictions;
- uncommon hierarchy rows from `hierarchy_conflicts`;
- every exact low-resolution group with a target conflict;
- the confirmed blank image at ID `44998`;
- likely non-fashion rows from Sporting Goods, Home, and other EDA scope warnings; and
- all 343 low-resolution grayscale products.

Assign a stable candidate key:

```python
candidate_key = f"{issue_type}:{field}:{min(group_ids)}"
```

Sort by `issue_type`, `field`, and integer ID. Candidate generation may suggest `keep` or
`mask`, but it must not create an approved correction from a text rule.

- [ ] **Step 6: Build a static visual review page**

Write plain HTML with one section per candidate group. Show current metadata, product name,
sharp image, blurry image, related IDs, and evidence source. Use repository-relative links
and no JavaScript dependency. The page is evidence only; `label_review.csv` remains the
source of approved actions.

- [ ] **Step 7: Implement review validation and application**

Start each output frame with:

```python
result = population.copy()
result["is_in_scope"] = True
for target in TARGET_COLUMNS:
    result[f"{target}_valid"] = result[target].astype("string").str.strip().ne("")
```

Then apply rows in sorted ID/field order. Treat literal usage `"NA"` as invalid only through
its explicit approved mask row.

- [ ] **Step 8: Add the first staged CLI command**

Support:

```bash
./.venv/bin/python scripts/build_phase1_data.py --through review
```

It must build the quarantine and population in memory, write candidate evidence, seed
approved policy actions, and stop with a non-zero exit plus a count of unresolved visual
candidates until their decisions are recorded.

- [ ] **Step 9: Run tests and generate the real review pack**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_review.py -q
./.venv/bin/python scripts/build_phase1_data.py --through review
```

Expected before manual decisions: focused tests pass; the command reports unresolved
candidate groups and does not claim the review table is final.

- [ ] **Step 10: Complete targeted visual review**

Review related gender products together, uncommon hierarchy rows, all 22 conflicting exact
groups, ID `44998`, suspected non-fashion products, and all 343 grayscale pairs. Record one
explicit approved action for each required candidate. Exclude ID `44998` as visually
unusable. Use `mask` when a target remains uncertain. Do not infer a correction from
product-name words alone.

- [ ] **Step 11: Re-run review validation**

Run:

```bash
./.venv/bin/python scripts/build_phase1_data.py --through review
```

Expected: zero pending required candidates, 20 season masks, 72 usage masks before any
additional manual masks, both approved article-type corrections, and a printed count for
every manual keep, correction, mask, and exclusion.

- [ ] **Step 12: Stop for Checkpoint 3**

Show the user the action counts and every correction, mask, and exclusion. Wait for explicit
approval of `data/processed/label_review.csv` before Task 4.

---

### Task 4: Build the One-Row-Per-Product Manifest

**Files:**
- Create: `src/fashion/data/manifest.py`
- Create: `tests/test_manifest.py`
- Modify: `src/fashion/data/__init__.py`
- Modify: `scripts/build_phase1_data.py`
- Generate: `data/processed/train_manifest.csv`

**Interfaces:**
- Consumes: reviewed population from `apply_review_table(...)`
- Produces: `MANIFEST_COLUMNS`
- Produces: `build_exact_group_map(population: pd.DataFrame) -> pd.Series`
- Produces: `build_train_manifest(population: pd.DataFrame, review: pd.DataFrame, root: Path) -> pd.DataFrame`
- Produces: `write_stable_csv(frame: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> str`
- Returns the written file's SHA-256 digest from `write_stable_csv`.

- [ ] **Step 1: Write failing manifest-contract tests**

Require one sorted row per usable ID and these column groups:

```python
identity = ["id"]
metadata = [
    "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "year", "usage", "productDisplayName",
]
paths = ["original_image_path", "lowres_image_path"]
control = [
    "exact_group_id", "is_in_scope",
    "articleType_valid", "season_valid", "gender_valid", "usage_valid",
]
```

Assert no test ID appears, both paths are repository-relative POSIX strings, IDs and paths
resolve to the expected files, and exact copies receive the same `exact_group_id`.

- [ ] **Step 2: Write failing target-specific mask tests**

Use one row with invalid usage and valid other labels:

```python
row = manifest.set_index("id").loc[7]
assert bool(row["usage_valid"]) is False
assert bool(row["gender_valid"]) is True
assert bool(row["articleType_valid"]) is True
assert bool(row["season_valid"]) is True
assert bool(row["is_in_scope"]) is True
```

Also assert a reviewed product exclusion keeps an auditable manifest row with
`is_in_scope == False`.

- [ ] **Step 3: Write failing deterministic-byte tests**

Build the same fixture in two different absolute temporary directories. Assert both
manifest frames and written bytes are identical because paths are repository-relative:

```python
assert first_frame.equals(second_frame)
assert first_path.read_bytes() == second_path.read_bytes()
assert first_sha256 == second_sha256
```

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_manifest.py -q
```

Expected: collection fails because `fashion.data.manifest` does not exist.

- [ ] **Step 5: Implement exact low-resolution group IDs**

Hash low-resolution file bytes with SHA-256. Give duplicate groups
`exact:<smallest-id>` and singleton products `id:<id>`. Verify that the real data produces
636 non-singleton groups covering 1,399 IDs before reviewed exclusions.

- [ ] **Step 6: Implement manifest construction**

Apply the approved review table, convert image paths with:

```python
relative = path.resolve().relative_to(root.resolve()).as_posix()
```

Order columns exactly as the contract, sort by integer ID with a stable sort, reject
duplicate IDs, and reject absent files.

- [ ] **Step 7: Implement stable CSV writing**

Write to a sibling temporary file, use `lineterminator="\n"`, then atomically replace the
destination. Never include dataframe indexes or timestamps.

- [ ] **Step 8: Extend the staged CLI and run focused tests**

Support:

```bash
./.venv/bin/python scripts/build_phase1_data.py --through manifest
```

Then run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_manifest.py -q
./.venv/bin/python scripts/build_phase1_data.py --through manifest
```

Expected: 38,612 manifest rows before scope filtering, sorted unique IDs, two valid relative
paths per row, and printed counts for each target-valid flag.

- [ ] **Step 9: Stop for Checkpoint 4**

Show row count, excluded count, target-valid counts, article-type vocabulary size, duplicate
group counts, and manifest SHA-256. Wait for user approval before Task 5.

---

### Task 5: Build the Grouped Development Split and Catalogue Holdout

**Files:**
- Create: `src/fashion/data/splits.py`
- Create: `tests/test_splits.py`
- Modify: `src/fashion/data/__init__.py`
- Modify: `scripts/build_phase1_data.py`
- Modify: `pyproject.toml:10-20`
- Modify: `.gitignore:10-12`
- Generate: `results/figures/phase1/provisional-split.csv`
- Generate: `results/figures/phase1/cross-boundary-near-review.csv`
- Generate: `results/figures/phase1/split-distributions.csv`
- Generate: `data/processed/splits.csv`

**Interfaces:**
- Consumes: `train_manifest.csv`, `label_review.csv`, and sampled EDA near candidates.
- Produces: `SPLIT_COLUMNS = ("id", "partition", "group_id")`
- Produces: `merge_groups(ids: Iterable[int], pairs: Iterable[tuple[int, int]]) -> dict[int, str]`
- Produces: `catalogue_holdout_ids(manifest: pd.DataFrame, group_map: Mapping[int, str]) -> set[int]`
- Produces: `candidate_development_folds(manifest: pd.DataFrame, group_map: Mapping[int, str]) -> list[SplitCandidate]`
- Produces: `select_split(candidates: Sequence[SplitCandidate], manifest: pd.DataFrame) -> SplitCandidate`
- Produces: `reviewed_near_pairs(review: pd.DataFrame) -> tuple[tuple[int, int], ...]`
- Produces: `build_splits(manifest: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Add and install scikit-learn**

Add `"scikit-learn"` to project dependencies, then run:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
```

- [ ] **Step 2: Write failing group and holdout tests**

Use exact-copy groups `(46833, 50194, 50195)`, `(41236, 47151)`, and
`(46732, 49843)`. Assert:

```python
holdout = catalogue_holdout_ids(manifest, exact_group_map)
assert {50194, 50195, 47151, 49843}.issubset(holdout)
assert {41236, 46732, 46833}.issubset(holdout)
```

Also assert transitive pairs `(1, 2)` and `(2, 3)` create one group whose stable ID uses
the smallest member.

- [ ] **Step 3: Write failing partition-contract tests**

Assert every manifest ID appears exactly once. Out-of-scope products receive `excluded`.
Every in-scope ID receives `train`, `validation`, or `catalogue_holdout`. No group crosses
partitions, and no official test ID is accepted as input.

- [ ] **Step 4: Write failing rare-class and balance-selection tests**

Build a grouped fixture where candidate folds have different validation size and target
distribution errors. Require:

```python
assert selected.validation_share == pytest.approx(0.20, abs=0.03)
assert selected.score == min(candidate.score for candidate in candidates)
assert selected.fold_number == min(
    candidate.fold_number
    for candidate in candidates
    if candidate.score == selected.score
)
```

Calculate each candidate score as:

```python
score = abs(validation_rows / development_rows - 0.20)
score += sum(
    total_variation(
        labelled_development[target],
        labelled_validation[target],
    )
    for target in TARGET_COLUMNS
)
```

Masked labels are omitted only for their target.

- [ ] **Step 5: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_splits.py -q
```

Expected: collection fails because `fashion.data.splits` does not exist.

- [ ] **Step 6: Implement deterministic union-find grouping**

Start with manifest exact groups. Merge approved near-copy pairs transitively. Use the
smallest integer member as `group:<id>`. Assign every singleton `group:<id>` too, so later
checks never need null group values.

- [ ] **Step 7: Implement forced holdout and excluded assignments**

Assign `excluded` first for `is_in_scope == False`. For each remaining group, assign
`catalogue_holdout` if any member is inside `46_919..51_999` or is one of the three approved
extra IDs. Do not move a high-range member back into development.

- [ ] **Step 8: Implement five grouped development candidates**

Use:

```python
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED,
)
```

Pass cleaned `articleType` as `y` and final group IDs as `groups`. Treat each fold once as
validation and the other four as training. Select the lowest score defined in Step 4, then
the lowest fold number. Use the fixed sentinel `"__MASKED__"` as `y` only for rows whose
article-type label is masked, so every group is assigned; omit those rows from the
article-type balance score.

- [ ] **Step 9: Implement provisional near-copy boundary review**

Join `results/figures/eda/near-duplicates.csv` to the provisional partitions. Export every
cross-boundary candidate with dHash distance 0 or 1. Include sharp image paths, blurry image
paths, current labels, and provisional partitions.

Review each exported pair using sharp originals. Record only confirmed same-product or
same-photo pairs as group merges. Record every reviewed pair in `label_review.csv` with
`field == "duplicate_group"` and evidence formatted as
`near_pair:<left>|<right>;dhash:<distance>`. Use `action == "correct"` and
`new_value == "merge"` for a confirmed merge; use `action == "keep"` and a blank
`new_value` for a rejected merge. These rows do not change target labels.
Explicitly reject `(8855, 8860)` if it appears.

- [ ] **Step 10: Rebuild after confirmed near-copy decisions**

Merge confirmed pairs, force any newly expanded holdout group into the holdout, regenerate
the five development candidates, and select again from seed `2753`. Fail if any reviewed
candidate remains unresolved.

- [ ] **Step 11: Make only the two versioned processed files trackable**

Replace the broad `data/` ignore with:

```gitignore
data/*
!data/processed/
data/processed/*
!data/processed/label_review.csv
!data/processed/splits.csv
```

Confirm raw data and `train_manifest.csv` remain ignored.

- [ ] **Step 12: Run focused tests and build the real split**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_splits.py -q
./.venv/bin/python scripts/build_phase1_data.py --through split
git check-ignore data/raw/original/styles.csv
git check-ignore data/processed/train_manifest.csv
git check-ignore -v data/processed/splits.csv || true
```

Expected: tests pass; raw data and manifest are ignored; `splits.csv` is not ignored.

- [ ] **Step 13: Stop for Checkpoint 5**

Show train, validation, catalogue-holdout, and excluded counts; the exact validation share;
all four target-distribution distances; the three holdout-expansion IDs; confirmed
near-copy merges; and proof that no group crosses a boundary. Wait for user approval before
Task 6.

---

### Task 6: Verify the Full Build, Notebook, and Documentation

**Files:**
- Create: `src/fashion/data/validation.py`
- Create: `src/fashion/data/pipeline.py`
- Create: `tests/test_phase1_pipeline.py`
- Create: `scripts/verify_phase1_data.py`
- Create: `notebooks/01_preprocessing.ipynb`
- Modify: `scripts/build_phase1_data.py`
- Modify: `src/fashion/data/__init__.py`
- Modify: `README.md:23-127`
- Modify: `docs/eda-problem-review.md:27-37,377-412`
- Modify: `docs/assignment-roadmap.md:18-30`

**Interfaces:**
- Consumes: all Phase 1 modules and three processed artifacts.
- Produces: `validate_phase1_artifacts(...) -> ValidationReport`
- Produces: `build_phase1(paths: Phase1Paths = Phase1Paths(), through: str = "verify") -> Phase1Result`
- Produces: `verify_rebuild_exactness(paths: Phase1Paths) -> dict[str, str]`
- Produces: CLI success line:
  `OK — Phase 1 counts, quarantine, reviews, manifests, splits, and rebuild hashes reconcile`

- [ ] **Step 1: Write the failing miniature integration test**

Build a raw-data tree containing:

- separate teacher train and test IDs;
- one missing image pair;
- one blank target;
- one literal `"NA"` usage;
- one correction;
- one excluded product;
- one exact group crossing the holdout boundary; and
- one confirmed near pair.

Run the full pipeline and assert all three CSVs exist, test IDs are absent, masks are
target-specific, group members share a partition, and a second build has identical bytes.

- [ ] **Step 2: Write failing real-artifact validation tests**

Create deliberately bad artifacts and assert one clear failure per case: test-ID leakage,
duplicate manifest IDs, missing split IDs, unknown partition, cross-partition group,
unresolved review, wrong mask, absent image, and mismatched rebuild hash.

- [ ] **Step 3: Run integration tests and confirm RED**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_phase1_pipeline.py -q
```

Expected: collection fails because validation and pipeline modules do not exist.

- [ ] **Step 4: Implement one validation report**

Return an immutable report containing counts, partition shares, target-valid counts,
distribution distances, file hashes, and a tuple of failures. Provide:

```python
report.raise_for_failures()
```

that raises one `ValueError` containing every failed invariant, not just the first.

- [ ] **Step 5: Implement staged and full orchestration**

Support these exact stages:

```python
STAGES = ("quarantine", "population", "review", "manifest", "split", "verify")
```

The CLI `--through` option stops after the named stage. The default runs through `verify`.
Never write later-stage artifacts before earlier validation passes.

- [ ] **Step 6: Implement clean rebuild exactness**

Build all artifacts twice into separate temporary processed directories from the same raw
inputs and approved review decisions. Compare bytes for `label_review.csv`,
`train_manifest.csv`, and `splits.csv`. Then compare those hashes with the normal output.

- [ ] **Step 7: Implement the standalone verifier**

`scripts/verify_phase1_data.py` must load existing artifacts without rewriting them, run all
cross-artifact checks, print a compact count/hash report, and exit non-zero on any failure.

- [ ] **Step 8: Create the narrative preprocessing notebook**

Create a short notebook that:

1. states the quarantine rule;
2. imports the pipeline result and validation report;
3. displays population and review-action counts;
4. displays partition sizes and target shares;
5. explains duplicate grouping and the three extra holdout IDs; and
6. ends with the exact rebuild hashes.

The notebook must not edit review decisions or assign partitions.

- [ ] **Step 9: Update README commands**

Document:

```bash
cd /localhome/local-lintran/MLA2
./.venv/bin/python scripts/build_phase1_data.py
./.venv/bin/python scripts/verify_phase1_data.py
```

Also document the three processed artifacts, which two are tracked, and that all later work
must read `data/processed/splits.csv`.

- [ ] **Step 10: Run the complete automated test suite**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 11: Build, verify, and execute the notebook**

Run:

```bash
./.venv/bin/python scripts/build_phase1_data.py
./.venv/bin/python scripts/verify_phase1_data.py
MPLBACKEND=Agg ./.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/01_preprocessing.ipynb
```

Expected verifier output:

```text
OK — Phase 1 counts, quarantine, reviews, manifests, splits, and rebuild hashes reconcile
```

- [ ] **Step 12: Re-run existing dataset checks**

Run:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/verify_eda.py
PYTHONPATH=src ./.venv/bin/python scripts/verify_dataset_comparison.py
```

Expected: both existing evidence bundles verify after the shared population refactor.

- [ ] **Step 13: Check tracked and ignored files**

Run:

```bash
git status --short
git diff --check
git check-ignore data/raw/teacher/train/styles_train.csv
git check-ignore data/processed/train_manifest.csv
```

Expected: no raw data or manifest is staged or trackable; `label_review.csv` and
`splits.csv` appear as the only processed data files available to Git.

- [ ] **Step 14: Stop for Checkpoint 6**

Show the user the final counts, all three SHA-256 hashes, test totals, notebook result,
split-safety result, and exact `git status --short` output. Wait for approval.

- [ ] **Step 15: Update progress documents only after approval**

After the user approves Checkpoint 6 and asks to finalize Phase 1:

- mark all seven `Action` cells complete in `docs/eda-problem-review.md`;
- add a discussion-log entry with the final counts and split policy; and
- mark Phase 1 complete in `docs/assignment-roadmap.md`.

Run the full verifier once more after these documentation changes.

- [ ] **Step 16: Commit only if explicitly requested**

If the user asks for a commit, stage source, tests, docs, notebook, `label_review.csv`, and
`splits.csv`. Do not stage raw data, generated review pages, or `train_manifest.csv`. Use a
message that explains that the fixed data contract prevents test-label and duplicate-image
leakage.
