# Gender labels from product names

Created a separate dataset variant at
`data/processed/variants/gender_name_truth_v1/`.

The experiment's assumption is that the product name is the source of truth
when it contains exactly one explicit gender-class cue. This applies uniformly
to all five development folds, including their validation rows. It is not
restricted to model errors or to folds 0 and 4.

## What changed

- Exactly **350 of 32,773 development gender labels** changed, across 239 product
  families. They are exactly the 350 cases from the preceding name-conflict audit.
- **182 Men → Boys** and **156 Women → Girls** account for most changes.
  The other 12 changes follow the same rule.
- Changes by validation fold: **73, 85, 58, 50, 84** for folds 0–4.
- The returned dataset still contains all **38,612 rows**. The 5,778 holdout
  rows and 61 quarantine rows retain blank protected targets and false masks.
- Only `gender` changes. IDs, fold assignments, partitions, family groups,
  image paths, image hashes, other targets and every other canonical field
  remain identical. No images are copied, moved, added or removed.

`data/processed/splits.csv` remains the only split source. The separate
`labels.csv` deliberately contains no fold or partition assignments. The loader
reads the canonical split first and joins the verified labels by ID.

## Frozen name rule

Version: `single_explicit_gender_cue_v1`.

Match whole words, case-insensitively, after normalizing curly apostrophes:

- `boy`, `boys`, `boy's` → Boys;
- `girl`, `girls`, `girl's` → Girls;
- `men`, `mens`, `men's` → Men;
- `women`, `womens`, `women's` → Women;
- `unisex` → Unisex.

Repeated mentions of the same class count as one class cue. Embedded words
such as `Boyfriend` do not match `boy`.

Exactly one class cue makes the name label authoritative, even when the old
label differs. Names do not enter the image model as input features.

There are **31,453 single-cue names**, **1,283 names with no cue**, and **37 names
with multiple cues**. The latter two groups retain their original labels and
are listed in `unclear_names.csv`. For example, “Mr. Men Boys Tee” has two
cues; the rule does not choose a winner using the old label or a prediction.
No semantic guess is made from a photo or from words such as “Kids” alone.

The name rule creates **5 groups of identical images with different labels
(11 rows)**. Their names differ. They remain in their original folds and keep
their per-name labels; no rows are dropped or moved. See
`same_image_label_conflicts.csv`. Canonical quarantine flags still describe the
original teacher-data audit, not this variant's new label conflicts.

## Build and load

Run from the repository root:

```bash
./.venv/bin/python -m fashion.data.gender_name_truth
```

The generated dataset is local and rebuildable; the existing data-directory
ignore policy is unchanged. Building it does not need raw teacher labels or
image access. An identical rebuild is safe; a differing existing variant is
rejected rather than overwritten.

Load it explicitly:

```python
from fashion.data import get_cv_split
from fashion.data.gender_name_truth import load_gender_name_truth_variant

name_truth = load_gender_name_truth_variant()
training, validation = get_cv_split(name_truth, validation_fold=0)
```

Both returned folds use the new label convention. Ordinary `load_splits()` and
all existing training entry points still use the original labels. A new
training experiment must opt into this loader and record a new experiment
identity plus the label-variant hashes. Do not reuse old run IDs or silently
replace labels inside a frozen experiment.

The loader checks the canonical split hash, artifact hashes, complete ID set,
original labels and the name rule. Stale, edited or incomplete label files are
rejected. Provenance is also available in the returned frame's
`attrs["gender_label_variant"]`; a future training entry point must explicitly
persist that provenance in its run configuration and registry evidence.

## Files

- `labels.csv`: every eligible development ID, original/new gender, matched
  cues, label source and change flag.
- `changes.csv`: only the 350 flips, with names, canonical folds and image IDs.
- `unclear_names.csv`: no-cue and multiple-cue rows, kept unchanged.
- `same_image_label_conflicts.csv`: the 11 rows whose identical images receive
  different name-based labels.
- `class_counts.csv`: old and new class counts.
- `summary.json`: rule, source and file hashes, fold counts and experiment scope.

## Checks and interpretation

All 37 selected data tests pass, including 20 tests for name parsing, unchanged
splits and protected rows, round-trip loading, artifact tampering, repeat builds
and identical-image conflicts. A full-data check confirms exactly 350 flips,
all other cells unchanged, identical training/validation IDs in all five folds,
and an unchanged canonical split file.

No model has been trained or rescored with this variant. This is a test of a
different label convention, not proof that the teacher labels are wrong. It
was proposed after inspecting development errors, so it is not an independent
blind evaluation. Keep future name-truth scores separate from the original
teacher-label scores and preserve the current acceptance rules unless a new
protocol is explicitly agreed.

## Training entry point

`notebooks/04ad_task3_gender_name_truth_screen.ipynb` trains folds 0 and 4 from
scratch. It keeps the 04ac recipe: dropout 0.30, grayscale probability 0.10,
the same mild darkening, 30 epochs and seed 2753. Run it on a fresh Colab L4
after pushing the notebook and its source changes. The existing teacher ZIP
is enough; the notebook rebuilds and verifies the small label files.

The new run hashes include the label contract. Original G2/E6/04ac artifacts
are checked against their original labels, then all comparison models are
evaluated on the same name-truth labels. The 19 numerical screen rules stay
unchanged. Separate original-label diagnostics reuse the exact saved
probabilities and add no acceptance gates. No training has run yet.

Results go to
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_dropout_030_grayscale_010/gender`.
The notebook stops after two folds for review. These research runs are marked
`submission_eligible=false` in the registry.
