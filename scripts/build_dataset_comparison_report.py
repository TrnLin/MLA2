from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/figures/dataset-comparison/comparison-summary.json"
REPORT = ROOT / "docs/dataset-quality-comparison.md"


def number(value: int | float) -> str:
    return f"{value:,.0f}"


def percent(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pop = data["populations"]
    loss = data["teacher_train_image_loss"]
    labels = data["labels"]
    images = data["images"]
    json_audit = data["artifact_consistency"]["json"]
    article_shift = labels["split_comparison"]["articleType"]
    same_id = images["same_id_comparison"]
    exact = images["exact_duplicates"]
    original_audit = images["audits"]["original"]
    teacher_audit = images["audits"]["teacher_train_present"]
    test_audit = images["audits"]["teacher_test"]

    unseen = ", ".join(
        f"{row['label']} ({number(row['test_rows'])})"
        for row in article_shift["unseen_test_labels"]
    )
    known_candidates = "; ".join(
        f"{row['id']} — {row['articleType']} — {row['productDisplayName']}"
        for row in labels["known_review_candidates"]
    )
    conflict_counts = ", ".join(
        f"{target}: {count}"
        for target, count in exact["conflict_counts_by_target"].items()
    ) or "none"
    near_sweep = "; ".join(
        f"d≤{row['threshold']}: {number(row['candidate_pairs'])} pairs, "
        f"largest component {number(row['largest_component'])}"
        for row in images["near_duplicate_sample"]["sweep"]
    )
    mismatch_total = sum(json_audit["mismatch_counts"].values())

    report = f"""# Fashion Dataset Quality Comparison

Generated from the fresh machine-readable audit at
`results/figures/dataset-comparison/comparison-summary.json`
({data['generated_at_utc']}).

## Bottom line

The original dataset is better for **image quality**, but it is not a cleaner independent
source of labels. Every shared teacher-train label matches the original metadata, and the
restored teacher directory now has every source image available for its train IDs. The
safest assignment dataset is:

1. the official {number(pop['teacher_train_metadata_rows'])} teacher-train IDs;
2. joined to the original high-resolution images and metadata for those IDs only; and
3. with all {number(pop['teacher_test_rows'])} official test IDs and recovered labels
   quarantined from training, tuning, feature selection, and model comparison.

Training on all {number(pop['original_metadata_rows'])} original rows would expose the
official answer key and invalidate the held-out evaluation.

## What was compared

- Original source: {number(pop['original_metadata_rows'])} metadata rows,
  {number(pop['original_image_files'])} JPGs,
  {number(pop['original_json_files'])} JSON records, and
  {number(pop['original_link_rows'])} URL records.
- Teacher train: {number(pop['teacher_train_metadata_rows'])} metadata rows and
  {number(pop['teacher_train_image_files'])} JPGs currently present.
- Teacher test: {number(pop['teacher_test_rows'])} blank prediction rows and
  {number(pop['teacher_test_image_files'])} JPGs.
- ID proof: the original IDs are exactly the disjoint union of teacher train and teacher
  test IDs.

![Metadata and image coverage](../results/figures/dataset-comparison/population-coverage.png)

## Teacher training copy is restored

The current `data/train-old/images_train/` directory now reconciles with the source:

- expected metadata rows: {number(loss['metadata_rows'])};
- present JPGs: {number(loss['present_images'])};
- metadata rows without a JPG: {number(loss['missing_images'])};
- missing JPGs recoverable from the original image directory:
  {number(loss['recoverable_from_original'])};
- known metadata-only IDs missing from both sources:
  {', '.join(loss['unrecoverable_known_orphans'])};
- teacher-train image coverage among source-available rows: 100%.

The older EDA also recorded {number(38_612)} teacher images. The five remaining gaps are
the same metadata-only IDs absent from the original image collection, so they are source
orphans rather than failed transfers.

## Label quality

### Shared labels are the same

All nine compared metadata fields match on every shared teacher-train ID. The original CSV
is therefore not an independent relabelling. It cannot be called “better labelled” merely
because it is larger.

The original JSON audit found {number(mismatch_total)} CSV-to-JSON field disagreements
across {number(json_audit['files'])} files, with
{number(len(json_audit['parse_errors']))} JSON parse errors. This is useful corroboration,
but both artefacts come from the same catalogue source.

### The official test population is shifted

`articleType` has total-variation distance
{article_shift['total_variation']:.3f} between teacher train and quarantined test labels.
There are {number(len(article_shift['unseen_test_labels']))} article types absent from
official training but present in the test set, covering
{number(article_shift['unseen_test_rows'])} test rows
({percent(article_shift['unseen_test_rows'] / pop['teacher_test_rows'])}).

Those unseen labels are: {unseen}.

This is a dataset-design limitation: a classifier cannot learn a class with zero training
examples. It also means a single random validation split from teacher train will not mimic
the official test distribution.

![Train-to-test label shift](../results/figures/dataset-comparison/label-shift.png)

![articleType long tail](../results/figures/dataset-comparison/article-type-long-tail.png)

### Weird and suspicious labels

- `articleType` classes in official train:
  {number(labels['article_type_classes'])}.
- Singleton `articleType` classes:
  {number(labels['article_type_singletons'])}.
- `articleType` classes with fewer than 10 rows:
  {number(labels['article_type_under_10'])}.
- `articleType` hierarchy conflicts:
  {number(labels['hierarchy_conflicts'])}.
- Product-name versus gender review candidates:
  {number(labels['name_gender_contradictions']['count'])}.
- Exact-image groups with at least one target conflict:
  {number(exact['label_conflict_groups'])}.
- Exact-image conflict groups by target: {conflict_counts}.

Two high-priority known candidates remain: {known_candidates}.

These are review candidates, not automatic corrections. In particular, season and usage
are merchandising labels and may genuinely differ even when pixels are identical.

![Label review candidates](../results/figures/dataset-comparison/label-review-candidates.png)

## Image quality

The original collection has a median resolution of
{original_audit['megapixels_median']:.2f} megapixels and a median file size of
{original_audit['bytes_median'] / 1024:.0f} KiB. The currently present teacher-train
images have a median resolution of {teacher_audit['megapixels_median']:.4f} megapixels
and a median file size of {teacher_audit['bytes_median'] / 1024:.0f} KiB.

The same-ID sample compared {number(same_id['sample_size'])} teacher images with their
original files:

- median source-to-teacher pixel ratio:
  {same_id['median_pixel_ratio']:.1f}×;
- median source-to-teacher file-size ratio:
  {same_id['median_file_size_ratio']:.1f}×;
- median dHash distance:
  {same_id['median_dhash_distance']:.1f};
- pairs at dHash distance 6 or less:
  {percent(same_id['share_dhash_at_most_6'])};
- median resized-image PSNR:
  {same_id['median_psnr_db']:.1f} dB.

This supports using the original image for the same official train ID: it retains much more
detail while still depicting the same product. It does not justify adding the official
test IDs.

Integrity checks found {number(len(original_audit['header_errors']))} original header
errors and {number(len(original_audit['decode_errors']))} failures in the
{number(original_audit['decoded_files'])}-file original decode sample.
All {number(teacher_audit['decoded_files'])} currently present teacher-train JPGs and all
{number(test_audit['decoded_files'])} test JPGs were decoded; their failure counts are
{number(len(teacher_audit['decode_errors']))} and
{number(len(test_audit['decode_errors']))}.

## Duplicate and leakage risk

Full-file SHA-256 hashing found {number(exact['groups'])} exact duplicate groups covering
{number(exact['images'])} original images. Of these,
{number(exact['cross_split_groups'])} groups cross the official train/test boundary.
Those groups can inflate apparent generalisation if duplicate products appear on both
sides.

The perceptual analysis is deliberately a sample and a review queue. Threshold growth was:
{near_sweep}. A large component at a loose threshold is not proof that all members are
duplicates; transitive chaining can join unrelated catalogue photos.

## Recommendation

Use **official teacher-train IDs plus original high-resolution files**. Keep the teacher
test ID list immutable and exclude those IDs before any preprocessing. Preserve literal
`"NA"` separately from true blanks. Group confirmed exact duplicates before making a
development split. Review suspicious labels manually and record any correction policy;
do not silently edit raw CSVs.

For evaluation, report macro-F1 and per-class support. The severe long tail and unseen test
classes make accuracy alone misleading.

## Limits

- Full image headers and hashes were scanned, but full pixel decoding of the 14 GiB original
  collection used a deterministic sample of {number(original_audit['decoded_files'])}.
- Near-duplicate analysis used a deterministic sample of
  {number(images['near_duplicate_sample']['sample_size'])}; it is not an exhaustive visual
  duplicate census.
- Aggregate quarantined test-label statistics have now been inspected. They must not be
  used to tune choices.
- Product-name checks are heuristics. Human review is required before changing labels.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
