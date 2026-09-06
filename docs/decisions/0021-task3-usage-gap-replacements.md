# 0021 — Replace outside Usage photos to cover observed data gaps

- Status: Accepted
- Date: 2026-09-06
- Extends: [0019](0019-task3-text-supported-usage-expansion.md); earlier versions remain
  valid for their historical runs.
- Amends: the named Task 3 dataset exceptions to 0014 and 0015.

## Context

The user requested a full data-gap review and replacement of some added images with
photos covering missing categories. Earlier additions had no Smart watches, wallets,
ties or sandals; no Party clutches, perfumes or watches; and weak Travel camera-bag
coverage. Photo style, audience coverage and concentrated product families also differed
from the teacher development set. A product's form alone does not establish its Usage.

## Decision

Create `data/processed/teacher_plus_rare_usage_v3_20260906/splits.csv` by replacing 130
outside rows from v2: Smart Casual 62, Party 44, Travel 17 and Home 7. The final dataset
still has 39,299 rows, including 38,612 teacher and 687 outside images. Preserve every
teacher field and all 557 retained outside rows, including IDs, partitions and folds.
Reserve retired IDs permanently and retain old data and photos. Only the named v3
manifest omits retired rows. Keep the nine-class Usage map, literal NA, masked non-Usage
targets, canonical teacher split, Task 4 gallery and existing experiment recipes intact.

Use the text/collection evidence allowance from 0019 and record inferred mappings,
source URLs, original photo hashes, rights status and review notes. Use only teacher
development labels for gap analysis. Teacher protected-role image identities may enter
duplicate checks without their target labels. Do not edit teacher labels to match new data.

Link source identities and conservative visual families across every old and new image.
Retain surviving families' saved fold anchors. New independent families use the existing
deterministic seed-2753 whole-family Usage balancing method. Reject any new linkage to
multiple conflicting anchors or labels. Also reject a replacement linked to a wholly
retired family, so removal cannot silently erase its historical fold anchor. No old row
is moved and no family is split for better class/type proportions.

All 130 admitted photos form 116 new families, at most four images each. Retiring the
49-image Smart family and reducing a repeated Travel family raises total outside families
from 549 to 597; the largest remaining family has 23 images. Frozen fold counts still
cause uneven per-type support. This limitation is recorded, not hidden by resplitting.

## Validation and consequences

All 130 candidates were visually reviewed in source and prepared views and compared with
all 45,128 prior image identities, including retired rows. No overlap was accepted under
the saved checks. Every final image loads and matches its hash; no ID, hash, normalized
name or linked family crosses any of the five train/validation boundaries. Inputs and
review receipts are hash-checked. The replacement/extension tests pass (18 tests).

The audit covers all nine classes and 21 rare teacher Usage × articleType combinations.
Tiny teacher classes, overlapping Usage meaning, photo/audience mismatch, source bias,
uneven type folds and limited source rights remain. One teacher Home example cannot
support a reliable independent Home evaluation. No model is trained in this change;
coverage improvement is not an accuracy result. Later experiments must register the v3
split hash, train from scratch, and report results on the same teacher validation rows.

Export the full split, fold lists, old/new/retired receipts, source geometry and validation
records. Supply a small local delta ZIP for use alongside the existing v2 data archive;
do not require a fresh full archive for each experiment.

## Evidence

- [Frozen v3 checks and MixUp + SAM recipe](../../src/fashion/train/task3_usage_replaced_v3.py)
- [Training and comparison tests](../../tests/train/test_task3_usage_replaced_v3.py)
- [Training guide](../../reports/task3_usage_replaced_v3_mixup_sam_20260907/README.md)
- [Real-data preflight](../../reports/task3_usage_replaced_v3_mixup_sam_20260907/preflight.json)
- The local intake review, original source receipts and visual gallery remain under
  `reports/task3_rare_replacement_20260906/`; product photos are distributed in the private
  data archives rather than the public code repository.
