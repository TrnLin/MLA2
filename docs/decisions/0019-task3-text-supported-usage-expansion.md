# 0019 — Text-supported rare Usage expansion

- Status: Accepted
- Date: 2026-09-06
- Extends: [0018](0018-task3-external-source-label-intake.md) with a second named dataset;
  the earlier dataset and its evidence policy remain valid for historical runs.
- Amends: the named Task 3 exceptions to [0014](0014-development-holdout-cv-boundary.md)
  and [0015](0015-teacher-only-shared-image-preparation.md).

## Context

The user requested at least 100 additional images for each rare Usage class and explicitly
allowed suitable product names/descriptions to support a proposed Usage without an exact source
label. The user then excluded NA and requested a new training dataset with the same fold method
as before. The reviewed collection contains 567 images: Home 120, Party 127, Smart Casual 156,
and Travel 164. This record documents those authorised choices.

## Decision

Create `data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv` with all 38,732 earlier
rows plus the 567 new images: **39,299 rows in total**. Preserve every previous field, ID,
partition and CV fold, including those of the earlier 120 external images. Append new numeric
IDs after the earlier reserved IDs. Keep the same nine-class `usage` map, including literal NA.
Do not assign NA to any new image. Other targets remain blank with false validity masks.

Treat both `product_text_inference` and `explicit_source_label` as reviewed evidence for this
version. The latter can mean retailer collection membership or product tags; it is not proof
of a native nine-class Usage annotation. Retain original product text, source URLs, evidence
basis, rights information and the explicit inferred mapping in each new row. The older exact
occasion/style validator is unchanged.

All 687 external images were linked across both intakes using source identities, normalized
names, image identity and conservative visual-family rules. The result has 549 family groups,
with no new-to-old family links and no conflicts between earlier folds. Place the 440 new
families into five folds deterministically with seed 2753, balancing Usage counts against the
existing fold counts. Keep each whole family together. If a future input links to an existing
family, it must follow that family's fold; conflicting existing fold anchors fail validation.
The current Smart Casual collection includes a 49-image conservative family, so exact 80/20
class proportions are not possible in every fold. Family separation takes priority.

The full split is the authority. Export fold-0 `train.csv` and `validation.csv`, all five train
and validation ID lists, class/fold counts, all 687 added rows, the 567 new rows, unchanged label
maps, source-border geometry, source evidence and validation results. Select the new split
explicitly through the normal `load_splits`, `get_cv_split`, and `FashionDataset` APIs.
Do not redirect the original teacher split, Task 4 gallery, or old training recipes.

## Validation and consequences

The intake compared original, resized and foreground image views against all 44,441 teacher-role
images plus the earlier 120 additions, using image/identity fields only. No accepted overlap
was found. A separate within-collection check removed two near-duplicate images. These checks
reduce duplicate risk but cannot prove that every edited or related image was found.

The dataset build checks the reviewed collection package and current image bytes. It proves
that original rows and folds survive, protected targets remain blank, source membership is
complete, and IDs, hashes, normalized names and linked families never cross any saved train/
validation boundary. Every row is loaded through the normal RGB 60×80 transform, and a mixed
teacher/old-external/new-external batch is checked. Learned preprocessing and class weights must
be fitted on each training fold only; no old source-only statistics are reused.

Seller-purpose labels can overlap and may differ from teacher labelling. The new sources also
have different product and photo distributions. Keep source-specific results separate when
evaluating a later model, compare on the same teacher validation IDs, and do not claim that more
images must improve accuracy. Retailer rights limits and the ABO licence conflict remain in
the source manifests. No model fit is part of this dataset build; every future fit must start
from scratch and register its new split hash.

## Evidence

- [Reusable extension and validation](../../src/fashion/data/usage_extension.py)
- [Boundary and input tests](../../tests/data/test_usage_extension.py)
- [Versioned dataset build](../../reports/task3_usage_expanded_v2_20260906/build_dataset.py)
- [Family linking](../../reports/task3_usage_expanded_v2_20260906/audit_family_links.py)
- [Source intake](../../reports/task3_rare_expansion_20260906/README.md)
- [Usage guide and split counts](../../reports/task3_usage_expanded_v2_20260906/README.md)
