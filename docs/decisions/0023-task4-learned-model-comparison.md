# Decision 0023: Task 4 learned model and gallery

**Status:** Accepted  
**Date:** 2026-09-03  
**Scope:** Development-only Task 4 selection

## Context

Task 4 needs one scratch-trained embedding and one gallery policy before the
labelled holdout is opened. The frozen input is `240×320` letterboxed RGB. The
frozen primary measure is mean per-query linear nDCG@10 on development data.
Every training attempt is recorded in `results/runs.csv`.

## Alternatives

- Scratch R1–R4: incremental CNN/VICReg candidates, ending with R4 adding
  batch-hard product-family triplet loss to R3.
- Scratch R5: convolutional autoencoder with an L2-normalized 128-value
  bottleneck.
- B1: pretrained benchmark, comparison-only and ineligible for submission.
- Teacher-only, V1-only, or two-view gallery. Two-view collapses each product
  to its minimum distance before Top-K.

## Evidence

The six final manifests live under `results/evidence/task4/learned/`. They
compare quality, four teacher/V1 source directions, synthetic wide/tall
canvases, CPU batch-one latency, and index bytes under one split and metric.

The ten `stability_evidence.json` files cover folds 0–4 for both R5 and R3.
`task9-post-stability-deployment.json` revalidates those files and records the
five-fold means, sample standard deviations, mean gap, pooled spread, source
ratio, canvas scores, and all deployment gates. R5's mean gap over R3 is much
larger than the pooled spread.

`task9-final-gallery-decision.json` revalidates the three gallery policies.
Teacher-only has the best development quality, the lowest measured p95 of the
three, and half the two-view index bytes.

## Decision

Select scratch R5 as the final development model. Use its L2-normalized
bottleneck with exact cosine distance. Keep one result per product by taking
the minimum distance before Top-K.

Use the teacher-only gallery. B1 remains useful comparison evidence but is
permanently ineligible for final refit, holdout judgement, and submission.

## Gallery provenance

R5's original `cost.json` says `selected_gallery_policy: v1`. That field was a
pre-study cost assumption used to finish candidate evidence. It is not the
final gallery choice. The canonical post-study source is:

`results/evidence/task4/phase_results/task9-final-gallery-decision.json`

That artifact explicitly labels the old V1 value
`pre_study_cost_assumption` and freezes `teacher`.

## Deployment assumptions

- Arbitrary queries use the saved teacher-source normalization.
- CPU timing is batch size one, one thread, on the recorded machine.
- Index bytes describe the measured exact-search artifacts.
- Metadata relevance is a proxy for human visual similarity.

## Consequences

Notebook 06 must refit this exact R5 choice on all development folds, create a
registered `final_refit` run, and evaluate the labelled holdout once. It may
not change the model, gallery, preprocessing, fusion, or metric after that.

Teacher-only uses less storage than two-view and avoids treating V1 as new
data. R5 still loses substantial quality on large white canvases, so the app
and report must state that limit.

## Failures kept visible

R1/R2 non-finite-gradient attempts and abandoned stability attempts remain in
the registry. The real gallery command wrote valid durable artifacts and then
failed to shut down. The native shutdown cause is unknown. Bounded supervision
now prevents this hang from being reported as successful command completion.

## Holdout boundary

Only `data/processed/splits.csv` defines membership. This decision uses
development artifacts only. It does not open holdout, quarantine, or official
teacher-test images. The holdout stays sealed until Notebook 06.
