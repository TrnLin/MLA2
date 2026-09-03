# Task 4 Progress Tracker

Use this page to follow the big milestones. We will add detailed checklists only
when we start each milestone.

Status words:

- `[x]` Done
- `[ ]` Not started
- `[~]` In progress
- `[!]` Blocked

## HD direction

The goal is a strong investigation, not only the highest search score.

- Compare several methods: colour/shape baseline, scratch CNN embedding,
  scratch autoencoder, and scratch Siamese/triplet model.
- Use any pretrained model as a comparison only, never as the submitted model.
- Compare teacher images against V1 with the same fair rules.
- Change one main thing per experiment and record every run.
- Judge quality together with speed, memory, storage, robustness, and failures.
- Test difficult queries such as crops, blur, compression, odd sizes, and backgrounds.
- Open the sealed holdout once, after the final method is frozen.
- Compare the final result with published work; V1 is not independent data.
- Demonstrate the final search through a simple GUI.
- Keep only figures that support a clear report claim.

## 1. Data ready

- [x] Audit teacher data
- [x] Audit V1 high-resolution images
- [x] Match V1 images to teacher IDs
- [x] Exclude official test images from the training index
- [x] Reuse the saved development, holdout, quarantine, and CV-fold rules

Evidence: `01_v1_eda.ipynb`, `results/evidence/task4/`

## 2. Evaluation rules

- [x] Choose one fold or all five folds
- [x] Define query and gallery rules
- [x] Define what “similar” means
- [x] Choose Top-K values
- [x] Choose the main score and supporting scores

Done when: every model can be compared using the same frozen rules.

Evidence: `docs/decisions/0019-task4-retrieval-evaluation.md`,
`05_task4_visual_search.ipynb`, and
`results/evidence/task4/retrieval_protocol_coverage.csv`

## 3. Image preprocessing

- [x] Compare useful image sizes
- [x] Choose resize, padding, and colour handling
- [x] Define arbitrary query-image handling
- [x] Fit any learned image values on training folds only

Done when: teacher, V1, and user images use one clear input contract.

Decision: `240×320` RGB, EXIF-corrected LANCZOS letterboxing with white
padding and a content mask. The fixed probe ranked `96×128` first; decision
0021 explicitly chooses `240×320` to preserve V1 detail for learned models.
Learned RGB values are fitted inside each round's training folds and never
refitted on queries.

Important failure: letterboxing is deterministic, but wide and tall
white-canvas queries lost most retrieval quality. Later robustness work must
address small clothing regions and large backgrounds.

Evidence: `docs/decisions/0020-task4-image-preprocessing.md`,
`docs/decisions/0021-task4-high-resolution-input.md`,
`results/evidence/task4/preprocessing_*.{csv,json}`, and
`results/figures/task4/preprocessing_comparison.png`

## 4. Baseline search

- [x] Build a simple non-deep-learning baseline
- [x] Measure quality, speed, memory, and common failures

Done when: later models have a fair comparison point.

Decision 0022 freezes the untrained `spatial-hsv-edge-4x4-v2` baseline. It
beats random, but its `0.94250242` cross-source ratio fails and rejects the 95%
source-robustness hypothesis. Quality, CPU timing, index cost, failure slices,
and all eight qualitative examples are tracked under `results/`.

## 5. Model comparisons

- [x] Pick several reasonable model families
- [x] Check scratch-training compliance
- [x] Freeze the experiment matrix and training budget

Done when: we know exactly which comparisons answer each project question.

Evidence: scratch R1–R5, comparison-only pretrained B1, and the approved
incremental matrix in `results/evidence/task4/learned/`.

## 6. Train and validate

- [x] Build the run registry
- [x] Train models from scratch
- [x] Validate with the frozen evaluation rules
- [x] Compare teacher images against V1 images
- [x] Save checkpoints, figures, timings, and run records

Done when: all planned runs are repeatable and recorded.

Evidence: `results/runs.csv`, six final learned manifests, and ten R5/R3
stability records.

## 7. Error analysis and final choice

- [x] Review good and bad search results
- [x] Check rare products, colours, image quality, and query-size failures
- [x] Compare quality against runtime and storage cost
- [x] Choose and freeze the final method

Done when: the winner is justified by evidence, not only one score.

Decision 0023 freezes scratch R5 with a teacher-only gallery. R5 beat R3 over
five fresh folds, passed the scratch/speed/storage gates, and won the
teacher/V1/two-view gallery study. Canvas sensitivity and failed/abandoned
attempts remain visible.

Task 10 fix round 1/5 restored the full HOG/fusion comparison and added the
final learned appendix. The notebook and report now read the compact,
strictly validated `results/evidence/task4/final/task4-final-comparison.json`.
Its source and template are ready to track, but they are not staged or
committed. The holdout remains sealed.

## 8. Final test

- [ ] Refit the frozen method on all development data
- [ ] Open the sealed holdout once in Notebook 06
- [ ] Record final results and limitations

Done when: no model or evaluation rule changes after seeing holdout results.

## 9. App and report

- [ ] Build the final search index
- [ ] Add image-query and Top-K results to the app
- [ ] Add comparison, failure, and judgement evidence to the report
- [ ] Check all assignment deliverables

Done when: another person can run the search, inspect results, and understand the
final choice.

## Current next milestone

**Milestone 8 — Final test**

The holdout remains sealed until Milestone 8.
