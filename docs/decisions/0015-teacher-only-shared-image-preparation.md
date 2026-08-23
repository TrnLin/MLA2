# 0015 — Teacher-only shared image preparation

- Status: Accepted
- Date: 2026-08-23
- Supersedes: 0008, 0013, and the fixed shared protocol parts of 0009

## Context

The assignment supplies teacher train and prediction images. The separate Fashion Product Images
collection is optional, has mixed dimensions, and contains IDs from both teacher roles. Treating it
as mandatory high-resolution training data makes every task slower and fixes image choices before
there is task evidence.

## Decision

- Shared preparation reads only `data/raw/teacher`.
- The optional collection lives at `data/raw/external/fashion_product_images_v1/` and is owned by
  Task 4 if that owner later chooses to test it.
- Shared code never parses the external `images.csv`. It may be byte-hashed later for provenance.
- Notebook 01 audits source geometry and demonstrates stretch, crop, padding, and resize risks, but
  does not select an image size or transformation policy.
- Dataset callers must pass transforms explicitly. Shared preparation does not create normalization
  values because learned image statistics must be fit inside each task's training fold.
- Task 4 owns arbitrary query sizes, image-size comparisons, query/gallery rules, relevance, cutoff,
  index design, ranking evidence, and optional external-image experiments.

## Consequences

Paired low/high manifests, paired normalization, and shared retrieval protocol artifacts are retired.
Tasks 1–3 use teacher images only. Task 4 remains free to compare resize or multi-size strategies, but
must record that choice in its own notebook and must not use official prediction targets.

The external collection is not called `train_4k`: it is neither uniformly 4K nor train-only.

## Verification

- Full and cached shared preparation pass when the external folder is absent.
- Shared runtime neither scans the external tree nor opens its CSV.
- Notebook 01 contains no selected image transform, paired-image policy, or retrieval protocol.
