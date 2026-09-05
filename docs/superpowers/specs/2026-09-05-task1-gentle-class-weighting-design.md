# Task 1 Gentle Class-Weighting Design

## Purpose

Add one new scratch-CNN experiment that tests whether gentle class weighting improves
article-type macro-F1, especially for rare classes. Keep the ten completed unweighted CNN
folds and train only five new weighted folds. The change must preserve the canonical split,
append-only run registry, from-scratch training rule, and untouched holdout.

## Experiment question

The current development data are highly uneven: article-type support ranges from 1 to 5,748
products. The new experiment asks:

> With the architecture, image preprocessing, optimizer, seed, epoch budget, and folds fixed,
> does gentle class-weighted cross-entropy improve five-fold validation macro-F1 and rare-class
> F1 without unacceptable fold instability or loss of common-class performance?

The new candidate uses the existing no-augmentation preprocessing. This isolates the loss change.
Validation metrics and validation loss remain unweighted so all candidates stay comparable.

## Loss design

Create an explicit loss configuration rather than encoding a loss choice in a preprocessing ID.
The two supported loss identities are:

- `cross_entropy_unweighted_v1`
- `cross_entropy_sqrt_class_weighted_v1`

For each validation fold, calculate weights from that fold's development-training rows only:

1. Count the training examples for every fixed article-type class.
2. For each present class, calculate `sqrt(median_positive_count / class_count)`.
3. Divide present-class weights by their arithmetic mean.
4. Clamp the result to the range 0.25 through 4.0.
5. Give an absent training class weight 0.0. Record its absence; the model cannot learn a class
   that is absent from that fold's training rows.

Use these weights only in the training cross-entropy call. Use ordinary unweighted
cross-entropy for validation loss. Keep checkpoint selection based on the fixed validation
macro-F1 over all 124 classes.

Store the loss configuration, class counts, and final weights in the checkpoint. Include the
loss configuration in the run configuration hash and record the exact loss ID in
`results/runs.csv`.

## Candidate identity

Add an explicit CNN candidate identity so candidates that share preprocessing cannot be mixed.
Each fold result and evidence row carries `candidate_id`, `preprocessing_id`, and `loss_id`.

Use these candidate IDs:

- `task1_cnn_no_aug_unweighted_v1`
- `task1_cnn_mild_aug_unweighted_v1`
- `task1_cnn_no_aug_sqrt_weighted_v1`

The physical run `experiment_id` includes the candidate ID. This keeps run directories and
registry rows readable and unique.

## Incremental run flow

Keep the current unweighted controller behavior for existing callers. Add a focused weighted
experiment path with two stages:

- Weighted smoke: fold 0, one epoch, two train batches, two validation batches, not final eligible.
- Weighted full: folds 0 through 4, 20 epochs each, final eligible.

Every physical run appends through `fashion.train.registry`. A failure leaves the old comparison
evidence untouched and records the failed physical run normally.

After all five weighted folds complete, build combined evidence as follows:

1. Read the ten old unweighted fold rows from the current Task 1 evidence.
2. Match their run IDs to completed, final-eligible registry rows with the canonical split hash,
   unweighted loss ID, expected preprocessing ID, and folds 0 through 4.
3. Verify every old prediction file exists and matches its recorded SHA-256 hash.
4. Validate the five new weighted fold rows and prediction files in the same way.
5. Validate that each candidate has exactly one prediction for every expected development ID.
6. Atomically replace the shared CNN fold, comparison, OOF, and per-class evidence only after all
   three candidates pass validation.

Old checkpoints, histories, predictions, and registry rows are never deleted or rewritten.

## Evidence and figures

The combined evidence keeps these files under `results/evidence/task1/`:

- `fold_metrics.csv`: 15 rows with candidate, preprocessing, loss, fold, run ID, and metrics.
- `comparison.csv`: five-fold mean and sample standard deviation for all three candidates.
- `oof_metrics.csv`: pooled out-of-fold metrics for all three candidates.
- one per-class CSV for each candidate.

Add learning-curve figures under `results/figures/task1/`. Each figure shows training loss and
unweighted validation loss by epoch. A second panel shows validation macro-F1 by epoch. Existing
histories do not contain training macro-F1, so the figures must not invent it or require old runs
to be repeated.

The comparison figure groups by candidate ID rather than preprocessing ID. Its title and axis
labels describe a CNN candidate comparison, not only a preprocessing comparison.

## Notebook story

Reorder the Task 1 notebook narrative without deleting prior evidence:

1. Classical HOG baselines.
2. Scratch CNN without augmentation.
3. Learning curve and overfitting diagnosis.
4. Mild-augmentation test.
5. Gentle class-weighted-loss test.
6. Five-fold comparison and pooled OOF metrics.
7. Rare-class and confusion analysis.
8. Final development choice and handoff to the untouched holdout notebook.

The notebook states that augmentation reduced validation-loss overfitting but did not improve the
current mean macro-F1. It must wait for the weighted evidence before judging the new candidate.

## Failure handling

- Reject weights calculated from validation or holdout rows.
- Reject unknown loss IDs, non-finite weights, negative weights, or the wrong 124-element shape.
- Reject final-eligible runs that use a non-approved model, split, preprocessing, loss, seed, or
  training budget.
- Reject mixed candidates, duplicate folds, missing folds, stale/missing artifacts, hash
  mismatches, and incomplete OOF coverage before rewriting comparison evidence.
- Keep validation loss unweighted even for the weighted training candidate.

## Tests

Use test-first development. Add tests that prove:

- the hand-checked gentle-weight formula, normalization, cap, and absent-class behavior;
- weights use training-fold rows only;
- unweighted training remains unchanged;
- weighted training passes weights only to training cross-entropy;
- validation loss remains unweighted;
- run and checkpoint metadata contain the correct loss and candidate identities;
- smoke schedules only weighted fold 0 and is not final eligible;
- full schedules only the five weighted folds;
- combined evidence accepts exactly the two old candidates plus the new weighted candidate;
- broken registry identity, artifact hashes, fold coverage, or OOF coverage prevents evidence
  replacement;
- plots accept the existing history schema and write readable files;
- the notebook contains the approved experiment order and no new split operation.

Run the focused Task 1 test suite, lint the changed Python files, validate the notebook, run the
weighted smoke stage, then run five weighted folds. Finally regenerate and inspect the combined
tables and figures.

## Success rule

Do not call the weighted model the winner merely because it is new. Compare five-fold mean
macro-F1 first, then fold standard deviation, pooled OOF macro-F1, rare-class F1, common-class
impact, learning curves, runtime, and model size. A negative result is still useful evidence and
must be reported honestly.
