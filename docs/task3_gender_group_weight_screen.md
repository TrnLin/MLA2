# Gender/product loss-weight trial

Run `notebooks/04ae_task3_gender_group_weight_screen.ipynb` on a fresh Colab L4
after pushing its source changes. It trains folds **0 and 4 only**, from scratch,
and stops for review. The completed 04ad models are the direct parents.
The local machine has not run this trial.

## Reason

The completed name-label screen passed 17 of 19 checks. It still failed the
fold-0 gap reduction and the mean 0.050 gap-reduction requirement. The deeper
review found Boys, Girls and Unisex explain 87% of its clean training–validation
gap. Many Unisex shoes, watches and sunglasses become Men or Women. Conversely,
all 23 Men validation backpacks became Unisex. These patterns motivate one
training-weight trial; they do not prove what the model uses internally.

## Frozen change

For every observed `(articleType, gender)` training group, let `n` be its row
count and `m` the largest gender count in that article type. Set:

```
raw_weight = min(3, sqrt(m / n))
weight = raw_weight / mean_raw_weight_across_training_rows_of_this_articleType
training_loss = sum(weight * cross_entropy) / sum(weight)  # within each batch
```

The square root softens inverse-frequency weighting. The cap limits the
minority-to-majority weight ratio to 3. Normalizing inside each article type
keeps its total training weight equal to its original image count. Balanced
groups and types with only one observed gender have weight 1.

This uses every observed group, not a hand-picked list of validation failures.
Counts use valid gender rows in the four training folds only. Every image is
seen once per epoch; there is no replacement sampling, extra data, filtering,
or new split. Empty groups are not created. Group counts are image counts,
not independent family counts. A tiny group can still lack useful variety.
The cap and exponent are fixed for this single trial; there is no sweep.

On the current training data:

| Group | Fold 0 weight | Fold 4 weight |
|---|---:|---:|
| Unisex casual shoes | 2.179 | 2.172 |
| Unisex watches | 2.450 | 2.453 |
| Unisex sunglasses | 1.645 | 1.623 |
| Men backpacks | 2.366 | 2.390 |

The normalized weights can be below 1 for dominant groups. Gender totals also
change: Boys gain roughly twice their original weight, while Men lose weight.
The article-type totals stay fixed. More minority recall may come with false
positives; weight cannot supply missing visual cues.

Keep the 04ad labels, all canonical fold IDs, model, optimizer, normalization,
augmentation and random streams: dropout 0.30, grayscale probability 0.10,
mild darkening, seed 2753, batch 128, 30 epochs and the final checkpoint.
Clean training, validation and corruption scores use **no group weights**.
The model still has 390,181 parameters. It is not submission eligible.

## Evidence and checks

The source audit checks the exact completed 04ad runs and their prior chain.
It binds label hashes, training-only weight contracts and the weighting code
hash before training. Each fit saves `training_selection.csv` containing every
training row and its weight. After fitting or reuse, a fresh fold-only fit must
reproduce that file's hash and configuration contract. Each fit is registered
through `fashion.train.registry` and mirrored to Drive and the local registry.

Results go to:

```
MyDrive/MLA2/task3/experiments/t3_gender_name_truth_article_weight_sqrt_cap3/gender
```

`training_weight_preview.csv` contains all 245 fold-0 and 242 fold-4 observed
groups. `screen_decision.json` keeps the same 19 checks against matched G2/E6
on the same name-truth labels. No old failure is erased or threshold relaxed.

`incremental_comparison.json` compares against the completed 04ad NameTruth
parent on identical labels: pooled F1, a 10,000-draw paired-family interval,
class F1, per-fold clean gaps and corruption scores. Its inherited `dropout_*`
field names refer to the **NameTruth** parent. Candidate and parent metric
blocks include NLL and ECE. This direct comparison is descriptive and adds no
acceptance gate. Both name-truth and original teacher-label diagnostic scores
are retained without another GPU pass.

Review Unisex recall and precision, Boys/Girls F1, newly wrong Men/Women
examples, and raw corruption scores alongside clean scores. A smaller training
gap alone is not enough. This is another development experiment on reused
folds, not an independent blind evaluation. Do not auto-run folds 1–3 or the
held-out test.
