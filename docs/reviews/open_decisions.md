# Open human decisions

These decisions belong to later task work. They do not block teacher-only preparation.

## 1. Cross-role visual matches

Review whether the listed teacher/prediction pairs show the same product. The safe current
rule keeps every affected labelled product in quarantine and keeps prediction products out
of development.

`50723/52131`, `49743/58884`, `50311/58893`, `48604/53103`, `48716/59550`,
`49740/58893`, `50305/58893`, `49743/58888`, `48708/59545`, `48591/58619`.

## 2. Broad product-name families

Review whether some normalized names make split blocks too broad. Keeping each family
whole prevents leakage but reduces the number of independent units. The largest current
development family is `family_3a8dd25529104cb0`, with 80 products. Full membership is in
`data/processed/splits.csv`.

The deterministic development-only review in
`results/figures/data_preparation/broad_name_family_review.png` shows that this
`Lucera Women Silver Earrings` group contains several earring designs. Keep the broad
group whole for assignment split safety. Treat 22,905 as conservative split groups, not
verified independent products. An industry rebuild would need real SKU identifiers or a
human family-review step.

## 3. Task-owner experiment choices

Each owner must choose and justify one fixed `cv_fold` or all five folds, task-specific
preprocessing, comparison models, metrics, and error slices. Nothing is selected here.

## 4. Task 4 choices

Decision 0018 fixes one boundary: V1 is a high-resolution copy of the same teacher
catalogue, inherits `data/processed/splits.csv`, and is never split again. Its focused
audit is in `notebooks/task-4/01_v1_eda.ipynb`.

Decisions 0019, 0021, and 0022 now fix the evaluation contract, the `240×320`
input size, the arbitrary-query letterbox policy, and the untrained search
baseline. The Task 4 owner must still decide whether V1 improves learned
retrieval enough to justify its cost, index construction, and the final learned
representation. The final winner remains open. The relevance rule remains a
proxy, not real-world similarity ground truth.
