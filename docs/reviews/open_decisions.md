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

## 3. Task-owner experiment choices

Each owner must choose and justify one fixed `cv_fold` or all five folds, task-specific
preprocessing, comparison models, metrics, and error slices. Nothing is selected here.

## 4. Task 4 choices

The Task 4 owner must decide the arbitrary-query image policy, one or more image sizes,
whether to use optional external images, query/gallery isolation, relevance, K, index
construction, and ranking evaluation. The relevance rule must be labelled as a proxy, not
real-world similarity ground truth.
