# Data-preparation figures

Notebook 01 creates these report-ready figures during Run All:

- `target_distributions.png`: all target classes on a log scale, including the long tail;
- `family_size_profile.png`: 32,773 development rows form 22,905 conservative split groups—safe
  blocks for data division, not 22,905 verified independent products; 83.1% of
  groups are singletons, 41.9% of products belong to multi-row groups, and the largest group
  has 80 products;
- `family_group_evidence.png`: the exact-hash, accepted-near-duplicate, and normalized-name
  evidence used to join products into conservative split groups;
- `broad_name_family_review.png`: the four widest non-empty normalized-name groups, with six
  evenly spaced sorted IDs per group; the 80-row `Lucera Women Silver Earrings` group visibly
  contains several earring designs, so the name block is not a verified SKU family;
- `near_duplicate_threshold_review.png`: a small development-only sample immediately on each side
  of the fixed automatic threshold; this is evidence only and never changes the pipeline;
- `class_support_products_vs_groups.png`: product counts versus conservative split-group counts;
- `rare_class_fold_support.png`: rare-class validation counts for each fold, with red boxes where
  the corresponding training side has zero products;
- `shortcut_risk_heatmaps.png`: articleType associations with season, usage, and gender;
- `shortcut_majority_benchmark.png`: global-majority versus articleType-majority descriptive
  agreement; this is not model accuracy;
- `acquisition_shortcut_risk.png`: image count and Season share by year; 69.9% of valid-Season
  rows are from 2011-2012, and year-majority agreement is 74.5% versus 49.6% globally;
- `season_file_size_shortcut.png`: compressed file size by Season; Fall's median is 2.2 KiB
  versus 15.0-18.1 KiB for the other seasons. Both acquisition figures are warning signals,
  not causal proof or model evaluation;
- `joint_target_nmi.png`: appendix-only NMI evidence on a 0-to-1 scale with no universal
  good/bad threshold; the largest off-diagonal value is 0.254;
- `image_quality_distributions.png`: sampled brightness and contrast distributions;
- `image_quality_spearman.png`: pixel-diagnostic correlations; brightness-contrast is -0.862
  and brightness-white-background share is 0.792, so the five columns are not five independent
  kinds of image quality;
- `development_contact_sheet.png`: a labelled development sample;
- `exact_duplicate_examples.png`: one exact duplicate pair shown side by side;
- `quality_extreme_examples.png`: sampled dark, bright, and low-edge examples;
- `transform_risk.png`: stretch, crop, and padding risks.

The plots use development rows only. Holdout and quarantine outcomes stay sealed.
Teacher images are the only shared image source. The transform figure shows risks; it
does not select a transform or define a Task 4 protocol. The family review is evidence
only and does not feed a human decision back into grouping, partitions, or folds.
