# Data-preparation figures

Notebook 01 creates these report-ready figures during Run All:

- `target_distributions.png`: all target classes on a log scale, including the long tail;
- `family_duplicate_profile.png`: 32,773 development rows collapse to 22,905 independent
  families; 83.1% of families are singletons, 41.9% of products belong to multi-row families,
  the largest family has 80 products, and both partition/fold crossings are zero;
- `near_duplicate_threshold_review.png`: a small development-only sample immediately on each side
  of the fixed automatic threshold; this is evidence only and never changes the pipeline;
- `class_support_by_fold.png`: product-versus-family support and rare-class validation counts for
  each fold, with red boxes where the corresponding training side has zero products;
- `shortcut_risk_heatmaps.png`: articleType associations with season, usage, and gender plus
  descriptive majority benchmarks; these are not causal claims or model accuracy;
- `joint_target_nmi.png`: appendix-only normalized mutual information evidence;
- `image_quality_and_spearman.png`: pixel diagnostics only; width, height, and aspect ratio are not
  in the correlation heatmap;
- `development_contact_sheet.png` and `data_quality_examples.png`: labelled development examples;
- `transform_risk.png`: stretch, crop, and padding risks.

The plots use development rows only. Holdout and quarantine outcomes stay sealed.
Teacher images are the only shared image source. The transform figure shows risks; it
does not select a transform or define a Task 4 protocol.
