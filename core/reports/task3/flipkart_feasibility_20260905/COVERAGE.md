# Flipkart type coverage

The rules accept metadata matches for **85 of 124 teacher article types**. This is rule-accepted metadata coverage, not verified image coverage or proof that source labels mean the same thing as teacher labels.

- 11,347 of 20,000 source rows match one type under the saved rules.
- 407 rows have uncertain evidence; 8,246 have no exact rule match.
- 9,588 matched rows have a source Occasion field.
- 5,650 matched rows, across 52 types, have exactly one distinct source occasion token that exactly equals a teacher usage name. This is a text overlap check only, not label validation.
- 39 teacher types have no accepted match. The full list and per-type counts are in `teacher_type_coverage.csv` and `coverage_summary.json`.

## Method and limits

`coverage_rules.json` lists every alias. Matching uses whole category segments and explicit `Type` specification values, ignoring case and extra whitespace. It never uses a broad root such as Clothing as proof of a specific type. It does not use images or product names to create positive labels. Names containing combo, combination, material, unstitched or semi-stitched are withheld. Different candidate types are also withheld. Gardening, towel-holder and watch-accessory contexts are withheld to avoid homonyms such as a towel ring becoming jewellery or a watch strap becoming a wristband.

The rules preserve sports, formal and casual shoe distinctions. Generic shoes are not assigned a subtype. Mixed bins such as Leggings & Jeggings, Scarves & Stoles, Slippers & Flip Flops, and Camisoles & Slips need a more specific exact Type value. One-product sets such as bangle sets are permitted where explicitly listed. Garment sets are assigned only by an explicit set alias; conflicting garment evidence remains uncertain.

Aliases such as wedges → Heels and ballerinas → Flats are candidate semantic matches. Source errors can remain even where the rule matches one type. For example, a misleading source category cannot be fully detected without a separate manual audit. Rule-accepted metadata coverage is not a statistical bound on true labels.

Unmatched does not mean absent from the source. The map deliberately leaves unclear cases unresolved: pyjamas versus lounge pants, vests versus innerwear vests, tablet covers versus tablet sleeves, scarves versus stoles, and broad fragrance/combo bins. Expanding these requires source evidence, not guessing. No name-similarity search was used to inflate coverage.

Occasion tokens come from the source parser's comma split. The exact-single check uses one distinct token, compares case-sensitive names, and excludes teacher NA. Repeated identical tokens count as one distinct value. Counts say nothing about whether Casual, Party, Sports or other names are annotation-compatible across datasets.

## Reproduce and checks

Run from the repository root:

```sh
./.venv/bin/python reports/task3/flipkart_feasibility_20260905/coverage.py
```

The script reads only parsed source metadata, the development taxonomy and development class counts. It writes one coverage row per teacher type and one evidence row per source row. Checks passed for all 124 types, all 20,000 unique row indexes, one assignment per source row, and reconciliation of accepted rows with per-type totals. Counts are product rows; separate unique nonblank product-ID counts are supplied for each type. Multiple source rows with the same product ID are not assumed to be distinct products.

No training, image labeling, teacher protected-label reads, commits or pushes were done.
