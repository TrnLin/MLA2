# Boys, Girls and grayscale error inspection

The main patterns are **color sensitivity, confusion between child and adult
catalog categories, and possible catalog inconsistencies**. Increasing dropout
did not resolve them. This is an inspection of saved predictions and images;
no model was trained, no fresh inference ran, and no labels or splits changed.

The analysis uses all 13,110 validation rows on canonical development folds 0
and 4, including 274 Boys and 216 Girls items. All 16 input prediction files
match their saved evaluation-manifest hashes and the canonical IDs, labels,
folds and product families. We visually inspected 94 panels covering 93 product
IDs. The illustrated images also match their canonical SHA-256 hashes.

## 1. Stronger dropout loses useful child-item decisions

| Catalog class | Correct at 0.30 | Correct at 0.45 | New mistakes | Mistakes fixed |
| --- | ---: | ---: | ---: | ---: |
| Boys, 274 items | 180 | 172 | 25 | 17 |
| Girls, 216 items | 125 | 117 | 17 | 9 |

The 42 new mistakes represent 30 canonical product families and 30 distinct
source-image hashes: 19 Boys families and 11 Girls families. All 30 families
are pictured below, using one lowest-ID example per family. Repeated catalog
copies must not be counted as independent confirmation of the pattern.

The Boys regression is concentrated in T-shirts: 16 new misses and 8 recoveries
among 171 Boys T-shirts. These are also the most common Boys article type, so
the count alone does not establish a special T-shirt failure rate. Girls changes
are spread across tops, shorts, dresses, shoes and other small slices.

The changes differ by fold. On fold 0, Girls have 13 new misses and only 2
recoveries; on fold 4 they have 4 new misses and 7 recoveries. Boys have more
new misses than recoveries on both folds. This helps explain why the overall
gap improvement was weak on fold 0.

Visual inspection found normal, readable product photos rather than empty or
obviously corrupted inputs. Examples include striped shirts, plain shorts,
jeans and slip-on shoes. Their photos contain limited evidence about intended
age or physical size. A Boys shirt predicted as Men is often visually plausible,
although it remains wrong under the fixed catalog labels. At dropout 0.30,
114 of the 185 mistakes on Boys/Girls items predict Men or Women. At 0.45,
125 of 201 do. The ambiguity is not fixed by hiding more features.

There are also 159 child-item mistakes shared by both models. The image sheets
include high-confidence shared errors and recoveries as comparison examples;
they are not a random sample from which to estimate visual-pattern prevalence.

- [All Boys regression families, page 1](boys_regressions_1.png)
- [All Boys regression families, page 2](boys_regressions_2.png)
- [All Girls regression families](girls_regressions_1.png)
- [Confident mistakes shared by both models](shared_child_errors.png)
- [Examples that stronger dropout fixes](child_recoveries.png)

## 2. Grayscale is especially damaging to Girls at dropout 0.30

| Catalog class | Correct in color, dropout 0.30 | Correct in grayscale, dropout 0.30 |
| --- | ---: | ---: |
| Boys | 180 / 274 | 159 / 274 |
| Girls | 125 / 216 | 60 / 216 |

For Girls, grayscale breaks 68 previously correct predictions and fixes only 3.
Those 68 new errors predict Men 27 times, Women 21, Boys 18 and Unisex 2.
For Boys, it breaks 36 correct predictions and fixes 15; 25 of the newly wrong
predictions are Men.

Pink Girls items are a clear descriptive slice: 45 of 61 are correct in color,
but only 12 are correct in grayscale. These items span article types, so this
does not isolate a universal effect of pink from clothing shape or product mix.
The visual examples show that removing color changes predictions even when
the product outline remains visible. Some modeled apparel and undergarments
also flip from Women to Men. This supports color sensitivity; it does not prove
exactly which internal features the network uses.

Girls contribute −0.022231 to the dropout-0.30 grayscale gate difference versus
E6, whose overall value is −0.027437. Thus Girls alone contribute more than the
allowed −0.020 macro-F1 loss; other classes add some loss and some offset.
These are averages of per-fold F1 changes, matching the actual screen rule.

An important distinction: dropout 0.45 **improves Girls grayscale F1**, from
0.379747 to 0.406250 pooled. It still weakens their clean F1, from 0.595238 to
0.572127. It also lowers grayscale F1 for Boys, Men, Unisex and Women. Hence
the overall grayscale score falls, despite the smaller induced drop and the
Girls grayscale improvement. Raising dropout is not a reliable grayscale fix.

[Paired original/grayscale examples](grayscale_failures.png) show four Boys,
eight Girls and four Women cases. Selection takes the largest losses in true-
label probability within those classes, with one example per canonical family.
Grayscale uses the exact evaluation conversion: RGB → L → RGB. Enlargements
use nearest-neighbor scaling, not image enhancement.

## 3. Some recorded child predictions disagree with inconsistent metadata

There are **356 development rows from 244 families** whose names explicitly
contain Boys or Girls but whose catalog gender differs. There are **154 such
rows from 105 families** in the inspected validation folds: 82 labeled Men
with Boys wording, 71 labeled Women with Girls wording, and one labeled Girls
with Boys wording. The check uses explicit word tokens, not a judgment based
on color, appearance or a person's gender.

Examples checked against the images:

- ID 4875: catalog **Men**; name **Levis Kids Boy's Davis White Polo Tshirt**.
  Dropout 0.30 predicts Men; 0.45 predicts Boys.
- ID 4886: catalog **Women**; name **Levis Kids Girl's Dana Pink Teens Kidswear**.
  Dropout 0.30 predicts Women; 0.45 predicts Girls.
- ID 4930: catalog **Women**; name **Gini and Jony Girl's Vandia Blue Dungree
  Infant Kidswear**. Both models predict Girls.

At dropout 0.30, 32 of 82 false Boys predictions and 31 of 79 false Girls
predictions agree with this explicit name signal. At dropout 0.45, the counts
are 35 of 92 and 33 of 76. Seven new false Boys predictions and eight new false
Girls predictions after increasing dropout fall in these flagged rows.

These are **audit candidates, not confirmed wrong labels**. Names can be wrong,
and a photo alone may not establish the intended audience. The inconsistency
may affect training and evaluation, but it does not explain all model failures
or justify changing labels after inspecting validation errors. The official
scores and acceptance decision remain unchanged.

- [Visual label–name examples](name_label_conflicts.png)
- [All validation audit candidates](validation_name_label_conflicts.csv)
- [All development audit candidates](development_name_label_conflicts.csv)

## Recommended next step

For the next single-factor model trial, use **dropout 0.30 plus the same mild
darkening**, and test a small probability of grayscale during training. That
directly tests the observed color sensitivity. Predeclare the probability and
keep the same folds, references and all acceptance rules. It could hurt useful
color cues or fail to close the clean training–validation gap; no benefit is
claimed before the experiment.

Keep the catalog-conflict list as a separate data audit. Any future training-
label intervention needs a documented, independently justified rule; do not
relabel or remove validation rows to improve a score. Neither that intervention
nor grayscale training has been implemented here.

## Reproduce and inspect

Run from the repository root:

```bash
./.venv/bin/python reports/task3/gender_error_inspection_20260905/inspect_errors.py
```

Outputs include exact [input hashes](input_hashes.json), [class precision,
recall and F1](class_metrics.csv), [confusion counts](confusions.csv), [paired
error changes](paired_switches.csv), [article slices](article_slices.csv),
[training support](training_support.csv), [grayscale gate decomposition](grayscale_gate_decomposition.csv),
and [the visual-selection manifest](visually_inspected_examples.csv).
Small article slices are descriptive. The reused development folds do not
provide an independent final-test estimate. No held-out test images were used.
