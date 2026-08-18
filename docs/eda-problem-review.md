# EDA Problem Review

This file records the dataset discussion. It separates measured facts from policy
decisions.

## Status meanings

- **Evidence complete:** the EDA measured the problem.
- **Decision complete:** we agreed how the project will handle it.
- **Action complete:** the agreed policy was implemented and checked.

## Decision records

- `docs/decisions/0004-use-review-driven-cleaning-and-target-masking.md`
- `docs/decisions/0005-use-group-aware-splits-with-a-catalogue-holdout.md`
- `docs/decisions/0006-preserve-class-prevalence-and-test-imbalance-handling.md`

## Already settled

- Dataset population: 38,617 official training IDs.
- Usable population: 38,612 products; five have no source image.
- Image source: use the original high-resolution image for each official training ID.
- Official test set: all 5,829 IDs and labels remain quarantined.
- Paired original and 60×80 images are two views of one product.
- Split source: `data/processed/splits.csv` will be the only split.

## Problem tracker

| Problem | Evidence | Decision | Action |
|---|---|---|---|
| 1. Target imbalance and long tail | Complete | Complete | Not started |
| 2. Dataset scope and fashion relevance | Complete | Complete | Not started |
| 3. Missing, unusual, or suspicious labels | Complete | Complete | Not started |
| 4. Image-quality problems | Complete | Complete | Not started |
| 5. Duplicate and leakage risk | Complete | Complete | Not started |
| 6. Distribution drift and evaluation risk | Complete | Complete | Not started |
| 7. Task-specific risks | Complete | Complete | Not started |

## 1. Target imbalance and long tail

### Measured facts

- `articleType` has 124 classes among the 38,612 usable products.
- Six `articleType` classes have one product each.
- Eight have two products each.
- In total, 32 classes have fewer than 10 products.
- Tshirts are the largest `articleType`: 6,780 products (17.6%).
- `usage` is most skewed: Casual has 29,636 products (76.8%).
- `gender` is led by Men: 20,913 products (54.2%).
- `season` is led by Summer: 19,135 products (49.6%).

### What is already clear

- The imbalance is mainly a genuine dataset difficulty, not proof of bad labels.
- Rare classes are also a coverage limit: the dataset has too few examples to learn or
  test them fairly.
- Plain accuracy is not enough.
- Macro-F1 and per-class support must be reported.
- Singleton classes cannot appear in train, validation, and test at the same time.
- A split cannot give reliable class-level results for very small classes.

### Agreed handling

- Keep every valid product and keep the official label names.
- Do not generate fake products with a generative model.
- Use normal image augmentation only inside the training fold.
- Compare ordinary training with one imbalance-aware method instead of assuming that
  weighting will help.
- Report macro-F1, per-class support, and results grouped by class-support band.
- Mark classes that are too small for fair validation as unsupported rather than claiming
  that their score is reliable.
- Consider extra real data only if it has the same label definitions, clear permission,
  and enough examples to change the rare-class problem.

### Agreed report framing

- Preserve the natural class distribution in validation and holdout data.
- Do not remove majority products or generate minority products merely to make class
  counts equal.
- Compare ordinary training with one imbalance-aware training method. This changes model
  learning, not the underlying evaluation distribution.
- State that rare-class performance is limited by data coverage, but support that statement
  with macro-F1, per-class support, and error analysis.
- Do not use dataset skew as a general excuse for weak results.

### Agreed usage-specific handling

- After excluding missing usage labels, the class counts are: Casual 29,636; Sports
  3,940; Ethnic 2,570; Formal 2,300; Smart Casual 55; Travel 25; Party 13; and Home 1.
- Use an always-Casual prediction as the 76.8% accuracy sanity baseline.
- Compare ordinary cross-entropy with focal loss using the same model and split. Focal
  loss reduces the influence of easy majority examples without deleting data.
- Do not heavily oversample tiny classes; repeating Home or Party images would mostly
  teach memorisation.
- Keep every valid label, but mark classes without enough validation support as
  unsupported.
- Judge the comparison with macro-F1, per-class recall, confusion matrices, calibration,
  and the majority baseline rather than accuracy alone.

## 2. Dataset scope and fashion relevance

### Measured facts

- Apparel, Accessories, and Footwear contain 37,502 products (97.1%).
- Personal Care contains 1,001 products. Most are fragrance, makeup, or nail products.
- Free Items contains 83 products. This is often a promotion label placed on real fashion
  items, such as watch ID 10595.
- Sporting Goods contains 25 products. At least 21 are footballs or basketballs; ID 1550
  is a football.
- Home contains one Cushion Covers product.
- `articleType == "Ipad"` has one row, ID 45824, but its name and image show women's
  flats. This is a label-review case, not evidence that the image is an out-of-scope iPad.

### What is already clear

- A broad catalogue category is not enough to decide whether a product is fashion-related.
- Free Items describes how a product was sold, not what the product is.
- Personal-care products are fashion-adjacent, so their scope is a project definition.
- Balls and cushion covers look outside the assignment's fashion-item goal.
- Suspicious metadata must be reviewed separately from genuinely out-of-scope images.

### Agreed handling

- Keep fashion and beauty products.
- Do not exclude a row only because its broad category looks unusual.
- Exclude only products visually confirmed as non-fashion, such as balls and cushion
  covers.
- Treat wrong labels separately from genuine non-fashion products.

## 3. Missing, unusual, or suspicious labels

### Approved correction for the cleanup session

- ID 45824 is a valid fashion image of women's blue flats.
- Its current hierarchy is `Free Items → Vouchers → Ipad`.
- Its product name is `Senorita Women Blue Flats`.
- Keep the product and override its hierarchy to
  `Footwear → Shoes → Flats` in a versioned correction table.
- After this correction, the cleaned `articleType` count would fall from 124 to 123 and
  the singleton-class count would fall from six to five.
- The raw CSV must remain unchanged.

### Missing target labels

#### Measured facts

- `season` has 20 blank values.
- `usage` has one blank value.
- `usage` also has 71 literal `"NA"` values. These are distinct from the blank value.
- Products of the same type can have either `usage == "Casual"` or `usage == "NA"`.
  This is common among personal-care products, so `"NA"` does not act like a stable
  visual occasion.

#### Agreed handling

- Treat blank targets as missing, not as classes.
- Treat literal `usage == "NA"` as a missing usage label, not as a ninth usage class.
- Keep every affected product for targets that do have labels and for visual search.
- Exclude or mask only the missing target when training and evaluating that target.
- Do not fill missing season or usage labels with the majority class.

### Suspicious label candidates

#### Measured facts

- The product-name check found 392 gender review candidates.
- This check is only a text rule. It does not prove that all 392 labels are wrong.
- ID 38223 is labelled `Ties and Cufflinks`, but its product name is
  `Polaroid Women Sunglasses` and its image shows sunglasses.

#### Agreed handling

- Keep ID 38223 and correct its article type to `Sunglasses` in the cleanup session.
- Review candidate records using metadata and images before correcting them.
- Do not auto-correct all 392 gender candidates from product-name words alone.
- Review related product groups together. For cases such as ID 3319, correct `Men` to
  `Boys` only when the group evidence is consistent; otherwise mask only the uncertain
  gender label.
- Record each accepted correction in the same versioned correction table as ID 45824.
- Keep the raw CSV unchanged.

### Category-hierarchy conflicts

#### Measured facts

- Twenty `articleType` labels appear under more than one `masterCategory` or
  `subCategory`.
- This means 20 article types are affected, not that only 20 product rows are affected.
- Some conflicts are easy to explain:
  - Backpacks appear under `Accessories → Bags` and promotional `Free Items → Free Gifts`.
  - Dresses appear under the `Dress` and `Topwear` subcategories.
  - Kajal and Eyeliner appears under `Eyes` and `Makeup`.
  - Wristbands appear under both Accessories and Sporting Goods.
- Other conflicts look more suspicious. Tshirts appear mostly under Apparel and Topwear,
  but at least one row uses Accessories and Belts.

#### Agreed handling

- Do not automatically force all 20 article types into one hierarchy.
- Do not use `masterCategory` or `subCategory` as model inputs.
- During cleanup, review only the rows using the uncommon hierarchy mapping.
- Correct clear metadata errors but keep defensible overlaps such as sports wristbands.
- Record all accepted corrections without editing the raw CSV.

## 4. Image-quality problems

### Measured facts

- Five official training products have no image in either source. They are already outside
  the 38,612-product usable population.
- Every usable 60×80 image decoded successfully.
- The low-resolution set has 38,269 RGB images and 343 grayscale images.
- Most low-resolution images are 60×80, but a small number have unusual dimensions; the
  observed minimum is 53×60.
- The grayscale review shows valid products rather than corrupt files.
- Brightness, contrast, and colour outliers also mostly look like valid catalogue-photo
  variation: dark products, white products, models, and unusual backgrounds.
- All 2,048 sampled original images decoded successfully and were RGB.
- The sampled originals have a median size of 1080×1440, while the teacher copies are
  usually 60×80.
- Sharpness values from the two resolutions are not directly comparable because the
  metric changes with image scale.
- Copying a grayscale value into three channels changes the file shape, not its colour.
  Lost colour cannot be recovered from a grayscale image alone.
- A fresh 256-original-image round-trip test against a 360×480 reference measured:
  - 60×80: mean PSNR 26.10 dB and edge correlation 0.560;
  - 192×256: mean PSNR 33.62 dB and edge correlation 0.855;
  - 240×320: mean PSNR 36.27 dB and edge correlation 0.914.
- Higher is better for both measures. The 240×320 review preserves collars, stripe
  patterns, and pose differences that disappear at 60×80.
- This verifies retained image detail, not classification performance. A controlled
  192×256 versus 240×320 model run is still required.

### Agreed handling

- Keep grayscale and unusual-but-readable images.
- Compare all 343 low-resolution grayscale images with their same-ID originals during
  cleanup.
- Use the original image as the source. If it contains colour, keep that real colour.
- If the original is also visually grayscale, copy its values into three channels only to
  give the model a consistent RGB-shaped input; do not invent colours.
- Exclude only missing, unreadable, or visually unusable images.
- Train from the original high-resolution source, then resize consistently in the loader.
- Preserve aspect ratio and pad when needed instead of stretching odd-sized images.
- The originals will normally be reduced to the model input size. Do not enlarge a small
  image to claim extra detail; place it unchanged on the padded input canvas instead.
- Do not automatically shrink every original to the teacher copy's 60×80 size.
- Run one controlled article-type pilot at 60×80, 192×256, and 240×320 with the same
  split, model, seed, and training settings.
- Choose one final input size from macro-F1, training cost, GPU memory, and inference
  speed. Train the full model comparisons only at that winning size.
- Use training-only brightness, contrast, and colour augmentation for robustness.
- Do not automatically remove statistical image outliers without visual review.

## 5. Duplicate and leakage risk

### Measured facts

- The usable low-resolution set contains 636 exact-image groups covering 1,399 product
  IDs.
- Twenty-two exact-image groups have at least one conflicting target label.
- Conflict-group counts by target are: gender 8, article type 7, season 9, and usage 3.
  One group can conflict on more than one target.
- The near-duplicate audit found 4,383 candidate pairs in a 2,048-product sample.
- Near-duplicate candidates are not confirmed duplicates. The review includes visibly
  different shirts, trousers, shoes, and deodorants with similar layouts.
- IDs 8855 and 8860 have identical low-resolution difference hashes, but their originals
  show different poses, collars, stripe patterns, and shirt details. They are similar
  products, not duplicates.

### Why this matters

- Putting the same pixels in training and validation lets the model memorise the image and
  makes validation results look better than real performance.
- Repeating the same image many times can give that product too much training weight.
- Identical pixels with different labels describe a task an image-only model cannot solve
  consistently.
- Near-duplicate hashes can join unrelated catalogue images and must not be trusted as
  automatic merge rules.

### Agreed handling

- Keep all official product IDs, but place every exact-image group wholly in one split.
- Review the 22 conflicting exact groups during cleanup.
- Correct clear metadata errors. If an identical image still has genuinely conflicting
  season or usage labels, mask that target for the conflicting group.
- For classification training, sample one representative per epoch only when the original
  high-resolution files are also exact and their target labels are consistent. Otherwise
  keep the products but still group them in the split.
- Keep all product IDs for search, but collapse confirmed original-image copies in
  displayed Top-K results so they do not fill the result list.
- Do not auto-group all 4,383 near candidates.
- After creating the group-aware split, review the strongest near matches that cross split
  boundaries using the original images, not 60×80 thumbnails. Join only pairs confirmed
  to show the same product/photo; do not join IDs 8855 and 8860.

## 6. Distribution drift and evaluation risk

### Measured facts

- Product mix changes across both catalogue ID ranges and recorded years.
- Total-variation distance measures this change: zero means the same mix and one means no
  overlap.
- Ten ID-range bins each contain about 3,400–4,300 products. Their article-type distances
  from the overall mix range from 0.213 to 0.445.
- The earliest ID bin has distance 0.445 and the latest has 0.432. Several middle bins are
  near 0.21–0.27.
- Year groups are very uneven. Years 2007, 2008, and 2009 contain only 2, 7, and 20
  products, so their large-looking drift values are not stable.
- Years 2011 and 2012 contain 27,050 products, about 70% of the usable data.
- Catalogue ID order is related to collection order but is not proven to be exact time.
- Official test labels remain quarantined and are not used to choose this policy.

### Why this matters

- A fully mixed split can make validation easier than later catalogue products.
- A strict year split is unstable because several years are tiny.
- A high-ID holdout is a useful stress test, but it may contain classes with little or no
  earlier support.

### Agreed handling

- Store every partition in the one allowed file: `data/processed/splits.csv`.
- Use a duplicate-group-aware and rare-class-aware training/validation split for model
  development.
- Reserve the highest official-training ID range as an untouched catalogue-shift holdout.
- Tune only on validation. Use the catalogue holdout once for final stress evaluation.
- Report holdout metrics with class support and clearly mark unseen or unsupported
  classes.
- Report year and ID-range breakdowns, but do not make claims from tiny year groups.
- Never inspect official test labels.

## 7. Task-specific risks

### Measured facts

- After the approved Ipad correction, article type will have 123 classes. It has the
  strongest long tail and needs fine image details.
- Season has four labelled classes and 20 missing labels. Summer covers about half the
  data, but season can be a catalogue choice rather than a visible property.
- Gender has five classes. Adult versus child intent can be impossible to infer from an
  isolated product image with no scale.
- After treating blank and `"NA"` as missing, usage has eight labelled classes. Casual
  covers about 77%, and occasion is often subjective.
- Article type is associated with usage, season, and gender in the metadata, but this does
  not prove that one shared image model will improve every target.
- Visual search has no supplied relevance labels saying which returned items are truly
  similar.

### Why this matters

- One loss, metric, or model design may help one target and hurt another.
- Combining gender and usage into one joint label would create many sparse class
  combinations.
- Season and usage may have a lower image-only performance ceiling than article type.
- Search accuracy cannot be honestly claimed from classification labels alone.

### Agreed handling

- Use image pixels only at inference; do not depend on metadata unavailable for official
  test images.
- Keep gender and usage as separate outputs.
- Compare separate from-scratch classifiers with a from-scratch shared-backbone,
  multi-head model. Do not assume multi-task learning wins.
- Use masked loss so each target ignores only its own missing or unresolved labels.
- Use macro-F1 and per-class support for every classification target.
- For article type, also report Top-K accuracy and support-band results.
- For season, gender, and usage, report confusion matrices and balanced class results.
- Select final models per target; do not force every target to use the same architecture
  if evidence favours different models.
- Evaluate visual search in two ways:
  - proxy Precision@K using labels such as article type;
  - a fixed human-rated query set for actual visual similarity.
- Keep evaluation queries out of the search gallery, collapse exact-copy results, and
  report search latency and index size.

## Discussion log

- 2026-08-18: Created the tracker. No cleaning, class, split, or modelling policy
  was chosen.
- 2026-08-18: Proposed keeping valid rare classes, avoiding generated products, using
  training-only augmentation, and experimentally comparing one imbalance-aware method.
  This proposal was approved.
- 2026-08-18: Approved keeping fashion and beauty products and excluding only visually
  confirmed non-fashion products.
- 2026-08-18: Proposed keeping ID 45824 and correcting its hierarchy from
  `Free Items → Vouchers → Ipad` to `Footwear → Shoes → Flats`. The correction was
  approved but deferred to a separate cleanup session.
- 2026-08-18: Proposed treating the 20 blank season labels, one blank usage label, and
  71 literal `"NA"` usage labels as target-specific missing values. This proposal was
  approved.
- 2026-08-18: Approved correcting clear label errors, manually reviewing the 392 gender
  candidates in related product groups, and masking only labels that remain uncertain.
- 2026-08-18: Approved reviewing only uncommon hierarchy mappings, correcting clear
  errors, and keeping defensible overlaps.
- 2026-08-18: Approved using original images, reviewing low-resolution grayscale pairs,
  keeping valid image variation, preserving aspect ratio, and running a controlled
  60×80 versus 192×256 versus 240×320 article-type resolution pilot.
- 2026-08-18: Approved exact-image group splitting, review and masking of conflicting
  exact groups, duplicate-aware classification sampling and search display, and
  original-image review without automatic grouping of near-match candidates.
- 2026-08-19: Approved a group-aware training/validation split plus an untouched
  high-ID catalogue-shift holdout, all recorded in the single allowed split file.
- 2026-08-19: Approved task-specific metrics and model selection, separate gender and
  usage outputs, a single-task versus multi-task comparison, and proxy plus human-rated
  visual-search evaluation.
- 2026-08-19: Approved preserving natural skew in validation and holdout data, testing
  one imbalance-aware training method, and using measured rare-class evidence rather than
  dataset skew as an excuse.
- 2026-08-19: Approved an always-Casual usage baseline and a controlled cross-entropy
  versus focal-loss comparison without aggressive rare-class oversampling.
