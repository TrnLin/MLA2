# Rare Usage gaps and 130 replacements

The new dataset is **`data/processed/teacher_plus_rare_usage_v3_20260906/splits.csv`**.
It replaces 130 of the 687 outside images. All 38,612 teacher rows and all 557 retained
outside rows keep every saved field, ID, partition and fold. The old datasets and image
files remain available. There are still 39,299 total rows and the same nine Usage classes.

Open [photo_review.html](photo_review.html) to inspect all 130 new and 130 retired photos,
with source links, fold numbers and reasons. “Retired” means omitted from v3, not deleted.
The pictures are the exact RGB 60×80 training files, enlarged for review.

## What changed

| Usage | Replaced | New photos | Why those old photos were chosen |
|---|---:|---|---|
| Smart Casual | 62 | 35 watches, 8 wallets, 4 ties, 3 sandals, 12 shirts | Remove one whole 49-image shoe family, 9 jackets/coats and 4 polos; keep all earlier shirts and trousers and 38 ambiguous formal/casual shoes |
| Party | 44 | 18 clutches, 8 perfumes, 6 watches, 12 dark mini dresses | Remove 32 teacher-absent form/audience examples and exchange 12 bright/pale gowns or styled dress views; the dress total stays 130 |
| Travel | 17 | 11 shoulder/tote/handbag forms and 6 camera bags | Remove 13 wheeled/hybrid bags and 4 repeated Hugger variants; retain the other family members in their old fold |
| Home | 7 | 7 printed/patterned covers, including set/pair presentations | Exchange 7 single rectangular pillow views for photos closer to the one teacher Home example |

Outside totals stay Home 136, Party 224, Smart Casual 162 and Travel 165.
The 130 replacements form 116 conservative product families, with at most 4 new images
per family. All outside data now has 597 families, up from 549. The largest old family
falls from 49 to 23 images; the remaining 23-image Hugger family is still a limitation.

The Smart selection also includes women's shirts, watches and sandals, absent from the
earlier additions. Audience is recorded only where source text supports it. Some watch
tags conflict or are broad, so a whole visual subgroup is not declared solved.

## What the audit found

The audit covers all 687 old outside images, all nine teacher development Usage classes,
every one of the 21 rare Usage × articleType combinations, source concentration, product
families, all five folds, source evidence, audience, colour and photo presentation. All
rare teacher contact sheets, old additions and selected replacements were visually reviewed.
The detailed evidence is in [audit/README.md](audit/README.md),
[audit/AUDIENCE.md](audit/AUDIENCE.md), and [type_coverage_before_after.csv](type_coverage_before_after.csv).

1. **Missing forms:** no Smart watches/wallets/ties/sandals; no Party clutches/perfumes/watches;
   thin Travel camera-bag/handbag coverage. The new intake adds these forms.
2. **Photo and audience mismatch:** most new Party dresses were bright long gowns or
   lifestyle views; Smart support was overwhelmingly men's products; Home was mainly
   single cushions. The swaps reduce these gaps, without proving that style defines Usage.
3. **Related images gave less variety than the count suggested:** all 49 examples in one
   Smart shoe family occupied fold 0. That whole family is retired. Families are never
   split across folds merely to improve counts.
4. **Teacher label noise changed the sourcing target:** Smart ID 2633 is a closed men's
   shoe labelled Heels; Travel ID 12348 is a folded shirt with a placeholder name;
   Travel IDs 35828/39487 are camera bags/pouches. The audit does not source high heels,
   folded shirts or phone-only cases to imitate those noisy labels. Teacher labels stay fixed.
5. **Usage meaning overlaps:** watches, shoes, shirts, wallets and perfumes occur in other
   Usage classes. Retailer Party/Smart Casual/Travel text is evidence of intended use, not
   proof of agreement with the teacher catalogue. Count increases alone cannot solve this.
6. **The teacher examples are very small:** development has Smart 47, Travel 22, Party 12
   and Home 1. All rare classes except Home appear in all five teacher folds. The one Home
   example is in fold 4, so that fold has no teacher Home training example. Outside photos
   cannot create an independent teacher Home evaluation population.

## Limits that remain

- The women's Smart shirts added here are mostly brown/green isolated blouses; teacher
  examples are white, often modelled shirts. Women's sporty-watch shapes and Travel's
  women's outdoor-handbag/pink-backpack subgroup remain thin. No person-level gender
  inference is used, and no gender training labels are added.
- Party source meaning is broader than the teacher's all-Women development subset:
  three new perfumes say Man, one says Unisex, and three watches are marketed Girls.
  These have Party evidence but do not establish women's subgroup coverage. Four clutch
  mappings use Evening/Night Out rather than literal Party. Smart includes Business Casual
  and collection-level mappings; two Timex mappings are weaker collection extrapolations.
- Brands remain tied to particular product types and Usage labels. Most clutches come
  from Olga Berg, perfumes from Bella Vita, and new cover photos from Swayam. More sources
  and independent families do not remove this source bias.
- Frozen old folds and whole-family assignment limit per-type balance. Smart watches
  distribute 21/5/5/1/3 across folds 0–4; Party perfumes distribute 1/0/1/0/6. Each fold's
  training side still contains examples of every new form, but this is sparse in places.
  See the dataset's `new_type_fold_counts.csv`. Do not treat external validation gains as
  teacher gains.
- Final prepared-photo review found no blockers, but five items have small product area
  or low contrast and one wallet has an indoor background. See
  [final prepared QA](smart_casual/final_prepared_visual_qa.md). The fixed letterbox transform
  is used for every image; there are no custom crops or generated product photos.
- Product-type/audience counts are audit annotations, not new articleType/gender targets.
  Formal/casual shoe counts cannot be allocated cleanly between those teacher labels.
  Older retailer rights limits remain. New public product photos have no established open
  redistribution licence; source receipts preserve this status. No public upload was made.

## Validation

- Compared all 130 selected photos against 44,441 teacher-role images plus all 687 old
  outside images, including retired candidates. The comparison used image identity,
  resized and foreground views; protected teacher labels were not used. No overlap was
  accepted under the saved rules. Within-intake and source-family checks also passed.
  These checks reduce duplicate risk; they cannot prove every related image was found.
- Every one of the 39,299 final images loaded through the normal RGB 60×80 loader and
  matched its saved file hash. All five train/validation boundaries have zero shared IDs,
  hashes, duplicate groups, normalized names, product families or extension families.
- Teacher and retained outside rows match v2; original split and label-map hashes match.
  New IDs start above every earlier ID, including retired IDs. New rows have Usage only;
  other targets are blank and masked off. Literal NA remains a class with no outside additions.
- Relevant replacement/extension tests: **18 passed**. Code lint passed. No model or learned
  preprocessing was fitted; no accuracy gain is claimed. [validation.json](validation.json)
  records the checks and frozen hashes.

## Use and reproduce

Select v3 explicitly in a future training recipe:

```python
from fashion.data import get_cv_split, load_splits

splits = load_splits("data/processed/teacher_plus_rare_usage_v3_20260906/splits.csv")
train, validation = get_cv_split(splits, 0)
```

Fit preprocessing/weights on the training side only, train from scratch, and register
the new split hash. Compare against the existing v2 baseline using the same teacher
validation IDs. Do not use holdout/test labels to choose more replacements.

The local `teacher_plus_rare_usage_v3_delta.zip` contains the new prepared images and v3
manifests (about 10 MB). Extract it at the repository root alongside the existing
`MyDrive/MLA2/data/task3-data.zip` teacher archive and the existing
`teacher_plus_rare_usage_v2_training.zip` (or `teacher_plus_rare_usage_v2.zip`).
It reuses those images and does not replace their files. Source originals remain in the
local intake, not the training delta. No new full dataset upload is needed.

`audit_collection.py prepare`, `audit_collection.py audit`, then `audit_collection.py finalize`
reproduce the admitted manifest from frozen selected inputs and source bytes.
`select_retirements.py` recreates the proposed retirement sheet and resets its review status;
visual review must be recorded before `build_dataset.py` accepts it. Run each with
`./.venv/bin/python reports/task3/rare_replacement_20260906/<script>`.
`package_review.py` rebuilds the coverage tables, offline gallery, figure and delta ZIP.
The original audit's 128-row shortlist and quotas are advisory history; **the final 130-row
`retirement_review.csv` and accepted manifest are the authority**.
