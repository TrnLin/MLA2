# Rare Usage replacement audit

**Audience supplement:** see `AUDIENCE.md` and `audience_gap_table.csv`, refreshed after the women's substitution. Smart now has evidence-supported women's shirts, watches and sandals; colour/photo gaps and some conflicting watch-audience tags remain. Party includes some men's perfumes/Girls watches despite all teacher Party development rows being Women. No gender target changes.

Read-only audit of all 687 external rows and teacher **development only**, from `data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv`. CSVs were read with `dtype=str, keep_default_na=False`. No holdout/test labels were used. No training data, labels, folds, gallery, or model files were changed.

## Correction review and advisory shortlist

`teacher_tiny_type_review.csv` reviews all 30 teacher images in rare type groups of four or fewer. Smart ID2633 is a men's closed leather shoe, despite its Heels label: **do not source high heels**. Travel ID12348 is a folded shirt with placeholder name: exclude it from bag sourcing rationale. Travel IDs35828 and39487 are camera pouches/bags, despite Handbags/Mobile Pouch labels: source small camera bags, not phone-only pouches. Party ID45340 is a heeled sandal; its Heels label is compatible with the Sandals title. All teacher labels remain unchanged.

`retirement_shortlist_128.csv` and its four class files propose exactly Party38, Smart60, Travel20, Home10 retirements. Every selected family is retired whole. All existing source counts survive. Smart's whole49-image family is included, leaving **zero existing external Smart rows in fold0**: retirement must wait until admitted independent replacement families restore fold0 coverage. No existing row is reassigned to another fold. Travel's27-image family is retained because it exceeds the20-row budget; this concentration remains unresolved. The shortlist protects all Party dresses/tops/heels, all Smart shirts/trousers, and both Home sources. Retirement is conditional on replacement acceptance, not an instruction to modify the dataset now.

Run `build_audit.py` followed by `correction_review.py` to reproduce the corrected gap table and advisory shortlist. That128-row shortlist does not govern final v3. The final current plan is130 replacements: Smart62, Party44, Travel17, Home7. Final retained counts must come from the actual retirement receipt; the audience audit labels its128-row retained projection ADVISORY.

## Main finding

More examples did not fill several teacher product-type gaps. Source labels also use a broader meaning of Usage than the teacher catalogue. A clean photo and an exact retailer tag do not prove agreement with the teacher labels.

| Usage | External rows | Conservative families | Largest family | Main gap |
|---|---:|---:|---:|---|
| Home | 136 | 134 | 2 | Teacher image has a group/set presentation, light background, multicolor print; most additions show single cushions |
| Party | 224 | 216 | 3 | Zero clutches, perfumes, watches; old intake includes14 bras |
| Smart Casual | 162 | 83 | 49 | Zero watches, wallets, ties, sandals; Heels is a noisy teacher type, not a true gap |
| Travel | 165 | 116 | 27 | Small camera-pouch/bag gap; broad tote/weekender group does not fill outdoor-handbag styles |

`rare_type_gap_table.csv` covers every rare teacher Usage × articleType pair. Counts for external types are **audit annotations**, never training targets. Formal/casual shoes cannot be allocated reliably:87 Smart shoes are possible matches for both types, not174 distinct images. Home has30 explicit cover titles and106 pillow titles with cover status unspecified in the title; this does not prove106 filled pillows. Descriptions need review before any semantic rejection.

## Teacher limits and nine classes

Valid teacher development counts: Casual25,151; Sports3,346; Ethnic2,183; Formal1,949; NA61; Smart Casual47; Travel22; Party12; Home1. One further development row has blank Usage and is excluded from the nine-class summary. Literal `NA` remains a class; no new external NA is proposed.

Casual spans many types, including1,837 watches. Sports is led by1,663 sports shoes and809 Tshirts. Ethnic has1,318 kurtas. Formal has723 shirts and498 formal shoes. NA includes17 shoe accessories,9 perfumes,8 nail polishes and7 wallets. These overlaps mean a product type cannot determine Usage. Full counts are in `teacher_type_counts.csv` and `teacher_cross_class_type_counts.csv`.

All classes except Home occur in all five teacher folds. Rare fold0–4 counts: Party3/2/2/2/3; Smart9/10/9/10/9; Travel4/5/4/5/4; Home0/0/0/0/1. Teacher Home therefore has no training example when fold4 is validation. Adding outside data cannot create an independent teacher Home test population. One-example types do not establish a stable class definition.

## Sources, families and folds

Home has120 ABO +16 Amazon Reviews. Party has97 Flipkart +60 Beginning Boutique +48 Meshki +19 Red Dress. Smart has84 ABO +78 M&S. Travel has52 Dbjourney +46 ABO +27 Eagle Creek +19 Cotopaxi +13 Dakine +4 Topo +3 Tom Bihn +1 Flipkart. Counts and exact identities are saved separately.

External fold0–4 counts: Home28/27/27/28/26; Party45/45/45/45/44; Smart49/28/29/28/28; Travel33/32/34/33/33. Smart fold0 is one49-image conservative family. This is concentrated support, not49 independent styles. Preserve saved folds of retained rows and never split this family to make fold counts look balanced. Travel also has a27-image family. Product titles and visual groupings can join styles conservatively; large families are not automatically duplicate errors.

## Evidence and photo review

Inspected all four new prepared contact sheets, all four teacher-development sheets generated here, and the complete original120 contact sheet generated here. Also inspected old Flipkart's four source sheets and Amazon's three source sheets, which include candidates that were not admitted. Retained-row claims use the admitted manifest, not raw sheet totals.

New Party127 are all modelled dresses, often with lifestyle backgrounds, bare skin, long gowns, bright colours or pale studio scenery. Teacher Party dresses are mostly short black/dark catalogue views on light backgrounds. Old97 Party have3 dresses,22 tops,22 heel/wedge titles and50 other-type titles. The50 include14 bras,16 shirts,10 other shoes,3 flats, and7 singletons (blazer, cap, kurta, lehenga, saree, suit, Tshirt). These are poor type coverage matches, not proven wrong labels. Heel/wedge titles include visibly low or flat sandals; titles do not guarantee teacher-type agreement.

Smart additions are mostly isolated footwear and folded/isolated menswear; many teacher shirts/trousers are modelled. There are87 shoes with uncertain formal/casual assignment,5 boots,20 trousers,12 shirts and38 other titles. The teacher item labelled Heels (2633) is named Carlton London Men Black Leather Casual Shoes and looks like a closed men's shoe. Preserve its label, but source no high heels for this spurious gap.

Travel has91 backpack/travel-pack titles,41 duffels,13 rucksack/hiking-pack titles,13 wheeled/hybrid luggage titles and7 shoulder/tote/weekender titles. Most are clean product photos. Db colour/view repetition and wheeled luggage broaden the shape distribution. Teacher Handbags includes an erroneous folded-shirt image and a camera pouch, alongside two outdoor handbags. Mobile Pouch is actually named/pictured as a camera bag. Use those corrected visual kinds for sourcing only.

Home teacher ID40826 (`data/raw/teacher/train/images_train/40826.jpg`, fold4) shows a group/set presentation, light background, multicolor print. Resolution is too low to establish whether the products are folded or unfilled. New sheets mostly show single puffed square/rectangular cushions; some sets and one room view remain. A small intake of real cover-set product photos is useful, but one teacher row cannot justify broad Home categories such as sleepwear or bedding.

Evidence levels in external rows: Party97 old exact occasion,79 explicit new source label,48 product-text inference; Smart6 exact style,78 explicit source label,78 product-text inference; Home16 exact occasion,120 product-text inference; Travel1 exact occasion,164 product-text inference. M&S collection evidence is stronger than interpreting semi-formal shoe prose, but may still differ from teacher labels. Generic travel/gym/school descriptions also span multiple purposes.

## Replacement proposal

`replacement_quotas.csv` records an earlier broad **95 Smart,60 Party,20 Travel,10 Home** sourcing pool. It is historical advice, not a command to delete185 rows. The advisory shortlist is60/38/20/10; final current plan is62/44/17/7. Smart's heels quota is zero. Prefer several independent families and sources, with a practical cap around five images per family where possible. Do not invent source variety when only one defensible source exists.

Keep class totals fixed only if each new image passes the evidence, rights, duplicate, visual and family checks. A quota must not force a weak label. Party's 15 new dresses should replace existing dress images rather than increase the dress total. Keep useful old tops and heels while reviewing the 50 teacher-absent title groups first; then review 10 redundant/background-heavy dresses. Travel: review 13 wheeled/hybrid rows, then repeated large-family variants. Home: review title-ambiguous single-cushion variants, but preserve useful explicit covers and set photos. Smart: review low-evidence semi-formal shoes and variants from its large family, plus teacher-absent types. Keep enough varied footwear; do not remove all footwear just because watches are missing.

The four `retirement_<class>.csv` files rank exact admitted IDs with original source/evidence and reasons. Scores are transparent review priorities, **not measured mislabel probabilities**: Party teacher-absent title+70; Smart teacher-absent title+45; Smart inferred label+25; Travel wheeled/hybrid+40; family size≥5 adds min(35,size); new Party dress+15; Home cover-unspecified title+20. Ties use ID order, not a quality difference. Deletions require reviewing the row and its proposed replacement; retain family fold integrity.

## Files and reproducibility

- `external_687_audit.csv`: all external IDs, sources, URLs, label evidence, folds, title types and review scores.
- `rare_type_gap_table.csv`: every rare teacher type, external broad/ambiguous coverage, families, sources and folds.
- `retirement_*.csv`: exact ranked rows per class; `retirement_ranked.csv` combines all four.
- `replacement_quotas.csv`: per-type proposal and evidence gates.
- `teacher_*counts.csv`, `teacher_nine_class_summary.csv`, `external_*counts.csv`: full supporting aggregates.
- Four `*_teacher_development.png` sheets and `original_120_contact_sheet.png`: visually inspected images.
- `build_audit.py`: rebuilds generated CSVs/sheets with `./.venv/bin/python reports/task3/rare_replacement_20260906/audit/build_audit.py`.

No accuracy improvement is claimed. Source shift, conflicting meanings of Usage, tiny teacher classes, and fold-specific missing types remain even after careful replacement. Use the existing teacher evaluation only after freezing choices; do not keep tuning sourcing to heldout errors.
