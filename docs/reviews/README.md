# Image-review ledgers

These files are triage ledgers. Their current calls are **provisional**. No team
reviewer name, date, or independent second review was available, so every row is
marked `pending_team_signoff`. The project does not claim these rows as completed
human review.

The pending status is safe for splitting:

- every pending labelled-versus-prediction visual match is quarantined;
- accepted labelled-only visual matches are used only to make a larger split block;
- normalized names are used only as conservative split blocks, never as proof that
  products are identical and never to relabel an item.

## Files

- `near_duplicate_policy_review.csv` freezes a deterministic, dHash-stratified
  sample of 120 non-exact pairs passing the automatic pixel rule.
- `cross_role_near_duplicate_review.csv` covers every non-exact
  labelled-versus-prediction pair passing that rule. A missing row stops a rebuild.
- `product_name_policy_review.csv` samples 30 normalized-name keys from each of
  four size bands in the pre-policy crossing set. It measures name-block noise.
- `product_name_pre_policy_triage.json` freezes the 2,549 name groups crossing the
  superseded split. It is historical evidence, not another usable split.
- `review_signoff_template.csv` gives the required fields for any new ledger.

`same_or_variant`, `different`, and `uncertain` have their usual visual meaning.
They stay provisional until sign-off.

## Team sign-off procedure

1. Run `notebooks/00_eda.ipynb`. It rebuilds the ID-only contact sheet at
   `results/reviews/review_contact_sheet.html`. Target labels, split names, model
   scores, and aggregate review rates stay hidden.
2. A team member inspects each full image and records `reviewer_initials`,
   `review_date` (`YYYY-MM-DD`), `review_method_or_tool`, and whether the review was
   blind to metrics and independent of the provisional call.
3. A second team member reviews every `uncertain`/`different` call and a fixed
   random check sample of proposed `same_or_variant` calls. Record initials,
   decision, and status. Do not change the sample after seeing agreement rates.
4. Record every disagreement and its resolution. A row with an unresolved
   disagreement stays pending.
5. Change `signoff_status` to `signed_off` only when the audit fields are complete.
   Rebuild all split and EDA evidence after any decision changes.

The ledger validator rejects signed rows with missing reviewer, date, method,
blindness, independence, or second-review status. It also rejects a resolved
disagreement without a written resolution.

## Current provisional result and limits

The stored calls are 119/120 `same_or_variant` for the automatic-rule sample,
6/10 `same_or_variant` for cross-role candidates, and 28/120
`same_or_variant` for normalized-name pairs. These numbers are **not signed human
precision evidence**. They are preserved only so a real reviewer can audit or
replace the calls without losing row identity.

Even signed samples would estimate precision, not recall. Small images and subtle
variants can escape the automatic rule. The normalized-name sample also cannot
prove that all rows sharing a name are one product.
