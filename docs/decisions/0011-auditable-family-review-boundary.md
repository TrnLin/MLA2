# 0011 — Auditable family review boundary

- Status: Accepted
- Date: 2026-08-21

## Context

The existing near-duplicate and product-name ledgers preserve useful pairs and
provisional calls, but do not contain real reviewer provenance. They cannot prove
human review precision. Leakage safety must not depend on those unsigned calls.

## Decision

- Mark every existing ledger row `pending_team_signoff`; never invent reviewer
  metadata.
- Quarantine every pending labelled-versus-prediction automatic visual match.
  Only a fully signed `different` or `uncertain` decision may reactivate it.
- Accepted labelled-only automatic matches may enlarge a split block. A mistaken
  acceptance can reduce sample size but cannot create a crossing.
- Keep normalized names as conservative split blocks only. Mixed labels inside a
  name block are triage evidence, not proof of label noise.
- Require reviewer initials, date, method/tool, blind-to-metrics flag,
  independence flag, second-review status, and disagreement resolution before
  calling a row signed. The exact procedure lives in `docs/reviews/README.md`.
- Fix IDs `41303/43587` as one train-only evidence family before allocation. The
  pair was named externally as the required visual-only Summer/Fall example; the
  constraint prevents EDA from selecting or displaying a validation/holdout
  outcome after splitting.

## Consequences

All active exact-SHA, accepted-visual, and normalized-name blocks must remain in
one partition. Pending human sign-off is an honest evidence limitation, but it
cannot leak a protected or prediction image into model development.

## Evidence

- All 250 existing review rows are explicitly `pending_team_signoff`; reviewer
  identity, date, method, independence, and second-review fields are blank.
- The blind contact sheet lists every ledger pair by ID and image while hiding
  labels, metrics, notes, and provisional decisions.
- All 10 non-exact cross-role rule matches remain quarantined while unsigned.
  The final split has 61 quarantine rows.
- The final active split has zero exact-SHA, product-family, and normalized-name
  crossings. Its counts are 26,992 train, 5,781 validation, and 5,778 holdout.
- Of 568 mixed-season multi-row train groups, 567 involve normalized-name
  blocking. The sole visual-only mixed group is IDs `41303/43587`, joined by the
  objective accepted-near rule; its human interpretation is still pending.
