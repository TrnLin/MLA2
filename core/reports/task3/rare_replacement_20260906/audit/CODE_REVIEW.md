# Replacement code review

Read `src/fashion/data/usage_replacement.py`, `reports/task3/rare_replacement_20260906/audit_collection.py` and their fold/validation dependencies. No production edits made by this review.

## Fixed during review

The initial finalize function recorded but did not verify the overlap-matches CSV hash and did not verify the selection receipt. Stale overlap decisions or a changed selected-candidate CSV could therefore be used without failure. Reported to the parent; re-read confirms checks now exist at audit_collection.py lines168–179 for selection input hashes, prepared hash, overlap matches hash, candidate hash and previous split hash.

## Reproduced edge case: fixed

The earlier implementation passed only retained rows to `extend_usage_dataset`. If a replacement belonged to a wholly retired old family, `assign_extension_folds` had no remaining member from which to recover its frozen fold. A synthetic reproduction retired the sole member in fold1, linked the new image to the same old family, and got fold3.

Re-read confirms `usage_replacement.py` now rejects new links to wholly retired families before extension (lines63–75). Families with retained members keep their existing anchors. A regression test covers this rejection; the final v3 validation passed. This finding is resolved.

## Checks that passed

The initial six replacement tests passed during this audit. After the fix, the parent review reports all18 replacement/extension tests passed, including the new regression test. Assertions preserve retained teacher rows, reserve retired IDs, enforce equal per-Usage counts, restrict additions to development and forbid external articleType/gender/season targets. Teacher overlap references read only image/identity fields across roles; no holdout/test labels enter source-label choices in the reviewed code.

The review is bounded. It does not prove the near-duplicate thresholds detect every visual relative or that retailer Usage prose matches teacher semantics. Those are explicitly separate data limitations.
