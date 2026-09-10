# 0017 - Product-name `NA` and development CV refreeze

- Status: Accepted
- Date: 2026-08-23
- Amends: the product-name rule in 0011 and the frozen fold assignment in 0014

## Context

The teacher CSV uses the literal token `NA` in two different ways. It is a valid `usage` target
needed by the teacher vocabulary, but it means missing data in `productDisplayName`. Treating it as
a real product name joined seven unrelated Clutches and Perfume and Body Mist rows. One of those
rows also has a real exact/perceptual duplicate link to a named product, so only that visual link
should remain.

Changing this rule changes the atomic family contract. Keeping the old CV folds would silently
preserve an assignment made for the wrong family graph.

## Decision

- Treat blank values and the case-insensitive literal `NA` as missing only in the product-name
  repair and family-name normalization path.
- Continue to treat `usage="NA"` as a valid target label.
- Keep the existing development, holdout, and quarantine IDs unchanged.
- Re-freeze all five development folds once, using the existing seed `2753` and the repaired family
  graph, before any training run.
- Require the explicit `refreeze_development_folds=True` flag when a rebuilt development family ID
  differs from the canonical split. A normal rebuild preserves existing folds.

## Recorded transition

- Development families: `22,899 -> 22,905`.
- Development family IDs changed for eight rows: the seven missing-name rows plus the named product
  that remains visually linked to one of them.
- Development fold assignments changed for `26,053` rows because the deterministic group allocator
  was run again from the repaired family graph.
- Old CV assignment SHA-256:
  `53dc9fee3b56c8dc7bf39701507c3a3406b8d43b55408250d595e9e2428c822b`.
- New CV assignment SHA-256:
  `bad7bc4ae65fbbfd815567f4ccfa308d6e57dc650bc15c0b8e798867a335f2fd`.
- Development, holdout, and quarantine ID-set SHA-256 values did not change.
- Active partition crossings and development fold crossings remain zero.

## Consequences

All task notebooks must consume the new `cv_fold` values from `data/processed/splits.csv`. No old
fold number may be cached elsewhere. `usage="NA"` remains in `label_maps.json`; it is a teacher
vocabulary token, not a claim that `NA` is a meaningful real-world occasion.

The split summary, CV summary, taxonomy, preparation cache, notebook evidence, artifact registry,
and saved HTML are regenerated from the repaired contract.
