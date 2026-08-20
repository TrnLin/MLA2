# 0004 — Product-group leakage check

- Status: Proposed
- Date: 2026-08-20

## Context

The shared split keeps byte-identical SHA-256 groups together, but that does not
prove that different images of the same product or product family stay together.
The current metadata has not yet been validated as a reliable product-group key.

## Decision

Keep Phase 1 open. Before model training, inspect candidate product relationships,
choose a defensible grouping rule, and add a split-level leakage check. Do not
change the shared split until that rule and its evidence are accepted.

## Why

An untested proxy could merge unrelated products or miss genuine variants. The
remaining risk should be explicit while the current exact-duplicate protection
continues to provide a narrower, verified guarantee.

## Consequences

The current split supports EDA and pipeline verification but is not yet approved
for model comparison. Model training remains gated on accepting and testing a
product-group policy, followed by rebuilding downstream evidence if the split changes.

## Evidence

- Repository requirement to control product-group leakage.
- Current `fashion.data.splits` validation covers exact hashes, not product groups.
- Product metadata needs an evidence-backed grouping analysis before it becomes a hard rule.
