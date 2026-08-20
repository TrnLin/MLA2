# 0006 — Dependency lock pending

- Status: Proposed
- Date: 2026-08-20

## Context

The template defines dependencies in `pyproject.toml` but has no established lock
or constraints-file convention. Editable-install, test, and rendering checks only
prove the package versions installed in the reviewed environment.

## Decision

Keep dependency locking as an explicit collaboration risk. Before shared model
training, choose one team-supported lock or constraints workflow, record it, and
verify installation in a clean environment. Do not introduce a one-off lock format
without team agreement.

## Why

An isolated convention would create competing sources of dependency truth. Leaving
the risk visible permits the team to choose a workflow compatible with every
development environment.

## Consequences

Current checks remain valid for the recorded runtime versions in EDA provenance,
but future dependency resolution may differ until this decision is accepted and
implemented.

## Evidence

- Dependency declarations in `pyproject.toml`.
- Runtime package versions in `results/evidence/eda/summary.json`.
- No existing lock or constraints convention in the team template.
