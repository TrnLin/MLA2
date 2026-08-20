# 0001 — Data roles and raw immutability

- Status: Accepted
- Date: 2026-08-20

## Context

The supplied labelled images and official prediction images have different roles.
Raw files must remain reproducible and must not be changed by cleaning.

## Decision

Store local teacher data under `data/raw/teacher/{train,test}`. Treat every raw
file as read-only. Record repairs, masks, hashes, and image checks only in generated
files under `data/processed/`.

## Why

This preserves source evidence, makes every change reviewable, and prevents the
official prediction template from becoming an accidental labelled dataset.

## Consequences

The pipeline needs enough local disk access to read all images. Raw data remains
ignored by Git, and local symbolic links are allowed.

## Evidence

- Assignment dataset roles and fixed prediction schema.
- `AGENTS.md` raw-data and split rules.
- Raw audit digests written by `fashion.data.audit`.
