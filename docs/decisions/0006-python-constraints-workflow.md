# 0006 — Python constraints workflow

- Status: Accepted
- Date: 2026-08-21

## Context

Unpinned package resolution can change calculations, notebook rendering, and
tests between machines.

## Decision

Support CPython 3.12 through 3.14. Pin direct packages in `pyproject.toml` and
resolve the complete CPython 3.12 environment in
`requirements/constraints-py312.txt`.

Install with:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -c requirements/constraints-py312.txt -e ".[dev]"
```

Normal wheel installs resolve the checkout from the current directory when
possible. Set `FASHION_PROJECT_ROOT=/absolute/path/to/MLA2-eda` when running from
elsewhere; repository data is intentionally not bundled inside the wheel.

When a package changes, resolve the whole constraints file in a clean Python 3.12
environment, run `pip check`, the full tests, and notebook rendering, then review
the diff. Do not hand-edit only one transitive package without resolving again.

## Why

This keeps `pyproject.toml` readable while making the tested environment exact.
Python 3.12 is a stable common baseline; 3.14 remains accepted for the current
local environment.

## Consequences

The constraints file is authoritative for reproducibility checks. Other Python
minor versions may need a separately reviewed constraints file if their wheels
differ.

## Evidence

- A clean CPython 3.12.13 editable install resolves every pinned package.
- The clean environment passes `pip check` and the full test suite.
- A normal, non-editable wheel import resolves `fashion` from `site-packages` and
  finds the checkout both from its working directory and from
  `FASHION_PROJECT_ROOT`; the wheel environment also passes the full suite.
- Runtime versions and the constraints-file hash are recorded in EDA provenance.
