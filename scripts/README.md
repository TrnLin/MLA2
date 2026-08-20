# Scripts

Put small command-line entry points here.

Scripts should call reusable functions from `src/fashion/` instead of containing
project logic.

- `prepare_data.py` rebuilds audits, manifests, `data/processed/splits.csv`, and
  train-only normalization statistics.
- `generate_eda.py` rebuilds compact evidence and report figures.
- `audit_perceptual_duplicates.py` runs an optional label-free diagnostic and
  never rewrites the shared split.
