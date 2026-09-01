# Task 1 report: deterministic grayscale HOG extraction

## RED evidence

Command: `./.venv/bin/python -m pytest tests/task1/test_classical.py -q`

Result: collection failed with the expected `ModuleNotFoundError: No module named 'fashion.task1.classical'`.

## GREEN evidence

Commands:

```text
PYTHONPATH=src ./.venv/bin/python -m pytest tests/task1/test_classical.py -q
....                                                                     [100%]
4 passed

PYTHONPATH=src ./.venv/bin/python -m ruff check src/fashion/task1/classical.py tests/task1/test_classical.py src/fashion/config.py
All checks passed!
```

## Files changed

- `src/fashion/task1/classical.py`: frozen HOG specs and deterministic grayscale extractor.
- `tests/task1/test_classical.py`: fixed IDs, widths, dtype, determinism, and validation tests.
- `src/fashion/config.py`: ignored processed HOG cache path.
- `pyproject.toml`: pinned `scikit-image==0.26.0`.
- `requirements/constraints-py312.txt`: pinned `scikit-image==0.26.0`.

## Self-review

The extractor uses the existing letterbox transform at fixed 80x60, converts to grayscale, and returns finite float32 vectors. Geometry validation computes the expected HOG width and rejects mismatches. No split, holdout, augmentation, scaling, PCA, color HOG, or pretrained model code was added.

## Concerns

The worktree virtual environment imports the separately installed main checkout by default, so test commands need `PYTHONPATH=src` to exercise this worktree. The brief's unqualified command therefore reports the missing module until the package is installed editable from this worktree.
