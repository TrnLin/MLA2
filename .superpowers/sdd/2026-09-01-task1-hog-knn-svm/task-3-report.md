# Task 3 report — KNN and Linear SVM adapters

## RED evidence

Added the required KNN, Linear SVM, and reused-neighbour tests to
`tests/task1/test_classical.py` before changing production code.

Ran:

```sh
PYTHONPATH=src ./.venv/bin/python -m pytest tests/task1/test_classical.py -k "knn or svm or neighbour" -q
```

Result: collection failed with the expected missing-interface error:
`ImportError: cannot import name 'Task1KNNConfig' from fashion.task1.classical`.

## GREEN evidence

Implemented the validated fixed grids, fixed 124-class expansion, stable softmax
for LinearSVC decision scores, exact batched Euclidean neighbour queries, and
scikit-learn-equivalent uniform/distance votes. Distance voting follows the
zero-distance rule. `ConvergenceWarning` is promoted to an error during SVM
training.

Fresh verification after formatting:

```sh
PYTHONPATH=src ./.venv/bin/python -m pytest tests/task1/test_classical.py -k "knn or svm or neighbour" -q
# 8 passed, 10 deselected

PYTHONPATH=src ./.venv/bin/python -m pytest tests/task1/test_evaluation.py -q
# 13 passed

PYTHONPATH=src ./.venv/bin/python -m ruff check src/fashion/task1/classical.py tests/task1/test_classical.py
# All checks passed!

git diff --check
# exit 0
```

## Changed files

- `src/fashion/task1/classical.py`
  - Added `Task1KNNConfig`, `Task1LinearSVMConfig`, approved grids/defaults,
    score helpers, and pure KNN/SVM prediction adapters.
- `tests/task1/test_classical.py`
  - Added tests for fixed-size expanded probabilities and reusable neighbour
    vote equivalence with scikit-learn.

## Self-review

- No split access, PCA, scaling, RBF kernel, or pretrained model was added.
- Every adapter output is finite `(rows, 124)` data, with missing classes kept
  at zero.
- KNN uses brute-force Euclidean distance and validation batches no larger than
  512 in the fit/predict adapter.
- SVM score conversion is a stable softmax for common score artifacts only; it
  does not claim calibration.

## Concerns

No known concerns. The SVM probabilities are score-normalised artifacts, not
calibrated probabilities; callers and report text should describe them that way.
