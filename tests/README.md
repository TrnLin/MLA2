# Tests

Tests use small temporary teacher datasets. They do not copy or change the supplied raw
dataset.

The suites check the sole split, deterministic five-fold allocation, family and duplicate
isolation, sealed protected targets, cache validation, exact artifact hashes, notebook
structure, saved HTML, task scaffolds, and active documentation.

Pipeline fixtures also rebuild without `data/raw/external/`. This proves shared
preparation is teacher-only. A protected-label sentinel test changes sealed target
values and proves that public folds, cache, label maps, tables, and figures do not change.
