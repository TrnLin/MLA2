# Tests

Add automated tests here as reusable code is added to `src/fashion/`.

Use small temporary data in tests. Never change or copy the supplied raw dataset.

Tests are grouped under `data/`, `eda/`, and `retrieval/`. The shared fixtures
create a tiny temporary teacher dataset with missing, duplicate, rare-class, and
cross-role cases. EDA tests execute the official notebook twice in fresh temporary
projects with tiny high-resolution images and compare its stable outputs. A
sentinel test changes sealed target values, regenerates every small public table, and
proves normal EDA state does not change or expose them. Pipeline tests also prove the
lean gzip pack is deterministic, cached validation accepts an unchanged contract, and
a changed artifact is rejected. Source-image tests pin the narrower
path/size-plus-sample guarantee.
