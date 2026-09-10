# Integration with the latest main branch

The reviewed Task 3 head was `20af25bbcc3f7ad755a62eeb65394292a6544cd2`.
Main advanced to `0b1d69a60069d5f150e80c9f4c16777a412f9dcb`, adding the
Task 2 and Task 4 implementations. The merge retains their code, reports,
dependencies and notebook contracts alongside the completed Task 3 work.

## Shared contracts

- `fashion.train.metrics` retains both tasks' metric functions. Task 2's
  scikit-learn confusion function has an explicit alias so Task 3's fixed-class
  function does not replace it.
- `fashion.train.registry.RunRegistry` retains Task 2's `append/finalize/read`
  interface and Task 3's `start/update/complete/fail` interface. Task 4 keeps
  its separate registry view. The union schema accepts earlier task schemas
  and preserves other tasks' rows during writes. Task 3's lifecycle lives in
  `task3_registry.py`; a Task 3 update cannot change another task's row.
- The shared ledger contains 27 existing Task 4 rows and 411 existing Task 3
  rows. Every original field value is preserved. The original Task 3 CSV bytes
  remain in the private, hash-checked `results/task3/registry_before_main_merge.csv`.
  Dated report snapshots are unchanged. No new training runs were created.
- Training exports remain lazy, as required by main, with Task 3 exports added.
- The shared environment keeps main's pins and Task 3's Kymatio dependency.
  Historical model environments remain recorded in their original receipts.
- Pytest uses importlib mode to separate equal test filenames across tasks.
  Existing Task 3 fixture imports retain their two test-directory search paths.
- Notebook tests retain the completed Task 2 and Task 4 checks and the completed
  45-section Task 3 checks. Only obsolete Task 3 scaffold placeholders are replaced.

## Frozen evidence

The main Task 3 notebook, its saved outputs and its 81 embedded charts are
unchanged by this merge. All 40 training companions remain unchanged.
The frozen E1 manifest and evaluated inference archives are unchanged.
`scripts/verify_task3_usage_manifest.py` still verifies all 38 frozen references
and both adapted scripts separately. The private-asset verifier still checks
all 7,229 assets, using the updated path for the original registry snapshot.

No training or new model inference is part of this integration. Tests use
their own fixtures; no source checkout or remote asset store is changed.

## Integration checks

- All 38 frozen E1 references and both adapted scripts pass the hash verifier.
- All 7,229 private Task 3 asset hashes pass, including the original registry snapshot.
- All 438 combined registry rows preserve their original field values.
- The affected suite passed 751 tests and found one obsolete documentation check
  that read a planning file intentionally removed from main. That check now reads
  the accepted public decision; all 70 final documentation/notebook checks pass.
- All 56 final registry checks pass, including mixed-task writes, mirrors,
  duplicate IDs, terminal rows and protection against cross-task updates.
- Ruff passes for the changed Python files. Project TOML parses.
- A broader run passed 692 tests and failed one Task 2 test because the private
  `tmp/task2/checkpoints/g3-c2-t0-resnet18-f0-s2753-66ee7a85d5c6.pt` checkpoint
  is absent. The run was interrupted during unchanged Task 4 deployment fixtures
  after 7 minutes 44 seconds. This was not a full-suite pass; later tests were
  not reached. No checkpoint was rebuilt to satisfy that test.

The earlier passing runs used the existing `./.venv/bin/python` with
`PYTHONPATH=src:/tmp/mla2-mixup-torch`. The extra directory contains a real
PyTorch 2.11.0+cpu installation, including its compiled CPython 3.14 extension;
it is not installed in the base virtual environment. No dependency reinstall
or training was performed during integration.
The test environment is Python 3.14.7, NumPy 2.5.2, pandas 3.0.5 and SciPy
1.16.3. A fresh install with main's SciPy 1.18.1 pin has not been tested here.

## Registry ownership correction

The initial merge still let Task 2 interrupt a running Task 3 row and let
Task 4 append a row labelled Task 3. The earlier test attempted interruption
only after completion and therefore missed the running-row bug.

Task 2 read, append, finalize and interrupt now require `task2` ownership.
Task 4 validation again requires exactly `task4`. New regressions exercise
the running Task 3 case with a mirror and verify that neither CSV changes.
They also check Task 2 append/finalize with wrong task labels and Task 4
append with wrong labels against both missing and existing files. Nine of
these cases failed before the fix; all 13 now pass. Existing legitimate
Task 2 lifecycle and Task 4 checks remain enabled.

Current validation, each in a fresh process from the repository root:

```bash
PYTHONPATH=src FASHION_PROJECT_ROOT="$PWD" ./.venv/bin/python -m pytest tests/train/test_registry.py tests/train/test_metrics.py tests/test_documentation.py tests/test_task3_frozen_sources.py tests/test_task3_layout.py -q
```

This base-environment command gives **116 passed, 4 failed**. All four failures
are `ModuleNotFoundError: torch`: the parent Torch-module preservation test,
both tracked-run lifecycle tests, and the duplicate/final-rewrite test.
No test was skipped, weakened or supplied with a fake dependency.

```bash
PYTHONPATH=src:/tmp/mla2-mixup-torch FASHION_PROJECT_ROOT="$PWD" ./.venv/bin/python -m pytest tests/train/test_registry.py -q
PYTHONPATH=src:/tmp/mla2-mixup-torch FASHION_PROJECT_ROOT="$PWD" ./.venv/bin/python -m pytest tests/train/test_registry.py tests/train/test_metrics.py tests/test_documentation.py tests/test_task3_frozen_sources.py tests/test_task3_layout.py -q
```

With the separate real Torch installation, the independent registry run gives
**69 passed** and the focused suite gives **120 passed**. Its import path was
verified as `/tmp/mla2-mixup-torch/torch/__init__.py`, with native extension
`torch/_C.cpython-314-x86_64-linux-gnu.so`; a tensor sum also succeeded.
This temporary dependency path is local test setup, not a portable installation
instruction or evidence that main's Torch 2.13.0 pin was tested.

Ruff and the focused diff check pass. The unchanged E1 manifest still verifies
all 38 frozen references and both separately pinned current scripts. No
training, model inference, registry-data changes or private checkpoint rebuild
was performed for this correction.
