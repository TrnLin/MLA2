# Training package verification

- Focused trainer, earlier expansion and baseline contract tests: **30 passed**.
- Planned notebook inventory check: **1 passed**.
- Ruff format, Ruff check and `git diff --check`: passed.
- Actual notebook result cells ran against a small, synthetic CPU fit.
- Local Colab setup simulation passed against the delivered ZIP, including all ten
  reference folds, training arguments, repeat setup and archive path checks.
- The package builder decoded and checked hashes for all 33,459 Usage development
  images. Teacher, earlier expansion and new split hashes stayed fixed.
- The notebook HTML and five-fold plot layout were visually inspected. Plot layout
  used earlier saved histories only; it is not a new training result.
- Full GPU training has not started. The local Colab check stopped before fitting;
  it does not verify GPU execution or Google Drive connectivity.

See `bundle_receipt.json` for the final ZIP hash and `colab_bootstrap_check.json`
for the exact setup check. The ZIP includes identical Python code and tests; its
notebook JSON is reformatted but has the same content and empty outputs.

Test commands used `./.venv/bin/python`. The local CPU Torch installation was
provided through `PYTHONPATH=/tmp/mla2-mixup-torch:src` for the training checks.
