# Verification

- 38 focused tests passed: the new v3 runner and archive setup, existing Usage MixUp + SAM,
  v2 training, SAM math and shared baseline contracts. One old notebook-output assertion
  was excluded because that earlier notebook contains user-saved training results.
- The planned-notebook inventory check passed separately: **39 passing checks total**.
- Tiny CPU fixtures performed actual fresh fits and verified both gradient passes, v3
  data routing, reference selection, completed-run reuse, changed outside IDs and corrupt
  receipt rejection. Loading old model weights was explicitly blocked in these tests.
- The new notebook is valid, all six code cells compile and all outputs are empty.
  Its HTML render was visually inspected for readable instructions and settings.
- Ruff formatting and lint passed for all five new/changed Python implementation and
  test files. The shared notebook-inventory file has only its new allowed-name line staged.
- The real-data preflight decoded/hash-checked all 33,459 eligible development images and
  verified the original v2 MixUp + SAM fold-0/fold-4 artifacts and training receipts.
  `preflight.json` records the frozen dataset and reference identities.

Tests used the project's `./.venv/bin/python` with the existing local CPU PyTorch dependency
directory added to `PYTHONPATH`. No Colab GPU session or production training was started.
The data delta remains a local/private-Drive artifact; public Git contains code and metadata.
