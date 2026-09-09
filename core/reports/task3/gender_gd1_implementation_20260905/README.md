# Gender G-D1 implementation check

Training code is written; GPU prerequisites and training have not run.

- Entry point: `notebooks/task3_training/gender_gd1_mild_darkening.ipynb`.
- Run All checks the saved E6/G2 bundles, GPU/build, numerical parity, memory and timing, then trains only screen folds 0 and 4 if all prerequisites pass.
- Candidate: scratch GeM with the existing two-pixel translation, plus mild darkening with probability 0.25 and brightness sampled from 0.90–1.00 using a separate random generator.
- Confirmation requires a passing screen and a separate call. The held-out test is untouched.
- Upload the changed code or publish it to the selected repository branch before using the notebook's Colab clone setup. Use an NVIDIA L4 and the saved G2 software build; the checks enforce the match.
- CPU offload uses PyTorch's documented [`save_on_cpu`](https://docs.pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.save_on_cpu). Real GPU parity, memory and timing remain unverified until Colab runs.

Validation: full local suite **161 passed, 4 skipped** because local PyTorch is absent. Changed training modules and notebook code cells compile. Ruff and `git diff --check` pass. The ten saved source bundles passed the source audit. New notebook narrative and the updated decision ledger were rendered and visually inspected in `review.html`.

Local PyTorch was removed as requested. No training, commit or push was performed.
