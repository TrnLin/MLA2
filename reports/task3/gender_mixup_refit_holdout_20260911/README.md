# Selected MixUp refit — final evaluation

This evidence is used only by
`notebooks/07_task3_part2_final_evaluation.ipynb`.
The development notebook contains no holdout references or results.

The scored model is the single scratch MixUp 0.20 refit at epoch 30:
`t3_gender_name_truth_mixup_alpha020_refit_20260911T041436Z_3af0b94e`.
Checkpoint SHA-256:
`860f688162cccfcd903e8e4874b368697c0637b6e6a15baae3b4d4a3008f4ef9`.

The model gets 5,227 of 5,778 original-label images right: 90.46% accuracy
and 78.74% five-class macro-F1. Whole-family bootstrap, 10,000 draws with
seed 2753, gives a 95% F1 interval of 75.30%–81.66%. Unisex recall is 54.98%
(171 of 311). This is a follow-up on previously opened reserved rows, not
a new blind test. No model or threshold was tuned from these results.

`evaluate.py` verifies the exact training artifact and image hashes, saves
a fixed plan, runs one inference pass with unchanged weights, freezes
predictions, then joins original reference labels through the protected
loader. There is no ensemble or comparison-model inference. It refuses
to replace an existing completed evaluation. `executed_source.txt` retains
the exact source used for that pass; the formatted entry point has the
same Python syntax tree. Both source identities are recorded in provenance.

For normal replay, run the final evaluation notebook with the project
`.venv` kernel. It reads the saved predictions through
`fashion.task3_refit_evaluation.load_selected_evaluation`, recomputes the
scores, checks file hashes, and draws the figures. It also reads only the
selected E8 refit's saved Usage predictions. No inference runs during replay.

`selected_refit_bootstrap.json` contains separate single-model intervals for
MixUp and E8. Families are resampled whole; absent classes retain zero F1 in
the fixed class map. These intervals do not cover training-seed variation,
repeated selection, or transfer to other catalogues. Bootstrap is implemented
by `fashion.task3_refit_evaluation.family_score_interval` and checked against
explicit family sampling in the report tests.

The development report audits 501 unique saved run IDs across both targets,
including incomplete attempts, sampled feature probes and refits. Its tables
keep fold scope, label basis and repeated runs separate. Main-notebook replay
was checked with a file-read guard that rejects final-evaluation assets.

Both notebooks execute successfully. The rewritten sections, tables and plots
were rendered and visually inspected. The focused report tests and lint checks
pass. The Gender artifact verifier also checks the selected MixUp checkpoint.
