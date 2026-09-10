# Usage investigation

Read [REPORT.md](REPORT.md), or open the rendered [REPORT.html](REPORT.html).

- Checked 13 Usage experiments and 53 fold runs.
- Found that U2 calibration lowers raw-margin F1 from **0.4059 to 0.3629**. The base models also still overfit.
- A fixed E2/E3/E8 probability average reaches **0.4231**, versus E2's **0.4082**, on all five folds. This small gain does not establish a fix for overfitting; the user has declined it as the next test.
- The local high resolution archive has **zero new product IDs** outside the teacher roles.
- The proposed two-stage CNN has since run as U3 and failed its two-fold screen. See the [U3 review](../usage_u3_review_20260905/README.md), including its missing fold-4 completion record.
- Next research direction: check independent data with existing Usage or occasion labels. Verify label fit, rights and teacher-data overlap before a training plan. No compatible new dataset is confirmed yet.
- No step needs human image labels. No training, large download, commit or branch change was made.

Exact run IDs: [RUNS.md](RUNS.md). Inactive historical average plan: [next_experiment_plan.json](next_experiment_plan.json). Source and licence notes: [SOURCES.md](SOURCES.md). Checks: [verification.json](verification.json).

Reproduce from `/home/dinhquan/personal/academic/RMIT/Machine-Learning/MLA2-eda`:

```bash
./.venv/bin/python reports/task3/usage_deep_investigation_20260905/analyse_evidence.py
./.venv/bin/python reports/task3/usage_deep_investigation_20260905/diagnose_u2.py
./.venv/bin/python reports/task3/usage_deep_investigation_20260905/diagnose_decisions.py
./.venv/bin/python reports/task3/usage_deep_investigation_20260905/diagnose_data.py
./.venv/bin/python reports/task3/usage_deep_investigation_20260905/summarise_diagnostics.py
./.venv/bin/python reports/task3/usage_deep_investigation_20260905/audit_eda_probes.py
./.venv/bin/python reports/task3/usage_deep_investigation_20260905/render_report.py
```

The first six scripts use saved evidence and development images only. They fit no models. The final script builds the readable run ledger and HTML report. The current local environment lacks PyTorch. The average's corruption inference was not run and is no longer the proposed next test.
