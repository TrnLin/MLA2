#!/usr/bin/env bash
# R5 stability folds on GPU 0: fold 1 retry, then folds 2, 3, 4.
# Stops at the first failure so a systematic problem does not waste the queue.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=./.venv/bin/python
RUNNER=scripts/task4/run_model_comparisons.py
GATE=results/evidence/task4/model_comparison_budget_gate.json
REQ=results/evidence/task4/phase_requests
OUT=results/evidence/task4/phase_results

run_fold() {
  local name="$1"
  echo "=== $(date -Is) starting ${name}"
  "${PY}" "${RUNNER}" \
    --phase stability \
    --gpu 0 \
    --phase-request "${REQ}/${name}-request.json" \
    --budget-gate "${GATE}" \
    --phase-output "${OUT}/${name}-result.json" \
    --registry results/runs.csv
  echo "=== $(date -Is) finished ${name}"
}

run_fold task9-stability-r5-f02-retry1
run_fold task9-stability-r5-f03
run_fold task9-stability-r5-f04
run_fold task9-stability-r5-f05

echo "=== $(date -Is) GPU0 R5 stability queue complete"
