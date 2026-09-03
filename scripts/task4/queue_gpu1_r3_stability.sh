#!/usr/bin/env bash
# R3 stability folds on GPU 1: fold 2 retry, then folds 3 and 4.
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
    --gpu 1 \
    --phase-request "${REQ}/${name}-request.json" \
    --budget-gate "${GATE}" \
    --phase-output "${OUT}/${name}-result.json" \
    --registry results/runs.csv
  echo "=== $(date -Is) finished ${name}"
}

run_fold task9-stability-r3-f03-retry1
run_fold task9-stability-r3-f04
run_fold task9-stability-r3-f05

echo "=== $(date -Is) GPU1 R3 stability queue complete"
