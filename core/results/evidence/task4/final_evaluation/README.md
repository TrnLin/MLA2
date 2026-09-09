# Local Task 4 ranking artifact

GitHub cannot store `holdout_primary_rankings.csv` because it is 292 MB.
The MR therefore leaves that one file local and ignores its path.

To run the full artifact audit or replay Notebook 06, place the original file at:

`results/evidence/task4/final_evaluation/holdout_primary_rankings.csv`

Expected identity:

- Rows: `3,813,480`
- Bytes: `292,330,199`
- SHA-256: `ed8e5864417b25450627b426ce3017aa6f1b947db147ca845cb7aadcd12e99f2`

`prediction_receipt.json` is the authoritative record. The committed scorecards,
figures, receipts, and HTML report were produced from this frozen local file.
