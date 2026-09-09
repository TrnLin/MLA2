"""Recheck saved G2/E6 evidence; no model fitting or checkpoint inference."""
from pathlib import Path

from fashion.train.task3_g2_audit import audit_g2_confirmation

root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
report = Path(__file__).resolve().parent
result = audit_g2_confirmation(
    g2_directory=report / "drive",
    e6_directory=root / "results/evidence/task3/experiments/t3_gender_e6_gem_p3/gender",
    registry_path=report / "drive/runs.csv",
    report_directory=report,
    root=root,
)
print(result["status"])
for check in result["checks"]:
    if check["status"] != "pass":
        print(check)
