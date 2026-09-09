"""Save human-reviewed cushion photos; contact numbers refer to all_downloaded.csv."""

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
frame = pd.read_csv(OUT / "all_downloaded.csv", keep_default_na=False)
excluded = {
    "bedding_or_folded_fabric_without_clear_decorative_cushion": [
        3,
        4,
        6,
        11,
        24,
        30,
        34,
        36,
        37,
        43,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        61,
        65,
        72,
        76,
        80,
        84,
        86,
        87,
        93,
        94,
        97,
        99,
        104,
        105,
        108,
        113,
        116,
        119,
        120,
        123,
        130,
        137,
        144,
        149,
        157,
        158,
        161,
        163,
        170,
        173,
        175,
        179,
    ],
    "sleep_or_therapy_pillow_outside_decorative_cushion_scope": [16, 32, 101],
    "logo_or_placeholder": [35, 142],
    "multi_panel_product_grid": [45, 59, 85, 103, 128, 156],
}
reasons = {number: reason for reason, numbers in excluded.items() for number in numbers}
frame["visual_review"] = frame.contact_number.map(
    lambda n: reasons.get(n, "passed_contact_sheet_review")
)
frame["visual_reviewer_note"] = (
    "All five contact sheets inspected at 1200x1200. Keep decorative cushions/covers/throw pillows "
    "with clear product form; reject bedding, folded fabric, grids and placeholders. "
    "A pack shown as overlapping cushions remains one image."
)
frame.to_csv(OUT / "visual_qa.csv", index=False)
frame.loc[frame.contact_number.isin(reasons)].to_csv(OUT / "rejected.csv", index=False)
frame.loc[~frame.contact_number.isin(reasons)].to_csv(OUT / "candidates.csv", index=False)
(OUT / "visual_qa.json").write_text(
    json.dumps(
        {
            "sheets_reviewed": [f"contact_{n}.jpg" for n in range(1, 6)],
            "reasons_by_contact_number": excluded,
            "downloaded": len(frame),
            "visually_accepted": int((~frame.contact_number.isin(reasons)).sum()),
            "label_policy": (
                "Home inferred from decorative cushion/throw-pillow name; "
                "no native usage annotation"
            ),
        },
        indent=2,
    )
    + "\n"
)
print(f"Home visual acceptance: {(~frame.contact_number.isin(reasons)).sum()} / {len(frame)}")
