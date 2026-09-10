"""Record manual inspection of all four saved contact sheets (2026-09-06)."""

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent
MODELS = {
    2,
    11,
    15,
    16,
    17,
    18,
    20,
    21,
    34,
    37,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    49,
    50,
    51,
    53,
    54,
    60,
    61,
    67,
    69,
    70,
    71,
    72,
    73,
    74,
    78,
    80,
    81,
    82,
    83,
    86,
    87,
    89,
    91,
    93,
    96,
    99,
    100,
    101,
    102,
    103,
    104,
    106,
    108,
    109,
    110,
    112,
}
FLAT_CONCERN = {14, 23, 32, 52, 58, 62, 105}
rows = list(csv.DictReader((OUT / "candidates.csv").open()))
qa = []
for index, row in enumerate(rows, 1):
    flags = []
    if index in MODELS:
        flags.append("model_worn_outfit_context")
    if index == 54:
        flags.append("collage_three_models_three_color_variants")
    if index in FLAT_CONCERN:
        flags.append("mapped_Heels_may_be_flat_or_low_wedge_review_type")
    if index in {68, 83}:
        flags.append("mapped_top_or_tunic_has_dress_like_presentation_review_type")
    if index == 4:
        flags.append("large_product_lettering")
    qa.append(
        {
            "source_id": row["source_id"],
            "contact_index": index,
            "contact_sheet": f"contact_{(index - 1) // 30 + 1:02d}.jpg",
            "visual_flags": ";".join(flags) if flags else "single_product_presentation",
            "recommendation": "exclude_collage" if index == 54 else "root_final_QA",
            "notes": "Contact-sheet review; no Usage relabel; no teacher duplicate assessment",
        }
    )
with (OUT / "visual_qa.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(qa[0]))
    writer.writeheader()
    writer.writerows(qa)
print({"reviewed": len(qa), "model_context": len(MODELS), "collage": 1})
