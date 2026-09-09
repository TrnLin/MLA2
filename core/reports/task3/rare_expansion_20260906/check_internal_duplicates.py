"""Compare new images, including siblings in the same source family."""

from fashion.task3_paths import resolve_task3_path

from types import SimpleNamespace

import numpy as np
import pandas as pd
from build_collection import CLASSES, DATA, OUT, ROOT, read_csv, save_json

from fashion.data.external_usage_audit import _cropped_path, _normalized_path
from fashion.data.perceptual import compute_pair_pixel_metrics, meets_near_duplicate_rule


def main():
    frame = pd.concat([read_csv(OUT / slug / "audited.csv") for slug in CLASSES], ignore_index=True)
    prefixes = ("", "view_", "crop_")
    dhashes = {
        p: np.array([int(v, 16) for v in frame[p + "dhash_hex"]], dtype=np.uint64) for p in prefixes
    }
    ahashes = {
        p: np.array([int(v, 16) for v in frame[p + "ahash_hex"]], dtype=np.uint64) for p in prefixes
    }
    matches = []
    accepted = set()
    candidates = {}
    positions = np.arange(len(frame))
    for first in range(len(frame)):
        for p in prefixes:
            for q in prefixes:
                dh = np.bitwise_count(dhashes[q] ^ dhashes[p][first])
                ah = np.bitwise_count(ahashes[q] ^ ahashes[p][first])
                for second in np.flatnonzero((positions > first) & (dh <= 2) & (ah <= 1)):
                    candidates.setdefault((first, int(second)), []).append(
                        (p, q, int(dh[second]), int(ah[second]))
                    )
    for key in ("file_sha256", "pixel_sha256", "view_pixel_sha256", "crop_pixel_sha256"):
        for _, group in frame.groupby(key):
            indices = list(group.index)
            for second in indices[1:]:
                first = indices[0]
                pair = (first, second)
                if pair not in accepted:
                    accepted.add(pair)
                    matches.append(
                        {
                            "first_id": frame.iloc[first].external_id,
                            "second_id": frame.iloc[second].external_id,
                            "reason": "exact_" + key,
                        }
                    )
    cache = DATA / "comparison_views" / "new_internal"
    views = {}

    def path(index, prefix):
        key = (index, prefix)
        if key not in views:
            source = resolve_task3_path(frame.iloc[index].original_path, root=ROOT)
            views[key] = (
                _cropped_path(source, cache)
                if prefix == "crop_"
                else _normalized_path(source, cache)
                if prefix == "view_"
                else source
            )
        return views[key]

    print(f"Internal candidates: {len(candidates)} pairs among {len(frame)} images", flush=True)
    for number, ((first, second), signals) in enumerate(candidates.items(), 1):
        if (first, second) in accepted:
            continue
        for p, q, dh, ah in signals:
            metrics = compute_pair_pixel_metrics(path(first, p), path(second, q))
            if meets_near_duplicate_rule(
                SimpleNamespace(dhash_distance=dh, ahash_distance=ah, **metrics)
            ):
                matches.append(
                    {
                        "first_id": frame.iloc[first].external_id,
                        "second_id": frame.iloc[second].external_id,
                        "reason": "accepted_near_duplicate",
                        "first_view": p or "original",
                        "second_view": q or "original",
                        "dhash_distance": dh,
                        "ahash_distance": ah,
                        **metrics,
                    }
                )
                accepted.add((first, second))
                break
        if number % 100 == 0:
            print(
                f"Internal pairs checked: {number}/{len(candidates)}; matches={len(matches)}",
                flush=True,
            )
    pd.DataFrame(
        matches,
        columns=[
            "first_id",
            "second_id",
            "reason",
            "first_view",
            "second_view",
            "dhash_distance",
            "ahash_distance",
            "mse",
            "mae",
            "max_difference",
            "crop_mse",
            "crop_mae",
            "foreground_fraction_1",
            "foreground_fraction_2",
            "foreground_ratio",
        ],
    ).to_csv(OUT / "internal_near_duplicate_matches.csv", index=False)
    save_json(
        OUT / "internal_duplicate_summary.json",
        {
            "input_images": len(frame),
            "prefiltered_pairs": len(candidates),
            "accepted_duplicate_edges": len(matches),
            "source_family_equality_does_not_skip_comparison": True,
        },
    )
    print(f"Internal duplicate edges: {len(matches)}", flush=True)


if __name__ == "__main__":
    main()
