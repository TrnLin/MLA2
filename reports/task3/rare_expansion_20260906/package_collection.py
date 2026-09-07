"""Package only accepted images and evidence, then verify every archived image hash."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import hashlib
import io
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from build_collection import CLASSES, DATA, OUT, ROOT, file_sha256, read_csv, save_json
from PIL import Image


def main():
    frame = read_csv(DATA / "manifest.csv")
    licenses = OUT / "licenses"
    licenses.mkdir(exist_ok=True)
    prior = ROOT / "reports/task3/rare_external_intake_20260906/sources/abo"
    for source, name in [
        (prior / "raw/LICENSE-CC-BY-4.0.txt", "ABO-LICENSE-CC-BY-4.0.txt"),
        (prior / "ATTRIBUTION.md", "ABO-ATTRIBUTION.md"),
        (prior / "raw/README.md", "ABO-RELEASE-README.md"),
    ]:
        shutil.copyfile(source, licenses / name)
    files = {DATA / "manifest.csv"}
    for column in ("path", "original_path", "source_original_path"):
        files.update(resolve_task3_path(value, root=ROOT) for value in frame[column])
    for name in [
        "README.md",
        "accepted_manifest.csv",
        "class_counts.csv",
        "summary.json",
        "central_rejected.csv",
        "internal_near_duplicate_matches.csv",
        "internal_duplicate_summary.json",
        "family_edges.csv",
        "gallery.html",
        "preview.png",
        "prepared_contact_index.csv",
        "review_artifacts.json",
        "build_collection.py",
        "check_internal_duplicates.py",
        "make_review.py",
        "package_collection.py",
    ]:
        files.add(OUT / name)
    files.update(licenses.iterdir())
    for slug in CLASSES:
        files.add(OUT / f"prepared_{slug}.png")
        files.add(OUT / slug / "audit_summary.json")
        files.add(OUT / slug / "candidates.csv")
    for name in [
        "home/acquire.py",
        "home/finalize.py",
        "home/visual_qa.json",
        "home/selected_native_records.json",
        "party/acquisition_log.md",
        "smart_casual/ACQUISITION_LOG.md",
        "travel/ACQUISITION_LOG.md",
    ]:
        files.add(OUT / name)
    destination = OUT / "rare_usage_567_images.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            assert path.is_file(), path
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for row in frame.itertuples():
            for path_column, hash_column in [
                ("path", "sha256"),
                ("original_path", "file_sha256"),
                ("source_original_path", "source_file_sha256"),
            ]:
                content = archive.read(getattr(row, path_column))
                assert hashlib.sha256(content).hexdigest() == getattr(row, hash_column)
            with Image.open(io.BytesIO(archive.read(row.path))) as im:
                im.load()
                assert im.size == (60, 80) and im.mode == "RGB"
        names = archive.namelist()
        assert len(names) == len(set(names))
    prepared_paths = set(frame.path)
    # Move the two earlier prepared duplicate copies into this report's audit-only folder.
    extras = [
        p
        for p in (DATA / "images_60x80").rglob("*.png")
        if str(p.relative_to(ROOT)) not in prepared_paths
    ]
    for path in extras:
        rejected = OUT / "rejected_prepared" / path.parent.name / path.name
        rejected.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), rejected)
    assert len(list((DATA / "images_60x80").rglob("*.png"))) == len(frame)
    result = {
        "archive": str(destination.relative_to(ROOT)),
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
        "zip_members": len(files),
        "accepted_images": len(frame),
        "all_image_hashes_verified_inside_archive": True,
        "all_prepared_images_decoded_inside_archive": True,
        "archive_crc_passed": True,
        "rejected_images_in_archive": False,
        "prepared_directory_matches_manifest": True,
    }
    light = OUT / "rare_usage_567_prepared.zip"
    original_paths = {
        resolve_task3_path(value, root=ROOT) for col in ("original_path", "source_original_path") for value in frame[col]
    }
    with ZipFile(light, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files - original_paths):
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(light) as archive:
        assert archive.testzip() is None
        for row in frame.itertuples():
            assert hashlib.sha256(archive.read(row.path)).hexdigest() == row.sha256
    result["prepared_archive"] = str(light.relative_to(ROOT))
    result["prepared_archive_sha256"] = file_sha256(light)
    result["prepared_archive_bytes"] = light.stat().st_size
    result["prepared_archive_hashes_and_crc_passed"] = True
    save_json(OUT / "package_validation.json", result)
    print(result)


if __name__ == "__main__":
    main()
