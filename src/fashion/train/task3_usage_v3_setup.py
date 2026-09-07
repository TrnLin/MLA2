"""Install only data from existing Drive archives into the Usage v3 code checkout."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from fashion.data.hashing import compute_sha256

BASE_ARCHIVES = (
    "usage_mixup_sam_training.zip",
    "teacher_plus_rare_usage_v2_training.zip",
    "teacher_plus_rare_usage_v2.zip",
)
DELTA_ARCHIVE = "teacher_plus_rare_usage_v3_delta.zip"
BASE_FILES = {
    "data/processed/splits.csv",
    "data/processed/label_maps.json",
    "data/processed/teacher_plus_rare_usage_20260906/splits.csv",
}
BASE_PREFIXES = (
    "data/processed/teacher_plus_rare_usage_v2_20260906/",
    "data/external/rare_usage_20260906/images_60x80/",
    "data/external/rare_usage_expansion_20260906/images_60x80/",
)
DELTA_PREFIXES = (
    "data/processed/teacher_plus_rare_usage_v3_20260906/",
    "data/external/rare_usage_replacement_20260906/images_60x80/",
)
TEACHER_PREFIXES = ("data/raw/teacher/train/images_train/",)


def copy_archive(source, cache):
    """Cache a complete, hash-checked copy while keeping reads off Drive during training."""
    source, cache = Path(source), Path(cache)
    if not source.is_file():
        raise FileNotFoundError(f"Required Drive archive is missing: {source}")
    digest = compute_sha256(source)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{digest[:16]}-{source.name}"
    if not destination.is_file() or compute_sha256(destination) != digest:
        partial = destination.with_suffix(".zip.partial")
        shutil.copyfile(source, partial)
        if compute_sha256(partial) != digest:
            raise RuntimeError(f"Archive copy is incomplete: {source.name}")
        partial.replace(destination)
    return destination, digest


def extract_data(archive_path, root, *, files=(), prefixes=()):
    """Reject unsafe archives and extract a fixed data allowlist, never bundled code."""
    root = Path(root).resolve()
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate file names")
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe archive path: {name}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive symlink is not allowed: {name}")
        selected = [
            m
            for m in members
            if not m.is_dir() and (m.filename in files or m.filename.startswith(tuple(prefixes)))
        ]
        if not selected:
            raise ValueError("Archive contains none of the required data paths")
        for member in selected:
            destination = root / member.filename
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Local symlink escapes the checkout: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # The final dataset validator checks exact image hashes. Reuse matching-size
            # files here to avoid reading every image twice during repeated Run All.
            if destination.is_file() and destination.stat().st_size == member.file_size:
                continue
            if destination.exists():
                raise ValueError(f"Local data differs; use a fresh runtime: {destination}")
            partial = destination.with_name(destination.name + ".partial")
            with archive.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            partial.replace(destination)
    return len(selected)


def prepare_data(*, root, drive_project, cache):
    """Reuse teacher and v2 images, install the v3 delta and locate existing SAM results."""
    root, drive_project = Path(root), Path(drive_project)
    data = drive_project / "data"
    baseline = next((data / name for name in BASE_ARCHIVES if (data / name).is_file()), None)
    if baseline is None:
        raise FileNotFoundError(f"Keep one existing v2 data archive in {data}: {BASE_ARCHIVES}")
    delta = data / DELTA_ARCHIVE
    teacher = data / "task3-data.zip"
    for path in (delta, teacher):
        if not path.is_file():
            raise FileNotFoundError(f"Required archive is missing: {path}")
    references = (
        drive_project / "task3_usage_mixup_sam/experiments/t3_usage_expanded_v2_mixup_sam/usage"
    )
    registry = references / "results/runs.csv"
    if not registry.is_file():
        raise FileNotFoundError(f"The completed v2 MixUp + SAM reference is missing: {registry}")
    receipts = []
    for source, files, prefixes in (
        (baseline, BASE_FILES, BASE_PREFIXES),
        (delta, (), DELTA_PREFIXES),
        (teacher, (), TEACHER_PREFIXES),
    ):
        local, digest = copy_archive(source, cache)
        count = extract_data(local, root, files=files, prefixes=prefixes)
        receipts.append({"source": str(source), "sha256": digest, "data_files": count})
        print(f"Ready: {source.name} ({count} data files)", flush=True)
    return {
        "archives": receipts,
        "baseline_directory": str(references),
        "baseline_registry_path": str(registry),
        "output_root": str(drive_project / "task3_usage_replaced_v3_mixup_sam"),
        "archive_code_loaded": False,
    }
