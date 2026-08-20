"""Reconcile the allowed product population behind the official test-ID quarantine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from PIL import Image

from fashion.config import (
    ORIGINAL_CSV,
    ORIGINAL_IMAGE_DIR,
    TEACHER_TRAIN_CSV,
    TEACHER_TRAIN_IMAGE_DIR,
    TEST_CSV,
)
from fashion.data.quarantine import (
    QuarantineAudit,
    QuarantinePaths,
    establish_quarantine,
)
from fashion.data_audit import CsvAudit, audit_csv

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class PopulationPaths:
    """Metadata, image locations, and expected source counts for the allowed population."""

    teacher_train_csv: Path = TEACHER_TRAIN_CSV
    original_csv: Path = ORIGINAL_CSV
    test_csv: Path = TEST_CSV
    original_image_dir: Path = ORIGINAL_IMAGE_DIR
    lowres_image_dir: Path = TEACHER_TRAIN_IMAGE_DIR
    expected_test_count: int = 5_829
    expected_train_count: int = 38_617
    expected_original_count: int = 44_446

    def quarantine_paths(self) -> QuarantinePaths:
        """Return the ID-only boundary that must hold before labels may load."""
        return QuarantinePaths(
            teacher_test_csv=self.test_csv,
            teacher_train_csv=self.teacher_train_csv,
            original_csv=self.original_csv,
            expected_test_count=self.expected_test_count,
            expected_train_count=self.expected_train_count,
            expected_original_count=self.expected_original_count,
        )


@dataclass(frozen=True)
class ImageInventory:
    """One image directory reconciled by filename, with ambiguity kept visible."""

    paths: Mapping[int, Path] = field(default_factory=dict)
    ambiguous_paths: Mapping[int, tuple[Path, ...]] = field(default_factory=dict)
    invalid_filenames: tuple[str, ...] = ()

    @property
    def duplicate_ids(self) -> tuple[int, ...]:
        """Return IDs claimed by more than one file, so no file may be chosen."""
        return tuple(sorted(self.ambiguous_paths))

    @property
    def known_ids(self) -> frozenset[int]:
        """Return every ID that has at least one file, ambiguous ones included."""
        return frozenset(self.paths).union(self.ambiguous_paths)

    @property
    def candidate_paths(self) -> tuple[tuple[int, Path], ...]:
        """Return every inventoried ID and file pair once, in deterministic order."""
        pairs = [(product_id, path) for product_id, path in self.paths.items()]
        pairs.extend(
            (product_id, path)
            for product_id, paths in self.ambiguous_paths.items()
            for path in paths
        )
        return tuple(sorted(pairs, key=lambda pair: (pair[0], pair[1].name)))


@dataclass(frozen=True)
class PopulationAudit:
    """Immutable reconciliation of allowed metadata IDs and paired image files."""

    source_train_ids: int
    usable_products: int
    quarantined_test_ids: tuple[int, ...]
    teacher_duplicate_ids: tuple[int, ...]
    missing_original_metadata_ids: tuple[int, ...]
    missing_original_image_ids: tuple[int, ...]
    missing_lowres_image_ids: tuple[int, ...]
    missing_both_image_ids: tuple[int, ...]
    unmatched_original_image_ids: tuple[int, ...]
    unmatched_lowres_image_ids: tuple[int, ...]
    invalid_original_image_filenames: tuple[str, ...]
    invalid_lowres_image_filenames: tuple[str, ...]
    duplicate_original_image_ids: tuple[int, ...]
    duplicate_lowres_image_ids: tuple[int, ...]
    unreadable_original_image_ids: tuple[int, ...]
    unreadable_lowres_image_ids: tuple[int, ...]
    teacher_csv_audit: CsvAudit


def inventory_images(directory: Path) -> ImageInventory:
    """Match image files to product IDs by filename, never choosing between two files."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    candidates: dict[int, list[Path]] = {}
    invalid: list[str] = []
    for path in sorted(
        (item for item in directory.iterdir() if item.is_file()),
        key=lambda item: item.name,
    ):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            product_id = int(path.stem)
        except ValueError:
            invalid.append(path.name)
            continue
        candidates.setdefault(product_id, []).append(path)
    paths = {
        product_id: files[0]
        for product_id, files in sorted(candidates.items())
        if len(files) == 1
    }
    ambiguous = {
        product_id: tuple(files)
        for product_id, files in sorted(candidates.items())
        if len(files) > 1
    }
    return ImageInventory(
        paths=MappingProxyType(paths),
        ambiguous_paths=MappingProxyType(ambiguous),
        invalid_filenames=tuple(invalid),
    )


def _unreadable_ids(inventory: ImageInventory) -> tuple[int, ...]:
    """Return inventoried IDs with any file whose pixel data will not fully decode."""
    unreadable: set[int] = set()
    for product_id, path in inventory.candidate_paths:
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, SyntaxError, ValueError):
            unreadable.add(product_id)
    return tuple(sorted(unreadable))


def _allowed_labels(
    paths: PopulationPaths, quarantine: QuarantineAudit
) -> tuple[pd.DataFrame, CsvAudit]:
    """Load target labels from teacher training metadata only, after the quarantine."""
    teacher, teacher_csv_audit = audit_csv(paths.teacher_train_csv)
    teacher = teacher.loc[teacher["id"].isin(quarantine.train_ids)].copy()
    return teacher, teacher_csv_audit


def build_allowed_population(
    paths: PopulationPaths = PopulationPaths(),
) -> tuple[pd.DataFrame, PopulationAudit]:
    """Return allowed products with teacher labels and one readable image pair each."""
    quarantine = establish_quarantine(paths.quarantine_paths())
    teacher, teacher_csv_audit = _allowed_labels(paths, quarantine)

    allowed_ids = pd.Index(quarantine.train_ids, dtype="int64")
    original_inventory = inventory_images(paths.original_image_dir)
    lowres_inventory = inventory_images(paths.lowres_image_dir)

    missing_original = tuple(
        sorted(allowed_ids.difference(pd.Index(sorted(original_inventory.known_ids))))
    )
    missing_lowres = tuple(
        sorted(allowed_ids.difference(pd.Index(sorted(lowres_inventory.known_ids))))
    )
    missing_both = tuple(sorted(set(missing_original).intersection(missing_lowres)))
    allowed = set(allowed_ids)
    unmatched_original = tuple(sorted(original_inventory.known_ids.difference(allowed)))
    unmatched_lowres = tuple(sorted(lowres_inventory.known_ids.difference(allowed)))

    unreadable_original = _unreadable_ids(original_inventory)
    unreadable_lowres = _unreadable_ids(lowres_inventory)
    unreadable_ids = set(unreadable_original).union(unreadable_lowres)

    population = (
        teacher.drop_duplicates("id", keep=False)
        .assign(
            original_image_path=lambda frame: frame["id"].map(original_inventory.paths),
            lowres_image_path=lambda frame: frame["id"].map(lowres_inventory.paths),
        )
        .dropna(subset=["original_image_path", "lowres_image_path"])
        .loc[lambda frame: ~frame["id"].isin(unreadable_ids)]
        .sort_values("id", ignore_index=True)
    )
    audit = PopulationAudit(
        source_train_ids=len(allowed_ids),
        usable_products=len(population),
        quarantined_test_ids=quarantine.test_ids,
        teacher_duplicate_ids=tuple(
            sorted(int(value) for value in teacher_csv_audit.duplicate_ids)
        ),
        missing_original_metadata_ids=tuple(
            sorted(allowed_ids.difference(pd.Index(quarantine.original_ids)))
        ),
        missing_original_image_ids=missing_original,
        missing_lowres_image_ids=missing_lowres,
        missing_both_image_ids=missing_both,
        unmatched_original_image_ids=unmatched_original,
        unmatched_lowres_image_ids=unmatched_lowres,
        invalid_original_image_filenames=original_inventory.invalid_filenames,
        invalid_lowres_image_filenames=lowres_inventory.invalid_filenames,
        duplicate_original_image_ids=original_inventory.duplicate_ids,
        duplicate_lowres_image_ids=lowres_inventory.duplicate_ids,
        unreadable_original_image_ids=unreadable_original,
        unreadable_lowres_image_ids=unreadable_lowres,
        teacher_csv_audit=teacher_csv_audit,
    )
    return population, audit
