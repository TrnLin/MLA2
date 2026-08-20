from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from data_helpers import save_image, save_truncated_jpeg, write_metadata
import fashion.data.population as population_module
from fashion.data.population import (
    PopulationPaths,
    build_allowed_population,
    inventory_images,
)


TEACHER_ROWS = [
    "1,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,Casual,Teacher one",
    "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,NA,Teacher two",
    "3,Men,Apparel,Bottomwear,Jeans,Black,Spring,2014,,Teacher three",
    "4,Women,Apparel,Topwear,Tshirts,White,Summer,2015,Casual,Teacher four",
]
ORIGINAL_ROWS = [
    "1,Girls,Personal Care,Fragrance,Perfume,Pink,Fall,2020,Formal,Original one",
    "2,Girls,Personal Care,Fragrance,Perfume,Pink,Fall,2020,Formal,Original two",
    "3,Girls,Personal Care,Fragrance,Perfume,Pink,Fall,2020,Formal,Original three",
    "4,Girls,Personal Care,Fragrance,Perfume,Pink,Fall,2020,Formal,Original four",
    "90,Women,Accessories,Bags,Bags,Green,Fall,2016,Casual,Original ninety",
    "91,Men,Footwear,Shoes,Casual Shoes,Grey,Summer,2016,Casual,Original ninety one",
]


@pytest.fixture
def population_paths(tmp_path: Path) -> PopulationPaths:
    """Four allowed products, two quarantined IDs, and three complete image pairs."""
    teacher_csv = write_metadata(tmp_path / "teacher.csv", TEACHER_ROWS)
    original_csv = write_metadata(tmp_path / "original.csv", ORIGINAL_ROWS)
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(
        "id,gender,articleType,season,usage\n"
        "90,Women,Bags,Fall,Casual\n"
        "91,Men,Casual Shoes,Summer,Casual\n",
        encoding="utf-8",
    )
    original_images = tmp_path / "original-images"
    lowres_images = tmp_path / "lowres-images"
    for product_id in (1, 2, 3):
        save_image(original_images / f"{product_id}.jpg")
        save_image(lowres_images / f"{product_id}.jpg")
    save_image(original_images / "90.jpg")
    save_image(lowres_images / "91.jpg")
    return PopulationPaths(
        teacher_train_csv=teacher_csv,
        original_csv=original_csv,
        test_csv=test_csv,
        original_image_dir=original_images,
        lowres_image_dir=lowres_images,
        expected_test_count=2,
        expected_train_count=4,
        expected_original_count=6,
    )


def test_allowed_population_uses_teacher_labels_and_requires_both_image_views(
    population_paths: PopulationPaths,
) -> None:
    """Only allowed IDs with one sharp and one blurry readable image may be trained on."""
    population, audit = build_allowed_population(population_paths)

    assert population["id"].tolist() == [1, 2, 3]
    assert audit.source_train_ids == 4
    assert audit.usable_products == 3
    assert audit.missing_both_image_ids == (4,)
    assert audit.missing_original_image_ids == (4,)
    assert audit.missing_lowres_image_ids == (4,)
    assert not set(population["id"]).intersection({90, 91})
    assert population.loc[1, "usage"] == "NA"
    assert population.loc[2, "usage"] == ""
    assert {"original_image_path", "lowres_image_path"}.issubset(population)
    assert pd.api.types.is_integer_dtype(population["id"])
    assert audit.quarantined_test_ids == (90, 91)
    assert audit.unmatched_original_image_ids == (90,)
    assert audit.unmatched_lowres_image_ids == (91,)
    assert audit.teacher_csv_audit.row_count == 4
    assert audit.teacher_csv_audit.literal_na_usage_count == 1


def test_allowed_population_keeps_both_image_views_on_one_product_row(
    population_paths: PopulationPaths,
) -> None:
    """Splitting the two views into separate rows would double every product count."""
    population, _ = build_allowed_population(population_paths)

    assert len(population) == population["id"].nunique()
    assert [Path(path).parent.name for path in population["original_image_path"]] == [
        "original-images"
    ] * 3
    assert [Path(path).parent.name for path in population["lowres_image_path"]] == [
        "lowres-images"
    ] * 3
    assert [Path(path).name for path in population["original_image_path"]] == [
        "1.jpg",
        "2.jpg",
        "3.jpg",
    ]


def test_labels_load_only_after_the_quarantine_has_succeeded(
    population_paths: PopulationPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading teacher targets before the boundary is proven would void the quarantine."""
    calls: list[str] = []
    real_quarantine = population_module.establish_quarantine
    real_audit_csv = population_module.audit_csv

    def record_quarantine(paths: object) -> object:
        audit = real_quarantine(paths)  # type: ignore[arg-type]
        calls.append("quarantine")
        return audit

    def record_audit_csv(path: Path) -> object:
        calls.append(f"audit_csv:{Path(path).name}")
        return real_audit_csv(path)

    monkeypatch.setattr(population_module, "establish_quarantine", record_quarantine)
    monkeypatch.setattr(population_module, "audit_csv", record_audit_csv)

    build_allowed_population(population_paths)

    assert calls == ["quarantine", "audit_csv:teacher.csv"]

    calls.clear()
    with pytest.raises(ValueError, match="Expected 99 official test IDs"):
        build_allowed_population(replace(population_paths, expected_test_count=99))

    assert calls == []


def test_original_metadata_is_read_only_through_the_id_only_path(
    population_paths: PopulationPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading original target columns would keep unreviewed labels in process memory."""
    real_read_csv = pd.read_csv
    reads: list[tuple[Path, object]] = []

    def record_read_csv(source: object, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append((Path(str(source)), kwargs.get("usecols")))
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", record_read_csv)

    population, _ = build_allowed_population(population_paths)

    original_reads = [
        usecols for path, usecols in reads if path == population_paths.original_csv
    ]
    assert original_reads == [["id"]]
    assert any(path == population_paths.teacher_train_csv for path, _ in reads)
    assert population["productDisplayName"].tolist() == [
        "Teacher one",
        "Teacher two",
        "Teacher three",
    ]
    assert population["gender"].tolist() == ["Men", "Women", "Men"]


def test_duplicate_teacher_ids_are_rejected_before_any_row_is_chosen(
    population_paths: PopulationPaths,
) -> None:
    """Silently keeping one of two conflicting label rows hides a source contradiction."""
    write_metadata(
        population_paths.teacher_train_csv,
        [
            *TEACHER_ROWS,
            "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,Formal,Teacher two again",
        ],
    )

    with pytest.raises(ValueError, match="Duplicate IDs in Teacher training metadata"):
        build_allowed_population(replace(population_paths, expected_train_count=5))


def test_invalid_image_filenames_are_reported_and_never_matched(
    population_paths: PopulationPaths,
) -> None:
    """A non-numeric filename cannot be matched to a product and must stay visible."""
    save_image(population_paths.original_image_dir / "sharp-copy.jpg")
    save_image(population_paths.lowres_image_dir / "thumb_4.jpg")
    (population_paths.lowres_image_dir / "notes.txt").write_text("x", encoding="utf-8")

    population, audit = build_allowed_population(population_paths)

    assert audit.invalid_original_image_filenames == ("sharp-copy.jpg",)
    assert audit.invalid_lowres_image_filenames == ("thumb_4.jpg",)
    assert population["id"].tolist() == [1, 2, 3]


def test_ambiguous_duplicate_image_stems_are_reported_and_excluded(
    population_paths: PopulationPaths,
) -> None:
    """Two files for one view make the chosen pixels unknowable, so the product must drop."""
    save_image(population_paths.original_image_dir / "1.png")
    save_image(population_paths.lowres_image_dir / "2.png")

    population, audit = build_allowed_population(population_paths)

    assert audit.duplicate_original_image_ids == (1,)
    assert audit.duplicate_lowres_image_ids == (2,)
    assert population["id"].tolist() == [3]
    assert audit.usable_products == 1
    assert audit.missing_original_image_ids == (4,)
    assert audit.missing_lowres_image_ids == (4,)


def test_a_missing_sharp_image_alone_excludes_the_product(
    population_paths: PopulationPaths,
) -> None:
    """One view is not enough evidence for a paired sharp-and-blurry product row."""
    save_image(population_paths.lowres_image_dir / "4.jpg")

    population, audit = build_allowed_population(population_paths)

    assert population["id"].tolist() == [1, 2, 3]
    assert audit.missing_original_image_ids == (4,)
    assert audit.missing_lowres_image_ids == ()
    assert audit.missing_both_image_ids == ()


def test_a_missing_blurry_image_alone_excludes_the_product(
    population_paths: PopulationPaths,
) -> None:
    """The blurry teacher view is the modelling input and cannot be substituted."""
    save_image(population_paths.original_image_dir / "4.jpg")

    population, audit = build_allowed_population(population_paths)

    assert population["id"].tolist() == [1, 2, 3]
    assert audit.missing_lowres_image_ids == (4,)
    assert audit.missing_original_image_ids == ()
    assert audit.missing_both_image_ids == ()


def test_unreadable_images_are_reported_and_kept_out_of_the_population(
    population_paths: PopulationPaths,
) -> None:
    """An undecodable file would fail later during training, not during reconciliation."""
    (population_paths.lowres_image_dir / "2.jpg").write_bytes(b"not an image")
    (population_paths.original_image_dir / "3.jpg").write_bytes(b"not an image")

    population, audit = build_allowed_population(population_paths)

    assert audit.unreadable_lowres_image_ids == (2,)
    assert audit.unreadable_original_image_ids == (3,)
    assert population["id"].tolist() == [1]
    assert audit.usable_products == 1


def test_truncated_images_that_pass_structure_checks_are_reported_and_excluded(
    population_paths: PopulationPaths,
) -> None:
    """A structure-only check passes truncated pixels that break during training."""
    truncated = save_truncated_jpeg(population_paths.lowres_image_dir / "3.jpg")
    with Image.open(truncated) as image:
        image.verify()

    population, audit = build_allowed_population(population_paths)

    assert audit.unreadable_lowres_image_ids == (3,)
    assert audit.unreadable_original_image_ids == ()
    assert population["id"].tolist() == [1, 2]
    assert audit.usable_products == 2


def test_unreadable_candidates_outside_the_allowed_ids_are_still_audited(
    population_paths: PopulationPaths,
) -> None:
    """Silently skipping unmatched files would understate the collection's decode health."""
    (population_paths.original_image_dir / "95.jpg").write_bytes(b"not an image")
    save_truncated_jpeg(population_paths.lowres_image_dir / "96.jpg")

    population, audit = build_allowed_population(population_paths)

    assert audit.unreadable_original_image_ids == (95,)
    assert audit.unreadable_lowres_image_ids == (96,)
    assert population["id"].tolist() == [1, 2, 3]
    assert audit.usable_products == 3
    assert 95 in audit.unmatched_original_image_ids
    assert 96 in audit.unmatched_lowres_image_ids


def test_every_duplicate_stem_path_is_decoded_and_reported_once(
    population_paths: PopulationPaths,
) -> None:
    """Skipping ambiguous paths would hide a broken file behind a readable sibling."""
    (population_paths.original_image_dir / "1.png").write_bytes(b"not an image")
    save_truncated_jpeg(population_paths.lowres_image_dir / "2.jpeg")

    population, audit = build_allowed_population(population_paths)

    assert audit.duplicate_original_image_ids == (1,)
    assert audit.duplicate_lowres_image_ids == (2,)
    assert audit.unreadable_original_image_ids == (1,)
    assert audit.unreadable_lowres_image_ids == (2,)
    assert population["id"].tolist() == [3]


def test_inventory_images_reports_invalid_and_ambiguous_filenames(tmp_path: Path) -> None:
    """Filename reconciliation must be inspectable on its own, without label loading."""
    directory = tmp_path / "images"
    save_image(directory / "1.jpg")
    save_image(directory / "1.png")
    save_image(directory / "2.jpg")
    save_image(directory / "not-an-id.png")
    (directory / "notes.txt").write_text("ignored", encoding="utf-8")

    inventory = inventory_images(directory)

    assert set(inventory.paths) == {2}
    assert inventory.paths[2].name == "2.jpg"
    assert inventory.duplicate_ids == (1,)
    assert inventory.invalid_filenames == ("not-an-id.png",)
    assert inventory.known_ids == frozenset({1, 2})
    assert [(product_id, path.name) for product_id, path in inventory.candidate_paths] == [
        (1, "1.jpg"),
        (1, "1.png"),
        (2, "2.jpg"),
    ]


def test_inventory_images_requires_an_existing_directory(tmp_path: Path) -> None:
    """A mistyped image directory must fail loudly, not report an empty population."""
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        inventory_images(tmp_path / "missing")
