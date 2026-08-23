from __future__ import annotations

import builtins
import errno
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from fashion.config import NORMALIZATION_JSON, PAIRED_NORMALIZATION_JSON
from fashion.data.dataset import FashionDataset
from fashion.data.hashing import compute_sha256
from fashion.data.pipeline import prepare_data, validate_prepared_data_cache
from fashion.data.statistics import compute_paired_normalization_stats, train_ids_digest
from fashion.data.variants import (
    build_training_variant_manifest,
    catalogue_high_resolution_dataset,
    discover_high_resolution_root,
    load_image_variants,
)


def _symlink_directory_or_skip(source: Path, target: Path) -> None:
    try:
        os.symlink(source, target, target_is_directory=True)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EPERM} or getattr(error, "winerror", None) == 1314:
            pytest.skip("directory symlink privilege is unavailable on this platform")
        raise


def _high_resolution_tree(root: Path, ids: list[int], *, images_csv: bool = True) -> Path:
    dataset = root / "data/fashion-dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    links = []
    for item_id in ids:
        Image.new(
            "RGB",
            (600, 800),
            color=((item_id * 17) % 255, (item_id * 53) % 255, (item_id * 91) % 255),
        ).save(images / f"{item_id}.jpg", "JPEG")
        links.append({"filename": f"{item_id}.jpg", "link": f"https://example.test/{item_id}"})
    if images_csv:
        pd.DataFrame(links).to_csv(dataset / "images.csv", index=False)
    return dataset


def _catalogue_and_manifest(prepared_project):
    dataset = _high_resolution_tree(prepared_project.root, [*range(1, 13), 101, 102])
    catalogue_dir = prepared_project.processed / "high_resolution"
    catalogue = catalogue_high_resolution_dataset(
        dataset_dir=dataset,
        splits_csv=prepared_project.splits,
        official_prediction_csv=prepared_project.prediction_csv,
        output_dir=catalogue_dir,
        root=prepared_project.root,
        workers=2,
    )
    variants = prepared_project.processed / "training_image_variants.csv.gz"
    summary = prepared_project.processed / "training_image_variants_summary.json"
    proof = build_training_variant_manifest(
        splits_csv=prepared_project.splits,
        high_resolution_catalogue_csv=catalogue["image_catalogue"],
        official_prediction_csv=prepared_project.prediction_csv,
        output_csv=variants,
        summary_output=summary,
        root=prepared_project.root,
    )
    return catalogue, variants, summary, proof


def _resign_manifest(variants: Path, summary: Path) -> None:
    proof = json.loads(summary.read_text(encoding="utf-8"))
    proof["manifest_sha256"] = compute_sha256(variants)
    proof["model_variant_count"] = len(pd.read_csv(variants))
    summary.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(prepared_project, variants: Path, summary: Path, partition: str = "train"):
    return load_image_variants(
        partition=partition,
        variants_csv=variants,
        summary_json=summary,
        splits_csv=prepared_project.splits,
        official_prediction_csv=prepared_project.prediction_csv,
        root=prepared_project.root,
    )


def test_discovery_supports_flat_and_nested_image_only_layouts(tmp_path):
    flat = _high_resolution_tree(tmp_path / "flat", [1, 2], images_csv=False)
    canonical, evidence = discover_high_resolution_root(flat, root=tmp_path / "flat")
    assert canonical == flat
    assert evidence["canonical_root"] == "data/fashion-dataset"

    nested_base = tmp_path / "nested/data/fashion-dataset"
    nested = nested_base / "fashion-dataset"
    (nested / "images").mkdir(parents=True)
    for item_id in (1, 2):
        Image.new("RGB", (600, 800), (item_id, 2, 3)).save(nested / "images" / f"{item_id}.jpg")
    canonical, evidence = discover_high_resolution_root(nested_base, root=tmp_path / "nested")
    assert canonical == nested
    assert evidence["canonical_root"] == "data/fashion-dataset/fashion-dataset"


def test_discovery_proves_separate_outer_copy(tmp_path):
    dataset = _high_resolution_tree(tmp_path, [1, 2], images_csv=False)
    nested = dataset / "fashion-dataset"
    shutil.copytree(dataset / "images", nested / "images")
    _, evidence = discover_high_resolution_root(dataset, root=tmp_path, duplicate_sample_size=2)
    relation = evidence["duplicate_tree"]["images"]
    assert evidence["duplicate_tree"]["safe_to_scan_canonical_once"] is True
    assert relation["relation"] == "separate_probable_copy"
    assert relation["same_device_inode_pairs"] == 0


def test_discovery_rejects_mismatched_symlink_tree(tmp_path):
    dataset = _high_resolution_tree(tmp_path, [1], images_csv=False)
    other = tmp_path / "other"
    other.mkdir()
    Image.new("RGB", (601, 800), (9, 9, 9)).save(other / "1.jpg")
    nested = dataset / "fashion-dataset"
    nested.mkdir()
    _symlink_directory_or_skip(other, nested / "images")
    with pytest.raises(ValueError, match="do not match"):
        discover_high_resolution_root(dataset, root=tmp_path, duplicate_sample_size=1)


def test_catalogue_is_image_only_and_never_opens_target_metadata(prepared_project, monkeypatch):
    dataset = _high_resolution_tree(
        prepared_project.root, [*range(1, 13), 101, 102], images_csv=False
    )
    (dataset / "styles").mkdir()
    (dataset / "styles.csv").write_text("id,gender\n1,SENTINEL\n", encoding="utf-8")
    (dataset / "styles/1.json").write_text('{"gender":"SENTINEL"}', encoding="utf-8")
    original_path_open = Path.open
    original_builtin_open = builtins.open

    def reject_path_open(path, *args, **kwargs):
        if Path(path).name == "styles.csv" or "styles" in Path(path).parts:
            raise AssertionError(f"target-bearing metadata opened: {path}")
        return original_path_open(path, *args, **kwargs)

    def reject_builtin_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            if path.name == "styles.csv" or "styles" in path.parts:
                raise AssertionError(f"target-bearing metadata opened: {path}")
        return original_builtin_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_path_open)
    monkeypatch.setattr(builtins, "open", reject_builtin_open)
    paths = catalogue_high_resolution_dataset(
        dataset_dir=dataset,
        splits_csv=prepared_project.splits,
        official_prediction_csv=prepared_project.prediction_csv,
        output_dir=prepared_project.processed / "high_resolution",
        root=prepared_project.root,
        workers=2,
    )
    summary = json.loads(paths["catalogue"].read_text(encoding="utf-8"))
    assert summary["source_scope"]["policy"] == "image_only"
    assert summary["source_scope"]["target_bearing_metadata_opened"] is False
    assert summary["source_scope"]["images_csv"] is None
    assert summary["images"]["decoded_count"] == 14


def test_catalogue_keeps_lexical_project_paths_through_symlink(prepared_project):
    ids = [*range(1, 13), 101, 102]
    external = _high_resolution_tree(prepared_project.root / "external", ids, images_csv=False)
    dataset = prepared_project.root / "data/fashion-dataset"
    dataset.mkdir(parents=True)
    _symlink_directory_or_skip(external / "images", dataset / "images")
    paths = catalogue_high_resolution_dataset(
        dataset_dir=dataset,
        splits_csv=prepared_project.splits,
        official_prediction_csv=prepared_project.prediction_csv,
        output_dir=prepared_project.processed / "high_resolution",
        root=prepared_project.root,
        workers=2,
    )
    image_paths = pd.read_csv(paths["image_catalogue"])["path"]
    assert image_paths.str.startswith("data/fashion-dataset/images/").all()
    assert not image_paths.str.contains(str(prepared_project.root), regex=False).any()


def test_variant_manifest_pairs_every_model_partition(prepared_project):
    _, variants_path, summary, proof = _catalogue_and_manifest(prepared_project)
    variants = pd.read_csv(variants_path)
    assert variants["variant_key"].is_unique
    assert set(variants["partition"]) == {"train", "val", "holdout"}
    assert variants.groupby("id")["variant"].agg(set).eq({"original", "high_resolution"}).all()
    assert not set(variants["id"]).intersection({101, 102})
    assert all(value == 0 for key, value in proof.items() if key.endswith("intersection"))
    assert proof["high_resolution_duplicate_membership_mismatch_count"] == 0
    for partition, counts in proof["partition_coverage"].items():
        assert counts["variant_count"] == 2 * counts["product_count"], partition
        assert counts["complete_pair_count"] == counts["product_count"], partition
        assert counts["incomplete_pair_count"] == 0, partition

    for partition in ("train", "val", "holdout"):
        paired = _load(prepared_project, variants_path, summary, partition)
        assert paired.groupby("id")["per_product_weight"].sum().eq(1).all()
        assert not any(column.endswith(("_x", "_y")) for column in paired)
        if partition == "holdout":
            for target in ("articleType", "season", "gender", "usage"):
                assert paired[target].eq("").all()
            unlocked = load_image_variants(
                partition="holdout",
                variants_csv=variants_path,
                summary_json=summary,
                splits_csv=prepared_project.splits,
                official_prediction_csv=prepared_project.prediction_csv,
                root=prepared_project.root,
                evaluation_unlocked=True,
            )
            assert len(unlocked) == len(paired)

    with pytest.raises(ValueError, match="only valid for holdout"):
        load_image_variants(
            partition="train",
            variants_csv=variants_path,
            summary_json=summary,
            splits_csv=prepared_project.splits,
            official_prediction_csv=prepared_project.prediction_csv,
            root=prepared_project.root,
            evaluation_unlocked=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cherry_pick", "exactly cover"),
        ("third_variant", "exactly two"),
        ("duplicate_variant", "exactly two"),
        ("two_originals", "one original and one high-resolution"),
        ("prediction_id", "exactly cover"),
        ("absolute_path", "project-relative"),
        ("traversal_path", "project-relative"),
        ("missing_path", "does not exist"),
        ("wrong_partition", "forbidden partition|differs from splits"),
        ("wrong_family", "product_family_group differs"),
        ("wrong_duplicate_group", "duplicate_group differs"),
        ("wrong_canonical_sha", "canonical SHA-256 differs"),
        ("wrong_weight", "weights must be exactly"),
        ("target_column", "unexpected columns"),
    ],
)
def test_variant_loader_rejects_adversarial_manifests(prepared_project, mutation, message):
    _, variants_path, summary, _ = _catalogue_and_manifest(prepared_project)
    frame = pd.read_csv(variants_path, keep_default_na=False)
    if mutation == "cherry_pick":
        frame = frame[frame["id"].eq(frame["id"].iloc[0])]
    elif mutation == "third_variant":
        extra = frame.iloc[[0]].copy()
        extra["variant"] = "third"
        extra["variant_key"] = extra["id"].astype(str) + ":third"
        frame = pd.concat([frame, extra], ignore_index=True)
    elif mutation == "duplicate_variant":
        extra = frame.iloc[[0]].copy()
        extra["variant_key"] = extra["id"].astype(str) + ":duplicate"
        frame = pd.concat([frame, extra], ignore_index=True)
    elif mutation == "two_originals":
        row = frame.index[
            frame["id"].eq(frame["id"].iloc[0]) & frame["variant"].eq("high_resolution")
        ][0]
        frame.loc[row, "variant"] = "original"
        frame.loc[row, "variant_key"] = f"{int(frame.loc[row, 'id'])}:second_original"
    elif mutation == "prediction_id":
        frame.loc[frame.index[0], "id"] = 101
        frame.loc[frame.index[0], "variant_key"] = "101:original"
    elif mutation == "absolute_path":
        frame.loc[frame.index[0], "path"] = "/tmp/injected.jpg"
    elif mutation == "traversal_path":
        frame.loc[frame.index[0], "path"] = "../injected.jpg"
    elif mutation == "missing_path":
        high_index = frame.index[frame["variant"].eq("high_resolution")][0]
        frame.loc[high_index, "path"] = "data/fashion-dataset/images/missing.jpg"
    elif mutation == "wrong_partition":
        frame.loc[frame.index[0], "partition"] = "quarantine"
    elif mutation == "wrong_family":
        frame.loc[frame.index[0], "product_family_group"] = "injected"
    elif mutation == "wrong_duplicate_group":
        frame.loc[frame.index[0], "duplicate_group"] = "injected"
    elif mutation == "wrong_canonical_sha":
        frame.loc[frame.index[0], "canonical_sha256"] = "0" * 64
    elif mutation == "wrong_weight":
        frame.loc[frame["id"].eq(frame["id"].iloc[0]), "per_product_weight"] = [0.4, 0.6]
    elif mutation == "target_column":
        frame["articleType"] = "INJECTED"
    frame.to_csv(variants_path, index=False)
    _resign_manifest(variants_path, summary)
    with pytest.raises(ValueError, match=message):
        _load(prepared_project, variants_path, summary)


def test_variant_loader_rejects_untrusted_hashes(prepared_project):
    _, variants_path, summary, _ = _catalogue_and_manifest(prepared_project)
    frame = pd.read_csv(variants_path)
    frame.loc[frame.index[0], "sample_group"] = "changed"
    frame.to_csv(variants_path, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        _load(prepared_project, variants_path, summary)

    proof = json.loads(summary.read_text(encoding="utf-8"))
    proof["splits_sha256"] = "0" * 64
    proof["manifest_sha256"] = compute_sha256(variants_path)
    summary.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="splits.csv SHA-256"):
        _load(prepared_project, variants_path, summary)


def test_pair_weighted_train_normalization_provenance(prepared_project):
    _, variants_path, summary, _ = _catalogue_and_manifest(prepared_project)
    train = _load(prepared_project, variants_path, summary)
    output = prepared_project.processed / "paired_normalization.json"
    stats = compute_paired_normalization_stats(
        variants_csv=variants_path,
        variants_summary_json=summary,
        splits_csv=prepared_project.splits,
        official_prediction_csv=prepared_project.prediction_csv,
        output_path=output,
        root=prepared_project.root,
        workers=2,
    )
    train_ids = train["id"].drop_duplicates().astype(int).tolist()
    assert stats["source_partition"] == "train"
    assert stats["source_variants"] == ["original", "high_resolution"]
    assert stats["variant_count"] == 2 * stats["product_count"]
    assert stats["total_effective_product_weight"] == stats["product_count"]
    assert stats["train_ids_digest"] == train_ids_digest(train_ids)
    assert stats["variant_manifest_sha256"] == compute_sha256(variants_path)
    assert "splits_sha256" not in stats

    sample = FashionDataset(train.iloc[[0]], root=prepared_project.root, targets=())[0]
    for field in (
        "variant",
        "variant_key",
        "sample_group",
        "independence_group",
        "duplicate_group",
        "product_family_group",
        "per_product_weight",
    ):
        assert field in sample
    assert sample["per_product_weight"] == 0.5
    assert NORMALIZATION_JSON == PAIRED_NORMALIZATION_JSON


def test_full_pipeline_builds_paired_stats_and_alignment_from_empty_output(tiny_project):
    _high_resolution_tree(tiny_project.root, [*range(1, 13), 101, 102])
    assert not tiny_project.processed.exists()
    prepare_data(root=tiny_project.root, workers=2, include_high_resolution_variants=True)

    paired = json.loads(tiny_project.paired_normalization.read_text(encoding="utf-8"))
    variants = tiny_project.processed / "training_image_variants.csv.gz"
    assert paired["policy"] == "pair_weighted_original_and_high_resolution"
    assert paired["variant_manifest_sha256"] == compute_sha256(variants)
    assert paired["source_partition"] == "train"
    assert paired["variant_count"] == 2 * paired["product_count"]
    assert (tiny_project.processed / "normalization_original_only.json").is_file()
    assert (tiny_project.processed / "high_resolution/all_alignment_pairs.csv.gz").is_file()
    assert (tiny_project.processed / "high_resolution/alignment_summary.json").is_file()
    result = validate_prepared_data_cache(
        root=tiny_project.root,
        include_high_resolution_variants=True,
    )
    assert result["critical_prepared_artifacts_fully_hashed"] is True
    assert result["raw_image_content_fully_hashed"] is False


def test_cache_rejects_stale_paired_stats(tiny_project):
    _high_resolution_tree(tiny_project.root, [*range(1, 13), 101, 102])
    prepare_data(root=tiny_project.root, workers=2, include_high_resolution_variants=True)
    stats = json.loads(tiny_project.paired_normalization.read_text(encoding="utf-8"))
    stats["variant_manifest_sha256"] = "0" * 64
    tiny_project.paired_normalization.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="critical prepared-data artifact is stale"):
        validate_prepared_data_cache(
            root=tiny_project.root,
            include_high_resolution_variants=True,
        )


def test_variant_manifest_fails_when_any_model_pair_is_missing(prepared_project):
    catalogue, _, _, _ = _catalogue_and_manifest(prepared_project)
    splits = pd.read_csv(prepared_project.splits)
    val_id = int(splits.loc[splits["partition"].eq("val"), "id"].iloc[0])
    images = pd.read_csv(catalogue["image_catalogue"])
    incomplete = prepared_project.processed / "high_resolution/incomplete.csv"
    images[images["id"].ne(val_id)].to_csv(incomplete, index=False)
    with pytest.raises(ValueError, match="missing for eligible labelled IDs"):
        build_training_variant_manifest(
            splits_csv=prepared_project.splits,
            high_resolution_catalogue_csv=incomplete,
            official_prediction_csv=prepared_project.prediction_csv,
            output_csv=prepared_project.processed / "incomplete_variants.csv",
            summary_output=prepared_project.processed / "incomplete_summary.json",
            root=prepared_project.root,
        )
