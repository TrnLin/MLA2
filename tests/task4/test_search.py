from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import fashion.task4 as task4
import fashion.task4.search as search_module
from fashion.data.hashing import compute_sha256
from fashion.task4.gallery_artifact import TeacherGallery
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.search import (
    CropBox,
    SearchBundle,
    SearchHit,
    load_search_bundle,
    run_search,
)


class FakeEncoder(nn.Module):
    def __init__(self, embedding: np.ndarray | None = None) -> None:
        super().__init__()
        values = np.zeros(128, dtype=np.float32)
        values[0] = 1.0
        self.embedding = values if embedding is None else embedding
        self.batches: list[torch.Tensor] = []

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        self.batches.append(batch.detach().cpu().clone())
        return torch.from_numpy(self.embedding).to(batch.device).unsqueeze(0)


def _split_row(
    product_id: int,
    *,
    fold: int | None,
    path: str,
    digest: str,
    article_type: str,
    base_colour: str,
    partition: str = "development",
) -> dict[str, object]:
    return {
        "id": product_id,
        "path": path,
        "sha256": digest,
        "articleType": article_type,
        "baseColour": base_colour,
        "productDisplayName": f"Product {product_id}",
        "duplicate_group": f"duplicate-{product_id}",
        "product_name_key": f"product-{product_id}",
        "product_family_group": f"family-{product_id}",
        "partition": partition,
        "cv_fold": fold,
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "quarantine_reason": "",
        "season": "Summer",
        "gender": "Unisex",
        "usage": "Casual",
        "has_articleType_label": True,
        "has_season_label": True,
        "has_gender_label": True,
        "has_usage_label": True,
    }


def _gallery(tmp_path: Path) -> TeacherGallery:
    ids = np.array([20, 30, 40], dtype=np.int64)
    features = np.zeros((3, 128), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, :2] = [0.8, 0.6]
    features[2, 1] = 1.0
    metadata = pd.DataFrame(
        [
            _split_row(
                20,
                fold=0,
                path="data/raw/teacher/train/images/20.jpg",
                digest="2" * 64,
                article_type="Tshirts",
                base_colour="Blue",
            ),
            _split_row(
                30,
                fold=2,
                path="data/raw/teacher/train/images/30.jpg",
                digest="3" * 64,
                article_type="Tshirts",
                base_colour="Red",
            ),
            _split_row(
                40,
                fold=3,
                path="data/raw/teacher/train/images/40.jpg",
                digest="4" * 64,
                article_type="Jeans",
                base_colour="Blue",
            ),
        ]
    )
    return TeacherGallery(
        directory=tmp_path / "gallery",
        ids=ids,
        features=features,
        metadata=metadata,
        manifest={
            "artifact_identity_sha256": "a" * 64,
            "r5_checkpoint": {"run_id": "task4-r5-test", "sha256": "c" * 64},
            "fold": 1,
            "rows": 3,
            "contract": PreprocessingContract(240, 320).to_dict(),
        },
        identity_sha256="a" * 64,
    )


@pytest.fixture
def bundle(tmp_path: Path) -> SearchBundle:
    known_path = tmp_path / "known.png"
    Image.new("RGB", (75, 100), (64, 96, 128)).save(known_path)
    known = _split_row(
        10,
        fold=1,
        path=str(known_path),
        digest=compute_sha256(known_path),
        article_type="Tshirts",
        base_colour="Blue",
    )
    outside_fold = _split_row(
        11,
        fold=0,
        path=str(tmp_path / "outside-fold.png"),
        digest="1" * 64,
        article_type="Tshirts",
        base_colour="Blue",
    )
    gallery = _gallery(tmp_path)
    splits = pd.concat(
        [
            pd.DataFrame([known, outside_fold]),
            gallery.metadata,
            pd.DataFrame(
                [
                    _split_row(
                        99,
                        fold=None,
                        path="redacted",
                        digest="9" * 64,
                        article_type="Unknown",
                        base_colour="Unknown",
                        partition="holdout",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    return SearchBundle(
        model=FakeEncoder(),
        device=torch.device("cpu"),
        contract=PreprocessingContract(240, 320),
        teacher_mean=(0.5, 0.5, 0.5),
        teacher_std=(0.25, 0.25, 0.25),
        model_manifest={
            "method": "R5",
            "architecture": "resnet18",
            "source_checkpoint": {
                "run_id": "task4-r5-test",
                "sha256": "c" * 64,
            },
        },
        model_manifest_sha256="b" * 64,
        gallery=gallery,
        query_catalogue=pd.DataFrame([known]),
        splits=splits.drop(columns=["path"]),
    )


def _save_image(path: Path, size: tuple[int, int] = (100, 80)) -> Path:
    Image.new("RGBA", size, (255, 0, 0, 0)).save(path)
    return path


def test_outside_search_prepares_encodes_ranks_and_redacts(
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    outside_path = _save_image(tmp_path / "outside.png")

    response = run_search(
        bundle,
        image_path=outside_path,
        crop=CropBox(10, 5, 70, 75),
        top_k=2,
        rating="mixed",
        note="background remains visible",
    )

    query = response.record.query
    assert query.kind == "outside"
    assert query.known_id is None
    assert query.source_sha256 == compute_sha256(outside_path)
    assert query.source_dimensions == (100, 80)
    assert query.effective_dimensions == (60, 70)
    assert query.crop == CropBox(10, 5, 70, 75)
    assert query.preprocessed.pixels.shape == (320, 240, 3)
    assert query.preprocessed.pixels.dtype == np.uint8
    assert np.all(query.preprocessed.pixels[query.preprocessed.content_mask] == 255)
    assert response.record.rating == "mixed"
    assert response.record.note == "background remains visible"
    assert [hit.rank for hit in response.record.results] == [1, 2]
    assert all(hit.grade is None for hit in response.record.results)
    assert response.result_metadata["id"].tolist() == [20, 30]

    payload = response.record.to_dict()
    assert set(payload["query"]) == {
        "kind",
        "known_id",
        "source_sha256",
        "source_dimensions",
        "effective_dimensions",
        "crop",
        "aspect_ratio",
        "content_fraction",
        "warnings",
    }
    assert "source_path" not in payload["query"]
    assert "preprocessed" not in payload["query"]
    assert "known_row" not in payload["query"]
    json.dumps(payload, allow_nan=False)
    assert response.record.query_key.startswith(
        f"outside-{compute_sha256(outside_path)[:12]}-"
    )

    model = bundle.model
    assert isinstance(model, FakeEncoder)
    assert len(model.batches) == 1
    model_input = model.batches[0]
    assert model_input.dtype == torch.float32
    assert tuple(model_input.shape) == (1, 3, 320, 240)
    padding = torch.from_numpy(~query.preprocessed.content_mask)
    assert torch.all(model_input[0, :, padding] == 0)


def test_known_fold_one_search_adds_literal_primary_grades(bundle: SearchBundle) -> None:
    response = run_search(bundle, query_id=10, top_k=3)

    assert response.record.query.kind == "known"
    assert response.record.query.known_id == 10
    assert [hit.candidate_id for hit in response.record.results] == [20, 30, 40]
    assert [hit.grade for hit in response.record.results] == [2, 1, 0]
    assert response.record.rating is None
    assert response.record.note is None
    assert response.record.query_key.startswith("known-10-")
    assert "source_path" not in response.record.to_dict()["query"]


def test_search_hit_maps_exact_persisted_names() -> None:
    hit = SearchHit(
        rank=1,
        candidate_id=20,
        distance=0.25,
        article_type="Tshirts",
        base_colour="Blue",
        product_display_name="Blue tee",
        grade=2,
    )

    assert hit.to_dict() == {
        "rank": 1,
        "candidate_id": 20,
        "distance": 0.25,
        "articleType": "Tshirts",
        "baseColour": "Blue",
        "productDisplayName": "Blue tee",
        "grade": 2,
    }


@pytest.mark.parametrize(
    ("image_path", "query_id"),
    [(None, None), (Path("outside.png"), 10)],
)
def test_search_requires_exactly_one_query_source(
    bundle: SearchBundle,
    image_path: Path | None,
    query_id: int | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        run_search(bundle, image_path=image_path, query_id=query_id)


def test_search_rejects_known_id_outside_fixed_fold(bundle: SearchBundle) -> None:
    with pytest.raises(ValueError, match="fold 1"):
        run_search(bundle, query_id=11)


def test_protected_path_is_rejected_before_file_or_decode_checks(
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    protected = tmp_path / "teacher" / "test" / "missing.png"

    with pytest.raises(ValueError, match="teacher-test path"):
        run_search(bundle, image_path=protected)


def test_copied_protected_bytes_are_rejected_before_decode(
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied.bin"
    copied.write_bytes(b"sealed image bytes, not an image")
    digest = compute_sha256(copied)
    bundle.splits.loc[bundle.splits["id"].eq(99), "sha256"] = digest

    with pytest.raises(ValueError, match="protected image hash"):
        run_search(bundle, image_path=copied)


def test_search_decodes_the_same_open_file_bytes_that_passed_hash_safety(
    monkeypatch: pytest.MonkeyPatch,
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.png"
    protected = tmp_path / "protected.png"
    Image.new("RGB", (75, 100), (255, 255, 255)).save(outside)
    Image.new("RGB", (75, 100), (0, 0, 0)).save(protected)
    safe_digest = compute_sha256(outside)
    bundle.splits.loc[bundle.splits["id"].eq(99), "sha256"] = compute_sha256(
        protected
    )
    original_reject = search_module.reject_protected_image_sha256

    def replace_path_after_hash_check(digest: str, splits: pd.DataFrame) -> None:
        original_reject(digest, splits)
        protected.replace(outside)

    monkeypatch.setattr(
        search_module,
        "reject_protected_image_sha256",
        replace_path_after_hash_check,
    )

    response = run_search(bundle, image_path=outside, top_k=1)

    query = response.record.query
    assert query.source_sha256 == safe_digest
    assert np.all(query.preprocessed.pixels[query.preprocessed.content_mask] == 255)


@pytest.mark.parametrize(
    "coordinates",
    [
        (-1, 0, 1, 1),
        (0, -1, 1, 1),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0.5, 0, 1, 1),
        (True, 0, 1, 1),
    ],
)
def test_crop_box_rejects_non_integer_unordered_or_negative_coordinates(
    coordinates: tuple[object, object, object, object],
) -> None:
    with pytest.raises(ValueError, match="crop"):
        CropBox(*coordinates)  # type: ignore[arg-type]


def test_crop_must_fit_inside_oriented_source_bounds(
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    outside = _save_image(tmp_path / "outside.png", (20, 10))

    with pytest.raises(ValueError, match="bounds"):
        run_search(bundle, image_path=outside, crop=CropBox(0, 0, 21, 10))


@pytest.mark.parametrize(
    ("rating", "note", "message"),
    [
        ("good", None, "both"),
        (None, "useful", "both"),
        ("excellent", "useful", "rating"),
        ("bad", "   ", "blank"),
    ],
)
def test_outside_rating_and_note_are_validated_together(
    bundle: SearchBundle,
    tmp_path: Path,
    rating: str | None,
    note: str | None,
    message: str,
) -> None:
    outside = _save_image(tmp_path / "outside.png")

    with pytest.raises(ValueError, match=message):
        run_search(bundle, image_path=outside, rating=rating, note=note)


def test_known_query_rejects_rating_and_note(bundle: SearchBundle) -> None:
    with pytest.raises(ValueError, match="known"):
        run_search(bundle, query_id=10, rating="good", note="clear match")


@pytest.mark.parametrize("top_k", [0, 21, True, 1.5])
def test_search_rejects_k_outside_integer_one_through_twenty(
    bundle: SearchBundle,
    tmp_path: Path,
    top_k: object,
) -> None:
    outside = _save_image(tmp_path / "outside.png")

    with pytest.raises(ValueError, match="1 through 20"):
        run_search(bundle, image_path=outside, top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("size", "expected_warnings"),
    [
        ((751, 1000), ()),
        ((761, 1000), ("unusual_aspect_ratio",)),
        (
            (200, 20),
            ("unusual_aspect_ratio", "heavy_letterbox_padding"),
        ),
    ],
)
def test_query_warnings_use_fixed_aspect_and_content_thresholds(
    bundle: SearchBundle,
    tmp_path: Path,
    size: tuple[int, int],
    expected_warnings: tuple[str, ...],
) -> None:
    outside = _save_image(tmp_path / f"{size[0]}-{size[1]}.png", size)

    response = run_search(bundle, image_path=outside, top_k=1)

    assert response.record.query.warnings == expected_warnings


@pytest.mark.parametrize(
    "embedding",
    [
        np.zeros(127, dtype=np.float32),
        np.full(128, np.nan, dtype=np.float32),
        np.full(128, 0.5, dtype=np.float32),
    ],
    ids=["wrong-shape", "nonfinite", "not-unit"],
)
def test_search_rejects_invalid_model_embeddings(
    bundle: SearchBundle,
    tmp_path: Path,
    embedding: np.ndarray,
) -> None:
    outside = _save_image(tmp_path / "outside.png")
    object.__setattr__(bundle, "model", FakeEncoder(embedding))

    with pytest.raises(ValueError, match="128|finite|unit"):
        run_search(bundle, image_path=outside)


def _portable_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "task4_r5_inference_package",
        "method": "R5",
        "architecture": "resnet18",
        "objective": "content_mask_mse",
        "pretrained": False,
        "weight_origin": "trained_from_scratch",
        "embedding_dim": 128,
        "source_checkpoint": {
            "run_id": "task4-r5-test",
            "epoch": 1,
            "score": 0.5,
            "sha256": "c" * 64,
        },
        "weights": {
            "path": "weights.pt",
            "sha256": "d" * 64,
            "bytes": 123,
        },
        "input_contract": PreprocessingContract(240, 320).to_dict(),
        "normalization": {
            "teacher": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
            "v1": {
                "mean": [0.6, 0.6, 0.6],
                "std": [0.3, 0.3, 0.3],
            },
        },
        "loader": {
            "module": "fashion.task4",
            "function": "load_r5_inference_package",
        },
    }


def test_bundle_loader_checks_identities_builds_fixed_views_and_redacts_paths(
    monkeypatch: pytest.MonkeyPatch,
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    package = tmp_path / "task4_r5"
    package.mkdir()
    manifest_path = package / "manifest.json"
    manifest_path.write_text(json.dumps(_portable_manifest()), encoding="utf-8")
    model = FakeEncoder()
    calls: dict[str, object] = {}

    def fake_model_loader(path: Path, *, device: torch.device | str) -> nn.Module:
        calls["model"] = (Path(path), torch.device(device))
        return model

    def fake_gallery_loader(path: Path) -> TeacherGallery:
        calls["gallery"] = Path(path)
        return bundle.gallery

    loader_splits = pd.concat(
        [bundle.query_catalogue, bundle.gallery.metadata],
        ignore_index=True,
    )
    monkeypatch.setattr(search_module, "load_r5_inference_package", fake_model_loader)
    monkeypatch.setattr(search_module, "load_teacher_gallery_artifact", fake_gallery_loader)
    monkeypatch.setattr(search_module, "load_splits", lambda path: loader_splits)

    loaded = load_search_bundle(
        model_package=package,
        gallery_directory=tmp_path / "gallery",
        splits_path=tmp_path / "splits.csv",
    )

    assert calls == {
        "model": (package, torch.device("cpu")),
        "gallery": tmp_path / "gallery",
    }
    assert loaded.model is model
    assert loaded.model.training is False
    assert loaded.device == torch.device("cpu")
    assert loaded.contract == PreprocessingContract(240, 320)
    assert loaded.teacher_mean == (0.5, 0.5, 0.5)
    assert loaded.teacher_std == (0.25, 0.25, 0.25)
    assert loaded.model_manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert loaded.query_catalogue["id"].tolist() == [10]
    assert loaded.gallery.ids.tolist() == [20, 30, 40]
    assert "path" not in loaded.splits
    assert set(loaded.splits) == {"id", "sha256", "partition", "cv_fold"}


def test_bundle_loader_rejects_manifest_changed_while_model_loads(
    monkeypatch: pytest.MonkeyPatch,
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    package = tmp_path / "task4_r5"
    package.mkdir()
    manifest_path = package / "manifest.json"
    manifest = _portable_manifest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loader_splits = pd.concat(
        [bundle.query_catalogue, bundle.gallery.metadata],
        ignore_index=True,
    )

    def mutate_manifest_while_loading(
        path: Path,
        *,
        device: torch.device | str,
    ) -> nn.Module:
        changed = _portable_manifest()
        changed["normalization"]["teacher"]["mean"] = [0.1, 0.2, 0.3]
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        return FakeEncoder()

    monkeypatch.setattr(
        search_module,
        "load_r5_inference_package",
        mutate_manifest_while_loading,
    )
    monkeypatch.setattr(
        search_module,
        "load_teacher_gallery_artifact",
        lambda path: bundle.gallery,
    )
    monkeypatch.setattr(search_module, "load_splits", lambda path: loader_splits)

    with pytest.raises(ValueError, match="manifest.*changed"):
        load_search_bundle(
            model_package=package,
            gallery_directory=tmp_path / "gallery",
            splits_path=tmp_path / "splits.csv",
        )


def test_bundle_loader_rejects_gallery_from_another_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    bundle: SearchBundle,
    tmp_path: Path,
) -> None:
    package = tmp_path / "task4_r5"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(_portable_manifest()),
        encoding="utf-8",
    )
    bad_manifest = dict(bundle.gallery.manifest)
    bad_manifest["r5_checkpoint"] = {
        "run_id": "other",
        "sha256": "e" * 64,
    }
    bad_gallery = TeacherGallery(
        directory=bundle.gallery.directory,
        ids=bundle.gallery.ids,
        features=bundle.gallery.features,
        metadata=bundle.gallery.metadata,
        manifest=bad_manifest,
        identity_sha256=bundle.gallery.identity_sha256,
    )
    monkeypatch.setattr(
        search_module,
        "load_r5_inference_package",
        lambda *args, **kwargs: FakeEncoder(),
    )
    monkeypatch.setattr(
        search_module,
        "load_teacher_gallery_artifact",
        lambda *args, **kwargs: bad_gallery,
    )
    monkeypatch.setattr(
        search_module,
        "load_splits",
        lambda path: bundle.splits.assign(path="safe.png"),
    )

    with pytest.raises(ValueError, match="checkpoint"):
        load_search_bundle(
            model_package=package,
            gallery_directory=tmp_path / "gallery",
            splits_path=tmp_path / "splits.csv",
        )


def test_bundle_loader_rejects_unavailable_cuda_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA"):
        load_search_bundle(
            model_package=tmp_path / "missing-model",
            gallery_directory=tmp_path / "missing-gallery",
            splits_path=tmp_path / "missing-splits.csv",
            device="cuda",
        )


def test_search_api_is_public() -> None:
    assert task4.CropBox is CropBox
    assert task4.SearchBundle is SearchBundle
    assert task4.SearchHit is SearchHit
    assert task4.load_search_bundle is load_search_bundle
    assert task4.run_search is run_search
