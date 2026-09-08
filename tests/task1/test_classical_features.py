from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.task1.classical_features import (
    TASK1_HOG_COARSE,
    TASK1_HOG_FINE,
    Task1HogSpec,
    extract_task1_hog,
    load_or_build_task1_hog_features,
)
from fashion.task1.image_contract import TASK1_IMAGE_SIZE


def test_hog_specs_have_fixed_ids_and_feature_lengths() -> None:
    assert TASK1_HOG_COARSE.hog_id == "task1_gray_hog_ppc16_v1"
    assert TASK1_HOG_COARSE.expected_features == 288
    assert TASK1_HOG_COARSE.pixels_per_cell == (16, 16)
    assert TASK1_HOG_FINE.hog_id == "task1_gray_hog_ppc10_v1"
    assert TASK1_HOG_FINE.expected_features == 1260
    assert TASK1_HOG_FINE.pixels_per_cell == (10, 10)
    for spec in (TASK1_HOG_COARSE, TASK1_HOG_FINE):
        assert spec.orientations == 9
        assert spec.cells_per_block == (2, 2)
        assert spec.block_norm == "L2-Hys"
        assert spec.transform_sqrt is True
        assert spec.image_size == TASK1_IMAGE_SIZE


@pytest.mark.parametrize("spec", [TASK1_HOG_COARSE, TASK1_HOG_FINE])
def test_extract_task1_hog_is_float32_deterministic_and_fixed_width(
    tmp_path: Path, spec: Task1HogSpec
) -> None:
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    pixels[20:60, 15:45] = (220, 80, 40)
    path = tmp_path / "sample.png"
    Image.fromarray(pixels, mode="RGB").save(path)

    first = extract_task1_hog(path, spec)
    second = extract_task1_hog(path, spec)

    assert first.shape == (spec.expected_features,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_hog_spec_rejects_unapproved_geometry() -> None:
    with pytest.raises(ValueError, match="expected feature count"):
        Task1HogSpec(hog_id="bad", pixels_per_cell=(8, 8), expected_features=1)


def _cache_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [11, 12],
            "path": ["images/11.jpg", "images/12.jpg"],
            "sha256": ["a" * 64, "b" * 64],
            "partition": ["development", "development"],
            "articleType": ["class-000", "class-001"],
        }
    )


def test_hog_cache_reuses_valid_ordered_float32_features(tmp_path: Path) -> None:
    calls: list[int] = []

    def extractor(path: Path, spec: Task1HogSpec) -> np.ndarray:
        calls.append(int(path.stem))
        return np.full(spec.expected_features, int(path.stem), dtype=np.float32)

    first = load_or_build_task1_hog_features(
        _cache_rows(), TASK1_HOG_COARSE, root=tmp_path, cache_root=tmp_path / "cache",
        split_sha256="c" * 64, extractor=extractor,
    )
    second = load_or_build_task1_hog_features(
        _cache_rows(), TASK1_HOG_COARSE, root=tmp_path, cache_root=tmp_path / "cache",
        split_sha256="c" * 64, extractor=lambda *_: pytest.fail("valid cache should be reused"),
    )

    assert calls == [11, 12]
    np.testing.assert_array_equal(first.ids, np.array([11, 12]))
    np.testing.assert_array_equal(first.features, second.features)
    assert second.features.dtype == np.float32


def test_hog_cache_rebuilds_when_source_inventory_changes(tmp_path: Path) -> None:
    rows = _cache_rows()
    load_or_build_task1_hog_features(rows, TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32))
    changed = rows.copy()
    changed.loc[0, "sha256"] = "d" * 64
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    rebuilt = load_or_build_task1_hog_features(changed, TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64, extractor=replacement)

    assert calls == 2
    assert np.all(rebuilt.features == 1.0)


def test_hog_cache_rebuilds_when_split_identity_changes(tmp_path: Path) -> None:
    load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32),
    )
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    rebuilt = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="d" * 64,
        extractor=replacement,
    )

    assert calls == 2
    assert np.all(rebuilt.features == 1.0)


def test_hog_cache_rebuilds_when_hog_config_changes(tmp_path: Path) -> None:
    load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32),
    )
    changed_config = Task1HogSpec(
        hog_id=TASK1_HOG_COARSE.hog_id,
        pixels_per_cell=TASK1_HOG_COARSE.pixels_per_cell,
        expected_features=TASK1_HOG_COARSE.expected_features,
        transform_sqrt=False,
    )
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    rebuilt = load_or_build_task1_hog_features(
        _cache_rows(),
        changed_config,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=replacement,
    )

    assert calls == 2
    assert np.all(rebuilt.features == 1.0)


def test_hog_cache_rebuilds_an_invalid_cached_feature_shape(tmp_path: Path) -> None:
    first = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32),
    )
    with np.load(first.cache_path, allow_pickle=False) as stored:
        metadata = stored["metadata"]
    np.savez_compressed(
        first.cache_path,
        ids=first.ids,
        features=np.zeros((len(first.ids), 1), dtype=np.float32),
        metadata=metadata,
    )
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    rebuilt = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=replacement,
    )

    assert calls == 2
    assert rebuilt.features.shape == (2, 288)
    assert np.all(rebuilt.features == 1.0)


def test_hog_cache_rebuilds_when_implementation_or_library_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = [{"implementation_sha256": "a" * 64, "scikit-image": "0.26.0"}]
    monkeypatch.setattr("fashion.task1.classical_features._hog_cache_implementation_identity",
        lambda: identity[0], raising=False)
    first = load_or_build_task1_hog_features(_cache_rows(), TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32))
    identity[0] = {"implementation_sha256": "b" * 64, "scikit-image": "0.27.0"}
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    second = load_or_build_task1_hog_features(_cache_rows(), TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64, extractor=replacement)

    assert calls == 2
    assert first.cache_path != second.cache_path
    assert np.all(second.features == 1.0)


def test_hog_cache_rebuilds_when_a_hashed_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = {
        "classical_features.py": "a" * 64,
        "image_contract.py": "b" * 64,
        "images.py": "c" * 64,
    }

    def source_hash(path: Path) -> str:
        return hashes[path.name]

    monkeypatch.setattr("fashion.task1.classical_features.compute_sha256", source_hash)
    first = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32),
    )
    hashes["image_contract.py"] = "d" * 64
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    second = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=replacement,
    )

    assert calls == 2
    assert first.cache_path != second.cache_path
    assert np.all(second.features == 1.0)


def test_hog_cache_rebuilds_when_a_producing_package_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages = {"numpy": "1.0.0", "Pillow": "1.0.0", "scikit-image": "1.0.0"}
    monkeypatch.setattr("fashion.task1.classical_features.version", packages.__getitem__)
    first = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32),
    )
    packages["scikit-image"] = "2.0.0"
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    second = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=replacement,
    )

    assert calls == 2
    assert first.cache_path != second.cache_path
    assert np.all(second.features == 1.0)


def test_hog_cache_rejects_non_development_or_duplicate_rows(tmp_path: Path) -> None:
    rows = _cache_rows()
    rows.loc[1, "id"] = 11
    with pytest.raises(ValueError, match="unique development rows"):
        load_or_build_task1_hog_features(
            rows,
            TASK1_HOG_COARSE,
            root=tmp_path,
            cache_root=tmp_path / "cache",
            split_sha256="c" * 64,
        )


def test_hog_cache_rejects_holdout_rows(tmp_path: Path) -> None:
    rows = _cache_rows()
    rows.loc[1, "partition"] = "holdout"
    with pytest.raises(ValueError, match="unique development rows"):
        load_or_build_task1_hog_features(
            rows,
            TASK1_HOG_COARSE,
            root=tmp_path,
            cache_root=tmp_path / "cache",
            split_sha256="c" * 64,
        )


def test_hog_cache_rejects_ids_that_duplicate_after_integer_normalization(
    tmp_path: Path,
) -> None:
    rows = _cache_rows()
    rows["id"] = ["01", "1"]
    with pytest.raises(ValueError, match="unique development rows"):
        load_or_build_task1_hog_features(
            rows,
            TASK1_HOG_COARSE,
            root=tmp_path,
            cache_root=tmp_path / "cache",
            split_sha256="c" * 64,
            extractor=lambda *_: np.zeros(288, dtype=np.float32),
        )


def test_hog_cache_leaves_no_final_file_when_atomic_write_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted_write(*_: object) -> Path:
        raise RuntimeError("write stopped")

    monkeypatch.setattr("fashion.task1.classical_features.atomic_write_bytes", interrupted_write)
    with pytest.raises(RuntimeError, match="write stopped"):
        load_or_build_task1_hog_features(_cache_rows(), TASK1_HOG_COARSE, root=tmp_path,
            cache_root=tmp_path / "cache", split_sha256="c" * 64,
            extractor=lambda *_: np.zeros(288, dtype=np.float32))
    assert not list((tmp_path / "cache").glob("*.npz"))
