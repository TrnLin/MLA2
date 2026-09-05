"""Check diagnostic image integrity and saved checkpoint metadata contracts."""

import hashlib
from copy import deepcopy

import pandas as pd
import pytest

from fashion.train.task3_gender_diagnostic import verify_checkpoint_metadata, verify_input_images


def test_image_verification_rejects_same_length_content_changes(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"original")
    frame = pd.DataFrame(
        [{"id": 1, "path": image.name, "sha256": hashlib.sha256(b"original").hexdigest()}]
    )
    assert verify_input_images(frame, tmp_path) == 1
    image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="differs from canonical split"):
        verify_input_images(frame, tmp_path)
    image.unlink()
    with pytest.raises(ValueError, match="differs from canonical split"):
        verify_input_images(frame, tmp_path)


@pytest.fixture
def checkpoint_metadata():
    stats = {
        "channels": 3,
        "mean": [0.8, 0.7, 0.6],
        "std": [0.2, 0.3, 0.4],
        "total_pixels": 4800,
    }
    checkpoint = {
        "run_id": "saved_gender_fold_0",
        "config": {"channels": [32, 64, 128, 256]},
        "class_names": ["Boys", "Girls", "Men", "Unisex", "Women"],
        "normalization": stats,
    }
    source = {
        "run_id": checkpoint["run_id"],
        "config": deepcopy(checkpoint["config"]),
        "classes": checkpoint["class_names"].copy(),
        "normalization": {
            **deepcopy(stats),
            "fit_scope": "fold_training_content_pixels_only",
            "validation_fold": 0,
            "padding_excluded": True,
            "input_view": "full",
        },
    }
    return checkpoint, source


def test_checkpoint_accepts_statistics_with_json_scope_notes(checkpoint_metadata):
    checkpoint, source = checkpoint_metadata
    verify_checkpoint_metadata(checkpoint, **source)


@pytest.mark.parametrize("field", ["channels", "mean", "std", "total_pixels"])
def test_checkpoint_rejects_changed_statistics(checkpoint_metadata, field):
    checkpoint, source = checkpoint_metadata
    checkpoint["normalization"][field] = 999
    with pytest.raises(ValueError, match="field=normalization"):
        verify_checkpoint_metadata(checkpoint, **source)


@pytest.mark.parametrize("field", ["run_id", "config", "class_names", "normalization"])
@pytest.mark.parametrize("missing", [False, True])
def test_checkpoint_rejects_changed_or_missing_metadata(checkpoint_metadata, field, missing):
    checkpoint, source = checkpoint_metadata
    if missing:
        del checkpoint[field]
    else:
        checkpoint[field] = "different"
    with pytest.raises(ValueError, match=f"field={field}"):
        verify_checkpoint_metadata(checkpoint, **source)
