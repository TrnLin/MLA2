"""Check that the inference diagnostic rejects changed source images."""

import hashlib

import pandas as pd
import pytest

from fashion.train.task3_gender_diagnostic import verify_input_images


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
