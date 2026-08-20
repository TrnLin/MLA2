from __future__ import annotations

import pytest
from PIL import Image

from fashion.eda.notebook import figure_html, validate_exported_image_descriptions


def test_figure_html_preserves_alt_text_for_nbconvert(tmp_path):
    path = tmp_path / "figure.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path)

    rendered = figure_html(path, "A useful description").data

    assert 'alt="A useful description"' in rendered
    assert 'src="data:image/png;base64,' in rendered
    validate_exported_image_descriptions(rendered)


def test_export_validation_rejects_missing_or_placeholder_descriptions():
    with pytest.raises(ValueError, match="2 image"):
        validate_exported_image_descriptions(
            '<img src="one.png"><img src="two.png" alt="No description has been provided">'
        )


def test_figure_html_rejects_empty_alt_text(tmp_path):
    path = tmp_path / "figure.png"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(ValueError, match="must not be empty"):
        figure_html(path, "  ")
