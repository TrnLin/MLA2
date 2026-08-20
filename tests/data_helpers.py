from __future__ import annotations

import io
from pathlib import Path
import random

from PIL import Image

METADATA_HEADER = (
    "id,gender,masterCategory,subCategory,articleType,baseColour,"
    "season,year,usage,productDisplayName"
)


def write_id_csv(path: Path, ids: list[str | int]) -> Path:
    """Write a minimal metadata file used by ID-only boundary tests."""
    path.write_text(
        "id\n" + "\n".join(str(identifier) for identifier in ids) + "\n",
        encoding="utf-8",
    )
    return path


def write_metadata(path: Path, rows: list[str], header: str = METADATA_HEADER) -> Path:
    """Write fashion metadata rows under the ten-column product header."""
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def save_image(path: Path, color: str = "blue") -> Path:
    """Save one tiny readable image, creating the parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color).save(path)
    return path


def save_truncated_jpeg(path: Path, kept_fraction: float = 0.6) -> Path:
    """Save a JPEG that passes structure checks but fails when pixels are decoded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = random.Random(2753)
    image = Image.new("RGB", (64, 64))
    image.putdata(
        [
            (noise.randrange(256), noise.randrange(256), noise.randrange(256))
            for _ in range(64 * 64)
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = buffer.getvalue()
    path.write_bytes(encoded[: int(len(encoded) * kept_fraction)])
    return path
