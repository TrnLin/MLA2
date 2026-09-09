"""Content-addressed original uploads. Files are retained, never auto-deleted."""
from __future__ import annotations

import hashlib
import io
import re
import warnings
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

MIME_FORMATS = {'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP'}


class UploadStore:
    def __init__(self, directory: Path, *, max_bytes: int, max_pixels: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels

    def save(self, data: bytes, content_type: str) -> tuple[str, Path]:
        if len(data) > self.max_bytes:
            raise HTTPException(413, 'Image must be at most 10 MiB.')
        content_type = content_type.split(';', 1)[0].strip().lower()
        if content_type not in MIME_FORMATS:
            raise HTTPException(415, 'Use a JPG, PNG, or WebP image.')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != MIME_FORMATS[content_type]:
                        raise HTTPException(415, 'Image format does not match its content type.')
                    if image.width * image.height > self.max_pixels:
                        raise HTTPException(413, 'Decoded image is too large.')
                    image.verify()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise HTTPException(413, 'Decoded image is too large.') from error
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise HTTPException(422, 'Image is corrupt or cannot be read.') from error
        image_id = hashlib.sha256(data).hexdigest()
        path = self.directory / image_id
        try:
            with path.open('xb') as output:
                output.write(data)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise HTTPException(503, 'Stored image failed its integrity check.')
        return image_id, path

    def get(self, image_id: str) -> Path:
        if re.fullmatch('[0-9a-f]{64}', image_id) is None:
            raise HTTPException(404, 'Unknown image.')
        path = self.directory / image_id
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, 'Unknown image.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != image_id:
            raise HTTPException(503, 'Stored image failed its integrity check.')
        return path
