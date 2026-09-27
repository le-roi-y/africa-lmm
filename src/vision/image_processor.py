from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from PIL import Image


class ImageProcessor:
    SUPPORTED_FORMATS: ClassVar[set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tiff",
        ".bmp",
    }

    def load(self, path: str | Path) -> Image.Image:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        with Image.open(path) as image:
            return image.copy()

    def resize(self, image: Image.Image, max_size: int = 2048) -> Image.Image:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0.")

        resized = image.copy()
        resized.thumbnail((max_size, max_size))

        return resized
