"""Input schemas for image sources."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

from PIL import Image


@dataclass
class ImageInput:
    """Represents an image input source.

    Supports multiple source types: URL, file path, PIL Image, or raw bytes.
    """

    source: Union[str, Path, Image.Image, bytes]
    source_type: Literal["url", "path", "pil", "bytes"]

    @classmethod
    def from_url(cls, url: str) -> "ImageInput":
        """Create ImageInput from a URL string."""
        return cls(source=url, source_type="url")

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "ImageInput":
        """Create ImageInput from a file path."""
        return cls(source=Path(path) if isinstance(path, str) else path, source_type="path")

    @classmethod
    def from_pil(cls, image: Image.Image) -> "ImageInput":
        """Create ImageInput from a PIL Image."""
        return cls(source=image, source_type="pil")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ImageInput":
        """Create ImageInput from raw bytes."""
        return cls(source=data, source_type="bytes")
