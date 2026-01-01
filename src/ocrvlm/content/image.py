"""Image input handling and validation."""

import io
import logging
from pathlib import Path

import requests
from PIL import Image

from ..config import Settings
from ..schemas.input import ImageInput

logger = logging.getLogger(__name__)


class ImageHandler:
    """Handles image input validation and loading from various sources.

    Supports loading images from:
    - URLs (http/https)
    - File paths (local filesystem)
    - PIL Images (already loaded)
    - Raw bytes
    """

    def __init__(self, settings: Settings):
        """Initialize ImageHandler.

        Args:
            settings: Application settings for validation limits
        """
        self.settings = settings

    def validate_and_load(self, image_input: ImageInput) -> Image.Image:
        """Validate and load image from any source type.

        Args:
            image_input: ImageInput specifying source and type

        Returns:
            Loaded PIL Image

        Raises:
            ValueError: If image is invalid, too large, or unsupported format
            FileNotFoundError: If file path doesn't exist
            requests.RequestException: If URL download fails
        """
        logger.debug(f"Loading image from {image_input.source_type}")

        # Load based on source type
        if image_input.source_type == "url":
            image = self._load_from_url(str(image_input.source))
        elif image_input.source_type == "path":
            image = self._load_from_path(image_input.source)  # type: ignore
        elif image_input.source_type == "pil":
            image = self._validate_pil(image_input.source)  # type: ignore
        elif image_input.source_type == "bytes":
            image = self._load_from_bytes(image_input.source)  # type: ignore
        else:
            raise ValueError(f"Unknown source type: {image_input.source_type}")

        # Validate after loading
        self._validate_format(image)
        self._validate_size(image)

        logger.debug(f"Image loaded successfully: {image.size} {image.format}")
        return image

    def _load_from_url(self, url: str) -> Image.Image:
        """Load image from URL.

        Args:
            url: HTTP(S) URL to image

        Returns:
            Loaded PIL Image

        Raises:
            requests.RequestException: If download fails
            ValueError: If response is not a valid image
        """
        logger.debug(f"Downloading image from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image
        except requests.RequestException as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load image from URL response: {e}")
            raise ValueError(f"Invalid image data from URL: {e}")

    def _load_from_path(self, path: Path) -> Image.Image:
        """Load image from file path.

        Args:
            path: Path to image file

        Returns:
            Loaded PIL Image

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid image
        """
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        logger.debug(f"Loading image from file: {path}")
        try:
            image = Image.open(path)
            return image
        except Exception as e:
            logger.error(f"Failed to load image from {path}: {e}")
            raise ValueError(f"Invalid image file: {e}")

    def _load_from_bytes(self, data: bytes) -> Image.Image:
        """Load image from raw bytes.

        Args:
            data: Raw image data

        Returns:
            Loaded PIL Image

        Raises:
            ValueError: If bytes are not a valid image
        """
        logger.debug(f"Loading image from {len(data)} bytes")
        try:
            image = Image.open(io.BytesIO(data))
            return image
        except Exception as e:
            logger.error(f"Failed to load image from bytes: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def _validate_pil(self, image: Image.Image) -> Image.Image:
        """Validate an already-loaded PIL Image.

        Args:
            image: PIL Image to validate

        Returns:
            The same image if valid

        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        return image

    def _validate_format(self, image: Image.Image) -> None:
        """Check if image format is supported.

        Args:
            image: PIL Image to validate

        Raises:
            ValueError: If format is not supported
        """
        if image.format is None:
            logger.warning("Image format is None, skipping format validation")
            return

        format_lower = image.format.lower()
        if format_lower not in self.settings.supported_formats:
            raise ValueError(
                f"Unsupported image format: {image.format}. "
                f"Supported: {', '.join(self.settings.supported_formats)}"
            )

    def _validate_size(self, image: Image.Image) -> None:
        """Check if image size is within limits.

        Args:
            image: PIL Image to validate

        Raises:
            ValueError: If image is too large
        """
        # Estimate size in bytes (width * height * channels * bytes_per_channel)
        # This is approximate - actual file size depends on compression
        width, height = image.size
        channels = len(image.getbands())
        estimated_size_mb = (width * height * channels * 1) / (1024 * 1024)

        if estimated_size_mb > self.settings.max_image_size_mb:
            raise ValueError(
                f"Image too large: {estimated_size_mb:.2f}MB "
                f"(max: {self.settings.max_image_size_mb}MB)"
            )
