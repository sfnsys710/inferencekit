"""Image input handling and validation."""

import io
import logging
from pathlib import Path
from typing import Union

import requests
from PIL import Image

from ..schemas.config import Settings

logger = logging.getLogger(__name__)


class ImageHandler:
    """Handles image loading and validation from URLs or file paths."""

    def __init__(self, settings: Settings):
        """Initialize ImageHandler.

        Args:
            settings: Application settings for validation limits
        """
        self.settings = settings

    def load_from_url(self, url: str) -> Image.Image:
        """Load and validate image from URL.

        Args:
            url: HTTP(S) URL to image

        Returns:
            Validated PIL Image

        Raises:
            requests.RequestException: If download fails
            ValueError: If image is invalid, too large, or unsupported format
        """
        logger.debug(f"Downloading image from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        except requests.RequestException as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load image from URL response: {e}")
            raise ValueError(f"Invalid image data from URL: {e}")

        # Validate after loading
        self._validate_format(image)
        self._validate_size(image)

        logger.debug(f"Image loaded successfully: {image.size} {image.format}")
        return image

    def load_from_path(self, path: Union[str, Path]) -> Image.Image:
        """Load and validate image from file path.

        Args:
            path: Path to image file (string or Path object)

        Returns:
            Validated PIL Image

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If image is invalid, too large, or unsupported format
        """
        path_obj = Path(path) if isinstance(path, str) else path

        if not path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {path_obj}")

        logger.debug(f"Loading image from file: {path_obj}")
        try:
            image = Image.open(path_obj)
        except Exception as e:
            logger.error(f"Failed to load image from {path_obj}: {e}")
            raise ValueError(f"Invalid image file: {e}")

        # Validate after loading
        self._validate_format(image)
        self._validate_size(image)

        logger.debug(f"Image loaded successfully: {image.size} {image.format}")
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
