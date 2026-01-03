"""Abstract base class for OCR models."""

from abc import ABC, abstractmethod

from PIL import Image

from ..schemas.config import Settings


class BaseOCRModel(ABC):
    """Abstract base class for OCR models.

    Defines the interface that all OCR model implementations must follow.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model and processor into memory.

        Raises:
            Exception: If model loading fails
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Free model resources from memory."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded.

        Returns:
            True if model is loaded and ready for inference
        """
        pass

    @abstractmethod
    def generate(self, image: Image.Image, settings: Settings) -> str:
        """Run OCR inference on an image.

        Args:
            image: PIL Image to perform OCR on
            settings: Application settings including generation parameters

        Returns:
            Extracted text from the image

        Raises:
            Exception: If model is not loaded or inference fails
        """
        pass
