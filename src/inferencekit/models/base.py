"""Abstract base class for inference models."""

from abc import ABC, abstractmethod
from typing import Any

from ..schemas.config import Settings


class BaseModel(ABC):
    """Abstract base class for inference models.

    Defines the interface that all model implementations must follow.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model and processor/tokenizer into memory.

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
    def generate(self, input_data: Any, settings: Settings) -> str:
        """Run inference.

        Args:
            input_data: Input for the model (image, text, etc.)
            settings: Application settings including generation parameters

        Returns:
            Generated output text

        Raises:
            Exception: If model is not loaded or inference fails
        """
        pass
