"""
inferencekit - Model inference toolkit

Main public API for the inferencekit library.
Provides building blocks for running inference on various models (OCR, text generation, etc.).
"""

from .content.image import ImageHandler
from .models.base import BaseOCRModel
from .models.got_ocr import GOTOCRModel
from .schemas.config import Settings, get_settings
from .schemas.output import OCRResult

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "Settings",
    "get_settings",
    # Core components
    "ImageHandler",
    # Models
    "BaseOCRModel",
    "GOTOCRModel",
    # Schemas
    "OCRResult",
]
