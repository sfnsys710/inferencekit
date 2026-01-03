"""
inferencekit - Model inference toolkit

Main public API for the inferencekit library.
Provides building blocks for running inference on various models (OCR, text generation, etc.).
"""

from .content.image import ImageHandler
from .models.base import BaseModel
from .models.got_ocr import GOTOCRModel
from .models.qwen3 import Qwen3Model
from .schemas.config import Settings, get_settings
from .schemas.output import OCRResult, TextResult

__version__ = "0.2.0"

__all__ = [
    # Configuration
    "Settings",
    "get_settings",
    # Core components
    "ImageHandler",
    # Models
    "BaseModel",
    "GOTOCRModel",
    "Qwen3Model",
    # Schemas
    "OCRResult",
    "TextResult",
]
