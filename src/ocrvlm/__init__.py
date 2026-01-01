"""
ocrvlm - OCR using Vision Language Models

Main public API for the ocrvlm library.
Provides building blocks for OCR with GOT-OCR-2.0-hf model.
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
