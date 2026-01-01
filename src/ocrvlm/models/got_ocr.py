"""GOT-OCR-2.0-hf model implementation."""

import logging
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..schemas.config import Settings
from .base import BaseOCRModel

logger = logging.getLogger(__name__)


class GOTOCRModel(BaseOCRModel):
    """Implementation for GOT-OCR-2.0-hf model from HuggingFace.

    This model uses Vision-Language capabilities for OCR tasks.
    Based on validation in notebooks/stepfun-got-ocr.ipynb.
    """

    def __init__(self, settings: Settings):
        """Initialize GOT-OCR model.

        Args:
            settings: Application settings including model_id and device
        """
        self.settings = settings
        # Use Any type to avoid incomplete type stubs from transformers
        self.processor: Any = None
        self.model: Any = None
        self._loaded = False

    def load(self) -> None:
        """Load model and processor from HuggingFace.

        This operation takes approximately 26 seconds on M2.

        Raises:
            Exception: If model or processor loading fails
        """
        if self._loaded:
            logger.info("Model already loaded, skipping")
            return

        try:
            logger.info(f"Loading model {self.settings.model_id} on device={self.settings.device}")
            if self.settings.model_cache_dir:
                logger.debug(f"Using cache dir: {self.settings.model_cache_dir}")

            # Warn if MPS is used (CPU is faster for this model on M2)
            if self.settings.device == "mps":
                logger.warning(
                    "MPS device selected, but CPU is ~2.7x faster for GOT-OCR-2.0-hf on M2. "
                    "Consider setting DEVICE=cpu in .env"
                )

            # Load processor
            logger.debug("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.settings.model_id,
                cache_dir=self.settings.model_cache_dir,
            )

            # Load model
            logger.debug("Loading model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.settings.model_id,
                cache_dir=self.settings.model_cache_dir,
            )

            # Move to device
            logger.debug(f"Moving model to {self.settings.device}")
            self.model.to(self.settings.device)

            self._loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load GOT-OCR model: {e}", exc_info=True)
            self._loaded = False
            raise

    def unload(self) -> None:
        """Free model resources from memory."""
        if not self._loaded:
            logger.debug("Model not loaded, nothing to unload")
            return

        logger.info("Unloading model...")
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear CUDA cache if using CUDA
        if self.settings.device == "cuda":
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("Model unloaded successfully")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded.

        Returns:
            True if model and processor are loaded and ready
        """
        return self._loaded

    def generate(self, image: Image.Image, settings: Settings) -> str:
        """Run OCR inference on an image.

        Implementation based on notebooks/stepfun-got-ocr.ipynb validation.
        Inference takes approximately 80 seconds on M2 CPU.

        Args:
            image: PIL Image to perform OCR on
            settings: Application settings including generation parameters

        Returns:
            Extracted text from the image

        Raises:
            RuntimeError: If model is not loaded
            Exception: If inference fails
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.processor is None or self.model is None:
            raise RuntimeError("Processor or model is None. Call load() first.")

        try:
            logger.debug("Processing image...")
            # Process image - processor handles PIL images directly
            inputs = self.processor(image, return_tensors="pt").to(self.settings.device)

            logger.debug("Running inference...")
            # Generate with no_grad context for memory efficiency
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    do_sample=settings.do_sample,
                    tokenizer=self.processor.tokenizer,
                    stop_strings=settings.stop_strings,
                    max_new_tokens=settings.max_new_tokens,
                )

            # Decode output, skipping the input tokens
            logger.debug("Decoding output...")
            result = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            logger.debug(f"Inference complete, extracted {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise
