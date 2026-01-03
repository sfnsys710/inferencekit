"""Qwen3 text generation model implementation."""

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..schemas.config import Settings
from .base import BaseModel

logger = logging.getLogger(__name__)


class Qwen3Model(BaseModel):
    """Implementation for Qwen3 text generation model.

    Lightweight text-to-text model for generation tasks.
    """

    def __init__(self, settings: Settings):
        """Initialize Qwen3 model.

        Args:
            settings: Application settings including model_id and device
        """
        self.settings = settings
        self.tokenizer: Any = None
        self.model: Any = None
        self._loaded = False

    def load(self) -> None:
        """Load model and tokenizer from HuggingFace.

        Raises:
            Exception: If model or tokenizer loading fails
        """
        if self._loaded:
            logger.info("Model already loaded, skipping")
            return

        try:
            logger.info(
                f"Loading model {self.settings.qwen_model_id} on device={self.settings.device}"
            )
            if self.settings.model_cache_dir:
                logger.debug(f"Using cache dir: {self.settings.model_cache_dir}")

            # Load tokenizer
            logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.qwen_model_id,
                cache_dir=self.settings.model_cache_dir,
            )

            # Load model
            logger.debug("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.qwen_model_id,
                cache_dir=self.settings.model_cache_dir,
            )

            # Move to device
            logger.debug(f"Moving model to {self.settings.device}")
            self.model.to(self.settings.device)

            self._loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Qwen3 model: {e}", exc_info=True)
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

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if using CUDA
        if self.settings.device == "cuda":
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("Model unloaded successfully")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded.

        Returns:
            True if model and tokenizer are loaded and ready
        """
        return self._loaded

    def generate(self, prompt: str, settings: Settings) -> str:
        """Run text generation on a prompt.

        Args:
            prompt: Input text prompt
            settings: Application settings including generation parameters

        Returns:
            Generated text

        Raises:
            RuntimeError: If model is not loaded
            Exception: If inference fails
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Tokenizer or model is None. Call load() first.")

        try:
            logger.debug("Tokenizing input...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.settings.device)

            logger.debug("Running inference...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=settings.do_sample,
                )

            # Decode output
            logger.debug("Decoding output...")
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input prompt from the result
            if result.startswith(prompt):
                result = result[len(prompt) :].strip()

            logger.debug(f"Inference complete, generated {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise
