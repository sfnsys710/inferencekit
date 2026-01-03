#!/usr/bin/env python3
"""
CLI script for text generation using Qwen3 model.

Usage:
    python scripts/qwen3_generate.py --prompt "Hello, how are you?"
    python scripts/qwen3_generate.py --prompt "Explain quantum physics" --output result.txt
    python scripts/qwen3_generate.py --prompt "Write a poem" --device cpu --max_tokens 512
"""

import logging
import time
from pathlib import Path

import fire

from inferencekit import Qwen3Model, TextResult, get_settings

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def qwen3_generate(
    prompt: str,
    device: str | None = None,
    output: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run text generation using Qwen3 model.

    Args:
        prompt: Text prompt for generation
        device: Override device (cpu/mps/cuda), defaults to settings
        output: Optional file path to save result
        max_tokens: Override max_new_tokens, defaults to settings

    Returns:
        Generated text

    Raises:
        ValueError: If prompt is empty
        Exception: If model loading or generation fails

    Examples:
        python scripts/qwen3_generate.py --prompt "Hello, how are you?"
        python scripts/qwen3_generate.py --prompt "Explain AI" --output result.txt
        python scripts/qwen3_generate.py --prompt "Write code" --device cpu --max_tokens 512
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    # Validate input
    if not prompt:
        raise ValueError("Prompt cannot be empty")

    # Override settings if provided
    if device:
        settings.device = device  # type: ignore
    if max_tokens:
        settings.max_new_tokens = max_tokens

    logger.info(f"Using device: {settings.device}")
    logger.info(f"Prompt: {prompt[:100]}...")  # Log first 100 chars

    # Initialize and load model
    logger.info("Initializing model...")
    model = Qwen3Model(settings)

    logger.info("Loading model...")
    load_start = time.time()
    model.load()
    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.2f} seconds")

    # Run inference
    logger.info("Running text generation...")
    inference_start = time.time()
    text = model.generate(prompt, settings)
    inference_time = time.time() - inference_start

    # Create result
    TextResult(
        text=text,
        model_id=settings.qwen_model_id,
        inference_time_seconds=inference_time,
        device=settings.device,
    )

    logger.info(f"Generation completed in {inference_time:.2f} seconds")
    logger.info(f"Generated {len(text)} characters")

    # Save to file if output path provided
    if output:
        output_path = Path(output)
        output_path.write_text(text, encoding="utf-8")
        logger.info(f"Result saved to: {output}")

    # Cleanup
    logger.info("Unloading model...")
    model.unload()
    logger.info("Done!")

    return text


if __name__ == "__main__":
    fire.Fire(qwen3_generate)
