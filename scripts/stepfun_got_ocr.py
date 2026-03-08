#!/usr/bin/env python3
"""
CLI script for running OCR using GOT-OCR-2.0-hf model.

Usage:
    python scripts/stepfun_got_ocr.py --path image.jpg
    python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg
    python scripts/stepfun_got_ocr.py --path image.jpg --device cpu --output result.txt
"""

import logging
import time
from pathlib import Path

import fire

from ocrkit import GOTOCRModel, ImageHandler, OCRResult, get_settings

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def stepfun_got_ocr(
    path: str | None = None,
    url: str | None = None,
    device: str | None = None,
    output: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run OCR on an image from a file path or URL.

    Args:
        path: Path to local image file
        url: URL to image file
        device: Override device (cpu/mps/cuda), defaults to settings
        output: Optional file path to save OCR result
        max_tokens: Override max_new_tokens, defaults to settings

    Returns:
        Extracted text from the image

    Raises:
        ValueError: If neither or both path and url are provided, or if image is invalid
        FileNotFoundError: If the specified file path doesn't exist

    Examples:
        python scripts/stepfun_got_ocr.py --path image.jpg
        python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg
        python scripts/stepfun_got_ocr.py --path image.jpg --device cpu --output result.txt
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    # Validate input: must provide exactly one of path or url
    if not path and not url:
        raise ValueError("Must specify either --path or --url")

    if path and url:
        raise ValueError("Cannot specify both --path and --url, choose one")

    # Override settings if provided
    if device:
        settings.device = device  # type: ignore
    if max_tokens:
        settings.max_new_tokens = max_tokens

    # Load image
    handler = ImageHandler(settings)
    if url:
        logger.info(f"Loading image from URL: {url}")
        image = handler.load_from_url(url)
    else:
        logger.info(f"Loading image from path: {path}")
        image = handler.load_from_path(path)  # type: ignore

    logger.info(f"Image loaded: {image.size} {image.format}")
    logger.info(f"Using device: {settings.device}")

    # Initialize and load model
    logger.info("Initializing model...")
    model = GOTOCRModel(settings)

    logger.info("Loading model (this may take ~26 seconds)...")
    load_start = time.time()
    model.load()
    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.2f} seconds")

    # Run inference
    logger.info("Running OCR inference (this may take ~80 seconds on CPU)...")
    inference_start = time.time()
    text = model.generate(image, settings)
    inference_time = time.time() - inference_start

    # Create result
    OCRResult(
        text=text,
        model_id=settings.model_id,
        inference_time_seconds=inference_time,
        device=settings.device,
    )

    logger.info(f"OCR completed in {inference_time:.2f} seconds")
    logger.info(f"Extracted {len(text)} characters")

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
    fire.Fire(stepfun_got_ocr)
