"""Configuration schemas for OCR generation parameters."""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation from OCR model."""

    max_new_tokens: int = 4096
    do_sample: bool = False
    stop_strings: str = "<|im_end|>"
