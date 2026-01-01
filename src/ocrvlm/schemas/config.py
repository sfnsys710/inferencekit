"""Configuration schemas for application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Settings are loaded from .env file if present, with fallback to defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model Configuration
    model_id: str = Field(
        default="stepfun-ai/GOT-OCR-2.0-hf",
        description="HuggingFace model identifier"
    )
    model_cache_dir: Optional[Path] = Field(
        default=None,
        description="Optional directory for caching downloaded models"
    )

    # Device Settings
    device: Literal["cpu", "mps", "cuda"] = Field(
        default="cpu",
        description="Device to use for inference"
    )

    # Generation Parameters
    max_new_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate"
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to use sampling for generation"
    )
    stop_strings: str = Field(
        default="<|im_end|>",
        description="String(s) that signal end of generation"
    )

    # Image Processing
    max_image_size_mb: int = Field(
        default=10,
        description="Maximum image size in megabytes"
    )
    supported_formats: set[str] = Field(
        default={"jpg", "jpeg", "png", "bmp", "tiff"},
        description="Supported image formats"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # API Settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8080,
        description="API server port"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Singleton Settings instance
    """
    return Settings()
