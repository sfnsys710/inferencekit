"""Output schemas for OCR results."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class OCRResult:
    """OCR operation result with metadata.

    Attributes:
        text: Extracted text from the image
        model_id: Model identifier used for OCR
        inference_time_seconds: Time taken for inference
        timestamp: When the OCR operation was performed
        device: Device used for inference (cpu, mps, cuda)
        error: Error message if operation failed, None otherwise
    """

    text: str
    model_id: str
    inference_time_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device: str = "cpu"
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if the OCR operation was successful."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON/API responses."""
        return {
            "text": self.text,
            "model_id": self.model_id,
            "inference_time_seconds": self.inference_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "device": self.device,
            "error": self.error,
            "success": self.success,
        }
