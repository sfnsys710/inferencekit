"""Request and response schemas for the OCR API."""

from pydantic import BaseModel, Field, HttpUrl


class OCRUrlRequest(BaseModel):
    """Request schema for URL-based OCR.

    Attributes:
        url: HTTP(S) URL of the image to process
    """

    url: HttpUrl = Field(..., description="Image URL to process")


class OCRResponse(BaseModel):
    """Unified response schema for OCR operations.

    Attributes:
        text: Extracted text from the image
        model_id: Model identifier used for OCR
        inference_time_seconds: Time taken for inference
        timestamp: ISO format timestamp of when OCR was performed
        device: Device used for inference (cpu, mps, cuda)
        success: Whether the OCR operation was successful
        error: Error message if operation failed, None otherwise
    """

    text: str
    model_id: str
    inference_time_seconds: float
    timestamp: str
    device: str
    success: bool
    error: str | None = None
