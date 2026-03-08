"""FastAPI server for model inference (OCR and text generation)."""

import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ocrkit import GOTOCRModel, ImageHandler, OCRResult, Qwen3Model, TextResult, get_settings

from .schemas import OCRResponse, OCRUrlRequest, TextGenerateRequest, TextResponse

# Initialize settings and logging
settings = get_settings()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title="OcrKit API",
    description="Model inference API for OCR and text generation",
    version="0.2.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint redirecting to API info."""
    return {
        "message": "OcrKit API",
        "docs": "/docs",
        "endpoints": {
            "ocr": "/ocr",
            "text_generation": "/text",
            "health": "/health",
        },
    }


@app.get("/ocr")
async def get_ocr_info():
    """Get information about OCR API capabilities.

    Returns information about supported formats, model, and usage examples.
    """
    return {
        "service": "OCR VLM API",
        "model": settings.model_id,
        "device": settings.device,
        "supported_formats": list(settings.supported_formats),
        "max_image_size_mb": settings.max_image_size_mb,
        "endpoints": {
            "upload": "POST /ocr/upload - Upload image file for OCR",
            "url": "POST /ocr/url - Provide image URL for OCR",
            "health": "GET /health - Health check",
        },
        "example_usage": {
            "upload": "curl -X POST -F 'file=@image.jpg' http://localhost:8080/ocr/upload",
            "url": (
                'curl -X POST -H "Content-Type: application/json" '
                '-d \'{"url":"https://example.com/image.jpg"}\' http://localhost:8080/ocr/url'
            ),
        },
        "performance": {
            "model_load_time_seconds": "~26",
            "inference_time_seconds": "~80 (CPU on M2)",
            "total_per_request_seconds": "~106",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring.

    Returns service status and configuration.
    """
    return {
        "status": "healthy",
        "ocr_model": settings.model_id,
        "text_model": settings.qwen_model_id,
        "device": settings.device,
        "service": "OcrKit API",
    }


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(file: UploadFile = File(...)):
    """Process an uploaded image file with OCR.

    Args:
        file: Image file to process (jpg, jpeg, png, bmp, tiff)

    Returns:
        OCRResponse with extracted text and metadata

    Raises:
        HTTPException: 400 for invalid images, 500 for processing errors
    """
    logger.info(f"OCR upload request: {file.filename}")
    temp_file = None

    try:
        # Save uploaded file to temp location
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename or "image").suffix
        )
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        temp_path = temp_file.name

        logger.debug(f"Saved upload to temp file: {temp_path}")

        # Load and validate image
        handler = ImageHandler(settings)
        image = handler.load_from_path(temp_path)
        logger.info(f"Image loaded: {image.size} {image.format}")

        # Initialize and load model
        logger.info("Loading model...")
        model = GOTOCRModel(settings)
        load_start = time.time()
        model.load()
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        # Run inference
        logger.info("Running OCR inference...")
        inference_start = time.time()
        text = model.generate(image, settings)
        inference_time = time.time() - inference_start
        logger.info(
            f"OCR completed in {inference_time:.2f} seconds, extracted {len(text)} characters"
        )

        # Cleanup model
        model.unload()
        logger.debug("Model unloaded")

        # Create result
        result = OCRResult(
            text=text,
            model_id=settings.model_id,
            inference_time_seconds=inference_time,
            device=settings.device,
        )

        # Convert to response schema
        response_dict = result.to_dict()
        return OCRResponse(**response_dict)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OCR upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file is not None:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
                logger.debug(f"Cleaned up temp file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


@app.post("/ocr/url", response_model=OCRResponse)
async def ocr_url(request: OCRUrlRequest):
    """Process an image from a URL with OCR.

    Args:
        request: OCRUrlRequest containing the image URL

    Returns:
        OCRResponse with extracted text and metadata

    Raises:
        HTTPException: 400 for invalid images, 422 for URL errors, 500 for processing errors
    """
    logger.info(f"OCR URL request: {request.url}")

    try:
        # Load and validate image from URL
        handler = ImageHandler(settings)
        image = handler.load_from_url(str(request.url))
        logger.info(f"Image loaded: {image.size} {image.format}")

        # Initialize and load model
        logger.info("Loading model...")
        model = GOTOCRModel(settings)
        load_start = time.time()
        model.load()
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        # Run inference
        logger.info("Running OCR inference...")
        inference_start = time.time()
        text = model.generate(image, settings)
        inference_time = time.time() - inference_start
        logger.info(
            f"OCR completed in {inference_time:.2f} seconds, extracted {len(text)} characters"
        )

        # Cleanup model
        model.unload()
        logger.debug("Model unloaded")

        # Create result
        result = OCRResult(
            text=text,
            model_id=settings.model_id,
            inference_time_seconds=inference_time,
            device=settings.device,
        )

        # Convert to response schema
        response_dict = result.to_dict()
        return OCRResponse(**response_dict)

    except ValueError as e:
        logger.error(f"Invalid image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OCR URL processing failed: {e}", exc_info=True)
        # Check if it's a request exception (URL download failure)
        if "RequestException" in str(type(e)):
            raise HTTPException(status_code=422, detail=f"Failed to download image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get("/text")
async def get_text_info():
    """Get information about text generation API capabilities.

    Returns information about model, device, and usage examples.
    """
    return {
        "service": "Text Generation API",
        "model": settings.qwen_model_id,
        "device": settings.device,
        "endpoints": {
            "generate": "POST /text/generate - Generate text from prompt",
            "health": "GET /health - Health check",
        },
        "example_usage": {
            "generate": (
                'curl -X POST -H "Content-Type: application/json" '
                '-d \'{"prompt":"Hello, how are you?"}\' http://localhost:8080/text/generate'
            ),
        },
    }


@app.post("/text/generate", response_model=TextResponse)
async def text_generate(request: TextGenerateRequest):
    """Generate text from a prompt using Qwen3 model.

    Args:
        request: TextGenerateRequest containing the prompt

    Returns:
        TextResponse with generated text and metadata

    Raises:
        HTTPException: 400 for invalid prompts, 500 for processing errors
    """
    logger.info(f"Text generation request: {request.prompt[:100]}...")

    try:
        # Initialize and load model
        logger.info("Loading model...")
        model = Qwen3Model(settings)
        load_start = time.time()
        model.load()
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        # Run inference
        logger.info("Running text generation...")
        inference_start = time.time()
        text = model.generate(request.prompt, settings)
        inference_time = time.time() - inference_start
        logger.info(
            f"Generation completed in {inference_time:.2f} seconds"
            f", generated {len(text)} characters"
        )

        # Cleanup model
        model.unload()
        logger.debug("Model unloaded")

        # Create result
        result = TextResult(
            text=text,
            model_id=settings.qwen_model_id,
            inference_time_seconds=inference_time,
            device=settings.device,
        )

        # Convert to response schema
        response_dict = result.to_dict()
        return TextResponse(**response_dict)

    except ValueError as e:
        logger.error(f"Invalid prompt: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
