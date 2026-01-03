# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

inferencekit provides building blocks for running inference on various models (OCR, text generation, etc.). Currently supports GOT-OCR-2.0-hf Vision Language Model and Qwen3 text model, optimized for Apple Silicon M2.

**Performance characteristics on M2:**
- Model loading: ~26 seconds
- Inference: ~80 seconds per image on CPU
- **Important:** CPU mode is ~2.7x faster than MPS for this model - always use `DEVICE=cpu`

## Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
```

### Running Models

**OCR (GOT-OCR):**
```bash
python scripts/stepfun_got_ocr.py --path image.jpg
python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg
python scripts/stepfun_got_ocr.py --path image.jpg --device cpu --output result.txt
```

**Text Generation (Qwen3):**
```bash
python scripts/qwen3_generate.py --prompt "Hello, how are you?"
python scripts/qwen3_generate.py --prompt "Explain AI" --output result.txt
python scripts/qwen3_generate.py --prompt "Write code" --device cpu --max_tokens 512
```

**API Server:**
```bash
# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/ocr   # OCR API info
curl http://localhost:8080/text  # Text generation API info
curl -X POST -F "file=@image.jpg" http://localhost:8080/ocr/upload
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' http://localhost:8080/ocr/url
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"Hello"}' http://localhost:8080/text/generate

# View auto-generated docs
# Swagger UI: http://localhost:8080/docs
# ReDoc: http://localhost:8080/redoc
```

### Development
```bash
# Run Jupyter notebooks for experimentation
jupyter notebook notebooks/
```

### Docker

**Building and running locally:**
```bash
# Build Docker image
docker build -t inferencekit:latest .

# Run container
docker run -p 8080:8080 inferencekit:latest

# Run with environment overrides
docker run -p 8080:8080 -e DEVICE=cpu -e LOG_LEVEL=DEBUG inferencekit:latest

# Test the containerized API
curl http://localhost:8080/health
curl -X POST -F "file=@image.jpg" http://localhost:8080/ocr/upload
```

## Architecture

### Core Design Principles

1. **Explicit over implicit**: Use explicit `--path` or `--url` parameters rather than generic `--source`. Force users to be clear about their inputs.

2. **Logging only**: No print statements in code. Use Python's logging module exclusively. Scripts configure logging via `setup_logging()`.

3. **Raise errors, don't exit**: Prefer `raise ValueError()` over `sys.exit(1)`. Let calling code handle errors.

4. **Single Settings class**: All configuration (model, device, generation params, logging) consolidated in one `Settings` class in `src/inferencekit/schemas/config.py`. No separate GenerationConfig or multiple config files.

### Package Structure

```
src/inferencekit/        # Core reusable modules
├── __init__.py          # Public API exports
├── content/
│   └── image.py         # ImageHandler with load_from_url() and load_from_path()
├── models/
│   ├── base.py          # BaseModel abstract interface
│   ├── got_ocr.py       # GOTOCRModel implementation (vision → text)
│   └── qwen3.py         # Qwen3Model implementation (text → text)
└── schemas/
    ├── config.py        # Settings class (all configuration including API)
    └── output.py        # OCRResult and TextResult dataclasses

scripts/                 # CLI entry points (standalone, no __init__.py)
├── stepfun_got_ocr.py  # OCR CLI using Fire
└── qwen3_generate.py   # Text generation CLI using Fire

api/                     # FastAPI server (simple structure)
├── main.py             # FastAPI app with all endpoints
└── schemas.py          # API request/response models
```

### Key Components

**Settings (`schemas/config.py`):**
- Loads from `.env` file using pydantic-settings
- Single source of truth for all configuration
- Includes model config, device settings, generation parameters, image processing limits, logging level, and **API settings** (api_host, api_port)
- Cached singleton via `get_settings()`
- Used by both CLI and API

**ImageHandler (`content/image.py`):**
- Two explicit methods: `load_from_url()` and `load_from_path()`
- Validates image format and size against Settings
- Returns PIL Image objects
- Raises ValueError or FileNotFoundError on errors

**BaseModel (`models/base.py`):**
- Abstract interface defining `load()`, `unload()`, `is_loaded()`, `generate()`
- Generic for all model types (vision, text, etc.)

**GOTOCRModel (`models/got_ocr.py`):**
- Vision-to-text OCR implementation
- Uses `AutoProcessor` and `AutoModelForImageTextToText`
- CPU is ~2.7x faster than MPS on M2

**Qwen3Model (`models/qwen3.py`):**
- Text-to-text generation implementation
- Uses `AutoTokenizer` and `AutoModelForCausalLM`
- Same lazy loading pattern as GOTOCRModel

**OCRResult & TextResult (`schemas/output.py`):**
- Dataclasses for inference results with metadata
- Both have `to_dict()` method for JSON serialization

**FastAPI Server (`api/main.py`):**
- Simple structure: all endpoints in one file (main.py)
- Separate schema file for request/response models (schemas.py)
- **Model lifecycle**: Load per request, unload after completion (~106s total: 26s load + 80s inference)
- Uses same core modules (ImageHandler, GOTOCRModel, OCRResult)
- Settings loaded from `.env` only (no per-request overrides)
- No API versioning (direct paths like `/ocr/upload`)

**API Endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check for monitoring
- `GET /ocr` - OCR API info
- `POST /ocr/upload` - File upload OCR
- `POST /ocr/url` - URL-based OCR
- `GET /text` - Text generation API info
- `POST /text/generate` - Text generation

**API Schemas (`api/schemas.py`):**
- `OCRUrlRequest`, `OCRResponse` - OCR request/response models
- `TextGenerateRequest`, `TextResponse` - Text generation request/response models

### Data Flow

**OCR CLI Flow (scripts/stepfun_got_ocr.py):**
1. Validate explicit `--path` or `--url` parameter (mutually exclusive, one required)
2. ImageHandler loads and validates image via appropriate method
3. GOTOCRModel loads model
4. Model runs inference with Settings parameters
5. Returns extracted text as string
6. Optional: Save to output file if `--output` specified
7. Model cleanup via `unload()`

**Text Generation CLI Flow (scripts/qwen3_generate.py):**
1. Validate `--prompt` parameter (required)
2. Qwen3Model loads model
3. Model runs text generation with Settings parameters
4. Returns generated text as string
5. Optional: Save to output file if `--output` specified
6. Model cleanup via `unload()`

**API Flow:**
- **OCR**: Image upload/URL → ImageHandler → GOTOCRModel → OCRResult → JSON
- **Text**: Prompt → Qwen3Model → TextResult → JSON
- Same pattern: per-request load/unload, timing logs, error handling
- CLI can override settings via parameters; API uses `.env` settings only

## Important Notes

### Device Configuration
Always set `DEVICE=cpu` in `.env` for GOT-OCR-2.0-hf on M2. MPS acceleration is significantly slower than CPU for this specific model.

### Script Development Guidelines
When creating new scripts in `scripts/`:
- Use function-based approach (not classes) with Fire library
- Explicit parameters (e.g., `--path` and `--url`, not `--source`)
- Logging only (import logging, use logger, call `setup_logging()`)
- Raise errors (ValueError, FileNotFoundError) instead of sys.exit()
- Scripts are standalone - no `__init__.py` in scripts/ directory

### Configuration Management
- All settings in one place: `src/inferencekit/schemas/config.py`
- No separate config files or generation config classes
- Settings passed directly to model methods
- Override settings at runtime via script parameters

### Image Processing
- ImageHandler has exactly two public methods matching script parameters
- Always validate format and size after loading
- Let exceptions propagate (ValueError, FileNotFoundError)

### API Development Guidelines
When working with the FastAPI server in `api/`:
- Keep it simple: all endpoints in `main.py`, schemas in `schemas.py`
- Model loads per request (not at startup) - this is intentional for occasional use pattern
- Use `tempfile.NamedTemporaryFile` for upload handling
- Clean up temp files in finally block
- Map exceptions to HTTP status codes:
  - `ValueError` → 400 Bad Request
  - `FileNotFoundError` → 404 Not Found
  - URL download errors → 422 Unprocessable Entity
  - Other errors → 500 Internal Server Error
- No per-request setting overrides (simpler than CLI)
- No API versioning (keep paths simple)
- FastAPI auto-generates docs at `/docs` and `/redoc`

### Docker Development Guidelines
When working with Docker containerization:
- **Multi-stage builds**: Use builder stage for dependency installation, slim runtime for production
- **Python version consistency**: Match python version in both Dockerfile stages with pyproject.toml (3.12)
- **UV dependency groups**: Separate dependencies into groups in pyproject.toml:
  - Core dependencies (main): torch, transformers, pillow, pydantic, pydantic-settings, fire
  - API group: fastapi, uvicorn[standard], python-multipart
  - Dev group: jupyter (excluded from Docker build)
- **UV environment variables**:
  - `UV_PYTHON_DOWNLOADS=0` - Use system Python, don't download
  - `UV_COMPILE_BYTECODE=1` - Compile bytecode for faster startup
- **Layer caching**: Install dependencies before copying source code
- **Minimal runtime**: Use `python:3.12-slim-bookworm` for small image size
- **Port 8080**: Standard for Cloud Run compatibility
- **Environment variables**: Support runtime overrides via `-e` flags
- **.dockerignore**: Exclude notebooks, tests, dev files, .env, .git from build context

## Project Components

The project is complete and includes:

- **Core modules** (`src/inferencekit/`): Reusable components for running inference on various models (OCR, text generation)
- **CLI interface** (`scripts/stepfun_got_ocr.py`): Command-line tool for local OCR processing
- **REST API** (`api/`): FastAPI server with file upload and URL-based OCR endpoints
- **Docker support** (`Dockerfile`): Multi-stage containerization for cloud deployment
- **Experimentation** (`notebooks/`): Jupyter notebooks for model testing and validation (GOT-OCR, Qwen3)
