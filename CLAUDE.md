# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ocrvlm is an OCR experimentation project using the GOT-OCR-2.0-hf Vision Language Model from HuggingFace, optimized for Apple Silicon M2. The project provides building blocks for OCR with both CLI and future API interfaces.

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

### Running OCR

**CLI (Phase 3):**
```bash
python scripts/stepfun_got_ocr.py --path image.jpg
python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg
python scripts/stepfun_got_ocr.py --path image.jpg --device cpu --output result.txt

# Optional parameters
# --device: cpu/mps/cuda (overrides .env setting)
# --output: file path to save result
# --max_tokens: override max_new_tokens setting
```

**API Server (Phase 4):**
```bash
# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/ocr  # API info
curl -X POST -F "file=@image.jpg" http://localhost:8080/ocr/upload
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' http://localhost:8080/ocr/url

# View auto-generated docs
# Swagger UI: http://localhost:8080/docs
# ReDoc: http://localhost:8080/redoc
```

### Development
```bash
# Run Jupyter notebooks for experimentation
jupyter notebook notebooks/
```

## Architecture

### Core Design Principles

1. **Explicit over implicit**: Use explicit `--path` or `--url` parameters rather than generic `--source`. Force users to be clear about their inputs.

2. **Logging only**: No print statements in code. Use Python's logging module exclusively. Scripts configure logging via `setup_logging()`.

3. **Raise errors, don't exit**: Prefer `raise ValueError()` over `sys.exit(1)`. Let calling code handle errors.

4. **Single Settings class**: All configuration (model, device, generation params, logging) consolidated in one `Settings` class in `src/ocrvlm/schemas/config.py`. No separate GenerationConfig or multiple config files.

### Package Structure

```
src/ocrvlm/              # Core reusable modules
├── __init__.py          # Public API exports
├── content/
│   └── image.py         # ImageHandler with load_from_url() and load_from_path()
├── models/
│   ├── base.py          # BaseOCRModel abstract interface
│   └── got_ocr.py       # GOTOCRModel implementation
└── schemas/
    ├── config.py        # Settings class (all configuration including API)
    └── output.py        # OCRResult dataclass

scripts/                 # CLI entry points (standalone, no __init__.py)
└── stepfun_got_ocr.py  # Function-based CLI using Fire

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

**BaseOCRModel (`models/base.py`):**
- Abstract interface defining `load()`, `unload()`, `is_loaded()`, `generate()`
- `generate()` accepts `Image.Image` and `Settings` (not separate config objects)

**GOTOCRModel (`models/got_ocr.py`):**
- Concrete implementation using HuggingFace transformers
- Uses `AutoProcessor` and `AutoModelForImageTextToText`
- Lazy loading pattern - model loads on first `load()` call
- Uses `torch.no_grad()` for inference efficiency
- Warns if MPS is used (CPU is faster for this model)

**OCRResult (`schemas/output.py`):**
- Dataclass for inference results
- Includes text, model_id, inference_time, timestamp, device
- Has `to_dict()` method for JSON serialization (used by API)

**FastAPI Server (`api/main.py`):**
- Simple structure: all endpoints in one file (main.py)
- Separate schema file for request/response models (schemas.py)
- **Model lifecycle**: Load per request, unload after completion (~106s total: 26s load + 80s inference)
- Uses same core modules (ImageHandler, GOTOCRModel, OCRResult)
- Settings loaded from `.env` only (no per-request overrides)
- No API versioning (direct paths like `/ocr/upload`)

**API Endpoints:**
- `GET /` - Root endpoint
- `GET /ocr` - API info and capabilities
- `GET /health` - Health check for monitoring
- `POST /ocr/upload` - File upload OCR (mirrors CLI `--path`)
- `POST /ocr/url` - URL-based OCR (mirrors CLI `--url`)

**API Schemas (`api/schemas.py`):**
- `OCRUrlRequest` - Request model for URL endpoint (pydantic HttpUrl validation)
- `OCRResponse` - Unified response model (mirrors OCRResult.to_dict())

### Data Flow

**CLI Flow (scripts/stepfun_got_ocr.py):**
1. Validate explicit `--path` or `--url` parameter (mutually exclusive, one required)
2. ImageHandler loads and validates image via appropriate method
3. GOTOCRModel loads model (if not already loaded)
4. Model runs inference with Settings parameters (max_new_tokens, do_sample, stop_strings)
5. Returns extracted text as string
6. Optional: Save to output file if `--output` specified
7. Model cleanup via `unload()`

**API Flow (api/main.py):**
1. HTTP request received (file upload or URL)
2. For uploads: Save to tempfile, use `load_from_path()`; For URLs: Use `load_from_url()`
3. ImageHandler validates format and size
4. GOTOCRModel instance created and loaded (~26s)
5. Model runs inference (~80s on CPU)
6. Model unloaded to free memory
7. OCRResult created and converted to JSON response via `to_dict()`
8. Temp file cleanup (upload endpoint only)

**Key difference**: CLI can override settings via parameters; API uses `.env` settings only

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
- All settings in one place: `src/ocrvlm/schemas/config.py`
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

## Project Status

- ✅ **Phase 1**: M2 compatibility validation (notebooks/first.ipynb)
- ✅ **Phase 2**: Core reusable modules (src/ocrvlm/)
- ✅ **Phase 3**: CLI script (scripts/stepfun_got_ocr.py)
- ✅ **Phase 4**: FastAPI REST API (api/main.py, api/schemas.py)
- 🔜 **Phase 5**: Docker containerization for GCP Cloud Run
