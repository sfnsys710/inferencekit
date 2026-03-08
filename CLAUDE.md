# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ocrkit is an OCR research and deployment toolkit covering traditional, LLM-based, and hybrid approaches — applied to the Moroccan context (Arabic, French, Darija). It includes notebooks for POC experimentation and production-ready packaged code.

**Three OCR layers:**
- **Traditional**: Tesseract, doctr — rule-based and DL pipelines
- **LLM-based**: GOT-OCR-2.0-hf (VLM), Qwen3 (text post-processing/extraction)
- **Hybrid**: traditional detection + LLM understanding/correction

**Deployment:**
- Development: M2 CPU (transformers)
- Production: Cloud Run GPU NVIDIA L4/T4 (transformers + CUDA)
- Use case: Moderately frequent, non-concurrent (10-50 requests/day)

**Performance (GOT-OCR):**
- M2 CPU: ~106s/request (~26s load + ~80s inference) — use `DEVICE=cpu` (MPS is 2.7x slower!)
- Cloud Run GPU: ~20-35s/request (~5-10s load + ~15-25s inference) — use `DEVICE=cuda`

## Commands

### Setup
```bash
uv sync
uv sync --group api          # + FastAPI
uv sync --group experiment   # + docling, doctr, tesserocr
cp .env.example .env
```

### OCR CLI
```bash
python scripts/stepfun_got_ocr.py --path image.jpg
python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg
python scripts/stepfun_got_ocr.py --path image.jpg --device cpu --output result.txt
```

### Text generation CLI
```bash
python scripts/qwen3_generate.py --prompt "Hello, how are you?"
python scripts/qwen3_generate.py --prompt "Explain AI" --output result.txt
```

### API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
curl http://localhost:8080/health
curl -X POST -F "file=@image.jpg" http://localhost:8080/ocr/upload
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' http://localhost:8080/ocr/url
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"Hello"}' http://localhost:8080/text/generate
# Docs: http://localhost:8080/docs
```

### Docker
```bash
docker build -t ocrkit:latest .
docker run -p 8080:8080 ocrkit:latest
docker run -p 8080:8080 -e DEVICE=cpu -e LOG_LEVEL=DEBUG ocrkit:latest
```

### Notebooks
```bash
uv sync --group dev
jupyter notebook notebooks/
```

## Architecture

### Core Design Principles

1. **Explicit over implicit**: Use `--path` or `--url`, not `--source`. Be clear about inputs.
2. **Logging only**: No print statements. Use `logging` module + `setup_logging()` in scripts.
3. **Raise errors, don't exit**: `raise ValueError()` over `sys.exit(1)`.
4. **Single Settings class**: All config in `src/ocrkit/schemas/config.py`. No separate config files.

### Package Structure

```
src/ocrkit/              # Deployable package
├── __init__.py          # Public API exports
├── content/
│   └── image.py         # ImageHandler — load_from_url(), load_from_path()
├── models/
│   ├── base.py          # BaseModel abstract interface
│   ├── got_ocr.py       # GOTOCRModel (vision → text)
│   └── qwen3.py         # Qwen3Model (text → text)
└── schemas/
    ├── config.py        # Settings (pydantic-settings, cached singleton)
    └── output.py        # OCRResult, TextResult dataclasses

scripts/                 # CLI entry points (no __init__.py)
├── stepfun_got_ocr.py
└── qwen3_generate.py

api/
├── main.py              # All FastAPI endpoints
└── schemas.py           # API request/response models

notebooks/               # POC experiments (not imported by src/)
├── gotocr.ipynb
├── doctr.ipynb
├── docling.ipynb
└── qwen3-06b.ipynb
```

### Inference Options (studied in notebooks)

| Runtime | VLM support | Dev (M2) | GPU prod | Concurrency |
|---------|-------------|----------|----------|-------------|
| **transformers** (current) | ✓✓ | CPU | CUDA | Sequential |
| vLLM | ⚠ Limited | ✗ CUDA only | ✓✓ throughput | Concurrent |
| Ollama | ✗ Chat only | ✓ Metal | ✓ | Sequential |
| llama-cpp | ✓ | ✓✓ Metal | ⚠ | Sequential |

**Why transformers:** GOT-OCR VLM support required; full precision; Cloud Run GPU target.

### Model Lifecycle

Per-request load/unload — intentional for Cloud Run scale-to-zero cost savings:
1. Request → Load model (5-10s GPU, 26s M2)
2. Inference (15-25s GPU, 80s M2)
3. Unload → free memory

Cost: ~$50-100/month (scale-to-zero) vs ~$500/month (always-on GPU).

### Key Components

**Settings (`schemas/config.py`):**
- pydantic-settings, loads from `.env`
- Single source of truth for model, device, generation params, image limits, API config
- Cached singleton via `get_settings()`

**ImageHandler (`content/image.py`):**
- `load_from_url()` and `load_from_path()` — explicit, two methods only
- Validates format + size against Settings
- Raises `ValueError` / `FileNotFoundError`

**GOTOCRModel (`models/got_ocr.py`):** Vision-to-text, `AutoProcessor` + `AutoModelForImageTextToText`

**Qwen3Model (`models/qwen3.py`):** Text-to-text, `AutoTokenizer` + `AutoModelForCausalLM`

**FastAPI (`api/main.py`):**
- All endpoints in one file; schemas in `schemas.py`
- Model loads per request (not at startup)
- Exception mapping: `ValueError` → 400, `FileNotFoundError` → 404, URL errors → 422, else → 500

## Important Notes

### Device Configuration
- **M2 dev**: `DEVICE=cpu` — MPS is 2.7x slower for GOT-OCR (architecture-specific)
- **Production**: `DEVICE=cuda` — Cloud Run L4/T4 GPU

### Script Guidelines (`scripts/`)
- Function-based with Fire library (not classes)
- Explicit `--path` / `--url` params
- `setup_logging()` call, no prints
- `raise` errors, never `sys.exit()`
- Standalone — no `__init__.py` in `scripts/`

### API Guidelines (`api/`)
- All endpoints in `main.py`, request/response models in `schemas.py`
- Per-request model loading (Cloud Run scale-to-zero)
- `tempfile.NamedTemporaryFile` for uploads, cleanup in `finally`
- No API versioning, no per-request setting overrides

### Docker Guidelines
- Multi-stage: builder + `python:3.12-slim-bookworm` runtime
- `UV_PYTHON_DOWNLOADS=0`, `UV_COMPILE_BYTECODE=1`
- Install deps before copying source (layer cache)
- Port 8080 (Cloud Run standard)
