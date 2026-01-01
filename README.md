# OCR VLM

OCR experimentation project using GOT-OCR-2.0-hf model from HuggingFace, optimized for Apple Silicon M2.

## Features

- Vision Language Model-based OCR using GOT-OCR-2.0-hf
- Optimized for Apple Silicon M2 (CPU mode)
- Dual interfaces: CLI and REST API
- Containerized deployment ready for GCP Cloud Run
- Designed for low-volume, occasional use (~106s per request including model load)

## Quick Start

### Setup

```bash
# Install uv if you haven't already
# Clone and setup
git clone <repo-url>
cd ocrvlm
uv sync --group api  # Include API dependencies
cp .env.example .env
```

### CLI Usage

```bash
# Process local image
python scripts/stepfun_got_ocr.py --path image.jpg

# Process from URL
python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg

# Save output to file
python scripts/stepfun_got_ocr.py --path image.jpg --output result.txt

# Override device setting
python scripts/stepfun_got_ocr.py --path image.jpg --device cpu
```

### API Usage

```bash
# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Upload image
curl -X POST -F "file=@image.jpg" http://localhost:8080/ocr/upload

# Process from URL
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' \
  http://localhost:8080/ocr/url

# Health check
curl http://localhost:8080/health

# API documentation: http://localhost:8080/docs
```

### Docker Usage

```bash
# Build image
docker build -t ocrvlm:latest .

# Run container
docker run -p 8080:8080 ocrvlm:latest

# Run with environment overrides
docker run -p 8080:8080 -e DEVICE=cpu -e LOG_LEVEL=DEBUG ocrvlm:latest
```

## Architecture

### Tech Stack

- **Dependency Management**: uv with dependency groups
- **ML Framework**: transformers + torch
- **API**: FastAPI with uvicorn
- **CLI**: Fire library
- **Deployment**: Multi-stage Docker for GCP Cloud Run

### Project Structure

```
ocrvlm/
├── src/ocrvlm/        # Core reusable Python modules
│   ├── content/       # Image loading and validation
│   ├── models/        # OCR model implementations
│   └── schemas/       # Configuration and output schemas
├── scripts/           # CLI entry points
├── api/              # FastAPI server
│   ├── main.py       # All endpoints
│   └── schemas.py    # API request/response models
├── notebooks/        # Jupyter notebooks for experiments
└── Dockerfile        # Multi-stage container build
```

### Model Lifecycle: Per-Request Load/Unload

The API uses a **per-request model lifecycle** intentionally designed for occasional use:

1. Request received → Load model (~26s)
2. Run inference (~80s on M2 CPU)
3. Unload model to free memory
4. Total: ~106s per request

**Why not keep model in memory?**
- This project targets low-volume, occasional use (not high-throughput serving)
- Freeing memory between requests allows the server to run on smaller instances
- For high-volume use, consider dedicated serving solutions like vLLM or Ollama

### Hardware Tradeoffs

**Critical: Use CPU mode on M2 for this model**

Performance comparison on Apple Silicon M2:
- **CPU**: ~80s inference (recommended)
- **MPS**: ~220s inference (2.7x slower)
- **CUDA**: Not tested (requires NVIDIA GPU)

The GOT-OCR-2.0-hf model performs better on CPU than MPS for this specific architecture. This is configured via the `DEVICE` setting in `.env`.

**Configuration:**
```bash
# .env file
DEVICE=cpu  # Use CPU (fastest on M2)
# DEVICE=mps  # Not recommended - slower than CPU
# DEVICE=cuda  # For NVIDIA GPUs
```

### Configuration Management

All configuration is centralized in `src/ocrvlm/schemas/config.py` using pydantic-settings.

**Environment variables (.env):**
```bash
# Model settings
MODEL_ID=stepfun-ai/GOT-OCR2_0
DEVICE=cpu
MAX_NEW_TOKENS=4096

# API settings
API_HOST=0.0.0.0
API_PORT=8080

# Logging
LOG_LEVEL=INFO
```

**Override priority:**
1. CLI: Command-line arguments override `.env` settings
   ```bash
   python scripts/stepfun_got_ocr.py --path img.jpg --device cuda
   ```

2. API: Uses `.env` settings only (no per-request overrides for simplicity)

3. Docker: Environment variables passed to container override defaults
   ```bash
   docker run -e DEVICE=cpu -e LOG_LEVEL=DEBUG ocrvlm:latest
   ```

### API Monitoring

The API includes built-in logging and monitoring capabilities:

**Health endpoint:** `GET /health`
- Returns service status, model info, and device configuration
- Use for container health checks and uptime monitoring

**Logging:**
- All requests logged with timestamps and processing times
- Errors logged with full stack traces
- Configure log level via `LOG_LEVEL` environment variable

**Response metadata:**
Every OCR response includes:
- `inference_time_seconds`: Actual inference duration
- `model_id`: Model version used
- `device`: Hardware used (cpu/mps/cuda)
- `timestamp`: ISO format timestamp

**Example monitoring in production:**
```bash
# Cloud Run health check
gcloud run services update ocrvlm \
  --region us-central1 \
  --max-instances 10 \
  --cpu 2 \
  --memory 4Gi \
  --timeout 300s
```

## Design Decisions

### Why Docker + FastAPI?

**FastAPI chosen over Ollama/vLLM:**
- **Ollama**: Designed for chat models, not vision-language OCR tasks
- **vLLM**: Optimized for high-throughput LLM serving with continuous batching
- **FastAPI**: Lightweight, gives full control over model lifecycle, perfect for occasional-use pattern

**Our use case:**
- Low volume (occasional requests, not continuous serving)
- Vision-language model (not pure LLM)
- Need per-request load/unload to minimize resource usage
- Simple REST API without complex serving optimizations

**For high-volume OCR serving**, consider:
- vLLM with continuous batching
- Dedicated GPU instances
- Model kept in memory
- Request queuing and batching

### Dependency Groups

Dependencies are organized in `pyproject.toml`:
- **Core**: torch, transformers, pillow, pydantic (needed by all)
- **API group**: fastapi, uvicorn, python-multipart
- **Dev group**: jupyter

Install only what you need:
```bash
uv sync              # Core + CLI only
uv sync --group api  # Core + CLI + API
uv sync --group dev  # Core + CLI + dev tools
```

## Performance Characteristics

**On Apple Silicon M2:**
- Model loading: ~26 seconds
- Inference: ~80 seconds per image (CPU)
- Total per request: ~106 seconds
- Memory: ~3GB during inference

**Recommendations:**
- **Device**: Use `DEVICE=cpu` (not MPS)
- **RAM**: 4GB+ for reliable operation
- **Timeout**: Set HTTP timeout to 300s+ for API
- **Instances**: Scale horizontally for concurrent requests

## Development

### Running Tests
```bash
# Jupyter notebooks for experimentation
uv sync --group dev
jupyter notebook notebooks/
```

### Local Development
```bash
# API with auto-reload
uvicorn api.main:app --reload

# CLI development
python scripts/stepfun_got_ocr.py --path test.jpg
```

## Requirements

- Python 3.12+
- 4GB+ RAM recommended
- Apple Silicon M2 (or any CPU/CUDA device)
- Docker (for containerized deployment)

## License

MIT License
