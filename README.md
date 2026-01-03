# InferenceKit

Building blocks for running inference on various models (OCR, text generation, etc.). Currently supports GOT-OCR-2.0-hf vision language model and Qwen3 text model.

## Features

- Multi-model support: Vision Language Models (GOT-OCR) and text models (Qwen3)
- Optimized for Apple Silicon M2 (CPU mode for OCR)
- Dual interfaces: CLI and REST API
- Containerized deployment ready for GCP Cloud Run
- Designed for low-volume, occasional use (~106s per OCR request including model load)

## Quick Start

### Setup

```bash
# Install uv if you haven't already
# Clone and setup
git clone https://github.com/sfnsys710/inferencekit.git
cd inferencekit
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
docker build -t inferencekit:latest .

# Run container
docker run -p 8080:8080 inferencekit:latest

# Run with environment overrides
docker run -p 8080:8080 -e DEVICE=cpu -e LOG_LEVEL=DEBUG inferencekit:latest
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
inferencekit/
├── src/inferencekit/  # Core reusable Python modules
│   ├── content/       # Image loading and validation
│   ├── models/        # Model implementations (OCR, text generation)
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

All configuration is centralized in `src/inferencekit/schemas/config.py` using pydantic-settings.

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
   docker run -e DEVICE=cpu -e LOG_LEVEL=DEBUG inferencekit:latest
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
gcloud run services update inferencekit \
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
- **Pre-commit group**: pre-commit hooks and linting tools

Install only what you need:
```bash
uv sync                    # Core + CLI only
uv sync --group api        # Core + CLI + API
uv sync --group dev        # Core + CLI + dev tools
uv sync --group pre-commit # Core + CLI + pre-commit hooks
```

## Critical Analysis & Tradeoffs

This section provides honest assessment of our architectural choices, their limitations, and when you should NOT use this approach.

### Model Lifecycle: Per-Request Load/Unload

**What we chose:**
- Load model on each request (~26s)
- Run inference (~80s)
- Unload to free memory
- Total: ~106s per request

**Tradeoffs:**

✅ **Advantages:**
- Minimal idle memory usage (model unloaded between requests)
- Can run on smaller/cheaper instances (4GB RAM vs 8GB+)
- No state management complexity
- Perfect for sporadic, low-volume use (few requests per hour)

❌ **Disadvantages:**
- **26-second cold start penalty on EVERY request** (not acceptable for many use cases)
- Cannot handle concurrent requests efficiently (each needs full model load)
- Wasteful for high-volume scenarios (repeatedly loading same model)
- No request batching possible

**When NOT to use this approach:**
- **High volume**: >10 requests/hour → Use vLLM with model kept in memory
- **Low latency required**: <10s response time needed → Keep model loaded
- **Concurrent requests**: Multiple simultaneous users → Use model pooling or queuing
- **Predictable traffic**: Steady request rate → Persistent model loading is more efficient

**Alternatives:**
1. **Keep model in memory**: 0s load time, but ~8GB RAM constantly used
2. **vLLM with PagedAttention**: Efficient memory management + batching for high throughput
3. **Model pooling**: Multiple instances with load balancing for concurrency
4. **Hybrid approach**: Keep model loaded for N minutes after last request (lazy unload)

### Hardware: CPU vs MPS vs CUDA

**What we found (GOT-OCR-2.0-hf on M2):**
- CPU: ~80s inference ✅ Recommended
- MPS: ~220s inference (2.7x slower)
- CUDA: Not tested

**Critical analysis:**

**CPU Mode (Recommended for M2):**
- ✅ Fastest for this specific model on M2
- ✅ Available on all hardware
- ✅ Consistent performance
- ❌ Slower than CUDA on NVIDIA GPUs
- ❌ No parallelization benefits for this model

**MPS Mode (Apple Silicon GPU):**
- ❌ **2.7x slower than CPU for GOT-OCR-2.0-hf**
- Why? Model architecture doesn't benefit from Metal acceleration
- Memory transfers between CPU/GPU add overhead
- **Do NOT use MPS for this specific model**

**CUDA Mode (NVIDIA GPUs):**
- ✅ Likely fastest option (15-30s inference on T4/A10G GPUs)
- ✅ Better for high-volume production
- ❌ Requires NVIDIA hardware ($$$)
- ❌ Higher cloud costs (GPU instances 3-5x more expensive)
- ❌ Not tested in this project

**Cost implications:**
- **M2 CPU**: Free (local) or ~$0.10/hour (cloud CPU instance)
- **NVIDIA T4**: ~$0.35/hour (GCP/AWS GPU instance)
- **NVIDIA A10G**: ~$0.80/hour (GCP/AWS GPU instance)

**Decision matrix:**
- **Development/testing**: M2 CPU (free, good enough)
- **Low volume production (<50 requests/day)**: CPU instance ($5-10/month)
- **High volume production (>100 requests/day)**: CUDA GPU (faster, but $250+/month)
- **Apple Silicon only**: Stick with CPU, never use MPS

### API Monitoring & Management Limitations

**What we implemented:**
- Basic Python logging (stdout/stderr)
- Health endpoint for uptime checks
- Response metadata (inference_time, model_id, device)
- Error logging with stack traces

**What we're MISSING (critical for production):**

❌ **No structured logging**
- Logs are plain text, not JSON
- Hard to parse/query in log aggregation tools
- No correlation IDs for request tracing

❌ **No metrics collection**
- No Prometheus/StatsD integration
- Can't track: requests/sec, error rates, latency percentiles
- No dashboards (Grafana, etc.)

❌ **No usage tracking**
- No per-user/API key tracking
- Can't bill based on usage
- Can't detect abuse or unusual patterns

❌ **No rate limiting**
- Vulnerable to abuse (someone can spam requests)
- No quotas or throttling
- Can exhaust server resources

❌ **No request queuing**
- Concurrent requests compete for resources
- No backpressure mechanism
- Server can be overwhelmed

❌ **No distributed tracing**
- Can't track requests across services
- No OpenTelemetry/Jaeger integration
- Hard to debug in multi-service environments

**When our simple approach is insufficient:**
1. **Production SaaS**: Need billing, quotas, abuse prevention → Add API gateway (Kong, Tyk)
2. **Multi-tenant**: Need per-user tracking → Add authentication + usage metering
3. **High reliability**: Need SLAs, monitoring, alerts → Add Prometheus + Grafana + PagerDuty
4. **Microservices**: Need distributed tracing → Add OpenTelemetry
5. **Compliance**: Need audit logs → Add structured JSON logging with retention policies

**Production-ready monitoring stack:**
```bash
# What you'd need to add:
- API Gateway: Kong/Tyk (rate limiting, auth, usage tracking)
- Metrics: Prometheus + Grafana (dashboards, alerts)
- Logging: Structured JSON → ELK/Loki (search, analysis)
- Tracing: OpenTelemetry → Jaeger (distributed tracing)
- Alerting: PagerDuty/Opsgenie (incident response)
```

### FastAPI vs Ollama vs vLLM: What We're Giving Up

**What we chose: FastAPI with manual model management**

**vs Ollama:**

✅ **We gain:**
- Full control over model lifecycle
- Works with vision-language models (Ollama focused on chat LLMs)
- Simpler for OCR-specific tasks
- No Ollama installation/configuration needed

❌ **We lose:**
- No built-in model management (pull, update, list models)
- No built-in prompt templating
- No multi-model serving out of the box
- No chat history/conversation management
- Manual implementation of everything

**vs vLLM:**

✅ **We gain:**
- Simpler codebase (no complex serving infrastructure)
- Works for low-volume use (vLLM optimized for throughput)
- Lower resource requirements (vLLM needs GPU + more RAM)
- Vision-language model support without constraints

❌ **We lose:**
- **No PagedAttention** (efficient memory management for KV cache)
- **No continuous batching** (process multiple requests in parallel)
- **No request queuing** (vLLM handles concurrent requests gracefully)
- **3-10x slower throughput** for high-volume scenarios
- No tensor parallelism (vLLM can split model across GPUs)
- No automatic quantization/optimization

**When you SHOULD use the alternatives:**

**Choose Ollama if:**
- You want a complete LLM serving solution with minimal code
- You're serving chat models (Llama, Mistral, etc.)
- You want model management features (pull, update, versioning)
- You need conversation history and prompt templating

**Choose vLLM if:**
- High throughput required (>100 requests/hour)
- You have GPU resources available
- You need concurrent request handling
- You want state-of-the-art serving optimizations
- You're serving pure LLMs (vLLM has best support for text models)

**Choose our FastAPI approach if:**
- Low volume, occasional use (<50 requests/day)
- Vision-language OCR tasks (not pure chat)
- Need full control over model lifecycle
- Want to minimize idle resource usage
- Experimenting or prototyping

### Summary: This Project Is Optimized For...

✅ **Perfect use cases:**
- Personal OCR tool (few images per day)
- Internal company tool (sporadic document processing)
- Development/testing of GOT-OCR model
- Cost-sensitive deployments (minimize idle costs)
- Learning project for VLM deployment

❌ **Wrong use cases:**
- High-volume OCR service (>100 images/day) → Use vLLM + GPU
- Low-latency requirements (<10s response) → Keep model in memory
- Multi-tenant SaaS → Need API gateway, metering, monitoring
- Production-critical system → Need proper monitoring, alerting, SLAs
- Concurrent users → Need request queuing or model pooling

**If your use case is in the "wrong" category, this architecture will cause pain. Choose the right tool for your actual needs.**

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

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
uv sync --group pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on staged files (automatic on git commit)
git commit -m "your message"
```

**Hooks included:**
- **File hygiene**: trailing whitespace, end-of-file fixer, line endings
- **Validation**: YAML, TOML, JSON, large files, merge conflicts
- **Python**: Ruff linting and formatting (100 char line length)
- **Security**: Secret detection (excluding notebooks)
- **Notebooks**: Ruff linting, strip outputs while keeping metadata

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

## Roadmap

Future improvements and tasks:

- [ ] **Configuration validation**: Ensure all .env settings in config.py are actually used throughout the codebase
- [ ] **Configuration documentation**: Document complete override hierarchy (CLI args > environment vars > .env defaults)
- [ ] **Dependency cleanup**: Simplify pydantic-settings and python-multipart usage if possible
- [ ] **Docker hardware testing**: Test and document Docker performance across different hardware (M2 CPU, Intel CPU, NVIDIA GPU)
- [ ] **Documentation cleanup**: Review and consolidate documentation across README, CLAUDE.md, and code comments
- [ ] **Performance benchmarking**: Create reproducible benchmarks for different hardware configurations
- [ ] **Cloud deployment guide**: Add detailed GCP Cloud Run deployment instructions with cost estimates
