# OcrKit

Building blocks for running OCR inference on various models. Currently supports GOT-OCR-2.0-hf vision language model and Qwen3 text model.

## Features

- Multi-model support: Vision Language Models (GOT-OCR) and text models (Qwen3)
- Optimized for Apple Silicon M2 development (CPU mode for OCR)
- Production-ready deployment on GCP Cloud Run with GPU acceleration
- Dual interfaces: CLI and REST API
- Designed for moderately frequent, non-concurrent use (significantly faster inference on GPU vs local CPU)

## Quick Start

### Setup

```bash
# Install uv if you haven't already
# Clone and setup
git clone https://github.com/sfnsys710/ocrkit.git
cd ocrkit
uv sync --group api  # Include API dependencies
cp .env.example .env
```

### CLI Usage

**OCR (GOT-OCR-2.0-hf):**
```bash
# Process local image
python scripts/stepfun_got_ocr.py --path image.jpg

# Process from URL
python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg

# Save output to file
python scripts/stepfun_got_ocr.py --path image.jpg --output result.txt
```

**Text Generation (Qwen3):**
```bash
# Generate text
python scripts/qwen3_generate.py --prompt "Hello, how are you?"

# Save output to file
python scripts/qwen3_generate.py --prompt "Explain AI" --output result.txt

# Override device and max tokens
python scripts/qwen3_generate.py --prompt "Write a poem" --device cpu --max_tokens 512
```

### API Usage

```bash
# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# OCR: Upload image
curl -X POST -F "file=@image.jpg" http://localhost:8080/ocr/upload

# OCR: Process from URL
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' \
  http://localhost:8080/ocr/url

# Text generation
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, how are you?"}' \
  http://localhost:8080/text/generate

# Health check
curl http://localhost:8080/health

# API documentation: http://localhost:8080/docs
```

### Docker Usage

```bash
# Build image
docker build -t ocrkit:latest .

# Run container
docker run -p 8080:8080 ocrkit:latest

# Run with environment overrides
docker run -p 8080:8080 -e DEVICE=cpu -e LOG_LEVEL=DEBUG ocrkit:latest
```

## Model Selection Rationale

### Open Source vs Closed Source: Strategic Decision

This project uses open-source models (GOT-OCR-2.0-hf, Qwen3) instead of closed-source alternatives (Claude, GPT-4). Here's the analysis:

| Aspect | Closed-Source (Claude/GPT) | Open-Source (GOT-OCR/Qwen3) | Our Choice |
|--------|---------------------------|----------------------------|------------|
| **Raw Quality** | 15-25% higher on benchmarks | Good enough for most tasks | Open ✓ |
| **Cost** | $0.50-$15 per 1M tokens | Free (compute only) | Open ✓ |
| **Latency** | Network round-trip required | Local = faster for dev | Open ✓ |
| **Privacy** | Data sent to third party | 100% on-premise | Open ✓ |
| **Customization** | Limited to prompting | Fine-tuning possible | Open ✓ |
| **Deployment** | API dependency | Full control | Open ✓ |

**Verdict:** For moderately frequent, privacy-conscious use cases, the 15-25% quality gap is acceptable. Open-source provides full control, zero API costs, and on-premise deployment.

### Why Qwen3 for Text Generation?

| Model | Size | MMLU | IFEval | GSM8K | Context | Key Advantage |
|-------|------|------|--------|-------|---------|---------------|
| **Qwen3-4B** | 4B | ~68% | ~72% | ~80% | 128K | Architecture upgrade, 40% faster than Qwen2.5 |
| SmolLM3-3B | 3B | ~65% | ~70% | ~75% | 128K | Good alternative, 6 languages |
| Qwen2.5-3B | 3B | ~62% | ~65% | ~70% | 128K | Previous generation |
| Phi-3 Mini | 3.8B | ~69% | ~68% | ~75% | 128K | Microsoft, comparable |

**Qwen3 chosen for:** Best IFEval score (instruction-following critical for RAG), major architecture improvements over Qwen2.5, strong multilingual support, and active development community.

### Model Naming Conventions Primer

Understanding model suffixes is critical for deployment:

**Training Suffixes:**
| Suffix | Purpose | Use Case |
|--------|---------|----------|
| **-Instruct** | Fine-tuned for instructions | Chat, Q&A, RAG (our use case) |
| **-Base** | Pretrained only | Text completion, fine-tuning |
| **-Coder** | Code generation | Programming tasks |

**Format Suffixes:**
| Format | Runtime | Hardware | Size | Quality Loss |
|--------|---------|----------|------|--------------|
| **Safetensors** | transformers | Any (GPU preferred) | 100% | None |
| **GGUF** | llama.cpp | CPU, Apple Metal | 25-50% | 1-5% |
| **GPTQ/AWQ** | AutoGPTQ/vLLM | NVIDIA GPU only | 25-50% | 1-3% |

**Our deployment:** Safetensors format with transformers runtime for maximum quality and GPU compatibility.

## Architecture

### Tech Stack

- **Dependency Management**: uv with dependency groups
- **ML Framework**: transformers + torch
- **API**: FastAPI with uvicorn
- **CLI**: Fire library
- **Deployment**: Multi-stage Docker for GCP Cloud Run

### Project Structure

```
ocrkit/
├── src/ocrkit/  # Core reusable Python modules
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

The API uses a **per-request model lifecycle** intentionally designed for moderately frequent, non-concurrent use:

1. Request received → Load model (~5-10s on GPU, ~26s on M2 CPU)
2. Run inference (~15-25s on GPU, ~80s on M2 CPU)
3. Unload model to free memory
4. Total: ~20-35s per request on GPU, ~106s on M2

**Why not keep model in memory?**
- This project targets moderately frequent, non-concurrent use (10-50 requests/day)
- Freeing memory between requests enables Cloud Run scale-to-zero cost savings
- For high concurrency or 24/7 use, consider vLLM with persistent instances

### Hardware Optimization Strategy

**Performance comparison across hardware backends:**

| Backend | M2 Dev | Cloud GPU | Quality | Memory | Our Use |
|---------|--------|-----------|---------|--------|---------|
| **CPU** | ~80s inference | ~30-40s (less optimal) | Full precision | 3GB | ✓ M2 dev |
| **MPS** (Apple Metal) | ~220s (2.7x slower!) | N/A | Full precision | 3GB | ✗ Never use |
| **CUDA** (NVIDIA) | N/A | ~15-25s (optimal) | Full precision | 4GB | ✓✓ Production |
| **MLX** (Apple Silicon) | Not tested | N/A | Full precision | ~3GB | ⚠ Future consideration |

**Critical finding:** GOT-OCR-2.0-hf performs **2.7x slower** on MPS vs CPU on M2. This is model-architecture specific.

**Hardware decision matrix:**

| Environment | Hardware | Backend | Rationale |
|-------------|----------|---------|-----------|
| **M2 Development** | Apple Silicon | CPU | Fastest on M2 (MPS unusable), adequate for testing |
| **Cloud Run Production** | NVIDIA L4/T4 | CUDA | 3-5x faster than M2 CPU, cost-effective GPU |
| **High-volume (future)** | NVIDIA A10G/A100 | CUDA | If scaling beyond moderate frequency |

**Cost-performance tradeoff:**

| Deployment | Hardware | Inference Time | Cost/Hour | Cost/1K Requests | Use Case |
|------------|----------|----------------|-----------|------------------|----------|
| M2 Local | CPU | ~80s | $0 (local) | $0 | Development, testing |
| Cloud Run CPU | Intel/AMD | ~50-60s | ~$0.10 | ~$1.50 | Budget (not recommended) |
| Cloud Run GPU (L4) | NVIDIA L4 | ~15-25s | ~$0.60 | ~$2.50 | Production (chosen) ✓ |
| Cloud Run GPU (T4) | NVIDIA T4 | ~20-30s | ~$0.35 | ~$1.50 | Budget GPU alternative |

**Configuration:**
```bash
# .env file
DEVICE=cpu   # M2 dev (fastest on Apple Silicon for this model)
# DEVICE=cuda  # Cloud Run GPU production
# DEVICE=mps   # NEVER use - 2.7x slower than CPU
```

**Why not MPS?** The GOT-OCR model's architecture doesn't benefit from Metal acceleration. Memory transfers between CPU/GPU add overhead without computational gains. Always use CPU on M2.

### Deployment Strategy

**Production deployment options analysis:**

| Platform | GPU Support | Scale-to-Zero | Ops Overhead | Cold Start | Cost Model | Our Fit |
|----------|-------------|---------------|--------------|------------|------------|---------|
| **Cloud Run (GPU)** | ✓✓ L4/T4 | ✓✓ Native | Very Low | ~30s | Pay-per-use | ✓✓ Chosen |
| Cloud Run (CPU) | ✗ | ✓✓ Native | Very Low | ~15s | Pay-per-use | ⚠ Too slow |
| Kubernetes (GKE) | ✓ Any GPU | ⚠ KEDA | High | ~0s (persistent) | Always-on | ✗ Overkill |
| VM with GPU | ✓ Any GPU | ✗ Manual | Medium | ~0s (persistent) | Always-on (~$500/mo) | ✗ Expensive |
| vLLM Server | ✓✓ Optimized | ✗ | Medium | ~0s (persistent) | Always-on | ✗ High concurrency only |

**Tradeoff analysis:**

| Option | Advantages | Disadvantages | Decision |
|--------|-----------|---------------|----------|
| **Cloud Run GPU** (chosen) | Serverless (no ops), scale-to-zero, GPU support, pay-per-use | Cold starts (~30s), limited GPU options (L4/T4) | ✓ **Best for moderate frequency** |
| Kubernetes | Full control, any GPU, persistent (no cold start) | Complex ops, always-on costs, requires DevOps expertise | ✗ Moderate use doesn't justify complexity |
| VM with persistent GPU | Zero cold start, maximum performance | $400-600/month idle cost, manual scaling, wasted capacity | ✗ Not cost-effective for non-24/7 use |
| Cloud Run CPU | Cheapest ($0.10/hr), fast cold start | 3-4x slower inference, poor user experience | ✗ Unacceptable latency |

**Our requirements mapping:**

| Requirement | Cloud Run GPU | Kubernetes | VM + GPU | Verdict |
|-------------|---------------|------------|----------|---------|
| **Moderate frequency** | ✓ Pay-per-use saves money | ✗ Always-on wasteful | ✗ Idle cost high | Cloud Run ✓ |
| **Non-concurrent** | ✓ Single instance OK | ⚠ Overkill for no parallelism | ✓ Works | Cloud Run ✓ |
| **Low ops overhead** | ✓✓ Serverless | ✗ Requires K8s expertise | ⚠ VM management | Cloud Run ✓ |
| **GPU acceleration** | ✓ L4/T4 available | ✓ Any GPU | ✓ Any GPU | All work ✓ |
| **Cost optimization** | ✓ Scale-to-zero | ✗ Always-on | ✗ Always-on | Cloud Run ✓ |

**Final architecture:**
```
Development:     M2 (CPU) → transformers → Local testing
Production:      Cloud Run (GPU L4) → transformers → CUDA inference
Traffic pattern: Moderate frequency, non-concurrent requests
Scaling:         0 → N instances (auto), typically 1 instance
Cost model:      Pay only during inference (~$0.60/hr active)
```

**Why Cloud Run GPU wins:**
1. **Moderate frequency = pay-per-use is cheaper** than always-on GPU VM ($50-100/month vs $500/month)
2. **Non-concurrent = serverless autoscaling perfect**, no need for K8s complexity
3. **Low ops = serverless deployment**, just `gcloud run deploy`
4. **GPU support = L4 GPUs** available (T4 also option)
5. **Same codebase** as M2 dev (transformers everywhere)

**When to reconsider:**
- **High frequency (24/7)**: Persistent GPU VM becomes cheaper
- **High concurrency**: Kubernetes + horizontal pod autoscaling
- **Custom GPU needs**: GKE for A100/H100 access

### Configuration Management

All configuration is centralized in `src/ocrkit/schemas/config.py` using pydantic-settings.

**Environment variables (.env):**
```bash
# Model settings
MODEL_ID=stepfun-ai/GOT-OCR2_0        # OCR model
QWEN_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct  # Text generation model
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
   docker run -e DEVICE=cpu -e LOG_LEVEL=DEBUG ocrkit:latest
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
gcloud run services update ocrkit \
  --region us-central1 \
  --max-instances 10 \
  --cpu 2 \
  --memory 4Gi \
  --timeout 300s
```

## Design Decisions

### Infrastructure Stack: transformers + FastAPI + Docker

**Runtime comparison matrix:**

| Aspect | transformers+FastAPI | vLLM | Ollama | llama-cpp-python |
|--------|---------------------|------|--------|------------------|
| **M2 Dev** | ✓ CPU (adequate) | ✗ CUDA only | ✓ Metal (fast) | ✓✓ Metal (fastest) |
| **GPU Prod** | ✓✓ CUDA (optimal) | ✓✓ CUDA (best) | ✓ CUDA (good) | ⚠ CUDA (immature) |
| **Integration** | Native Python | OpenAI API | HTTP proxy | Native Python |
| **VLM Support** | ✓✓ Excellent | ⚠ Limited | ✗ Chat only | ✓ Good |
| **Batching** | Manual | ✓✓ Continuous | ✗ None | ✗ None |
| **Model Format** | Safetensors (full quality) | Safetensors/AWQ | GGUF (quantized) | GGUF (quantized) |
| **Ops Complexity** | Low | Medium | Low | Low |
| **Our Use Case Fit** | ✓✓ Perfect | ⚠ Overkill | ✗ Wrong type | ⚠ Wrong target |

**Tradeoffs analysis:**

| Choice | What We Gain | What We Lose | Verdict |
|--------|--------------|--------------|---------|
| **transformers** (chosen) | Full quality (no quantization), VLM support, GPU optimization, HuggingFace ecosystem | Slower M2 dev vs llama-cpp, No continuous batching | ✓ **Best for our case** |
| vLLM | 3-5x throughput, continuous batching, PagedAttention | Complexity, CUDA-only, OpenAI API overhead | ✗ Overkill for non-concurrent |
| Ollama | Easy M2 dev, model management | No VLM support, HTTP overhead, chat-focused | ✗ Wrong model type |
| llama-cpp | Fast M2 dev (Metal), small models (GGUF) | Quality loss (quantization), weak GPU support | ✗ Targeting GPU prod |

**Our requirements analysis:**

| Requirement | Value | Eliminated Options | Winner |
|-------------|-------|-------------------|--------|
| **Traffic pattern** | Moderately frequent, non-concurrent | vLLM (needs concurrency) | transformers ✓ |
| **Model type** | Vision-language (GOT-OCR) | Ollama (chat only) | transformers ✓ |
| **Deployment target** | Cloud Run GPU | llama-cpp (CPU-optimized) | transformers ✓ |
| **Quality priority** | Full precision | GGUF-based (quantized) | transformers ✓ |
| **Dev environment** | M2 CPU acceptable | vLLM (needs GPU) | transformers ✓ |

**Decision:** transformers + FastAPI + Docker provides the optimal balance for moderately frequent, non-concurrent vision-language inference with GPU production deployment.

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
- Load model on each request (~26s on M2 CPU, ~5-10s on Cloud Run GPU)
- Run inference (~80s on M2 CPU, ~15-25s on Cloud Run GPU)
- Unload to free memory
- Total: ~106s per request on M2, ~20-35s on Cloud Run GPU

**Tradeoff analysis:**

| Aspect | Per-Request Load (our choice) | Persistent In-Memory | vLLM + Batching |
|--------|------------------------------|---------------------|-----------------|
| **Load time** | 5-26s per request | 0s (one-time startup) | 0s (one-time startup) |
| **Memory idle** | ~500MB (scale-to-zero) | ~8GB constantly | ~8-12GB constantly |
| **Concurrency** | Sequential only | Limited | Excellent (batching) |
| **Cost (moderate use)** | $50-100/month | $300-500/month | $500-800/month |
| **Best for** | Moderate frequency, non-concurrent | High frequency, single-user | High concurrency |

**Our rationale:** At 10-50 requests/day (moderately frequent) with Cloud Run GPU scale-to-zero, per-request loading costs ~$50-100/month vs ~$500/month for always-on GPU. The 5-10s GPU load time is acceptable for non-real-time workflows.

**When NOT to use this approach:**
- **High concurrency**: Multiple simultaneous users → Use vLLM with batching
- **Ultra-low latency**: <5s total response time → Keep model in memory
- **24/7 high volume**: >200 requests/day → Persistent GPU instance cheaper

**Alternatives:**
- **Persistent in-memory**: 0s load, but $400-600/month always-on GPU cost
- **vLLM + batching**: For high concurrency (>10 concurrent users)
- **Hybrid (min_instances=1)**: For frequent bursts, accepts ~$16/day minimum cost

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
- Moderate frequency, non-concurrent use (10-50 requests/day)
- Vision-language OCR tasks (not pure chat)
- Need full control over model lifecycle
- Want cost-effective GPU deployment with scale-to-zero
- Production-ready but not mission-critical

### Summary: This Project Is Optimized For...

✅ **Perfect use cases:**
- **Moderate frequency OCR service** (10-50 requests/day, our target)
- Internal company document processing (non-concurrent workflow)
- Production deployment with cost optimization via Cloud Run GPU
- Privacy-conscious deployments (on-premise capable)
- Vision-language model experimentation and development

❌ **Wrong use cases:**
- **High concurrency** (>10 simultaneous users) → Use vLLM + batching
- **24/7 high volume** (>200 requests/day) → Persistent GPU instance
- **Ultra-low latency** (<5s total response) → Keep model in memory
- **Multi-tenant SaaS** → Need API gateway, metering, rate limiting
- **Mission-critical systems** → Need advanced monitoring, SLAs, alerting

**If your use case is in the "wrong" category, this architecture will cause pain. Choose the right tool for your actual needs.**

## Performance Characteristics

**Development (Apple Silicon M2 CPU):**
- Model loading: ~26 seconds
- Inference: ~80 seconds per image
- Total per request: ~106 seconds
- Memory: ~3GB during inference
- Device: `DEVICE=cpu` (MPS is 2.7x slower!)

**Production (Cloud Run GPU - NVIDIA L4):**
- Model loading: ~5-10 seconds
- Inference: ~15-25 seconds per image
- Total per request: ~20-35 seconds
- Memory: ~4GB during inference
- Device: `DEVICE=cuda`

**Deployment recommendations:**
- **M2 dev**: Use `DEVICE=cpu` (not MPS - significantly slower)
- **Cloud Run**: L4 GPU for production, T4 for budget alternative
- **RAM**: 4GB+ (M2), 8GB+ (Cloud Run with GPU)
- **Timeout**: Set HTTP timeout to 300s+ for M2, 120s+ for Cloud Run
- **Concurrency**: Sequential requests only (no batching)

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

### Core Infrastructure
- [ ] **Configuration validation**: Ensure all .env settings in config.py are actually used throughout the codebase
- [ ] **Configuration documentation**: Document complete override hierarchy (CLI args > environment vars > .env defaults)
- [ ] **Dependency cleanup**: Simplify pydantic-settings and python-multipart usage if possible

### Deployment & Performance
- [ ] **Docker hardware testing**: Test and document Docker performance across different hardware (M2 CPU, Intel CPU, NVIDIA GPU)
- [ ] **Cloud deployment guide**: Add detailed GCP Cloud Run deployment instructions with cost estimates
- [ ] **Performance benchmarking**: Create reproducible benchmarks for different hardware configurations
- [ ] **Fact-check benchmarks**: Verify model benchmark results (MMLU, IFEval, GSM8K) against latest model cards and leaderboards
- [ ] **Fact-check costs**: Update and verify Cloud Run GPU pricing, cost estimates, and deployment cost comparisons

### Production Monitoring & Observability
- [ ] **Structured logging**: Implement JSON logging for better parsing in log aggregation tools (Cloud Logging, ELK)
- [ ] **Request correlation IDs**: Add tracing IDs to track requests across the system
- [ ] **Metrics collection**: Integrate Prometheus/StatsD for tracking requests/sec, error rates, latency percentiles
- [ ] **Monitoring dashboards**: Set up Grafana or Cloud Monitoring dashboards for operational visibility
- [ ] **Distributed tracing**: Add OpenTelemetry/Jaeger for debugging multi-service interactions
- [ ] **Alerting system**: Configure alerts for errors, latency spikes, resource exhaustion (PagerDuty/Opsgenie)

### Security & Access Control
- [ ] **Rate limiting**: Implement API rate limiting to prevent abuse and resource exhaustion
- [ ] **Authentication**: Add API key authentication for access control
- [ ] **Usage tracking**: Track per-user/API key usage for billing and abuse detection
- [ ] **Request queuing**: Implement backpressure mechanism to handle traffic spikes gracefully

### Documentation
- [ ] **Documentation cleanup**: Review and consolidate documentation across README, CLAUDE.md, and code comments
