# ocrkit

OCR research toolkit — traditional, LLM-based, and hybrid approaches, applied to the Moroccan context (Arabic, French, Darija). Includes notebooks for POC experimentation and production-ready packaged code with Docker + REST API.

## What's inside

| Layer | Purpose |
|-------|---------|
| `notebooks/` | POC experiments comparing OCR approaches |
| `src/ocrkit/` | Packaged, deployable inference code |
| `api/` | FastAPI REST API |
| `scripts/` | CLI entry points |
| `Dockerfile` | Multi-stage build for GCP Cloud Run GPU |

## OCR Approaches

### Traditional
- **Tesseract** (`tesserocr`) — open-source OCR, good Arabic/French support with trained models
- **doctr** (from Mindee) — deep-learning OCR pipeline, layout-aware

### LLM-based
- **GOT-OCR-2.0-hf** — vision-language model (VLM), end-to-end OCR with layout understanding
- **Qwen3** — LLM for post-processing, correction, and structured extraction

### Hybrid
- Traditional detection → LLM extraction/correction
- Layout analysis (doctr) + VLM understanding (GOT-OCR)

## Moroccan Context

The project specifically targets Arabic script, French/Arabic bilingual documents, and informal Moroccan Arabic (Darija) — common in CINs, administrative forms, receipts, and mixed-language documents.

## Inference Options

The `notebooks/` compare different deployment strategies:

| Option | Dev (M2) | GPU Prod | Concurrency | Use case |
|--------|----------|----------|-------------|----------|
| **transformers** (current) | ✓ CPU | ✓✓ CUDA | Sequential | VLM, full quality |
| **vLLM** | ✗ CUDA only | ✓✓ Best throughput | Concurrent | High volume text |
| **Ollama** | ✓ Metal | ✓ CUDA | Sequential | Easy serving, chat LLMs |
| **llama-cpp** | ✓✓ Metal | ⚠ Immature | Sequential | CPU/Metal, GGUF |

Current production choice: **transformers** — required for VLM (GOT-OCR) support and full precision.

## Quick Start

```bash
git clone https://github.com/sfnsys710/ocrkit.git
cd ocrkit
uv sync
cp .env.example .env
```

### Run OCR (CLI)

```bash
# GOT-OCR VLM
python scripts/stepfun_got_ocr.py --path image.jpg
python scripts/stepfun_got_ocr.py --url https://example.com/image.jpg

# Text generation (Qwen3)
python scripts/qwen3_generate.py --prompt "Extract structured data from this text: ..."
```

### Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

curl -X POST -F "file=@image.jpg" http://localhost:8080/ocr/upload
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' http://localhost:8080/ocr/url
curl http://localhost:8080/health
# Docs: http://localhost:8080/docs
```

### Docker

```bash
docker build -t ocrkit:latest .
docker run -p 8080:8080 -e DEVICE=cpu ocrkit:latest
docker run -p 8080:8080 -e DEVICE=cuda ocrkit:latest  # GPU
```

## Performance

**M2 CPU (development):**
- GOT-OCR: ~26s load + ~80s inference = ~106s/request
- Use `DEVICE=cpu` — MPS is 2.7x slower for this model

**Cloud Run GPU (NVIDIA L4, production):**
- GOT-OCR: ~5-10s load + ~15-25s inference = ~20-35s/request
- Use `DEVICE=cuda`

## Project Structure

```
ocrkit/
├── src/ocrkit/          # Deployable Python package
│   ├── content/         # Image loading (URL + path)
│   ├── models/          # GOTOCRModel, Qwen3Model (+ base)
│   └── schemas/         # Settings, OCRResult, TextResult
├── scripts/             # CLI scripts (Fire-based)
├── api/                 # FastAPI server
│   ├── main.py          # All endpoints
│   └── schemas.py       # Request/response models
├── notebooks/           # POC experiments
│   ├── gotocr.ipynb
│   ├── doctr.ipynb
│   ├── docling.ipynb
│   └── qwen3-06b.ipynb
└── Dockerfile
```

## Configuration

All config via `.env` (see `.env.example`):

```bash
MODEL_ID=stepfun-ai/GOT-OCR2_0
QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct
DEVICE=cpu          # cpu | cuda (never mps for GOT-OCR)
MAX_NEW_TOKENS=4096
LOG_LEVEL=INFO
```

## License

MIT
