# ocrkit

OCR research toolkit — traditional, LLM-based, and hybrid approaches, applied to the Moroccan context (Arabic, French, Darija). Includes notebooks for POC experimentation and production-ready packaged code with Docker + REST API.

## What's inside

| Layer | Purpose |
|-------|---------|
| `notebooks/` | POC experiments comparing OCR approaches |
| `src/ocrkit/` | Packaged, deployable inference code |
| `api/` | FastAPI REST API |
| `scripts/` | CLI entry points |
| `evals/` | Evaluation pipeline (RAGAS + Anthropic judge LLM) |
| `dashboard/` | Dash app for visualizing eval results |
| `Dockerfile` | Multi-stage build for GCP Cloud Run GPU |

## OCR Approaches

### Traditional
- **Tesseract** (`tesserocr`) — open-source OCR, good Arabic/French support with trained models
- **doctr** (from Mindee) — deep-learning OCR pipeline, layout-aware

### LLM-based
- **GOT-OCR-2.0-hf** — vision-language model (VLM), end-to-end OCR with layout understanding
- **Qwen3** — LLM for post-processing, correction, and structured extraction

### Hybrid
- docling: VLM Layout analysis + Tesseract or easyocr OCR engine

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

## Evaluation Pipeline

Evaluating OCR quality is hard to automate — extracting structured fields from identity documents requires judging both format correctness and field-level accuracy. Human annotation at scale is costly and slow. This project uses **[RAGAS](https://docs.ragas.io/)** to implement a reproducible LLM-as-judge evaluation pipeline.

> "Without evaluation, you're flying blind regarding whether changes improve performance or break functionality." — Arize AI

### Why LLM-as-judge?

LLM-based evaluation addresses a core constraint: getting human feedback on even a fraction of model outputs is expensive. An LLM judge automates this while maintaining consistency and scalability. RAGAS formalizes this with structured metrics, ground truth comparison, and CSV-backed experiment tracking.

### Benchmarking strategy

The evaluation deliberately starts with **closed-source frontier models** (Claude Haiku, Claude Sonnet) to establish a quality ceiling. These results serve as the benchmark that open-source models (GOT-OCR, doctr, Tesseract) and hybrid approaches will be measured against.

| Model | Type | Role |
|-------|------|------|
| `claude-haiku-4-5` | Closed-source | Fast baseline, lower cost |
| `claude-sonnet-4-6` | Closed-source | Quality ceiling / gold standard |
| GOT-OCR-2.0-hf | Open-source VLM | Primary open-source candidate |
| doctr / Tesseract | Traditional | Rule-based / DL baseline |

### Metrics

Two RAGAS metrics judge each extraction:

| Metric | Type | Description |
|--------|------|-------------|
| `correct_fields` | `NumericMetric` (0–8) | How many of the 8 RECTO fields were extracted correctly |
| `format_compliance` | `DiscreteMetric` (correct/incorrect) | Whether the output format matches the expected JSON schema |

A separate `claude-sonnet-4-6` instance acts as the judge LLM, scoring each model's output against ground truth JSON.

### Running experiments

```bash
cd evals
uv run --group evals python run.py --model claude-haiku-4-5-20251001 --doc cin --prompt_version v1
uv run --group evals python run.py --model claude-sonnet-4-6 --doc cin --prompt_version v2
```

Results are saved as CSV files in `evals/experiments/`.

### Dashboard

A Dash app visualizes experiment results across models and prompt versions:

```bash
uv run --group evals python dashboard/app.py
# → http://127.0.0.1:8050
```

**Dashboard sections:**
- **Correct Fields** — bar chart of average field extraction accuracy per model × prompt version
- **Format Compliance** — % of runs with incorrect output format
- **Execution Time** — average inference latency per model × prompt version
- **Reason Summarizer** — uses an LLM to synthesize the judge's reasoning across filtered results, surfacing common failure patterns
- **All Results** — full filterable/sortable table of every experiment run
- **Row Detail** — per-row inspection of generated vs. expected output with judge reasoning

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
├── evals/               # Evaluation pipeline
│   ├── run.py           # Experiment runner (RAGAS + Anthropic)
│   ├── utils.py         # Dataset + prompt loading
│   ├── datasets/cin/    # CIN dataset (images, ground truth, prompts)
│   └── experiments/     # Output CSVs from each run
├── dashboard/           # Results visualization (Dash)
│   ├── app.py           # Main app + callbacks
│   ├── components.py    # Charts + filter panel
│   ├── llm.py           # Reason summarizer (LLM)
│   └── utils.py         # Data loading helpers
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
