# AGENTS.md

Guidance for agentic coding agents working with the ocrkit codebase.

ocrkit is an OCR toolkit covering traditional, LLM-based, and hybrid approaches applied to the Moroccan context (Arabic, French, Darija). Notebooks hold POC experiments; `src/ocrkit/` holds deployable code.

## Build/Lint/Test Commands

```bash
# Setup
uv sync
uv sync --group dev --group api --group experiment

# Linting / formatting
ruff check --fix --unsafe-fixes && ruff format

# Pre-commit
pre-commit install
pre-commit run --all-files

# Tests (when added)
pytest
pytest --cov=ocrkit

# Run
python scripts/stepfun_got_ocr.py --path image.jpg
python scripts/qwen3_generate.py --prompt "Hello"
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Docker
docker build -t ocrkit:latest .
docker run -p 8080:8080 ocrkit:latest
```

## Code Style

- **Explicit params**: `--path` / `--url`, not `--source`
- **Logging only**: `import logging`, `logger = logging.getLogger(__name__)`, `setup_logging()` in scripts — no `print()`
- **Raise, don't exit**: `raise ValueError()` not `sys.exit(1)`
- **Single Settings**: all config in `src/ocrkit/schemas/config.py`
- **Type hints**: `str | None` not `Optional[str]`, return types on all functions
- **Imports**: isort order (stdlib → third-party → local); relative imports inside package
- **Naming**: PascalCase classes, snake_case functions/vars, UPPER_SNAKE constants, `_prefix` private

## Architecture

```
src/ocrkit/         # Deployable package (imported by api/, scripts/)
notebooks/          # POC experiments — not imported by src/
api/                # FastAPI (main.py + schemas.py)
scripts/            # CLI entry points (Fire, standalone)
```

### OCR Approaches in this repo

| Approach | Tools | Location |
|----------|-------|----------|
| Traditional | Tesseract (tesserocr), doctr | notebooks/, experiment group |
| LLM-based | GOT-OCR-2.0-hf, Qwen3 | src/ocrkit/models/ |
| Hybrid | layout + VLM | notebooks/ |

### Inference runtimes (studied in notebooks)

| Runtime | VLM | Dev (M2) | GPU | Notes |
|---------|-----|----------|-----|-------|
| transformers (current) | ✓✓ | CPU | CUDA | Required for GOT-OCR |
| vLLM | ⚠ | ✗ | ✓✓ | High concurrency only |
| Ollama | ✗ | ✓ Metal | ✓ | Chat LLMs |
| llama-cpp | ✓ | ✓✓ Metal | ⚠ | GGUF, quantized |

### Model lifecycle

Per-request load/unload (intentional — Cloud Run scale-to-zero):
- `model.load()` → inference → `model.unload()`
- Never keep model in memory between requests

### Error handling (API)

- `ValueError` → 400
- `FileNotFoundError` → 404
- URL download failure → 422
- Other → 500

### Device

- M2 dev: `DEVICE=cpu` (MPS is 2.7x slower for GOT-OCR — never use MPS)
- Production: `DEVICE=cuda`

## Development Workflow

1. Edit code following style above
2. `ruff check --fix --unsafe-fixes && ruff format`
3. Test CLI / API endpoints manually
4. `pre-commit run --all-files` before committing
5. Add notebook POCs in `notebooks/` for new approaches
