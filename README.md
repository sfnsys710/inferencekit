# OCR VLM

OCR experimentation project using GOT-OCR-2.0-hf model from HuggingFace, optimized for Apple Silicon M2.

## Features

- Vision Language Model-based OCR using GOT-OCR-2.0-hf
- MPS (Metal Performance Shaders) support for Apple Silicon M2
- Dual entry points: CLI and REST API
- Designed for low-volume, occasional use

## Tech Stack

- **Dependency Management**: uv
- **ML Framework**: transformers + torch (with MPS support)
- **API**: FastAPI (Phase 4)
- **CLI**: Fire library (Phase 3)
- **Deployment**: Docker for GCP Cloud Run (Phase 5)

## Project Structure

```
ocrvlm/
├── notebooks/          # Jupyter notebooks for experiments
├── src/ocrvlm/        # Core reusable Python modules
├── scripts/           # CLI entry points (Phase 3)
├── api/              # FastAPI server (Phase 4)
└── Dockerfile        # Container deployment (Phase 5)
```

## Setup

1. Install uv if you haven't already

2. Clone and setup:
   ```bash
   git clone <repo-url>
   cd ocrvlm
   uv sync
   ```

3. Copy environment template:
   ```bash
   cp .env.example .env
   ```

## Development

### Phase 1: M2 Compatibility Testing (Current)

### Future Phases

- **Phase 2**: Core implementation (model manager, processing utilities)
- **Phase 3**: CLI scripts with Fire
- **Phase 4**: FastAPI REST API
- **Phase 5**: Docker containerization

## Requirements

- Python 3.12+
- Apple Silicon M2 (for MPS acceleration)
- 8GB+ RAM recommended

## License

MIT License