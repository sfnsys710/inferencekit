# AGENTS.md

Guidance for agentic coding agents working with the inferencekit codebase.

## Build/Lint/Test Commands

```bash
# Environment setup
uv sync
uv sync --group dev --group api --group pre-commit --group experiment

# Code quality
ruff check --fix --unsafe-fixes
ruff format
ruff check --fix --unsafe-fixes --exit-non-zero-on-fix && ruff format

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Testing (when pytest is added)
pytest                           # All tests
pytest tests/test_specific.py    # Single file
pytest tests/test_specific.py::test_function_name  # Single function
pytest --cov=inferencekit        # With coverage

# Running the application
python scripts/stepfun_got_ocr.py --path image.jpg
python scripts/qwen3_generate.py --prompt "Hello"
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Docker
docker build -t inferencekit:latest .
docker run -p 8080:8080 inferencekit:latest
```

## Code Style Guidelines

### General Principles
- **Explicit over implicit**: Use explicit parameters (`--path`/`--url`) rather than generic ones
- **Logging only**: No print statements - use Python's logging module exclusively
- **Raise errors, don't exit**: Prefer `raise ValueError()` over `sys.exit(1)`
- **Single settings class**: All configuration in `schemas/config.py` using pydantic-settings

### Import Style
- Use `isort` rules (handled by ruff)
- Group imports: standard library, third-party, local imports
- Use relative imports for local modules: `from .schemas.config import Settings`
- Import specific names, not modules: `from pathlib import Path` not `import pathlib`

### Type Hints
- Use modern type hints: `str | None` instead of `Optional[str]`
- Use `Literal` for string enums: `Literal["cpu", "mps", "cuda"]`
- Add return types to all functions
- Use `Any` sparingly, prefer specific types

### Naming Conventions
- **Classes**: PascalCase (e.g., `ImageHandler`, `GOTOCRModel`)
- **Functions/variables**: snake_case (e.g., `load_from_url`, `max_new_tokens`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_IMAGE_SIZE_MB`)
- **Private methods**: prefix with underscore (e.g., `_validate_image`)

### Error Handling
- Use specific exception types: `ValueError`, `FileNotFoundError`, `HTTPException`
- Include descriptive error messages
- Let exceptions propagate to calling code
- In API, map exceptions to HTTP status codes:
  - `ValueError` → 400 Bad Request
  - `FileNotFoundError` → 404 Not Found
  - Other errors → 500 Internal Server Error

### Documentation
- Use triple quotes for all module, class, and function docstrings
- Follow Google-style docstring format
- Include Args, Returns, Raises sections where applicable
- Add usage examples in CLI scripts

### Code Structure
- **Core modules** (`src/inferencekit/`): Reusable components with clear interfaces
- **CLI scripts** (`scripts/`): Standalone, function-based with Fire library
- **API server** (`api/`): Simple FastAPI with all endpoints in `main.py`
- **Configuration**: Single `Settings` class in `schemas/config.py`

### Logging
- Import logging: `import logging`
- Get module logger: `logger = logging.getLogger(__name__)`
- Configure via `setup_logging()` in scripts, `basicConfig()` in modules
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR

### File Organization
- Keep files focused on single responsibility
- Use `__init__.py` to control public API exports
- Scripts are standalone (no `__init__.py` in `scripts/`)
- Separate schemas from implementation

### Device & Deployment
- **M2 Development**: Use `DEVICE=cpu` (MPS is 2.7x slower for GOT-OCR)
- **Cloud Run Production**: Use `DEVICE=cuda` for GPU acceleration
- Never use MPS for GOT-OCR-2.0-hf model
- Model loads per request (not at startup) for Cloud Run scale-to-zero
- Use `tempfile.NamedTemporaryFile` for uploads; cleanup in finally blocks
- No per-request setting overrides (use .env settings only)

### Docker
- Multi-stage builds with builder and runtime stages
- Use `python:3.12-slim-bookworm` for minimal runtime
- Set `UV_PYTHON_DOWNLOADS=0` and `UV_COMPILE_BYTECODE=1`
- Support runtime environment overrides (`-e DEVICE=cuda`)

## Architecture Notes

### Model Lifecycle
- Lazy loading: Models load only when needed
- Per-request loading: Intentional for Cloud Run cost optimization
- Explicit cleanup: Always call `unload()` after inference

### Settings Management
- Environment-based configuration via `.env` file
- Cached singleton using `@lru_cache`
- CLI can override settings via parameters
- API uses environment settings only

### Image Processing
- Validate format and size after loading
- Support explicit `load_from_path()` and `load_from_url()` methods
- Raise appropriate exceptions for validation failures

## Development Workflow

1. Make changes following code style guidelines
2. Run `ruff check --fix --unsafe-fixes && ruff format`
3. Test CLI scripts and API endpoints
4. Run `pre-commit run --all-files` before committing
5. Use `uv sync` to update dependencies as needed
