import base64
from pathlib import Path

import pandas as pd
import yaml


def get_prompt(doc: str, version: str, **kwargs: str) -> str:
    """Load a versioned prompt from evals/datasets/<doc>/prompts/<version>.yaml.

    Args:
        doc: Doc name (e.g. "cin").
        version: Version file name without extension (e.g. "v1").
        **kwargs: Dynamic values to interpolate (for prompts with template_vars).

    Example:
        prompt = load_prompt("cin", "v1")
        prompt = load_prompt("rp", "v2", today_date="17 February 2026")
    """
    prompt_file = Path(__file__).parent / "datasets" / doc / "prompts" / f"{version}.yaml"

    if not prompt_file.exists():
        available = [p.stem for p in prompt_file.parent.glob("v*.yaml")]
        raise FileNotFoundError(
            f"No '{version}.yaml' found for dataset '{doc}'. Available: {available}"
        )

    data = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))
    prompt: str = data["prompt"]
    template_vars: list = data.get("template_vars", [])

    if template_vars:
        missing = [v for v in template_vars if v not in kwargs]
        if missing:
            raise ValueError(f"Missing template variables for '{version}': {missing}")
        prompt = prompt.format(**kwargs)

    return prompt


def get_prompts(doc: str, tags: list[str] | None = None) -> list[dict]:
    """List available prompt versions for a doc, optionally filtered by tags.

    Args:
        doc: Doc name (e.g. "cin").
        tags: If provided, only return versions that have ALL the given tags.

    Returns:
        List of dicts with keys: version, description, tags, template_vars.

    Example:
        get_prompts("cin")
        get_prompts("cin", tags=["ignore_noise"])
    """
    prompts_dir = Path(__file__).parent / "datasets" / doc / "prompts"

    if not prompts_dir.exists():
        raise FileNotFoundError(f"No prompts directory found for dataset '{doc}'.")

    results = []
    for prompt_file in sorted(prompts_dir.glob("v*.yaml")):
        data = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))
        file_tags: list[str] = data.get("tags", [])

        if tags and not all(t in file_tags for t in tags):
            continue

        results.append(
            {
                "version": prompt_file.stem,
                "description": data.get("description", ""),
                "tags": file_tags,
                "template_vars": data.get("template_vars", []),
            }
        )

    return results


_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def load_file_for_api(file_path: Path) -> dict:
    """Base64-encode a file and return an Anthropic API content block.

    Returns a ``document`` block for PDFs and an ``image`` block for images.

    Args:
        file_path: Absolute path to the file.

    Example:
        block = load_file_for_api(Path("datasets/cin/ins/cin1.png"))
    """
    suffix = file_path.suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {list(_MEDIA_TYPES)}")

    data = base64.standard_b64encode(file_path.read_bytes()).decode("utf-8")
    block_type = "document" if suffix == ".pdf" else "image"
    return {
        "type": block_type,
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def load_dataset(doc: str) -> pd.DataFrame:
    """Load a dataset CSV as a DataFrame.

    'in' and 'out' columns are resolved to absolute paths.

    Args:
        doc: Doc name (e.g. "cin").

    Example:
        load_dataset("cin")
    """
    dataset_dir = Path(__file__).parent / "datasets" / doc
    df = pd.read_csv(dataset_dir / f"{doc}.csv")
    df["in"] = df["in"].apply(lambda p: str(dataset_dir / p))
    df["out"] = df["out"].apply(lambda p: str(dataset_dir / p))
    return df
