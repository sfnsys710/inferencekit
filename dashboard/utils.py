import json
from pathlib import Path

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).parent.parent / "evals" / "experiments"

DISPLAY_COLS = [
    "id",
    "doc",
    "model",
    "prompt_version",
    "tags",
    "notes",
    "timestamp",
    "execution_time",
    "correct_fields",
    "format_compliance",
]

DETAIL_COLS = [
    "correct_fields_reason",
    "format_compliance_reason",
    "expected",
    "generated",
]


def dropdown_options(series) -> list[dict]:
    vals = sorted(series.dropna().unique().tolist())
    return [{"label": str(v), "value": str(v)} for v in vals]


def tag_options(df) -> list[dict]:
    tags: set[str] = set()
    if "tags" in df.columns:
        for val in df["tags"].dropna():
            for t in str(val).split(","):
                t = t.strip()
                if t:
                    tags.add(t)
    return [{"label": t, "value": t} for t in sorted(tags)]


def pretty_json(value: str) -> str:
    try:
        return json.dumps(json.loads(value), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return str(value) if value else ""


def load_experiments() -> pd.DataFrame:
    dfs = []
    for csv_file in sorted(EXPERIMENTS_DIR.glob("*.csv")):
        df = pd.read_csv(csv_file)
        # Extract prompt version from filename: <doc>_prompt_<version>_<model>.csv
        parts = csv_file.stem.split("_prompt_")
        if len(parts) == 2:
            version_model = parts[1].split("_", 1)
            df["prompt_version"] = version_model[0] if version_model else ""
        else:
            df["prompt_version"] = ""
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=DISPLAY_COLS + DETAIL_COLS)
    combined = pd.concat(dfs, ignore_index=True)
    all_cols = DISPLAY_COLS + [c for c in DETAIL_COLS if c in combined.columns]
    available = [c for c in all_cols if c in combined.columns]
    return combined[available].sort_values("timestamp", ascending=False)
