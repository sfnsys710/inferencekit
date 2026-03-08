import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def summarize_reasons(reasons: list[str], metric_name: str) -> str:
    """Summarize the most common reasons/mistakes for a metric using an LLM."""
    model = os.environ.get("EVAL_SUMMARIZER_LLM", "claude-haiku-4-5-20251001")
    reasons_text = "\n".join(f"- {r}" for r in reasons if r and str(r).strip())

    if not reasons_text:
        return "No reasons available to summarize."

    prompt = (
        f"You are analyzing evaluation results for an OCR/LLM document extraction system.\n"
        f"The following are judge LLM explanations for the metric '{metric_name}'"
        f" across multiple test cases.\n"
        f"Summarize the most common mistakes or patterns concisely and actionably.\n\n"
        f"Reasons:\n{reasons_text}\n\nSummary:"
    )

    message = _get_client().messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in message.content if b.type == "text"), None)
    text = text_block.text if text_block else ""
    # Demote all headings by one level so # → ## etc. (avoids oversized titles in the box)
    return re.sub(r"^(#{1,5})", r"#\1", text, flags=re.MULTILINE)
