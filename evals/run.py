import asyncio
import os
import time
from datetime import datetime
from pathlib import Path

import fire
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from ragas import Dataset, experiment
from ragas.backends.local_csv import LocalCSVBackend
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric, NumericMetric
from utils import get_prompt, load_dataset, load_file_for_api

load_dotenv()
client = AsyncAnthropic()
judge_llm = llm_factory(os.environ["JUDGE_LLM"], provider="anthropic", client=Anthropic())
judge_llm.model_args.pop("top_p", None)  # Claude 4.x rejects temperature+top_p together

format_compliance = DiscreteMetric(
    name="format_compliance",
    prompt=(
        "Check if the format of this generated reponse: {generated} is same as the expected"
        " format {expected}. Return 'correct' or 'incorrect'."
    ),
    allowed_values=["correct", "incorrect"],
)


correct_fields = NumericMetric(
    name="correct_fields",
    prompt=(
        "We're expected to perform OCR to extract multi-field information from an iage,"
        " How many of the following generated feilds are correct compared to the expected?"
        " \ngenerated: \n{generated} \nexpected: \n{expected}"
    ),
    allowed_values=(0, 8),
)


@experiment()
async def ocr_experiment(row, doc, model, prompt):
    file_block = load_file_for_api(Path(row["in"]))
    t0 = time.perf_counter()
    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    file_block,
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        model=model,
    )
    execution_time = round(time.perf_counter() - t0, 3)
    text_block = next((b for b in message.content if b.type == "text"), None)

    format_score = format_compliance.score(
        llm=judge_llm,
        generated=text_block.text if text_block else "",
        expected=Path(row["out"]).read_text(encoding="utf-8"),
    )

    correct_fields_score = correct_fields.score(
        llm=judge_llm,
        generated=text_block.text if text_block else "",
        expected=Path(row["out"]).read_text(encoding="utf-8"),
    )

    return {
        "id": row["id"],
        "doc": doc,
        "model": model,
        "tags": row["tags"],
        "notes": row["notes"],
        "timestamp": datetime.now().isoformat(),
        "execution_time": execution_time,
        "correct_fields": correct_fields_score.value,
        "correct_fields_reason": correct_fields_score.reason,
        "format_compliance": format_score.value,
        "format_compliance_reason": format_score.reason,
        "expected": Path(row["out"]).read_text(encoding="utf-8"),
        "generated": text_block.text if text_block else "",
    }


async def _main(model: str, doc: str, prompt_version: str):
    backend = LocalCSVBackend(root_dir=str(Path(__file__).parent))
    prompt = get_prompt(doc=doc, version=prompt_version)
    df = load_dataset(doc)
    dataset = Dataset.from_pandas(df, name=f"{doc}_dataset", backend=backend)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    await ocr_experiment.arun(
        dataset,
        name=f"{doc}_prompt_{prompt_version}_{model}_{timestamp}",
        backend=backend,
        doc=doc,
        model=model,
        prompt=prompt,
    )


def main(model: str, doc: str, prompt_version: str):
    asyncio.run(_main(model, doc, prompt_version))


if __name__ == "__main__":
    fire.Fire(main)
