import json
import os
import pickle

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN = 4096

def render_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    llm = LLM(
        model=MODEL_DIR,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        tokenizer_mode="auto",
        seed=0,
    )

    prompts = [render_prompt(tokenizer, row["question"]) for row in rows]

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        top_k=-1,
    )

    outputs = llm.generate(prompts, sampling_params=sampling)

    result = [
        {"rid": row["rid"], "answer": out.outputs[0].text.strip()}
        for row, out in zip(rows, outputs)
    ]

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
