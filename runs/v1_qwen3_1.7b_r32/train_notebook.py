import os
import json
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sh(cmd):
    print("$", cmd)
    subprocess.check_call(cmd, shell=True)


sh(f"{sys.executable} -m pip install -q --upgrade "
   "'transformers==4.46.3' 'peft==0.13.2' 'trl==0.12.2' "
   "'datasets==3.1.0' 'accelerate==1.1.1' 'bitsandbytes==0.44.1' "
   "'huggingface_hub>=0.26' 'rouge-score==0.1.2'")

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import snapshot_download
from rouge_score import rouge_scorer


CONFIG = {
    "run_id": os.environ.get("RUN_ID", "v1_qwen3_1.7b_r32"),
    "base_model": "Qwen/Qwen3-1.7B",
    "max_len": 2048,
    "epochs": 2.0,
    "lr": 2e-4,
    "batch": 2,
    "grad_accum": 8,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "seed": 42,
    "holdout_size": 500,
    "gen_max_new_tokens": 1536,
    "gen_temperature": 0.0,
}
print("CONFIG:", json.dumps(CONFIG, indent=2))


WORK = Path("/kaggle/working")
WEIGHTS = WORK / "base_weights"
RUN_DIR = WORK / "run"
RUN_DIR.mkdir(parents=True, exist_ok=True)

DATA_PARQUET = None
for p in Path("/kaggle/input").rglob("*.parquet"):
    DATA_PARQUET = p
    break
assert DATA_PARQUET is not None, "no parquet attached"
print("dataset:", DATA_PARQUET)


if not (WEIGHTS / "config.json").exists():
    snapshot_download(
        repo_id=CONFIG["base_model"],
        local_dir=str(WEIGHTS),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.jinja"],
    )
print("base weights ready at", WEIGHTS)


tokenizer = AutoTokenizer.from_pretrained(WEIGHTS, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

df = pd.read_parquet(DATA_PARQUET, engine="pyarrow").dropna(subset=["query", "answer"]).reset_index(drop=True)
df = df.sample(frac=1.0, random_state=CONFIG["seed"]).reset_index(drop=True)

holdout = df.iloc[:CONFIG["holdout_size"]].reset_index(drop=True)
train_df = df.iloc[CONFIG["holdout_size"]:].reset_index(drop=True)
print(f"train rows: {len(train_df)}  holdout rows: {len(holdout)}")


def to_text(row):
    messages = [
        {"role": "user", "content": str(row["query"])},
        {"role": "assistant", "content": str(row["answer"])},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )}

train_ds = Dataset.from_pandas(train_df[["query", "answer"]]).map(to_text, remove_columns=["query", "answer"])

def length_ok(ex):
    return len(tokenizer(ex["text"], add_special_tokens=False)["input_ids"]) <= CONFIG["max_len"]

train_ds = train_ds.filter(length_ok, num_proc=2)
print("train after length filter:", len(train_ds))


model = AutoModelForCausalLM.from_pretrained(
    WEIGHTS, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_cfg = LoraConfig(
    r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

sft_cfg = SFTConfig(
    output_dir=str(WORK / "lora_out"),
    num_train_epochs=CONFIG["epochs"],
    per_device_train_batch_size=CONFIG["batch"],
    gradient_accumulation_steps=CONFIG["grad_accum"],
    learning_rate=CONFIG["lr"],
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    seed=CONFIG["seed"],
    report_to="none",
    dataset_text_field="text",
    max_seq_length=CONFIG["max_len"],
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

t0 = time.time()
trainer = SFTTrainer(
    model=model, args=sft_cfg, train_dataset=train_ds,
    peft_config=lora_cfg, tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(str(WORK / "lora_out"))
train_seconds = time.time() - t0
print(f"training took {train_seconds:.1f}s")

del trainer, model
gc.collect()
torch.cuda.empty_cache()


print("merging LoRA into base...")
base = AutoModelForCausalLM.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, str(WORK / "lora_out")).merge_and_unload()

MERGED_DIR = RUN_DIR / "merged_weights"
MERGED_DIR.mkdir(exist_ok=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
for fname in ("chat_template.jinja", "generation_config.json"):
    src = WEIGHTS / fname
    if src.exists():
        shutil.copy(src, MERGED_DIR / fname)

del base, merged
gc.collect()
torch.cuda.empty_cache()


print("installing vllm for eval inference...")
sh(f"{sys.executable} -m pip install -q 'vllm==0.11.0'")

from vllm import LLM, SamplingParams
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

llm = LLM(
    model=str(MERGED_DIR),
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tokenizer_mode="auto",
    seed=CONFIG["seed"],
)

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": str(q)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    for q in holdout["query"].tolist()
]

sampling = SamplingParams(
    temperature=CONFIG["gen_temperature"],
    max_tokens=CONFIG["gen_max_new_tokens"],
    top_k=-1,
)

t0 = time.time()
outputs = llm.generate(prompts, sampling_params=sampling)
infer_seconds = time.time() - t0

preds = [o.outputs[0].text.strip() for o in outputs]
refs = [str(a) for a in holdout["answer"].tolist()]


scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
rouge1_f = []
rougel_f = []
for pred, ref in zip(preds, refs):
    s = scorer.score(ref, pred)
    rouge1_f.append(s["rouge1"].fmeasure)
    rougel_f.append(s["rougeL"].fmeasure)

eval_metrics = {
    "run_id": CONFIG["run_id"],
    "rouge1_f_mean": sum(rouge1_f) / len(rouge1_f),
    "rougeL_f_mean": sum(rougel_f) / len(rougel_f),
    "holdout_n": len(holdout),
    "infer_seconds_total": infer_seconds,
    "infer_seconds_per_sample": infer_seconds / len(holdout),
    "train_seconds": train_seconds,
    "pred_len_chars_mean": sum(len(p) for p in preds) / len(preds),
    "ref_len_chars_mean": sum(len(r) for r in refs) / len(refs),
}
print("EVAL:", json.dumps(eval_metrics, indent=2, ensure_ascii=False))

with open(RUN_DIR / "eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

with open(RUN_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=2)

with open(RUN_DIR / "predictions.jsonl", "w", encoding="utf-8") as f:
    for q, ref, pred, r1, rl in zip(holdout["query"].tolist(), refs, preds, rouge1_f, rougel_f):
        f.write(json.dumps({
            "query": str(q), "reference": ref, "prediction": pred,
            "rouge1_f": r1, "rougeL_f": rl,
        }, ensure_ascii=False) + "\n")


print("packing merged weights for download...")
ARCHIVE = WORK / f"{CONFIG['run_id']}_merged.tar"
sh(f"tar -cf {ARCHIVE} -C {MERGED_DIR.parent} {MERGED_DIR.name}")
print("archive:", ARCHIVE, ARCHIVE.stat().st_size, "bytes")

shutil.rmtree(MERGED_DIR)
shutil.rmtree(WEIGHTS, ignore_errors=True)
shutil.rmtree(WORK / "lora_out", ignore_errors=True)
print("done")
