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
   "'transformers==4.51.3' 'peft==0.14.0' 'accelerate==1.2.1' "
   "'rouge-score==0.1.2' 'huggingface_hub>=0.26'")

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer


CONFIG = {
    "run_id": "v2_qwen3_1.7b_r32_fp16",
    "max_eval": 500,
    "gen_max_new_tokens": 1536,
    "batch_size": 4,
    "seed": 42,
    "holdout_size": 500,
}
print("CONFIG:", json.dumps(CONFIG, indent=2))


cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")


WORK = Path("/kaggle/working")
RUN_DIR = WORK / "run"
RUN_DIR.mkdir(parents=True, exist_ok=True)


SRC_WEIGHTS = None
for cand in Path("/kaggle/input").rglob("merged_weights"):
    if cand.is_dir() and (cand / "config.json").exists():
        SRC_WEIGHTS = cand
        break
if SRC_WEIGHTS is None:
    for cand in Path("/kaggle/input").rglob("config.json"):
        if (cand.parent / "tokenizer.json").exists() and any(cand.parent.glob("*.safetensors")):
            SRC_WEIGHTS = cand.parent
            break
assert SRC_WEIGHTS is not None, "merged weights not found in /kaggle/input"
print("found weights at:", SRC_WEIGHTS)


DATA_PARQUET = None
for p in Path("/kaggle/input").rglob("*.parquet"):
    DATA_PARQUET = p
    break
assert DATA_PARQUET is not None, "no parquet attached"
print("dataset:", DATA_PARQUET)


tokenizer = AutoTokenizer.from_pretrained(SRC_WEIGHTS, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


df = pd.read_parquet(DATA_PARQUET, engine="pyarrow").dropna(subset=["query", "answer"]).reset_index(drop=True)
df = df.sample(frac=1.0, random_state=CONFIG["seed"]).reset_index(drop=True)
holdout = df.iloc[:CONFIG["holdout_size"]].reset_index(drop=True)
print(f"holdout rows: {len(holdout)}")


print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    SRC_WEIGHTS, torch_dtype=torch.float16, device_map="cuda",
    attn_implementation="sdpa",
)
model.eval()
model.config.use_cache = True


prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": str(q)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    for q in holdout["query"].tolist()
]
refs = [str(a) for a in holdout["answer"].tolist()]


preds = []
B = CONFIG["batch_size"]
t0 = time.time()
with torch.inference_mode():
    for i in range(0, len(prompts), B):
        batch = prompts[i:i+B]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        out = model.generate(
            **enc,
            max_new_tokens=CONFIG["gen_max_new_tokens"],
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds.extend([d.strip() for d in decoded])
        if (i // B) % 10 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            eta = elapsed / max(done, 1) * (len(prompts) - done)
            print(f"  {done}/{len(prompts)} elapsed={elapsed:.0f}s eta={eta:.0f}s")
infer_seconds = time.time() - t0


scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
rouge1_f, rougel_f = [], []
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
    "pred_len_chars_mean": sum(len(p) for p in preds) / len(preds),
    "ref_len_chars_mean": sum(len(r) for r in refs) / len(refs),
}
print("EVAL:", json.dumps(eval_metrics, indent=2, ensure_ascii=False))

with open(RUN_DIR / "eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

with open(RUN_DIR / "predictions.jsonl", "w", encoding="utf-8") as f:
    for q, ref, pred, r1, rl in zip(holdout["query"].tolist(), refs, preds, rouge1_f, rougel_f):
        f.write(json.dumps({
            "query": str(q), "reference": ref, "prediction": pred,
            "rouge1_f": r1, "rougeL_f": rl,
        }, ensure_ascii=False) + "\n")

print("packing merged weights for download...")
DST = WORK / "merged_weights_out"
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC_WEIGHTS, DST)
ARCHIVE = WORK / f"{CONFIG['run_id']}_merged.tar"
sh(f"tar -cf {ARCHIVE} -C {DST.parent} {DST.name}")
print("archive:", ARCHIVE, ARCHIVE.stat().st_size, "bytes")
shutil.rmtree(DST)
print("done")
