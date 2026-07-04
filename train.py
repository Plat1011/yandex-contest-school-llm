import argparse
import os
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

def build_dataset(parquet_path: str, tokenizer, max_len: int) -> Dataset:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = df.dropna(subset=["query", "answer"]).reset_index(drop=True)

    def to_text(row):
        messages = [
            {"role": "user", "content": str(row["query"])},
            {"role": "assistant", "content": str(row["answer"])},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    ds = Dataset.from_pandas(df[["query", "answer"]])
    ds = ds.map(to_text, remove_columns=["query", "answer"])

    def length_ok(ex):
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        return len(ids) <= max_len

    ds = ds.filter(length_ok, num_proc=2)
    return ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./weights")
    p.add_argument("--data", default="./dataset_ml_challenge.parquet")
    p.add_argument("--output_dir", default="./lora_out")
    p.add_argument("--merged_dir", default="./merged_weights")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = build_dataset(args.data, tokenizer, args.max_len)
    print(f"train samples after length filter: {len(ds)}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=args.seed,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_len,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    print("merging LoRA into base...")
    del trainer, model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
    )
    merged = PeftModel.from_pretrained(base, args.output_dir)
    merged = merged.merge_and_unload()

    out = Path(args.merged_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    for fname in ("chat_template.jinja", "generation_config.json"):
        src = Path(args.model_dir) / fname
        if src.exists():
            (out / fname).write_bytes(src.read_bytes())

    print(f"merged weights saved to {out}")

if __name__ == "__main__":
    main()
