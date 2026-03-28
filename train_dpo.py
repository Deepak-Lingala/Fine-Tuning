from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO tuning on top of the SFT adapter.")
    parser.add_argument("--sft-checkpoint", default="outputs/sft-qwen2.5-3b-json")
    parser.add_argument("--preference-file", default="data/dpo_pairs/train.jsonl")
    parser.add_argument("--val-file", default="data/dpo_pairs/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/dpo-qwen2.5-3b-json")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.5)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files={"train": args.preference_file, "validation": args.val_file},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        is_trainable=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "logs"),
        max_length=args.max_length,
        max_prompt_length=768,
        seed=args.seed,
    )

    dpo_signature = inspect.signature(DPOTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
    }

    if "tokenizer" in dpo_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in dpo_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    metrics["eval_samples"] = len(dataset["validation"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
