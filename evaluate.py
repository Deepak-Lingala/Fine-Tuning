from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.structured_json_ft.metrics import canonical_json, extract_json_block, field_level_f1, refusal_correct
from src.structured_json_ft.prompts import SYSTEM_PROMPT, build_user_prompt
from src.structured_json_ft.serialization import make_json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline, SFT, or DPO checkpoints.")
    parser.add_argument("--dataset-file", default="data/processed/test.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-file", default="results/latest_eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_model(base_model: str, adapter_path: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_prediction(model, tokenizer, input_text: str, schema_hint: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(input_text, schema_hint)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()


def evaluate_rows(rows: list[dict[str, Any]], model, tokenizer, max_new_tokens: int) -> dict[str, Any]:
    total = len(rows)
    valid_count = 0
    exact_count = 0
    refusal_count = 0
    f1_sum = 0.0
    predictions: list[dict[str, Any]] = []

    for row in rows:
        raw_output = generate_prediction(model, tokenizer, row["input_text"], row["schema_hint"], max_new_tokens)
        parsed = extract_json_block(raw_output)
        target = make_json_safe(row["target_json"])
        valid_count += int(parsed is not None)
        exact_count += int(canonical_json(parsed) == canonical_json(target))
        refusal_count += int(refusal_correct(parsed, target))
        f1_sum += field_level_f1(parsed, target)
        predictions.append(
            {
                "id": row["id"],
                "raw_output": raw_output,
                "parsed_output": parsed,
                "target_json": target,
            }
        )

    return {
        "samples": total,
        "json_validity_rate": round(valid_count / total * 100, 2),
        "exact_match_accuracy": round(exact_count / total * 100, 2),
        "field_level_f1": round(f1_sum / total * 100, 2),
        "refusal_correctness": round(refusal_count / total * 100, 2),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    rows = list(load_dataset("json", data_files=args.dataset_file)["train"])
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    metrics = evaluate_rows(rows, model, tokenizer, args.max_new_tokens)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
