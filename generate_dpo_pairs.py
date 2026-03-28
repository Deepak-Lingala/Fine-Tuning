from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.structured_json_ft.prompts import SYSTEM_PROMPT, build_user_prompt
from src.structured_json_ft.serialization import make_json_safe

LABELER_PROMPT = """You are ranking structured JSON extraction outputs.
Choose the best and worst answer for the given target.

Prefer answers that:
1. Are valid JSON.
2. Match the schema exactly.
3. Match the target record accurately.
4. Refuse only when the target should refuse.

Return JSON:
{"chosen_index": int, "rejected_index": int, "reason": "..."}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DPO preference pairs using GPT-4o-mini.")
    parser.add_argument("--candidates-file", default=None, help="Optional existing JSON file with generated candidates.")
    parser.add_argument("--dataset-file", default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", default="data/dpo_pairs")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default="outputs/sft-qwen2.5-3b-json")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def build_prompt(row: dict[str, Any], candidates: list[str]) -> str:
    payload = {
        "instruction": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(row["input_text"], row["schema_hint"]),
        "target_json": make_json_safe(row["target_json"]),
        "candidates": candidates,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def parse_labeler_output(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Labeler did not return JSON.")
    return json.loads(text[start : end + 1])


def load_generation_model(base_model: str, adapter_path: str | None):
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


def sample_candidates(model, tokenizer, row: dict[str, Any], count: int, max_new_tokens: int) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row["input_text"], row["schema_hint"])},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    candidates: list[str] = []
    with torch.no_grad():
        for _ in range(count):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            candidates.append(text)
    return candidates


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required to generate DPO pairs.")

    client = OpenAI(api_key=api_key)
    rows = list(load_dataset("json", data_files=args.dataset_file)["train"])
    if args.candidates_file:
        candidates_payload = json.loads(Path(args.candidates_file).read_text(encoding="utf-8"))
    else:
        model, tokenizer = load_generation_model(args.base_model, args.adapter_path)
        candidates_payload = {
            row["id"]: sample_candidates(model, tokenizer, row, args.num_candidates, args.max_new_tokens)
            for row in rows
        }

    pairs = []
    for row in rows:
        candidate_texts = candidates_payload[row["id"]]
        response = client.responses.create(
            model=args.model,
            input=[
                {"role": "system", "content": LABELER_PROMPT},
                {"role": "user", "content": build_prompt(row, candidate_texts)},
            ],
        )
        parsed = parse_labeler_output(response.output_text)
        chosen = candidate_texts[parsed["chosen_index"]]
        rejected = candidate_texts[parsed["rejected_index"]]
        pairs.append(
            {
                "id": row["id"],
                "prompt": build_user_prompt(row["input_text"], row["schema_hint"]),
                "chosen": chosen,
                "rejected": rejected,
                "label_reason": parsed["reason"],
            }
        )

    split_index = int(len(pairs) * args.train_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(candidates_payload, handle, indent=2)
    for split_name, split_rows in {"train": pairs[:split_index], "val": pairs[split_index:]}.items():
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(pairs)} preference pairs to {output_dir}")


if __name__ == "__main__":
    main()
