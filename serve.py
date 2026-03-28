from __future__ import annotations

import os
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.structured_json_ft.metrics import extract_json_block
from src.structured_json_ft.prompts import SYSTEM_PROMPT, build_user_prompt

app = FastAPI(title="Structured JSON Extraction Fine-Tuning", version="1.0.0")


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Raw medical note or other messy text.")
    schema_hint: str = Field(default="medical_intake_v1")


class ExtractResponse(BaseModel):
    structured_json: dict[str, Any] | None
    confidence: float
    raw_output: str


def load_runtime():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH")
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


MODEL, TOKENIZER = load_runtime()


def infer(text: str, schema_hint: str) -> tuple[str, dict[str, Any] | None]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text, schema_hint)},
    ]
    prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        output = MODEL.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=TOKENIZER.pad_token_id,
        )
    raw_output = TOKENIZER.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return raw_output, extract_json_block(raw_output)


def score_confidence(parsed: dict[str, Any] | None) -> float:
    if parsed is None:
        return 0.0
    if parsed.get("can_extract") is False:
        return 0.92
    record = parsed.get("record") or {}
    expected_fields = 9
    populated = sum(1 for value in record.values() if value not in (None, "", [], {}))
    return round(min(0.99, 0.45 + (populated / expected_fields) * 0.5), 2)


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    raw_output, parsed = infer(request.text, request.schema_hint)
    return ExtractResponse(structured_json=parsed, confidence=score_confidence(parsed), raw_output=raw_output)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
