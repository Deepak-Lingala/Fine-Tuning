from __future__ import annotations

import json
from typing import Any


def extract_json_block(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def canonical_json(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def flatten_json(data: Any, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_json(value, next_prefix))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            next_prefix = f"{prefix}[{idx}]"
            flat.update(flatten_json(value, next_prefix))
    else:
        flat[prefix] = json.dumps(data, sort_keys=True)
    return flat


def field_level_f1(prediction: dict[str, Any] | None, target: dict[str, Any]) -> float:
    if prediction is None:
        return 0.0
    pred_items = flatten_json(prediction)
    target_items = flatten_json(target)
    if not pred_items and not target_items:
        return 1.0
    if not pred_items or not target_items:
        return 0.0
    pred_set = set(pred_items.items())
    target_set = set(target_items.items())
    true_positive = len(pred_set & target_set)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(pred_set)
    recall = true_positive / len(target_set)
    return 2 * precision * recall / (precision + recall)


def refusal_correct(prediction: dict[str, Any] | None, target: dict[str, Any]) -> bool:
    if prediction is None:
        return False
    return prediction.get("can_extract") == target.get("can_extract")
