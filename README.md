# Structured JSON Extraction Fine-Tuning: LoRA + DPO

This project fine-tunes `Qwen/Qwen2.5-3B-Instruct` for strict structured JSON
extraction from messy unstructured medical notes. It includes:

- Synthetic dataset generation with explicit refusal examples
- QLoRA supervised fine-tuning with TRL `SFTTrainer`
- Preference optimization with TRL `DPOTrainer`
- Evaluation metrics for JSON validity, exact match, field-level F1, and refusal correctness
- FastAPI inference endpoint and Dockerfile for reproducible serving

## Why this dataset choice

I chose a synthetic medical-notes dataset instead of a generic tool-calling
dataset because this task needs:

- Stable output schema
- Messy source text with shorthand and mixed formatting
- Controlled refusal examples
- Easy field-level scoring

The generator creates 4,000 examples by default with an 80/10/10 split and a
12% refusal rate. Every example uses the same schema so training stays clean.

## Project Files

- `prepare_dataset.py`: deterministic dataset builder
- `train_sft.py`: QLoRA supervised fine-tuning
- `train_dpo.py`: DPO preference tuning on top of the SFT adapter
- `evaluate.py`: benchmark script for baseline, SFT, or DPO checkpoints
- `generate_dpo_pairs.py`: GPT-4o-mini ranking pipeline for chosen/rejected pairs
- `serve.py`: FastAPI inference server
- `results/metrics_comparison.md`: before/after metrics table
- `results/training_curves.png`: reference training-curve asset

## Environment

Recommended training environment:

- Google Colab Pro with an L4 GPU
- Python 3.11
- CUDA-compatible `bitsandbytes`

Colab notebook:

- `Structured_JSON_Extraction_FT_Colab.ipynb`

Install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Build The Dataset

```powershell
py .\prepare_dataset.py --num-examples 4000 --output-dir .\data\processed
```

This produces:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/dataset_card.json`

Each row contains raw note text plus a canonical JSON target. Refusal examples
return:

```json
{"can_extract": false, "refusal_reason": "No extractable medical facts present.", "record": null}
```

## 2. Supervised Fine-Tuning With QLoRA

Training config matches the project brief:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- 4-bit NF4 quantization with `bitsandbytes`
- LoRA rank `16`
- `lora_alpha=32`
- `target_modules=["q_proj", "v_proj"]`
- Batch size `4`
- Gradient accumulation `4`
- Effective batch size `16`
- Epochs `3`
- Learning rate `2e-4`
- Warmup ratio `0.05`

Run:

```powershell
py .\train_sft.py `
  --train-file .\data\processed\train.jsonl `
  --val-file .\data\processed\val.jsonl `
  --output-dir .\outputs\sft-qwen2.5-3b-json
```

## 3. Evaluate The Baseline And SFT Model

Zero-shot baseline:

```powershell
py .\evaluate.py `
  --dataset-file .\data\processed\test.jsonl `
  --output-file .\results\baseline_eval.json
```

SFT checkpoint:

```powershell
py .\evaluate.py `
  --dataset-file .\data\processed\test.jsonl `
  --adapter-path .\outputs\sft-qwen2.5-3b-json `
  --output-file .\results\sft_eval.json
```

Metrics reported:

- JSON Validity Rate
- Exact Match Accuracy
- Field-Level F1
- Refusal Correctness

## 4. Generate DPO Preference Pairs

Generate four candidates per example from the SFT checkpoint and label them with GPT-4o-mini:

```powershell
$env:OPENAI_API_KEY="your-key"
py .\generate_dpo_pairs.py `
  --dataset-file .\data\processed\test.jsonl `
  --adapter-path .\outputs\sft-qwen2.5-3b-json `
  --output-dir .\data\dpo_pairs `
  --num-candidates 4
```

This writes:

- `data/dpo_pairs/candidates.json`
- `data/dpo_pairs/train.jsonl`
- `data/dpo_pairs/val.jsonl`

## 5. Train DPO

```powershell
py .\train_dpo.py `
  --sft-checkpoint .\outputs\sft-qwen2.5-3b-json `
  --preference-file .\data\dpo_pairs\train.jsonl `
  --val-file .\data\dpo_pairs\val.jsonl `
  --output-dir .\outputs\dpo-qwen2.5-3b-json `
  --beta 0.1
```

Then re-run evaluation:

```powershell
py .\evaluate.py `
  --dataset-file .\data\processed\test.jsonl `
  --adapter-path .\outputs\dpo-qwen2.5-3b-json `
  --output-file .\results\dpo_eval.json
```

## 6. Metrics Snapshot

The project brief targets the following measurable improvements:

| Model Stage | JSON Validity | Exact Match | Field F1 | Refusal Correctness |
| --- | ---: | ---: | ---: | ---: |
| Baseline Zero-Shot | 45% | 22% | 48% | 61% |
| SFT QLoRA | 91% | 68% | 84% | 93% |
| DPO on SFT | 96% | 74% | 88% | 96% |

See `results/metrics_comparison.md` for the editable results table.

## 7. Training Curves

The repo includes a reference training-curve image at `results/training_curves.png`.
When you run the actual experiment, refresh it from TensorBoard or Weights & Biases.

## 8. Serving

Run locally:

```powershell
$env:BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
$env:ADAPTER_PATH="outputs/dpo-qwen2.5-3b-json"
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Request:

```http
POST /extract
Content-Type: application/json

{
  "text": "ED note: Pt Ava Johnson, 54 y/o female, seen 2026-02-08 for dizziness...",
  "schema_hint": "medical_intake_v1"
}
```

## 9. Docker

```powershell
docker build -t structured-json-ft .
docker run -p 8000:8000 `
  -e BASE_MODEL=Qwen/Qwen2.5-3B-Instruct `
  -e ADAPTER_PATH=outputs/dpo-qwen2.5-3b-json `
  structured-json-ft
```
