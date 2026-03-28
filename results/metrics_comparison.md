# Metrics Comparison

The table below is a target-results template for the three checkpoints this
repo evaluates. Replace all values with your actual `evaluate.py` outputs
after training.

| Model Stage | JSON Validity Rate | Exact Match Accuracy | Field-Level F1 | Refusal Correctness |
| --- | ---: | ---: | ---: | ---: |
| Baseline Zero-Shot Target | 45.00% | 22.00% | 48.00% | 61.00% |
| SFT QLoRA Target | 91.00% | 68.00% | 84.00% | 93.00% |
| DPO on SFT Target | 96.00% | 74.00% | 88.00% | 96.00% |

## How to refresh with actual results

1. Run `py .\evaluate.py --dataset-file .\data\processed\test.jsonl --output-file .\results\baseline_eval.json`
2. Run `py .\evaluate.py --dataset-file .\data\processed\test.jsonl --adapter-path .\outputs\sft-qwen2.5-3b-json --output-file .\results\sft_eval.json`
3. Run `py .\evaluate.py --dataset-file .\data\processed\test.jsonl --adapter-path .\outputs\dpo-qwen2.5-3b-json --output-file .\results\dpo_eval.json`
4. Copy the reported metrics into this table.
