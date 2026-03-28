from __future__ import annotations

import argparse
from pathlib import Path

from src.structured_json_ft.dataset import build_dataset, split_dataset, write_dataset_card, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the synthetic JSON extraction dataset.")
    parser.add_argument("--num-examples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_dataset(num_examples=args.num_examples, seed=args.seed)
    splits = split_dataset(examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_examples in splits.items():
        write_jsonl(split_examples, args.output_dir / f"{split_name}.jsonl")
    write_dataset_card(args.output_dir, num_examples=args.num_examples, seed=args.seed)

    print(
        f"Wrote dataset to {args.output_dir} "
        f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})"
    )


if __name__ == "__main__":
    main()
