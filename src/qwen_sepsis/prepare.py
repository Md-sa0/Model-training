import argparse
import json
from pathlib import Path

from .data import download_dataset, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/sepsis_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-train", type=int, default=40000)
    parser.add_argument("--max-eval", type=int, default=20000)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    provenance = download_dataset(args.raw)
    metadata = prepare_dataset(args.raw, args.output, args.seed, args.negative_ratio, args.max_train, args.max_eval)
    print(json.dumps({"provenance": provenance, "dataset": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

