from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .inference import load_sepsis_model, score_prompts


def score(model, tokenizer, prompts: list[str], batch_size: int = 4) -> np.ndarray:
    records, _tokens = score_prompts(model, tokenizer, prompts, batch_size=batch_size, max_length=512)
    return np.asarray([record["raw_probability"] for record in records], dtype=float)


def best_threshold(y, probability):
    candidates = np.linspace(0.01, 0.99, 99)
    values = [f1_score(y, probability >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    return {"threshold": threshold, "average_precision": average_precision_score(y, probability),
            "auroc": roc_auc_score(y, probability), "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0),
            "brier": brier_score_loss(y, probability), "confusion_matrix": confusion_matrix(y, predicted).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args()
    model, tokenizer = load_sepsis_model(args.base_model, args.adapter)
    datasets = {split: load_dataset("json", data_files=str(args.data_dir / f"{split}.jsonl"), split="train") for split in ["validation", "test"]}
    probabilities = {split: score(model, tokenizer, data["prompt"]) for split, data in datasets.items()}
    threshold = best_threshold(np.asarray(datasets["validation"]["label"], dtype=int), probabilities["validation"])
    result = {split: metrics(np.asarray(data["label"], dtype=int), probabilities[split], threshold) for split, data in datasets.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
