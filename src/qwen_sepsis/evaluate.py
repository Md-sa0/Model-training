from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .train import SYSTEM


def score(model, tokenizer, prompts: list[str], batch_size: int = 4) -> np.ndarray:
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in ["0", "1"]]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError(f"Rótulos não são tokens únicos: {label_ids}")
    scores = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        texts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) for prompt in prompts[start:start + batch_size]]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.inference_mode(): logits = model(**batch).logits
        last = batch.attention_mask.sum(dim=1) - 1
        next_logits = logits[torch.arange(len(texts), device=model.device), last]
        selected = next_logits[:, [label_ids[0][0], label_ids[1][0]]]
        scores.extend(torch.softmax(selected.float(), dim=-1)[:, 1].cpu().tolist())
    return np.asarray(scores)


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
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quantization, device_map={"": 0}, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    datasets = {split: load_dataset("json", data_files=str(args.data_dir / f"{split}.jsonl"), split="train") for split in ["validation", "test"]}
    probabilities = {split: score(model, tokenizer, data["prompt"]) for split, data in datasets.items()}
    threshold = best_threshold(np.asarray(datasets["validation"]["label"], dtype=int), probabilities["validation"])
    result = {split: metrics(np.asarray(data["label"], dtype=int), probabilities[split], threshold) for split, data in datasets.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
