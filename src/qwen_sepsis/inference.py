"""Inferência do Qwen3-8B + LoRA já treinado. Não gera texto e não altera pesos."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .scoring import probability_from_class_logits, resolve_binary_token_ids
from .train import SYSTEM


def load_sepsis_model(base_model: str, adapter: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def score_prompts(model, tokenizer, prompts, batch_size: int = 4, max_length: int = 512, log=None):
    prompts = list(prompts)
    token_0, token_1, sequences = resolve_binary_token_ids(tokenizer)
    meta = {
        "token_id_0": token_0,
        "token_id_1": token_1,
        "token_ids_0": sequences["0"],
        "token_ids_1": sequences["1"],
        "single_token_labels": all(len(token_ids) == 1 for token_ids in sequences.values()),
    }
    if not meta["single_token_labels"] and log is not None:
        log(
            "Os rótulos 0 e 1 não são tokens únicos; a probabilidade usa o primeiro token, "
            "que é o token supervisionado no treino."
        )
    if not prompts:
        return [], meta
    records = []
    model.eval()
    total = len(prompts)
    for start in range(0, total, batch_size):
        chunk = prompts[start:start + batch_size]
        texts = [render_prompt(tokenizer, prompt) for prompt in chunk]
        batch = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**batch).logits
        last = batch.attention_mask.sum(dim=1) - 1
        next_logits = logits[torch.arange(len(texts), device=logits.device), last]
        selected = next_logits[:, [token_0, token_1]].float().cpu().numpy()
        for logit_0, logit_1 in selected:
            records.append(probability_from_class_logits(float(logit_0), float(logit_1)))
        if log is not None:
            log(f"Scores: {min(start + batch_size, total)}/{total}")
    return records, meta


def get_binary_probability(model, tokenizer, prompt: str, max_length: int = 512) -> dict[str, float]:
    records, _meta = score_prompts(model, tokenizer, [prompt], batch_size=1, max_length=max_length)
    return records[0]
