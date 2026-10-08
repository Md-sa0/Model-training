"""Inferência do Qwen3-8B + LoRA já treinado. Não gera texto e não altera pesos."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
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


def _last_relevant_positions(attention_mask, padding_side: str):
    if padding_side == "right":
        return attention_mask.sum(dim=1) - 1
    if padding_side == "left":
        return torch.full(
            (attention_mask.shape[0],), attention_mask.shape[1] - 1,
            dtype=torch.long, device=attention_mask.device,
        )
    raise ValueError(f"padding_side desconhecido: {padding_side}")


def _sequence_log_probability(model, prompt_ids, token_sequence: list[int], first_log_probs) -> float:
    score = float(first_log_probs[token_sequence[0]])
    prefix = list(token_sequence[:1])
    for token_id in token_sequence[1:]:
        continuation = torch.tensor(prefix, dtype=prompt_ids.dtype, device=prompt_ids.device).unsqueeze(0)
        input_ids = torch.cat((prompt_ids, continuation), dim=1)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].float()
        score += float(functional.log_softmax(logits, dim=-1)[token_id])
        prefix.append(token_id)
    return score


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
            output = model(**batch)
            logits = output.logits
        last = _last_relevant_positions(batch.attention_mask, tokenizer.padding_side)
        next_logits = logits[torch.arange(len(texts), device=logits.device), last]
        if all(len(token_ids) == 1 for token_ids in sequences.values()):
            selected = next_logits[:, [token_0, token_1]].float().cpu().numpy()
            for logit_0, logit_1 in selected:
                records.append(probability_from_class_logits(float(logit_0), float(logit_1)))
        else:
            for row_index, tokenized_row in enumerate(batch.input_ids):
                prompt_ids = tokenized_row[batch.attention_mask[row_index].bool()].unsqueeze(0)
                first_log_probs = functional.log_softmax(next_logits[row_index].float(), dim=-1)
                sequence_scores = [
                    _sequence_log_probability(model, prompt_ids, sequences[label], first_log_probs)
                    for label in ("0", "1")
                ]
                records.append(probability_from_class_logits(*sequence_scores))
        if log is not None:
            if start == 0:
                meta["logits_shape"] = [int(value) for value in logits.shape]
                meta["last_relevant_positions"] = [int(value) for value in last.tolist()]
                meta["padding_side"] = tokenizer.padding_side
                log(
                    f"Logits shape={tuple(logits.shape)}; posições relevantes={last.tolist()}; "
                    f"padding_side={tokenizer.padding_side}."
                )
            log(f"Scores: {min(start + batch_size, total)}/{total}")
        elif start == 0:
            meta["logits_shape"] = [int(value) for value in logits.shape]
            meta["last_relevant_positions"] = [int(value) for value in last.tolist()]
            meta["padding_side"] = tokenizer.padding_side
    return records, meta


def get_binary_probability(model, tokenizer, prompt: str, max_length: int = 512) -> dict[str, float]:
    records, _meta = score_prompts(model, tokenizer, [prompt], batch_size=1, max_length=max_length)
    return records[0]
