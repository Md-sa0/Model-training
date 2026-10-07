"""Conversão dos logits dos tokens 0 e 1 em probabilidade bruta."""

from __future__ import annotations

import math

import numpy as np


def probability_from_class_logits(logit_0: float, logit_1: float) -> dict[str, float]:
    """P(classe=1) a partir dos logits dos tokens de 0 e de 1."""
    first = float(logit_0)
    second = float(logit_1)
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("Logit não finito")
    logits = np.array([first, second], dtype=np.float64)
    logits -= np.max(logits)
    expanded = np.exp(logits)
    probability = float(expanded[1] / expanded.sum())
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probabilidade fora de [0, 1]")
    return {"logit_0": first, "logit_1": second, "raw_probability": probability}


def resolve_binary_token_ids(tokenizer) -> tuple[int, int, dict[str, list[int]]]:
    """Confere como o tokenizer representa 0 e 1, sem supor um único id."""
    sequences: dict[str, list[int]] = {}
    for label in ("0", "1"):
        encoded = list(tokenizer.encode(label, add_special_tokens=False))
        if not encoded:
            raise ValueError(f"O tokenizer não produziu token para {label!r}.")
        sequences[label] = [int(token) for token in encoded]
    if sequences["0"][0] == sequences["1"][0]:
        raise ValueError(
            "Os primeiros tokens de '0' e '1' coincidem. "
            f"Sequências: {sequences}."
        )
    return sequences["0"][0], sequences["1"][0], sequences
