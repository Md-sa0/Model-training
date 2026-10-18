"""Validation and median imputation shared by the CLI and REST service."""

from __future__ import annotations

import math

import pandas as pd

from .data import DYNAMIC, FEATURES


def prepare_observation(
    values: dict[str, float | int | None], medians: pd.Series
) -> tuple[pd.Series, list[str]]:
    unknown = set(values).difference(FEATURES)
    if unknown:
        raise ValueError(f"Campos não reconhecidos: {sorted(unknown)}")
    row: dict[str, float | int] = {}
    imputed_features: list[str] = []
    for feature in FEATURES:
        value = values.get(feature)
        if value is None:
            value = float(medians[feature])
            if feature in DYNAMIC:
                row[f"{feature}_missing"] = 1
                imputed_features.append(feature)
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Valor inválido para {feature}")
            if feature in DYNAMIC:
                row[f"{feature}_missing"] = 0
        row[feature] = value
    return pd.Series(row), imputed_features
