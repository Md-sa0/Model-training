"""Risco operacional e anormalidades observadas, sem texto gerado pelo modelo."""

from __future__ import annotations

import pandas as pd

from .data import hour_value

DISCLAIMER = (
    "Sistema experimental de apoio à decisão clínica para previsão precoce de risco de sepse. "
    "Não substitui o diagnóstico médico."
)
RISK_LEVEL_NOTE = "Limites operacionais do sistema experimental, não limites clínicos universais."
RISK_SCORE_MEANING = (
    "Probabilidade calibrada na validação. Não é diagnóstico nem chance clínica validada."
)
RANGE_NOTE = (
    "Faixas usadas somente para descrever valores observados. Não são critérios diagnósticos universais."
)

# Cortes operacionais para o texto determinístico. Não são limites clínicos universais.
OPERATIONAL_RANGES = {
    "HR": {"low": 60.0, "high": 100.0, "low_text": "Frequência cardíaca reduzida", "high_text": "Frequência cardíaca elevada"},
    "O2Sat": {"low": 94.0, "low_text": "Saturação de oxigênio reduzida"},
    "Temp": {"low": 36.0, "high": 38.0, "low_text": "Temperatura abaixo da faixa operacional", "high_text": "Temperatura acima da faixa operacional"},
    "SBP": {"low": 90.0, "low_text": "Pressão arterial sistólica reduzida"},
    "MAP": {"low": 65.0, "low_text": "PAM reduzida"},
    "DBP": {"low": 60.0, "low_text": "Pressão arterial diastólica reduzida"},
    "Resp": {"high": 20.0, "high_text": "Frequência respiratória elevada"},
    "WBC": {"low": 4.0, "high": 12.0, "low_text": "Leucócitos abaixo da faixa operacional", "high_text": "Leucócitos acima da faixa operacional"},
    "Lactate": {"high": 2.0, "high_text": "Lactato acima da faixa de referência operacional"},
    "Creatinine": {"high": 1.2, "high_text": "Creatinina acima da faixa operacional"},
    "Platelets": {"low": 150.0, "low_text": "Plaquetas abaixo da faixa operacional"},
}


def _lookup(row, key):
    if isinstance(row, pd.Series):
        return row[key] if key in row.index else None
    return row.get(key)


def observed_abnormalities(row) -> list[str]:
    findings = []
    for name, rule in OPERATIONAL_RANGES.items():
        missing = _lookup(row, f"{name}_missing")
        if missing is not None and not pd.isna(missing) and int(missing) == 1:
            continue
        value = _lookup(row, name)
        if value is None or pd.isna(value):
            continue
        number = float(value)
        if "high" in rule and number > rule["high"]:
            findings.append(rule["high_text"])
        if "low" in rule and number < rule["low"]:
            findings.append(rule["low_text"])
    return findings


def abnormalities_until(frame: pd.DataFrame, current_hour) -> list[str]:
    """Descreve somente a última linha disponível até a hora pedida."""
    if "Hour" not in frame.columns:
        return []
    work = frame.copy()
    work["Hour"] = work["Hour"].map(hour_value)
    work = work.loc[work["Hour"] <= hour_value(current_hour)]
    if work.empty:
        return []
    current = work.sort_values("Hour", kind="stable").iloc[-1]
    return observed_abnormalities(current)


def classify_risk(probability: float, low_below: float, high_at_or_above: float) -> str:
    if high_at_or_above < low_below:
        raise ValueError("O limite alto não pode ser menor que o limiar operacional.")
    if probability < low_below:
        return "LOW"
    if probability < high_at_or_above:
        return "MODERATE"
    return "HIGH"


def freeze_risk_levels(operating_threshold: float, high_at_or_above: float) -> dict[str, object]:
    threshold = float(operating_threshold)
    high_cut = float(high_at_or_above)
    if high_cut <= threshold:
        high_cut = threshold
    return {
        "low_below": threshold,
        "high_at_or_above": high_cut,
        "moderate_band": high_cut > threshold,
        "note": RISK_LEVEL_NOTE,
    }


def classify_trend(probabilities, delta: float = 0.02) -> str:
    values = [float(probability) for probability in probabilities]
    if len(values) < 2:
        return "INSUFFICIENT_HISTORY"
    change = values[-1] - values[-2]
    if change > delta:
        return "INCREASING"
    if change < -delta:
        return "DECREASING"
    return "STABLE"


def build_decision_record(
    patient_id,
    current_hour,
    raw_probability,
    calibrated_probability,
    risk_level,
    first_alert_hour,
    persistent_alert_hours,
    trend,
    clinical_review_recommended,
    abnormalities,
    hourly_alert,
) -> dict[str, object]:
    return {
        "patient_id": str(patient_id),
        "current_hour": int(current_hour),
        "raw_probability": None if raw_probability is None else float(raw_probability),
        "calibrated_probability": float(calibrated_probability),
        "risk_score": float(calibrated_probability),
        "risk_score_meaning": RISK_SCORE_MEANING,
        "risk_level": risk_level,
        "risk_level_note": RISK_LEVEL_NOTE,
        "first_alert_hour": None if first_alert_hour is None else int(first_alert_hour),
        "persistent_alert_hours": int(persistent_alert_hours),
        "trend": trend,
        "hourly_alert": bool(hourly_alert),
        "clinical_review_recommended": bool(clinical_review_recommended),
        "clinical_review_basis": "A recomendação segue a regra temporal congelada na validação, não um diagnóstico.",
        "observed_abnormalities": list(abnormalities),
        "disclaimer": DISCLAIMER,
    }
