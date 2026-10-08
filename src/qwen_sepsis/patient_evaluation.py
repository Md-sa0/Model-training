"""Agregação temporal e métricas no nível do paciente."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .calibration import binary_metrics, expected_calibration_error
from .data import hour_value, normalize_patient_id
from .explain import abnormalities_until, build_decision_record, classify_risk, classify_trend

STRATEGIES = {
    "any_alert": {
        "kind": "consecutive",
        "hours": 1,
        "description": "Um alerta horário classifica o paciente como positivo.",
    },
    "persist_2h": {
        "kind": "consecutive",
        "hours": 2,
        "description": "Duas horas consecutivas acima do limiar.",
    },
    "persist_3h": {
        "kind": "consecutive",
        "hours": 3,
        "description": "Três horas consecutivas acima do limiar.",
    },
    "window_2_in_3h": {
        "kind": "window",
        "hours": 3,
        "min_alerts": 2,
        "description": "Dois ou mais alertas dentro de três horas.",
    },
}

STRATEGY_SELECTION_RULE = (
    "Na validação, entre as estratégias que atingem a sensibilidade alvo, "
    "prioriza maior sensibilidade, depois menos falsos negativos, depois maior lead time mediano "
    "e depois menos falsos positivos. Se nenhuma atingir a meta, a mesma ordem vale sobre todas "
    "e o relatório registra que a meta não foi alcançada."
)

METRIC_DEFINITIONS = {
    "evaluation_unit": "Paciente, não hora. As horas são agrupadas por Patient_ID.",
    "patient_prediction": "A decisão binária segue a regra temporal congelada na validação.",
    "max_risk": (
        "Maior probabilidade calibrada horária do paciente. Entra em AUROC, AUPRC, Brier e ECE. "
        "Não substitui a regra temporal."
    ),
    "persistent_positive_hours": "Maior sequência de horas consecutivas acima do limiar na série avaliada.",
    "persistent_alert_hours": "Sequência de horas consecutivas acima do limiar que termina na hora corrente.",
    "number_of_alerts": "Número de horas em que a estratégia temporal selecionada dispara; não é o total de horas brutas acima do limiar.",
    "lead_time": "Hora de início do rótulo de sepse menos a hora do primeiro alerta da regra. Positivo antecipa o evento.",
    "pct_detected_before": "Fração dos pacientes com sepse cujo lead time é pelo menos o horizonte pedido. Quem não foi detectado entra no denominador.",
    "false_alarms_per_patient": "Média, em todos os pacientes, das horas acima do limiar ocorridas em pacientes sem sepse.",
    "time_axis": "Hour. A hora t usa somente observações com Hour <= t.",
}


def strategy_fields(name: str) -> dict[str, object]:
    spec = STRATEGIES[name]
    if spec["kind"] == "consecutive":
        return {
            "aggregation_strategy": name,
            "persistence_hours": spec["hours"],
            "window_hours": None,
            "window_min_alerts": None,
            "strategy_description": spec["description"],
        }
    return {
        "aggregation_strategy": name,
        "persistence_hours": None,
        "window_hours": spec["hours"],
        "window_min_alerts": spec["min_alerts"],
        "strategy_description": spec["description"],
    }


def _strategy(strategy) -> dict:
    if isinstance(strategy, str):
        if strategy not in STRATEGIES:
            raise ValueError(f"Estratégia temporal desconhecida: {strategy}")
        return STRATEGIES[strategy]
    return strategy


def _labels(frame: pd.DataFrame) -> pd.Series:
    if "SepsisLabel" in frame.columns:
        return frame["SepsisLabel"].astype(int)
    if "label" in frame.columns:
        return frame["label"].astype(int)
    raise ValueError("SepsisLabel ausente")


def _triggers(flag_at: dict[int, bool], hour: int, strategy: dict) -> bool:
    # offset 0 é a hora atual; offsets maiores olham somente horas anteriores.
    if strategy["kind"] == "consecutive":
        return all(flag_at.get(hour - offset, False) for offset in range(strategy["hours"]))
    if strategy["kind"] == "window":
        alerts = sum(flag_at.get(hour - offset, False) for offset in range(strategy["hours"]))
        return alerts >= strategy["min_alerts"]
    raise ValueError(f"Tipo de estratégia desconhecido: {strategy['kind']}")


def _scan(hours: list[int], flags: list[bool], strategy: dict) -> dict[str, object]:
    flag_at = {int(hour): bool(flag) for hour, flag in zip(hours, flags, strict=True)}
    trigger_hours = [hour for hour in sorted(flag_at) if _triggers(flag_at, hour, strategy)]
    if not trigger_hours:
        return {"patient_prediction": 0, "first_alert_hour": None, "trigger_hours": []}
    return {
        "patient_prediction": 1,
        "first_alert_hour": int(trigger_hours[0]),
        "trigger_hours": [int(hour) for hour in trigger_hours],
    }


def max_consecutive(hours: list[int], flags: list[bool]) -> int:
    best = current = 0
    previous = None
    for hour, flag in zip(hours, flags, strict=True):
        if flag and previous is not None and hour == previous + 1:
            current += 1
        elif flag:
            current = 1
        else:
            current = 0
        best = max(best, current)
        previous = hour
    return int(best)


def streak_ending_at(hours: list[int], flags: list[bool], current_hour: int) -> int:
    flag_at = {int(hour): bool(flag) for hour, flag in zip(hours, flags, strict=True)}
    streak = 0
    hour = int(current_hour)
    while flag_at.get(hour, False):
        streak += 1
        hour -= 1
    return int(streak)


def summarize_patient(rows: pd.DataFrame, threshold: float, strategy, current_hour=None,
                      probability_column: str = "calibrated_probability") -> dict[str, object]:
    """Estado causal até a hora pedida. Linhas futuras não entram."""
    spec = _strategy(strategy)
    work = rows.copy()
    if "Patient_ID" in work.columns and work["Patient_ID"].map(normalize_patient_id).nunique() > 1:
        raise ValueError("summarize_patient recebe um único paciente")
    work["Hour"] = work["Hour"].map(hour_value)
    work["__label"] = _labels(work)
    if probability_column not in work.columns:
        raise ValueError(f"Coluna de probabilidade ausente: {probability_column}")
    work[probability_column] = work[probability_column].astype(float)
    requested = None if current_hour is None else hour_value(current_hour)
    if requested is not None:
        work = work.loc[work["Hour"] <= requested]
    if work.empty:
        raise ValueError("Nenhuma observação disponível até a hora atual")
    work = work.sort_values("Hour", kind="stable")
    if work["Hour"].duplicated().any():
        raise ValueError("Hora duplicada para o mesmo paciente")
    hours = [int(hour) for hour in work["Hour"].tolist()]
    labels = [int(label) for label in work["__label"].tolist()]
    probabilities = [float(probability) for probability in work[probability_column].tolist()]
    flags = [probability >= threshold for probability in probabilities]
    available_hour = hours[-1]
    scan = _scan(hours, flags, spec)
    positive_hours = [hour for hour, label in zip(hours, labels, strict=True) if label == 1]
    onset = min(positive_hours) if positive_hours else None
    has_sepsis = int(onset is not None)
    prediction = int(scan["patient_prediction"])
    first_alert = scan["first_alert_hour"]
    alert_hours = scan["trigger_hours"]
    lead_time = None if onset is None or first_alert is None else float(onset - first_alert)
    hour_gap = int(any(hours[index] - hours[index - 1] > 1 for index in range(1, len(hours))))
    return {
        "has_sepsis": has_sepsis,
        "ground_truth": has_sepsis,
        "max_risk": float(max(probabilities)),
        "max_calibrated_probability": float(max(probabilities)),
        "first_alert_hour": first_alert,
        "last_alert_hour": int(alert_hours[-1]) if alert_hours else None,
        "number_of_alerts": int(len(alert_hours)),
        "persistent_positive_hours": max_consecutive(hours, flags),
        "persistent_alert_hours": streak_ending_at(hours, flags, available_hour),
        "patient_prediction": prediction,
        "prediction": prediction,
        "false_negative": int(has_sepsis == 1 and prediction == 0),
        "false_positive": int(has_sepsis == 0 and prediction == 1),
        "onset_hour": onset,
        "reference_event_hour": onset,
        "lead_time": lead_time,
        "current_hour": available_hour,
        "current_probability": probabilities[-1],
        "probability_history": probabilities,
        "hour_history": hours,
        "hours_above_threshold": int(sum(flags)),
        "false_alarm_hours": int(sum(flags)) if has_sepsis == 0 else 0,
        "hour_gap": hour_gap,
    }


def _table_row(patient_id: str, state: dict) -> dict:
    keys = [
        "has_sepsis", "ground_truth", "max_risk", "max_calibrated_probability", "first_alert_hour",
        "last_alert_hour", "number_of_alerts", "persistent_positive_hours", "patient_prediction", "prediction",
        "false_negative", "false_positive", "onset_hour", "lead_time", "persistent_alert_hours",
        "reference_event_hour", "hours_above_threshold", "false_alarm_hours", "hour_gap", "current_hour",
        "current_probability",
    ]
    return {"Patient_ID": patient_id, **{key: state[key] for key in keys}}


def iter_patient_states(frame: pd.DataFrame, threshold: float, strategy, probability_column: str):
    if "Patient_ID" not in frame.columns:
        raise ValueError("Patient_ID é obrigatório para agrupar as horas do mesmo paciente.")
    ordered = frame.copy()
    ordered["Patient_ID"] = ordered["Patient_ID"].map(normalize_patient_id)
    if ordered.duplicated(["Patient_ID", "Hour"]).any():
        raise ValueError("Há horas duplicadas para o mesmo paciente")
    for patient_id, part in ordered.groupby("Patient_ID", sort=True):
        state = summarize_patient(part, threshold, strategy, probability_column=probability_column)
        yield str(patient_id), part, state


def build_patient_table(frame: pd.DataFrame, threshold: float, strategy,
                        probability_column: str = "calibrated_probability") -> pd.DataFrame:
    rows = [
        _table_row(patient_id, state)
        for patient_id, _part, state in iter_patient_states(frame, threshold, strategy, probability_column)
    ]
    if not rows:
        raise ValueError("Nenhum paciente para avaliar.")
    return pd.DataFrame(rows)


def _finite(value):
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _ranking(y_true, scores, function):
    actual = np.asarray(y_true).astype(int)
    if len(np.unique(actual)) < 2:
        return None
    return float(function(actual, scores))


def patient_level_metrics(table: pd.DataFrame) -> dict[str, object]:
    if table.empty:
        raise ValueError("Nenhum paciente para avaliar.")
    actual = table["has_sepsis"].astype(int)
    predicted = table["patient_prediction"].astype(int)
    scores = table["max_risk"].astype(float).clip(0.0, 1.0)
    counts = binary_metrics(actual, predicted)
    sepsis = table[table["has_sepsis"] == 1]
    detected = sepsis[sepsis["patient_prediction"] == 1]["lead_time"].dropna()
    n_sepsis = int(len(sepsis))

    def early(hours: int):
        if n_sepsis == 0:
            return None
        return float((sepsis["lead_time"] >= hours).sum() / n_sepsis)

    ece, _curve = expected_calibration_error(actual, scores)
    return {
        "sensitivity": counts["sensitivity"],
        "specificity": counts["specificity"],
        "ppv": counts["ppv"],
        "npv": counts["npv"],
        "f1": counts["f1"],
        "mcc": counts["mcc"],
        "auroc": _ranking(actual, scores, roc_auc_score),
        "auprc": _ranking(actual, scores, average_precision_score),
        "brier_score": float(brier_score_loss(actual, scores)),
        "ece": ece,
        "true_positives": counts["TP"],
        "true_negatives": counts["TN"],
        "false_positives": counts["FP"],
        "false_negatives": counts["FN"],
        "mean_lead_time": _finite(detected.mean()) if len(detected) else None,
        "median_lead_time": _finite(detected.median()) if len(detected) else None,
        "minimum_lead_time": _finite(detected.min()) if len(detected) else None,
        "maximum_lead_time": _finite(detected.max()) if len(detected) else None,
        "pct_detected_6h_before": early(6),
        "pct_detected_12h_before": early(12),
        "pct_detected_24h_before": early(24),
        "n_undetected": int(((table["has_sepsis"] == 1) & (table["patient_prediction"] == 0)).sum()),
        "n_patients": int(len(table)),
        "n_sepsis_patients": n_sepsis,
        "n_detected": int(((table["has_sepsis"] == 1) & (table["patient_prediction"] == 1)).sum()),
        "false_alarms_per_patient": float(table["false_alarm_hours"].mean()),
        "patients_with_hour_gaps": int(table["hour_gap"].sum()),
        "confusion_matrix": [[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]],
    }


def hourly_calibration(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    actual = _labels(frame)
    scores = np.clip(frame[probability_column].to_numpy(dtype=float), 0.0, 1.0)
    ece, _curve = expected_calibration_error(actual, scores)
    return {
        "hourly_brier_score": float(brier_score_loss(actual, scores)),
        "hourly_ece": ece,
        "hourly_log_loss": float(log_loss(actual, np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }


def hourly_evaluation(frame: pd.DataFrame, threshold: float,
                      probability_column: str = "calibrated_probability") -> dict[str, object]:
    actual = _labels(frame).to_numpy(dtype=int)
    probabilities = frame[probability_column].to_numpy(dtype=float)
    counts = binary_metrics(actual, probabilities >= threshold)
    patient_labels = frame.assign(__label=actual).groupby("Patient_ID", sort=False)["__label"].max()
    return {
        "prediction_unit": "hourly observation",
        "evaluation_unit": "hourly observation",
        "total_samples": int(len(actual)),
        "positive_samples": int(actual.sum()),
        "negative_samples": int((actual == 0).sum()),
        "positive_rate": float(actual.mean()),
        "negative_rate": float((actual == 0).mean()),
        "total_patients": int(len(patient_labels)),
        "positive_patients": int(patient_labels.sum()),
        "negative_patients": int((patient_labels == 0).sum()),
        "patient_positive_rate": float(patient_labels.mean()),
        **counts,
        **hourly_calibration(frame, probability_column),
    }


def _value_at_hour(part: pd.DataFrame, column: str, hour: int):
    if column not in part.columns:
        return None
    hours = part["Hour"].map(hour_value)
    selected = part.loc[hours == hour, column]
    if selected.empty or pd.isna(selected.iloc[-1]):
        return None
    return float(selected.iloc[-1])


def evaluate_patients(frame, threshold: float, strategy_name: str, risk_levels: dict, trend_delta: float,
                      probability_column: str = "calibrated_probability"):
    rows = []
    decisions = []
    for patient_id, part, state in iter_patient_states(frame, threshold, strategy_name, probability_column):
        rows.append(_table_row(patient_id, state))
        risk_level = classify_risk(
            state["current_probability"], risk_levels["low_below"], risk_levels["high_at_or_above"],
        )
        decisions.append(build_decision_record(
            patient_id=patient_id,
            current_hour=state["current_hour"],
            raw_probability=_value_at_hour(part, "raw_probability", state["current_hour"]),
            calibrated_probability=state["current_probability"],
            risk_level=risk_level,
            first_alert_hour=state["first_alert_hour"],
            persistent_alert_hours=state["persistent_alert_hours"],
            trend=classify_trend(state["probability_history"], trend_delta),
            clinical_review_recommended=state["patient_prediction"] == 1,
            abnormalities=abnormalities_until(part, state["current_hour"]),
            hourly_alert=state["current_probability"] >= threshold,
        ))
    table = pd.DataFrame(rows)
    if len(table) != frame["Patient_ID"].map(normalize_patient_id).nunique():
        raise RuntimeError("A quantidade de predições não corresponde aos pacientes únicos")
    metrics = patient_level_metrics(table)
    return table, metrics, decisions


def compare_temporal_strategies(frame: pd.DataFrame, threshold: float, split: str = "validation") -> pd.DataFrame:
    if split != "validation":
        raise ValueError("As estratégias temporais só podem ser comparadas para escolha na validação.")
    rows = []
    for name in STRATEGIES:
        table = build_patient_table(frame, threshold, name)
        metrics = patient_level_metrics(table)
        rows.append({
            "strategy": name,
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "PPV": metrics["ppv"],
            "NPV": metrics["npv"],
            "F1": metrics["f1"],
            "MCC": metrics["mcc"],
            "FN": metrics["false_negatives"],
            "FP": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "false_positives": metrics["false_positives"],
            "lead_time": metrics["median_lead_time"],
            "median_lead_time": metrics["median_lead_time"],
            "mean_lead_time": metrics["mean_lead_time"],
            "pct_detected_6h_before": metrics["pct_detected_6h_before"],
            "pct_detected_12h_before": metrics["pct_detected_12h_before"],
            "pct_detected_24h_before": metrics["pct_detected_24h_before"],
            "n_undetected": metrics["n_undetected"],
            "description": STRATEGIES[name]["description"],
        })
    return pd.DataFrame(rows)


def select_temporal_strategy(comparison: pd.DataFrame, target_sensitivity: float, split: str = "validation") -> dict:
    if split != "validation":
        raise ValueError("A regra temporal só pode ser escolhida na validação.")
    if comparison.empty:
        raise ValueError("Nenhuma estratégia temporal foi calculada")
    eligible = comparison[comparison["sensitivity"].fillna(-1) >= target_sensitivity]
    pool = eligible if not eligible.empty else comparison
    ranked = pool.sort_values(
        by=["sensitivity", "false_negatives", "lead_time", "false_positives"],
        ascending=[False, True, False, True],
        na_position="last",
        kind="mergesort",
    )
    chosen = ranked.iloc[0]
    return {
        "strategy": str(chosen["strategy"]),
        "target_met": bool(not eligible.empty),
        "metrics": chosen.to_dict(),
    }
