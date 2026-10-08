from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_probability(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        raise ValueError("Nenhum score para avaliar.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Há scores não finitos na entrada.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Os scores fora do intervalo [0, 1] não são válidos para análise de qualidade.")
    return values


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def summarize_raw_score_distribution(
    frame: pd.DataFrame,
    score_column: str = "raw_probability",
    label_column: str = "SepsisLabel",
) -> dict[str, dict[str, float | int]]:
    if score_column not in frame.columns:
        raise ValueError(f"A coluna {score_column!r} não foi encontrada.")
    if label_column not in frame.columns:
        raise ValueError(f"A coluna {label_column!r} não foi encontrada.")

    summary: dict[str, dict[str, float | int]] = {}
    for label in sorted(frame[label_column].dropna().unique().tolist()):
        values = frame.loc[frame[label_column] == label, score_column].astype(float).to_numpy()
        if values.size == 0:
            continue
        summary[f"label_{int(label)}"] = {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=0)),
            "p01": _percentile(values, 0.01),
            "p05": _percentile(values, 0.05),
            "p10": _percentile(values, 0.10),
            "p25": _percentile(values, 0.25),
            "p50": _percentile(values, 0.50),
            "p75": _percentile(values, 0.75),
            "p90": _percentile(values, 0.90),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
        }
    return summary


def _calc_sensitivity(tp: int, fn: int) -> float:
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def _calc_specificity(tn: int, fp: int) -> float:
    return 0.0 if tn + fp == 0 else tn / (tn + fp)


def _calc_precision(tp: int, fp: int) -> float:
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def _calc_npv(tn: int, fn: int) -> float:
    return 0.0 if tn + fn == 0 else tn / (tn + fn)


def _calc_f1(tp: int, fp: int, fn: int) -> float:
    precision = _calc_precision(tp, fp)
    recall = _calc_sensitivity(tp, fn)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _calc_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denominator


def compute_threshold_analysis(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    thresholds: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=int)
    scores = _safe_probability(probabilities)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels e scores devem ter o mesmo número de elementos.")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    threshold_values = np.asarray(thresholds, dtype=float)
    rows: list[dict[str, float | int]] = []

    for threshold in threshold_values:
        prediction = scores >= float(threshold)
        tp = int(np.sum((prediction == 1) & (labels == 1)))
        tn = int(np.sum((prediction == 0) & (labels == 0)))
        fp = int(np.sum((prediction == 1) & (labels == 0)))
        fn = int(np.sum((prediction == 0) & (labels == 1)))
        sensitivity = _calc_sensitivity(tp, fn)
        specificity = _calc_specificity(tn, fp)
        ppv = _calc_precision(tp, fp)
        npv = _calc_npv(tn, fn)
        f1 = _calc_f1(tp, fp, fn)
        mcc = _calc_mcc(tp, tn, fp, fn)
        rows.append(
            {
                "threshold": float(threshold),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "ppv": float(ppv),
                "npv": float(npv),
                "f1": float(f1),
                "mcc": float(mcc),
                "youden_j": float(sensitivity + specificity - 1.0),
                "positive_rate": float(np.mean(prediction)),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold", ascending=True, kind="mergesort").reset_index(drop=True)


def find_best_threshold(table: pd.DataFrame, criterion: str) -> dict[str, float | int]:
    allowed = {"youden_j", "f1", "mcc", "sensitivity_ge_95"}
    if criterion not in allowed:
        raise ValueError(f"Critério inválido: {criterion!r}. Opções: {sorted(allowed)}")
    if criterion == "sensitivity_ge_95":
        feasible = table.loc[table["sensitivity"] >= 0.95].copy()
        if feasible.empty:
            return {"threshold": float("nan"), "sensitivity": float("nan"), "specificity": float("nan")}
        return feasible.sort_values(["specificity", "threshold"], ascending=[False, True], kind="mergesort").iloc[0].to_dict()
    if criterion == "youden_j":
        target = "youden_j"
    elif criterion == "f1":
        target = "f1"
    else:
        target = "mcc"
    best_row = table.loc[table[target].idxmax()].to_dict()
    return best_row


def aggregate_patient_scores(
    frame: pd.DataFrame,
    score_column: str = "raw_probability",
    label_column: str = "SepsisLabel",
    patient_column: str = "Patient_ID",
) -> pd.DataFrame:
    if score_column not in frame.columns:
        raise ValueError(f"A coluna {score_column!r} não foi encontrada.")
    if label_column not in frame.columns:
        raise ValueError(f"A coluna {label_column!r} não foi encontrada.")
    if patient_column not in frame.columns:
        raise ValueError(f"A coluna {patient_column!r} não foi encontrada.")

    patient_frame = (
        frame.groupby(patient_column, sort=False)
        .agg(
            patient_label=(label_column, "max"),
            max_probability=(score_column, "max"),
            mean_probability=(score_column, "mean"),
            median_probability=(score_column, "median"),
            p95_probability=(score_column, lambda values: float(np.quantile(values, 0.95))),
            alert_hours=(score_column, lambda values: int(np.sum(values >= 0.5))),
        )
        .reset_index()
    )
    return patient_frame


def _auroc_auprc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    values = _safe_probability(scores)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("labels e scores devem ter o mesmo tamanho.")
    auroc = float(roc_auc_score(labels, values))
    auprc = float(average_precision_score(labels, values))
    return {"auroc": auroc, "auprc": auprc}


def compute_patient_aggregation_metrics(frame: pd.DataFrame, score_column: str = "raw_probability") -> dict[str, dict[str, float]]:
    patient_scores = aggregate_patient_scores(frame, score_column=score_column)
    metrics: dict[str, dict[str, float]] = {}
    for column_name in ("max_probability", "mean_probability", "median_probability", "p95_probability"):
        labels = patient_scores["patient_label"].to_numpy(dtype=int)
        values = patient_scores[column_name].to_numpy(dtype=float)
        metrics[column_name] = _auroc_auprc(labels, values)
    return metrics


def _baseline_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    pred = np.asarray(prediction, dtype=int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    return {
        "sensitivity": _calc_sensitivity(tp, fn),
        "specificity": _calc_specificity(tn, fp),
        "ppv": _calc_precision(tp, fp),
        "npv": _calc_npv(tn, fn),
        "f1": _calc_f1(tp, fp, fn),
        "mcc": _calc_mcc(tp, tn, fp, fn),
        "FN": fn,
        "FP": fp,
    }


def build_model_quality_report(
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    current_threshold: float = 0.0,
    score_column: str = "raw_probability",
    label_column: str = "SepsisLabel",
) -> dict[str, Any]:
    validation_labels = validation_frame[label_column].to_numpy(dtype=int)
    validation_scores = _safe_probability(validation_frame[score_column].to_numpy(dtype=float))
    test_labels = test_frame[label_column].to_numpy(dtype=int)
    test_scores = _safe_probability(test_frame[score_column].to_numpy(dtype=float))

    validation_thresholds = compute_threshold_analysis(validation_labels, validation_scores)
    test_thresholds = compute_threshold_analysis(test_labels, test_scores)
    best_youden = find_best_threshold(validation_thresholds, "youden_j")
    best_f1 = find_best_threshold(validation_thresholds, "f1")
    best_mcc = find_best_threshold(validation_thresholds, "mcc")
    best_sensitivity = find_best_threshold(validation_thresholds, "sensitivity_ge_95")
        
    current_validation = validation_thresholds.loc[np.isclose(validation_thresholds["threshold"], current_threshold, atol=1e-8)].head(1)
    current_validation_metrics = current_validation.iloc[0].to_dict() if not current_validation.empty else {
        "threshold": current_threshold,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "ppv": 0.0,
        "npv": 0.0,
        "f1": 0.0,
        "mcc": 0.0,
        "FN": 0,
        "FP": 0,
    }
    current_test = test_thresholds.loc[np.isclose(test_thresholds["threshold"], current_threshold, atol=1e-8)].head(1)
    current_test_metrics = current_test.iloc[0].to_dict() if not current_test.empty else {
        "threshold": current_threshold,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "ppv": 0.0,
        "npv": 0.0,
        "f1": 0.0,
        "mcc": 0.0,
        "FN": 0,
        "FP": 0,
    }

    positive_prevalence = float(np.mean(test_labels == 1)) if test_labels.size else 0.0
    validation_metrics = _auroc_auprc(validation_labels, validation_scores)
    test_metrics = _auroc_auprc(test_labels, test_scores)
    patient_metrics = compute_patient_aggregation_metrics(test_frame, score_column=score_column)

    always_negative = _baseline_metrics(test_labels, np.zeros_like(test_labels, dtype=int))
    always_positive = _baseline_metrics(test_labels, np.ones_like(test_labels, dtype=int))

    positive_scores = validation_scores[validation_labels == 1]
    negative_scores = validation_scores[validation_labels == 0]

    report = {
        "validation": {
            "auroc": validation_metrics["auroc"],
            "auprc": validation_metrics["auprc"],
            "positive_prevalence": float(np.mean(validation_labels == 1)),
            "mean_positive_score": float(np.mean(positive_scores)) if positive_scores.size else 0.0,
            "mean_negative_score": float(np.mean(negative_scores)) if negative_scores.size else 0.0,
        },
        "test": {
            "auroc": test_metrics["auroc"],
            "auprc": test_metrics["auprc"],
            "positive_prevalence": positive_prevalence,
            "mean_positive_score": float(np.mean(test_scores[test_labels == 1])) if np.any(test_labels == 1) else 0.0,
            "mean_negative_score": float(np.mean(test_scores[test_labels == 0])) if np.any(test_labels == 0) else 0.0,
        },
        "positive_prevalence": positive_prevalence,
        "current_threshold": float(current_threshold),
        "current_validation_metrics": current_validation_metrics,
        "current_test_metrics": current_test_metrics,
        "best_youden_threshold": best_youden,
        "best_f1_threshold": best_f1,
        "best_mcc_threshold": best_mcc,
        "best_sensitivity_ge_95_threshold": best_sensitivity,
        "patient_level": patient_metrics,
        "always_negative": always_negative,
        "always_positive": always_positive,
        "score_discrimination": {
            "validation_mean_positive_score": float(np.mean(positive_scores)) if positive_scores.size else 0.0,
            "validation_mean_negative_score": float(np.mean(negative_scores)) if negative_scores.size else 0.0,
            "validation_median_positive_score": float(np.median(positive_scores)) if positive_scores.size else 0.0,
            "validation_median_negative_score": float(np.median(negative_scores)) if negative_scores.size else 0.0,
        },
    }
    return report
