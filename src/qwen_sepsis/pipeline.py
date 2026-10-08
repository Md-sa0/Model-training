"""Seleção na validação e avaliação congelada no teste."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import (
    FrozenCalibrator,
    compare_calibrators,
    load_calibrator,
    save_calibrator,
    select_threshold,
)
from .explain import DISCLAIMER, OPERATIONAL_RANGES, RANGE_NOTE, freeze_risk_levels
from .patient_evaluation import (
    METRIC_DEFINITIONS,
    STRATEGY_SELECTION_RULE,
    _labels,
    compare_temporal_strategies,
    evaluate_patients,
    hourly_evaluation,
    select_temporal_strategy,
    strategy_fields,
)
from .plotting import save_calibration_curve, save_confusion_matrix, save_threshold_curve

REQUIRED_REPORT_KEYS = [
    "prediction_unit",
    "evaluation_unit",
    "calibration_method",
    "threshold",
    "target_sensitivity",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "mcc",
    "auroc",
    "auprc",
    "brier_score",
    "ece",
    "false_negatives",
    "false_positives",
    "mean_lead_time",
    "median_lead_time",
]

LIMITATIONS = [
    DISCLAIMER,
    "A calibração melhora a leitura probabilística, mas não transforma o modelo em dispositivo clinicamente validado.",
    "Não se assume que todos os falsos negativos possam ser eliminados. A meta é alta sensibilidade com a carga de falsos positivos definida na validação.",
    "Brier e ECE usados para escolher o calibrador são calculados na validação, no mesmo conjunto em que Platt e a isotônica foram ajustados.",
    "O conjunto de teste não altera calibrador, limiar nem regra temporal.",
    "max_risk ordena pacientes para AUROC, AUPRC, Brier e ECE. A decisão binária segue a regra temporal.",
    "O lead time usa a coluna Hour e as horas presentes no split.",
]


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Valor não serializável: {type(value)!r}")


def attach_raw_scores(frame: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    if len(frame) != len(records):
        raise ValueError("Quantidade de scores diferente da quantidade de linhas")
    scored = frame.copy()
    scored["logit_0"] = [record["logit_0"] for record in records]
    scored["logit_1"] = [record["logit_1"] for record in records]
    scored["raw_probability"] = [record["raw_probability"] for record in records]
    return scored


@dataclass
class ValidationSelection:
    calibrator: FrozenCalibrator
    config: dict
    threshold_table: pd.DataFrame
    strategy_table: pd.DataFrame
    calibration_metrics: dict
    scored_frame: pd.DataFrame


def select_from_validation(frame: pd.DataFrame, target_sensitivity: float, seed: int,
                           high_risk_probability: float, trend_delta: float, ece_bins: int = 10,
                           split: str = "validation") -> ValidationSelection:
    if split != "validation":
        raise ValueError("Calibrador, limiar e regra temporal só podem ser definidos na validação.")
    work = frame.copy()
    if "raw_probability" not in work.columns:
        raise ValueError("A validação precisa da coluna raw_probability")
    labels = _labels(work).to_numpy()
    scores = work["raw_probability"].to_numpy(dtype=float)
    calibrator, metrics = compare_calibrators(scores, labels, split="validation", seed=seed, n_bins=ece_bins)
    work["calibrated_probability"] = calibrator.predict(scores)
    threshold_info, threshold_table = select_threshold(
        labels, work["calibrated_probability"].to_numpy(dtype=float),
        target_sensitivity=target_sensitivity, split="validation",
    )
    strategy_table = compare_temporal_strategies(work, threshold_info["threshold"], split="validation")
    chosen_strategy = select_temporal_strategy(strategy_table, target_sensitivity, split="validation")
    chosen_metrics = {
        key: chosen_strategy["metrics"][key]
        for key in (
            "sensitivity", "specificity", "PPV", "NPV", "FN", "FP", "lead_time",
            "mean_lead_time", "n_undetected",
        )
    }
    maximum_sensitivity = threshold_table.sort_values(
        ["sensitivity", "specificity", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    config = {
        "method": calibrator.method,
        "threshold": threshold_info["threshold"],
        "target_sensitivity": float(target_sensitivity),
        "target_met": threshold_info["target_met"],
        "achieved_sensitivity_validation_hourly": threshold_info["achieved_sensitivity"],
        "achieved_specificity_validation_hourly": threshold_info["achieved_specificity"],
        "maximum_validation_sensitivity": float(maximum_sensitivity["sensitivity"]),
        "maximum_sensitivity_threshold": float(maximum_sensitivity["threshold"]),
        "maximum_sensitivity_fp": int(maximum_sensitivity["FP"]),
        "maximum_sensitivity_fn": int(maximum_sensitivity["FN"]),
        "threshold_message": threshold_info["message"],
        **strategy_fields(chosen_strategy["strategy"]),
        "strategy_target_met": chosen_strategy["target_met"],
        "strategy_selection_rule": STRATEGY_SELECTION_RULE,
        "validation_strategy_metrics": chosen_metrics,
        "risk_levels": freeze_risk_levels(threshold_info["threshold"], high_risk_probability),
        "trend_delta": float(trend_delta),
        "calibration_feature": "raw_probability",
        "calibration_selection_rule": "Menor Brier na validação; empate resolve por menor ECE e depois menor log loss.",
        "calibration_comparison": {
            name: {"brier": values["brier"], "ece": values["ece"], "log_loss": values["log_loss"]}
            for name, values in metrics.items()
        },
        "validation_brier": metrics[calibrator.method]["brier"],
        "validation_ece": metrics[calibrator.method]["ece"],
        "seed": int(seed),
        "ece_bins": int(ece_bins),
        "fitted_on": "validation",
        "n_validation_rows": int(len(work)),
        "n_validation_patients": int(work["Patient_ID"].map(lambda value: str(value)).nunique()),
        "operational_ranges": OPERATIONAL_RANGES,
        "operational_ranges_note": RANGE_NOTE,
        "test_used_for_fitting": False,
    }
    return ValidationSelection(
        calibrator=calibrator,
        config=json_ready(config),
        threshold_table=threshold_table,
        strategy_table=strategy_table,
        calibration_metrics=metrics,
        scored_frame=work,
    )


def save_validation_artifacts(output_dir: Path, selection: ValidationSelection, context: dict) -> None:
    if context.get("seed") not in (None, selection.config["seed"]):
        raise ValueError("Seed divergente entre o contexto e a validação")
    calibration_dir = Path(output_dir) / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    save_calibrator(selection.calibrator, calibration_dir / "calibrator.pkl")
    config = json_ready({**selection.config, **context, "fitted_on": "validation", "test_used_for_fitting": False})
    (calibration_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selection.threshold_table.to_csv(calibration_dir / "threshold_analysis.csv", index=False)
    selection.strategy_table.to_csv(calibration_dir / "temporal_strategies.csv", index=False)
    (calibration_dir / "comparison.json").write_text(
        json.dumps(json_ready(selection.calibration_metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_calibration_curve(calibration_dir / "calibration_curve.png", selection.calibration_metrics)
    save_threshold_curve(
        calibration_dir / "threshold_curve.png",
        selection.threshold_table,
        float(selection.config["threshold"]),
        float(selection.config["target_sensitivity"]),
    )
    preferred = ["Patient_ID", "Hour", "SepsisLabel", "logit_0", "logit_1", "raw_probability", "calibrated_probability"]
    columns = [column for column in preferred if column in selection.scored_frame.columns]
    selection.scored_frame[columns].to_csv(calibration_dir / "validation_scores.csv", index=False)


def load_frozen_decision(output_dir: Path):
    calibration_dir = Path(output_dir) / "calibration"
    config = json.loads((calibration_dir / "config.json").read_text(encoding="utf-8"))
    calibrator = load_calibrator(calibration_dir / "calibrator.pkl")
    if calibrator.method != config["method"]:
        raise ValueError("O calibrador salvo não corresponde ao config.json")
    if config.get("fitted_on") != "validation" or config.get("test_used_for_fitting") is not False:
        raise ValueError("A configuração salva não está restrita à validação")
    return calibrator, config


def evaluate_test_frame(frame: pd.DataFrame, calibrator: FrozenCalibrator, config: dict):
    if config.get("fitted_on") != "validation" or config.get("test_used_for_fitting") is not False:
        raise RuntimeError("A avaliação de teste recebeu uma configuração que não foi congelada na validação.")
    if not calibrator.locked:
        raise RuntimeError("Calibrador não está congelado.")
    before = calibrator._parameters()
    scores = frame["raw_probability"].to_numpy(dtype=float)
    calibrated = calibrator.predict(scores)
    if not _same_parameters(before, calibrator._parameters()):
        raise RuntimeError("A aplicação no teste modificou o calibrador")
    work = frame.copy()
    work["calibrated_probability"] = calibrated
    table, metrics, decisions = evaluate_patients(
        work,
        threshold=float(config["threshold"]),
        strategy_name=config["aggregation_strategy"],
        risk_levels=config["risk_levels"],
        trend_delta=float(config["trend_delta"]),
    )
    if len(table) != work["Patient_ID"].nunique():
        raise RuntimeError("A avaliação por paciente não contém exatamente uma linha por Patient_ID único")
    return {
        "table": table,
        "metrics": metrics,
        "hourly_metrics": hourly_evaluation(work, float(config["threshold"])),
        "decisions": decisions,
        "frame": work,
    }


def _same_parameters(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return all(np.allclose(first, second) for first, second in zip(left, right, strict=True))


def build_final_report(metrics: dict, config: dict, context: dict, hourly_metrics: dict | None = None) -> dict:
    report = {
        "prediction_unit": "hourly observation",
        "evaluation_unit": "patient",
        "split": "test",
        "calibration_method": config["method"],
        "threshold": float(config["threshold"]),
        "maximum_validation_sensitivity": float(config["maximum_validation_sensitivity"]),
        "maximum_sensitivity_threshold": float(config["maximum_sensitivity_threshold"]),
        "maximum_sensitivity_fp": int(config["maximum_sensitivity_fp"]),
        "target_sensitivity": float(config["target_sensitivity"]),
        "aggregation_strategy": config["aggregation_strategy"],
        "persistence_hours": config["persistence_hours"],
        "window_hours": config["window_hours"],
        "window_min_alerts": config["window_min_alerts"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "ppv": metrics["ppv"],
        "npv": metrics["npv"],
        "f1": metrics["f1"],
        "mcc": metrics["mcc"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "brier_score": metrics["brier_score"],
        "ece": metrics["ece"],
        "false_negatives": metrics["false_negatives"],
        "false_positives": metrics["false_positives"],
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "mean_lead_time": metrics["mean_lead_time"],
        "median_lead_time": metrics["median_lead_time"],
        "minimum_lead_time": metrics["minimum_lead_time"],
        "maximum_lead_time": metrics["maximum_lead_time"],
        "pct_detected_6h_before": metrics["pct_detected_6h_before"],
        "pct_detected_12h_before": metrics["pct_detected_12h_before"],
        "pct_detected_24h_before": metrics["pct_detected_24h_before"],
        "n_undetected": metrics["n_undetected"],
        "n_patients": metrics["n_patients"],
        "n_sepsis_patients": metrics["n_sepsis_patients"],
        "n_detected": metrics["n_detected"],
        "patients_not_detected": metrics["n_undetected"],
        "false_alarms_per_patient": metrics["false_alarms_per_patient"],
        "patients_with_hour_gaps": metrics["patients_with_hour_gaps"],
        "confusion_matrix": metrics["confusion_matrix"],
        "hourly_brier_score": metrics.get("hourly_brier_score"),
        "hourly_ece": metrics.get("hourly_ece"),
        "hourly_log_loss": metrics.get("hourly_log_loss"),
        "hourly_evaluation": hourly_metrics,
        "validation_selection": {
            "threshold_message": config["threshold_message"],
            "target_met": config["target_met"],
            "achieved_sensitivity_hourly": config["achieved_sensitivity_validation_hourly"],
            "achieved_specificity_hourly": config["achieved_specificity_validation_hourly"],
            "maximum_sensitivity": config["maximum_validation_sensitivity"],
            "maximum_sensitivity_threshold": config["maximum_sensitivity_threshold"],
            "maximum_sensitivity_false_positives": config["maximum_sensitivity_fp"],
            "strategy_target_met": config["strategy_target_met"],
            "strategy_description": config["strategy_description"],
            "calibration_comparison": config["calibration_comparison"],
            "validation_strategy_metrics": config["validation_strategy_metrics"],
        },
        "test_used_for_fitting": False,
        "calibrator_locked": True,
        "metric_definitions": METRIC_DEFINITIONS,
        "reproducibility": context,
        "disclaimer": DISCLAIMER,
        "how_to_read": (
            "Métricas principais são por paciente no conjunto de teste. "
            "Calibrador, limiar e regra temporal foram congelados na validação. "
            "risk_score é a probabilidade calibrada, não um diagnóstico."
        ),
    }
    limitations = list(LIMITATIONS)
    tokens = context.get("tokens") or {}
    if tokens and not tokens.get("single_token_labels", True):
        limitations.append(
            "Os textos '0' e '1' não são tokens únicos neste tokenizer. "
            "A probabilidade usa o primeiro token de cada rótulo, que é o token supervisionado no treino."
        )
    if report["patients_with_hour_gaps"]:
        limitations.append(
            "Há pacientes com horas faltantes no split. O lead time usa apenas as horas presentes."
        )
    warnings = []
    if report["sensitivity"] is not None and report["sensitivity"] < report["target_sensitivity"]:
        warnings.append(
            "A sensibilidade no teste ficou abaixo da meta definida na validação. O limiar não foi reajustado."
        )
    if float(config["threshold"]) <= 0.0 or float(config["achieved_specificity_validation_hourly"]) == 0.0:
        warnings.append(
            "WARNING: A meta de sensibilidade foi alcançada na validação somente com threshold zero ou "
            "especificidade zero; isso gera falsos positivos em todas as observações sem sepse. "
            f"FP horários na validação={config['maximum_sensitivity_fp']}."
        )
    if report["specificity"] == 0.0 and report["false_positives"]:
        warnings.append(
            "WARNING: A avaliação por paciente no test teve especificidade zero: "
            f"{report['false_positives']} de {report['n_patients'] - report['n_sepsis_patients']} pacientes sem sepse foram sinalizados."
        )
    report["limitations"] = limitations
    report["warnings"] = warnings
    return json_ready(report)


def _predictions_frame(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Patient_ID", "has_sepsis", "max_risk", "first_alert_hour", "persistent_positive_hours",
        "patient_prediction", "false_negative", "false_positive", "onset_hour", "lead_time",
        "ground_truth", "prediction", "max_calibrated_probability", "last_alert_hour", "number_of_alerts",
        "reference_event_hour", "persistent_alert_hours",
        "hours_above_threshold", "false_alarm_hours", "hour_gap",
        "current_hour", "current_probability",
    ]
    out = table[[column for column in columns if column in table.columns]].copy()
    for column in ("first_alert_hour", "onset_hour", "current_hour"):
        if column not in out.columns:
            continue
        out[column] = ["" if pd.isna(value) else int(value) for value in out[column].tolist()]
    return out


def write_evaluation_artifacts(output_dir: Path, result: dict, report: dict) -> None:
    evaluation_dir = Path(output_dir) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "final_patient_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    scalar = {key: value for key, value in report.items() if not isinstance(value, (dict, list))}
    pd.DataFrame([scalar]).to_csv(evaluation_dir / "final_patient_metrics.csv", index=False)
    _predictions_frame(result["table"]).to_csv(evaluation_dir / "patient_predictions.csv", index=False)
    pd.DataFrame([json_ready(result["hourly_metrics"])]).to_csv(evaluation_dir / "hourly_metrics.csv", index=False)
    save_confusion_matrix(
        evaluation_dir / "patient_confusion_matrix.png",
        result["table"]["has_sepsis"],
        result["table"]["patient_prediction"],
    )
    with (evaluation_dir / "patient_decisions.jsonl").open("w", encoding="utf-8") as target:
        for record in result["decisions"]:
            target.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
    preferred = ["Patient_ID", "Hour", "SepsisLabel", "logit_0", "logit_1", "raw_probability", "calibrated_probability"]
    columns = [column for column in preferred if column in result["frame"].columns]
    result["frame"][columns].to_csv(evaluation_dir / "test_scores.csv", index=False)
