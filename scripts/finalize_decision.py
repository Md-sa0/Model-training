"""Freeze the existing validation decision and regenerate final reports from saved scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qwen_sepsis.calibration import binary_metrics, load_calibrator, select_threshold
from qwen_sepsis.data import load_prepared_split
from qwen_sepsis.patient_evaluation import build_patient_table, patient_level_metrics
from qwen_sepsis.pipeline import (
    build_final_report,
    evaluate_test_frame,
    json_ready,
    write_evaluation_artifacts,
)
from qwen_sepsis.plotting import save_confusion_matrix, save_threshold_curve


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Patient_ID": str})
    required = {"Patient_ID", "Hour", "SepsisLabel", "raw_probability", "calibrated_probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")
    return frame


def _hourly_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    labels = frame["SepsisLabel"].astype(int).to_numpy()
    scores = frame["calibrated_probability"].astype(float).to_numpy()
    metrics = binary_metrics(labels, scores >= threshold)
    return {
        **metrics,
        "n_observations": int(len(frame)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    artifacts = ROOT / "artifacts"
    calibration_dir = artifacts / "calibration"
    decision_dir = artifacts / "decision"
    config_path = calibration_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibrator = load_calibrator(calibration_dir / "calibrator.pkl")
    if calibrator.method != config["method"] or config.get("fitted_on") != "validation":
        raise RuntimeError("O calibrador salvo não está bloqueado e associado à validation.")
    if config.get("test_used_for_fitting") is not False:
        raise RuntimeError("O calibrador salvo registra uso do test no ajuste.")

    validation = _scores(calibration_dir / "validation_scores.csv")
    recalculated = calibrator.predict(validation["raw_probability"].to_numpy(dtype=float))
    saved = validation["calibrated_probability"].to_numpy(dtype=float)
    if not np.allclose(recalculated, saved, rtol=0.0, atol=1e-10):
        raise RuntimeError("Scores salvos não correspondem ao calibrador congelado.")

    comparison = json.loads((calibration_dir / "comparison.json").read_text(encoding="utf-8"))
    preference = {"raw": 0, "platt": 1, "isotonic": 2}
    selected_calibrator = min(
        comparison,
        key=lambda name: (
            comparison[name]["brier"], comparison[name]["ece"],
            comparison[name]["log_loss"], preference[name],
        ),
    )
    if selected_calibrator != calibrator.method:
        raise RuntimeError("O calibrador existente não é o melhor pela regra congelada na validation.")

    threshold_info, threshold_table = select_threshold(
        validation["SepsisLabel"].astype(int).to_numpy(),
        validation["calibrated_probability"].astype(float).to_numpy(),
        target_sensitivity=float(config.get("target_sensitivity", 0.95)),
        split="validation",
        thresholds=np.round(np.arange(1, 101) / 100, 2),
    )
    threshold = float(threshold_info["threshold"])
    if threshold <= 0.0:
        raise RuntimeError("Threshold operacional precisa ser maior que zero.")
    selected_row = threshold_table.loc[threshold_table["threshold"] == threshold].iloc[0]
    hourly = _hourly_metrics(validation, threshold)
    validation_zero_hourly = _hourly_metrics(validation, 0.0)
    threshold_message = (
        "Nenhum threshold atende ao requisito mínimo de sensibilidade. "
        f"Melhor compromisso encontrado: threshold={threshold:.2f}, "
        f"sensibilidade={selected_row['sensitivity']:.4f}, "
        f"especificidade={selected_row['specificity']:.4f}."
        if not threshold_info["target_met"] else threshold_info["message"]
    )
    threshold_table["patient_sensitivity"] = np.nan
    threshold_table["patient_specificity"] = np.nan
    threshold_table["patient_PPV"] = np.nan
    threshold_table["patient_NPV"] = np.nan
    threshold_table["patient_MCC"] = np.nan
    threshold_table["patient_TP"] = np.nan
    threshold_table["patient_TN"] = np.nan
    threshold_table["patient_FP"] = np.nan
    threshold_table["patient_FN"] = np.nan
    for index, row in threshold_table.iterrows():
        patient_metrics = patient_level_metrics(build_patient_table(validation, float(row["threshold"]), "any_alert"))
        for source, target in (
            ("sensitivity", "patient_sensitivity"), ("specificity", "patient_specificity"),
            ("ppv", "patient_PPV"), ("npv", "patient_NPV"), ("mcc", "patient_MCC"),
            ("true_positives", "patient_TP"), ("true_negatives", "patient_TN"),
            ("false_positives", "patient_FP"), ("false_negatives", "patient_FN"),
        ):
            threshold_table.loc[index, target] = patient_metrics[source]

    validation_patient_table = build_patient_table(validation, threshold, "any_alert")
    validation_patient_metrics = patient_level_metrics(validation_patient_table)
    validation_zero_patient_metrics = patient_level_metrics(build_patient_table(validation, 0.0, "any_alert"))
    risk_levels = {
        "low_below": threshold,
        "high_at_or_above": threshold,
        "moderate_band": False,
        "note": "Classificação binária escolhida; não há segundo corte validado na validation.",
    }
    config.update({
        "threshold": threshold,
        "threshold_locked": True,
        "calibrator_locked": True,
        "test_used_for_fitting": False,
        "retraining": False,
        "selection_unit": "hourly_observation",
        "aggregation_strategy": "any_alert",
        "risk_levels": risk_levels,
        "threshold_message": threshold_message,
        "target_met": bool(threshold_info["target_met"]),
        "achieved_sensitivity_validation_hourly": float(selected_row["sensitivity"]),
        "achieved_specificity_validation_hourly": float(selected_row["specificity"]),
        "validation_patient_metrics": validation_patient_metrics,
        "validation_hourly_metrics": hourly,
        "threshold_zero_validation_hourly": validation_zero_hourly,
        "threshold_zero_validation_patient": validation_zero_patient_metrics,
        "threshold_selection_rule": (
            "Na validation, usar as observações do SepsisLabel horário, exigir sensibilidade >=95%; "
            "maximizar especificidade; desempatar por MCC, PPV, menos FP e maior threshold. "
            "Se nenhum threshold atingir a meta, maximizar sensibilidade e registrar o compromisso. "
            "A avaliação patient-level é reportada separadamente com any_alert."
        ),
        "risk_level_selection": "Binário: baixo risco abaixo do threshold e risco elevado a partir dele; sem T2 validado.",
    })
    _write_json(config_path, config)

    decision_dir.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(decision_dir / "threshold_comparison.csv", index=False)
    save_threshold_curve(
        decision_dir / "threshold_curve.png", threshold_table, threshold,
        float(config.get("target_sensitivity", 0.95)), evaluation_unit="observação horária",
    )
    save_confusion_matrix(
        decision_dir / "confusion_matrix_validation.png",
        validation_patient_table["has_sepsis"], validation_patient_table["patient_prediction"],
        title="Matriz de confusão por paciente na validation",
    )

    # The test scores are opened only after both locked states have been persisted.
    if config.get("calibrator_locked") is not True or config.get("threshold_locked") is not True:
        raise RuntimeError("Calibrador e threshold precisam estar congelados antes do test.")
    test_scores = _scores(artifacts / "evaluation" / "test_scores.csv")
    test_data = load_prepared_split(ROOT / "data" / "processed", "test")
    test_frame = test_data.merge(
        test_scores[["Patient_ID", "Hour", "raw_probability"]],
        on=["Patient_ID", "Hour"], how="left", validate="one_to_one",
    )
    if test_frame["raw_probability"].isna().any():
        raise RuntimeError("Scores do test não estão alinhados ao split de pacientes salvo.")
    result = evaluate_test_frame(test_frame, calibrator, config)
    context = {
        "model_id": config.get("model_id", "Qwen/Qwen3-8B"),
        "base_model_path": config.get("base_model_path"),
        "adapter_path": config.get("adapter_path"),
        "dataset": config.get("dataset"),
        "seed": config.get("seed"),
        "tokens": config.get("tokens"),
        "time_axis": config.get("time_axis", "Hour"),
        "system_role": config.get("system_role"),
        "retraining": False,
    }
    report = build_final_report(result["metrics"], config, context, result["hourly_metrics"])
    report["threshold_locked"] = True
    report["retraining"] = False
    write_evaluation_artifacts(artifacts, result, report)
    save_confusion_matrix(
        decision_dir / "confusion_matrix_test.png",
        result["table"]["has_sepsis"], result["table"]["patient_prediction"],
        title="Matriz de confusão por paciente no test",
    )

    calibration_metrics_by_method = {
        name: {key: comparison[name][key] for key in ("brier", "ece", "log_loss")}
        for name in ("raw", "platt", "isotonic")
    }
    decision_report = {
        "model": context,
        "calibration": {
            "metrics_validation_in_sample": calibration_metrics_by_method,
            "selected": calibrator.method,
            "locked": True,
            "fitted_on": "validation",
            "test_used_for_fitting": False,
            "selection_rule": config["calibration_selection_rule"],
        },
        "threshold": {
            "value": threshold,
            "locked": True,
            "selection_unit": "hourly_observation",
            "strategy": "any_alert",
            "target_sensitivity": float(config["target_sensitivity"]),
            "target_met": bool(threshold_info["target_met"]),
            "selection_rule": config["threshold_selection_rule"],
            "validation_patient": validation_patient_metrics,
            "validation_hourly": hourly,
            "threshold_zero_validation_hourly": validation_zero_hourly,
            "threshold_zero_validation_patient": validation_zero_patient_metrics,
            "message": threshold_message,
            "target_met": bool(threshold_info["target_met"]),
        },
        "test_final": {
            "patient_level": result["metrics"],
            "hourly": result["hourly_metrics"],
            "test_used_for_fitting": False,
        },
        "risk_levels": {
            "three_level_supported": False,
            "classification": "binary",
            "low_risk_below": threshold,
            "elevated_risk_at_or_above": threshold,
            "reason": "A validation não sustenta um segundo corte; manter classificação binária evita limites arbitrários.",
        },
        "controls": {
            "calibrator_locked": True,
            "threshold_locked": True,
            "test_used_for_fitting": False,
            "retraining": False,
        },
        "clinical_note": (
            "Sistema de suporte à decisão para estimativa de risco de desenvolvimento de sepse. "
            "O resultado não constitui diagnóstico médico."
        ),
    }
    _write_json(decision_dir / "decision_threshold_report.json", decision_report)

    selected_levels = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    markdown_rows = threshold_table[
        threshold_table["threshold"].isin(selected_levels + [threshold])
    ].drop_duplicates("threshold").sort_values("threshold")
    lines = [
        "# Relatório de decisão de risco de sepse",
        "",
        f"Modelo: `{context['model_id']}`; adapter: `{context['adapter_path']}`.",
        "Nenhum retraining foi executado. O calibrador isotônico salvo foi reutilizado sem ajuste.",
        "",
        "## Calibração na validation",
        "",
        "| Método | Brier | ECE | Log Loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in calibration_metrics_by_method.items():
        lines.append(f"| {name} | {values['brier']:.6f} | {values['ece']:.6f} | {values['log_loss']:.6f} |")
    lines.extend([
        "",
        f"Calibrador escolhido: **{calibrator.method}** (congelado). Métricas de calibração são in-sample na validation.",
        "",
        "## Threshold horário na validation",
        "",
        config["threshold_selection_rule"],
        "",
        f"Threshold: **{threshold:.2f}**. Sensibilidade={selected_row['sensitivity']:.4f}; especificidade={selected_row['specificity']:.4f}; "
        f"PPV={selected_row['PPV']:.4f}; NPV={selected_row['NPV']}; FP={int(selected_row['FP'])}; "
        f"FN={int(selected_row['FN'])}; MCC={selected_row['MCC']:.4f}.",
        "",
        (
            "Nenhum threshold atende ao requisito mínimo de sensibilidade. "
            "Foi selecionado o melhor compromisso encontrado."
            if not threshold_info["target_met"] else ""
        ),
        f"Baseline validation em threshold 0.00: sensibilidade horária={validation_zero_hourly['sensitivity']:.4f}, "
        f"especificidade horária={validation_zero_hourly['specificity']:.4f}, FP={validation_zero_hourly['FP']}, "
        f"FN={validation_zero_hourly['FN']}; por paciente, sensibilidade={validation_zero_patient_metrics['sensitivity']:.4f}, "
        f"especificidade={validation_zero_patient_metrics['specificity']:.4f}, FP={validation_zero_patient_metrics['false_positives']}, "
        f"FN={validation_zero_patient_metrics['false_negatives']}.",
        "",
        "| Threshold | Sensibilidade | Especificidade | PPV | NPV | FP | FN | MCC |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in markdown_rows.iterrows():
        lines.append(
            f"| {row['threshold']:.2f} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{row['PPV']:.4f} | {row['NPV'] if pd.notna(row['NPV']) else 'N/A'} | "
            f"{int(row['FP'])} | {int(row['FN'])} | {row['MCC']:.4f} |"
        )
    lines.extend([
        "",
        f"Resultado patient-level na validation com `any_alert`: {validation_patient_metrics['n_patients']} pacientes, "
        f"sensibilidade={validation_patient_metrics['sensitivity']:.4f}, especificidade={validation_patient_metrics['specificity']:.4f}, "
        f"PPV={validation_patient_metrics['ppv']:.4f}, NPV={validation_patient_metrics['npv']}, "
        f"TP={validation_patient_metrics['true_positives']}, TN={validation_patient_metrics['true_negatives']}, "
        f"FP={validation_patient_metrics['false_positives']}, FN={validation_patient_metrics['false_negatives']}.",
    ])
    lines.extend([
        "",
        "### Resultado patient-level no test",
        "",
        "| N pacientes | Positivos | Negativos | Sensibilidade | Especificidade | PPV | NPV | F1 | MCC | TP | TN | FP | FN | AUROC | AUPRC |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {result['metrics']['n_patients']} | {result['metrics']['n_sepsis_patients']} | "
        f"{result['metrics']['n_patients'] - result['metrics']['n_sepsis_patients']} | "
        f"{result['metrics']['sensitivity']:.4f} | {result['metrics']['specificity']:.4f} | "
        f"{result['metrics']['ppv']:.4f} | {result['metrics']['npv']} | {result['metrics']['f1']:.4f} | "
        f"{result['metrics']['mcc']:.4f} | "
        f"{result['metrics']['true_positives']} | {result['metrics']['true_negatives']} | "
        f"{result['metrics']['false_positives']} | {result['metrics']['false_negatives']} | "
        f"{result['metrics']['auroc']:.4f} | {result['metrics']['auprc']:.4f} |",
        "",
        "Controles: calibrator_locked=true; threshold_locked=true; test_used_for_fitting=false; retraining=false.",
        "",
        "Decisão em dois níveis: baixo risco abaixo do threshold e risco elevado a partir dele. Não há base validada para T1/T2.",
        "Este sistema é suporte à decisão para estimativa de risco e não constitui diagnóstico médico.",
        "",
        "Ver `threshold_comparison.csv` para a busca completa em passos de 0,01 e os resultados horários complementares.",
    ])
    (decision_dir / "decision_threshold_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "calibrator": calibrator.method,
        "threshold": threshold,
        "validation_patient": validation_patient_metrics,
        "test_patient": result["metrics"],
        "controls": decision_report["controls"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()