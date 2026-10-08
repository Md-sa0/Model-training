from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qwen_sepsis.model_quality import (
    aggregate_patient_scores,
    build_model_quality_report,
    compute_patient_aggregation_metrics,
    compute_threshold_analysis,
    summarize_raw_score_distribution,
)


def save_distribution_plot(frame: pd.DataFrame, output_path: Path) -> None:
    positive = frame.loc[frame["SepsisLabel"] == 1, "raw_probability"].to_numpy(dtype=float)
    negative = frame.loc[frame["SepsisLabel"] == 0, "raw_probability"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(negative, bins=30, alpha=0.7, label="classe 0", color="tab:blue")
    ax.hist(positive, bins=30, alpha=0.7, label="classe 1", color="tab:orange")
    ax.set_xlabel("P(class=1)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de scores brutos por classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curve(validation: pd.DataFrame, test: pd.DataFrame, output_path: Path) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, frame in (("validation", validation), ("test", test)):
        y_true = frame["SepsisLabel"].to_numpy(dtype=int)
        scores = frame["raw_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {frame['raw_probability'].mean() if False else 'n/a'})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_pr_curve(validation: pd.DataFrame, test: pd.DataFrame, output_path: Path) -> None:
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, frame in (("validation", validation), ("test", test)):
        y_true = frame["SepsisLabel"].to_numpy(dtype=int)
        scores = frame["raw_probability"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    project_root = ROOT
    artifact_dir = project_root / "artifacts"
    quality_dir = artifact_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    validation_path = artifact_dir / "calibration" / "validation_scores.csv"
    test_path = artifact_dir / "evaluation" / "test_scores.csv"
    config_path = artifact_dir / "calibration" / "config.json"
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError("Os scores de validação e teste não foram encontrados em artifacts/. Execute a calibração antes desta auditoria.")

    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    current_threshold = float(config.get("threshold", 0.0))

    score_summary = summarize_raw_score_distribution(validation, score_column="raw_probability", label_column="SepsisLabel")
    distribution_df = pd.DataFrame(
        [
            {"label": key, **values}
            for key, values in score_summary.items()
        ]
    )
    distribution_df.to_csv(quality_dir / "raw_score_distribution.csv", index=False)
    save_distribution_plot(validation, quality_dir / "raw_score_distribution.png")

    validation_analysis = compute_threshold_analysis(
        validation["SepsisLabel"].to_numpy(dtype=int),
        validation["raw_probability"].to_numpy(dtype=float),
    )
    test_analysis = compute_threshold_analysis(
        test["SepsisLabel"].to_numpy(dtype=int),
        test["raw_probability"].to_numpy(dtype=float),
    )
    validation_analysis.to_csv(quality_dir / "validation_threshold_analysis.csv", index=False)
    test_analysis.to_csv(quality_dir / "test_threshold_analysis.csv", index=False)

    report = build_model_quality_report(validation, test, current_threshold=current_threshold)
    (quality_dir / "model_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    save_roc_curve(validation, test, quality_dir / "roc_curve.png")
    save_pr_curve(validation, test, quality_dir / "pr_curve.png")

    patient_scores = aggregate_patient_scores(test, score_column="raw_probability")
    patient_scores.to_csv(quality_dir / "patient_aggregation.csv", index=False)
    patient_metrics = compute_patient_aggregation_metrics(test, score_column="raw_probability")
    patient_metrics_path = quality_dir / "patient_aggregation_metrics.json"
    patient_metrics_path.write_text(json.dumps(patient_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    validation_auroc = report["validation"]["auroc"]
    validation_auprc = report["validation"]["auprc"]
    test_auroc = report["test"]["auroc"]
    test_auprc = report["test"]["auprc"]
    positive_prevalence = report["positive_prevalence"]

    if validation_auroc >= 0.8 and validation_auprc >= max(0.2, positive_prevalence * 2):
        category = "GOOD"
    elif validation_auroc >= 0.65 and validation_auprc >= max(0.1, positive_prevalence * 1.5):
        category = "MODERATE"
    elif validation_auroc >= 0.55:
        category = "POOR"
    else:
        category = "INSUFFICIENT"

    main_finding = (
        "O modelo apresenta separação limitada entre classes quando se considera a distribuição dos scores brutos "
        "e o trade-off sensibilidade/especificidade. O limiar atual produz sensibilidade máxima apenas em troca de "
        "especificidade agressivamente baixa."
    )

    markdown = f"""# Model Quality Report

## Summary

- Validation AUROC: {validation_auroc:.4f}
- Validation AUPRC: {validation_auprc:.4f}
- Test AUROC: {test_auroc:.4f}
- Test AUPRC: {test_auprc:.4f}
- Positive prevalence: {positive_prevalence:.4f}
- Current threshold: {current_threshold:.2f}
- Best Youden threshold: {report['best_youden_threshold']['threshold']:.2f}
- Best F1 threshold: {report['best_f1_threshold']['threshold']:.2f}
- Best MCC threshold: {report['best_mcc_threshold']['threshold']:.2f}
- Classification: {category}

## Main finding

{main_finding}

## Validation score distribution

- Mean positive score: {report['validation']['mean_positive_score']:.4f}
- Mean negative score: {report['validation']['mean_negative_score']:.4f}
- Median positive score: {report['score_discrimination']['validation_median_positive_score']:.4f}
- Median negative score: {report['score_discrimination']['validation_median_negative_score']:.4f}

## Decision analysis

- Current sensitivity: {report['current_validation_metrics']['sensitivity']:.4f}
- Current specificity: {report['current_validation_metrics']['specificity']:.4f}
- Current FP: {report['current_validation_metrics']['FP']}
- Current FN: {report['current_validation_metrics']['FN']}

## Next step

A próxima etapa recomendada é definir uma política clínica e operacional de alertas antes de qualquer alteração de pipeline, porque a capacidade discriminativa real deve ser interpretada em conjunto com AUROC, AUPRC, taxa de falsos positivos e separação dos scores.
"""
    (quality_dir / "model_quality_report.md").write_text(markdown, encoding="utf-8")

    print("=== QWEN MODEL QUALITY ===")
    print(f"Validation: AUROC={validation_auroc:.4f}, AUPRC={validation_auprc:.4f}")
    print(f"Test: AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f}")
    print(f"Positive prevalence={positive_prevalence:.4f}")
    print(f"Current threshold={current_threshold:.2f}")
    print(f"Current validation sensitivity={report['current_validation_metrics']['sensitivity']:.4f}")
    print(f"Current validation specificity={report['current_validation_metrics']['specificity']:.4f}")
    print(f"Best Youden threshold={report['best_youden_threshold']['threshold']:.2f}")
    print(f"Best F1 threshold={report['best_f1_threshold']['threshold']:.2f}")
    print(f"Best MCC threshold={report['best_mcc_threshold']['threshold']:.2f}")
    print(f"Classification={category}")


if __name__ == "__main__":
    main()
