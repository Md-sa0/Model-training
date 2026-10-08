from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qwen_sepsis.data import FEATURES, DYNAMIC, REQUIRED, validate

DEFAULT_RAW_PATH = Path("data/raw/sepsis_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/dataset_audit")
TEMPORAL_WINDOWS = (1, 3, 6)

FEATURE_GROUPS: dict[str, list[str]] = {
    "A": ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"],
    "B": ["WBC", "Lactate", "Creatinine", "Platelets"],
    "C": ["Age", "Gender"],
    "D": ["ICULOS"],
    "E": list(FEATURES),
    "F": [
        *FEATURES,
        "HR_delta_1h",
        "HR_delta_3h",
        "HR_delta_6h",
        "O2Sat_delta_1h",
        "O2Sat_delta_3h",
        "O2Sat_delta_6h",
        "Temp_delta_1h",
        "Temp_delta_3h",
        "Temp_delta_6h",
    ],
}

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "HR": (30.0, 220.0),
    "O2Sat": (50.0, 100.0),
    "Temp": (32.0, 45.0),
    "SBP": (40.0, 220.0),
    "MAP": (20.0, 200.0),
    "DBP": (20.0, 140.0),
    "Resp": (5.0, 80.0),
    "WBC": (1.0, 50.0),
    "Lactate": (0.25, 20.0),
    "Creatinine": (0.2, 12.0),
    "Platelets": (20.0, 800.0),
    "Age": (0.0, 120.0),
    "Gender": (0.0, 1.0),
    "ICULOS": (0.0, 100.0),
}


def build_feature_groups() -> dict[str, list[str]]:
    return {key: list(value) for key, value in FEATURE_GROUPS.items()}


def normalize_patient_ids(frame: pd.DataFrame) -> pd.Series:
    return frame["Patient_ID"].map(lambda value: str(value).strip())


def compute_prevalence_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if "SepsisLabel" not in frame.columns:
        raise ValueError("SepsisLabel ausente")
    if "Patient_ID" not in frame.columns:
        raise ValueError("Patient_ID ausente")
    positive_rows = int(frame["SepsisLabel"].astype(int).sum())
    negative_rows = int((1 - frame["SepsisLabel"].astype(int)).sum())
    patient_positive = (
        frame.groupby("Patient_ID", sort=False)["SepsisLabel"].max().astype(int)
    )
    patient_positive_count = int(patient_positive.sum())
    patient_negative_count = int((patient_positive == 0).sum())
    rows = int(len(frame))
    patients = int(frame["Patient_ID"].nunique())
    patient_prevalence = patient_positive_count / patients if patients else 0.0
    return {
        "rows": rows,
        "patients": patients,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "positive_observations": positive_rows,
        "negative_observations": negative_rows,
        "positive_prevalence_observation": float(positive_rows / rows) if rows else 0.0,
        "positive_prevalence_patient": patient_prevalence,
        "negative_prevalence_patient": (patient_negative_count / patients) if patients else 0.0,
        "positive_patients": patient_positive_count,
        "negative_patients": patient_negative_count,
    }


def compute_missingness(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, dict[str, float | int]]:
    target = list(columns) if columns is not None else list(frame.columns)
    output: dict[str, dict[str, float | int]] = {}
    for column in target:
        series = frame[column]
        missing_count = int(series.isna().sum())
        output[column] = {
            "missing_count": missing_count,
            "missing_pct": float(missing_count / len(frame) * 100.0) if len(frame) else 0.0,
            "non_missing_count": int(len(frame) - missing_count),
        }
    return output


def summarise_distribution(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, dict[str, Any]]:
    if columns is None:
        columns = [column for column in FEATURES if column in frame.columns]
    output: dict[str, dict[str, Any]] = {}
    for column in columns:
        output[column] = summarize_numeric_feature(frame[column], column)
    return output


def summarize_numeric_feature(series: pd.Series, feature_name: str) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    stats = {
        "feature": feature_name,
        "count": int(valid.count()),
        "missing_count": int(numeric.isna().sum()),
        "missing_pct": float(numeric.isna().mean() * 100.0) if len(numeric) else 0.0,
        "mean": float(valid.mean()) if not valid.empty else np.nan,
        "median": float(valid.median()) if not valid.empty else np.nan,
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "min": float(valid.min()) if not valid.empty else np.nan,
        "max": float(valid.max()) if not valid.empty else np.nan,
        "p01": float(valid.quantile(0.01)) if not valid.empty else np.nan,
        "p05": float(valid.quantile(0.05)) if not valid.empty else np.nan,
        "p25": float(valid.quantile(0.25)) if not valid.empty else np.nan,
        "p50": float(valid.quantile(0.50)) if not valid.empty else np.nan,
        "p75": float(valid.quantile(0.75)) if not valid.empty else np.nan,
        "p95": float(valid.quantile(0.95)) if not valid.empty else np.nan,
        "p99": float(valid.quantile(0.99)) if not valid.empty else np.nan,
    }
    return stats


def _cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    var_a = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    var_b = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (mean_a - mean_b) / pooled


def compare_positive_vs_negative(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        raise KeyError(f"Coluna ausente: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    positive = values[frame["SepsisLabel"].astype(int) == 1]
    negative = values[frame["SepsisLabel"].astype(int) == 0]
    diff_means = float(positive.mean() - negative.mean()) if positive.notna().any() and negative.notna().any() else np.nan
    diff_medians = float(positive.median() - negative.median()) if positive.notna().any() and negative.notna().any() else np.nan
    corr = float(values.corr(frame["SepsisLabel"].astype(float))) if values.notna().all() and frame["SepsisLabel"].notna().all() else np.nan
    return {
        "feature": column,
        "positive_mean": float(positive.mean()) if positive.notna().any() else np.nan,
        "negative_mean": float(negative.mean()) if negative.notna().any() else np.nan,
        "positive_median": float(positive.median()) if positive.notna().any() else np.nan,
        "negative_median": float(negative.median()) if negative.notna().any() else np.nan,
        "mean_difference": diff_means,
        "median_difference": diff_medians,
        "cohens_d": float(_cohens_d(positive, negative)) if positive.notna().any() and negative.notna().any() else np.nan,
        "corr_with_label": corr,
    }


def compute_feature_separation(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    target = columns or [column for column in FEATURES if column in frame.columns]
    results = {column: compare_positive_vs_negative(frame, column) for column in target}
    return results


def _rolling_window(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=1)
    return rolling.mean()


def build_temporal_features(frame: pd.DataFrame, feature_name: str, window_hours: tuple[int, ...] = TEMPORAL_WINDOWS) -> pd.DataFrame:
    work = frame.sort_values(["Patient_ID", "Hour"], kind="stable").copy()
    if feature_name not in work.columns:
        raise KeyError(f"Feature temporal ausente: {feature_name}")
    output = work.copy()
    output[f"{feature_name}_delta_1h"] = output.groupby("Patient_ID", sort=False)[feature_name].transform(
        lambda s: s.diff(1)
    )
    for window in window_hours:
        if window <= 1:
            continue
        shifted = output.groupby("Patient_ID", sort=False)[feature_name].transform(
            lambda s: s.shift(window)
        )
        output[f"{feature_name}_delta_{window}h"] = output[feature_name].sub(shifted)
        output[f"{feature_name}_rolling_mean_{window}h"] = output.groupby("Patient_ID", sort=False)[feature_name].transform(
            lambda s: _rolling_window(s, window)
        )
        output[f"{feature_name}_rolling_std_{window}h"] = output.groupby("Patient_ID", sort=False)[feature_name].transform(
            lambda s: s.rolling(window=window, min_periods=1).std(ddof=1)
        )
        output[f"{feature_name}_rolling_min_{window}h"] = output.groupby("Patient_ID", sort=False)[feature_name].transform(
            lambda s: s.rolling(window=window, min_periods=1).min()
        )
        output[f"{feature_name}_rolling_max_{window}h"] = output.groupby("Patient_ID", sort=False)[feature_name].transform(
            lambda s: s.rolling(window=window, min_periods=1).max()
        )
    output[f"{feature_name}_delta_1h"] = output[f"{feature_name}_delta_1h"].where(output["Hour"] > 0, np.nan)
    if "Hour" in output.columns:
        output[f"{feature_name}_trend_6h"] = output.groupby("Patient_ID", sort=False)[feature_name].transform(
            lambda s: s.iloc[-1] - s.iloc[0] if len(s) > 1 else 0.0
        )
    return output


def build_temporal_features_for_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values(["Patient_ID", "Hour"], kind="stable").copy()
    for feature_name in DYNAMIC:
        work = build_temporal_features(work, feature_name)
    return work


def compute_hours_by_patient(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("Patient_ID", sort=False).size()


def compute_iculos_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    if "ICULOS" not in frame.columns:
        raise KeyError("ICULOS ausente")
    values = pd.to_numeric(frame["ICULOS"], errors="coerce")
    return summarize_numeric_feature(values, "ICULOS")


def compute_extreme_summary(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    target = columns or list(FEATURES)
    output: dict[str, Any] = {}
    for column in target:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if column in FEATURE_RANGES:
            lo, hi = FEATURE_RANGES[column]
            count = int(((values < lo) | (values > hi)).sum())
            output[column] = {"extreme_count": count, "extreme_pct": float(count / len(frame) * 100.0) if len(frame) else 0.0}
        else:
            output[column] = {"extreme_count": 0, "extreme_pct": 0.0}
    return output


def compute_split_distribution(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    positives = int(frame["SepsisLabel"].astype(int).sum())
    total = int(len(frame))
    negatives = total - positives
    return {
        "split": split_name,
        "rows": total,
        "positive": positives,
        "negative": negatives,
        "positive_pct": float(positives / total * 100.0) if total else 0.0,
        "negative_pct": float(negatives / total * 100.0) if total else 0.0,
        "prevalence": float(positives / total) if total else 0.0,
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


def build_dataset_audit_report(raw_path: Path = DEFAULT_RAW_PATH) -> dict[str, Any]:
    frame = pd.read_csv(raw_path)
    frame, invalid_rows = validate(frame)
    frame = frame.sort_values(["Patient_ID", "Hour"], kind="stable").reset_index(drop=True)
    prevalence = compute_prevalence_summary(frame)
    missingness = compute_missingness(frame, list(FEATURES))
    feature_stats = summarise_distribution(frame, list(FEATURES))
    separation = compute_feature_separation(frame, list(FEATURES))
    extreme_summary = compute_extreme_summary(frame, list(FEATURES))
    hours_by_patient = compute_hours_by_patient(frame)
    hours_summary = {
        "distribution": dict(hours_by_patient.value_counts().sort_index().astype(int).to_dict()),
        "median": float(hours_by_patient.median()),
        "mean": float(hours_by_patient.mean()),
        "min": int(hours_by_patient.min()),
        "max": int(hours_by_patient.max()),
    }
    iculos_stats = compute_iculos_distribution(frame)
    return {
        "source_path": str(raw_path),
        "rows": int(len(frame)),
        "patients": int(frame["Patient_ID"].nunique()),
        "hours_observations": int(len(frame)),
        "invalid_rows": invalid_rows,
        "prevalence": prevalence,
        "missingness": missingness,
        "feature_stats": feature_stats,
        "feature_separation": separation,
        "extreme_summary": extreme_summary,
        "hours_by_patient": hours_summary,
        "iculos_distribution": iculos_stats,
        "feature_groups": build_feature_groups(),
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    prevalence = report["prevalence"]
    rows = report["rows"]
    positive_pct = prevalence["positive_prevalence_observation"] * 100.0
    patient_pct = prevalence["positive_prevalence_patient"] * 100.0
    missing_lines = []
    for column, values in report["missingness"].items():
        missing_lines.append(f"- {column}: {values['missing_count']} ausentes ({values['missing_pct']:.2f}%)")
    stats_lines = []
    for column in FEATURES:
        stat = report["feature_stats"][column]
        stats_lines.append(f"- {column}: média={stat['mean']:.2f}, mediana={stat['median']:.2f}, p05={stat['p05']:.2f}, p95={stat['p95']:.2f}, missing={stat['missing_pct']:.2f}%")
    md = f"""# Dataset audit report

## Overview

- Total de linhas: {report['rows']}
- Total de pacientes: {report['patients']}
- Observações positivas: {prevalence['positive_rows']}
- Observações negativas: {prevalence['negative_rows']}
- Prevalência por observação: {positive_pct:.2f}%
- Prevalência por paciente: {patient_pct:.2f}%

## Missingness

{chr(10).join(missing_lines)}

## Feature summary

{chr(10).join(stats_lines)}

## Notes

- O objetivo desta auditoria é documentar o comportamento atual do dataset sem alterar dados, pesos, calibrador ou pipeline de produção.
- Esta etapa é de investigação, não de retraining.
"""
    output_path.write_text(md, encoding="utf-8")


def _write_plots(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.Series(["0", "1"], name="Classe")
    frame = pd.read_csv(report["source_path"])
    if "SepsisLabel" in frame.columns:
        counts = frame["SepsisLabel"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["0", "1"], [int(counts.get(0, 0)), int(counts.get(1, 0))], color=["#4C72B0", "#DD8452"])
        ax.set_title("Distribuição das classes")
        ax.set_ylabel("Quantidade")
        fig.tight_layout()
        fig.savefig(output_dir / "class_distribution.png", dpi=200)
        plt.close(fig)

    for feature_name in ["HR", "Temp", "MAP", "O2Sat", "Lactate"]:
        if feature_name in frame.columns:
            positive = frame.loc[frame["SepsisLabel"] == 1, feature_name]
            negative = frame.loc[frame["SepsisLabel"] == 0, feature_name]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(negative.dropna(), bins=25, alpha=0.7, label="classe 0", color="#4C72B0")
            ax.hist(positive.dropna(), bins=25, alpha=0.7, label="classe 1", color="#DD8452")
            ax.set_title(f"Distribuição de {feature_name}")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Frequência")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"feature_{feature_name}.png", dpi=200)
            plt.close(fig)

    missing = pd.DataFrame(
        [{"feature": feature, "missing_pct": report["missingness"][feature]["missing_pct"]} for feature in FEATURES]
    )
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(missing["feature"], missing["missing_pct"], color="#4C72B0")
        ax.set_title("Missingness por feature")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Missing (%)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "missingness.png", dpi=200)
        plt.close(fig)


def run_audit(raw_path: Path = DEFAULT_RAW_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    report = build_dataset_audit_report(raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_audit_report.json"
    json_path.write_text(json.dumps(_safe_json(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, output_dir / "dataset_audit_report.md")
    _write_plots(report, output_dir)
    return report


def main() -> None:
    run_audit(DEFAULT_RAW_PATH, DEFAULT_OUTPUT_DIR)
    print(json.dumps({"status": "ok", "output_dir": str(DEFAULT_OUTPUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
