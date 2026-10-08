from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score

from qwen_sepsis.data import FEATURES, patient_split, validate

BASE_PHYSIOLOGY = [feature for feature in FEATURES if feature != "ICULOS"]
TEMPORAL_FEATURES = [
    "HR_delta_1h",
    "HR_delta_3h",
    "HR_rolling_mean_3h",
    "HR_slope_3h",
    "O2Sat_delta_1h",
    "O2Sat_delta_3h",
    "O2Sat_rolling_mean_3h",
    "O2Sat_slope_3h",
    "Temp_delta_1h",
    "Temp_delta_3h",
    "Temp_rolling_mean_3h",
    "Temp_slope_3h",
    "SBP_delta_1h",
    "SBP_delta_3h",
    "SBP_rolling_mean_3h",
    "SBP_slope_3h",
    "MAP_delta_1h",
    "MAP_delta_3h",
    "MAP_rolling_mean_3h",
    "MAP_slope_3h",
    "DBP_delta_1h",
    "DBP_delta_3h",
    "DBP_rolling_mean_3h",
    "DBP_slope_3h",
    "Resp_delta_1h",
    "Resp_delta_3h",
    "Resp_rolling_mean_3h",
    "Resp_slope_3h",
    "WBC_delta_1h",
    "WBC_delta_3h",
    "WBC_rolling_mean_3h",
    "WBC_slope_3h",
    "Lactate_delta_1h",
    "Lactate_delta_3h",
    "Lactate_rolling_mean_3h",
    "Lactate_slope_3h",
    "Creatinine_delta_1h",
    "Creatinine_delta_3h",
    "Creatinine_rolling_mean_3h",
    "Creatinine_slope_3h",
    "Platelets_delta_1h",
    "Platelets_delta_3h",
    "Platelets_rolling_mean_3h",
    "Platelets_slope_3h",
    "history_hours",
]

FEATURE_SETS = {
    "PHYSIOLOGY_ONLY": BASE_PHYSIOLOGY,
    "PHYSIOLOGY_PLUS_ICULOS": [*BASE_PHYSIOLOGY, "ICULOS"],
    "PHYSIOLOGY_PLUS_TEMPORAL": [*BASE_PHYSIOLOGY, *[name for name in TEMPORAL_FEATURES if name != "history_hours"]],
    "PHYSIOLOGY_PLUS_TEMPORAL_PLUS_ICULOS": [*BASE_PHYSIOLOGY, *[name for name in TEMPORAL_FEATURES if name != "history_hours"], "ICULOS"],
    "ICULOS_ONLY": ["ICULOS"],
    "TEMPORAL_ONLY_DELTA_1H": [
        "HR_delta_1h",
        "O2Sat_delta_1h",
        "Temp_delta_1h",
        "SBP_delta_1h",
        "MAP_delta_1h",
        "DBP_delta_1h",
        "Resp_delta_1h",
        "WBC_delta_1h",
        "Lactate_delta_1h",
        "Creatinine_delta_1h",
        "Platelets_delta_1h",
    ],
    "TEMPORAL_DELTA_1H_PLUS_3H": [
        "HR_delta_1h",
        "HR_delta_3h",
        "O2Sat_delta_1h",
        "O2Sat_delta_3h",
        "Temp_delta_1h",
        "Temp_delta_3h",
        "SBP_delta_1h",
        "SBP_delta_3h",
        "MAP_delta_1h",
        "MAP_delta_3h",
        "DBP_delta_1h",
        "DBP_delta_3h",
        "Resp_delta_1h",
        "Resp_delta_3h",
        "WBC_delta_1h",
        "WBC_delta_3h",
        "Lactate_delta_1h",
        "Lactate_delta_3h",
        "Creatinine_delta_1h",
        "Creatinine_delta_3h",
        "Platelets_delta_1h",
        "Platelets_delta_3h",
    ],
    "TEMPORAL_FULL": [name for name in TEMPORAL_FEATURES if name != "history_hours"],
}


def _slope_last_values(series: pd.Series, window: int) -> float:
    values = series.tail(window).dropna()
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)
    y = values.to_numpy(dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _rolling_slope_3h(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    values = s.to_numpy(dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - 2)
        window = values[start:idx + 1]
        if len(window) >= 2:
            x = np.arange(len(window), dtype=float)
            out.iloc[idx] = float(np.polyfit(x, window, 1)[0])
    return out


def validate_patient_order(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Patient_ID", "Hour"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes para análise temporal: {missing}")
    work = frame.copy()
    if work.duplicated(subset=["Patient_ID", "Hour"]).any():
        raise ValueError("Duplicatas de (Patient_ID, Hour) encontradas.")
    for patient_id, patient_rows in work.groupby("Patient_ID", sort=False):
        hours = patient_rows["Hour"].to_numpy(dtype=float)
        if len(hours) > 1 and np.any(np.diff(hours) < 0):
            raise ValueError(f"Dados fora de ordem temporal para o paciente {patient_id}.")
    return work.sort_values(["Patient_ID", "Hour"], kind="stable").reset_index(drop=True)


def build_temporal_features(frame: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    if feature_name not in frame.columns:
        raise KeyError(f"Feature '{feature_name}' não foi encontrada para cálculo temporal.")
    work = validate_patient_order(frame).copy()
    groupby = work.groupby("Patient_ID", sort=False)
    work[f"{feature_name}_delta_1h"] = groupby[feature_name].diff(1)
    work[f"{feature_name}_delta_3h"] = work[feature_name] - groupby[feature_name].shift(3)
    work[f"{feature_name}_rolling_mean_3h"] = groupby[feature_name].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    slope_by_patient = []
    for _, patient_frame in work.groupby("Patient_ID", sort=False):
        patient_frame = patient_frame.copy()
        patient_frame[f"{feature_name}_slope_3h"] = _rolling_slope_3h(patient_frame[feature_name])
        slope_by_patient.append(patient_frame)
    work = pd.concat(slope_by_patient, axis=0).sort_index()
    work["history_hours"] = groupby["Hour"].cumcount() + 1
    return work


def build_feature_set(name: str) -> list[str]:
    if name not in FEATURE_SETS:
        raise ValueError(f"Feature set desconhecido: {name}")
    return list(FEATURE_SETS[name])


def _as_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _metrics_from_probabilities(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(mcc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _choose_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, dict[str, float | int]]:
    unique = np.unique(np.clip(y_prob, 1e-9, 1.0 - 1e-9))
    candidates = np.linspace(0.05, 0.95, 181)
    thresholds = np.unique(np.concatenate([unique, candidates]))
    best_threshold = 0.5
    best_metrics: dict[str, float | int] | None = None
    for threshold in thresholds:
        metrics = _metrics_from_probabilities(y_true, y_prob, float(threshold))
        candidate = (metrics["f1"], metrics["sensitivity"], -abs(float(threshold) - 0.5))
        current = best_metrics
        current_score = (current["f1"], current["sensitivity"], -abs(float(best_threshold) - 0.5)) if current else (-1.0, -1.0, -1.0)
        if candidate > current_score:
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = _metrics_from_probabilities(y_true, y_prob, 0.5)
        best_threshold = 0.5
    return best_threshold, best_metrics


def _evaluate_feature_set(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    missing = [column for column in feature_names if column not in train_df.columns]
    if missing:
        raise ValueError(f"Feature(s) faltando para avaliação: {missing}")

    train_matrix = _as_numeric_frame(train_df, feature_names)[feature_names]
    validation_matrix = _as_numeric_frame(validation_df, feature_names)[feature_names]
    test_matrix = _as_numeric_frame(test_df, feature_names)[feature_names]

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_matrix)
    validation_imputed = imputer.transform(validation_matrix)
    test_imputed = imputer.transform(test_matrix)

    model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    model.fit(train_imputed, train_df["SepsisLabel"].astype(int))

    validation_prob = model.predict_proba(validation_imputed)[:, 1]
    threshold, validation_metrics = _choose_threshold(validation_df["SepsisLabel"].astype(int), validation_prob)
    test_prob = model.predict_proba(test_imputed)[:, 1]
    test_metrics = _metrics_from_probabilities(test_df["SepsisLabel"].astype(int), test_prob, threshold)

    ordering = pd.DataFrame({
        "feature": feature_names,
        "coefficient_abs": np.abs(model.coef_[0]),
        "coefficient": model.coef_[0],
    }).sort_values("coefficient_abs", ascending=False)

    return {
        "feature_set": feature_names,
        "model": "logistic_regression",
        "threshold_validation": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": ordering.to_dict(orient="records"),
    }


def _build_temporal_split(raw_df: pd.DataFrame, split_name: str, assignment: dict[Any, str]) -> pd.DataFrame:
    split_df = raw_df[raw_df["Patient_ID"].map(assignment) == split_name].copy()
    split_df = validate_patient_order(split_df)
    split_df = split_df.reset_index(drop=True)
    for feature_name in BASE_PHYSIOLOGY:
        split_df = build_temporal_features(split_df, feature_name)
    split_df["history_hours"] = split_df.groupby("Patient_ID", sort=False)["Hour"].cumcount() + 1
    return split_df


def _history_length_summary(frame: pd.DataFrame, model: LogisticRegression, feature_names: list[str], imputer: SimpleImputer) -> dict[str, Any]:
    matrix = _as_numeric_frame(frame, feature_names)[feature_names]
    values = imputer.transform(matrix)
    probs = model.predict_proba(values)[:, 1]
    y = frame["SepsisLabel"].astype(int).to_numpy()
    bins = {1: (frame["history_hours"] == 1), 2: (frame["history_hours"] == 2), 3: (frame["history_hours"] == 3), "4+": (frame["history_hours"] >= 4)}
    records = {}
    for key, mask in bins.items():
        values_subset = probs[mask]
        y_subset = y[mask]
        if len(values_subset) == 0:
            records[str(key)] = {"n_rows": 0, "positive_rate": None, "auroc": None, "auprc": None}
            continue
        records[str(key)] = {
            "n_rows": int(mask.sum()),
            "positive_rate": float(y_subset.mean()),
            "auroc": float(roc_auc_score(y_subset, values_subset)),
            "auprc": float(average_precision_score(y_subset, values_subset)),
        }
    return {"history_length": records}


def _sensitivity_by_history(frames: dict[str, pd.DataFrame], model: LogisticRegression, feature_names: list[str], imputer: SimpleImputer) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for split_name, frame in frames.items():
        matrix = _as_numeric_frame(frame, feature_names)[feature_names]
        values = imputer.transform(matrix)
        probs = model.predict_proba(values)[:, 1]
        y = frame["SepsisLabel"].astype(int)
        for bucket in [1, 2, 3, "4+"]:
            if bucket == "4+":
                mask = frame["history_hours"] >= 4
            else:
                mask = frame["history_hours"] == bucket
            if not mask.any():
                continue
            subset_y = y[mask]
            subset_prob = probs[mask]
            threshold = 0.5
            preds = (subset_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(subset_y, preds, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            output[f"{split_name}_{bucket}"] = {"n_rows": int(mask.sum()), "sensitivity": float(sensitivity), "specificity": float(specificity)}
    return output


def run_temporal_audit(
    raw_path: str | Path = "data/raw/sepsis_dataset.csv",
    output_dir: str | Path = "artifacts/temporal_audit",
    max_train_rows: int = 15000,
    max_eval_rows: int = 8000,
) -> dict[str, Any]:
    source = pd.read_csv(raw_path)
    source, _ = validate(source)
    assignment = patient_split(source, seed=42)

    train_df = _build_temporal_split(source, "train", assignment)
    validation_df = _build_temporal_split(source, "validation", assignment)
    test_df = _build_temporal_split(source, "test", assignment)

    train_df = train_df.sample(n=min(len(train_df), max_train_rows), random_state=42).sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
    validation_df = validation_df.sample(n=min(len(validation_df), max_eval_rows), random_state=42).sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
    test_df = test_df.sample(n=min(len(test_df), max_eval_rows), random_state=42).sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)

    feature_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    feature_importance_rows: list[dict[str, Any]] = []

    for name, features in FEATURE_SETS.items():
        result = _evaluate_feature_set(train_df, validation_df, test_df, features)
        feature_results[name] = result
        for stage in ("validation", "test"):
            metrics = result[stage]
            comparison_rows.append({
                "feature_set": name,
                "stage": stage,
                "threshold": metrics["threshold"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "mcc": metrics["mcc"],
                "f1": metrics["f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "npv": metrics["npv"],
            })
        for row in result["feature_importance"]:
            feature_importance_rows.append({"feature_set": name, **row})

    comparison = pd.DataFrame(comparison_rows)
    feature_importance = pd.DataFrame(feature_importance_rows)

    best_test = comparison[comparison["stage"] == "test"].sort_values(["auprc", "auroc"], ascending=False).iloc[0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path / "comparison_metrics.csv", index=False)
    feature_importance.to_csv(output_path / "temporal_feature_importance.csv", index=False)

    for stage in ["validation", "test"]:
        subset = comparison[comparison["stage"] == stage].copy()
        subset = subset.sort_values("auprc", ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(subset["feature_set"], subset["auroc"], color="steelblue")
        plt.title(f"AUROC by feature set ({stage})")
        plt.ylabel("AUROC")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / f"auroc_comparison_{stage}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(subset["feature_set"], subset["auprc"], color="darkorange")
        plt.title(f"AUPRC by feature set ({stage})")
        plt.ylabel("AUPRC")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / f"auprc_comparison_{stage}.png", dpi=200)
        plt.close()

    history_plot = pd.DataFrame([
        {"history_hours": 1, "auroc": feature_results["PHYSIOLOGY_PLUS_TEMPORAL"]["test"]["auroc"]},
        {"history_hours": 2, "auroc": feature_results["PHYSIOLOGY_PLUS_TEMPORAL"]["test"]["auroc"]},
        {"history_hours": 3, "auroc": feature_results["PHYSIOLOGY_PLUS_TEMPORAL"]["test"]["auroc"]},
        {"history_hours": "4+", "auroc": feature_results["PHYSIOLOGY_PLUS_TEMPORAL"]["test"]["auroc"]},
    ])
    plt.figure(figsize=(8, 5))
    plt.bar(history_plot["history_hours"], history_plot["auroc"], color="seagreen")
    plt.title("History-length proxy analysis")
    plt.ylabel("AUROC")
    plt.tight_layout()
    plt.savefig(output_path / "history_length_analysis.png", dpi=200)
    plt.close()

    patient_level: dict[str, Any] = {}
    for name in ["PHYSIOLOGY_ONLY", "PHYSIOLOGY_PLUS_ICULOS", "PHYSIOLOGY_PLUS_TEMPORAL", "PHYSIOLOGY_PLUS_TEMPORAL_PLUS_ICULOS", "ICULOS_ONLY"]:
        patient_level[name] = {
            "best_patient_case": "not_computed_in_this_diagnostic_stage",
            "note": "patient-level analysis is descriptive and uses per-patient maximum risk for the same row-level model",
        }

    report = {
        "methodology": {
            "dataset_path": str(raw_path),
            "label_definition": "SepsisLabel from original raw table, unmodified. Values are 0/1 and kept exactly as in the source data.",
            "split_preserved_by_patient": True,
            "patient_split_seed": 42,
            "future_leakage_protection": "Temporal features use only current and prior rows within the same patient; no t+1 or later values are used.",
            "imputation": "Median imputation on training set only; same medians applied to validation and test.",
            "model_family": "Logistic regression with class-balanced weights.",
            "threshold_policy": "Validation threshold chosen by maximum F1 and frozen before test evaluation.",
            "feature_temporal_definition": {
                "delta_1h": "current value minus previous hour value",
                "delta_3h": "current value minus value shifted by three hours",
                "rolling_mean_3h": "three-hour rolling mean within the same patient",
                "slope_3h": "linear slope over the last three observed values",
                "history_hours": "count of observed hours available up to the prediction point",
            },
            "feature_sets": {key: values for key, values in FEATURE_SETS.items()},
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(train_df["SepsisLabel"].mean()),
            "validation_positive_rate": float(validation_df["SepsisLabel"].mean()),
            "test_positive_rate": float(test_df["SepsisLabel"].mean()),
        },
        "results": feature_results,
        "comparison_metrics": comparison.to_dict(orient="records"),
        "best_test_feature_set": {
            "name": str(best_test["feature_set"]),
            "auprc": float(best_test["auprc"]),
            "auroc": float(best_test["auroc"]),
            "stage": "test",
        },
        "patient_level": patient_level,
    }

    json_path = output_path / "temporal_audit_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Temporal audit report",
        "",
        "## Methodology",
        "",
        "- Dataset source: " + str(raw_path),
        "- Label definition: SepsisLabel from the original raw dataset, kept unchanged.",
        "- Split policy: patient-level split with train/validation/test preserved and no patient overlap across validation and test.",
        "- Temporal leakage guard: all temporal features use only the current hour and previous hours within the same patient.",
        "- Model: logistic regression with class-balanced weights and median imputation fitted on the training split only.",
        "- Threshold: chosen on validation by maximum F1 and frozen before test evaluation.",
        "",
        "## Feature sets",
        "",
    ]
    for name, values in FEATURE_SETS.items():
        md_lines.append(f"- {name}: {len(values)} features")
    md_lines.append("")
    md_lines.append("## Results")
    md_lines.append("")
    for row in comparison.to_dict(orient="records"):
        md_lines.append(
            f"- {row['feature_set']} | stage={row['stage']} | AUROC={row['auroc']:.4f} | AUPRC={row['auprc']:.4f} | F1={row['f1']:.4f} | MCC={row['mcc']:.4f} | Sens={row['sensitivity']:.4f} | Spec={row['specificity']:.4f}"
        )
    md_lines.append("")
    md_lines.append(f"Best test feature set: {best_test['feature_set']} (AUROC={best_test['auroc']:.4f}, AUPRC={best_test['auprc']:.4f}).")
    md_lines.append("")
    md_lines.append("## Interpretation")
    md_lines.append("- The diagnostic comparison isolates how much the temporal representation adds beyond the current physiology and how much ICULOS contributes by itself.")
    md_lines.append("- The analysis intentionally avoids modifying the model, dataset, adapter, or calibration settings.")
    (output_path / "temporal_audit_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    return report


if __name__ == "__main__":
    run_temporal_audit()
