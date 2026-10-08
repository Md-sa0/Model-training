"""Calibração probabilística e limiar de alta sensibilidade, ajustados só na validação."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, matthews_corrcoef


def classification_counts(y_true, y_pred) -> dict[str, int]:
    actual = np.asarray(y_true).astype(int).reshape(-1)
    predicted = np.asarray(y_pred).astype(int).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("y_true e y_pred têm tamanhos diferentes")
    if not set(np.unique(actual)).issubset({0, 1}) or not set(np.unique(predicted)).issubset({0, 1}):
        raise ValueError("As classes precisam ser 0 ou 1")
    return {
        "TP": int(np.sum((actual == 1) & (predicted == 1))),
        "TN": int(np.sum((actual == 0) & (predicted == 0))),
        "FP": int(np.sum((actual == 0) & (predicted == 1))),
        "FN": int(np.sum((actual == 1) & (predicted == 0))),
    }


def _ratio(numerator: int, denominator: int):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def binary_metrics(y_true, y_pred) -> dict[str, object]:
    counts = classification_counts(y_true, y_pred)
    actual = np.asarray(y_true).astype(int).reshape(-1)
    predicted = np.asarray(y_pred).astype(int).reshape(-1)
    return {
        **counts,
        "sensitivity": _ratio(counts["TP"], counts["TP"] + counts["FN"]),
        "specificity": _ratio(counts["TN"], counts["TN"] + counts["FP"]),
        "ppv": _ratio(counts["TP"], counts["TP"] + counts["FP"]),
        "npv": _ratio(counts["TN"], counts["TN"] + counts["FN"]),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(actual, predicted)) if len(actual) else None,
    }


def expected_calibration_error(y_true, probabilities, n_bins: int = 10):
    actual = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0.0, 1.0)
    if len(actual) != len(scores) or len(actual) == 0:
        raise ValueError("Rótulos e probabilidades precisam ter o mesmo tamanho positivo")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(actual)
    error = 0.0
    curve = []
    for index in range(n_bins):
        low = float(edges[index])
        high = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        count = int(mask.sum())
        if count == 0:
            curve.append({
                "bin_lower": low, "bin_upper": high, "count": 0,
                "mean_predicted": None, "observed_frequency": None,
            })
            continue
        mean_predicted = float(scores[mask].mean())
        observed = float(actual[mask].mean())
        error += (count / total) * abs(observed - mean_predicted)
        curve.append({
            "bin_lower": low, "bin_upper": high, "count": count,
            "mean_predicted": mean_predicted, "observed_frequency": observed,
        })
    return float(error), curve


def calibration_metrics(y_true, probabilities, n_bins: int = 10) -> dict[str, object]:
    actual = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0.0, 1.0)
    ece, curve = expected_calibration_error(actual, scores, n_bins=n_bins)
    return {
        "brier": float(brier_score_loss(actual, scores)),
        "ece": ece,
        "log_loss": float(log_loss(actual, np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "curve": curve,
        "in_sample": True,
    }


def _supervised_arrays(scores, labels):
    features = np.asarray(scores, dtype=float).reshape(-1)
    target = np.asarray(labels).astype(int).reshape(-1)
    if len(features) != len(target) or len(features) == 0:
        raise ValueError("Scores e rótulos precisam ter o mesmo tamanho positivo")
    if np.isnan(features).any() or np.isnan(target.astype(float)).any():
        raise ValueError("Há valores ausentes nos scores ou nos rótulos")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("A calibração exige rótulos 0 e 1 na validação")
    return np.clip(features, 0.0, 1.0), target


def _parameters_equal(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return all(np.allclose(first, second) for first, second in zip(left, right, strict=True))


class FrozenCalibrator:
    """Calibrador já ajustado. A previsão não altera os parâmetros."""

    def __init__(self, method: str, estimator, n_samples: int, seed: int):
        self.method = method
        self.estimator = estimator
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.fitted_on = "validation"
        self.locked = True

    def fit(self, *args, **kwargs):
        raise RuntimeError("Calibrador congelado: o ajuste só pode ocorrer na validação.")

    def _parameters(self):
        if self.estimator is None:
            return None
        if self.method == "platt":
            return (self.estimator.coef_.copy(), self.estimator.intercept_.copy())
        if self.method == "isotonic":
            return (
                np.asarray(self.estimator.X_thresholds_).copy(),
                np.asarray(self.estimator.y_thresholds_).copy(),
            )
        return None

    def predict(self, scores) -> np.ndarray:
        before = self._parameters()
        values = np.asarray(scores, dtype=float).reshape(-1)
        if np.isnan(values).any():
            raise ValueError("Score ausente na previsão")
        if self.method == "raw":
            predicted = values
        elif self.method == "platt":
            predicted = self.estimator.predict_proba(values.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            predicted = np.asarray(self.estimator.predict(values), dtype=float)
        else:
            raise ValueError(f"Método de calibração desconhecido: {self.method}")
        predicted = np.clip(np.asarray(predicted, dtype=float), 0.0, 1.0)
        if not _parameters_equal(before, self._parameters()):
            raise RuntimeError("A previsão alterou o calibrador.")
        return predicted


def fit_probability_calibrator(method: str, scores, labels, split: str, seed: int = 42) -> FrozenCalibrator:
    if split != "validation":
        raise ValueError("O calibrador só pode ser ajustado na validação.")
    if method not in {"raw", "platt", "isotonic"}:
        raise ValueError(f"Método de calibração desconhecido: {method}")
    features, target = _supervised_arrays(scores, labels)
    estimator = None
    if method == "platt":
        estimator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(features.reshape(-1, 1), target)
    elif method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip")
        estimator.fit(features, target)
    return FrozenCalibrator(method=method, estimator=estimator, n_samples=len(target), seed=seed)


def compare_calibrators(scores, labels, split: str, seed: int = 42, n_bins: int = 10):
    if split != "validation":
        raise ValueError("A comparação de calibradores só pode usar a validação.")
    fitted = {
        method: fit_probability_calibrator(method, scores, labels, split="validation", seed=seed)
        for method in ("raw", "platt", "isotonic")
    }
    metrics = {
        method: calibration_metrics(labels, calibrator.predict(scores), n_bins=n_bins)
        for method, calibrator in fitted.items()
    }
    preference = {"raw": 0, "platt": 1, "isotonic": 2}
    best_name = min(
        metrics,
        key=lambda name: (metrics[name]["brier"], metrics[name]["ece"], metrics[name]["log_loss"], preference[name]),
    )
    return fitted[best_name], metrics


def format_threshold_message(target: float, achieved: float, threshold: float, target_met: bool) -> str:
    lines = [
        f"Target sensitivity: {target * 100:.2f}%",
        f"Achieved sensitivity: {achieved * 100:.2f}%",
        f"Selected threshold: {threshold:.2f}",
    ]
    if not target_met:
        lines.append(
            "Nenhum limiar atingiu a sensibilidade alvo. "
            "Foi selecionado o limiar com maior sensibilidade."
        )
        lines.append("WARNING: Target sensitivity of 95% was not achieved.")
    return "\n".join(lines)


def select_threshold(y_true, probabilities, target_sensitivity: float = 0.95, split: str = "validation", thresholds=None):
    if split != "validation":
        raise ValueError("O limiar só pode ser escolhido na validação.")
    actual = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if set(np.unique(actual)) != {0, 1}:
        raise ValueError("A validação precisa conter as duas classes para escolher o limiar.")
    if thresholds is None:
        thresholds = np.round(np.arange(0, 101) / 100, 2)
    rows = []
    for threshold in thresholds:
        metrics = binary_metrics(actual, scores >= float(threshold))
        rows.append({
            "threshold": float(threshold),
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "PPV": metrics["ppv"],
            "NPV": metrics["npv"],
            "F1": metrics["f1"],
            "FN": metrics["FN"],
            "FP": metrics["FP"],
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "MCC": metrics["mcc"],
            "false_positive_rate": 1.0 - metrics["specificity"],
            "false_negative_rate": 1.0 - metrics["sensitivity"],
        })
    table = pd.DataFrame(rows)
    eligible = table[table["sensitivity"] >= target_sensitivity]
    target_met = not eligible.empty
    if target_met:
        pool = eligible.sort_values(
            ["specificity", "MCC", "PPV", "FP", "threshold"],
            ascending=[False, False, False, True, False],
            kind="mergesort",
        )
    else:
        pool = table.sort_values(
            ["sensitivity", "specificity", "MCC", "PPV", "FP", "threshold"],
            ascending=[False, False, False, False, True, False],
            kind="mergesort",
        )
    chosen = pool.iloc[0]
    achieved = float(chosen["sensitivity"])
    threshold = float(chosen["threshold"])
    info = {
        "threshold": threshold,
        "target_sensitivity": float(target_sensitivity),
        "achieved_sensitivity": achieved,
        "achieved_specificity": float(chosen["specificity"]),
        "target_met": bool(target_met),
        "message": format_threshold_message(float(target_sensitivity), achieved, threshold, bool(target_met)),
    }
    return info, table


def save_calibrator(calibrator: FrozenCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: Path) -> FrozenCalibrator:
    calibrator = joblib.load(path)
    if not isinstance(calibrator, FrozenCalibrator):
        raise TypeError("Arquivo de calibrador inválido")
    if not calibrator.locked or calibrator.fitted_on != "validation":
        raise RuntimeError("Calibrador carregado sem o bloqueio da validação")
    return calibrator
