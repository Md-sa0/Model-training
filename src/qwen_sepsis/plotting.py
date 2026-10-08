"""Gráficos da calibração, do limiar e da matriz por paciente."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS = {"raw": "Bruto", "platt": "Platt", "isotonic": "Isotônica"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def save_calibration_curve(path: Path, metrics_by_method: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    for name, metrics in metrics_by_method.items():
        points = [item for item in metrics["curve"] if item["count"]]
        if not points:
            continue
        label = (
            f"{METHOD_LABELS.get(name, name)} "
            f"(Brier={metrics['brier']:.4f}, ECE={metrics['ece']:.4f})"
        )
        ax.plot(
            [item["mean_predicted"] for item in points],
            [item["observed_frequency"] for item in points],
            marker="o",
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Curva de calibração na validação\nPlatt e isotônica no mesmo conjunto em que foram ajustadas")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def save_threshold_curve(path: Path, table: pd.DataFrame, selected_threshold: float, target_sensitivity: float,
                         evaluation_unit: str = "observação horária") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(table["threshold"], table["sensitivity"], label="Sensibilidade")
    axes[0].plot(table["threshold"], table["specificity"], label="Especificidade")
    axes[0].axvline(selected_threshold, color="black", linestyle="--", label="Limiar escolhido")
    axes[0].axhline(target_sensitivity, color="tab:red", linestyle=":", label="Sensibilidade alvo")
    axes[0].set_xlabel("Limiar")
    axes[0].set_ylabel("Proporção")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("Sensibilidade e especificidade")

    axes[1].plot(table["threshold"], table["FN"], label="Falsos negativos")
    axes[1].plot(table["threshold"], table["FP"], label="Falsos positivos")
    axes[1].axvline(selected_threshold, color="black", linestyle="--", label="Limiar escolhido")
    axes[1].set_xlabel("Limiar")
    axes[1].set_ylabel(f"Contagem por {evaluation_unit} na validação")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_title("Falsos negativos e falsos positivos")
    fig.suptitle("Escolha do limiar na validação")
    fig.tight_layout()
    _save(fig, path)


def save_confusion_matrix(path: Path, y_true, y_pred, title: str = "Matriz de confusão por paciente no teste"):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Negativo", "Positivo"])
    ax.set_yticks([0, 1], ["Negativo", "Positivo"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)
    names = [["TN", "FP"], ["FN", "TP"]]
    peak = matrix.max() if matrix.size else 0
    for row in range(2):
        for col in range(2):
            color = "white" if peak and matrix[row, col] > peak / 2 else "black"
            ax.text(col, row, f"{names[row][col]}\n{matrix[row, col]}", ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    _save(fig, path)
    return matrix
