from __future__ import annotations

import numpy as np
import pandas as pd

from qwen_sepsis.model_quality import (
    aggregate_patient_scores,
    build_model_quality_report,
    compute_threshold_analysis,
    summarize_raw_score_distribution,
)


def test_summarize_raw_score_distribution_and_threshold_analysis():
    frame = pd.DataFrame(
        {
            "SepsisLabel": [0, 0, 1, 1],
            "raw_probability": [0.1, 0.4, 0.6, 0.9],
        }
    )

    summary = summarize_raw_score_distribution(frame, "raw_probability", "SepsisLabel")
    assert set(summary) == {"label_0", "label_1"}
    assert summary["label_0"]["min"] == 0.1
    assert summary["label_1"]["max"] == 0.9

    analysis = compute_threshold_analysis(frame["SepsisLabel"].to_numpy(), frame["raw_probability"].to_numpy())
    assert len(analysis) > 0
    assert analysis["threshold"].iloc[0] == 0.0
    assert analysis.loc[analysis["threshold"] == 0.0, "sensitivity"].iat[0] == 1.0


def test_patient_aggregation_uses_patient_level_labels():
    rows = [
        {"Patient_ID": "p1", "Hour": 0, "SepsisLabel": 0, "raw_probability": 0.2},
        {"Patient_ID": "p1", "Hour": 1, "SepsisLabel": 1, "raw_probability": 0.8},
        {"Patient_ID": "p2", "Hour": 0, "SepsisLabel": 0, "raw_probability": 0.6},
        {"Patient_ID": "p2", "Hour": 1, "SepsisLabel": 0, "raw_probability": 0.4},
        {"Patient_ID": "p3", "Hour": 0, "SepsisLabel": 0, "raw_probability": 0.1},
        {"Patient_ID": "p3", "Hour": 1, "SepsisLabel": 0, "raw_probability": 0.15},
    ]
    frame = pd.DataFrame(rows)

    patient_scores = aggregate_patient_scores(frame, score_column="raw_probability")
    assert set(patient_scores["Patient_ID"]) == {"p1", "p2", "p3"}
    assert patient_scores.loc[patient_scores["Patient_ID"] == "p1", "patient_label"].iat[0] == 1
    assert patient_scores.loc[patient_scores["Patient_ID"] == "p2", "max_probability"].iat[0] == 0.6


def test_build_model_quality_report_has_expected_sections():
    train = pd.DataFrame(
        {
            "Patient_ID": ["a", "a", "b", "b", "c", "c"],
            "Hour": [0, 1, 0, 1, 0, 1],
            "SepsisLabel": [0, 0, 1, 1, 0, 0],
            "raw_probability": [0.05, 0.08, 0.72, 0.83, 0.09, 0.11],
        }
    )
    val = pd.DataFrame(
        {
            "Patient_ID": ["d", "d", "e", "e", "f", "f"],
            "Hour": [0, 1, 0, 1, 0, 1],
            "SepsisLabel": [0, 0, 1, 1, 0, 0],
            "raw_probability": [0.10, 0.12, 0.75, 0.80, 0.11, 0.13],
        }
    )

    report = build_model_quality_report(train, val, current_threshold=0.5)
    assert "validation" in report
    assert "test" in report
    assert report["current_threshold"] == 0.5
    assert report["positive_prevalence"] > 0.0
    assert "best_youden_threshold" in report
