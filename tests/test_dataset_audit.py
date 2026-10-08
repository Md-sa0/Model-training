from __future__ import annotations

import pandas as pd
import pytest

from qwen_sepsis.data import FEATURES
from qwen_sepsis.dataset_audit import (
    build_feature_groups,
    build_temporal_features,
    compute_missingness,
    compute_prevalence_summary,
    summarize_numeric_feature,
)


def make_patient_frame() -> pd.DataFrame:
    rows = []
    for patient in [1, 2, 3]:
        for hour in [0, 1, 2, 3]:
            rows.append(
                {
                    "Patient_ID": patient,
                    "Hour": hour,
                    "ICULOS": hour + 1,
                    "SepsisLabel": 1 if patient == 1 and hour >= 2 else 0,
                    "HR": 70 + patient * 5 + hour,
                    "O2Sat": 97.0,
                    "Temp": 36.8,
                    "SBP": 120,
                    "MAP": 80,
                    "DBP": 60,
                    "Resp": 18,
                    "WBC": 7.5,
                    "Lactate": 1.2,
                    "Creatinine": 1.0,
                    "Platelets": 250,
                    "Age": 65,
                    "Gender": 0,
                }
            )
    return pd.DataFrame(rows)


def test_prevalence_summary_counts_observations_and_patients():
    frame = make_patient_frame()
    summary = compute_prevalence_summary(frame)
    assert summary["rows"] == 12
    assert summary["patients"] == 3
    assert summary["positive_rows"] == 2
    assert summary["negative_rows"] == 10
    assert summary["positive_prevalence_observation"] == pytest.approx(2 / 12)
    assert summary["positive_prevalence_patient"] == pytest.approx(1 / 3)


def test_missingness_and_feature_summary_are_numeric_and_complete():
    frame = make_patient_frame()
    frame.loc[0, "HR"] = None
    missing = compute_missingness(frame, columns=["HR", "WBC", "Lactate"])
    assert missing["HR"]["missing_count"] == 1
    assert missing["HR"]["missing_pct"] == pytest.approx(1 / 12 * 100)
    stats = summarize_numeric_feature(frame["HR"], feature_name="HR")
    assert stats["count"] == 11
    assert stats["missing_pct"] == pytest.approx(1 / 12 * 100)
    assert stats["mean"] > 0
    assert stats["median"] >= 0
    assert stats["p01"] <= stats["p99"]


def test_temporal_features_use_only_past_values_and_first_record_is_safe():
    frame = make_patient_frame().sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
    frame.loc[1, "HR"] = 100
    temporal = build_temporal_features(frame, feature_name="HR", window_hours=(1, 3, 6))
    assert "HR_delta_1h" in temporal.columns
    assert "HR_delta_3h" in temporal.columns
    assert "HR_rolling_mean_3h" in temporal.columns
    assert pd.isna(temporal.loc[0, "HR_delta_1h"])
    assert pd.isna(temporal.loc[0, "HR_delta_3h"])
    assert temporal.loc[0, "HR_rolling_mean_3h"] == pytest.approx(frame.loc[0, "HR"])
    assert temporal.loc[1, "HR_delta_1h"] == pytest.approx(frame.loc[1, "HR"] - frame.loc[0, "HR"]) 
    assert temporal.loc[2, "HR_delta_1h"] == pytest.approx(frame.loc[2, "HR"] - frame.loc[1, "HR"]) 


def test_audit_feature_groups_match_expected_columns():
    groups = build_feature_groups()
    assert groups["A"] == ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
    assert groups["B"] == ["WBC", "Lactate", "Creatinine", "Platelets"]
    assert groups["C"] == ["Age", "Gender"]
    assert groups["D"] == ["ICULOS"]
    assert groups["E"] == FEATURES
    assert groups["F"] == FEATURES + [
        "HR_delta_1h",
        "HR_delta_3h",
        "HR_delta_6h",
        "O2Sat_delta_1h",
        "O2Sat_delta_3h",
        "O2Sat_delta_6h",
        "Temp_delta_1h",
        "Temp_delta_3h",
        "Temp_delta_6h",
    ]


def test_audit_does_not_write_into_production_paths():
    from qwen_sepsis.dataset_audit import DEFAULT_OUTPUT_DIR

    assert "artifacts/dataset_audit" in DEFAULT_OUTPUT_DIR.as_posix()
    assert "data/processed" not in DEFAULT_OUTPUT_DIR.as_posix()
    assert "models" not in DEFAULT_OUTPUT_DIR.as_posix()
