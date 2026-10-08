from __future__ import annotations

import pandas as pd
import pytest

from qwen_sepsis.temporal_audit import (
    build_feature_set,
    build_temporal_features,
    validate_patient_order,
)


def make_temporal_frame() -> pd.DataFrame:
    rows = []
    for patient in [1, 2]:
        for hour in [0, 1, 2, 3]:
            rows.append(
                {
                    "Patient_ID": patient,
                    "Hour": hour,
                    "SepsisLabel": 1 if patient == 1 and hour >= 2 else 0,
                    "HR": 70 + patient * 5 + hour,
                    "O2Sat": 98.0,
                    "Temp": 37.0,
                    "SBP": 120.0,
                    "MAP": 80.0,
                    "DBP": 60.0,
                    "Resp": 18.0,
                    "WBC": 7.5,
                    "Lactate": 1.2,
                    "Creatinine": 1.0,
                    "Platelets": 250.0,
                    "Age": 65.0,
                    "Gender": 0.0,
                    "ICULOS": hour + 1,
                }
            )
    return pd.DataFrame(rows)


def test_temporal_features_are_past_only_and_ordered_by_patient_hour():
    frame = make_temporal_frame().sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
    temporal = build_temporal_features(frame, feature_name="HR")

    assert "HR_delta_1h" in temporal.columns
    assert "HR_delta_3h" in temporal.columns
    assert "HR_rolling_mean_3h" in temporal.columns
    assert "HR_slope_3h" in temporal.columns
    assert "history_hours" in temporal.columns

    assert pd.isna(temporal.loc[0, "HR_delta_1h"])
    assert pd.isna(temporal.loc[0, "HR_delta_3h"])
    assert temporal.loc[1, "HR_delta_1h"] == pytest.approx(frame.loc[1, "HR"] - frame.loc[0, "HR"])
    assert temporal.loc[3, "HR_delta_3h"] == pytest.approx(frame.loc[3, "HR"] - frame.loc[0, "HR"])
    assert temporal.loc[3, "history_hours"] == 4


def test_validate_patient_order_rejects_non_monotonic_data():
    frame = make_temporal_frame()
    bad = frame.copy()
    bad.loc[bad.index[0], "Hour"] = 5
    with pytest.raises(ValueError):
        validate_patient_order(bad)


def test_build_feature_set_returns_expected_groups():
    features = build_feature_set("PHYSIOLOGY_ONLY")
    assert "HR" in features
    assert "ICULOS" not in features
    assert "Temp" in features

    temporal = build_feature_set("PHYSIOLOGY_PLUS_TEMPORAL")
    assert "HR_delta_1h" in temporal
    assert "HR_rolling_mean_3h" in temporal

    combined = build_feature_set("PHYSIOLOGY_PLUS_TEMPORAL_PLUS_ICULOS")
    assert "ICULOS" in combined
    assert "HR_delta_1h" in combined
