from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qwen_sepsis.data import FEATURES, format_prompt, patient_split, prepare_dataset, temporal_fill, validate


def sample(patients=40):
    rows = []
    for patient in range(patients):
        for hour in [0, 1, 8]:
            row = dict.fromkeys(FEATURES, 10.0)
            row.update(Patient_ID=f"p{patient:03}", Hour=hour, SepsisLabel=patient % 2)
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_is_patient_level_reproducible_and_order_independent():
    frame = sample()
    first = patient_split(frame, 42)
    assert first == patient_split(frame.sample(frac=1), 42)
    assert set(first) == set(frame.Patient_ID)
    assert set(first.values()) == {"train", "validation", "test"}


def test_temporal_fill_uses_elapsed_hours_and_never_future():
    frame = sample().query("Patient_ID == 'p000'").copy()
    frame["HR"] = [12, np.nan, 50]
    result = temporal_fill(frame)
    assert result.iloc[1].HR == 12
    assert result.iloc[1].HR_missing == 1
    assert result.iloc[2].HR == 50
    frame["HR"] = [12, np.nan, np.nan]
    assert np.isnan(temporal_fill(frame).iloc[2].HR)


@pytest.mark.parametrize("problem", ["column", "label", "duplicate", "negative", "infinite"])
def test_validation_rejects_invalid_data(problem):
    frame = sample()
    if problem == "column": frame = frame.drop(columns="HR")
    if problem == "label": frame.loc[0, "SepsisLabel"] = 2
    if problem == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    if problem == "negative": frame.loc[0, "Hour"] = -1
    if problem == "infinite": frame.loc[0, "HR"] = np.inf
    with pytest.raises(ValueError): validate(frame)


def test_prompt_contains_values_missingness_and_no_target():
    frame = temporal_fill(sample().iloc[:3])
    frame[FEATURES] = frame[FEATURES].fillna(0)
    prompt = format_prompt(frame.iloc[0])
    assert "HR=" in prompt and "Variáveis originalmente ausentes" in prompt
    assert "SepsisLabel" not in prompt and "Patient_ID" not in prompt


def test_prepare_writes_disjoint_datasets(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    frame = sample(80)
    frame.to_csv(raw, index=False)
    metadata = prepare_dataset(raw, tmp_path / "processed", max_train=100, max_eval=40)
    assert metadata["valid_rows"] == len(frame)
    for split in ["train", "validation", "test"]:
        path = tmp_path / "processed" / f"{split}.jsonl"
        assert path.exists() and path.stat().st_size > 0

