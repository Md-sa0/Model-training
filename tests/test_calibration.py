import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from qwen_sepsis.calibration import (
    classification_counts,
    fit_probability_calibrator,
    load_calibrator,
    save_calibrator,
    select_threshold,
)
from qwen_sepsis.data import FEATURES, load_prepared_split, prepare_dataset
from qwen_sepsis.explain import abnormalities_until, build_decision_record, classify_risk
from qwen_sepsis.patient_evaluation import build_patient_table, select_temporal_strategy, summarize_patient
from qwen_sepsis.pipeline import (
    REQUIRED_REPORT_KEYS,
    build_final_report,
    evaluate_test_frame,
    load_frozen_decision,
    save_validation_artifacts,
    select_from_validation,
    write_evaluation_artifacts,
)
from qwen_sepsis.scoring import probability_from_class_logits, resolve_binary_token_ids


class _SingleToken:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"0": [15], "1": [16]}[text]


class _MultiToken:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"0": [3, 4], "1": [5]}[text]


class _SamePrefix:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [7, 8]


def _rows(patient, hours, labels, probabilities):
    return pd.DataFrame({
        "Patient_ID": patient,
        "Hour": hours,
        "SepsisLabel": labels,
        "calibrated_probability": probabilities,
        "raw_probability": probabilities,
    })


def _cohort(start, n_positive, n_negative):
    rows = []
    for index in range(n_positive):
        patient = f"s{start + index}"
        for hour, label, probability in [(0, 0, 0.2), (1, 0, 0.4), (2, 1, 0.8), (3, 1, 0.9)]:
            rows.append({
                "Patient_ID": patient, "Hour": hour, "SepsisLabel": label, "raw_probability": probability,
            })
    for index in range(n_negative):
        patient = f"n{start + index}"
        for hour, probability in [(0, 0.05), (1, 0.10), (2, 0.15), (3, 0.20)]:
            rows.append({
                "Patient_ID": patient, "Hour": hour, "SepsisLabel": 0, "raw_probability": probability,
            })
    return pd.DataFrame(rows)


def test_probability_is_between_zero_and_one():
    even = probability_from_class_logits(1.0, 1.0)
    assert even["raw_probability"] == pytest.approx(0.5)
    high = probability_from_class_logits(-1e6, 1e6)
    low = probability_from_class_logits(1e6, -1e6)
    assert high["raw_probability"] == pytest.approx(1.0)
    assert low["raw_probability"] == pytest.approx(0.0)
    for result in (even, high, low):
        assert 0.0 <= result["raw_probability"] <= 1.0
        assert set(result) == {"logit_0", "logit_1", "raw_probability"}

    token_0, token_1, sequences = resolve_binary_token_ids(_SingleToken())
    assert (token_0, token_1) == (15, 16)
    assert sequences == {"0": [15], "1": [16]}
    token_0, token_1, sequences = resolve_binary_token_ids(_MultiToken())
    assert (token_0, token_1, sequences["0"]) == (3, 5, [3, 4])
    with pytest.raises(ValueError):
        resolve_binary_token_ids(_SamePrefix())

    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    for method in ("raw", "platt", "isotonic"):
        calibrator = fit_probability_calibrator(method, scores, labels, split="validation", seed=42)
        predicted = calibrator.predict(np.array([-0.5, 0.0, 0.5, 1.0, 1.5]))
        assert np.all((predicted >= 0.0) & (predicted <= 1.0))

    source = (Path(__file__).resolve().parents[1] / "src" / "qwen_sepsis" / "inference.py").read_text(encoding="utf-8")
    assert ".generate(" not in source


def test_calibrator_is_fit_only_on_validation():
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_val = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])
    y_test = np.array([1, 1, 1, 1])
    p_test = np.array([0.1, 0.1, 0.1, 0.2])
    honest = fit_probability_calibrator("platt", p_val, y_val, split="validation", seed=42)
    leaked = fit_probability_calibrator(
        "platt", np.concatenate([p_val, p_test]), np.concatenate([y_val, y_test]), split="validation", seed=42,
    )
    assert honest.fitted_on == "validation"
    assert honest.n_samples == len(y_val)
    assert not np.allclose(honest.estimator.coef_, leaked.estimator.coef_)
    with pytest.raises(ValueError):
        fit_probability_calibrator("platt", p_test, y_test, split="test", seed=42)
    with pytest.raises(ValueError):
        select_threshold(y_test, p_test, split="test")
    with pytest.raises(ValueError):
        select_temporal_strategy(pd.DataFrame(), target_sensitivity=0.95, split="test")
    with pytest.raises(ValueError):
        select_from_validation(pd.DataFrame(), 0.95, 42, 0.7, 0.02, split="test")

    frame = pd.DataFrame({
        "Patient_ID": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "Hour": [0, 1, 0, 1, 0, 1, 0, 1],
        "SepsisLabel": [0, 0, 0, 0, 1, 1, 1, 1],
        "raw_probability": p_val,
    })
    selection = select_from_validation(frame, 0.95, 42, 0.7, 0.02, split="validation")
    assert selection.calibrator.n_samples == len(frame)
    assert selection.config["fitted_on"] == "validation"
    assert selection.config["test_used_for_fitting"] is False


def test_test_split_does_not_modify_calibrator(tmp_path):
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_val = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.7, 0.85])
    calibrator = fit_probability_calibrator("platt", p_val, y_val, split="validation", seed=42)
    coefficient = calibrator.estimator.coef_.copy()
    intercept = calibrator.estimator.intercept_.copy()
    test_scores = np.array([0.99, 0.01, 0.5, 0.6])
    predicted = calibrator.predict(test_scores)
    assert np.allclose(coefficient, calibrator.estimator.coef_)
    assert np.allclose(intercept, calibrator.estimator.intercept_)
    with pytest.raises(RuntimeError):
        calibrator.fit(test_scores, np.array([1, 0, 1, 0]))
    path = tmp_path / "calibrator.pkl"
    save_calibrator(calibrator, path)
    loaded = load_calibrator(path)
    assert np.allclose(loaded.predict(test_scores), predicted)
    assert np.allclose(loaded.estimator.coef_, coefficient)
    isotonic = fit_probability_calibrator("isotonic", p_val, y_val, split="validation", seed=42)
    thresholds = np.asarray(isotonic.estimator.X_thresholds_).copy()
    isotonic.predict(test_scores)
    assert np.allclose(thresholds, isotonic.estimator.X_thresholds_)

    validation = _cohort(0, 10, 10)
    test = _cohort(100, 6, 6)
    test["SepsisLabel"] = 1 - test["SepsisLabel"]
    selection = select_from_validation(validation, 0.95, 42, 0.7, 0.02)
    context = {
        "model_id": "Qwen/Qwen3-8B",
        "adapter_path": "models/qwen3-8b-sepsis-lora/final_adapter",
        "base_model_path": "Qwen/Qwen3-8B",
        "dataset": {"revision": "teste", "sha256": "teste"},
        "seed": 42,
        "tokens": {"token_id_0": 15, "token_id_1": 16, "single_token_labels": True},
    }
    save_validation_artifacts(tmp_path, selection, context)
    config_text = (tmp_path / "calibration" / "config.json").read_text(encoding="utf-8")
    frozen, config = load_frozen_decision(tmp_path)
    before = frozen._parameters()
    result = evaluate_test_frame(test, frozen, config)
    after = frozen._parameters()
    if before is None:
        assert after is None
    else:
        assert all(np.allclose(left, right) for left, right in zip(before, after, strict=True))
    assert (tmp_path / "calibration" / "config.json").read_text(encoding="utf-8") == config_text
    report = build_final_report(result["metrics"], config, context)
    write_evaluation_artifacts(tmp_path, result, report)
    assert report["threshold"] == pytest.approx(config["threshold"])
    assert report["calibration_method"] == config["method"]
    assert report["evaluation_unit"] == "patient"
    assert report["test_used_for_fitting"] is False
    for key in REQUIRED_REPORT_KEYS:
        assert key in report
    expected = [
        "calibration/calibrator.pkl",
        "calibration/config.json",
        "calibration/threshold_analysis.csv",
        "calibration/calibration_curve.png",
        "calibration/threshold_curve.png",
        "evaluation/final_patient_report.json",
        "evaluation/final_patient_metrics.csv",
        "evaluation/patient_confusion_matrix.png",
        "evaluation/patient_predictions.csv",
    ]
    for relative in expected:
        artifact = tmp_path / relative
        assert artifact.exists() and artifact.stat().st_size > 0
    columns = pd.read_csv(tmp_path / "calibration" / "threshold_analysis.csv").columns
    assert list(columns[:8]) == ["threshold", "sensitivity", "specificity", "PPV", "NPV", "F1", "FN", "FP"]
    assert (tmp_path / "calibration" / "config.json").read_text(encoding="utf-8") == config_text


def test_confusion_counts():
    actual = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 0, 1])
    counts = classification_counts(actual, predicted)
    assert counts == {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    assert matrix.tolist() == [[1, 1], [1, 1]]
    assert matrix[0, 0] == counts["TN"]
    assert matrix[0, 1] == counts["FP"]
    assert matrix[1, 0] == counts["FN"]
    assert matrix[1, 1] == counts["TP"]


def test_grouping_by_patient_id(tmp_path):
    frame = pd.concat([
        _rows("p1", [0, 1], [0, 1], [0.2, 0.8]),
        _rows("p2", [0], [0], [0.1]),
    ], ignore_index=True)
    table = build_patient_table(frame, threshold=0.5, strategy="any_alert")
    assert set(table["Patient_ID"]) == {"p1", "p2"}
    assert len(table) == 2
    indexed = table.set_index("Patient_ID")
    assert int(indexed.loc["p1", "has_sepsis"]) == 1
    assert int(indexed.loc["p1", "patient_prediction"]) == 1
    assert int(indexed.loc["p2", "has_sepsis"]) == 0
    assert int(indexed.loc["p2", "patient_prediction"]) == 0

    rows = []
    for patient in range(80):
        for hour in (0, 1, 8):
            row = dict.fromkeys(FEATURES, 10.0)
            row.update(Patient_ID=f"p{patient:03}", Hour=hour, SepsisLabel=patient % 2)
            rows.append(row)
    raw = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    prepare_dataset(raw, tmp_path / "processed", max_train=100, max_eval=40)
    for split in ("train", "validation", "test"):
        folder = tmp_path / "processed"
        prompts = [json.loads(line) for line in (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        index = [json.loads(line) for line in (folder / f"{split}_patients.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(prompts) == len(index) > 0
        assert [item["row_index"] for item in index] == list(range(len(index)))
        assert set(prompts[0]) == {"prompt", "label"}
        assert "Patient_ID" not in prompts[0]["prompt"]
        assert "SepsisLabel" not in prompts[0]["prompt"]
    loaded = load_prepared_split(tmp_path / "processed", "validation")
    assert loaded["label"].astype(int).tolist() == loaded["SepsisLabel"].astype(int).tolist()
    assert loaded["Patient_ID"].nunique() < len(loaded)


def test_persistence_of_two_and_three_hours():
    two = summarize_patient(_rows("p", [0, 1, 2], [0, 0, 1], [0.9, 0.9, 0.1]), 0.5, "persist_2h")
    assert two["patient_prediction"] == 1
    assert two["first_alert_hour"] == 1
    assert two["persistent_positive_hours"] == 2

    broken = summarize_patient(_rows("p", [0, 1, 2], [1, 1, 1], [0.9, 0.1, 0.9]), 0.5, "persist_2h")
    assert broken["patient_prediction"] == 0

    three = summarize_patient(_rows("p", [0, 1, 2], [1, 1, 1], [0.9, 0.9, 0.9]), 0.5, "persist_3h")
    assert three["patient_prediction"] == 1
    assert three["first_alert_hour"] == 2

    not_three = summarize_patient(_rows("p", [0, 1, 2], [1, 1, 1], [0.9, 0.9, 0.1]), 0.5, "persist_3h")
    assert not_three["patient_prediction"] == 0

    gap = summarize_patient(_rows("p", [0, 2], [1, 1], [0.9, 0.9]), 0.5, "persist_2h")
    assert gap["patient_prediction"] == 0
    assert gap["hour_gap"] == 1

    window = summarize_patient(_rows("p", [0, 1, 2], [0, 0, 1], [0.9, 0.1, 0.9]), 0.5, "window_2_in_3h")
    assert window["patient_prediction"] == 1
    assert window["first_alert_hour"] == 2


def test_future_data_does_not_enter_prediction():
    frame = _rows("p", [1, 2, 3, 4], [0, 0, 0, 1], [0.1, 0.2, 0.95, 0.99])
    early = summarize_patient(frame, 0.5, "persist_2h", current_hour=2)
    truncated = summarize_patient(frame[frame["Hour"] <= 2], 0.5, "persist_2h", current_hour=2)
    assert early == truncated
    assert early["patient_prediction"] == 0
    assert early["first_alert_hour"] is None
    assert early["probability_history"] == [0.1, 0.2]
    later = summarize_patient(frame, 0.5, "any_alert", current_hour=4)
    assert later["first_alert_hour"] == 3
    assert later["patient_prediction"] == 1

    window_early = summarize_patient(frame, 0.5, "window_2_in_3h", current_hour=2)
    assert window_early["patient_prediction"] == 0
    window_late = summarize_patient(frame, 0.5, "window_2_in_3h", current_hour=4)
    assert window_late["patient_prediction"] == 1
    assert window_late["first_alert_hour"] == 4

    observed = pd.DataFrame([
        {"Hour": 1, "Lactate": 1.0, "Lactate_missing": 0, "Resp": 16, "Resp_missing": 0},
        {"Hour": 2, "Lactate": 9.0, "Lactate_missing": 0, "Resp": 30, "Resp_missing": 0},
    ])
    assert abnormalities_until(observed, 1) == []
    findings = abnormalities_until(observed, 2)
    assert any("Lactato" in item for item in findings)
    assert any("respiratória" in item for item in findings)


def test_threshold_meets_target_sensitivity_when_possible():
    positives = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.80, 0.70, 0.60, 0.40])
    negatives = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.20, 0.30, 0.50, 0.60, 0.70])
    actual = np.array([1] * len(positives) + [0] * len(negatives))
    scores = np.concatenate([positives, negatives])
    info, table = select_threshold(actual, scores, target_sensitivity=0.95, split="validation")
    eligible = table[table["sensitivity"] >= 0.95]
    tied = eligible[np.isclose(eligible["specificity"], eligible["specificity"].max())]
    assert info["target_met"] is True
    assert info["achieved_sensitivity"] >= 0.95
    assert info["threshold"] == pytest.approx(tied["threshold"].max())
    assert info["threshold"] == pytest.approx(0.40)
    assert info["threshold"] != pytest.approx(0.50)
    assert "Target sensitivity:" in info["message"]
    assert "Achieved sensitivity:" in info["message"]
    assert "Selected threshold:" in info["message"]

    failed, failed_table = select_threshold(
        np.array([1, 1, 1, 1, 0, 0, 0, 0]),
        np.array([0.0, 0.0, 0.0, 0.2, 0.9, 0.9, 0.9, 0.9]),
        target_sensitivity=0.95,
        split="validation",
    )
    assert failed["target_met"] is False
    assert failed["achieved_sensitivity"] == pytest.approx(failed_table["sensitivity"].max())
    assert failed["achieved_sensitivity"] == pytest.approx(0.25)
    assert failed["threshold"] == pytest.approx(0.20)
    assert "Nenhum limiar" in failed["message"]


def test_risk_levels_follow_configured_cuts():
    assert classify_risk(0.2, low_below=0.25, high_at_or_above=0.7) == "LOW"
    assert classify_risk(0.25, low_below=0.25, high_at_or_above=0.7) == "MODERATE"
    assert classify_risk(0.7, low_below=0.25, high_at_or_above=0.7) == "HIGH"
    record = build_decision_record(
        patient_id="p1", current_hour=2, raw_probability=0.4, calibrated_probability=0.8,
        risk_level="HIGH", first_alert_hour=1, persistent_alert_hours=2, trend="INCREASING",
        clinical_review_recommended=True, abnormalities=["PAM reduzida"], hourly_alert=True,
    )
    assert record["risk_score"] == pytest.approx(0.8)
    assert record["calibrated_probability"] == pytest.approx(0.8)
    assert record["observed_abnormalities"] == ["PAM reduzida"]
    assert "diagnóstico médico" in record["disclaimer"]
    assert "universais" in record["risk_level_note"]
