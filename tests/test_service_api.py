from __future__ import annotations

import importlib.util
import json
import math
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from qwen_sepsis.data import DYNAMIC, FEATURES
from qwen_sepsis.explain import DISCLAIMER, RISK_SCORE_MEANING
from qwen_sepsis.inference_service import InferenceSettings
from qwen_sepsis.service_api import ROOT, ServiceSettings, create_app


class FakeEngine:
    config_version = "config-test-v1"
    model_version = "model-test-v1"
    calibrator = type("Calibrator", (), {"method": "isotonic"})()
    config = {
        "model_id": "test-model",
        "threshold": 0.01,
        "fitted_on": "validation",
        "test_used_for_fitting": False,
        "target_met": False,
    }

    def __init__(self):
        self.calls = 0

    def public_config(self, model_ready: bool):
        return {
            "model_ready": model_ready,
            "model_id": self.config["model_id"],
            "model_version": self.model_version,
            "config_version": self.config_version,
            "calibration_method": self.calibrator.method,
            "calibrator_locked": True,
            "threshold": self.config["threshold"],
            "threshold_locked": True,
            "threshold_target_met": False,
            "fitted_on": "validation",
            "test_used_for_fitting": False,
            "features": FEATURES,
            "dynamic_features": DYNAMIC,
            "decision_semantics": "RISK_ELEVATED quando risk_score >= threshold; caso contrário LOW_RISK.",
            "risk_score_meaning": RISK_SCORE_MEANING,
            "disclaimer": DISCLAIMER,
        }

    def evaluate(self, values):
        self.calls += 1
        return {
            "raw_score": 0.2,
            "risk_score": 0.02,
            "risk_score_meaning": RISK_SCORE_MEANING,
            "threshold": 0.01,
            "decision": "RISK_ELEVATED",
            "calibration_method": "isotonic",
            "model_version": self.model_version,
            "config_version": self.config_version,
            "imputed_features": [name for name in DYNAMIC if values.get(name) is None],
            "message": "Risco elevado de desenvolvimento de sepse.",
            "disclaimer": DISCLAIMER,
        }


def build_client(tmp_path, engine_loader=None):
    engine = FakeEngine()
    settings = ServiceSettings(
        database_path=tmp_path / "service.sqlite3",
        inference=InferenceSettings(
            artifact_dir=tmp_path / "artifacts",
            data_dir=tmp_path / "processed",
            base_model=tmp_path / "base",
            adapter=tmp_path / "adapter",
        ),
    )
    loader = engine_loader or (lambda _settings: engine)
    return TestClient(create_app(settings, loader)), engine


def register_patient(client: TestClient):
    response = client.post(
        "/api/patients",
        json={"external_id": "case-1", "initial_episode_label": "admission-1"},
    )
    assert response.status_code == 201
    return response.json()


def post_observation(client: TestClient, patient, observation_id=None, episode_id=None, values=None):
    return client.post(
        f"/api/patients/{patient['patient']['patient_id']}/observations",
        json={
            "observation_id": str(observation_id or uuid4()),
            "episode_id": episode_id or patient["episode"]["episode_id"],
            "observed_at": "2026-10-18T10:00:00-03:00",
            "values": values if values is not None else {"HR": 105, "Temp": 38.1, "Age": 60},
            "units": {"HR": "bpm", "Temp": "degC"},
        },
    )


def test_patients_episodes_observations_and_assessment_are_linked_and_idempotent(tmp_path):
    with build_client(tmp_path)[0] as client:
        patient = register_patient(client)
        patient_id = patient["patient"]["patient_id"]
        assert client.get("/api/patients").json()["total"] == 1
        assert client.get(f"/api/patients/{patient_id}/episodes").json()[0]["episode_id"] == patient["episode"]["episode_id"]
        observation_id = uuid4()
        first = post_observation(client, patient, observation_id)
        assert first.status_code == 201
        assert first.json()["missing_fields"]

        repeated = post_observation(client, patient, observation_id)
        assert repeated.status_code == 200
        assert repeated.json()["observation_id"] == str(observation_id)
        assert client.get(f"/api/patients/{patient_id}/observations").json()["total"] == 1

        engine = client.app.state.engine
        assessment_id = str(uuid4())
        body = {"assessment_id": assessment_id, "observation_id": str(observation_id)}
        assessment = client.post(f"/api/patients/{patient_id}/assessments", json=body)
        assert assessment.status_code == 201
        assert assessment.json()["config_version"] == engine.config_version
        assert assessment.json()["observation_id"] == str(observation_id)
        assert engine.calls == 1

        repeated_assessment = client.post(f"/api/patients/{patient_id}/assessments", json=body)
        assert repeated_assessment.status_code == 200
        assert repeated_assessment.json()["assessment_id"] == assessment_id
        assert engine.calls == 1
        assert client.get(f"/api/patients/{patient_id}/assessments").json()["total"] == 1
        assert client.get(f"/api/patients/{patient_id}").json()["episodes"][0]["episode_id"] == patient["episode"]["episode_id"]


def test_patient_and_episode_isolation_and_conflicts(tmp_path):
    with build_client(tmp_path)[0] as client:
        patient = register_patient(client)
        other = client.post("/api/patients", json={}).json()
        observation_id = uuid4()
        assert post_observation(client, patient, observation_id).status_code == 201
        other_patient_id = other["patient"]["patient_id"]
        assert client.get(
            f"/api/patients/{other_patient_id}/observations"
        ).json()["total"] == 0
        assert client.post(
            f"/api/patients/{other_patient_id}/assessments",
            json={"assessment_id": str(uuid4()), "observation_id": str(observation_id)},
        ).status_code == 404

        other_episode = client.post(
            f"/api/patients/{other_patient_id}/episodes", json={"label": "readmission"}
        ).json()
        mismatch = post_observation(client, patient, episode_id=other_episode["episode_id"])
        assert mismatch.status_code == 404

        conflict = post_observation(
            client, patient, observation_id, values={"HR": 107}
        )
        assert conflict.status_code == 409

        duplicate_time = post_observation(client, patient, values={"HR": 108})
        assert duplicate_time.status_code == 409


def test_invalid_observation_inputs_and_no_implicit_inference(tmp_path):
    client, engine = build_client(tmp_path)
    with client:
        patient = register_patient(client)
        invalid = post_observation(client, patient, values={"HR": "fast"})
        assert invalid.status_code == 422
        unknown = post_observation(client, patient, values={"UNKNOWN": 1})
        assert unknown.status_code == 422
        bad_time = client.post(
            f"/api/patients/{patient['patient']['patient_id']}/observations",
            json={
                "observation_id": str(uuid4()),
                "episode_id": patient["episode"]["episode_id"],
                "observed_at": "2026-10-18T10:00:00",
                "values": {"HR": 100},
            },
        )
        assert bad_time.status_code == 422
        saved = post_observation(client, patient)
        assert saved.status_code == 201
        assert engine.calls == 0


def test_stateless_risk_endpoint_reports_config_and_model_readiness(tmp_path):
    client, engine = build_client(tmp_path)
    with client:
        assert client.get("/health").json()["model_ready"] is True
        config = client.get("/api/config").json()
        assert config["threshold"] == 0.01
        result = client.post("/api/patient-risk", json={"values": {"HR": 110}})
        assert result.status_code == 200
        assert result.json()["config_version"] == engine.config_version
        assert engine.calls == 1


def test_model_unavailable_keeps_service_up_and_returns_503(tmp_path):
    def fail_loader(_settings):
        raise RuntimeError("weights not available")

    with build_client(tmp_path, fail_loader)[0] as client:
        assert client.get("/health").json() == {
            "status": "available", "model_ready": False, "model_error": "RuntimeError"
        }
        assert client.get("/api/config").json()["model_ready"] is False
        assert client.post("/api/patient-risk", json={"values": {"HR": 100}}).status_code == 503
        patient = register_patient(client)
        observation = post_observation(client, patient)
        assessment = client.post(
            f"/api/patients/{patient['patient']['patient_id']}/assessments",
            json={
                "assessment_id": str(uuid4()),
                "observation_id": observation.json()["observation_id"],
            },
        )
        assert assessment.status_code == 503
        assert client.get(
            f"/api/patients/{patient['patient']['patient_id']}/assessments"
        ).json()["total"] == 0


def test_database_persists_across_app_restart(tmp_path):
    with build_client(tmp_path)[0] as first_client:
        patient = register_patient(first_client)
        observation = post_observation(first_client, patient)
        assert observation.status_code == 201
        patient_id = patient["patient"]["patient_id"]
        observation_id = observation.json()["observation_id"]

    with build_client(tmp_path)[0] as restarted_client:
        assert restarted_client.get(f"/api/patients/{patient_id}").status_code == 200
        stored = restarted_client.get(f"/api/patients/{patient_id}/observations").json()
        assert stored["total"] == 1
        assert stored["items"][0]["observation_id"] == observation_id


def real_model_assets_available():
    required = [
        ROOT / "artifacts" / "calibration" / "config.json",
        ROOT / "artifacts" / "calibration" / "calibrator.pkl",
        ROOT / "data" / "processed" / "train_medians.csv",
        ROOT / "models" / "base" / "Qwen3-8B" / "config.json",
        ROOT / "models" / "qwen3-8b-sepsis-lora" / "final_adapter" / "adapter_model.safetensors",
    ]
    if any(not path.is_file() for path in required):
        return False
    if any(importlib.util.find_spec(module) is None for module in ("torch", "transformers", "peft", "bitsandbytes")):
        return False
    import torch

    return torch.cuda.is_available()


@pytest.mark.skipif(
    not real_model_assets_available(),
    reason="Real model inference requires local artifacts and compatible CUDA.",
)
def test_api_runs_real_model_inference_with_frozen_artifacts(tmp_path):
    defaults = ServiceSettings.from_environment()
    settings = ServiceSettings(tmp_path / "real-inference.sqlite3", defaults.inference)
    payload = json.loads((ROOT / "configs" / "demo_patient_example.json").read_text(encoding="utf-8"))
    values = {key: value for key, value in payload.items() if key in FEATURES}
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["model_ready"] is True
        result = client.post("/api/patient-risk", json={"values": values})
        assert result.status_code == 200
        result = result.json()
        assert 0 <= result["raw_score"] <= 1
        assert 0 <= result["risk_score"] <= 1
        assert math.isfinite(result["threshold"])
        assert result["threshold"] == 0.01
        assert result["calibration_method"] == "isotonic"
        assert len(result["config_version"]) == 64
        assert len(result["model_version"]) == 64
