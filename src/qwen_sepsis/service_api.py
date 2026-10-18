"""FastAPI application for frozen inference and longitudinal patient records."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse

from .data import DYNAMIC, FEATURES
from .explain import DISCLAIMER, RISK_SCORE_MEANING
from .inference_service import InferenceEngine, InferenceSettings
from .service_db import SQLiteStore
from .service_schemas import (
    AssessmentCreate,
    AssessmentPageResponse,
    AssessmentResponse,
    ConfigResponse,
    EpisodeCreate,
    EpisodeResponse,
    HealthResponse,
    ObservationCreate,
    ObservationPageResponse,
    ObservationResponse,
    PatientCreate,
    PatientCreatedResponse,
    PatientDetailResponse,
    PatientPageResponse,
    PatientResponse,
    PatientRiskRequest,
    PatientRiskResponse,
)

logger = logging.getLogger("qwen_sepsis.service")
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServiceSettings:
    database_path: Path
    inference: InferenceSettings

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        return cls(
            database_path=Path(os.environ.get("SEPSIS_DB_PATH", ROOT / "data" / "service.sqlite3")),
            inference=InferenceSettings(
                artifact_dir=Path(os.environ.get("SEPSIS_ARTIFACT_DIR", ROOT / "artifacts")),
                data_dir=Path(os.environ.get("SEPSIS_DATA_DIR", ROOT / "data" / "processed")),
                base_model=Path(os.environ.get("SEPSIS_BASE_MODEL", ROOT / "models" / "base" / "Qwen3-8B")),
                adapter=Path(os.environ.get(
                    "SEPSIS_ADAPTER", ROOT / "models" / "qwen3-8b-sepsis-lora" / "final_adapter"
                )),
            ),
        )


def create_app(
    settings: ServiceSettings | None = None,
    engine_loader: Callable[[InferenceSettings], InferenceEngine] = InferenceEngine.load,
) -> FastAPI:
    selected_settings = settings or ServiceSettings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = SQLiteStore(selected_settings.database_path)
        store.initialize()
        app.state.store = store
        app.state.inference_lock = threading.Lock()
        app.state.engine = None
        app.state.model_error = None
        try:
            app.state.engine = engine_loader(selected_settings.inference)
            logger.info("Inference model initialized and ready")
        except Exception as error:
            app.state.model_error = type(error).__name__
            logger.warning("Inference model initialization failed (%s)", type(error).__name__)
        yield

    app = FastAPI(
        title="Qwen Sepsis Risk Service",
        description=(
            "Experimental decision-support API. Not clinically validated and not a real-time "
            "hospital monitoring system."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(sqlite3.IntegrityError)
    async def integrity_error_handler(_request, _error):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "Conflict with an existing identifier or record constraint."},
        )

    def get_store() -> SQLiteStore:
        return app.state.store

    def get_engine() -> InferenceEngine:
        engine = app.state.engine
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "MODEL_NOT_READY", "message": "Inference model is not ready."},
            )
        return engine

    def to_utc(value: datetime | None) -> str:
        moment = value or datetime.now(timezone.utc)
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise HTTPException(status_code=422, detail="Date-time values must include a timezone.")
        return moment.astimezone(timezone.utc).isoformat()

    @app.get("/health", response_model=HealthResponse)
    def health():
        error = app.state.model_error
        return {
            "status": "available",
            "model_ready": app.state.engine is not None,
            "model_error": error,
        }

    @app.get("/api/config", response_model=ConfigResponse)
    def config():
        engine = app.state.engine
        if engine is not None:
            return engine.public_config(model_ready=True)
        return {
            "model_ready": False,
            "model_id": None,
            "model_version": None,
            "config_version": None,
            "calibration_method": None,
            "calibrator_locked": None,
            "threshold": None,
            "threshold_locked": None,
            "threshold_target_met": None,
            "fitted_on": None,
            "test_used_for_fitting": None,
            "features": FEATURES,
            "dynamic_features": DYNAMIC,
            "decision_semantics": "RISK_ELEVATED quando risk_score >= threshold; caso contrário LOW_RISK.",
            "risk_score_meaning": RISK_SCORE_MEANING,
            "disclaimer": DISCLAIMER,
        }

    @app.post("/api/patient-risk", response_model=PatientRiskResponse)
    def patient_risk(request: PatientRiskRequest):
        engine = get_engine()
        with app.state.inference_lock:
            result = engine.evaluate(request.values)
        return result

    @app.post(
        "/api/patients", response_model=PatientCreatedResponse, status_code=status.HTTP_201_CREATED
    )
    def create_patient(request: PatientCreate):
        patient_id, episode_id = uuid4(), uuid4()
        try:
            return get_store().create_patient(
                str(patient_id),
                str(episode_id),
                request.external_id,
                request.initial_episode_label,
                to_utc(request.episode_started_at),
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(status_code=409, detail="external_id already exists.") from error

    @app.get("/api/patients", response_model=PatientPageResponse)
    def list_patients(
        limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)
    ):
        items, total = get_store().list_patients(limit, offset)
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.get("/api/patients/{patient_id}", response_model=PatientDetailResponse)
    def get_patient(patient_id: UUID):
        patient = get_store().get_patient(str(patient_id))
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        return patient

    @app.post(
        "/api/patients/{patient_id}/episodes",
        response_model=EpisodeResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_episode(patient_id: UUID, request: EpisodeCreate):
        try:
            episode = get_store().create_episode(
                str(patient_id), str(uuid4()), request.label, to_utc(request.started_at)
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(status_code=409, detail="Episode conflicts with an existing record.") from error
        if episode is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        return episode

    @app.get("/api/patients/{patient_id}/episodes", response_model=list[EpisodeResponse])
    def list_episodes(patient_id: UUID):
        episodes = get_store().list_episodes(str(patient_id))
        if episodes is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        return episodes

    @app.post(
        "/api/patients/{patient_id}/observations",
        response_model=ObservationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_observation(patient_id: UUID, request: ObservationCreate, response: Response):
        missing_fields = sorted(feature for feature in FEATURES if request.values.get(feature) is None)
        try:
            observation, created = get_store().create_observation(
                str(request.observation_id),
                str(patient_id),
                str(request.episode_id),
                to_utc(request.observed_at),
                request.values,
                request.units,
                missing_fields,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except sqlite3.IntegrityError as error:
            raise HTTPException(
                status_code=409, detail="An observation already exists for this episode and timestamp."
            ) from error
        if not created:
            response.status_code = status.HTTP_200_OK
        return observation

    @app.get(
        "/api/patients/{patient_id}/observations", response_model=ObservationPageResponse
    )
    def list_observations(
        patient_id: UUID,
        episode_id: UUID | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        try:
            page = get_store().list_observations(
                str(patient_id), str(episode_id) if episode_id else None, limit, offset
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if page is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        items, total = page
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.post(
        "/api/patients/{patient_id}/assessments",
        response_model=AssessmentResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_assessment(patient_id: UUID, request: AssessmentCreate, response: Response):
        store = get_store()
        prior_id = store.get_assessment_by_id(str(request.assessment_id))
        if prior_id is not None:
            if prior_id["patient_id"] != str(patient_id) or prior_id["observation_id"] != str(request.observation_id):
                raise HTTPException(status_code=409, detail="assessment_id already belongs to another request.")
            response.status_code = status.HTTP_200_OK
            return prior_id
        observation = store.get_observation(str(request.observation_id), str(patient_id))
        if observation is None:
            raise HTTPException(status_code=404, detail="Observation not found for this patient.")

        engine = get_engine()
        existing = store.get_assessment(str(request.observation_id), engine.config_version)
        if existing is not None:
            response.status_code = status.HTTP_200_OK
            return existing

        with app.state.inference_lock:
            existing = store.get_assessment(str(request.observation_id), engine.config_version)
            if existing is not None:
                response.status_code = status.HTTP_200_OK
                return existing
            result = engine.evaluate(observation["values"])
            try:
                assessment = store.create_assessment(result, observation, str(request.assessment_id))
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
        return assessment

    @app.get(
        "/api/patients/{patient_id}/assessments", response_model=AssessmentPageResponse
    )
    def list_assessments(
        patient_id: UUID,
        episode_id: UUID | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        try:
            page = get_store().list_assessments(
                str(patient_id), str(episode_id) if episode_id else None, limit, offset
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if page is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        items, total = page
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    return app


app = create_app()
