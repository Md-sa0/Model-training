"""REST request and response contracts."""

from __future__ import annotations

import math
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

from .data import FEATURES

ClinicalValue = StrictInt | StrictFloat | None


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatientCreate(StrictRequest):
    external_id: str | None = Field(default=None, min_length=1, max_length=200)
    initial_episode_label: str | None = Field(default=None, max_length=200)
    episode_started_at: datetime | None = None


class EpisodeCreate(StrictRequest):
    label: str | None = Field(default=None, max_length=200)
    started_at: datetime | None = None


class ObservationCreate(StrictRequest):
    observation_id: UUID
    episode_id: UUID
    observed_at: datetime
    values: dict[str, ClinicalValue]
    units: dict[str, str] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at precisa incluir fuso horário")
        return value

    @field_validator("values")
    @classmethod
    def values_must_be_known_finite_features(cls, value: dict[str, ClinicalValue]):
        unknown = set(value).difference(FEATURES)
        if unknown:
            raise ValueError(f"Campos não reconhecidos: {sorted(unknown)}")
        for name, number in value.items():
            if number is not None and not math.isfinite(float(number)):
                raise ValueError(f"Valor não finito para {name}")
        return value

    @field_validator("units")
    @classmethod
    def units_must_be_known_and_nonempty(cls, value: dict[str, str]):
        unknown = set(value).difference(FEATURES)
        if unknown:
            raise ValueError(f"Unidades para campos não reconhecidos: {sorted(unknown)}")
        if any(not unit.strip() for unit in value.values()):
            raise ValueError("Unidades informadas não podem ser vazias")
        return value


class PatientRiskRequest(StrictRequest):
    values: dict[str, ClinicalValue]

    @field_validator("values")
    @classmethod
    def values_must_be_known_finite_features(cls, value: dict[str, ClinicalValue]):
        unknown = set(value).difference(FEATURES)
        if unknown:
            raise ValueError(f"Campos não reconhecidos: {sorted(unknown)}")
        for name, number in value.items():
            if number is not None and not math.isfinite(float(number)):
                raise ValueError(f"Valor não finito para {name}")
        return value


class AssessmentCreate(StrictRequest):
    assessment_id: UUID
    observation_id: UUID


class PatientResponse(BaseModel):
    patient_id: UUID
    external_id: str | None
    created_at: datetime


class EpisodeResponse(BaseModel):
    episode_id: UUID
    patient_id: UUID
    label: str | None
    started_at: datetime
    created_at: datetime


class PatientCreatedResponse(BaseModel):
    patient: PatientResponse
    episode: EpisodeResponse


class PatientDetailResponse(PatientResponse):
    episodes: list[EpisodeResponse]


class PatientPageResponse(BaseModel):
    items: list[PatientResponse]
    total: int
    limit: int
    offset: int


class ObservationResponse(BaseModel):
    observation_id: UUID
    patient_id: UUID
    episode_id: UUID
    observed_at: datetime
    received_at: datetime
    values: dict[str, ClinicalValue]
    units: dict[str, str]
    missing_fields: list[str]


class ObservationPageResponse(BaseModel):
    items: list[ObservationResponse]
    total: int
    limit: int
    offset: int


class AssessmentResponse(BaseModel):
    assessment_id: UUID
    observation_id: UUID
    patient_id: UUID
    episode_id: UUID
    evaluated_at: datetime
    raw_score: float
    risk_score: float
    risk_score_meaning: str
    threshold: float
    decision: str
    calibration_method: str
    model_version: str
    config_version: str
    imputed_features: list[str]
    message: str
    disclaimer: str


class AssessmentPageResponse(BaseModel):
    items: list[AssessmentResponse]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_error: str | None = None


class ConfigResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_ready: bool
    model_id: str | None = None
    model_version: str | None = None
    config_version: str | None = None
    calibration_method: str | None = None
    calibrator_locked: bool | None = None
    threshold: float | None = None
    threshold_locked: bool | None = None
    threshold_target_met: bool | None = None
    fitted_on: str | None = None
    test_used_for_fitting: bool | None = None
    features: list[str]
    dynamic_features: list[str]
    decision_semantics: str
    risk_score_meaning: str
    disclaimer: str


class PatientRiskResponse(BaseModel):
    raw_score: float
    risk_score: float
    risk_score_meaning: str
    threshold: float
    decision: str
    calibration_method: str
    model_version: str
    config_version: str
    imputed_features: list[str]
    message: str
    disclaimer: str

