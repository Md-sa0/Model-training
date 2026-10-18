"""SQLite persistence for patients, inpatient episodes, observations and assessments."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    external_id TEXT UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    label TEXT,
                    started_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (episode_id, patient_id)
                );
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    units_json TEXT NOT NULL,
                    missing_fields_json TEXT NOT NULL,
                    UNIQUE (episode_id, observed_at),
                    UNIQUE (observation_id, patient_id, episode_id),
                    FOREIGN KEY (episode_id, patient_id) REFERENCES episodes(episode_id, patient_id)
                );
                CREATE INDEX IF NOT EXISTS observations_patient_time
                    ON observations(patient_id, observed_at);
                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    observation_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    raw_score REAL NOT NULL CHECK(raw_score >= 0 AND raw_score <= 1),
                    risk_score REAL NOT NULL CHECK(risk_score >= 0 AND risk_score <= 1),
                    threshold REAL NOT NULL CHECK(threshold >= 0 AND threshold <= 1),
                    decision TEXT NOT NULL CHECK(decision IN ('RISK_ELEVATED', 'LOW_RISK')),
                    model_version TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    calibration_method TEXT NOT NULL,
                    imputed_features_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    UNIQUE (observation_id, config_version),
                    FOREIGN KEY (observation_id, patient_id, episode_id)
                        REFERENCES observations(observation_id, patient_id, episode_id)
                );
                CREATE INDEX IF NOT EXISTS assessments_patient_time
                    ON assessments(patient_id, evaluated_at);
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _patient(row: sqlite3.Row) -> dict[str, object]:
        return {
            "patient_id": row["patient_id"],
            "external_id": row["external_id"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _episode(row: sqlite3.Row) -> dict[str, object]:
        return {
            "episode_id": row["episode_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "started_at": row["started_at"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _observation(row: sqlite3.Row) -> dict[str, object]:
        return {
            "observation_id": row["observation_id"],
            "patient_id": row["patient_id"],
            "episode_id": row["episode_id"],
            "observed_at": row["observed_at"],
            "received_at": row["received_at"],
            "values": json.loads(row["values_json"]),
            "units": json.loads(row["units_json"]),
            "missing_fields": json.loads(row["missing_fields_json"]),
        }

    @staticmethod
    def _assessment(row: sqlite3.Row) -> dict[str, object]:
        return json.loads(row["result_json"])

    def create_patient(self, patient_id: str, episode_id: str, external_id: str | None,
                       episode_label: str | None, started_at: str) -> dict[str, object]:
        created_at = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO patients(patient_id, external_id, created_at) VALUES (?, ?, ?)",
                (patient_id, external_id, created_at),
            )
            connection.execute(
                "INSERT INTO episodes(episode_id, patient_id, label, started_at, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (episode_id, patient_id, episode_label, started_at, created_at),
            )
        return {
            "patient": {"patient_id": patient_id, "external_id": external_id, "created_at": created_at},
            "episode": {
                "episode_id": episode_id, "patient_id": patient_id, "label": episode_label,
                "started_at": started_at, "created_at": created_at,
            },
        }

    def list_patients(self, limit: int, offset: int) -> tuple[list[dict[str, object]], int]:
        with self.connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM patients ORDER BY created_at, patient_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._patient(row) for row in rows], int(total)

    def get_patient(self, patient_id: str) -> dict[str, object] | None:
        with self.connection() as connection:
            patient = connection.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
            if patient is None:
                return None
            episodes = connection.execute(
                "SELECT * FROM episodes WHERE patient_id = ? ORDER BY started_at, episode_id", (patient_id,)
            ).fetchall()
        return {**self._patient(patient), "episodes": [self._episode(row) for row in episodes]}

    def create_episode(self, patient_id: str, episode_id: str, label: str | None,
                       started_at: str) -> dict[str, object] | None:
        created_at = utc_now()
        with self.transaction() as connection:
            patient = connection.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
            if patient is None:
                return None
            connection.execute(
                "INSERT INTO episodes(episode_id, patient_id, label, started_at, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (episode_id, patient_id, label, started_at, created_at),
            )
        return {
            "episode_id": episode_id, "patient_id": patient_id, "label": label,
            "started_at": started_at, "created_at": created_at,
        }

    def list_episodes(self, patient_id: str) -> list[dict[str, object]] | None:
        with self.connection() as connection:
            if connection.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone() is None:
                return None
            rows = connection.execute(
                "SELECT * FROM episodes WHERE patient_id = ? ORDER BY started_at, episode_id", (patient_id,)
            ).fetchall()
        return [self._episode(row) for row in rows]

    def create_observation(
        self, observation_id: str, patient_id: str, episode_id: str, observed_at: str,
        values: dict[str, object], units: dict[str, str], missing_fields: list[str],
    ) -> tuple[dict[str, object], bool]:
        values_json = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        units_json = json.dumps(units, sort_keys=True, separators=(",", ":"), allow_nan=False)
        missing_json = json.dumps(missing_fields, separators=(",", ":"))
        received_at = utc_now()
        with self.transaction() as connection:
            prior = connection.execute(
                "SELECT * FROM observations WHERE observation_id = ?", (observation_id,)
            ).fetchone()
            if prior is not None:
                same_request = (
                    prior["patient_id"] == patient_id
                    and prior["episode_id"] == episode_id
                    and prior["observed_at"] == observed_at
                    and prior["values_json"] == values_json
                    and prior["units_json"] == units_json
                )
                if not same_request:
                    raise ValueError("observation_id já utilizado com outro conteúdo")
                return self._observation(prior), False
            episode = connection.execute(
                "SELECT 1 FROM episodes WHERE episode_id = ? AND patient_id = ?",
                (episode_id, patient_id),
            ).fetchone()
            if episode is None:
                raise LookupError("Episódio não encontrado para este paciente")
            connection.execute(
                """INSERT INTO observations(
                    observation_id, episode_id, patient_id, observed_at, received_at,
                    values_json, units_json, missing_fields_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (observation_id, episode_id, patient_id, observed_at, received_at,
                 values_json, units_json, missing_json),
            )
            row = connection.execute(
                "SELECT * FROM observations WHERE observation_id = ?", (observation_id,)
            ).fetchone()
        return self._observation(row), True

    def get_observation(self, observation_id: str, patient_id: str) -> dict[str, object] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM observations WHERE observation_id = ? AND patient_id = ?",
                (observation_id, patient_id),
            ).fetchone()
        return None if row is None else self._observation(row)

    def list_observations(self, patient_id: str, episode_id: str | None,
                          limit: int, offset: int) -> tuple[list[dict[str, object]], int] | None:
        with self.connection() as connection:
            if connection.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone() is None:
                return None
            if episode_id is not None and connection.execute(
                "SELECT 1 FROM episodes WHERE patient_id = ? AND episode_id = ?", (patient_id, episode_id)
            ).fetchone() is None:
                raise LookupError("Episódio não encontrado para este paciente")
            where = "patient_id = ?" + (" AND episode_id = ?" if episode_id else "")
            parameters: tuple[object, ...] = (patient_id, episode_id) if episode_id else (patient_id,)
            total = connection.execute(f"SELECT COUNT(*) FROM observations WHERE {where}", parameters).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM observations WHERE {where} ORDER BY observed_at, observation_id LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [self._observation(row) for row in rows], int(total)

    def get_assessment(self, observation_id: str, config_version: str) -> dict[str, object] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM assessments WHERE observation_id = ? AND config_version = ?",
                (observation_id, config_version),
            ).fetchone()
        return None if row is None else self._assessment(row)

    def get_assessment_by_id(self, assessment_id: str) -> dict[str, object] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
        return None if row is None else self._assessment(row)

    def create_assessment(self, result: dict[str, object], observation: dict[str, object],
                          assessment_id: str) -> dict[str, object]:
        with self.transaction() as connection:
            prior_id = connection.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            if prior_id is not None:
                existing = self._assessment(prior_id)
                if existing["observation_id"] != observation["observation_id"]:
                    raise ValueError("assessment_id já utilizado para outra observação")
                return existing
            prior_assessment = connection.execute(
                "SELECT * FROM assessments WHERE observation_id = ? AND config_version = ?",
                (observation["observation_id"], result["config_version"]),
            ).fetchone()
            if prior_assessment is not None:
                return self._assessment(prior_assessment)
            response = {
                "assessment_id": assessment_id,
                "observation_id": observation["observation_id"],
                "patient_id": observation["patient_id"],
                "episode_id": observation["episode_id"],
                "evaluated_at": utc_now(),
                **result,
            }
            connection.execute(
                """INSERT INTO assessments(
                    assessment_id, observation_id, patient_id, episode_id, evaluated_at,
                    raw_score, risk_score, threshold, decision, model_version, config_version,
                    calibration_method, imputed_features_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    assessment_id, observation["observation_id"], observation["patient_id"],
                    observation["episode_id"], response["evaluated_at"], result["raw_score"],
                    result["risk_score"], result["threshold"], result["decision"],
                    result["model_version"], result["config_version"], result["calibration_method"],
                    json.dumps(result["imputed_features"]), json.dumps(response, allow_nan=False),
                ),
            )
        return response

    def list_assessments(self, patient_id: str, episode_id: str | None,
                         limit: int, offset: int) -> tuple[list[dict[str, object]], int] | None:
        with self.connection() as connection:
            if connection.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone() is None:
                return None
            if episode_id is not None and connection.execute(
                "SELECT 1 FROM episodes WHERE patient_id = ? AND episode_id = ?", (patient_id, episode_id)
            ).fetchone() is None:
                raise LookupError("Episódio não encontrado para este paciente")
            where = "patient_id = ?" + (" AND episode_id = ?" if episode_id else "")
            parameters: tuple[object, ...] = (patient_id, episode_id) if episode_id else (patient_id,)
            total = connection.execute(f"SELECT COUNT(*) FROM assessments WHERE {where}", parameters).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM assessments WHERE {where} ORDER BY evaluated_at, assessment_id LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [self._assessment(row) for row in rows], int(total)
