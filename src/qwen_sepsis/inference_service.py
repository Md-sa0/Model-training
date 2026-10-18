"""Reusable service wrapper around the frozen sepsis inference pipeline."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import DYNAMIC, FEATURES, format_prompt
from .explain import DISCLAIMER, RISK_SCORE_MEANING
from .observation import prepare_observation


@dataclass(frozen=True)
class InferenceSettings:
    artifact_dir: Path
    data_dir: Path
    base_model: Path
    adapter: Path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_version(model_id: str, base_model: Path, adapter: Path) -> str:
    entries: list[dict[str, object]] = [{"model_id": model_id}]
    for label, root, manifests in (
        ("base", base_model, ("config.json", "model.safetensors.index.json")),
        (
            "adapter",
            adapter,
            (
                "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja",
            ),
        ),
    ):
        for name in manifests:
            path = root / name
            if path.is_file():
                entries.append({"component": label, "file": name, "sha256": _file_digest(path)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class InferenceEngine:
    model: object
    tokenizer: object
    calibrator: object
    config: dict[str, object]
    medians: pd.Series
    model_version: str
    config_version: str

    @classmethod
    def load(cls, settings: InferenceSettings) -> "InferenceEngine":
        from .calibration import load_calibrator
        from .inference import load_sepsis_model, render_prompt, score_prompts
        from .train import SYSTEM

        calibration_dir = settings.artifact_dir / "calibration"
        config_path = calibration_dir / "config.json"
        calibrator_path = calibration_dir / "calibrator.pkl"
        medians_path = settings.data_dir / "train_medians.csv"
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if config.get("calibrator_locked") is not True or config.get("threshold_locked") is not True:
            raise RuntimeError("A inferência exige calibrador e threshold congelados.")
        if config.get("fitted_on") != "validation":
            raise RuntimeError("O calibrador configurado não foi ajustado na validação.")
        threshold = float(config["threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise RuntimeError("Threshold congelado inválido.")
        calibrator = load_calibrator(calibrator_path)
        if calibrator.method != config["method"]:
            raise RuntimeError("Calibrador e configuração congelada não correspondem.")
        medians_frame = pd.read_csv(medians_path, index_col=0)
        if "train_median" not in medians_frame or not set(FEATURES).issubset(medians_frame.index):
            raise RuntimeError("O arquivo de medianas de treino não contém todas as features.")
        medians = medians_frame["train_median"].astype(float).reindex(FEATURES)
        if not medians.map(math.isfinite).all():
            raise RuntimeError("As medianas de treino devem ser valores finitos.")

        model, tokenizer = load_sepsis_model(str(settings.base_model), settings.adapter)
        model_version = _model_version(str(config["model_id"]), settings.base_model, settings.adapter)
        inference_contract = {
            "system_prompt": SYSTEM,
            "prompt_formatter": inspect.getsource(format_prompt),
            "prompt_renderer": inspect.getsource(render_prompt),
            "scorer": inspect.getsource(score_prompts),
            "features": FEATURES,
            "dynamic_features": DYNAMIC,
            "max_length": 512,
        }
        version_material = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "calibrator_sha256": _file_digest(calibrator_path),
            "medians_sha256": _file_digest(medians_path),
            "model_version": model_version,
            "inference_contract_sha256": hashlib.sha256(
                json.dumps(inference_contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        config_version = hashlib.sha256(
            json.dumps(version_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(model, tokenizer, calibrator, config, medians, model_version, config_version)

    def public_config(self, model_ready: bool) -> dict[str, object]:
        return {
            "model_ready": model_ready,
            "model_id": self.config["model_id"],
            "model_version": self.model_version,
            "config_version": self.config_version,
            "calibration_method": self.calibrator.method,
            "calibrator_locked": True,
            "threshold": float(self.config["threshold"]),
            "threshold_locked": True,
            "threshold_target_met": bool(self.config.get("target_met", False)),
            "fitted_on": self.config["fitted_on"],
            "test_used_for_fitting": bool(self.config.get("test_used_for_fitting", False)),
            "features": FEATURES,
            "dynamic_features": DYNAMIC,
            "decision_semantics": "RISK_ELEVATED quando risk_score >= threshold; caso contrário LOW_RISK.",
            "risk_score_meaning": RISK_SCORE_MEANING,
            "disclaimer": DISCLAIMER,
        }

    def evaluate(self, values: dict[str, float | int | None]) -> dict[str, object]:
        from .inference import score_prompts

        observation, imputed_features = prepare_observation(values, self.medians)
        prompt = format_prompt(observation)
        records, _metadata = score_prompts(self.model, self.tokenizer, [prompt], batch_size=1, max_length=512)
        raw_score = float(records[0]["raw_probability"])
        risk_score = float(self.calibrator.predict([raw_score])[0])
        threshold = float(self.config["threshold"])
        elevated = risk_score >= threshold
        return {
            "raw_score": raw_score,
            "risk_score": risk_score,
            "risk_score_meaning": RISK_SCORE_MEANING,
            "threshold": threshold,
            "decision": "RISK_ELEVATED" if elevated else "LOW_RISK",
            "calibration_method": self.calibrator.method,
            "model_version": self.model_version,
            "config_version": self.config_version,
            "imputed_features": imputed_features,
            "message": (
                "Risco elevado de desenvolvimento de sepse."
                if elevated else "Baixo risco estimado de desenvolvimento de sepse neste momento."
            ),
            "disclaimer": f"{DISCLAIMER} Este resultado não constitui diagnóstico médico.",
        }
