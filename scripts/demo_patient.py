"""Score one current patient observation with the frozen Qwen decision pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qwen_sepsis.calibration import load_calibrator
from qwen_sepsis.data import DYNAMIC, FEATURES, format_prompt
from qwen_sepsis.explain import DISCLAIMER
from qwen_sepsis.inference import load_sepsis_model, score_prompts


def prepare_observation(payload: dict, median_path: Path) -> tuple[pd.Series, list[str]]:
    unknown = set(payload).difference(set(FEATURES) | {"patient_id"})
    if unknown:
        raise ValueError(f"Campos não reconhecidos: {sorted(unknown)}")
    medians = pd.read_csv(median_path, index_col=0)["train_median"]
    row = {}
    missing = []
    for feature in FEATURES:
        value = payload.get(feature)
        if value is None:
            value = float(medians[feature])
            if feature in DYNAMIC:
                row[f"{feature}_missing"] = 1
                missing.append(feature)
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Valor inválido para {feature}")
            if feature in DYNAMIC:
                row[f"{feature}_missing"] = 0
        row[feature] = value
    return pd.Series(row), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstração acadêmica de risco por observação atual.")
    parser.add_argument("--input", type=Path, required=True, help="JSON com os dados atuais do paciente")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "qwen3-8b-sepsis-lora" / "final_adapter")
    args = parser.parse_args()

    calibration_dir = args.artifact_dir / "calibration"
    config = json.loads((calibration_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("calibrator_locked") is not True or config.get("threshold_locked") is not True:
        raise RuntimeError("A demonstração exige calibrador e threshold congelados.")
    calibrator = load_calibrator(calibration_dir / "calibrator.pkl")
    if calibrator.method != config["method"] or config.get("fitted_on") != "validation":
        raise RuntimeError("Calibrador e configuração congelada não correspondem.")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    observation, missing = prepare_observation(payload, args.data_dir / "train_medians.csv")
    prompt = format_prompt(observation)
    base = args.base_model or str(ROOT / "models" / "base" / "Qwen3-8B")
    model, tokenizer = load_sepsis_model(base, args.adapter)
    records, _metadata = score_prompts(model, tokenizer, [prompt], batch_size=1, max_length=512)
    raw_probability = float(records[0]["raw_probability"])
    calibrated_probability = float(calibrator.predict([raw_probability])[0])
    threshold = float(config["threshold"])
    elevated = calibrated_probability >= threshold
    result = {
        "patient_id": str(payload.get("patient_id", "demo")),
        "raw_score": raw_probability,
        "risk_score": calibrated_probability,
        "risk_score_meaning": "Probabilidade calibrada na validation; não é diagnóstico nem chance clínica validada.",
        "threshold": threshold,
        "risk_level": "RISCO ELEVADO" if elevated else "BAIXO RISCO",
        "calibrator": calibrator.method,
        "calibrator_locked": True,
        "threshold_locked": True,
        "test_used_for_fitting": False,
        "imputed_features": missing,
        "message": (
            "Risco elevado de desenvolvimento de sepse."
            if elevated else "Baixo risco estimado de desenvolvimento de sepse neste momento."
        ),
        "clinical_guidance": (
            "O resultado indica maior risco com base nos dados clínicos informados e deve ser utilizado "
            "como suporte à avaliação profissional."
        ),
        "disclaimer": f"{DISCLAIMER} Este resultado não constitui diagnóstico médico.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()