"""Score one current patient observation with the frozen Qwen decision pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qwen_sepsis.data import FEATURES
from qwen_sepsis.explain import DISCLAIMER
from qwen_sepsis.inference_service import InferenceEngine, InferenceSettings
from qwen_sepsis.observation import prepare_observation as _prepare_observation


def prepare_observation(payload: dict, median_path: Path) -> tuple[pd.Series, list[str]]:
    medians = pd.read_csv(median_path, index_col=0)["train_median"]
    values = {name: value for name, value in payload.items() if name in FEATURES}
    unknown = set(payload).difference(set(FEATURES) | {"patient_id"})
    if unknown:
        raise ValueError(f"Campos não reconhecidos: {sorted(unknown)}")
    return _prepare_observation(values, medians)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstração acadêmica de risco por observação atual.")
    parser.add_argument("--input", type=Path, required=True, help="JSON com os dados atuais do paciente")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "qwen3-8b-sepsis-lora" / "final_adapter")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    unknown = set(payload).difference(set(FEATURES) | {"patient_id"})
    if unknown:
        raise ValueError(f"Campos não reconhecidos: {sorted(unknown)}")
    prepare_observation(payload, args.data_dir / "train_medians.csv")
    base = args.base_model or str(ROOT / "models" / "base" / "Qwen3-8B")
    engine = InferenceEngine.load(InferenceSettings(
        artifact_dir=args.artifact_dir,
        data_dir=args.data_dir,
        base_model=Path(base),
        adapter=args.adapter,
    ))
    values = {name: value for name, value in payload.items() if name in FEATURES}
    result = engine.evaluate(values)
    elevated = result["decision"] == "RISK_ELEVATED"
    result = {
        "patient_id": str(payload.get("patient_id", "demo")),
        "raw_score": result["raw_score"],
        "risk_score": result["risk_score"],
        "risk_score_meaning": result["risk_score_meaning"],
        "threshold": result["threshold"],
        "risk_level": "RISCO ELEVADO" if elevated else "BAIXO RISCO",
        "calibrator": result["calibration_method"],
        "calibrator_locked": True,
        "threshold_locked": True,
        "test_used_for_fitting": False,
        "imputed_features": result["imputed_features"],
        "message": result["message"],
        "clinical_guidance": (
            "O resultado indica maior risco com base nos dados clínicos informados e deve ser utilizado "
            "como suporte à avaliação profissional."
        ),
        "disclaimer": result["disclaimer"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()