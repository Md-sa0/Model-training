"""Calibra o Qwen já treinado e avalia o teste por paciente.

O teste só é lido depois que calibrador, limiar e regra temporal foram gravados.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from qwen_sepsis import MODEL_ID
from qwen_sepsis.data import dataset_version, load_prepared_split
from qwen_sepsis.explain import DISCLAIMER
from qwen_sepsis.inference import load_sepsis_model, score_prompts
from qwen_sepsis.pipeline import (
    attach_raw_scores,
    build_final_report,
    evaluate_test_frame,
    load_frozen_decision,
    save_validation_artifacts,
    select_from_validation,
    write_evaluation_artifacts,
)


def load_runtime_config(path: Path) -> dict:
    defaults = {
        "target_sensitivity": 0.95,
        "high_risk_probability": 0.7,
        "trend_delta": 0.02,
        "seed": 42,
        "ece_bins": 10,
        "batch_size": 4,
        "max_length": 512,
    }
    if path.exists():
        defaults.update(json.loads(path.read_text(encoding="utf-8")))
    return defaults


def model_context(base_model: str, adapter: Path, token_meta: dict, data_dir: Path, seed: int) -> dict:
    context = {
        "model_id": MODEL_ID,
        "base_model_path": base_model,
        "adapter_path": str(adapter),
        "dataset": dataset_version(data_dir),
        "seed": seed,
        "tokens": token_meta,
        "time_axis": "Hour",
        "system_role": DISCLAIMER,
    }
    adapter_config = adapter / "adapter_config.json"
    if adapter_config.exists():
        context["adapter_config"] = json.loads(adapter_config.read_text(encoding="utf-8"))
    return context


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibra o Qwen3-8B já treinado, escolhe o limiar na validação e avalia o teste por paciente.",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "qwen3-8b-sepsis-lora" / "final_adapter")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "calibration.json")
    parser.add_argument("--target-sensitivity", type=float, default=None)
    parser.add_argument("--high-risk-probability", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    if args.target_sensitivity is not None:
        runtime["target_sensitivity"] = args.target_sensitivity
    if args.high_risk_probability is not None:
        runtime["high_risk_probability"] = args.high_risk_probability
    if args.batch_size is not None:
        runtime["batch_size"] = args.batch_size
    if args.seed is not None:
        runtime["seed"] = args.seed
    seed = int(runtime["seed"])
    random.seed(seed)
    np.random.seed(seed)

    test_files = [args.data_dir / "test.jsonl", args.data_dir / "test_patients.jsonl"]
    missing = [path for path in test_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "O teste precisa existir, mas só será lido depois da validação: "
            + ", ".join(str(path) for path in missing)
        )
    validation = load_prepared_split(args.data_dir, "validation")

    local_base = ROOT / "models" / "base" / "Qwen3-8B"
    base_model = args.base_model or (str(local_base) if local_base.exists() else MODEL_ID)
    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter não encontrado: {args.adapter}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. A calibração usa o mesmo carregamento 4 bits do treino.")
    torch.manual_seed(seed)
    print("Carregando o Qwen3-8B e o adapter já treinado, sem novo treino.", flush=True)
    model, tokenizer = load_sepsis_model(base_model, args.adapter)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Extraindo logits de 0 e 1 na validação.", flush=True)
    records, token_meta = score_prompts(
        model, tokenizer, validation["prompt"], batch_size=int(runtime["batch_size"]),
        max_length=int(runtime["max_length"]), log=lambda message: print(message, flush=True),
    )
    validation = attach_raw_scores(validation, records)
    print(
        f"Tokens de classe: 0 -> {token_meta['token_ids_0']}, 1 -> {token_meta['token_ids_1']}.",
        flush=True,
    )

    selection = select_from_validation(
        validation,
        target_sensitivity=float(runtime["target_sensitivity"]),
        seed=seed,
        high_risk_probability=float(runtime["high_risk_probability"]),
        trend_delta=float(runtime["trend_delta"]),
        ece_bins=int(runtime["ece_bins"]),
        split="validation",
    )
    context = model_context(base_model, args.adapter, token_meta, args.data_dir, seed)
    save_validation_artifacts(args.output_dir, selection, context)
    print(selection.config["threshold_message"], flush=True)
    print(
        f"Calibrador: {selection.config['method']}. Estratégia: {selection.config['aggregation_strategy']}.",
        flush=True,
    )
    if not selection.config["target_met"]:
        print("A meta de sensibilidade horária não foi atingida na validação.", flush=True)
    if not selection.config["strategy_target_met"]:
        print("A meta de sensibilidade por paciente não foi atingida na validação.", flush=True)

    calibrator, config = load_frozen_decision(args.output_dir)
    print("Validação congelada. O teste passa a ser lido somente para a avaliação final.", flush=True)
    test = load_prepared_split(args.data_dir, "test")
    test_records, test_tokens = score_prompts(
        model, tokenizer, test["prompt"], batch_size=int(runtime["batch_size"]),
        max_length=int(runtime["max_length"]), log=lambda message: print(message, flush=True),
    )
    if (test_tokens["token_id_0"], test_tokens["token_id_1"]) != (token_meta["token_id_0"], token_meta["token_id_1"]):
        raise RuntimeError("Os tokens de classe mudaram entre a validação e o teste.")
    test = attach_raw_scores(test, test_records)
    result = evaluate_test_frame(test, calibrator, config)
    report = build_final_report(result["metrics"], config, context)
    write_evaluation_artifacts(args.output_dir, result, report)
    print(
        "Teste por paciente: "
        f"sensibilidade={report['sensitivity']}, "
        f"falsos negativos={report['false_negatives']}, "
        f"falsos positivos={report['false_positives']}, "
        f"lead time mediano={report['median_lead_time']}.",
        flush=True,
    )
    for warning in report["warnings"]:
        print(warning, flush=True)
    print(DISCLAIMER, flush=True)
    print(f"Relatório: {args.output_dir / 'evaluation' / 'final_patient_report.json'}", flush=True)


if __name__ == "__main__":
    main()
