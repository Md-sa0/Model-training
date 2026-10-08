"""Calibra o Qwen já treinado e avalia o teste por paciente.

O teste só é lido depois que calibrador, limiar e regra temporal foram gravados.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from qwen_sepsis import MODEL_ID
from qwen_sepsis.data import assert_disjoint_patient_ids, dataset_version, load_patient_ids, load_prepared_split
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


def print_split_summary(frame, split: str) -> None:
    labels = frame["label"].astype(int)
    patient_labels = frame.assign(__label=labels).groupby("Patient_ID", sort=False)["__label"].max()
    print(
        f"{split.upper()} DISTRIBUTION: samples={len(frame)}, positive={int(labels.sum())}, "
        f"negative={int((labels == 0).sum())}, positive_rate={labels.mean():.6f}, "
        f"negative_rate={(labels == 0).mean():.6f}, patients={len(patient_labels)}, "
        f"positive_patients={int(patient_labels.sum())}, negative_patients={int((patient_labels == 0).sum())}, "
        f"patient_positive_rate={patient_labels.mean():.6f}",
        flush=True,
    )


def print_smoke_rows(frame, records, calibrator=None, threshold: float | None = None) -> None:
    import torch

    if len(frame) != len(records):
        raise RuntimeError("Smoke test: desalinhamento entre prompt, índice e logits")
    for (_, row), record in zip(frame.iterrows(), records, strict=True):
        probability = float(record["raw_probability"])
        expected = float(torch.softmax(
            torch.tensor([record["logit_0"], record["logit_1"]], dtype=torch.float64), dim=0,
        )[1])
        if not np.isclose(probability, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError("Smoke test: raw_probability diverge do softmax dos logits 0/1")
        calibrated = float(calibrator.predict([probability])[0]) if calibrator is not None else None
        prediction = int(calibrated >= threshold) if calibrated is not None and threshold is not None else None
        print(
            "SMOKE "
            f"Patient_ID={row['Patient_ID']} Hour={int(row['Hour'])} label={int(row['label'])} "
            f"logit_0={record['logit_0']:.6f} logit_1={record['logit_1']:.6f} "
            f"raw_probability={probability:.8f} "
            f"calibrated_probability={calibrated if calibrated is not None else 'pending'} "
            f"threshold={threshold if threshold is not None else 'pending'} "
            f"prediction={prediction if prediction is not None else 'pending'}",
            flush=True,
        )


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
    parser.add_argument("--smoke-only", action="store_true", help="Pontua 10 pacientes e encerra antes da calibração completa.")
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
    validation_ids = set(validation["Patient_ID"])
    test_ids = load_patient_ids(args.data_dir, "test")
    assert_disjoint_patient_ids(validation_ids, test_ids)
    if not validation.groupby("Patient_ID", sort=False)["Hour"].apply(lambda hours: hours.is_monotonic_increasing).all():
        raise ValueError("Hour fora de ordem crescente dentro de validation")
    print(f"Patient leakage: PASS ({len(validation_ids & test_ids)} IDs compartilhados).", flush=True)
    print_split_summary(validation, "validation")

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

    patient_labels = validation.groupby("Patient_ID", sort=True)["SepsisLabel"].max()
    positive_ids = patient_labels[patient_labels == 1].index.tolist()
    negative_ids = patient_labels[patient_labels == 0].index.tolist()
    random.Random(seed).shuffle(positive_ids)
    random.Random(seed + 1).shuffle(negative_ids)
    smoke_ids = positive_ids[:5] + negative_ids[:5]
    if len(smoke_ids) < 10:
        remaining = [patient_id for patient_id in patient_labels.index if patient_id not in set(smoke_ids)]
        smoke_ids.extend(remaining[:10 - len(smoke_ids)])
    smoke_rows = []
    for patient_id in smoke_ids:
        patient_rows = validation[validation["Patient_ID"] == patient_id]
        if patient_id in positive_ids[:5]:
            patient_rows = patient_rows[patient_rows["label"].astype(int) == 1]
        smoke_rows.append(patient_rows.head(1))
    smoke = pd.concat(smoke_rows, ignore_index=True)
    if set(smoke["label"].astype(int)) != {0, 1}:
        raise RuntimeError("Smoke test precisa conter observações rotuladas 0 e 1")
    print(f"Smoke test antes da validação completa: {len(smoke_ids)} pacientes, {len(smoke)} observações.", flush=True)
    smoke_records, token_meta = score_prompts(
        model, tokenizer, smoke["prompt"], batch_size=int(runtime["batch_size"]),
        max_length=int(runtime["max_length"]), log=lambda message: print(message, flush=True),
    )
    if not all(0.0 <= record["raw_probability"] <= 1.0 for record in smoke_records):
        raise RuntimeError("Smoke test produziu probabilidade fora de [0, 1]")
    print(
        f"Tokenizer: token_id('0')={token_meta['token_id_0']} token_id('1')={token_meta['token_id_1']} "
        f"sequences={token_meta['token_ids_0']}/{token_meta['token_ids_1']}; "
        f"logits_shape={token_meta['logits_shape']} last_positions={token_meta['last_relevant_positions']}.",
        flush=True,
    )
    print_smoke_rows(smoke, smoke_records)
    if args.smoke_only:
        print("Smoke test concluído; inferência completa e test não foram executados.", flush=True)
        return

    print("Extraindo logits de 0 e 1 em toda a validation.", flush=True)
    records, token_meta = score_prompts(
        model, tokenizer, validation["prompt"], batch_size=int(runtime["batch_size"]),
        max_length=int(runtime["max_length"]), log=lambda message: print(message, flush=True),
    )
    validation = attach_raw_scores(validation, records)
    print(
        f"Tokens de classe: token_id('0')={token_meta['token_id_0']}, "
        f"token_id('1')={token_meta['token_id_1']}; sequences="
        f"{token_meta['token_ids_0']}/{token_meta['token_ids_1']}; logits_shape={token_meta['logits_shape']}.",
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
    manifest = {
        "model": str(base_model),
        "adapter": str(args.adapter),
        "dataset": dataset_version(args.data_dir),
        "validation_patients": int(validation["Patient_ID"].nunique()),
        "test_patients": int(len(test_ids)),
        "calibration_method": selection.calibrator.method,
        "threshold": float(selection.config["threshold"]),
        "target_sensitivity": float(selection.config["target_sensitivity"]),
        "temporal_strategy": selection.config["aggregation_strategy"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(selection.config["threshold_message"], flush=True)
    print(
        "MAXIMUM VALIDATION SENSITIVITY: "
        f"{selection.config['maximum_validation_sensitivity']:.4f} at "
        f"threshold={selection.config['maximum_sensitivity_threshold']:.2f}, "
        f"FP={selection.config['maximum_sensitivity_fp']}, FN={selection.config['maximum_sensitivity_fn']}",
        flush=True,
    )
    print("TEMPORAL STRATEGIES ON VALIDATION:", flush=True)
    print(selection.strategy_table[[
        "strategy", "sensitivity", "specificity", "PPV", "NPV", "F1", "FN", "FP",
        "median_lead_time", "mean_lead_time",
    ]].to_string(index=False), flush=True)
    selected_threshold_row = selection.threshold_table.loc[
        selection.threshold_table.threshold == selection.config["threshold"]
    ].iloc[0]
    print(
        f"Selected threshold={selection.config['threshold']:.2f}; "
        f"validation sensitivity={selection.config['achieved_sensitivity_validation_hourly']:.4f}; "
        f"specificity={selection.config['achieved_specificity_validation_hourly']:.4f}; "
        f"hourly FN={int(selected_threshold_row['FN'])}; hourly FP={int(selected_threshold_row['FP'])}.",
        flush=True,
    )
    if not selection.config["target_met"]:
        print("WARNING: Target sensitivity of 95% was not achieved.", flush=True)
    print(
        f"Calibrador: {selection.config['method']}. Estratégia: {selection.config['aggregation_strategy']}.",
        flush=True,
    )
    if not selection.config["target_met"]:
        print("A meta de sensibilidade horária não foi atingida na validação.", flush=True)
    if not selection.config["strategy_target_met"]:
        print("A meta de sensibilidade por paciente não foi atingida na validação.", flush=True)
    print("CALIBRATION METRICS ON VALIDATION:", flush=True)
    print("Model              Brier       ECE       LogLoss", flush=True)
    for name, values in selection.calibration_metrics.items():
        print(f"{name:<18} {values['brier']:.6f}  {values['ece']:.6f}  {values['log_loss']:.6f}", flush=True)
    print("SANITY SAMPLE AFTER VALIDATION CALIBRATION:", flush=True)
    print_smoke_rows(smoke, smoke_records, selection.calibrator, float(selection.config["threshold"]))

    calibrator, config = load_frozen_decision(args.output_dir)
    print("Validação congelada. O teste passa a ser lido somente para a avaliação final.", flush=True)
    test = load_prepared_split(args.data_dir, "test")
    assert_disjoint_patient_ids(validation_ids, test["Patient_ID"])
    if not test.groupby("Patient_ID", sort=False)["Hour"].apply(lambda hours: hours.is_monotonic_increasing).all():
        raise ValueError("Hour fora de ordem crescente dentro de test")
    print_split_summary(test, "test")
    test_records, test_tokens = score_prompts(
        model, tokenizer, test["prompt"], batch_size=int(runtime["batch_size"]),
        max_length=int(runtime["max_length"]), log=lambda message: print(message, flush=True),
    )
    if (test_tokens["token_id_0"], test_tokens["token_id_1"]) != (token_meta["token_id_0"], token_meta["token_id_1"]):
        raise RuntimeError("Os tokens de classe mudaram entre a validação e o teste.")
    test = attach_raw_scores(test, test_records)
    result = evaluate_test_frame(test, calibrator, config)
    report = build_final_report(result["metrics"], config, context, result["hourly_metrics"])
    write_evaluation_artifacts(args.output_dir, result, report)
    print(
        "Teste por paciente: "
        f"sensibilidade={report['sensitivity']}, "
        f"falsos negativos={report['false_negatives']}, "
        f"falsos positivos={report['false_positives']}, "
        f"lead time mediano={report['median_lead_time']}.",
        flush=True,
    )
    print("HOURLY EVALUATION:", flush=True)
    print(json.dumps(result["hourly_metrics"], indent=2, ensure_ascii=False), flush=True)
    print("PATIENT-LEVEL EVALUATION:", flush=True)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False), flush=True)
    for warning in report["warnings"]:
        print(warning, flush=True)
    print(DISCLAIMER, flush=True)
    print(f"Relatório: {args.output_dir / 'evaluation' / 'final_patient_report.json'}", flush=True)


if __name__ == "__main__":
    main()
