from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

REVISION = "63b6d1472e3d2aa6fc9870e0fca54d5eadec6eae"
DATA_URL = f"https://huggingface.co/datasets/Martinsintel/sepsis-dataset/resolve/{REVISION}/sepsis_dataset.csv"
EXPECTED_SHA256 = "af7f8236de8ebb21fdd75c579863c0ba6bffbaac202028955d43a0e370e37e3e"
FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "WBC", "Lactate", "Creatinine", "Platelets", "Age", "Gender", "ICULOS"]
DYNAMIC = [c for c in FEATURES if c not in {"Age", "Gender", "ICULOS"}]
REQUIRED = {"Patient_ID", "Hour", "SepsisLabel", *FEATURES}


def download_dataset(path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urlopen(DATA_URL, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(path)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Hash do dataset diferente da revisão auditada")
    return {"url": DATA_URL, "revision": REVISION, "sha256": digest, "bytes": path.stat().st_size}


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    missing = sorted(REQUIRED.difference(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    invalid = df[["Patient_ID", "Hour", "SepsisLabel"]].isna().any(axis=1)
    invalid_count = int(invalid.sum())
    clean = df.loc[~invalid].copy()
    if not clean.SepsisLabel.isin([0, 1]).all():
        raise ValueError("SepsisLabel deve ser binário")
    if (clean.Hour < 0).any() or np.isinf(clean[["Hour", *FEATURES]].to_numpy(dtype=float)).any():
        raise ValueError("Hora negativa ou valor infinito")
    if clean.duplicated(["Patient_ID", "Hour"]).any():
        raise ValueError("Duplicata paciente-hora")
    return clean, invalid_count


def patient_split(df: pd.DataFrame, seed: int = 42) -> dict[object, str]:
    labels = df.groupby("Patient_ID", sort=True).SepsisLabel.max().astype(int)
    rng = np.random.default_rng(seed)
    assignment: dict[object, str] = {}
    for label in [0, 1]:
        ids = labels[labels == label].index.to_numpy(copy=True)
        rng.shuffle(ids)
        train_end = int(0.70 * len(ids))
        val_end = train_end + int(0.15 * len(ids))
        for patient in ids[:train_end]: assignment[patient] = "train"
        for patient in ids[train_end:val_end]: assignment[patient] = "validation"
        for patient in ids[val_end:]: assignment[patient] = "test"
    return assignment


def temporal_fill(df: pd.DataFrame, hours: int = 6) -> pd.DataFrame:
    work = df.sort_values(["Patient_ID", "Hour"], kind="stable").copy()
    for column in DYNAMIC:
        work[f"{column}_missing"] = work[column].isna().astype("int8")
        observed_at = work.Hour.where(work[column].notna())
        last_at = observed_at.groupby(work.Patient_ID).ffill()
        previous = work.groupby("Patient_ID")[column].ffill()
        work[column] = previous.where((work.Hour - last_at) <= hours)
    return work


def _sample_rows(df: pd.DataFrame, max_rows: int | None, seed: int, negative_ratio: int | None) -> pd.DataFrame:
    if negative_ratio is not None:
        positives = df[df.SepsisLabel == 1]
        negatives = df[df.SepsisLabel == 0].sample(n=min(len(df[df.SepsisLabel == 0]), len(positives) * negative_ratio), random_state=seed)
        df = pd.concat([positives, negatives]).sample(frac=1, random_state=seed)
    if max_rows and len(df) > max_rows:
        positives = df[df.SepsisLabel == 1]
        positive_n = min(len(positives), max(1, round(max_rows * df.SepsisLabel.mean())))
        negative_n = max_rows - positive_n
        df = pd.concat([
            positives.sample(n=positive_n, random_state=seed),
            df[df.SepsisLabel == 0].sample(n=negative_n, random_state=seed),
        ]).sample(frac=1, random_state=seed)
    return df


def format_prompt(row: pd.Series) -> str:
    values = ", ".join(f"{name}={float(row[name]):.4g}" for name in FEATURES)
    missing = ", ".join(name for name in DYNAMIC if int(row[f"{name}_missing"]) == 1) or "nenhuma"
    return (
        "Classifique esta observação horária de UTI usando somente os dados fornecidos. "
        "Responda apenas 0 (sem rótulo de sepse) ou 1 (rótulo de sepse). "
        f"Valores: {values}. Variáveis originalmente ausentes: {missing}."
    )


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for _, row in df.iterrows():
            target.write(json.dumps({"prompt": format_prompt(row), "label": str(int(row.SepsisLabel))}, ensure_ascii=False) + "\n")


def prepare_dataset(raw_path: Path, output_dir: Path, seed: int = 42, train_negative_ratio: int = 3,
                    max_train: int | None = 40000, max_eval: int | None = 20000) -> dict[str, object]:
    source = pd.read_csv(raw_path)
    raw_rows = len(source)
    if "Unnamed: 0" in source.columns:
        source = source.drop(columns="Unnamed: 0")
    source, invalid_rows = validate(source)
    assignment = patient_split(source, seed)
    work = source[["Patient_ID", "Hour", *FEATURES, "SepsisLabel"]].copy()
    work["split"] = work.Patient_ID.map(assignment)
    work = temporal_fill(work)
    medians = work.loc[work.split == "train", FEATURES].median()
    if medians.isna().any():
        raise ValueError("Feature inteiramente ausente no treino")
    work[FEATURES] = work[FEATURES].fillna(medians)
    counts: dict[str, object] = {}
    for split in ["train", "validation", "test"]:
        part = work[work.split == split]
        part = _sample_rows(part, max_train if split == "train" else max_eval, seed, train_negative_ratio if split == "train" else None)
        write_jsonl(part, output_dir / f"{split}.jsonl")
        counts[split] = {"rows": len(part), "positive": int(part.SepsisLabel.sum()), "prevalence": float(part.SepsisLabel.mean())}
    metadata = {"raw_rows": raw_rows, "valid_rows": len(source), "invalid_rows": invalid_rows, "patients": int(source.Patient_ID.nunique()), "splits": counts}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    medians.rename("train_median").to_csv(output_dir / "train_medians.csv")
    return metadata

