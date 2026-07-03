from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


OPERATIONAL_STATES = {"normal", "baseline", "teste", "acelerando", "motor_desligado"}

_TYPO_REPLACEMENTS = {
    "normla": "normal",
    "mortor": "motor",
    "desabalanceado": "desbalanceado",
    "desbanlanceado": "desbalanceado",
    "ddesbalanceado": "desbalanceado",
    "dedesbalanceado": "desbalanceado",
    "desbalanceamento": "desbalanceado",
    "cockecocked": "cocked",
    "tes": "teste",
}

_NOISE_TOKENS = {
    "new",
    "novo",
    "nova",
    "antigo",
    "antiga",
    "carga",
    "adxl",
    "pos",
}

NUMERIC_COLUMNS = [
    "z_rms_velocity_in_s",
    "z_rms_velocity_mm_s",
    "temperature_f",
    "temperature_c",
    "x_rms_velocity_in_s",
    "x_rms_velocity_mm_s",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_kurtosis",
    "x_kurtosis",
    "z_crest_factor",
    "x_crest_factor",
    "z_peak_velocity_in_s",
    "z_peak_velocity_mm_s",
    "x_peak_velocity_in_s",
    "x_peak_velocity_mm_s",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
    "rpm",
]


@dataclass(frozen=True)
class FaultSummary:
    fault_raw: str
    fault_normalized: str
    is_operational_state: bool


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _clean_token(token: str) -> str:
    token = _TYPO_REPLACEMENTS.get(token, token)
    token = re.sub(r"^\d+", "", token)
    return token


def normalize_fault_label(value: Any) -> str:
    """Normalize noisy operator labels while preserving the fault meaning."""
    if value is None or pd.isna(value):
        return "unknown"

    text = _strip_accents(str(value).lower().strip())
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    tokens = [_clean_token(token) for token in text.split("_") if token]
    tokens = [token for token in tokens if token and token not in _NOISE_TOKENS and not token.isdigit()]

    token_set = set(tokens)

    if "normal" in token_set:
        return "normal"
    if "baseline" in token_set:
        return "baseline"
    if "teste" in token_set:
        return "teste"
    if "acelerando" in token_set:
        return "acelerando"
    if {"motor", "desligado"}.issubset(token_set):
        return "motor_desligado"

    if "desbalanceado" in token_set:
        return "desbalanceado"
    if "desalinhado" in token_set:
        return "desalinhado"
    if {"falta", "fase"}.issubset(token_set):
        return "falta_fase"
    if "ventoinha" in token_set:
        return "ventoinha"
    if "polia" in token_set:
        return "polia"
    if "correia" in token_set:
        return "correia"

    if "rolamento" in token_set:
        if "outer" in token_set:
            return "rolamento_outer"
        if "inner" in token_set:
            return "rolamento_inner"
        if "ball" in token_set:
            return "rolamento_ball"
        if "combination" in token_set or "comb" in token_set:
            return "rolamento_combination"
        return "rolamento"

    if "eccentric" in token_set:
        return "eccentric_rotor"
    if "cocked" in token_set:
        return "cocked_rotor"

    return "_".join(tokens) if tokens else "unknown"


def summarize_fault(value: Any) -> FaultSummary:
    fault_raw = "" if value is None or pd.isna(value) else str(value)
    fault_normalized = normalize_fault_label(fault_raw)
    return FaultSummary(
        fault_raw=fault_raw,
        fault_normalized=fault_normalized,
        is_operational_state=fault_normalized in OPERATIONAL_STATES,
    )


def load_events(csv_path: str | Path) -> pd.DataFrame:
    """Load banner.csv and add normalized fault fields for downstream steps."""
    df = pd.read_csv(csv_path)
    if "fault" not in df.columns:
        raise ValueError("CSV must contain a 'fault' column.")

    df = df.copy()
    df["fault_raw"] = df["fault"].fillna("").astype(str)
    df["fault_normalized"] = df["fault_raw"].map(normalize_fault_label)
    df["fault_is_operational_state"] = df["fault_normalized"].isin(OPERATIONAL_STATES)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def fault_distribution(df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    counts = df["fault_normalized"].value_counts().head(limit)
    return [{"fault_normalized": key, "count": int(value)} for key, value in counts.items()]

