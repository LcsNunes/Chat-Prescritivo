from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.rag import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_BASE_URL, embed_texts


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


@dataclass(frozen=True)
class CanonicalFault:
    key: str
    display_name: str
    description: str
    related_documents: tuple[str, ...]
    has_documentation: bool = True


@dataclass(frozen=True)
class FaultMappingResult:
    fault_raw: str
    fault_normalized: str
    canonical_key: str
    display_name: str
    score: float
    confidence: str
    has_documentation: bool
    related_documents: tuple[str, ...]
    is_operational_state: bool


CANONICAL_FAULTS: tuple[CanonicalFault, ...] = (
    CanonicalFault(
        key="bearing_fault",
        display_name="Falha em rolamento",
        description=(
            "Falhas em rolamentos de maquinas rotativas: rolamento externo, rolamento interno, "
            "esferas, gaiola, bearing fault, outer race, inner race, ball fault, combination fault."
        ),
        related_documents=("Doc1.pdf",),
    ),
    CanonicalFault(
        key="misalignment",
        display_name="Desalinhamento",
        description=(
            "Desalinhamento de motor eletrico, eixo desalinhado, desalinhado, misalignment, "
            "desalinhamento angular, paralelo ou combinado."
        ),
        related_documents=("Doc2.pdf",),
    ),
    CanonicalFault(
        key="unbalance",
        display_name="Desbalanceamento",
        description=(
            "Desbalanceamento, desbalanceado, unbalance, massa desbalanceada, rotor com vibracao radial, "
            "desbalanceado por parafuso ou massa irregular."
        ),
        related_documents=("Doc3.pdf",),
    ),
    CanonicalFault(
        key="belt_fault",
        display_name="Falha em correia",
        description=(
            "Falha em correia de transmissao, correia frouxa, correia tensionada, desgaste, "
            "escorregamento, belt fault."
        ),
        related_documents=("Doc4.pdf",),
    ),
    CanonicalFault(
        key="pulley_fault",
        display_name="Falha em polia",
        description=(
            "Falha em polia, pulley fault, polia excentrica, polia desalinhada, desgaste de polias, "
            "transmissao por correia."
        ),
        related_documents=("Doc5.pdf",),
    ),
    CanonicalFault(
        key="cocked_rotor",
        display_name="Rotor inclinado / Cocked Rotor",
        description=(
            "Rotor inclinado, cocked rotor, rotor fora de esquadro, montagem inclinada, "
            "desvio angular do plano do rotor."
        ),
        related_documents=("Doc6.pdf",),
    ),
    CanonicalFault(
        key="fan_fault",
        display_name="Falha em ventoinha",
        description=(
            "Falha em ventoinha, ventilador, refrigeracao insuficiente, ventoinha parada, fan fault."
        ),
        related_documents=("Doc7.pdf",),
    ),
    CanonicalFault(
        key="phase_loss",
        display_name="Falta de fase",
        description="Falta de fase eletrica, phase loss, problema na alimentacao trifasica.",
        related_documents=("Doc8.pdf",),
    ),
    CanonicalFault(
        key="undocumented_eccentric_rotor",
        display_name="Rotor excentrico sem documento cadastrado",
        description=(
            "Rotor excentrico, eccentric rotor, excentricidade do rotor. Classe conhecida nos dados, "
            "mas sem procedimento tecnico especifico cadastrado."
        ),
        related_documents=(),
        has_documentation=False,
    ),
    CanonicalFault(
        key="operational_state",
        display_name="Estado operacional sem falha",
        description=(
            "Estado operacional sem defeito: normal, baseline, teste, acelerando ou motor desligado."
        ),
        related_documents=(),
        has_documentation=False,
    ),
)


def get_canonical_fault(key: str) -> CanonicalFault:
    for fault in CANONICAL_FAULTS:
        if fault.key == key:
            return fault
    raise KeyError(f"Unknown canonical fault: {key}")


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


def get_event_by_id(df: pd.DataFrame, event_id: int | str) -> dict[str, Any]:
    matches = df[df["id"].astype(str) == str(event_id)]
    if matches.empty:
        raise ValueError(f"Event id not found: {event_id}")
    return matches.iloc[0].to_dict()


def available_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in NUMERIC_COLUMNS if column in df.columns]


def find_similar_events(
    df: pd.DataFrame,
    event: dict[str, Any],
    limit: int = 5,
) -> dict[str, Any]:
    """Find historical events with similar numeric sensor behavior."""
    columns = available_numeric_columns(df)
    if not columns:
        return {"count": 0, "period": None, "common_faults": [], "examples": []}

    matrix = df[columns].apply(pd.to_numeric, errors="coerce")
    means = matrix.mean()
    stds = matrix.std(ddof=0).replace(0, 1)
    normalized_matrix = ((matrix.fillna(means) - means) / stds).to_numpy(dtype=np.float32)

    event_values = pd.Series({column: event.get(column) for column in columns})
    event_values = pd.to_numeric(event_values, errors="coerce").fillna(means)
    normalized_event = ((event_values - means) / stds).to_numpy(dtype=np.float32)

    distances = np.linalg.norm(normalized_matrix - normalized_event, axis=1)

    if "id" in event and "id" in df.columns:
        same_id = df["id"].astype(str) == str(event["id"])
        distances[same_id.to_numpy()] = np.inf

    nearest_indices = np.argsort(distances)[:limit]
    neighbors = df.iloc[nearest_indices].copy()
    neighbors["similarity_distance"] = distances[nearest_indices]

    created_at = pd.to_datetime(neighbors.get("created_at"), errors="coerce")
    period = None
    if created_at.notna().any():
        period = {
            "start": created_at.min().isoformat(),
            "end": created_at.max().isoformat(),
        }

    common_faults = (
        neighbors["fault_normalized"]
        .value_counts()
        .head(5)
        .rename_axis("fault_normalized")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    example_columns = [
        "id",
        "created_at",
        "fault_raw",
        "fault_normalized",
        "fault_is_operational_state",
        "similarity_distance",
    ]
    examples = []
    for row in neighbors[example_columns].to_dict(orient="records"):
        row["similarity_distance"] = float(row["similarity_distance"])
        examples.append(row)

    return {
        "count": int(len(examples)),
        "period": period,
        "common_faults": common_faults,
        "numeric_columns": columns,
        "examples": examples,
    }


def _confidence(score: float, min_score: float) -> str:
    if score < min_score:
        return "low"
    if score >= 0.70:
        return "high"
    return "medium"


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def map_fault_to_canonical(
    fault_raw: Any,
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    min_score: float = 0.55,
) -> FaultMappingResult:
    """Map a noisy fault label to a canonical class using semantic similarity."""
    summary = summarize_fault(fault_raw)

    if summary.is_operational_state:
        canonical = get_canonical_fault("operational_state")
        return FaultMappingResult(
            fault_raw=summary.fault_raw,
            fault_normalized=summary.fault_normalized,
            canonical_key=canonical.key,
            display_name=canonical.display_name,
            score=1.0,
            confidence="high",
            has_documentation=canonical.has_documentation,
            related_documents=canonical.related_documents,
            is_operational_state=True,
        )

    class_texts = [
        f"{fault.key}. {fault.display_name}. {fault.description}" for fault in CANONICAL_FAULTS
    ]
    query = (
        f"Falha informada pelo operador: {summary.fault_raw}. "
        f"Falha normalizada: {summary.fault_normalized}."
    )

    embeddings = embed_texts([query] + class_texts, model=model, base_url=base_url)
    query_embedding = _normalize_vector(embeddings[0])
    class_embeddings = np.array([_normalize_vector(vector) for vector in embeddings[1:]])
    scores = class_embeddings @ query_embedding
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    canonical = CANONICAL_FAULTS[best_index]

    if best_score < min_score:
        canonical = CanonicalFault(
            key="undocumented_unknown",
            display_name="Falha sem documento cadastrado",
            description="Falha nao mapeada com confianca suficiente.",
            related_documents=(),
            has_documentation=False,
        )

    return FaultMappingResult(
        fault_raw=summary.fault_raw,
        fault_normalized=summary.fault_normalized,
        canonical_key=canonical.key,
        display_name=canonical.display_name,
        score=best_score,
        confidence=_confidence(best_score, min_score),
        has_documentation=canonical.has_documentation,
        related_documents=canonical.related_documents,
        is_operational_state=False,
    )
