from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"


@dataclass
class VectorIndex:
    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    model: str
    fingerprint: str


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def chunks_fingerprint(chunks: list[dict[str, Any]]) -> str:
    payload = [
        {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def index_cache_file(cache_path: str | Path, model: str, fingerprint: str) -> Path:
    cache_dir = Path(cache_path)
    model_name = _safe_filename(model)
    return cache_dir / f"vector_index_{model_name}_{fingerprint[:12]}.json"


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    batch_size: int = 16,
    timeout: int = 180,
) -> np.ndarray:
    """Generate real embeddings through Ollama /api/embed."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    all_embeddings: list[list[float]] = []
    url = f"{base_url.rstrip('/')}/api/embed"

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = requests.post(
            url,
            json={"model": model, "input": batch},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise RuntimeError("Ollama did not return the expected embeddings.")
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def save_vector_index(index: VectorIndex, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": index.model,
        "fingerprint": index.fingerprint,
        "chunks": index.chunks,
        "embeddings": index.embeddings.tolist(),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_vector_index(path: str | Path) -> VectorIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return VectorIndex(
        chunks=payload["chunks"],
        embeddings=np.array(payload["embeddings"], dtype=np.float32),
        model=payload["model"],
        fingerprint=payload["fingerprint"],
    )


def build_vector_index(
    chunks: list[dict[str, Any]],
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    cache_path: str | Path = "cache",
    force_rebuild: bool = False,
) -> VectorIndex:
    """Build or load a local vector index using real Ollama embeddings."""
    fingerprint = chunks_fingerprint(chunks)
    cache_file = index_cache_file(cache_path, model, fingerprint)

    if cache_file.exists() and not force_rebuild:
        cached = load_vector_index(cache_file)
        if cached.model == model and cached.fingerprint == fingerprint:
            return cached

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts, model=model, base_url=base_url)
    index = VectorIndex(chunks=chunks, embeddings=embeddings, model=model, fingerprint=fingerprint)
    save_vector_index(index, cache_file)
    return index


def ollama_health(base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout: int = 5) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return {"ok": True, "models": [model.get("name") for model in data.get("models", [])]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

