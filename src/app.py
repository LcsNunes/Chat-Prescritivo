from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from src.chunking import build_document_chunks, document_extraction_report
from src.fault_mapping import (
    find_similar_events,
    get_event_by_id,
    load_events,
    map_fault_to_canonical,
    normalize_fault_label,
    summarize_fault,
)
from src.guardrails import build_undocumented_response, evaluate_guardrails, validate_llm_answer
from src.prompts import build_rag_messages
from src.rag import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    build_vector_index,
    chat_with_ollama,
    ollama_health,
    retrieve_chunks,
)


app = FastAPI(title="Chat Prescritivo", version="1.0.0")

EVENTS_DF: pd.DataFrame | None = None
DOCUMENT_CHUNKS: list[dict[str, Any]] | None = None
VECTOR_INDEX = None


class AnalyzeRequest(BaseModel):
    event_id: int | str | None = None
    event: dict[str, Any] | None = None
    top_k_chunks: int | None = None
    similar_events_limit: int | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "sim", "y"}


def _config() -> dict[str, Any]:
    return {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        "llm_model": os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        "embedding_model": os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "data_path": os.getenv("DATA_PATH", "data/banner.csv"),
        "docs_path": os.getenv("DOCS_PATH", "data/docs"),
        "cache_path": os.getenv("CACHE_PATH", "cache"),
        "min_fault_similarity": float(os.getenv("MIN_FAULT_SIMILARITY", "0.55")),
        "min_chunk_similarity": float(os.getenv("MIN_CHUNK_SIMILARITY", "0.50")),
        "top_k_chunks": int(os.getenv("TOP_K_CHUNKS", "5")),
        "similar_events_limit": int(os.getenv("SIMILAR_EVENTS_LIMIT", "5")),
        "enable_ocr": _env_bool("ENABLE_OCR", False),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def get_events_df() -> pd.DataFrame:
    global EVENTS_DF
    if EVENTS_DF is None:
        EVENTS_DF = load_events(_config()["data_path"])
    return EVENTS_DF


def get_document_chunks() -> list[dict[str, Any]]:
    global DOCUMENT_CHUNKS
    if DOCUMENT_CHUNKS is None:
        cfg = _config()
        DOCUMENT_CHUNKS = build_document_chunks(cfg["docs_path"], enable_ocr=cfg["enable_ocr"])
    return DOCUMENT_CHUNKS


def get_vector_index():
    global VECTOR_INDEX
    if VECTOR_INDEX is None:
        cfg = _config()
        VECTOR_INDEX = build_vector_index(
            get_document_chunks(),
            model=cfg["embedding_model"],
            base_url=cfg["ollama_base_url"],
            cache_path=cfg["cache_path"],
        )
    return VECTOR_INDEX


def _event_from_request(payload: AnalyzeRequest) -> dict[str, Any]:
    df = get_events_df()
    if payload.event_id is not None:
        return get_event_by_id(df, payload.event_id)
    if payload.event is not None:
        event = dict(payload.event)
        raw_fault = event.get("fault_raw", event.get("fault", ""))
        summary = summarize_fault(raw_fault)
        event["fault_raw"] = summary.fault_raw
        event["fault_normalized"] = summary.fault_normalized
        event["fault_is_operational_state"] = summary.is_operational_state
        return event
    raise HTTPException(status_code=400, detail="Send event_id or event.")


def _raw_fault_from_event(event: dict[str, Any]) -> str:
    return str(event.get("fault_raw") or event.get("fault") or "")


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return """
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Chat Prescritivo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f2ee;
      --panel: #ffffff;
      --ink: #17201d;
      --muted: #5b6762;
      --line: #c9d0ca;
      --accent: #0f6b57;
      --accent-strong: #0a493b;
      --warn: #8d3f1d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      gap: 0;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #ebe7de;
      padding: 28px;
    }
    section {
      padding: 28px;
    }
    h1 {
      font-size: 28px;
      line-height: 1.1;
      margin: 0 0 22px;
      font-weight: 750;
    }
    label {
      display: block;
      margin: 18px 0 8px;
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      padding: 11px 12px;
    }
    textarea {
      min-height: 180px;
      resize: vertical;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 13px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 12px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary { background: var(--accent-strong); }
    .status {
      min-height: 22px;
      margin-top: 14px;
      color: var(--warn);
      font-size: 14px;
    }
    pre {
      margin: 0;
      min-height: calc(100vh - 56px);
      white-space: pre-wrap;
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 20px;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      pre { min-height: 55vh; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>Chat Prescritivo</h1>
      <label for="eventId">Event ID</label>
      <input id="eventId" value="114387" />
      <div class="row">
        <button onclick="analyzeById()">Analisar ID</button>
        <button class="secondary" onclick="loadHealth()">Health</button>
      </div>
      <label for="eventJson">Evento JSON</label>
      <textarea id="eventJson" spellcheck="false"></textarea>
      <button style="width:100%; margin-top:10px" onclick="analyzeJson()">Analisar JSON</button>
      <div class="status" id="status"></div>
    </aside>
    <section>
      <pre id="output">Aguardando analise.</pre>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");

    async function postAnalyze(payload) {
      statusEl.textContent = "Processando...";
      outputEl.textContent = "";
      try {
        const res = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        outputEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = res.ok ? "Concluido." : "Erro na analise.";
      } catch (err) {
        statusEl.textContent = "Falha de comunicacao.";
        outputEl.textContent = String(err);
      }
    }

    function analyzeById() {
      postAnalyze({ event_id: document.getElementById("eventId").value });
    }

    function analyzeJson() {
      const raw = document.getElementById("eventJson").value.trim();
      if (!raw) {
        statusEl.textContent = "Informe um JSON.";
        return;
      }
      postAnalyze({ event: JSON.parse(raw) });
    }

    async function loadHealth() {
      statusEl.textContent = "Consultando health...";
      const res = await fetch("/health");
      outputEl.textContent = JSON.stringify(await res.json(), null, 2);
      statusEl.textContent = "Concluido.";
    }
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = _config()
    return _jsonable(
        {
            "status": "ok",
            "data_path_exists": Path(cfg["data_path"]).exists(),
            "docs_path_exists": Path(cfg["docs_path"]).exists(),
            "events_loaded": EVENTS_DF is not None,
            "chunks_loaded": DOCUMENT_CHUNKS is not None,
            "index_loaded": VECTOR_INDEX is not None,
            "ollama": ollama_health(cfg["ollama_base_url"]),
            "config": {
                "llm_model": cfg["llm_model"],
                "embedding_model": cfg["embedding_model"],
                "enable_ocr": cfg["enable_ocr"],
                "min_fault_similarity": cfg["min_fault_similarity"],
                "min_chunk_similarity": cfg["min_chunk_similarity"],
            },
        }
    )


@app.get("/document-report")
def document_report() -> list[dict[str, Any]]:
    cfg = _config()
    return _jsonable(document_extraction_report(cfg["docs_path"], enable_ocr=cfg["enable_ocr"]))


@app.get("/events/{event_id}")
def event_by_id(event_id: int | str) -> dict[str, Any]:
    return _jsonable(get_event_by_id(get_events_df(), event_id))


@app.get("/sample-events")
def sample_events(fault: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    df = get_events_df()
    sample = df
    if fault:
        normalized = normalize_fault_label(fault)
        sample = sample[sample["fault_normalized"] == normalized]
    rows = sample.head(max(1, min(limit, 100))).to_dict(orient="records")
    return _jsonable(rows)


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    cfg = _config()
    event = _event_from_request(payload)
    raw_fault = _raw_fault_from_event(event)

    fault_mapping = map_fault_to_canonical(
        raw_fault,
        model=cfg["embedding_model"],
        base_url=cfg["ollama_base_url"],
        min_score=cfg["min_fault_similarity"],
    )
    similar_events = find_similar_events(
        get_events_df(),
        event,
        limit=payload.similar_events_limit or cfg["similar_events_limit"],
    )

    retrieved_chunks: list[dict[str, Any]] = []
    if fault_mapping.has_documentation and not fault_mapping.is_operational_state:
        query = f"{fault_mapping.fault_normalized} {fault_mapping.display_name} {raw_fault}"
        retrieved_chunks = retrieve_chunks(
            query,
            get_vector_index(),
            model=cfg["embedding_model"],
            base_url=cfg["ollama_base_url"],
            top_k=payload.top_k_chunks or cfg["top_k_chunks"],
            min_score=0.0,
            document_filter=set(fault_mapping.related_documents),
        )

    guardrail_decision = evaluate_guardrails(
        fault_mapping,
        retrieved_chunks,
        min_chunk_score=cfg["min_chunk_similarity"],
    )

    if guardrail_decision["allowed"]:
        messages = build_rag_messages(event, fault_mapping, similar_events, retrieved_chunks)
        answer = chat_with_ollama(
            messages,
            model=cfg["llm_model"],
            base_url=cfg["ollama_base_url"],
        )
        validation = validate_llm_answer(answer, retrieved_chunks)
    else:
        answer = build_undocumented_response(fault_mapping, guardrail_decision)
        validation = {"ok": True, "documents": [], "cited_documents": []}

    return _jsonable(
        {
            "event": event,
            "fault_mapping": fault_mapping.__dict__,
            "similar_events": similar_events,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk["document"],
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "score": chunk.get("score"),
                    "text_preview": chunk["text"][:350],
                }
                for chunk in retrieved_chunks
            ],
            "guardrails": guardrail_decision,
            "answer_validation": validation,
            "answer": answer,
        }
    )

