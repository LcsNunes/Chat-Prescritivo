from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from src.chunking import build_document_chunks, document_extraction_report, ocr_status
from src.fault_mapping import (
    find_similar_events,
    get_event_by_id,
    load_events,
    map_fault_to_canonical,
    normalize_fault_label,
    summarize_fault,
)
from src.guardrails import build_undocumented_response, evaluate_guardrails, validate_llm_answer
from src.prompts import build_chat_messages, build_rag_messages
from src.rag import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    build_vector_index,
    chat_with_ollama,
    ollama_health,
    retrieve_chunks,
)


app = FastAPI(title="Chat Prescritivo", version="1.1.0")
app.mount("/assets", StaticFiles(directory="frontend"), name="assets")

EVENTS_DF: pd.DataFrame | None = None
DOCUMENT_CHUNKS: list[dict[str, Any]] | None = None
VECTOR_INDEX = None


class AnalyzeRequest(BaseModel):
    event_id: int | str | None = None
    event: dict[str, Any] | None = None
    top_k_chunks: int | None = None
    similar_events_limit: int | None = None


class ChatRequest(BaseModel):
    question: str
    top_k_chunks: int | None = None


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
        "ocr_strategy": os.getenv("OCR_STRATEGY", "auto"),
        "ocr_lang": os.getenv("OCR_LANG", "por+eng"),
        "tesseract_cmd": os.getenv("TESSERACT_CMD") or None,
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
        DOCUMENT_CHUNKS = build_document_chunks(
            cfg["docs_path"],
            enable_ocr=cfg["enable_ocr"],
            ocr_strategy=cfg["ocr_strategy"],
            tesseract_cmd=cfg["tesseract_cmd"],
            ocr_lang=cfg["ocr_lang"],
        )
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


def reset_document_cache() -> None:
    global DOCUMENT_CHUNKS, VECTOR_INDEX
    DOCUMENT_CHUNKS = None
    VECTOR_INDEX = None


def _safe_pdf_name(filename: str) -> str:
    name = Path(filename).name
    if not name or Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    return name


def _document_path(filename: str) -> Path:
    cfg = _config()
    docs_dir = Path(cfg["docs_path"]).resolve()
    target = (docs_dir / _safe_pdf_name(filename)).resolve()
    if docs_dir not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid document name.")
    return target


def _chunk_response(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "score": chunk.get("score"),
            "text_preview": chunk["text"][:350],
        }
        for chunk in chunks
    ]


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


def _document_inventory() -> list[dict[str, Any]]:
    cfg = _config()
    docs_dir = Path(cfg["docs_path"])
    docs_dir.mkdir(parents=True, exist_ok=True)
    report = {
        item["document"]: item
        for item in document_extraction_report(
            docs_dir,
            enable_ocr=cfg["enable_ocr"],
            ocr_strategy=cfg["ocr_strategy"],
            tesseract_cmd=cfg["tesseract_cmd"],
            ocr_lang=cfg["ocr_lang"],
        )
    }

    documents: list[dict[str, Any]] = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        stat = pdf_path.stat()
        item = {
            "document": pdf_path.name,
            "size_bytes": stat.st_size,
            "indexed": False,
            "pages": None,
            "text_pages": None,
            "image_pages": None,
            "ocr_pages": None,
            "ocr_failed_pages": None,
            "characters": None,
            "ocr_characters": None,
            "methods": [],
        }
        item.update(report.get(pdf_path.name, {}))
        item["indexed"] = bool(item.get("characters"))
        documents.append(item)
    return documents


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    return FileResponse("frontend/index.html")


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
            "ocr": ocr_status(cfg["tesseract_cmd"]),
            "config": {
                "llm_model": cfg["llm_model"],
                "embedding_model": cfg["embedding_model"],
                "enable_ocr": cfg["enable_ocr"],
                "ocr_strategy": cfg["ocr_strategy"],
                "ocr_lang": cfg["ocr_lang"],
                "tesseract_cmd_configured": bool(cfg["tesseract_cmd"]),
                "min_fault_similarity": cfg["min_fault_similarity"],
                "min_chunk_similarity": cfg["min_chunk_similarity"],
            },
        }
    )


@app.get("/document-report")
def document_report() -> list[dict[str, Any]]:
    cfg = _config()
    return _jsonable(
        document_extraction_report(
            cfg["docs_path"],
            enable_ocr=cfg["enable_ocr"],
            ocr_strategy=cfg["ocr_strategy"],
            tesseract_cmd=cfg["tesseract_cmd"],
            ocr_lang=cfg["ocr_lang"],
        )
    )


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    return _jsonable({"documents": _document_inventory()})


@app.post("/documents")
def add_document(
    file: UploadFile = File(...),
    overwrite: bool = Query(False),
) -> dict[str, Any]:
    filename = _safe_pdf_name(file.filename or "")
    target = _document_path(filename)

    if target.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail="Document already exists. Use overwrite=true to replace it.",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reset_document_cache()
    return _jsonable(
        {
            "message": "Document saved and RAG index invalidated.",
            "document": filename,
            "size_bytes": target.stat().st_size,
        }
    )


@app.delete("/documents/{filename}")
def delete_document(filename: str) -> dict[str, Any]:
    target = _document_path(filename)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    target.unlink()
    reset_document_cache()
    return {"message": "Document deleted and RAG index invalidated.", "document": target.name}


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


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    cfg = _config()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    fault_mapping = map_fault_to_canonical(
        question,
        model=cfg["embedding_model"],
        base_url=cfg["ollama_base_url"],
        min_score=cfg["min_fault_similarity"],
    )

    retrieved_chunks: list[dict[str, Any]] = []
    if fault_mapping.has_documentation and not fault_mapping.is_operational_state:
        retrieved_chunks = retrieve_chunks(
            question,
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
        messages = build_chat_messages(question, fault_mapping, retrieved_chunks)
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
            "question": question,
            "fault_mapping": fault_mapping.__dict__,
            "retrieved_chunks": _chunk_response(retrieved_chunks),
            "guardrails": guardrail_decision,
            "answer_validation": validation,
            "answer": answer,
        }
    )


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
            "retrieved_chunks": _chunk_response(retrieved_chunks),
            "guardrails": guardrail_decision,
            "answer_validation": validation,
            "answer": answer,
        }
    )
