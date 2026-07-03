from __future__ import annotations

import io
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pdfplumber


@dataclass(frozen=True)
class PageText:
    document: str
    page: int
    text: str
    extraction_method: str
    image_count: int = 0
    embedded_text_chars: int = 0
    ocr_text_chars: int = 0
    ocr_error: str | None = None


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def _ocr_pdf_page(
    pdf_path: Path,
    page_index: int,
    tesseract_cmd: str | None = None,
    ocr_lang: str = "por+eng",
) -> str:
    """Render one PDF page and run OCR over the rendered image."""
    import fitz
    import pytesseract
    from PIL import Image

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_index]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))

    try:
        return pytesseract.image_to_string(image, lang=ocr_lang)
    except Exception:
        return pytesseract.image_to_string(image)


def _ocr_availability_error(tesseract_cmd: str | None = None) -> str | None:
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        pytesseract.get_tesseract_version()
        return None
    except Exception as exc:
        return str(exc)


def ocr_status(tesseract_cmd: str | None = None) -> dict[str, Any]:
    error = _ocr_availability_error(tesseract_cmd)
    return {
        "available": error is None,
        "error": error,
        "tesseract_cmd_configured": bool(tesseract_cmd),
    }


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _merge_pdf_and_ocr_text(pdf_text: str, ocr_text: str) -> str:
    """Append OCR-only lines while avoiding obvious duplication of embedded text."""
    pdf_text = pdf_text.strip()
    ocr_text = ocr_text.strip()

    if not pdf_text:
        return _normalize_whitespace(ocr_text)
    if not ocr_text:
        return _normalize_whitespace(pdf_text)

    pdf_lines = [_normalize_whitespace(line) for line in pdf_text.splitlines()]
    pdf_lines = [line for line in pdf_lines if line]
    pdf_blob = _normalize_whitespace(pdf_text).lower()

    extra_lines: list[str] = []
    for raw_line in ocr_text.splitlines():
        line = _normalize_whitespace(raw_line)
        if len(line) < 4:
            continue
        normalized_line = line.lower()
        if normalized_line in pdf_blob:
            continue
        if any(_similarity(normalized_line, existing.lower()) >= 0.86 for existing in pdf_lines):
            continue
        if any(_similarity(normalized_line, existing.lower()) >= 0.86 for existing in extra_lines):
            continue
        extra_lines.append(line)

    if not extra_lines:
        return _normalize_whitespace(pdf_text)

    merged = pdf_text.rstrip() + "\n\n[Texto extraido por OCR de imagens ou renderizacao da pagina]\n"
    merged += "\n".join(extra_lines)
    return _normalize_whitespace(merged)


def _should_run_ocr(
    strategy: str,
    embedded_text: str,
    image_count: int,
    min_text_chars: int = 40,
) -> bool:
    strategy = strategy.lower().strip()
    if strategy == "always":
        return True
    if strategy == "missing_text":
        return len(_normalize_whitespace(embedded_text)) < min_text_chars
    return image_count > 0 or len(_normalize_whitespace(embedded_text)) < min_text_chars


def extract_pdf_pages(
    pdf_path: str | Path,
    enable_ocr: bool = False,
    ocr_strategy: str = "auto",
    tesseract_cmd: str | None = None,
    ocr_lang: str = "por+eng",
) -> list[PageText]:
    """Extract text per page and optionally merge OCR text from page images."""
    path = Path(pdf_path)
    pages: list[PageText] = []
    ocr_availability_error = _ocr_availability_error(tesseract_cmd) if enable_ocr else None

    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages):
            embedded_text = page.extract_text() or ""
            text = _normalize_whitespace(embedded_text)
            method = "pdf_text"
            image_count = len(page.images or [])
            ocr_text = ""
            ocr_error = None

            should_run_ocr = enable_ocr and _should_run_ocr(ocr_strategy, embedded_text, image_count)
            if should_run_ocr and ocr_availability_error:
                ocr_error = ocr_availability_error
                method = "ocr_unavailable" if not text else "pdf_text+ocr_unavailable"
            elif should_run_ocr:
                try:
                    ocr_text = _ocr_pdf_page(
                        path,
                        index,
                        tesseract_cmd=tesseract_cmd,
                        ocr_lang=ocr_lang,
                    )
                    if ocr_text:
                        text = _merge_pdf_and_ocr_text(embedded_text, ocr_text)
                        method = "ocr" if not _normalize_whitespace(embedded_text) else "pdf_text+ocr"
                except Exception as exc:
                    ocr_error = str(exc)
                    method = "ocr_failed" if not text else "pdf_text+ocr_failed"

            pages.append(
                PageText(
                    document=path.name,
                    page=index + 1,
                    text=text,
                    extraction_method=method,
                    image_count=image_count,
                    embedded_text_chars=len(_normalize_whitespace(embedded_text)),
                    ocr_text_chars=len(_normalize_whitespace(ocr_text)),
                    ocr_error=ocr_error,
                )
            )

    return pages


def split_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    """Split text into character-limited chunks with small word overlap."""
    text = _normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_size + extra > chunk_size:
            chunks.append(" ".join(current))

            overlap_words: list[str] = []
            overlap_size = 0
            for previous in reversed(current):
                previous_extra = len(previous) + (1 if overlap_words else 0)
                if overlap_size + previous_extra > overlap:
                    break
                overlap_words.insert(0, previous)
                overlap_size += previous_extra

            current = overlap_words
            current_size = len(" ".join(current))

        current.append(word)
        current_size += extra

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_document_chunks(
    docs_path: str | Path,
    enable_ocr: bool = False,
    ocr_strategy: str = "auto",
    tesseract_cmd: str | None = None,
    ocr_lang: str = "por+eng",
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[dict[str, Any]]:
    """Load PDFs from a folder and return RAG-ready chunks with metadata."""
    base = Path(docs_path)
    chunks: list[dict[str, Any]] = []

    for pdf_path in sorted(base.glob("*.pdf")):
        pages = extract_pdf_pages(
            pdf_path,
            enable_ocr=enable_ocr,
            ocr_strategy=ocr_strategy,
            tesseract_cmd=tesseract_cmd,
            ocr_lang=ocr_lang,
        )
        for page in pages:
            for index, chunk_text in enumerate(split_text(page.text, chunk_size, overlap), start=1):
                chunk_id = f"{page.document}:p{page.page}:c{index}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document": page.document,
                        "page": page.page,
                        "chunk_index": index,
                        "text": chunk_text,
                        "extraction_method": page.extraction_method,
                        "image_count": page.image_count,
                    }
                )

    return chunks


def document_extraction_report(
    docs_path: str | Path,
    enable_ocr: bool = False,
    ocr_strategy: str = "auto",
    tesseract_cmd: str | None = None,
    ocr_lang: str = "por+eng",
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for pdf_path in sorted(Path(docs_path).glob("*.pdf")):
        pages = extract_pdf_pages(
            pdf_path,
            enable_ocr=enable_ocr,
            ocr_strategy=ocr_strategy,
            tesseract_cmd=tesseract_cmd,
            ocr_lang=ocr_lang,
        )
        report.append(
            {
                "document": pdf_path.name,
                "pages": len(pages),
                "text_pages": sum(1 for page in pages if page.text),
                "image_pages": sum(1 for page in pages if page.image_count > 0),
                "ocr_pages": sum(
                    1 for page in pages if page.extraction_method in {"ocr", "pdf_text+ocr"}
                ),
                "ocr_failed_pages": sum(1 for page in pages if "ocr_failed" in page.extraction_method),
                "ocr_unavailable_pages": sum(
                    1 for page in pages if "ocr_unavailable" in page.extraction_method
                ),
                "characters": sum(len(page.text) for page in pages),
                "embedded_text_characters": sum(page.embedded_text_chars for page in pages),
                "ocr_characters": sum(page.ocr_text_chars for page in pages),
                "methods": sorted({page.extraction_method for page in pages}),
                "ocr_errors": sorted({page.ocr_error for page in pages if page.ocr_error}),
            }
        )
    return report
