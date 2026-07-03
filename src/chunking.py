from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber


@dataclass(frozen=True)
class PageText:
    document: str
    page: int
    text: str
    extraction_method: str


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def _ocr_pdf_page(pdf_path: Path, page_index: int) -> str:
    """Run OCR for one page when the PDF has no embedded text."""
    import fitz
    import pytesseract
    from PIL import Image

    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_index]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))

    try:
        return pytesseract.image_to_string(image, lang="por+eng")
    except Exception:
        return pytesseract.image_to_string(image)


def extract_pdf_pages(pdf_path: str | Path, enable_ocr: bool = False) -> list[PageText]:
    """Extract text per page, optionally falling back to OCR for scanned pages."""
    path = Path(pdf_path)
    pages: list[PageText] = []

    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages):
            text = _normalize_whitespace(page.extract_text() or "")
            method = "pdf_text"

            if enable_ocr and len(text) < 40:
                try:
                    ocr_text = _normalize_whitespace(_ocr_pdf_page(path, index))
                    if ocr_text:
                        text = ocr_text
                        method = "ocr"
                except Exception:
                    method = "ocr_failed"

            pages.append(
                PageText(
                    document=path.name,
                    page=index + 1,
                    text=text,
                    extraction_method=method,
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
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[dict[str, Any]]:
    """Load PDFs from a folder and return RAG-ready chunks with metadata."""
    base = Path(docs_path)
    chunks: list[dict[str, Any]] = []

    for pdf_path in sorted(base.glob("*.pdf")):
        pages = extract_pdf_pages(pdf_path, enable_ocr=enable_ocr)
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
                    }
                )

    return chunks


def document_extraction_report(docs_path: str | Path, enable_ocr: bool = False) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for pdf_path in sorted(Path(docs_path).glob("*.pdf")):
        pages = extract_pdf_pages(pdf_path, enable_ocr=enable_ocr)
        report.append(
            {
                "document": pdf_path.name,
                "pages": len(pages),
                "text_pages": sum(1 for page in pages if page.text),
                "characters": sum(len(page.text) for page in pages),
                "methods": sorted({page.extraction_method for page in pages}),
            }
        )
    return report

