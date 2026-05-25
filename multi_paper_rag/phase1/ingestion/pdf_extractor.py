"""
PDF Extractor Module
====================
Extracts text from PDF files using PyMuPDF (fitz).
Validates output quality and fails fast on scanned/corrupt PDFs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

logger = logging.getLogger("phase1.ingestion.pdf_extractor")


def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from every page of a PDF file.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A dict with keys:
            success    (bool)           — Whether extraction succeeded.
            total_pages (int)           — Number of pages in the PDF.
            pages      (List[dict])     — Per-page dicts with page_number & text.
            full_text  (str)            — Concatenated text of all pages.
            error      (str | None)     — Error message on failure, else None.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_file))
    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", pdf_path, exc)
        return _fail(0, f"Failed to open PDF: {exc}")

    total_pages: int = len(doc)

    # ── Validation: empty PDF ───────────────────────────────────────────
    if total_pages == 0:
        doc.close()
        logger.error("Empty PDF: %s", pdf_path)
        return _fail(0, "Empty PDF")

    # ── Extract per-page text ───────────────────────────────────────────
    pages: List[Dict[str, Any]] = []
    for page_num in range(total_pages):
        try:
            page = doc[page_num]
            text = page.get_text("text") or ""
            pages.append({"page_number": page_num + 1, "text": text})
        except Exception as exc:
            logger.warning(
                "Page %d extraction failed in '%s': %s", page_num + 1, pdf_path, exc
            )
            pages.append({"page_number": page_num + 1, "text": ""})

    doc.close()

    full_text = "\n".join(p["text"] for p in pages)

    # ── Validation: text too short (likely scanned image PDF) ──────────
    if len(full_text) < 500:
        logger.error(
            "Extracted text too short (%d chars) for '%s'", len(full_text), pdf_path
        )
        return _fail(
            total_pages,
            "Extracted text too short, likely scanned image PDF",
            pages=pages,
            full_text=full_text,
        )

    # ── Validation: high non-ASCII ratio (scanned / corrupted) ─────────
    non_ascii_count = sum(1 for ch in full_text if ord(ch) > 127)
    non_ascii_ratio = non_ascii_count / len(full_text) if full_text else 0
    if non_ascii_ratio > 0.40:
        logger.error(
            "Non-ASCII ratio %.1f%% for '%s' — likely scanned or corrupted",
            non_ascii_ratio * 100,
            pdf_path,
        )
        return _fail(
            total_pages,
            "Likely scanned or corrupted PDF",
            pages=pages,
            full_text=full_text,
        )

    logger.info(
        "Successfully extracted %d pages (%d chars) from '%s'",
        total_pages,
        len(full_text),
        pdf_path,
    )
    return {
        "success": True,
        "total_pages": total_pages,
        "pages": pages,
        "full_text": full_text,
        "error": None,
    }


def get_toc(pdf_path: str) -> list:
    """
    Return the embedded Table of Contents from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of TOC entries as returned by PyMuPDF's ``doc.get_toc()``.
        Each entry is ``[level, title, page_number]``.
        Returns an empty list if no TOC is embedded.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(pdf_file))
    toc = doc.get_toc()
    doc.close()
    logger.debug("TOC entries for '%s': %d", pdf_path, len(toc))
    return toc


# ── Private helpers ─────────────────────────────────────────────────────

def _fail(
    total_pages: int,
    error: str,
    pages: list | None = None,
    full_text: str = "",
) -> Dict[str, Any]:
    """Build a standardised failure result dict."""
    return {
        "success": False,
        "total_pages": total_pages,
        "pages": pages or [],
        "full_text": full_text,
        "error": error,
    }
