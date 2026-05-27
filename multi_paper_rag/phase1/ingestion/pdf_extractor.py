"""
PDF Extractor Module
====================
Extracts text from PDF files using PyMuPDF (fitz).
Provides font-aware extraction for heading detection, table/figure/equation
detection, and multi-column layout handling.
Validates output quality and fails fast on scanned/corrupt PDFs.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger("phase1.ingestion.pdf_extractor")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_structured_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text as semantic blocks with font and layout properties.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    blocks_data = []
    try:
        doc = fitz.open(str(pdf_file))
        for page_num in range(len(doc)):
            page = doc[page_num]
            dict_data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for b_idx, block in enumerate(dict_data.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                
                block_text = ""
                font_sizes = []
                is_bold = False
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if not span_text.strip():
                            continue
                        block_text += span_text
                        font_sizes.append(span.get("size", 0))
                        font_name = span.get("font", "").lower()
                        if "bold" in font_name or (span.get("flags", 0) & 2):
                            is_bold = True
                    block_text += "\n"
                    
                block_text = block_text.strip()
                if not block_text:
                    continue
                    
                dom_size = max(font_sizes) if font_sizes else 0
                    
                blocks_data.append({
                    "block_id": f"b_{page_num+1}_{b_idx+1}",
                    "page": page_num + 1,
                    "text": block_text,
                    "font_size": round(dom_size, 1),
                    "is_bold": is_bold,
                    "bbox": block.get("bbox")
                })
        doc.close()
    except Exception as exc:
        logger.error(f"Failed to extract blocks from '{pdf_path}': {exc}")
    
    return blocks_data



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
            font_info  (List[dict])     — Per-page font/span information.
            pdf_metadata (dict)         — PDF document-level metadata.

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

    # ── Extract PDF-level metadata ──────────────────────────────────────
    pdf_metadata = _extract_pdf_metadata(doc)

    # ── Extract per-page text with font info ────────────────────────────
    pages: List[Dict[str, Any]] = []
    font_info: List[Dict[str, Any]] = []

    for page_num in range(total_pages):
        try:
            page = doc[page_num]
            raw_text = page.get_text("text") or ""
            text = _clean_page_text(raw_text)

            # Extract font/span-level info for heading detection
            page_font_info = _extract_font_spans(page, page_num + 1)

            pages.append({"page_number": page_num + 1, "text": text})
            font_info.append({
                "page_number": page_num + 1,
                "spans": page_font_info,
            })
        except Exception as exc:
            logger.warning(
                "Page %d extraction failed in '%s': %s", page_num + 1, pdf_path, exc
            )
            pages.append({"page_number": page_num + 1, "text": ""})
            font_info.append({"page_number": page_num + 1, "spans": []})

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
            font_info=font_info,
            pdf_metadata=pdf_metadata,
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
            font_info=font_info,
            pdf_metadata=pdf_metadata,
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
        "font_info": font_info,
        "pdf_metadata": pdf_metadata,
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


def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using PyMuPDF's table detection.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of table dicts with keys: page, headers, rows, bbox.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return []

    tables = []
    try:
        doc = fitz.open(str(pdf_file))
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                page_tables = page.find_tables()
                for idx, table in enumerate(page_tables):
                    extracted = table.extract()
                    if extracted and len(extracted) > 0:
                        headers = extracted[0] if extracted else []
                        rows = extracted[1:] if len(extracted) > 1 else []
                        tables.append({
                            "table_id": f"table_{page_num+1}_{idx+1}",
                            "page": page_num + 1,
                            "headers": [str(h) if h else "" for h in headers],
                            "rows": [[str(c) if c else "" for c in row] for row in rows],
                            "bbox": list(table.bbox) if hasattr(table, 'bbox') else [],
                        })
            except Exception as exc:
                logger.debug("Table extraction failed on page %d: %s", page_num + 1, exc)
        doc.close()
    except Exception as exc:
        logger.warning("Table extraction failed for '%s': %s", pdf_path, exc)

    return tables


def extract_figures_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Detect figure captions and associate them with pages.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of figure dicts with keys: page, caption, figure_id.
    """
    figures = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text") or ""
            # Find figure captions
            for match in re.finditer(
                r"(Figure|Fig\.)\s+(\d+)[:\.]?\s*(.*?)(?:\n|$)",
                text, re.IGNORECASE
            ):
                fig_num = match.group(2)
                caption = match.group(3).strip()
                figures.append({
                    "page": page_num + 1,
                    "caption": caption,
                    "figure_id": f"fig_{fig_num}",
                })
        doc.close()
    except Exception as exc:
        logger.warning("Figure detection failed for '%s': %s", pdf_path, exc)

    return figures


def extract_equations_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Detect labeled equations from the PDF text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of equation dicts with keys: page, plain_text, label.
    """
    equations = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text") or ""
            # Find equation labels like (1), (2), etc.
            for match in re.finditer(
                r"(.{5,80})\s*\((\d+)\)\s*$",
                text, re.MULTILINE
            ):
                eq_text = match.group(1).strip()
                label = f"({match.group(2)})"
                # Filter out non-equation matches (references, etc.)
                if not re.match(r"^\[?\d", eq_text) and len(eq_text) > 3:
                    equations.append({
                        "equation_id": f"eq_{page_num+1}_{len(equations)+1}",
                        "page": page_num + 1,
                        "plain_text": eq_text,
                        "label": label,
                    })
        doc.close()
    except Exception as exc:
        logger.warning("Equation detection failed for '%s': %s", pdf_path, exc)

    return equations


def get_title_from_font_analysis(pdf_path: str) -> Optional[str]:
    """
    Extract the paper title by finding the largest font text on page 1.

    Uses font-size hierarchy: Title font > Section heading > Body text.
    Strips conference headers, page numbers, and running headers.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The extracted title string, or None if detection fails.
    """
    try:
        doc = fitz.open(str(pdf_path))
        if len(doc) == 0:
            doc.close()
            return None

        page = doc[0]
        spans = _extract_font_spans(page, 1)
        doc.close()

        if not spans:
            return None

        # Filter out tiny text and running headers
        # Group by font size and find the largest
        size_groups: Dict[float, List[str]] = {}
        for span in spans:
            size = round(span["size"], 1)
            text = span["text"].strip()
            if not text or len(text) < 3:
                continue
            # Skip known non-title patterns
            if _is_header_footer_text(text):
                continue
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(text)

        if not size_groups:
            return None

        # The title is typically the largest font
        max_size = max(size_groups.keys())
        title_parts = size_groups[max_size]

        # Combine and clean
        title = " ".join(title_parts).strip()
        # Remove line breaks and excessive whitespace
        title = re.sub(r"\s+", " ", title)
        # Limit length
        if len(title) > 300:
            title = title[:300]

        return title if len(title) > 5 else None

    except Exception as exc:
        logger.warning("Font-based title extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_pdf_metadata(doc: fitz.Document) -> Dict[str, Any]:
    """Extract document-level metadata from PDF."""
    meta = doc.metadata or {}
    return {
        "title": (meta.get("title") or "").strip(),
        "author": (meta.get("author") or "").strip(),
        "subject": (meta.get("subject") or "").strip(),
        "keywords": (meta.get("keywords") or "").strip(),
        "creator": (meta.get("creator") or "").strip(),
        "producer": (meta.get("producer") or "").strip(),
    }


def _extract_font_spans(page: fitz.Page, page_number: int) -> List[Dict[str, Any]]:
    """
    Extract text spans with font metadata from a page.

    Args:
        page: PyMuPDF page object.
        page_number: 1-indexed page number.

    Returns:
        List of span dicts with keys: text, size, font, flags, bbox, page.
    """
    spans = []
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    spans.append({
                        "text": text,
                        "size": span.get("size", 0),
                        "font": span.get("font", ""),
                        "flags": span.get("flags", 0),
                        "bbox": span.get("bbox", [0, 0, 0, 0]),
                        "page": page_number,
                    })
    except Exception as exc:
        logger.debug("Font span extraction failed on page %d: %s", page_number, exc)
    return spans


def _is_header_footer_text(text: str) -> bool:
    """Check if text is a running header/footer that should be excluded."""
    text_lower = text.strip().lower()

    # Page numbers
    if re.match(r"^\d{1,4}$", text.strip()):
        return True

    # Common conference headers
    header_patterns = [
        r"^www\s+companion",
        r"^proceedings\s+of",
        r"^accepted\s+at",
        r"^published\s+in",
        r"^preprint",
        r"^arxiv:\d{4}\.\d{4,5}",
        r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
    ]
    for pattern in header_patterns:
        if re.match(pattern, text_lower):
            return True

    # Very short text that looks like page markers
    if len(text.strip()) <= 3 and not text.strip().isalpha():
        return True

    return False


def _fail(
    total_pages: int,
    error: str,
    pages: list | None = None,
    full_text: str = "",
    font_info: list | None = None,
    pdf_metadata: dict | None = None,
) -> Dict[str, Any]:
    """Build a standardised failure result dict."""
    return {
        "success": False,
        "total_pages": total_pages,
        "pages": pages or [],
        "full_text": full_text,
        "error": error,
        "font_info": font_info or [],
        "pdf_metadata": pdf_metadata or {},
    }


def _clean_page_text(text: str) -> str:
    """Remove common headers, footers, and OCR noise from page text."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        sline = line.strip()
        if not sline:
            continue
        # Remove copyright/ACM/WWW footers
        if "Permission to make digital or hard copies" in sline:
            continue
        if sline.startswith("WWW Companion") or "Copyright held by" in sline:
            continue
        if re.match(r"^arxiv:\d{4}\.\d{4,5}(v\d+)?", sline, re.IGNORECASE):
            continue
        # Skip lines that are just numbers (like page numbers or lonely chart axis values)
        if re.match(r"^\d+$", sline):
            continue
        # Skip garbled math or mojibake (high density of non-ascii or single weird chars)
        if len(sline) > 0 and len([c for c in sline if ord(c) > 127]) / len(sline) > 0.5:
            continue
        # Skip isolated letters often found in graphs (e.g., 'a', 'b') if not part of a sentence
        if len(sline) == 1 and sline.lower() in "abcdefghijklmnopqrstuvwxyz":
            continue

        cleaned_lines.append(sline)
    return "\n".join(cleaned_lines)
