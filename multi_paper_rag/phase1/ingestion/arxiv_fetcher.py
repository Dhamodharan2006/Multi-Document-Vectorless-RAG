"""
arXiv Fetcher Module
====================
Downloads PDFs from arXiv and retrieves paper metadata via the arXiv API.
Handles full URLs, PDF URLs, and bare arXiv IDs.
"""

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

import requests

from multi_paper_rag.phase1.config import PDF_STORAGE_DIR

logger = logging.getLogger("phase1.ingestion.arxiv_fetcher")

# Namespace used in arXiv Atom XML responses
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

# Timeout for HTTP requests (seconds)
_REQUEST_TIMEOUT = 60


def _normalise_arxiv_id(arxiv_input: str) -> str:
    """
    Normalise any accepted arXiv input format to a bare arXiv ID.

    Accepted formats:
        - Full URL:  https://arxiv.org/abs/1706.03762  or  https://arxiv.org/abs/1706.03762v1
        - PDF URL:   https://arxiv.org/pdf/1706.03762  or  https://arxiv.org/pdf/1706.03762.pdf
        - Bare ID:   1706.03762  or  2301.12345v2

    Args:
        arxiv_input: Raw user input.

    Returns:
        Clean arXiv ID string (e.g. "1706.03762").

    Raises:
        ValueError: If the input cannot be parsed as an arXiv reference.
    """
    text = arxiv_input.strip()

    # Strip trailing .pdf
    if text.lower().endswith(".pdf"):
        text = text[:-4]

    # Try to extract from URL
    url_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s/?#]+)", text)
    if url_match:
        arxiv_id = url_match.group(1)
    else:
        arxiv_id = text

    # Strip version suffix (e.g. v1, v2)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    # Basic validation
    if not re.match(r"^[\d.]+$", arxiv_id) and not re.match(
        r"^[a-zA-Z\-]+/\d+$", arxiv_id
    ):
        raise ValueError(
            f"Cannot parse arXiv ID from input: '{arxiv_input}'. "
            "Expected a URL like https://arxiv.org/abs/1706.03762, "
            "or a bare ID like 1706.03762"
        )

    return arxiv_id


def fetch_arxiv_pdf(arxiv_input: str) -> str:
    """
    Download a PDF from arXiv and save it locally.

    Args:
        arxiv_input: arXiv URL, PDF URL, or bare ID.

    Returns:
        Local file path where the PDF was saved.

    Raises:
        ValueError:  If the input cannot be parsed.
        RuntimeError: If the download fails after retries.
    """
    arxiv_id = _normalise_arxiv_id(arxiv_input)
    safe_filename = arxiv_id.replace("/", "_")
    local_path = os.path.join(PDF_STORAGE_DIR, f"{safe_filename}.pdf")

    # Skip download if already cached
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        logger.info("PDF already cached at %s", local_path)
        return local_path

    download_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    logger.info("Downloading arXiv PDF from %s", download_url)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(download_url, timeout=_REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(local_path)
            logger.info(
                "Saved PDF (%d bytes) to %s", file_size, local_path
            )
            return local_path

        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "Download attempt %d/3 failed: %s", attempt, exc
            )
            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Failed to download arXiv PDF after 3 attempts: {last_error}"
    )


def get_paper_metadata(arxiv_id: str) -> Dict[str, Any]:
    """
    Retrieve paper metadata from the arXiv API.

    Args:
        arxiv_id: Bare arXiv ID (e.g. "1706.03762").

    Returns:
        Dict with keys: title, authors, abstract, published, arxiv_id.

    Raises:
        RuntimeError: If the API call fails or returns no results.
    """
    clean_id = _normalise_arxiv_id(arxiv_id)
    api_url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
    logger.info("Fetching metadata for arXiv:%s", clean_id)

    try:
        resp = requests.get(api_url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"arXiv API request failed: {exc}") from exc

    # Parse Atom XML
    root = ET.fromstring(resp.text)
    entries = root.findall(f"{_ATOM_NS}entry")

    if not entries:
        raise RuntimeError(f"No results from arXiv API for ID: {clean_id}")

    entry = entries[0]

    title = (entry.findtext(f"{_ATOM_NS}title") or "").strip().replace("\n", " ")
    abstract = (entry.findtext(f"{_ATOM_NS}summary") or "").strip().replace("\n", " ")
    published = (entry.findtext(f"{_ATOM_NS}published") or "").strip()

    authors = []
    for author_el in entry.findall(f"{_ATOM_NS}author"):
        name = (author_el.findtext(f"{_ATOM_NS}name") or "").strip()
        if name:
            authors.append(name)

    metadata = {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "published": published,
        "arxiv_id": clean_id,
    }
    logger.info("Metadata retrieved: '%s' by %s", title, ", ".join(authors[:3]))
    return metadata
