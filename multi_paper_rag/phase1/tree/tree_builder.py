"""
Tree Builder Module  (Core of Phase 1)
=======================================
Builds a hierarchical PageIndex section tree from a PDF.

Strategy:
    PATH A — TOC-Based:  Uses PyMuPDF's embedded TOC when available.
    PATH B — LLM-Inferred:  Sends text to NVIDIA NIM to identify sections.
    FALLBACK — Chunked:  Splits into ~1500-word chunks if both paths fail.

Inspired by the VectifyAI/PageIndex approach — builds a tree structure
that an LLM can reason over for retrieval, instead of vector similarity.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from multi_paper_rag.phase1.ingestion.pdf_extractor import extract_text_from_pdf, get_toc
from multi_paper_rag.phase1.llm.nvidia_client import call_nvidia, NVIDIAClientError
from multi_paper_rag.phase1.tree.tree_schema import PaperTree, SectionNode

logger = logging.getLogger("phase1.tree.tree_builder")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_tree(
    pdf_path: str,
    paper_id: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> PaperTree:
    """
    Build a hierarchical PageIndex tree from a PDF file.

    This function tries three strategies in order:
        1. TOC-based extraction (if the PDF has an embedded outline).
        2. LLM-inferred structure (NVIDIA NIM analyses the text).
        3. Fallback chunking (~1500 words per part).

    Args:
        pdf_path:  Path to the PDF file on disk.
        paper_id:  Sequential paper identifier (starts at 1).
        metadata:  Optional dict with keys: title, authors, arxiv_id, abstract.

    Returns:
        A fully populated ``PaperTree`` instance.
        Never raises — always returns a valid tree (falls back to chunking).
    """
    metadata = metadata or {}
    file_hash = _compute_file_hash(pdf_path)

    # ── Extract text ────────────────────────────────────────────────────
    extraction = extract_text_from_pdf(pdf_path)
    if not extraction["success"]:
        logger.warning(
            "PDF extraction had issues: %s — proceeding with available text",
            extraction["error"],
        )

    full_text: str = extraction["full_text"]
    pages: List[Dict] = extraction["pages"]
    total_pages: int = extraction["total_pages"]

    # ── PATH A: TOC-based ───────────────────────────────────────────────
    sections: List[SectionNode] = []
    build_method = ""

    try:
        toc_entries = get_toc(pdf_path)
        if toc_entries and len(toc_entries) >= 3:
            logger.info("TOC found with %d entries — using TOC-based path", len(toc_entries))
            sections = _build_from_toc(toc_entries, pages, total_pages, full_text)
            build_method = "toc_based"
    except Exception as exc:
        logger.warning("TOC extraction failed: %s", exc)

    # ── PATH B: LLM-inferred ───────────────────────────────────────────
    if len(sections) < 3:
        logger.info("TOC path yielded < 3 sections — trying LLM-inferred path")
        try:
            sections = _build_from_llm(full_text, pages, total_pages)
            build_method = "llm_inferred"
        except (NVIDIAClientError, Exception) as exc:
            logger.warning("LLM-inferred path failed: %s", exc)

    # ── FALLBACK: chunked ──────────────────────────────────────────────
    if len(sections) < 3:
        logger.warning(
            "Both TOC and LLM paths produced < 3 sections — falling back to chunking"
        )
        sections = _build_fallback_chunks(full_text, total_pages)
        build_method = "fallback_chunked"

    # ── Assemble PaperTree ─────────────────────────────────────────────
    extracted_meta = _extract_metadata_from_text(full_text, pdf_path)
    
    tree = PaperTree(
        paper_id=paper_id,
        title=metadata.get("title") or extracted_meta["title"],
        arxiv_id=metadata.get("arxiv_id") or extracted_meta["arxiv_id"],
        authors=metadata.get("authors") if metadata.get("authors") else extracted_meta["authors"],
        file_path=pdf_path,
        file_hash=file_hash,
        total_pages=total_pages,
        sections=sections,
        built_at=datetime.utcnow(),
        build_method=build_method,
    )

    top_level = sum(1 for s in sections if s.level == 1)
    logger.info(
        "Tree built: method=%s, top-level=%d, total_sections=%d",
        build_method,
        top_level,
        len(tree.get_all_sections_flat()),
    )
    return tree


# ---------------------------------------------------------------------------
# PATH A — TOC-based
# ---------------------------------------------------------------------------


def _build_from_toc(
    toc_entries: list,
    pages: List[Dict],
    total_pages: int,
    full_text: str,
) -> List[SectionNode]:
    """
    Map PyMuPDF TOC entries to SectionNode objects using character-level slicing.
    """
    flat_sections: List[Dict[str, Any]] = []
    
    # 1. Map TOC to text positions
    positions = []
    search_offset = 0
    for level, title, page_num in toc_entries:
        title = title.strip()
        # Restrict search around the expected page to avoid matching wrong occurrences
        expected_page_start = max(1, page_num)
        
        pos = _find_heading_position(full_text, title, start_offset=search_offset)
        if pos == -1:
            # Fallback: maybe title has slight numbering differences, let's try removing leading numbers
            clean_title = re.sub(r"^[\d\.\s]+", "", title)
            if clean_title:
                pos = _find_heading_position(full_text, clean_title, start_offset=search_offset)
                
        if pos != -1:
            positions.append({"level": level, "title": title, "pos": pos, "page_start": expected_page_start})
            search_offset = pos  # next heading must be after this one
        else:
            # If still not found, we just estimate based on page
            logger.warning("Could not find TOC heading '%s' in text", title)
            # Find character offset for the start of the expected page
            page_char_offset = 0
            cumulative = 0
            for p in pages:
                if p["page_number"] == expected_page_start:
                    page_char_offset = cumulative
                    break
                cumulative += len(p["text"]) + 1
            positions.append({"level": level, "title": title, "pos": max(search_offset, page_char_offset), "page_start": expected_page_start})

    # 1.5 Append standard tail sections if TOC missed them
    tail_sections = ["Limitations", "Acknowledgments", "References", "Appendix"]
    for tail in tail_sections:
        # Check if already in positions
        if any(tail.lower() in p["title"].lower() for p in positions):
            continue
        
        pos = _find_heading_position(full_text, tail, start_offset=search_offset)
        if pos != -1:
            # Found a missing tail section!
            expected_page = _find_page_for_position(pages, full_text, pos)
            positions.append({"level": 1, "title": tail, "pos": pos, "page_start": expected_page})
            search_offset = pos

    # 2. Slice content
    for i, sec in enumerate(positions):
        start_pos = sec["pos"]
        # Determine next position
        next_pos = len(full_text)
        for j in range(i + 1, len(positions)):
            if positions[j]["pos"] > start_pos:
                next_pos = positions[j]["pos"]
                break
                
        content = full_text[start_pos:next_pos].strip()
        page_end = _find_page_for_position(pages, full_text, max(start_pos, next_pos - 1))
        
        flat_sections.append({
            "title": sec["title"],
            "level": sec["level"],
            "page_start": sec["page_start"],
            "page_end": max(sec["page_start"], page_end),
            "content": content,
        })

    _assign_section_ids(flat_sections)
    return _nest_sections(flat_sections)


# ---------------------------------------------------------------------------
# PATH B — LLM-inferred
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a document structure analyser.
Read the following academic paper text and identify ALL section headings in the exact order they appear.
Do not miss any sections. Pay special attention to sections at the end like "Limitations", "Acknowledgments", and "References".
For each section return:
  - The exact heading text as it appears
  - Whether it is a top-level section (level 1) or subsection (level 2 or 3)
Return ONLY valid JSON. No explanation. No markdown fences.
Format:
{
  "sections": [
    {"title": "Abstract", "level": 1},
    {"title": "Introduction", "level": 1},
    {"title": "1.1 Background", "level": 2},
    ...
  ]
}"""


def _build_from_llm(
    full_text: str,
    pages: List[Dict],
    total_pages: int,
) -> List[SectionNode]:
    """
    Use NVIDIA NIM to infer section structure from the paper text.

    Args:
        full_text:   Full concatenated paper text.
        pages:       Per-page extracted text dicts.
        total_pages: Total page count.

    Returns:
        List of top-level SectionNode objects with nested children.

    Raises:
        NVIDIAClientError: If the LLM call fails.
        ValueError:        If the LLM response cannot be parsed.
    """
    # Send first ~6000 chars to keep within token limits
    sample_text = full_text[:6000]
    raw_response = call_nvidia(_LLM_SYSTEM_PROMPT, sample_text, temperature=0.1, max_tokens=2000)

    # Parse JSON from response (handle markdown fences)
    parsed = _parse_json_response(raw_response)
    if not parsed or "sections" not in parsed:
        raise ValueError(f"LLM response missing 'sections' key: {raw_response[:200]}")

    llm_sections = parsed["sections"]
    if not isinstance(llm_sections, list) or len(llm_sections) == 0:
        raise ValueError("LLM returned empty sections list")

    logger.info("LLM identified %d sections", len(llm_sections))

    # For each identified section, locate in full_text and extract content
    flat_sections: List[Dict[str, Any]] = []
    search_offset = 0
    positions = []
    
    for idx, sec in enumerate(llm_sections):
        title = sec.get("title", "").strip()
        level = int(sec.get("level", 1))
        if not title:
            continue

        # Find position in full text
        position = _find_heading_position(full_text, title, start_offset=search_offset)
        if position == -1:
            clean_title = re.sub(r"^[\d\.\s]+", "", title)
            if clean_title:
                position = _find_heading_position(full_text, clean_title, start_offset=search_offset)
                
        if position == -1:
            logger.debug("Could not locate heading '%s' in text — skipping", title)
            continue
            
        positions.append({"level": level, "title": title, "pos": position})
        search_offset = position

    for i, sec in enumerate(positions):
        start_pos = sec["pos"]
        next_pos = len(full_text)
        for j in range(i + 1, len(positions)):
            if positions[j]["pos"] > start_pos:
                next_pos = positions[j]["pos"]
                break

        content = full_text[start_pos:next_pos].strip()

        # Map to page numbers
        page_start = _find_page_for_position(pages, full_text, start_pos)
        page_end = _find_page_for_position(pages, full_text, min(next_pos - 1, len(full_text) - 1))

        flat_sections.append(
            {
                "title": sec["title"],
                "level": sec["level"],
                "page_start": max(1, page_start),
                "page_end": max(1, page_end),
                "content": content,
            }
        )

    _assign_section_ids(flat_sections)
    return _nest_sections(flat_sections)


# ---------------------------------------------------------------------------
# FALLBACK — chunked
# ---------------------------------------------------------------------------


def _build_fallback_chunks(
    full_text: str,
    total_pages: int,
    words_per_chunk: int = 1500,
) -> List[SectionNode]:
    """
    Split document into roughly equal word-count chunks.

    Args:
        full_text:       Full document text.
        total_pages:     Total page count.
        words_per_chunk: Target words per chunk (default 1500).

    Returns:
        List of SectionNode objects labelled "Part 1", "Part 2", etc.
    """
    words = full_text.split()
    sections: List[SectionNode] = []
    part_num = 1

    for start in range(0, len(words), words_per_chunk):
        chunk_words = words[start : start + words_per_chunk]
        content = " ".join(chunk_words)

        # Rough page mapping
        progress = start / max(len(words), 1)
        page_start = max(1, int(progress * total_pages) + 1)
        end_progress = min(1.0, (start + len(chunk_words)) / max(len(words), 1))
        page_end = max(page_start, int(end_progress * total_pages))

        sections.append(
            SectionNode(
                section_id=str(part_num),
                title=f"Part {part_num}",
                level=1,
                page_start=page_start,
                page_end=page_end,
                content=content,
                children=[],
            )
        )
        part_num += 1

    logger.info("Fallback chunking produced %d parts", len(sections))
    return sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hex digest of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_content_for_pages(
    pages: List[Dict], page_start: int, page_end: int
) -> str:
    """
    Concatenate text from pages in the given range (inclusive, 1-indexed).

    Args:
        pages:      List of {"page_number": int, "text": str}.
        page_start: First page (1-indexed).
        page_end:   Last page (1-indexed, inclusive).

    Returns:
        Concatenated text string.
    """
    texts = []
    for p in pages:
        if page_start <= p["page_number"] <= page_end:
            texts.append(p["text"])
    return "\n".join(texts).strip()


def _find_heading_position(full_text: str, heading: str, start_offset: int = 0) -> int:
    """
    Find the position of a heading in the full text using flexible matching.

    Args:
        full_text: Complete document text.
        heading:   Section heading to search for.
        start_offset: Search only after this character offset.

    Returns:
        Character offset of the heading, or -1 if not found.
    """
    search_text = full_text[start_offset:]
    
    # Try exact match first
    pos = search_text.find(heading)
    if pos != -1:
        return pos + start_offset

    # Try case-insensitive
    pos = search_text.lower().find(heading.lower())
    if pos != -1:
        return pos + start_offset

    # Try with collapsed whitespace
    collapsed_heading = re.sub(r"\s+", r"\\s+", re.escape(heading))
    match = re.search(collapsed_heading, search_text, re.IGNORECASE)
    if match:
        return match.start() + start_offset

    return -1


def _find_page_for_position(
    pages: List[Dict], full_text: str, char_position: int
) -> int:
    """
    Determine which page a character position falls on.

    Args:
        pages:         Per-page text dicts.
        full_text:     Full concatenated text.
        char_position: Character offset in full_text.

    Returns:
        Page number (1-indexed).
    """
    cumulative = 0
    for p in pages:
        page_len = len(p["text"]) + 1  # +1 for the join newline
        if cumulative + page_len > char_position:
            return p["page_number"]
        cumulative += page_len
    # Default to last page
    return pages[-1]["page_number"] if pages else 1


def _assign_section_ids(flat_sections: List[Dict]) -> None:
    """
    Assign hierarchical section IDs (e.g. "1", "1.1", "2") in-place.
    Prefers numbering found in the title (e.g. "4 Conclusion" -> "4").
    If no numbering is found, uses a slug of the title or generated counter.
    """
    counters = [0] * 10  # Support up to 10 levels deep
    for sec in flat_sections:
        level = sec["level"]
        counters[level] += 1
        # Reset all deeper counters
        for deeper in range(level + 1, len(counters)):
            counters[deeper] = 0
            
        title = sec["title"].strip()
        # Look for explicit numbering like "1", "1.1", "4.", "IV." 
        # But for now let's just handle decimal numbers "1.2"
        match = re.match(r"^([\d\.]+)\s+", title)
        if match:
            extracted_id = match.group(1).rstrip('.')
            sec["section_id"] = extracted_id
        else:
            # For unnumbered sections like "Acknowledgments", use a slug instead of sequential number
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            sec["section_id"] = slug if slug else ".".join(str(counters[lv]) for lv in range(1, level + 1))


def _nest_sections(flat_sections: List[Dict]) -> List[SectionNode]:
    """
    Convert a flat list of section dicts into a nested tree.

    Args:
        flat_sections: Ordered list of section dicts with level, section_id, etc.

    Returns:
        List of top-level SectionNode objects with children populated.
    """
    if not flat_sections:
        return []

    # Build SectionNode objects
    nodes = [
        SectionNode(
            section_id=s["section_id"],
            title=s["title"],
            level=s["level"],
            page_start=s["page_start"],
            page_end=s["page_end"],
            content=s["content"],
            children=[],
        )
        for s in flat_sections
    ]

    # Nest using a stack
    root_nodes: List[SectionNode] = []
    stack: List[SectionNode] = []

    for node in nodes:
        # Pop stack until we find the parent
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            root_nodes.append(node)

        stack.append(node)

    return root_nodes


def _extract_metadata_from_text(full_text: str, file_path: str) -> Dict[str, Any]:
    """
    Best-effort metadata extraction from the first few lines of text.

    Args:
        full_text: Full document text.
        file_path: Fallback filename for title.

    Returns:
        Dict with 'title', 'authors', and 'arxiv_id'.
    """
    meta = {
        "title": Path(file_path).name,
        "authors": [],
        "arxiv_id": None
    }
    
    lines = full_text[:2000].split("\n")
    cleaned_lines = [l.strip() for l in lines if l.strip()]
    
    # Extract arXiv ID
    arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", full_text[:3000], re.IGNORECASE)
    if arxiv_match:
        meta["arxiv_id"] = arxiv_match.group(1)
        
    # Title is usually the first non-trivial line
    for line in cleaned_lines:
        if len(line) > 10 and not line.lower().startswith("arxiv") and not line.lower().startswith("abstract"):
            meta["title"] = line[:200]
            break
            
    # Try to extract authors (heuristics: lines after title, before abstract, often separated by commas or containing known names)
    authors_text = []
    found_title = False
    for line in cleaned_lines[:20]:
        if meta["title"] in line:
            found_title = True
            continue
        if found_title:
            if line.lower().startswith("abstract"):
                break
            # A simple heuristic: if a line has commas and words, it might be authors or affiliations
            if len(line) > 5 and len(line.split()) < 20:
                # remove email addresses
                line = re.sub(r"\S+@\S+", "", line).strip()
                if line:
                    authors_text.append(line)
                    
    if authors_text:
        # Join and split by common separators
        joined = " ".join(authors_text)
        # Split by comma or 'and'
        parts = re.split(r",|\band\b", joined)
        meta["authors"] = [p.strip() for p in parts if p.strip() and len(p.strip().split()) <= 4]

    return meta


def _parse_json_response(text: str) -> Optional[Dict]:
    """
    Parse a JSON object from an LLM response, handling markdown fences.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict, or None on failure.
    """
    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse JSON from LLM response")
    return None
