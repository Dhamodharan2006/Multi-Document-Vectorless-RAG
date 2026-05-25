"""
Tree Navigator Module  (Query Engine for Phase 1)
==================================================
Navigates the PageIndex tree to answer natural-language questions.

Workflow:
    1. ``find_relevant_sections``  — ask Groq which sections are relevant.
    2. ``generate_answer``         — ask Groq to answer using those sections.
    3. ``query_paper``             — public orchestrator called from main.py.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from multi_paper_rag.phase1.llm.groq_client import call_groq, GroqClientError
from multi_paper_rag.phase1.tree.tree_schema import PaperTree, SectionNode

logger = logging.getLogger("phase1.llm.tree_navigator")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_paper(query: str, tree: PaperTree) -> Dict[str, Any]:
    """
    Answer a natural-language question about a paper using its tree.

    Orchestrates section retrieval and answer generation in two steps —
    this is a simple pipeline, **not** an agent.

    Args:
        query: The user's question.
        tree:  The PaperTree to query against.

    Returns:
        Dict with keys: answer, cited_sections, paper_id, paper_title.

    Raises:
        GroqClientError: If the Groq API is unreachable after retries.
    """
    logger.info("Querying paper '%s' (id=%d): %s", tree.title, tree.paper_id, query)

    relevant = find_relevant_sections(query, tree, top_n=3)
    if not relevant:
        logger.warning("No relevant sections found — returning all top-level sections")
        relevant = tree.sections[:3]  # Fallback: use first 3 sections

    metadata = {
        "paper_id": tree.paper_id,
        "title": tree.title,
        "arxiv_id": tree.arxiv_id,
        "authors": tree.authors,
    }

    return generate_answer(query, relevant, metadata)


# ---------------------------------------------------------------------------
# Step 1 — Section Selection
# ---------------------------------------------------------------------------

_SECTION_SELECT_SYSTEM = """You are a research paper section selector.
Given a question and a list of paper sections, identify the most relevant sections.
Return ONLY a valid JSON array of section_id strings.
No explanation. No markdown fences. Just the JSON array.
Example: ["1", "3.2", "4.1"]"""


def find_relevant_sections(
    query: str,
    tree: PaperTree,
    top_n: int = 3,
) -> List[SectionNode]:
    """
    Use Groq to select the most relevant sections for a query.

    Builds a compact section index (id, title, first 200 chars) and asks
    the LLM to rank them.

    Args:
        query: The user's question.
        tree:  The PaperTree to search.
        top_n: Maximum number of sections to return.

    Returns:
        List of matching SectionNode objects (up to top_n).

    Raises:
        GroqClientError: If the Groq call fails after retries.
    """
    all_sections = tree.get_all_sections_flat()
    if not all_sections:
        logger.warning("Tree has no sections")
        return []

    # Build compact index
    section_index = []
    for sec in all_sections:
        preview = sec.content[:200].replace("\n", " ").strip() if sec.content else ""
        section_index.append(
            {
                "section_id": sec.section_id,
                "title": sec.title,
                "preview": preview,
            }
        )

    user_prompt = (
        f"Question: {query}\n\n"
        f"Paper title: {tree.title}\n\n"
        f"Available sections:\n{json.dumps(section_index, indent=1)}\n\n"
        f"Return the section_ids of the {top_n} sections most likely "
        f"to contain the answer. Return ONLY a JSON array."
    )

    try:
        raw = call_groq(_SECTION_SELECT_SYSTEM, user_prompt, temperature=0.1, max_tokens=500)
    except GroqClientError:
        logger.error("Groq section selection failed — returning first %d sections", top_n)
        return all_sections[:top_n]

    # Parse the JSON array from the response
    selected_ids = _parse_id_list(raw)
    if not selected_ids:
        logger.warning("Could not parse section IDs from Groq — returning first %d", top_n)
        return all_sections[:top_n]

    # Fetch full SectionNode objects
    result: List[SectionNode] = []
    for sid in selected_ids[:top_n]:
        node = tree.find_section_by_id(sid)
        if node:
            result.append(node)
        else:
            logger.debug("Section ID '%s' not found in tree", sid)

    if not result:
        logger.warning("None of the selected IDs matched — returning first %d", top_n)
        return all_sections[:top_n]

    logger.info(
        "Selected %d sections: %s",
        len(result),
        [f"{s.section_id} ({s.title})" for s in result],
    )
    return result


# ---------------------------------------------------------------------------
# Step 2 — Answer Generation
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = """You are a research paper analysis assistant.
Answer the following question using ONLY the provided sections from the paper.

Rules:
- Only use information present in the provided sections
- For every claim, cite the section title and page range in brackets, e.g. [Section Title | Pages 3-5]
- If the answer is not in the provided sections, say: 'This information is not available in the retrieved sections'
- Do not hallucinate or add external knowledge
- Be thorough and detailed in your answer"""


def generate_answer(
    query: str,
    relevant_sections: List[SectionNode],
    paper_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a cited answer from the relevant sections using Groq.

    Args:
        query:             The user's question.
        relevant_sections: List of SectionNode objects selected in step 1.
        paper_metadata:    Dict with at least: paper_id, title.

    Returns:
        Dict with keys:
            answer          (str)  — Full answer text with citations.
            cited_sections  (list) — Metadata for each cited section.
            paper_id        (int)  — Paper identifier.
            paper_title     (str)  — Paper title.

    Raises:
        GroqClientError: If the Groq call fails after retries.
    """
    title = paper_metadata.get("title", "Unknown Paper")

    # Build context
    context_parts: List[str] = []
    cited_sections: List[Dict[str, Any]] = []
    for sec in relevant_sections:
        # Truncate very long sections to stay within token limits
        content = sec.content[:4000] if len(sec.content) > 4000 else sec.content
        context_parts.append(
            f"[{sec.title} | Pages {sec.page_start}-{sec.page_end}]\n{content}"
        )
        cited_sections.append(
            {
                "section_id": sec.section_id,
                "title": sec.title,
                "page_start": sec.page_start,
                "page_end": sec.page_end,
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    user_prompt = (
        f"Paper: {title}\n\n"
        f"Question: {query}\n\n"
        f"Paper sections:\n{context}"
    )

    answer_text = call_groq(
        _ANSWER_SYSTEM, user_prompt, temperature=0.2, max_tokens=2000
    )

    return {
        "answer": answer_text.strip(),
        "cited_sections": cited_sections,
        "paper_id": paper_metadata.get("paper_id", 0),
        "paper_title": title,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_id_list(text: str) -> List[str]:
    """
    Parse a JSON array of section-id strings from an LLM response.

    Handles markdown fences and other decorative text around the array.

    Args:
        text: Raw LLM response.

    Returns:
        List of section_id strings, or empty list on parse failure.
    """
    cleaned = text.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Try to extract array from text
    match = re.search(r"\[[\s\S]*?\]", cleaned)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    return []
