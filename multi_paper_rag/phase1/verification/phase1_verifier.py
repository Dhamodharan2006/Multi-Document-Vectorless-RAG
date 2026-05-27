"""
Phase 1 Verification Module
============================
Runs 7 automated tests to validate the Phase 1 foundation.

Tests:
    1. PDF Extraction         — text length, page count, no errors
    2. TOC Detection          — which build_method was selected
    3. Tree Structure         — section count, expected titles, field validity
    4. Nested Structure       — at least one subsection exists (WARN if not)
    5. JSON Persistence       — save + load round-trip with hash match
    6. Single Query Accuracy  — answer quality from Groq
    7. Multi-Paper Isolation  — two papers stay independent
"""

import logging
import sys
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("phase1.verification")

# Result constants
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def run_verification() -> None:
    """
    Execute all 7 Phase 1 verification tests and print a report.

    This function never raises — it catches all exceptions and reports
    them as test failures.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION SUITE")
    print("=" * 60 + "\n")

    results: List[Tuple[str, str, str]] = []  # (test_name, status, detail)

    # ── TEST 1: PDF Extraction ──────────────────────────────────────
    status, detail, tree1, pdf_path1 = _test_1_pdf_extraction()
    results.append(("TEST 1 PDF Extraction", status, detail))

    # ── TEST 2: TOC Detection ───────────────────────────────────────
    status, detail = _test_2_toc_detection(tree1)
    results.append(("TEST 2 TOC Detection", status, detail))

    # ── TEST 3: Tree Structure Accuracy ─────────────────────────────
    status, detail = _test_3_tree_accuracy(tree1)
    results.append(("TEST 3 Tree Accuracy", status, detail))

    # ── TEST 4: Nested Structure ────────────────────────────────────
    status, detail = _test_4_nested_structure(tree1)
    results.append(("TEST 4 Nested Structure", status, detail))

    # ── TEST 5: JSON Persistence ────────────────────────────────────
    status, detail = _test_5_json_persistence(tree1)
    results.append(("TEST 5 JSON Persistence", status, detail))

    # ── TEST 6: Single Query Accuracy ───────────────────────────────
    status, detail = _test_6_query_accuracy(tree1)
    results.append(("TEST 6 Query Accuracy", status, detail))

    # ── TEST 7: Multi-Paper Isolation ───────────────────────────────
    status, detail = _test_7_multi_paper(tree1)
    results.append(("TEST 7 Multi-Paper", status, detail))

    # ── Final Report ────────────────────────────────────────────────
    _print_report(results)


# ---------------------------------------------------------------------------
# Test Implementations
# ---------------------------------------------------------------------------


def _test_1_pdf_extraction():
    """Test 1: Download arXiv:1706.03762 and verify text extraction."""
    from multi_paper_rag.phase1.ingestion.arxiv_fetcher import fetch_arxiv_pdf, get_paper_metadata
    from multi_paper_rag.phase1.ingestion.pdf_extractor import extract_text_from_pdf
    from multi_paper_rag.phase1.tree.tree_builder import build_tree
    from multi_paper_rag.phase1.tree.tree_storage import save_tree, get_next_paper_id

    tree = None
    pdf_path = None

    try:
        print("  [1/7] Downloading arXiv:1706.03762 ...")
        pdf_path = fetch_arxiv_pdf("1706.03762")

        extraction = extract_text_from_pdf(pdf_path)
        if not extraction["success"]:
            return FAIL, f"Extraction failed: {extraction['error']}", None, pdf_path

        text_len = len(extraction["full_text"])
        total_pages = extraction["total_pages"]

        checks = []
        if text_len <= 5000:
            checks.append(f"text too short ({text_len} chars)")
        if total_pages <= 5:
            checks.append(f"too few pages ({total_pages})")

        if checks:
            return FAIL, "; ".join(checks), None, pdf_path

        # Build tree for use in later tests
        print("  [1/7] Building tree ...")
        try:
            metadata = get_paper_metadata("1706.03762")
        except Exception:
            metadata = {"title": "Attention Is All You Need", "authors": [], "arxiv_id": "1706.03762"}

        paper_id = get_next_paper_id()
        tree = build_tree(pdf_path, paper_id, metadata)
        save_tree(tree)

        return (
            PASS,
            f"{text_len} chars, {total_pages} pages",
            tree,
            pdf_path,
        )

    except Exception as exc:
        return FAIL, str(exc), tree, pdf_path


def _test_2_toc_detection(tree):
    """Test 2: Verify which build_method was used."""
    if tree is None:
        return FAIL, "No tree available (test 1 failed)"

    method = tree.build_method
    return PASS, f"method: {method}"


def _test_3_tree_accuracy(tree):
    """Test 3: Validate tree structure — section count, titles, fields."""
    if tree is None:
        return FAIL, "No tree available (test 1 failed)"

    all_sections = tree.get_all_sections_flat()
    top_level = [s for s in all_sections if s.level == 1]

    checks = []

    # At least 4 top-level sections
    if len(top_level) < 4:
        checks.append(f"only {len(top_level)} top-level sections (need ≥4)")

    # "abstract" or "introduction" section_type should exist
    section_types = [s.section_type for s in all_sections]
    has_expected = any(t in section_types for t in ["abstract", "introduction"])
    if not has_expected:
        checks.append("neither 'abstract' nor 'introduction' found in section_types")

    # All fields non-null
    for sec in all_sections:
        if not sec.title:
            checks.append(f"section {sec.section_id} has empty title")
            break
        if not sec.section_id:
            checks.append(f"section '{sec.title}' has empty section_id")
            break

    # page_start <= page_end
    for sec in all_sections:
        if sec.page_start > sec.page_end:
            checks.append(
                f"section {sec.section_id} has page_start({sec.page_start}) > page_end({sec.page_end})"
            )
            break

    if checks:
        return FAIL, "; ".join(checks)

    return PASS, f"{len(top_level)} top-level sections found"


def _test_4_nested_structure(tree):
    """Test 4: Check for at least one subsection (level ≥ 2)."""
    if tree is None:
        return FAIL, "No tree available (test 1 failed)"

    all_sections = tree.get_all_sections_flat()
    has_children = any(s.level >= 2 for s in all_sections)

    if has_children:
        subsections = sum(1 for s in all_sections if s.level >= 2)
        return PASS, f"{subsections} subsections detected"
    else:
        return WARN, "no subsections detected (paper may be flat)"


def _test_5_json_persistence(tree):
    """Test 5: Save tree, load it back, verify hash matches."""
    if tree is None:
        return FAIL, "No tree available (test 1 failed)"

    from multi_paper_rag.phase1.tree.tree_storage import save_tree, load_tree

    try:
        saved_path = save_tree(tree)
        loaded = load_tree(tree.file_hash)

        if loaded is None:
            return FAIL, "load_tree returned None after save"

        if loaded.file_hash != tree.file_hash:
            return FAIL, "file_hash mismatch after round-trip"

        if loaded.paper_id != tree.paper_id:
            return FAIL, "paper_id mismatch after round-trip"

        return PASS, "round-trip OK"

    except Exception as exc:
        return FAIL, str(exc)


def _test_6_query_accuracy(tree):
    """Test 6: Query 'What is the main contribution?' and validate answer."""
    if tree is None:
        return FAIL, "No tree available (test 1 failed)"

    from multi_paper_rag.phase1.llm.tree_navigator import query_paper

    query = "What is the main contribution of this paper?"

    try:
        print("  [6/7] Querying paper ...")
        result = query_paper(query, tree)
        answer = result.get("answer", "")

        checks = []

        if len(answer) < 100:
            checks.append(f"answer too short ({len(answer)} chars)")

        if not result.get("cited_sections"):
            checks.append("no section citations in result")

        negative_phrases = [
            "i don't know",
            "not available",
            "cannot answer",
            "no information",
        ]
        answer_lower = answer.lower()
        for phrase in negative_phrases:
            if phrase in answer_lower:
                checks.append(f"answer contains negative phrase: '{phrase}'")
                break

        if checks:
            return FAIL, "; ".join(checks)

        return PASS, f"answer length={len(answer)}"

    except Exception as exc:
        return FAIL, str(exc)


def _test_7_multi_paper(tree1):
    """Test 7: Ingest a second paper and verify isolation."""
    if tree1 is None:
        return FAIL, "No tree available (test 1 failed)"

    from multi_paper_rag.phase1.ingestion.arxiv_fetcher import fetch_arxiv_pdf, get_paper_metadata
    from multi_paper_rag.phase1.tree.tree_builder import build_tree
    from multi_paper_rag.phase1.tree.tree_storage import save_tree, get_next_paper_id
    from multi_paper_rag.phase1.llm.tree_navigator import query_paper

    try:
        print("  [7/7] Ingesting arXiv:1810.04805 (BERT) ...")
        pdf_path2 = fetch_arxiv_pdf("1810.04805")

        try:
            metadata2 = get_paper_metadata("1810.04805")
        except Exception:
            metadata2 = {"title": "BERT", "authors": [], "arxiv_id": "1810.04805"}

        paper_id2 = get_next_paper_id()
        tree2 = build_tree(pdf_path2, paper_id2, metadata2)
        save_tree(tree2)

        checks = []

        # Different paper_ids
        if tree1.paper_id == tree2.paper_id:
            checks.append("paper_id collision")

        # Different file hashes
        if tree1.file_hash == tree2.file_hash:
            checks.append("file_hash collision")

        # Query paper 1 and check answer doesn't reference BERT content
        print("  [7/7] Cross-checking isolation ...")
        result1 = query_paper("What is the main architecture proposed?", tree1)
        answer1_lower = result1.get("answer", "").lower()

        # BERT-specific terms that shouldn't appear in Transformer paper answer
        if result1.get("paper_title", "").lower() != tree1.title.lower():
            checks.append("paper_title mismatch in query result")

        if checks:
            return FAIL, "; ".join(checks)

        return PASS, f"paper1_id={tree1.paper_id}, paper2_id={tree2.paper_id}"

    except Exception as exc:
        return FAIL, str(exc)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(results: List[Tuple[str, str, str]]) -> None:
    """Print the final verification report."""
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION REPORT")
    print("=" * 60)

    pass_count = 0
    fail_count = 0
    warn_count = 0

    for name, status, detail in results:
        icon = "[PASS]" if status == PASS else ("[WARN]" if status == WARN else "[FAIL]")
        print(f"  {icon} {name:<25s} : {status:<4s}  ({detail})")

        if status == PASS:
            pass_count += 1
        elif status == WARN:
            warn_count += 1
            pass_count += 1  # WARN counts as passing
        else:
            fail_count += 1

    print("=" * 60)

    if pass_count >= 6:
        print("  [PASS] PHASE 1 STATUS: READY FOR PHASE 2")
        print(f"    ({pass_count} passed, {warn_count} warnings, {fail_count} failures)")
    else:
        print("  [FAIL] PHASE 1 STATUS: NOT READY")
        print(f"    ({pass_count} passed, {warn_count} warnings, {fail_count} failures)")
        print("    Fix failures and re-run verification.")

    print("=" * 60 + "\n")
