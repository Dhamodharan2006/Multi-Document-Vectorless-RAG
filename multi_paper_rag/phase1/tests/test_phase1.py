"""
Phase 1 — Manual Test Runner
==============================
Quick smoke tests that can be run independently of the full verification suite.
Usage:  python -m multi_paper_rag.phase1.tests.test_phase1
"""

import logging
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_phase1")


def test_config():
    """Test that config loads without errors."""
    from multi_paper_rag.phase1.config import (
        NVIDIA_BASE_URL, NVIDIA_MODEL, GROQ_MODEL,
        PDF_STORAGE_DIR, TREE_STORAGE_DIR
    )
    assert NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert NVIDIA_MODEL == "dracarys-llama-3.1-70b-instruct"
    assert GROQ_MODEL == "llama-3.3-70b-versatile"
    assert PDF_STORAGE_DIR
    assert TREE_STORAGE_DIR
    print("  [OK] config.py OK")


def test_schema():
    """Test Pydantic models validate correctly."""
    from multi_paper_rag.phase1.tree.tree_schema import SectionNode, PaperTree
    from datetime import datetime

    node = SectionNode(
        section_id="1",
        title="Introduction",
        level=1,
        page_start=1,
        page_end=3,
        content="This is a test section with some words in it.",
        children=[],
    )
    assert node.word_count == 10
    assert node.section_id == "1"

    tree = PaperTree(
        paper_id=1,
        title="Test Paper",
        file_path="/tmp/test.pdf",
        file_hash="abc123",
        total_pages=10,
        sections=[node],
        build_method="toc_based",
    )
    assert len(tree.get_all_sections_flat()) == 1
    print("  [OK] tree_schema.py OK")


def test_imports():
    """Test that all modules import without errors."""
    import multi_paper_rag.phase1.config
    import multi_paper_rag.phase1.tree.tree_schema
    import multi_paper_rag.phase1.tree.tree_builder
    import multi_paper_rag.phase1.tree.tree_storage
    import multi_paper_rag.phase1.ingestion.pdf_extractor
    import multi_paper_rag.phase1.ingestion.arxiv_fetcher
    import multi_paper_rag.phase1.llm.nvidia_client
    import multi_paper_rag.phase1.llm.groq_client
    import multi_paper_rag.phase1.llm.tree_navigator
    import multi_paper_rag.phase1.verification.phase1_verifier
    print("  [OK] All modules import OK")


def test_arxiv_id_normalise():
    """Test arXiv ID normalisation logic."""
    from multi_paper_rag.phase1.ingestion.arxiv_fetcher import _normalise_arxiv_id

    assert _normalise_arxiv_id("1706.03762") == "1706.03762"
    assert _normalise_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert _normalise_arxiv_id("https://arxiv.org/pdf/1706.03762") == "1706.03762"
    assert _normalise_arxiv_id("https://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
    assert _normalise_arxiv_id("1706.03762v1") == "1706.03762"
    print("  [OK] arXiv ID normalisation OK")


def test_tree_storage_roundtrip():
    """Test save + load of a dummy tree."""
    from multi_paper_rag.phase1.tree.tree_schema import SectionNode, PaperTree
    from multi_paper_rag.phase1.tree.tree_storage import save_tree, load_tree
    import tempfile, os

    node = SectionNode(
        section_id="1", title="Intro", level=1,
        page_start=1, page_end=2, content="Hello world", children=[]
    )
    tree = PaperTree(
        paper_id=999, title="Test", file_path="/tmp/test.pdf",
        file_hash="test_hash_12345", total_pages=5,
        sections=[node], build_method="toc_based"
    )

    # Use a temp directory for test isolation
    tmpdir = tempfile.mkdtemp()
    try:
        save_tree(tree, storage_dir=tmpdir)
        loaded = load_tree("test_hash_12345", storage_dir=tmpdir)
        assert loaded is not None
        assert loaded.file_hash == tree.file_hash
        assert loaded.paper_id == 999
        assert len(loaded.sections) == 1
        print("  [OK] tree_storage round-trip OK")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("\n  Phase 1 — Smoke Tests")
    print("  " + "-" * 40)

    tests = [test_config, test_schema, test_imports, test_arxiv_id_normalise, test_tree_storage_roundtrip]
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {test_fn.__name__} FAILED: {exc}")
            failed += 1

    print("  " + "-" * 40)
    print(f"  Results: {passed} passed, {failed} failed\n")
    sys.exit(1 if failed > 0 else 0)
