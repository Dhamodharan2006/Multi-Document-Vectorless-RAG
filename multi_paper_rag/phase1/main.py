"""
Phase 1 — Main CLI Entry Point
================================
Single command-line interface for ingestion, querying, verification,
and tree summary.

Usage:
    python -m multi_paper_rag.phase1.main ingest --arxiv 1706.03762
    python -m multi_paper_rag.phase1.main ingest --pdf path/to/file.pdf
    python -m multi_paper_rag.phase1.main query  --paper-id 1 --question "your question"
    python -m multi_paper_rag.phase1.main verify
    python -m multi_paper_rag.phase1.main summary --paper-id 1
"""

import argparse
import json
import logging
import sys
from typing import Optional

from multi_paper_rag.phase1.config import validate_config, logger as root_logger
from multi_paper_rag.phase1.ingestion.arxiv_fetcher import fetch_arxiv_pdf, get_paper_metadata
from multi_paper_rag.phase1.tree.tree_builder import build_tree
from multi_paper_rag.phase1.tree.tree_storage import (
    save_tree,
    load_tree_by_paper_id,
    get_tree_summary,
    get_next_paper_id,
    list_all_trees,
)
from multi_paper_rag.phase1.llm.tree_navigator import query_paper

logger = logging.getLogger("phase1.main")


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Ingest a paper from arXiv or a local PDF.

    Downloads (if arXiv), extracts text, builds the PageIndex tree,
    saves the tree JSON, and prints a summary.

    Args:
        args: Parsed CLI args with ``arxiv`` or ``pdf`` attribute.

    Raises:
        SystemExit: On fatal errors.
    """
    validate_config()

    metadata: dict = {}
    pdf_path: str = ""

    if args.arxiv:
        # ── arXiv path ──────────────────────────────────────────────
        logger.info("Ingesting arXiv paper: %s", args.arxiv)
        try:
            pdf_path = fetch_arxiv_pdf(args.arxiv)
        except Exception as exc:
            logger.error("Failed to fetch arXiv PDF: %s", exc)
            sys.exit(1)

        try:
            metadata = get_paper_metadata(args.arxiv)
        except Exception as exc:
            logger.warning("Could not fetch arXiv metadata: %s", exc)
            metadata = {"title": f"arXiv:{args.arxiv}", "authors": [], "arxiv_id": args.arxiv}

    elif args.pdf:
        # ── Local PDF path ──────────────────────────────────────────
        import os
        if not os.path.exists(args.pdf):
            logger.error("PDF file not found: %s", args.pdf)
            sys.exit(1)
        pdf_path = args.pdf
        metadata = {"title": os.path.basename(args.pdf), "authors": []}

    else:
        logger.error("Either --arxiv or --pdf must be provided")
        sys.exit(1)

    # ── Build tree ──────────────────────────────────────────────────
    paper_id = get_next_paper_id()
    logger.info("Building tree for paper_id=%d from %s", paper_id, pdf_path)

    tree = build_tree(pdf_path, paper_id, metadata)

    # ── Save tree ───────────────────────────────────────────────────
    saved_path = save_tree(tree)
    logger.info("Tree saved to %s", saved_path)

    # ── Print summary ───────────────────────────────────────────────
    summary = get_tree_summary(tree)
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Paper ID     : {summary['paper_id']}")
    print(f"  Title        : {summary['title']}")
    print(f"  arXiv ID     : {summary.get('arxiv_id', 'N/A')}")
    print(f"  Authors      : {', '.join(summary.get('authors', [])[:5])}")
    print(f"  Total Pages  : {summary['total_pages']}")
    print(f"  Sections     : {summary['total_sections']} ({summary['top_level_sections']} top-level)")
    print(f"  Build Method : {summary['build_method']}")
    print(f"  Tree File    : {saved_path}")
    print("=" * 60 + "\n")


def cmd_query(args: argparse.Namespace) -> None:
    """
    Query a previously ingested paper.

    Args:
        args: Parsed CLI args with ``paper_id`` and ``question``.

    Raises:
        SystemExit: If paper not found or query fails.
    """
    validate_config()

    tree = load_tree_by_paper_id(args.paper_id)
    if tree is None:
        logger.error("No tree found for paper_id=%d", args.paper_id)
        print(f"\nError: No ingested paper with id={args.paper_id}")
        print("Run 'summary' to see all ingested papers, or 'ingest' to add one.\n")
        sys.exit(1)

    logger.info("Querying paper_id=%d: %s", args.paper_id, args.question)

    try:
        result = query_paper(args.question, tree)
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  QUERY RESULT")
    print("=" * 60)
    print(f"  Paper  : [{result['paper_id']}] {result['paper_title']}")
    print(f"  Query  : {args.question}")
    print("-" * 60)
    print(f"\n{result['answer']}\n")
    print("-" * 60)
    print("  Cited sections:")
    for cs in result["cited_sections"]:
        print(f"    • {cs['section_id']}: {cs['title']}  (pp. {cs['page_start']}-{cs['page_end']})")
    print("=" * 60 + "\n")


def cmd_summary(args: argparse.Namespace) -> None:
    """
    Print the tree structure summary for a paper or list all papers.

    Args:
        args: Parsed CLI args with ``paper_id``.
    """
    if args.paper_id:
        tree = load_tree_by_paper_id(args.paper_id)
        if tree is None:
            print(f"\nNo tree found for paper_id={args.paper_id}\n")
            sys.exit(1)

        summary = get_tree_summary(tree)
        print("\n" + "=" * 60)
        print("  PAPER TREE SUMMARY")
        print("=" * 60)
        print(f"  Paper ID     : {summary['paper_id']}")
        print(f"  Title        : {summary['title']}")
        print(f"  arXiv ID     : {summary.get('arxiv_id', 'N/A')}")
        print(f"  Build Method : {summary['build_method']}")
        print(f"  Total Pages  : {summary['total_pages']}")
        print(f"  Sections     : {summary['total_sections']}")
        print("-" * 60)
        print("  Section tree:")
        _print_section_tree(tree.sections, indent=4)
        print("=" * 60 + "\n")
    else:
        # List all papers
        all_trees = list_all_trees()
        if not all_trees:
            print("\nNo papers ingested yet. Use 'ingest' to add one.\n")
            return
        print("\n" + "=" * 60)
        print("  INGESTED PAPERS")
        print("=" * 60)
        for s in all_trees:
            print(f"  [{s['paper_id']}] {s['title']}")
            print(f"      Method: {s['build_method']} | Pages: {s['total_pages']} | Sections: {s['total_sections']}")
        print("=" * 60 + "\n")


def cmd_verify(args: argparse.Namespace) -> None:
    """
    Run the Phase 1 verification suite.

    Args:
        args: Parsed CLI args (unused, but kept for interface consistency).
    """
    from multi_paper_rag.phase1.verification.phase1_verifier import run_verification
    run_verification()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_section_tree(sections, indent: int = 0) -> None:
    """Recursively print a section tree with indentation."""
    for sec in sections:
        prefix = " " * indent
        words = sec.word_count
        pages = f"pp. {sec.page_start}-{sec.page_end}"
        print(f"{prefix}|- [{sec.section_id}] {sec.title}  ({pages}, {words} words)")
        if sec.children:
            _print_section_tree(sec.children, indent + 4)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argparse parser with all Phase 1 subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="phase1",
        description="Phase 1 — Multi-Paper Research Intelligence Assistant",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG-level logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── ingest ──────────────────────────────────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="Ingest a paper (arXiv or local PDF)")
    ingest_group = p_ingest.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--arxiv", type=str, help="arXiv ID or URL")
    ingest_group.add_argument("--pdf", type=str, help="Path to a local PDF file")

    # ── query ───────────────────────────────────────────────────────
    p_query = subparsers.add_parser("query", help="Query an ingested paper")
    p_query.add_argument("--paper-id", type=int, required=True, help="Paper ID to query")
    p_query.add_argument("--question", type=str, required=True, help="Your question")

    # ── summary ─────────────────────────────────────────────────────
    p_summary = subparsers.add_parser("summary", help="Show tree summary")
    p_summary.add_argument("--paper-id", type=int, default=None, help="Paper ID (omit to list all)")

    # ── verify ──────────────────────────────────────────────────────
    subparsers.add_parser("verify", help="Run Phase 1 verification tests")

    return parser


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "summary": cmd_summary,
        "verify": cmd_verify,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
