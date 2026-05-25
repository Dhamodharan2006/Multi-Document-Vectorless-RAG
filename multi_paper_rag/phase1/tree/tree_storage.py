"""
Tree Storage Module
===================
Handles saving, loading, and summarising PaperTree objects as JSON on disk.
This is the disk-based cache for Phase 1 (Redis replaces it in Phase 3).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from multi_paper_rag.phase1.tree.tree_schema import PaperTree, SectionNode
from multi_paper_rag.phase1.config import TREE_STORAGE_DIR

logger = logging.getLogger("phase1.tree.tree_storage")


def save_tree(tree: PaperTree, storage_dir: Optional[str] = None) -> str:
    """
    Serialise a PaperTree to a human-readable JSON file on disk.

    The file is named ``{file_hash}.json`` so that the same PDF always
    maps to the same cache key.

    Args:
        tree:        The PaperTree instance to persist.
        storage_dir: Directory to write into.  Defaults to TREE_STORAGE_DIR.

    Returns:
        Absolute path of the saved JSON file.

    Raises:
        OSError: If the file cannot be written.
    """
    directory = storage_dir or TREE_STORAGE_DIR
    os.makedirs(directory, exist_ok=True)

    file_name = f"{tree.file_hash}.json"
    file_path = os.path.join(directory, file_name)

    data = tree.model_dump(mode="json")          # Pydantic v2 serialisation
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Tree saved → %s  (%d bytes)", file_path, os.path.getsize(file_path))
    return file_path


def load_tree(file_hash: str, storage_dir: Optional[str] = None) -> Optional[PaperTree]:
    """
    Load a PaperTree from its JSON cache file.

    Args:
        file_hash:   SHA-256 hex digest used as the file name stem.
        storage_dir: Directory to search.  Defaults to TREE_STORAGE_DIR.

    Returns:
        A PaperTree instance if the file exists, otherwise ``None``.

    Raises:
        ValueError: If the JSON is present but cannot be deserialised.
    """
    directory = storage_dir or TREE_STORAGE_DIR
    file_path = os.path.join(directory, f"{file_hash}.json")

    if not os.path.exists(file_path):
        logger.debug("No cached tree found for hash %s", file_hash)
        return None

    logger.info("Loading cached tree from %s", file_path)
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    try:
        tree = PaperTree.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"Failed to deserialise tree from {file_path}: {exc}"
        ) from exc

    logger.info("Loaded tree: paper_id=%d, title='%s'", tree.paper_id, tree.title)
    return tree


def load_tree_by_paper_id(
    paper_id: int, storage_dir: Optional[str] = None
) -> Optional[PaperTree]:
    """
    Scan the storage directory for a tree with the given paper_id.

    This is a linear scan — acceptable for Phase 1's small scale.

    Args:
        paper_id:    The sequential paper identifier to search for.
        storage_dir: Directory to scan.  Defaults to TREE_STORAGE_DIR.

    Returns:
        The matching PaperTree, or ``None`` if not found.
    """
    directory = storage_dir or TREE_STORAGE_DIR
    if not os.path.isdir(directory):
        return None

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("paper_id") == paper_id:
                return PaperTree.model_validate(data)
        except Exception:
            continue

    logger.debug("No tree found for paper_id=%d", paper_id)
    return None


def get_next_paper_id(storage_dir: Optional[str] = None) -> int:
    """
    Determine the next available paper_id by scanning existing trees.

    Args:
        storage_dir: Directory to scan.  Defaults to TREE_STORAGE_DIR.

    Returns:
        Next integer paper_id (starts at 1 if no trees exist).
    """
    directory = storage_dir or TREE_STORAGE_DIR
    if not os.path.isdir(directory):
        return 1

    max_id = 0
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            pid = data.get("paper_id", 0)
            if pid > max_id:
                max_id = pid
        except Exception:
            continue

    return max_id + 1


def get_tree_summary(tree: PaperTree) -> Dict[str, Any]:
    """
    Build a lightweight metadata summary of a PaperTree (no full content).

    Useful for logging, verification display, and the ``summary`` CLI command.

    Args:
        tree: The PaperTree to summarise.

    Returns:
        Dict with keys: paper_id, title, arxiv_id, total_sections,
        section_titles, build_method, total_pages, authors.
    """
    all_sections: List[SectionNode] = tree.get_all_sections_flat()
    section_titles = [s.title for s in all_sections]

    return {
        "paper_id": tree.paper_id,
        "title": tree.title,
        "arxiv_id": tree.arxiv_id,
        "authors": tree.authors,
        "total_sections": len(all_sections),
        "top_level_sections": sum(1 for s in all_sections if s.level == 1),
        "section_titles": section_titles,
        "build_method": tree.build_method,
        "total_pages": tree.total_pages,
        "built_at": str(tree.built_at),
    }


def list_all_trees(storage_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return summaries of all trees stored on disk.

    Args:
        storage_dir: Directory to scan.  Defaults to TREE_STORAGE_DIR.

    Returns:
        List of summary dicts (one per tree), sorted by paper_id.
    """
    directory = storage_dir or TREE_STORAGE_DIR
    if not os.path.isdir(directory):
        return []

    summaries: List[Dict[str, Any]] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            tree = PaperTree.model_validate(data)
            summaries.append(get_tree_summary(tree))
        except Exception as exc:
            logger.warning("Skipping corrupt tree file %s: %s", filename, exc)

    summaries.sort(key=lambda s: s["paper_id"])
    return summaries
