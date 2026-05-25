"""
Tree Schema Module
==================
Pydantic v2 data models for the PageIndex hierarchical section tree.

Models:
    - SectionNode : A single section/subsection in the paper.
    - PaperTree   : The full tree representation of one paper.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field


class SectionNode(BaseModel):
    """
    Represents a single section or subsection in an academic paper.

    Attributes:
        section_id: Hierarchical identifier, e.g. "1", "1.1", "1.2.3".
        title:      Exact section heading as it appears in the paper.
        level:      Depth in hierarchy — 1 = top-level, 2 = subsection, etc.
        page_start: First page where this section appears (1-indexed).
        page_end:   Last page where this section appears (1-indexed).
        content:    Full text of this section.
        children:   Nested subsections.
        word_count: Number of words in *content* (auto-calculated).
    """

    section_id: str = Field(..., description="Hierarchical section identifier, e.g. '1', '1.1'")
    title: str = Field(..., description="Exact section heading from the paper")
    level: int = Field(..., ge=1, le=5, description="Depth: 1=top, 2=sub, 3=subsub")
    page_start: int = Field(..., ge=0, description="First page of this section (1-indexed)")
    page_end: int = Field(..., ge=0, description="Last page of this section (1-indexed)")
    content: str = Field(default="", description="Full text content of this section")
    children: List[SectionNode] = Field(default_factory=list, description="Nested subsections")

    @computed_field  # type: ignore[misc]
    @property
    def word_count(self) -> int:
        """Auto-calculated word count from content."""
        return len(self.content.split()) if self.content else 0


class PaperTree(BaseModel):
    """
    Full hierarchical tree representation of a single academic paper.

    Attributes:
        paper_id:     Sequential identifier assigned at ingest time (starts at 1).
        title:        Paper title from arXiv metadata or PDF header.
        arxiv_id:     arXiv identifier, if applicable.
        authors:      List of author names.
        file_path:    Local path to the PDF file.
        file_hash:    SHA-256 hex digest of the PDF bytes.
        total_pages:  Total number of pages in the PDF.
        sections:     Top-level section nodes (children nest recursively).
        built_at:     Timestamp when the tree was constructed.
        build_method: One of "toc_based", "llm_inferred", or "fallback_chunked".
    """

    paper_id: int = Field(..., ge=1, description="Sequential paper identifier")
    title: str = Field(..., description="Paper title")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID if from arXiv")
    authors: List[str] = Field(default_factory=list, description="Author names")
    file_path: str = Field(..., description="Local PDF file path")
    file_hash: str = Field(..., description="SHA-256 hex digest of PDF bytes")
    total_pages: int = Field(..., ge=0, description="Total PDF page count")
    sections: List[SectionNode] = Field(default_factory=list, description="Top-level sections")
    built_at: datetime = Field(default_factory=datetime.utcnow, description="Build timestamp")
    build_method: str = Field(
        ...,
        description="Tree construction method: toc_based | llm_inferred | fallback_chunked",
    )

    def get_all_sections_flat(self) -> List[SectionNode]:
        """
        Flatten the tree into a single list of all SectionNode objects
        (depth-first traversal).

        Returns:
            List of every SectionNode in the tree.
        """
        result: List[SectionNode] = []

        def _walk(nodes: List[SectionNode]) -> None:
            for node in nodes:
                result.append(node)
                if node.children:
                    _walk(node.children)

        _walk(self.sections)
        return result

    def find_section_by_id(self, section_id: str) -> Optional[SectionNode]:
        """
        Find a SectionNode by its section_id.

        Args:
            section_id: The hierarchical section identifier to search for.

        Returns:
            The matching SectionNode, or None if not found.
        """
        for section in self.get_all_sections_flat():
            if section.section_id == section_id:
                return section
        return None
