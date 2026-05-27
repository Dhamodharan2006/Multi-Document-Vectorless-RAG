"""
Tree Schema Module
==================
Pydantic v2 data models for the PageIndex hierarchical section tree.

Models:
    - SectionNode        : A single section/subsection in the paper.
    - PaperTree          : The full tree representation of one paper.
    - TableObject        : Structured table extracted from the paper.
    - FigureObject       : Figure metadata and caption.
    - EquationObject     : Mathematical equation with LaTeX.
    - ReferenceObject    : Structured citation from the references section.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Section type enum values (as strings for JSON compatibility)
# ---------------------------------------------------------------------------
SECTION_TYPES = [
    "abstract", "introduction", "related_work", "background",
    "methodology", "experiments", "results", "discussion",
    "conclusion", "limitations", "future_work", "acknowledgments",
    "references", "appendix", "other",
]

PAPER_TYPES = [
    "conference_paper", "journal_article", "arxiv_preprint",
    "workshop_paper", "thesis", "technical_report", "survey", "other",
]


# ---------------------------------------------------------------------------
# Extracted object models (tables, figures, equations, references)
# ---------------------------------------------------------------------------

class TableObject(BaseModel):
    """Structured table extracted from the paper."""
    table_id: str = Field(..., description="e.g. 'table_1'")
    caption: str = Field(default="", description="Table caption text")
    page: int = Field(default=0, description="Page where the table appears")
    section_id: Optional[str] = Field(default=None, description="Section this table belongs to")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    rows: List[List[Any]] = Field(default_factory=list, description="Table rows")


class FigureObject(BaseModel):
    """Figure metadata and caption."""
    figure_id: str = Field(..., description="e.g. 'fig_1'")
    page: int = Field(default=0, description="Page where the figure appears")
    section_id: Optional[str] = Field(default=None, description="Section this figure belongs to")
    caption: str = Field(default="", description="Figure caption text")
    image_path: Optional[str] = Field(default=None, description="Path to extracted image file")


class EquationObject(BaseModel):
    """Mathematical equation with LaTeX representation."""
    equation_id: str = Field(..., description="e.g. 'eq_1'")
    page: int = Field(default=0, description="Page where the equation appears")
    section_id: Optional[str] = Field(default=None, description="Section this equation belongs to")
    latex: str = Field(default="", description="LaTeX representation")
    plain_text: str = Field(default="", description="Plain text fallback")
    label: Optional[str] = Field(default=None, description="Equation label e.g. '(1)'")


class ReferenceObject(BaseModel):
    """Structured citation from the references section."""
    ref_id: str = Field(..., description="Citation key e.g. '[1]'")
    authors: List[str] = Field(default_factory=list, description="Author names")
    year: Optional[int] = Field(default=None, description="Publication year")
    title: str = Field(default="", description="Paper/book title")
    venue: str = Field(default="", description="Journal/conference name")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID if available")
    doi: Optional[str] = Field(default=None, description="DOI if available")
    url: Optional[str] = Field(default=None, description="URL if available")


# ---------------------------------------------------------------------------
# Section Node
# ---------------------------------------------------------------------------

class SectionNode(BaseModel):
    """
    Represents a single section or subsection in an academic paper.

    Attributes:
        section_id:         Normalized identifier, e.g. "s1", "s1.1", "s_ref".
        title:              Exact section heading as it appears in the paper.
        level:              Depth in hierarchy — 1 = top-level, 2 = subsection, etc.
        page_start:         First page where this section appears (1-indexed).
        page_end:           Last page where this section appears (1-indexed).
        content:            Cleaned text content of this section.
        content_raw:        Original extracted text before cleaning.
        children:           Nested subsections.
        section_type:       Semantic classification of the section.
        summary:            1–3 sentence extractive summary.
        contains_table:     Whether this section contains a table.
        contains_figure:    Whether this section contains a figure.
        contains_equation:  Whether this section contains an equation.
        citation_count:     Number of in-text citations in this section.
        citations_used:     List of citation keys used in this section.
        reading_time_seconds: Estimated reading time in seconds.
    """

    section_id: str = Field(..., description="Normalized section identifier, e.g. 's1', 's1.1', 's_ref'")
    title: str = Field(..., description="Exact section heading from the paper")
    level: int = Field(..., ge=1, le=5, description="Depth: 1=top, 2=sub, 3=subsub")
    page_start: int = Field(..., ge=0, description="First page of this section (1-indexed)")
    page_end: int = Field(..., ge=0, description="Last page of this section (1-indexed)")
    content: str = Field(default="", description="Cleaned text content of this section")
    content_raw: str = Field(default="", description="Original extracted text before cleaning")
    children: List[SectionNode] = Field(default_factory=list, description="Nested subsections")

    # --- Structural improvements (Section 4) ---
    section_type: str = Field(
        default="other",
        description="Semantic type: abstract, introduction, methodology, etc.",
    )
    summary: str = Field(default="", description="1–3 sentence extractive summary")
    contains_table: bool = Field(default=False, description="Whether section contains a table")
    contains_figure: bool = Field(default=False, description="Whether section contains a figure")
    contains_equation: bool = Field(default=False, description="Whether section contains an equation")
    citation_count: int = Field(default=0, description="Number of in-text citations")
    citations_used: List[str] = Field(default_factory=list, description="List of citation keys e.g. ['[1]', '[3]']")
    reading_time_seconds: int = Field(default=0, description="Estimated reading time in seconds")

    @computed_field  # type: ignore[misc]
    @property
    def word_count(self) -> int:
        """Auto-calculated word count from content (own content only)."""
        return len(self.content.split()) if self.content else 0

    @computed_field  # type: ignore[misc]
    @property
    def total_word_count(self) -> int:
        """Word count including all descendant sections."""
        own = self.word_count
        for child in self.children:
            own += child.total_word_count
        return own


# ---------------------------------------------------------------------------
# Paper Tree
# ---------------------------------------------------------------------------

class PaperTree(BaseModel):
    """
    Full hierarchical tree representation of a single academic paper.

    Attributes:
        paper_id:          Sequential identifier assigned at ingest time (starts at 1).
        title:             Paper title extracted from content (NEVER the filename).
        arxiv_id:          arXiv identifier, if applicable.
        authors:           List of PERSON names only.
        affiliations:      List of institution/organization data.
        emails:            List of email addresses found in the paper.
        file_path:         Local path to the PDF file.
        file_hash:         SHA-256 hex digest of the PDF bytes.
        total_pages:       Total number of pages in the PDF.
        sections:          Top-level section nodes (children nest recursively).
        built_at:          Timestamp when the tree was constructed.
        build_method:      One of "toc_based", "llm_inferred", or "fallback_chunked".
        venue:             Conference/journal name.
        doi:               Digital Object Identifier.
        publication_year:  Year of publication.
        publisher:         Publisher name (ACM, IEEE, Springer, etc.).
        paper_type:        Classification of the paper type.
        keywords:          Extracted keywords.
        ccs_concepts:      ACM CCS Concepts if applicable.
        copyright:         Copyright notice text.
        tables:            Structured tables extracted from the paper.
        figures:           Figure metadata extracted from the paper.
        equations:         Equations extracted from the paper.
        references:        Structured references/citations.
        build_warnings:    Non-fatal issues encountered during parsing.
    """

    paper_id: int = Field(..., ge=1, description="Sequential paper identifier")
    title: str = Field(..., description="Paper title (extracted from content, NEVER filename)")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID if from arXiv")
    authors: List[str] = Field(default_factory=list, description="Author PERSON names only")
    affiliations: List[str] = Field(default_factory=list, description="Institution/organization data")
    emails: List[str] = Field(default_factory=list, description="Email addresses found")
    file_path: str = Field(..., description="Local PDF file path")
    file_hash: str = Field(..., description="SHA-256 hex digest of PDF bytes")
    total_pages: int = Field(..., ge=0, description="Total PDF page count")
    sections: List[SectionNode] = Field(default_factory=list, description="Top-level sections")
    built_at: datetime = Field(default_factory=datetime.utcnow, description="Build timestamp")
    build_method: str = Field(
        ...,
        description="Tree construction method: toc_based | llm_inferred | fallback_chunked",
    )

    # --- Metadata fields (Section 1) ---
    venue: Optional[str] = Field(default=None, description="Conference/journal name")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    publication_year: Optional[int] = Field(default=None, description="Year of publication")
    publisher: Optional[str] = Field(default=None, description="Publisher: ACM, IEEE, Springer, etc.")
    paper_type: str = Field(default="other", description="Paper type classification")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    ccs_concepts: Optional[List[str]] = Field(default=None, description="ACM CCS Concepts")
    copyright: Optional[str] = Field(default=None, description="Copyright notice text")

    # --- Extracted objects (Section 3) ---
    tables: List[TableObject] = Field(default_factory=list, description="Structured tables")
    figures: List[FigureObject] = Field(default_factory=list, description="Figure metadata")
    equations: List[EquationObject] = Field(default_factory=list, description="Equations")
    references: List[ReferenceObject] = Field(default_factory=list, description="Structured references")

    # --- Build quality (Section 4) ---
    build_warnings: List[str] = Field(default_factory=list, description="Non-fatal parsing issues")

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
