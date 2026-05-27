"""
Tree Builder Module  (Core of Phase 1)
=======================================
Builds a hierarchical PageIndex section tree from a PDF.

Implements a comprehensive pipeline for metadata extraction,
section detection, hierarchy resolution, and content extraction.
"""

import hashlib
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter

from multi_paper_rag.phase1.ingestion.pdf_extractor import (
    extract_text_from_pdf, get_toc, get_title_from_font_analysis,
    extract_tables_from_pdf, extract_figures_from_pdf, extract_equations_from_pdf,
    extract_structured_blocks
)
from multi_paper_rag.phase1.llm.nvidia_client import call_nvidia
from multi_paper_rag.phase1.tree.tree_schema import (
    PaperTree, SectionNode, TableObject, FigureObject, EquationObject, ReferenceObject,
    SECTION_TYPES, PAPER_TYPES
)

logger = logging.getLogger("phase1.tree.tree_builder")

def build_tree(
    pdf_path: str,
    paper_id: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> PaperTree:
    """Build a comprehensive hierarchical PageIndex tree from a PDF file."""
    metadata = metadata or {}
    file_hash = _compute_file_hash(pdf_path)
    build_warnings: List[str] = []

    # 1. Extraction (with enhanced font info)
    extraction = extract_text_from_pdf(pdf_path)
    if not extraction["success"]:
        logger.warning(f"PDF extraction issues: {extraction['error']}")
        build_warnings.append(f"PDF extraction issues: {extraction['error']}")

    full_text: str = extraction["full_text"]
    total_pages: int = extraction["total_pages"]
    pdf_meta: Dict = extraction.get("pdf_metadata", {})

    # 2. Extract Objects
    tables = extract_tables_from_pdf(pdf_path)
    figures = extract_figures_from_pdf(pdf_path)
    equations = extract_equations_from_pdf(pdf_path)

    # 3. Build Sections via PageIndex block mapping
    blocks = extract_structured_blocks(pdf_path)
    sections = _build_programmatic_tree(blocks)
    build_method = "pageindex_layout"
    
    extracted_metadata = _extract_metadata_heuristics(full_text, pdf_path, pdf_meta)
    
    # Clean abstract and extract references
    _clean_abstract_content(sections, extracted_metadata)
    refs_objects = _parse_references(sections)

    # 5. Metadata Merging
    final_title = metadata.get("title") or extracted_metadata.get("title") or get_title_from_font_analysis(pdf_path) or pdf_meta.get("title")
    if not final_title or final_title == Path(pdf_path).name:
        final_title = Path(pdf_path).stem.replace("_", " ")
        build_warnings.append("Title extracted from filename fallback — verify manually")

    final_authors = metadata.get("authors") or extracted_metadata.get("authors") or []
    final_arxiv_id = metadata.get("arxiv_id") or extracted_metadata.get("arxiv_id")

    # If author array looks like affiliation, warn
    if any(any(org in a.lower() for org in ["university", "institute", "taiwan", "china", "usa", "department"]) for a in final_authors):
        build_warnings.append("Authors array may contain organization/country names")

    # 6. Post-processing: Node flags, counts, recursive page/word updates
    _link_objects_to_sections(sections, tables, figures, equations)
    _calculate_recursive_metrics(sections)
    _normalize_section_ids(sections)

    # 7. Generate Summaries
    logger.info("Generating LLM summaries for sections...")
    _generate_llm_summaries(sections)

    # 8. Final Validation pass
    _run_post_build_validation(sections, final_title, final_authors, Path(pdf_path).name, build_warnings)

    tree = PaperTree(
        paper_id=paper_id,
        title=final_title,
        arxiv_id=final_arxiv_id,
        authors=final_authors,
        affiliations=extracted_metadata.get("affiliations", []),
        emails=extracted_metadata.get("emails", []),
        file_path=pdf_path,
        file_hash=file_hash,
        total_pages=total_pages,
        sections=sections,
        build_method=build_method,
        venue=extracted_metadata.get("venue"),
        doi=extracted_metadata.get("doi"),
        publication_year=extracted_metadata.get("publication_year"),
        publisher=extracted_metadata.get("publisher"),
        paper_type=extracted_metadata.get("paper_type", "other"),
        keywords=extracted_metadata.get("keywords", []),
        ccs_concepts=extracted_metadata.get("ccs_concepts"),
        copyright=extracted_metadata.get("copyright"),
        tables=[TableObject(**t) for t in tables],
        figures=[FigureObject(**f) for f in figures],
        equations=[EquationObject(**e) for e in equations],
        references=refs_objects,
        build_warnings=build_warnings
    )
    return tree


def _build_programmatic_tree(blocks: List[Dict]) -> List[SectionNode]:
    """Programmatically build tree mapping based on font sizes (PageIndex Workflow)."""
    if not blocks:
        return []
    
    # 1. Calculate body text size (most common size for text blocks > 20 chars)
    sizes = [b["font_size"] for b in blocks if len(b["text"]) > 20]
    if not sizes:
        sizes = [b["font_size"] for b in blocks]
        
    body_size = Counter(sizes).most_common(1)[0][0] if sizes else 10.0
    
    # 2. Filter artifacts
    filtered_blocks = []
    for b in blocks:
        text_clean = b["text"].replace("\n", " ").strip()
        if re.search(r"is available in .* format from", text_clean, re.IGNORECASE):
            continue
        filtered_blocks.append(b)
    blocks = filtered_blocks

    # 3. Identify Headings
    headings = []
    for b in blocks:
        if len(b["text"]) > 200:
            continue
            
        text_clean = b["text"].replace("\n", " ").strip()
        is_heading = False
        
        if "@" in text_clean or text_clean.startswith("http") or text_clean.startswith("arXiv:"):
            continue
            
        is_known_heading = bool(re.match(r'^(\d+\.?)+[ \t]+[A-Z]|Abstract|Introduction|References|Conclusion|Appendix|Methodology|Related Work|Experiments', text_clean, re.IGNORECASE))
        
        # Heading heuristic
        if b["font_size"] > body_size + 0.5:
            # Prevent massive title on page 1 from being a structural heading
            if b["page"] == 1 and b["font_size"] > body_size + 5 and not is_known_heading:
                continue
            is_heading = True
        elif b["font_size"] >= body_size and b["is_bold"] and len(text_clean) < 150:
            if is_known_heading:
                is_heading = True
                
        if is_heading:
            headings.append(b)

    # 3. Determine Heading Levels based on font sizes
    heading_sizes = sorted(list(set([b["font_size"] for b in headings])), reverse=True)
    
    def get_level(font_size):
        for i, size in enumerate(heading_sizes):
            if font_size >= size - 0.5:
                return min(i + 1, 5)
        return min(len(heading_sizes) + 1, 5)

    # 4. Build Tree
    root_nodes = []
    stack = []
    
    # Dummy node for initial metadata
    current_node = SectionNode(
        section_id="s_meta", title="Pre-Section Metadata", level=1,
        page_start=1, page_end=1, content="", children=[]
    )
    
    for b in blocks:
        if b in headings:
            text_clean = b["text"].replace("\n", " ").strip()
            level = get_level(b["font_size"])
            
            # Sub-sections might just be bold body text
            if b["font_size"] <= body_size + 0.5 and b["is_bold"]:
                level = min((stack[-1].level + 1) if stack else 1, 5)

            new_node = SectionNode(
                section_id="", title=text_clean, level=level,
                page_start=b["page"], page_end=b["page"], content="", children=[]
            )
            
            if not stack:
                root_nodes.append(new_node)
            else:
                while stack and stack[-1].level >= level:
                    stack.pop()
                if stack:
                    stack[-1].children.append(new_node)
                else:
                    root_nodes.append(new_node)
            
            stack.append(new_node)
            current_node = new_node
        else:
            current_node.content += b["text"] + "\n\n"
            current_node.page_end = max(current_node.page_end, b["page"])

    # Process node contents (citations, reading time, types)
    flat = []
    def _flatten(nodes):
        for n in nodes:
            flat.append(n)
            _flatten(n.children)
    _flatten(root_nodes)
    
    for node in flat:
        node.content = node.content.strip()
        t_low = node.title.lower()
        if "abstract" in t_low: node.section_type = "abstract"
        elif "introduction" in t_low: node.section_type = "introduction"
        elif "conclusion" in t_low: node.section_type = "conclusion"
        elif "reference" in t_low: node.section_type = "references"
        
        cites = re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", node.content)
        unique_cites = set()
        for c in cites:
            nums = re.findall(r"\d+", c)
            for n in nums: unique_cites.add(f"[{n}]")
        node.citations_used = list(unique_cites)
        node.citation_count = len(unique_cites)
        node.reading_time_seconds = int(len(node.content.split()) / 250 * 60)
        
    # Remove dummy node if it's empty
    if root_nodes and root_nodes[0].title == "Pre-Section Metadata":
        if not root_nodes[0].content.strip():
            root_nodes.pop(0)

    return root_nodes


def _clean_abstract_content(root_nodes: List[SectionNode], meta: Dict):
    """Strip keywords, CCS, and copyright from abstract content."""
    flat = []
    def _flatten(nodes):
        for n in nodes:
            flat.append(n)
            _flatten(n.children)
    _flatten(root_nodes)
    
    for node in flat:
        if node.section_type == "abstract" or "abstract" in node.title.lower():
            text = node.content
            # Remove CCS Concepts block
            ccs_match = re.search(r"(CCS Concepts.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
            if ccs_match:
                if not meta.get("ccs_concepts"): meta["ccs_concepts"] = [ccs_match.group(1).strip()]
                text = text.replace(ccs_match.group(1), "")
            
            # Remove Keywords
            kw_match = re.search(r"(Keywords.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
            if kw_match:
                kws = kw_match.group(1).replace("Keywords", "").strip().split(",")
                if not meta.get("keywords"): meta["keywords"] = [k.strip() for k in kws if k.strip()]
                text = text.replace(kw_match.group(1), "")
                
            # Remove Copyright
            copy_match = re.search(r"(Permission to make digital.*?https://doi\.org/.*?)", text, re.IGNORECASE | re.DOTALL)
            if copy_match:
                if not meta.get("copyright"): meta["copyright"] = copy_match.group(1).strip()
                text = text.replace(copy_match.group(1), "")
                
            node.content = text.strip()


def _parse_references(root_nodes: List[SectionNode]) -> List[ReferenceObject]:
    """Extract structured references from the references section."""
    flat = []
    def _flatten(nodes):
        for n in nodes:
            flat.append(n)
            _flatten(n.children)
    _flatten(root_nodes)
    
    refs = []
    for node in flat:
        if node.section_type == "references" or "reference" in node.title.lower():
            # Simple regex based reference parsing [1] Author. Year. Title.
            ref_blocks = re.split(r"(\[\d+\])", node.content)
            for i in range(1, len(ref_blocks)-1, 2):
                ref_id = ref_blocks[i]
                ref_text = ref_blocks[i+1].strip()
                
                # Basic heuristic extraction
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", ref_text)
                year = int(year_match.group(1)) if year_match else None
                
                arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5})", ref_text, re.IGNORECASE)
                arxiv_id = arxiv_match.group(1) if arxiv_match else None
                
                doi_match = re.search(r"(10\.\d{4,}/\S+)", ref_text)
                doi = doi_match.group(1) if doi_match else None
                
                refs.append(ReferenceObject(
                    ref_id=ref_id.strip(),
                    title=ref_text[:100] + "...", # Simplified for now
                    year=year,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    authors=[]
                ))
    return refs


def _link_objects_to_sections(nodes: List[SectionNode], tables: List[Dict], figures: List[Dict], equations: List[Dict]):
    flat = []
    def _flatten(nodes):
        for n in nodes:
            flat.append(n)
            _flatten(n.children)
    _flatten(nodes)
    
    for node in flat:
        node.contains_table = any(t["page"] >= node.page_start and t["page"] <= node.page_end for t in tables)
        node.contains_figure = any(f["page"] >= node.page_start and f["page"] <= node.page_end for f in figures)
        node.contains_equation = any(e["page"] >= node.page_start and e["page"] <= node.page_end for e in equations)
        
        # Assign IDs to objects
        for t in tables:
            if t["page"] >= node.page_start and t["page"] <= node.page_end and not t.get("section_id"):
                t["section_id"] = node.section_id
        for f in figures:
            if f["page"] >= node.page_start and f["page"] <= node.page_end and not f.get("section_id"):
                f["section_id"] = node.section_id
        for e in equations:
            if e["page"] >= node.page_start and e["page"] <= node.page_end and not e.get("section_id"):
                e["section_id"] = node.section_id


def _calculate_recursive_metrics(nodes: List[SectionNode]):
    """Update parent page_end and ensure recursive aggregation."""
    for node in nodes:
        if node.children:
            _calculate_recursive_metrics(node.children)
            # parent.page_start = min(all_children.page_start, own_content.page_start)
            node.page_start = min([node.page_start] + [c.page_start for c in node.children])
            # parent.page_end = max(all_children.page_end, own_content.page_end)
            node.page_end = max([node.page_end] + [c.page_end for c in node.children])


def _normalize_section_ids(root_nodes: List[SectionNode]):
    """Normalize section IDs to s0, s1, s1.1, s_ref format."""
    def _assign(nodes, prefix_counters):
        for i, node in enumerate(nodes):
            current_counters = prefix_counters.copy()
            
            # Special named sections
            t_low = node.title.lower()
            if "abstract" in t_low: node.section_id = "s0"
            elif "acknowledg" in t_low: node.section_id = "s_ack"
            elif "reference" in t_low: node.section_id = "s_ref"
            elif "appendix" in t_low: 
                app_match = re.search(r"appendix\s+([A-Z])", t_low)
                node.section_id = f"s_app_{app_match.group(1).upper()}" if app_match else "s_app"
            else:
                # Numeric hierarchy
                current_counters.append(str(i+1))
                node.section_id = "s" + ".".join(current_counters)
            
            if node.children:
                # If named section, start a new numeric branch under it (e.g., s_app_A.1)
                child_prefix = [] if not node.section_id.startswith("s") or "_" in node.section_id else current_counters
                # for appendix we might want s_app_A.1
                if "app" in node.section_id: child_prefix = [node.section_id.replace("s_","")]
                _assign(node.children, child_prefix)

    _assign(root_nodes, [])


def _run_post_build_validation(nodes: List[SectionNode], title: str, authors: List[str], filename: str, warnings: List[str]):
    """Execute validation rules."""
    if title == filename:
        warnings.append("Validation Failed: title == filename")
        
    for auth in authors:
        if any(org in auth.lower() for org in ["university", "institute", "country"]):
            warnings.append("Validation Failed: authors array contains organization/country names")
            break
            
    flat = []
    def _flatten(nodes):
        for n in nodes:
            flat.append(n)
            _flatten(n.children)
    _flatten(nodes)
    
    has_abstract = False
    has_refs = False
    for node in flat:
        if node.section_type == "abstract": has_abstract = True
        if node.section_type == "references": has_refs = True
        if node.word_count < 5 and not node.children:
            warnings.append(f"Validation Warning: section {node.section_id} has word_count < 5")
            
    if not has_abstract: warnings.append("Validation Failed: abstract section missing")
    if not has_refs: warnings.append("Validation Failed: references section missing")


def _extract_metadata_heuristics(full_text: str, file_path: str, pdf_meta: Dict) -> Dict[str, Any]:
    """Fallback metadata extraction."""
    meta = {
        "title": pdf_meta.get("title") or Path(file_path).name,
        "authors": [],
        "arxiv_id": None
    }
    arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", full_text[:3000], re.IGNORECASE)
    if arxiv_match: meta["arxiv_id"] = arxiv_match.group(1)
    
    doi_match = re.search(r"(10\.\d{4,}/\S+)", full_text[:3000])
    if doi_match: meta["doi"] = doi_match.group(1)
    return meta

def _compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def _generate_llm_summaries(sections: List[SectionNode]):
    """Generate 1-3 sentence summaries for each section using the LLM."""
    flat = []
    def _flatten(nodes):
        for n in nodes:
            flat.append(n)
            _flatten(n.children)
    _flatten(sections)
    
    for node in flat:
        if node.word_count > 50:
            try:
                # Take first 1000 chars to save tokens
                prompt = f"Summarize the following section of a research paper in 1 to 3 sentences. Focus on the key points. Do not use any introductory phrases.\n\nSection Title: {node.title}\n\nContent:\n{node.content[:1000]}"
                node.summary = call_nvidia("You are an expert academic summarizer.", prompt, temperature=0.1, max_tokens=150).strip()
            except Exception as exc:
                logger.warning(f"Failed to summarize section {node.section_id}: {exc}")
                node.summary = ""
