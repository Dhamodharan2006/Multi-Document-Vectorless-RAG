# 📘 PaperCompass — Navigate research papers by their own structure, not vector guesswork

---

## 🧠 What Is This Project?

A **Multi-Paper Research Intelligence Assistant** built in two phases:

| Phase | Purpose |
|-------|---------|
| **Phase 1** | Ingest PDFs → Build hierarchical section trees → Answer single-paper questions |
| **Phase 2** | Layer a LangGraph agentic loop on top → Multi-paper sessions, query routing, multi-hop retrieval, cross-paper synthesis |

### 🔑 Why "Vectorless"?

Traditional RAG systems:
1. Chunk text into fixed-size pieces
2. Embed chunks into a vector database
3. Retrieve by cosine similarity

**This project does NOT use vectors or embeddings.** Instead:
- The PDF is parsed into a **semantic section tree** (preserving the paper's own logical structure)
- An **LLM (Groq)** is asked *"which sections are relevant to this question?"* — using the tree's titles and summaries as an index
- The full section text is then passed to the LLM to generate a cited answer

**Why is this better?**
- No chunking artifacts (a section is never split mid-sentence)
- No embedding drift (meaning comes from the LLM's language understanding, not vector proximity)
- Sections are the paper's own organizational units — they naturally contain coherent ideas
- Traceability: every answer cites section title + page range

---

## 🗂️ Project Structure

```
PageIndex/
├── multi_paper_rag/
│   ├── phase1/
│   │   ├── ingestion/
│   │   │   ├── arxiv_fetcher.py       # Download PDFs from arXiv API
│   │   │   └── pdf_extractor.py       # Extract blocks, tables, figures, equations
│   │   ├── tree/
│   │   │   ├── tree_schema.py         # Pydantic models: SectionNode, PaperTree
│   │   │   ├── tree_builder.py        # 8-step PDF → tree pipeline (CORE)
│   │   │   └── tree_storage.py        # JSON serialization + SHA-256 caching
│   │   ├── llm/
│   │   │   ├── groq_client.py         # Groq API wrapper (with 3x retry)
│   │   │   ├── nvidia_client.py       # NVIDIA NIM API wrapper (for summaries)
│   │   │   └── tree_navigator.py      # Section selection + answer generation
│   │   └── config.py                  # API keys, model names, paths
│   └── phase2/
│       ├── agent/
│       │   ├── agent_state.py         # AgentState TypedDict (LangGraph state)
│       │   ├── agent_graph.py         # StateGraph definition + run_agent() API
│       │   ├── nodes/
│       │   │   ├── router_node.py     # Query classifier (Groq)
│       │   │   ├── tool_executor.py   # Tool dispatcher + hop counter
│       │   │   ├── synthesiser.py     # Answer generator (Groq)
│       │   │   └── citation_builder.py # Dedup + format citations
│       │   └── tools/
│       │       ├── query_paper.py         # single query type
│       │       ├── compare_sections.py    # comparison query type
│       │       ├── find_contradictions.py # contradiction multi-hop
│       │       └── get_paper_summary.py   # summary query type
│       ├── session/
│       │   └── session_manager.py     # In-memory session store
│       ├── chainlit_app.py            # Chainlit UI (main entry point)
│       ├── main.py                    # CLI entry point
│       └── config.py                  # Phase 2 settings (re-exports Phase 1)
```

---

## 🔬 Phase 1 — PDF Ingestion & Tree Building

### Overview

```
PDF file
  ↓
extract_structured_blocks()     ← PyMuPDF font-aware block extraction
  ↓
_build_programmatic_tree()      ← Heading detection + hierarchy assembly
  ↓
_clean_abstract_content()       ← Strip metadata noise from sections
  ↓
_parse_references()             ← Structured citation extraction
  ↓
_calculate_recursive_metrics()  ← Page ranges, word counts
  ↓
_normalize_section_ids()        ← s0, s1, s1.1, s_ref IDs
  ↓
_link_objects_to_sections()     ← Attach tables/figures/equations to sections
  ↓
_generate_llm_summaries()       ← NVIDIA NIM → Groq fallback summaries
  ↓
PaperTree (Pydantic model)      ← Saved as JSON with SHA-256 cache key
```

---

## 🌳 Section Tree Building — Step-by-Step Methodology

### Step 1: Block Extraction (`pdf_extractor.py`)

Uses **PyMuPDF (fitz)** to extract layout-aware blocks from the PDF. Each block contains:
```python
{
  "text": "2.1 Methodology",
  "font_size": 11.5,
  "is_bold": True,
  "page": 3
}
```

The extractor also separately extracts:
- **Tables** — using pdfplumber's table detection
- **Figures** — by detecting image bounding boxes
- **Equations** — by identifying LaTeX-like patterns

---

### Step 2: Body Size Calibration

The most common font size across all blocks with >20 characters is computed as the **body text baseline**:

```python
sizes = [b["font_size"] for b in blocks if len(b["text"]) > 20]
body_size = Counter(sizes).most_common(1)[0][0]
```

Everything measured against this baseline to identify headings.

---

### Step 3: Block Un-Fusion

Some PDFs compress multiple headings onto one line (e.g. `"3 Experiments 3.1 Experimental Setup"`). A regex splits these:

```python
match = re.match(r'^(\d+(?:\.\d+)*\s+[A-Za-z\s]+?)\s+(\d+\.\d+\s+[A-Za-z].*)$', text)
```

---

### Step 4: Heading Detection (3 Heuristics)

For each block, three rules determine if it's a section heading:

| Priority | Rule | Condition |
|----------|------|-----------|
| **Primary** | Font size larger than body | `font_size > body_size + 0.5` — skips page 1 title lines |
| **Secondary** | Bold text matching known names | `is_bold AND font_size >= body_size AND regex match` |
| **Tertiary** | Numbered pattern without bold | `re.match(r'^\d+(?:\.\d+)*\.?\s+[A-Z][a-z]', text)` |

Known heading names matched by regex:
`Abstract, Introduction, Background, Related Work, Methodology, Method, Approach, Experiment, Evaluation, Results, Discussion, Conclusion, Limitations, Future Work, Acknowledgments, References, Appendix, ...`

---

### Step 5: Level Assignment

Heading levels (1–5) are assigned by **sorted font size rank**:

```python
heading_sizes = sorted(set(h["font_size"] for h in headings), reverse=True)
# Largest font → Level 1 (top-level sections)
# Next size   → Level 2 (subsections)
# etc.
```

---

### Step 6: Tree Assembly (Stack-based)

A stack-based algorithm builds the nested tree:

```
Stack empty → append to root_nodes
New heading at same/higher level → pop stack until parent found
New heading at deeper level → append as child of stack top
Content blocks → append text to current_node.content
```

```python
for b in blocks:
    if b is heading:
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(new_node)   # nested
        else:
            root_nodes.append(new_node)            # top-level
        stack.append(new_node)
    else:
        current_node.content += b["text"]          # body text
```

**Fallback:** If zero headings are detected → entire document becomes one `"Full Document"` section.

---

### Step 7: Section ID Normalization

Assigns canonical IDs to every node:

| Section | ID Assigned |
|---------|------------|
| Abstract | `s0` |
| Top-level section 1 | `s1` |
| Subsection 1.2 | `s1.2` |
| Sub-subsection 1.2.3 | `s1.2.3` |
| References | `s_ref` |
| Acknowledgments | `s_ack` |
| Appendix A | `s_app_A` |

Page ranges propagate upward recursively (parent spans all children).

---

### Step 8: LLM Summary Generation

For every section with >20 words, the builder calls:
1. **NVIDIA NIM** (`meta/llama-3.3-70b-instruct`) for a 1–3 sentence summary
2. Falls back to **Groq** (`llama-3.3-70b-versatile`) if NVIDIA fails

Parent sections with little own content get a **rollup summary** from their children's summaries.

These summaries are used as the **section index** shown to the LLM during query-time section selection — not the raw text.

---

### Data Model: `SectionNode` & `PaperTree`

```python
class SectionNode(BaseModel):
    section_id: str          # "s1", "s1.2", "s_ref"
    title: str               # "2.1 Experimental Setup"
    level: int               # 1=top, 2=sub, 3=subsub
    page_start: int
    page_end: int
    content: str             # Full cleaned text
    children: List[SectionNode]   # Recursive
    section_type: str        # "abstract" | "methodology" | "references" | ...
    summary: str             # LLM-generated 1-3 sentence summary
    contains_table: bool
    contains_figure: bool
    contains_equation: bool
    citation_count: int
    citations_used: List[str]    # ["[1]", "[4]", "[12]"]
    reading_time_seconds: int

class PaperTree(BaseModel):
    paper_id: int
    title: str
    authors: List[str]
    file_hash: str           # SHA-256 — used for caching
    total_pages: int
    sections: List[SectionNode]    # Top-level sections
    build_method: str        # "pageindex_layout"
    tables: List[TableObject]
    figures: List[FigureObject]
    equations: List[EquationObject]
    references: List[ReferenceObject]
    build_warnings: List[str]
```

---

## 🔎 Phase 1 Query Pipeline (`tree_navigator.py`)

When a question is asked about a single paper:

### Step 1 — Section Selection (LLM as Index)

The section index is built from **titles + summaries** (NOT full text):

```
Section Index sent to Groq:
[
  {"id": "0",   "title": "Abstract",     "summary": "We propose..."},
  {"id": "1",   "title": "Introduction", "summary": "Recent work on..."},
  {"id": "2.1", "title": "Methodology",  "summary": "Our approach uses..."},
  ...
]
```

Groq responds with: `["2.1", "3", "4.2"]`

Groq is configured with `temperature=0.0` for deterministic section selection.

### Step 2 — Answer Generation

The full content of the selected sections is passed to Groq with strict citation instructions:

```
For every claim → cite as: *[Section Title | pp. X-Y]*
```

---

## ⚡ Phase 2 — LangGraph Agentic Pipeline

### Agent State (`agent_state.py`)

All data flowing between nodes is typed as `AgentState`:

```python
class AgentState(TypedDict):
    session_id: str
    original_query: str
    query_type: str              # "single" | "comparison" | "contradiction" | "summary"
    target_paper_ids: List[int]
    target_section: str
    retrieved_sections: List[RetrievedSection]
    hop_count: int
    max_hops: int                # Default: 5
    final_answer: str
    citations: List[Dict]
    error: Optional[str]
    session_index: List[Dict]    # Lightweight paper index for routing
```

---

### Graph Topology

```
START
  │
  ▼
┌─────────────┐
│ router_node │  ← Groq classifies query type + picks papers
└──────┬──────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│ tool_executor    │────▶│ tool_executor (loop) │  ← contradiction multi-hop
│ _node            │     └─────────────────────┘
└──────┬───────────┘
       │  _should_continue()
       ▼
┌──────────────┐
│ synthesiser  │  ← Groq generates cited answer
│ _node        │
└──────┬───────┘
       │
       ▼
┌───────────────────┐
│ citation_builder  │  ← Deduplicates, structures citations
│ _node             │
└──────┬────────────┘
       │
      END
```

---

### Node 1: Router Node (`router_node.py`)

**Purpose:** Classify the query intent and set routing parameters.

**LLM call:** Groq with `temperature=0.1`, `max_tokens=300`

**System prompt:** Instructs Groq to return JSON:
```json
{
  "query_type": "comparison",
  "target_paper_ids": [1, 2, 3],
  "target_section": "Methodology"
}
```

**Query types:**
| Type | Meaning |
|------|---------|
| `single` | Question about ONE paper |
| `comparison` | Comparing same aspect across papers |
| `contradiction` | Finding conflicting claims |
| `summary` | Summarising one or more papers |

**Fallback:** If Groq fails → `query_type="single"`, all papers targeted.

---

### Node 2: Tool Executor Node (`tool_executor.py`)

**Purpose:** Dispatch to the correct tool based on `query_type`. Track hops.

**Tool map:**
```python
tool_map = {
    "single":        query_paper_tool,
    "comparison":    compare_sections_tool,
    "contradiction": find_contradictions_tool,   # ← loops up to 3 times
    "summary":       get_paper_summary_tool,
}
```

**Hop guard:** If `hop_count >= max_hops`, returns error and exits.

**Conditional edge** (`_should_continue`):
- `contradiction` AND `hop_count < 3` → **loop back** to tool_executor
- All other types OR max hops reached → **proceed** to synthesiser

---

### Node 3: Synthesiser Node (`synthesiser.py`)

**Purpose:** Generate the final cited answer from all retrieved sections.

Uses **Groq** with `temperature=0.2` (slightly creative for synthesis).

Context format sent to Groq:
```
[Paper 1: Attention Is All You Need | Methodology | Pages 3-5]
<full section text>

---

[Paper 2: BERT | Methodology | Pages 2-4]
<full section text>
```

Rules enforced via system prompt:
- Every claim must cite `[Paper N, Section Name, p.X-Y]`
- Never add external knowledge
- Use markdown: headings, bullets, bold key terms

---

### Node 4: Citation Builder Node (`citation_builder.py`)

**Purpose:** Post-process the retrieved sections into a structured, deduplicated citation list.

Deduplication key: `(paper_id, section_id)`

Output per citation:
```python
{
    "paper_id": 1,
    "paper_title": "Attention Is All You Need",
    "section_title": "Multi-Head Attention",
    "page_start": 4,
    "page_end": 5
}
```

---

## 🔄 Multi-Hop Contradiction Detection (3 Hops)

The `find_contradictions_tool` is the only tool that loops. Here's what happens across 3 hops:

```
Hop 0: _hop0_extract_claims()
  → Calls find_relevant_sections() on EVERY target paper
  → Collects top 3 sections per paper
  → Appends all to retrieved_sections

Hop 1: _hop1_detect_contradictions()
  → Sends all retrieved sections to Groq
  → System prompt: "Identify contradictions. For each: state Claim A, Claim B, genuine vs framing difference"
  → Stores analysis as a SYNTHETIC section (paper_id=0, title="Cross-Paper Analysis")
  → Appended to retrieved_sections

Hop 2: _hop2_verify_context()
  → Fetches ADDITIONAL sections from each paper (methodology + results focused)
  → Skips sections already retrieved in Hop 0
  → Provides broader context for verification

Final synthesis: Groq synthesises the hop-0 sections + hop-1 analysis + hop-2 context
```

---

## 💾 Session Management (`session_manager.py`)

In-memory store for multi-paper sessions:

```python
Session {
    session_id: UUID
    papers: {1: PaperTree, 2: PaperTree, ...}   # keyed by session-local paper_id
    history: [{query, answer, timestamp}, ...]
}
```

**Key methods:**
| Method | Purpose |
|--------|---------|
| `create_session()` | Returns new UUID |
| `add_paper(session_id, tree)` | Returns 1-based paper_id. Raises ValueError if >5 papers |
| `get_paper(session_id, paper_id)` | Returns PaperTree or None |
| `get_session_index(session_id)` | Lightweight index for router: title + section names + page count |
| `add_to_history()` | Stores Q&A pair with UTC timestamp |
| `get_history()` | Returns full conversation list |

---

## 🖥️ Chainlit Interface (`chainlit_app.py`)

### Commands Available

| Command | Handler | What It Does |
|---------|---------|-------------|
| PDF Upload (paperclip) | `_handle_uploaded_file()` | Copy → build tree → add to session |
| `/add arxiv:1706.03762` | `_handle_add()` | Download from arXiv → build → add |
| `/add path.pdf` | `_handle_add()` | Local PDF → build → add |
| `/papers` | `_handle_papers()` | List all papers in session |
| `/history` | `_handle_history()` | Show past Q&A in session |
| `/help` | `_handle_help()` | Full usage reference |
| `/clear` | inline | Create new session |
| Any text | `_handle_question()` | Run full agentic pipeline |

### Query Progress Display

When a question is typed, a **live progress indicator** advances every 2.5 seconds while the agent runs in a background thread:

```
🧭 Step 1/4 — Routing query...
   > Classifying intent and identifying target papers

🔍 Step 2/4 — Retrieving sections...
   > Scanning paper trees for relevant content

🔄 Step 3/4 — Analysing content...
   > Cross-referencing sections across papers

📝 Step 4/4 — Synthesising answer...
   > Generating cited response from retrieved context
```

After completion, **3 `cl.Step` trace panels** show what actually happened:
- 🧭 Router Node → query type + tool selected
- ⚙️ Tool Executor → hops used + sections retrieved
- 📝 Synthesiser → answer length + citations

---

## 🔗 LLM Usage Summary

| LLM | Model | Used For | Temperature |
|-----|-------|---------|------------|
| **NVIDIA NIM** | `meta/llama-3.3-70b-instruct` | Section summaries during tree build | 0.1 |
| **Groq** | `llama-3.3-70b-versatile` | Section selection, answer generation, routing, contradiction detection | 0.0–0.2 |

Both clients implement **3x retry with 2s backoff**. NVIDIA is primary for summaries; Groq is the universal fallback and the primary inference engine at query time.

---

## 📦 Caching Strategy

Trees are cached to avoid re-processing the same PDF:

```python
# SHA-256 of PDF bytes → cache key
file_hash = sha256(pdf_bytes)
cached = load_tree(file_hash)   # Returns PaperTree from JSON, or None
if cached:
    return cached               # No re-build needed
```

Cache stored as JSON in `TREE_STORAGE_DIR`. Structure:
```
trees/
  <file_hash>.json    ← full PaperTree JSON
  index.json          ← list of all cached trees (for /papers command)
```

---

## 🔁 End-to-End Data Flow

```
User uploads PDF
      ↓
PyMuPDF extracts layout blocks (font size, bold, page)
      ↓
Body font size calibrated → headings identified (3 heuristics)
      ↓
Stack-based tree assembly → SectionNode hierarchy
      ↓
IDs normalized (s0, s1, s1.1, s_ref…)
      ↓
NVIDIA NIM generates 1-3 sentence summaries per section (Groq fallback)
      ↓
PaperTree saved as JSON (SHA-256 keyed cache)
      ↓
SessionManager.add_paper() → paper gets session-local ID

User types question
      ↓
LangGraph StateGraph invoked
      ↓
  [Router Node]
  Groq sees: session_index + query
  Returns: query_type, target_paper_ids, target_section
      ↓
  [Tool Executor Node]
  Calls correct tool (query_paper / compare / find_contradictions / summary)
  Tool calls find_relevant_sections() → Groq selects sections from index
  Returns: RetrievedSection list, hop_count++
      ↓
  _should_continue(): loop if contradiction & hop<3, else proceed
      ↓
  [Synthesiser Node]
  Groq sees: full section texts + citation rules
  Returns: markdown answer with inline citations
      ↓
  [Citation Builder Node]
  Deduplicates (paper_id, section_id) pairs
  Returns: structured citation list
      ↓
Chainlit renders:
  - Live progress steps (Router → Tool → Synthesiser)
  - Final answer (markdown)
  - Sources Used (paper / section / pages)
  - Footer: hops used + query type
```
