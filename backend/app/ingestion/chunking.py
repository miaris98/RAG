"""Staff-level chunking engine supporting:
1. Parent-Child (Small-to-Big) Chunking with Contextual Headers
2. Contextual Recursive Markdown-Aware Chunking
3. Fixed Recursive Baseline (for ablation comparisons)
4. Deterministic SHA-256 Content-Addressable Hashing for Idempotent Upserts
"""

import ast
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:

    class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
        """Pure-Python drop-in fallback for LangChain's RecursiveCharacterTextSplitter."""

        def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 150,
            separators: Optional[List[str]] = None,
        ):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        def split_text(self, text: str) -> List[str]:
            text = text.strip()
            if not text:
                return []
            if len(text) <= self.chunk_size:
                return [text]
            chunks: List[str] = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            start = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                if end < len(text):
                    # Try breaking on the highest-priority separator near the end window
                    window = text[start:end]
                    for sep in self.separators:
                        if sep and sep in window:
                            last_pos = window.rfind(sep)
                            if last_pos > self.chunk_size // 3:
                                end = start + last_pos + len(sep)
                                break
                piece = text[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(text):
                    break
                start = max(start + 1, end - self.chunk_overlap)
            return chunks

from app.config import settings
from app.models import ChunkRecord, ChunkingStrategy


def compute_content_hash(prefix: str, text: str) -> str:
    """Computes a deterministic 24-char SHA-256 hex digest for idempotent deduplication."""
    normalized = re.sub(r"\s+", " ", text.strip())
    payload = f"{prefix}::{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


FENCED_CODE_BLOCK_RE = re.compile(r"(```[a-zA-Z0-9_+-]*\r?\n[\s\S]*?```)", re.MULTILINE)


def extract_sections(raw_text: str) -> List[Tuple[str, str]]:
    """Splits Markdown or structured prose into (section_heading, section_body) tuples.

    Ignores '#' lines inside fenced code blocks (```...```) so Python/Shell comments never split sections.
    """
    lines = raw_text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: List[str] = []
    in_code_fence = False

    header_pattern = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        match = header_pattern.match(line) if not in_code_fence else None
        if match:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, body))
            current_heading = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    return sections if sections else [("Overview", raw_text.strip())]


def extract_code_symbols(text: str) -> Dict[str, List[str]]:
    """Extracts programming languages and code symbols (AST for Python, regex for polyglot fences)."""
    languages: List[str] = []
    symbols: List[str] = []
    seen_langs: set[str] = set()
    seen_syms: set[str] = set()

    for block in FENCED_CODE_BLOCK_RE.findall(text):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        lang = lines[0].strip().lstrip("`").strip().lower() or "plain"
        if lang not in seen_langs:
            seen_langs.add(lang)
            languages.append(lang)
        body = "\n".join(lines[1:-1])

        extracted: List[str] = []
        if lang in ("python", "py"):
            try:
                tree = ast.parse(body)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        extracted.append(node.name)
            except SyntaxError:
                pass

        if not extracted:
            patterns = [
                r"\b(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)",
                r"\b(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"\b(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                r"^\s*(?:public|private|protected)?\s*(?:async\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{",
                r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_.]*)",
            ]
            for pat in patterns:
                for match in re.finditer(pat, body, re.MULTILINE | re.IGNORECASE):
                    cand = match.group(1)
                    if cand.lower() not in {"if", "for", "while", "switch", "catch", "return"}:
                        extracted.append(cand)

        for sym in extracted:
            if sym not in seen_syms:
                seen_syms.add(sym)
                symbols.append(sym)

    return {"languages": languages, "symbols": symbols}


def generate_contextual_prefix(
    full_doc_text: str,
    title: str,
    section: str,
    code_symbols: Optional[List[str]] = None,
) -> str:
    """Anthropic-style situational context prefix grounding an isolated chunk in its parent document."""
    # Extract the first informative non-heading sentence from the document as global thesis
    thesis = ""
    for line in full_doc_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or cleaned.startswith("```") or cleaned.startswith("|"):
            continue
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
        if len(first_sentence) >= 20:
            thesis = first_sentence[:200]
            break

    parts = [f"From document '{title}' (section '{section}')"]
    if thesis:
        parts.append(f"Document summary: {thesis}")
    if code_symbols:
        parts.append(f"Defines code symbols: {', '.join(code_symbols[:8])}")
    return f"[Context: {' | '.join(parts)}]\n"


def format_contextual_header(
    title: str,
    source: str,
    section: str,
    code_symbols: Optional[List[str]] = None,
) -> str:
    """Builds an Anthropic-style contextual prefix so isolated chunks retain document identity."""
    sym_suffix = f" | Symbols: {', '.join(code_symbols[:8])}" if code_symbols else ""
    return f"[Document: {title} | Source: {source} | Section: {section}{sym_suffix}]\n"


class ChunkingEngine:
    """Produces ChunkRecord objects using configurable Staff RAG chunking strategies."""

    def __init__(
        self,
        parent_chunk_size: int = settings.parent_chunk_size,
        parent_chunk_overlap: int = settings.parent_chunk_overlap,
        child_chunk_size: int = settings.child_chunk_size,
        child_chunk_overlap: int = settings.child_chunk_overlap,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        self.contextual_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

    def chunk_document(
        self,
        content: str,
        title: str = "Untitled Document",
        source: str = "manual_input",
        strategy: ChunkingStrategy = "parent_child",
        tenant_id: str = "default",
        access_roles: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        enable_contextual_enrichment: bool = False,
    ) -> List[ChunkRecord]:
        """Splits a raw document into deterministic ChunkRecord items."""
        cleaned = content.strip()
        if not cleaned:
            return []

        doc_id = f"doc_{compute_content_hash(f'{tenant_id}:{title}', cleaned)[:16]}"
        roles = [r.strip().lower() for r in (access_roles or ["public"]) if r and r.strip()] or ["public"]
        base_meta = dict(extra_metadata or {})

        if strategy == "parent_child":
            records = self._chunk_parent_child(
                cleaned,
                doc_id,
                title,
                source,
                base_meta,
                enable_contextual_enrichment=enable_contextual_enrichment,
            )
        elif strategy == "contextual_recursive":
            records = self._chunk_contextual_recursive(
                cleaned,
                doc_id,
                title,
                source,
                base_meta,
                enable_contextual_enrichment=enable_contextual_enrichment,
            )
        else:
            records = self._chunk_fixed(cleaned, doc_id, title, source, base_meta)

        for rec in records:
            rec.tenant_id = tenant_id
            rec.access_roles = roles
        return records

    def _split_preserving_structure(
        self, text: str, splitter: RecursiveCharacterTextSplitter
    ) -> List[str]:
        """Keeps fenced code blocks atomic while preserving Markdown table headers across prose/table splits."""
        parts = FENCED_CODE_BLOCK_RE.split(text)
        if len(parts) == 1:
            return self._split_preserving_tables(text, splitter)

        chunks: List[str] = []
        for segment in parts:
            stripped = segment.strip()
            if not stripped:
                continue
            if stripped.startswith("```") and stripped.endswith("```"):
                # Keep the entire fenced code block atomic so AST/function context is never severed
                chunks.append(stripped)
            else:
                chunks.extend(self._split_preserving_tables(stripped, splitter))
        return chunks

    def _split_preserving_tables(self, text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
        """Splits text while re-injecting Markdown table headers if a table spans multiple chunks."""
        lines = text.splitlines()
        # Check if text contains a Markdown table header + separator (| --- |)
        table_header = None
        for i in range(len(lines) - 1):
            if "|" in lines[i] and re.match(r"^\s*\|?\s*:?-{2,}:?\s*\|", lines[i + 1]):
                table_header = f"{lines[i]}\n{lines[i + 1]}"
                break

        raw_splits = splitter.split_text(text)
        if not table_header or len(raw_splits) <= 1:
            return raw_splits

        enriched: List[str] = []
        for idx, chunk in enumerate(raw_splits):
            if idx > 0 and "|" in chunk and table_header.splitlines()[0] not in chunk:
                enriched.append(f"{table_header}\n{chunk}")
            else:
                enriched.append(chunk)
        return enriched

    def _chunk_parent_child(
        self,
        text: str,
        doc_id: str,
        title: str,
        source: str,
        base_meta: Dict[str, Any],
        enable_contextual_enrichment: bool = False,
    ) -> List[ChunkRecord]:
        """Small-to-Big (Parent-Child) strategy with Table-Header, Fenced-Code & Optional Contextual Prefix:
        - Indexes fine-grained Child chunks with Contextual Headers for high retrieval precision.
        - Stores the surrounding Parent chunk on each record so synthesis gets full context.
        """
        records: List[ChunkRecord] = []
        sections = extract_sections(text)
        chunk_idx = 0

        for section_name, section_body in sections:
            parent_texts = self._split_preserving_structure(section_body, self.parent_splitter)
            for p_text in parent_texts:
                parent_code_info = extract_code_symbols(p_text)
                parent_id = f"par_{compute_content_hash(f'{doc_id}:{section_name}', p_text)[:16]}"
                parent_header = format_contextual_header(
                    title, source, section_name, code_symbols=parent_code_info["symbols"]
                )
                parent_ctx = (
                    generate_contextual_prefix(text, title, section_name, parent_code_info["symbols"])
                    if enable_contextual_enrichment
                    else ""
                )
                enriched_parent = f"{parent_header}{parent_ctx}{p_text}"

                child_texts = self._split_preserving_structure(p_text, self.child_splitter)
                if not child_texts:
                    child_texts = [p_text]

                for c_text in child_texts:
                    child_code_info = extract_code_symbols(c_text)
                    child_symbols = child_code_info["symbols"] or parent_code_info["symbols"]
                    child_langs = child_code_info["languages"] or parent_code_info["languages"]
                    header = format_contextual_header(title, source, section_name, code_symbols=child_symbols)
                    ctx_prefix = (
                        generate_contextual_prefix(text, title, section_name, child_symbols)
                        if enable_contextual_enrichment
                        else ""
                    )
                    enriched_child = f"{header}{ctx_prefix}{c_text}"
                    chunk_id = f"chk_{compute_content_hash(parent_id, enriched_child)}"
                    meta = {
                        **base_meta,
                        "parent_id": parent_id,
                        "section": section_name,
                        "contextual_enriched": enable_contextual_enrichment,
                    }
                    if child_symbols:
                        meta["code_symbols"] = child_symbols
                    if child_langs:
                        meta["code_languages"] = child_langs
                    records.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            parent_id=parent_id,
                            title=title,
                            source=source,
                            section=section_name,
                            content=enriched_child,
                            parent_content=enriched_parent,
                            chunk_index=chunk_idx,
                            token_estimate=max(1, len(enriched_child.split())),
                            chunking_strategy="parent_child",
                            metadata=meta,
                        )
                    )
                    chunk_idx += 1

        return records

    def _chunk_contextual_recursive(
        self,
        text: str,
        doc_id: str,
        title: str,
        source: str,
        base_meta: Dict[str, Any],
        enable_contextual_enrichment: bool = False,
    ) -> List[ChunkRecord]:
        """Section-aware recursive chunking with code-block preservation and contextual headers."""
        records: List[ChunkRecord] = []
        sections = extract_sections(text)
        chunk_idx = 0

        for section_name, section_body in sections:
            splits = self._split_preserving_structure(section_body, self.contextual_splitter)
            for split_text in splits:
                code_info = extract_code_symbols(split_text)
                header = format_contextual_header(
                    title, source, section_name, code_symbols=code_info["symbols"]
                )
                ctx_prefix = (
                    generate_contextual_prefix(text, title, section_name, code_info["symbols"])
                    if enable_contextual_enrichment
                    else ""
                )
                enriched = f"{header}{ctx_prefix}{split_text}"
                chunk_id = f"chk_{compute_content_hash(doc_id, enriched)}"
                parent_id = f"par_{chunk_id[4:20]}"
                meta = {
                    **base_meta,
                    "section": section_name,
                    "contextual_enriched": enable_contextual_enrichment,
                }
                if code_info["symbols"]:
                    meta["code_symbols"] = code_info["symbols"]
                if code_info["languages"]:
                    meta["code_languages"] = code_info["languages"]
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        parent_id=parent_id,
                        title=title,
                        source=source,
                        section=section_name,
                        content=enriched,
                        parent_content=enriched,
                        chunk_index=chunk_idx,
                        token_estimate=max(1, len(enriched.split())),
                        chunking_strategy="contextual_recursive",
                        metadata=meta,
                    )
                )
                chunk_idx += 1

        return records

    def _chunk_fixed(
        self,
        text: str,
        doc_id: str,
        title: str,
        source: str,
        base_meta: Dict[str, Any],
    ) -> List[ChunkRecord]:
        """Baseline 1000-char / 150-overlap chunker matching the legacy implementation."""
        records: List[ChunkRecord] = []
        splits = self.fixed_splitter.split_text(text)
        for idx, split_text in enumerate(splits):
            chunk_id = f"chk_{compute_content_hash(doc_id, split_text)}"
            parent_id = f"par_{chunk_id[4:20]}"
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    parent_id=parent_id,
                    title=title,
                    source=source,
                    section="FixedChunk",
                    content=split_text,
                    parent_content=split_text,
                    chunk_index=idx,
                    token_estimate=max(1, len(split_text.split())),
                    chunking_strategy="fixed",
                    metadata=dict(base_meta),
                )
            )
        return records
