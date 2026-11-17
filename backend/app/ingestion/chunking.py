"""Staff-level chunking engine supporting:
1. Parent-Child (Small-to-Big) Chunking with Contextual Headers
2. Contextual Recursive Markdown-Aware Chunking
3. Fixed Recursive Baseline (for ablation comparisons)
4. Deterministic SHA-256 Content-Addressable Hashing for Idempotent Upserts
"""

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


def extract_sections(raw_text: str) -> List[Tuple[str, str]]:
    """Splits Markdown or structured prose into (section_heading, section_body) tuples.

    If no explicit Markdown headers (#, ##, ###) exist, falls back to 'Main Section'.
    """
    lines = raw_text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: List[str] = []

    header_pattern = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
    for line in lines:
        match = header_pattern.match(line)
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


def format_contextual_header(title: str, source: str, section: str) -> str:
    """Builds an Anthropic-style contextual prefix so isolated chunks retain document identity."""
    return f"[Document: {title} | Source: {source} | Section: {section}]\n"


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
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkRecord]:
        """Splits a raw document into deterministic ChunkRecord items."""
        cleaned = content.strip()
        if not cleaned:
            return []

        doc_id = f"doc_{compute_content_hash(title, cleaned)[:16]}"
        base_meta = dict(extra_metadata or {})

        if strategy == "parent_child":
            return self._chunk_parent_child(cleaned, doc_id, title, source, base_meta)
        elif strategy == "contextual_recursive":
            return self._chunk_contextual_recursive(cleaned, doc_id, title, source, base_meta)
        else:
            return self._chunk_fixed(cleaned, doc_id, title, source, base_meta)

    def _chunk_parent_child(
        self,
        text: str,
        doc_id: str,
        title: str,
        source: str,
        base_meta: Dict[str, Any],
    ) -> List[ChunkRecord]:
        """Small-to-Big (Parent-Child) strategy:
        - Indexes fine-grained Child chunks (380 chars) with Contextual Headers for high retrieval precision.
        - Stores the surrounding Parent chunk (1200 chars) on each record so synthesis gets full context.
        """
        records: List[ChunkRecord] = []
        sections = extract_sections(text)
        chunk_idx = 0

        for section_name, section_body in sections:
            parent_texts = self.parent_splitter.split_text(section_body)
            for p_text in parent_texts:
                parent_id = f"par_{compute_content_hash(f'{doc_id}:{section_name}', p_text)[:16]}"
                header = format_contextual_header(title, source, section_name)
                enriched_parent = f"{header}{p_text}"

                child_texts = self.child_splitter.split_text(p_text)
                if not child_texts:
                    child_texts = [p_text]

                for c_text in child_texts:
                    enriched_child = f"{header}{c_text}"
                    chunk_id = f"chk_{compute_content_hash(parent_id, enriched_child)}"
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
                            metadata={**base_meta, "parent_id": parent_id, "section": section_name},
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
    ) -> List[ChunkRecord]:
        """Section-aware recursive chunking with contextual headers."""
        records: List[ChunkRecord] = []
        sections = extract_sections(text)
        chunk_idx = 0

        for section_name, section_body in sections:
            header = format_contextual_header(title, source, section_name)
            splits = self.contextual_splitter.split_text(section_body)
            for split_text in splits:
                enriched = f"{header}{split_text}"
                chunk_id = f"chk_{compute_content_hash(doc_id, enriched)}"
                parent_id = f"par_{chunk_id[4:20]}"
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
                        metadata={**base_meta, "section": section_name},
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
