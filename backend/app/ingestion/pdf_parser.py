"""#10 NVIDIA-inspired Layout- & Table-Aware PDF Parser (pypdf + pure-Python fallback)."""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List


COL_SPLIT_RE = re.compile(r"\t+|\s*\|\s*|\s{2,}")
PDF_TEXT_OP_RE = re.compile(r"\(([^()\\]*(?:\\.[^()\\]*)*)\)\s*Tj")


def _extract_pages_fallback(pdf_bytes: bytes) -> List[str]:
    """Pure-Python fallback extracting literal PDF text operators when pypdf is not installed."""
    raw = pdf_bytes.decode("latin-1", errors="ignore")
    matches = PDF_TEXT_OP_RE.findall(raw)
    if not matches:
        return []
    cleaned = [m.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\").strip() for m in matches]
    return ["\n".join(line for line in cleaned if line)]


def _reconstruct_layout_and_tables(page_text: str) -> tuple[str, int]:
    """Reconstructs multi-column tabular runs into GitHub-Flavored Markdown tables."""
    lines = [ln.rstrip() for ln in page_text.splitlines()]
    output: List[str] = []
    tables_detected = 0
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        cols = [c.strip() for c in COL_SPLIT_RE.split(line) if c.strip()]
        if len(cols) >= 2:
            # Look ahead to gather consecutive rows with matching column counts
            table_rows: List[List[str]] = [cols]
            look = idx + 1
            while look < len(lines):
                nxt = lines[look].strip()
                if not nxt:
                    break
                nxt_cols = [c.strip() for c in COL_SPLIT_RE.split(nxt) if c.strip()]
                if len(nxt_cols) == len(cols):
                    table_rows.append(nxt_cols)
                    look += 1
                else:
                    break

            if len(table_rows) >= 2:
                tables_detected += 1
                header = "| " + " | ".join(table_rows[0]) + " |"
                sep = "| " + " | ".join(["---"] * len(cols)) + " |"
                output.append(header)
                output.append(sep)
                for row in table_rows[1:]:
                    output.append("| " + " | ".join(row) + " |")
                output.append("")
                idx = look
                continue

        output.append(line)
        idx += 1

    return "\n".join(output).strip(), tables_detected


def extract_structured_markdown_from_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
    """Parses PDF bytes into page-structured Markdown with reconstructed Markdown tables."""
    if not pdf_bytes:
        raise ValueError("Uploaded PDF payload is empty.")

    pages_raw: List[str] = []
    backend = "pypdf"
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            # Use layout extraction mode when available in modern pypdf to preserve table spacing
            try:
                txt = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                txt = page.extract_text() or ""
            if txt.strip():
                pages_raw.append(txt)
    except Exception:
        backend = "pure_python_fallback"
        pages_raw = _extract_pages_fallback(pdf_bytes)

    if not pages_raw:
        pages_raw = _extract_pages_fallback(pdf_bytes)
        backend = "pure_python_fallback"

    if not pages_raw:
        raise ValueError(f"No extractable text found in PDF '{filename}'.")

    md_sections: List[str] = [f"# {filename}"]
    total_tables = 0
    for page_num, raw_text in enumerate(pages_raw, start=1):
        structured_body, t_count = _reconstruct_layout_and_tables(raw_text)
        total_tables += t_count
        md_sections.append(f"## Page {page_num}\n{structured_body}")

    return {
        "filename": filename,
        "page_count": len(pages_raw),
        "tables_detected": total_tables,
        "parser_backend": backend,
        "markdown": "\n\n".join(md_sections),
    }
