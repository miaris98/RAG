"""Model Context Protocol (MCP) Tool Definitions for the Staff+ RAG Platform.

Exposes strict, schema-validated, RBAC-scoped RAG tools for external AI agents:
- retrieve_hybrid: Pre-filtered RBAC + Tri-Modal Hybrid Retrieval (Dense + BM25 + Entity Graph + Cross-Encoder)
- ingest_document: Idempotent Parent-Child ingestion with PII & Injection Guardrails
- gdpr_erase_document: GDPR Article 17 surgical erasure across all vector/lexical/graph stores
"""

from typing import Any, Dict, List, Optional
from app.agent.sql_tool import sql_inventory_tool
from app.agent.structured import execute_structured_taxonomy_query
from app.guardrails.audit import audit_logger
from app.ingestion.pipeline import ingestion_pipeline
from app.models import DocumentInput, StructuredQueryRequest
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.semantic_cache import semantic_cache


MCP_TOOL_MANIFEST: List[Dict[str, Any]] = [
    {
        "name": "retrieve_hybrid",
        "description": (
            "Search internal enterprise documents using pre-filtered RBAC, Milvus dense embeddings, "
            "BM25 sparse retrieval, Entity Knowledge Graph, and Cross-Encoder reranking. Read-only."
        ),
        "read_only": True,
        "parameters": ["query", "tenant_id", "user_roles", "top_k"],
    },
    {
        "name": "ingest_document",
        "description": (
            "Chunk, sanitize (PII redaction and prompt-injection defense), deduplicate (SHA-256), "
            "and index a document into the RAG vector and lexical stores."
        ),
        "read_only": False,
        "parameters": ["title", "content", "source", "tenant_id", "access_roles"],
    },
    {
        "name": "gdpr_erase_document",
        "description": (
            "Execute GDPR Article 17 Right-to-be-Forgotten surgical deletion of a document by doc_id "
            "across all retrieval indices and semantic caches, recording an immutable audit event."
        ),
        "read_only": False,
        "parameters": ["doc_id"],
    },
    {
        "name": "query_live_sql_inventory",
        "description": (
            "Run a validated read-only SELECT query against the live operational SQLite service_inventory "
            "and active_incidents database. Mutations are rejected."
        ),
        "read_only": True,
        "parameters": ["sql"],
    },
    {
        "name": "classify_structured_taxonomy",
        "description": (
            "Classify a query into one of the caller-supplied taxonomy enum labels grounded in "
            "RBAC-authorized RAG evidence and return verifiable rules with citations. Read-only."
        ),
        "read_only": True,
        "parameters": ["query", "allowed_labels", "tenant_id", "user_roles", "top_k"],
    },
]


def mcp_retrieve_hybrid(
    query: str,
    tenant_id: str = "default",
    user_roles: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """MCP tool handler for read-only RBAC-filtered hybrid retrieval."""
    roles = user_roles or ["public"]
    chunks = retrieval_engine.retrieve(
        queries=[query],
        mode="hybrid_graph_rerank",
        top_k=top_k,
        tenant_id=tenant_id,
        user_roles=roles,
    )
    audit_logger.log_event(
        event_type="mcp_retrieve_hybrid",
        actor_tenant=tenant_id,
        actor_roles=roles,
        doc_ids=sorted({c.doc_id for c in chunks}),
        chunk_ids=[c.chunk_id for c in chunks],
        summary=f"MCP retrieve_hybrid query='{query[:80]}'",
    )
    return {"count": len(chunks), "chunks": [c.model_dump() for c in chunks]}


def mcp_ingest_document(
    title: str,
    content: str,
    source: str = "mcp_client",
    tenant_id: str = "default",
    access_roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """MCP tool handler for guardrail-protected document ingestion."""
    return ingestion_pipeline.ingest(
        DocumentInput(
            title=title,
            content=content,
            source=source,
            tenant_id=tenant_id,
            access_roles=access_roles or ["public"],
            apply_guardrails=True,
        )
    )


def mcp_gdpr_erase_document(doc_id: str) -> Dict[str, Any]:
    """MCP tool handler for GDPR Article 17 surgical erasure."""
    deleted = retrieval_engine.delete_by_doc_id(doc_id)
    evicted = semantic_cache.invalidate_by_doc_id(doc_id)
    ev = audit_logger.log_event(
        event_type="mcp_gdpr_erasure",
        doc_ids=[doc_id],
        summary=f"MCP GDPR erasure for doc_id={doc_id}",
    )
    return {"status": "erased", "doc_id": doc_id, "chunks_deleted": deleted, "cache_evicted": evicted, "audit_hash": ev["event_hash"]}


def mcp_query_live_sql_inventory(sql: str) -> Dict[str, Any]:
    """MCP tool handler for read-only Agoda OpsBot SQL live inventory queries."""
    res = sql_inventory_tool.execute_readonly_sql(sql)
    audit_logger.log_event(
        event_type="mcp_sql_inventory_query",
        summary=f"MCP read-only SQL query returned {res['row_count']} rows",
        metadata={"sql": sql[:200], "row_count": res["row_count"]},
    )
    return res


def mcp_classify_structured_taxonomy(
    query: str,
    allowed_labels: List[str],
    tenant_id: str = "default",
    user_roles: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """MCP tool handler for DoorDash Inverted RAG constrained taxonomy + Ramp rule extraction."""
    res = execute_structured_taxonomy_query(
        StructuredQueryRequest(
            text=query,
            allowed_labels=allowed_labels,
            tenant_id=tenant_id,
            user_roles=user_roles or ["public"],
            top_k=top_k,
        )
    )
    return res.model_dump()
