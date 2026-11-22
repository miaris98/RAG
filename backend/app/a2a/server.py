"""P3.4: Google A2A (Agent-to-Agent) Protocol Server (Agent Card + JSON-RPC 2.0 + SSE + Anti-Confused-Deputy RBAC)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from app.agent.graph import execute_rag_pipeline
from app.agent.sql_tool import sql_inventory_tool
from app.agent.structured import execute_structured_taxonomy_query
from app.agent.subagents import DEFAULT_TAXONOMY_LABELS, supervisor_orchestrator
from app.guardrails.audit import audit_logger
from app.models import OrchestrateRequest, QueryInput, StructuredQueryRequest


# Anti-Confused-Deputy peer ceilings: external peer agents can never exceed their delegated RBAC scope
PEER_AGENT_ROLE_CEILINGS: Dict[str, Set[str]] = {
    "peer-support-bot": {"public"},
    "peer-ops-agent": {"public", "engineering"},
    "peer-compliance-auditor": {"public", "engineering", "compliance"},
    "local-dev-agent": {"public", "engineering", "compliance"},
}


def build_agent_card(base_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
    """Builds the spec-compliant A2A Agent Card served at GET /.well-known/agent.json."""
    clean_base = base_url.rstrip("/")
    return {
        "name": "Staff+ Enterprise RAG & Multi-Agent Supervisor",
        "description": (
            "Production multi-agent RAG server supporting pre-filtered RBAC Tri-Modal GraphRAG, "
            "read-only SQL OpsBot live inventory, DoorDash/Ramp constrained taxonomy classification, "
            "parallel Specialist Subagents, and tamper-evident SHA-256 audit logging."
        ),
        "url": f"{clean_base}/a2a",
        "version": "2.0.0",
        "protocolVersion": "0.2.5",
        "provider": {"organization": "StaffRAG Platform", "url": clean_base},
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "authentication": {
            "schemes": ["DelegatedRBAC"],
            "credentials": "Pass X-Peer-Agent-Id header; requested roles are intersected with peer ceiling.",
        },
        "skills": [
            {
                "id": "enterprise_rag_qa",
                "name": "Enterprise Grounded RAG Q&A",
                "description": "Answers technical & policy queries with Tri-Modal GraphRAG, CRAG gating, Self-RAG verification, and inline [Doc i] citations.",
                "tags": ["rag", "hybrid-search", "citations", "rbac"],
                "examples": ["What triggers ERR_LEDGER_9042 and what is the remediation command?"],
            },
            {
                "id": "live_sql_ops_audit",
                "name": "Live SQL OpsBot Inventory & Incidents",
                "description": "Queries read-only live SQLite service inventory and active incident status tables.",
                "tags": ["sql", "opsbot", "incidents", "read-only"],
                "examples": ["Show live service inventory status and active SEV1 incidents."],
            },
            {
                "id": "taxonomy_rule_classifier",
                "name": "Constrained Taxonomy & Rule-Grounded JSON Classifier",
                "description": "DoorDash Inverted RAG enum classifier + Ramp verifiable rule extractor with RBAC provenance.",
                "tags": ["taxonomy", "structured-json", "compliance"],
                "examples": ["Classify ERR_LEDGER_9042 quorum drift into the allowed incident taxonomy."],
            },
            {
                "id": "multi_agent_research",
                "name": "Parallel Specialist Subagent Supervisor",
                "description": "Fans out in parallel across retrieval, SQL OpsBot, and compliance auditor specialists with hierarchical span waterfalls.",
                "tags": ["subagents", "supervisor", "multi-agent", "audit"],
                "examples": ["Check live AegisPay status, RFC-402 remediation, and verify faithfulness."],
            },
        ],
    }


def resolve_effective_roles(
    peer_agent_id: Optional[str], requested_roles: Optional[List[str]]
) -> Tuple[List[str], List[str]]:
    """Enforces OWASP anti-confused-deputy RBAC delegation by intersecting requested roles with peer ceiling."""
    peer_key = (peer_agent_id or "anonymous-peer").strip().lower()
    ceiling = PEER_AGENT_ROLE_CEILINGS.get(peer_key, {"public"})
    req_list = [r.strip().lower() for r in (requested_roles or ["public"]) if r and r.strip()]
    effective = sorted(r for r in set(req_list) if r in ceiling)
    stripped = sorted(r for r in set(req_list) if r not in ceiling)
    return effective, stripped


def _extract_user_text(params: Dict[str, Any]) -> str:
    msg = params.get("message") or {}
    parts = msg.get("parts") or []
    texts: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            if part.get("type") == "text" and part.get("text"):
                texts.append(str(part["text"]))
            elif "text" in part and part["text"]:
                texts.append(str(part["text"]))
    return "\n".join(texts).strip()


class A2ATaskManager:
    """Manages Google A2A JSON-RPC 2.0 tasks, state transition history, SSE streaming, and RBAC auditing."""

    def __init__(self, max_tasks: int = 500):
        self._lock = threading.RLock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._max_tasks = max_tasks

    def _now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _create_or_update_task(
        self,
        task_id: str,
        state: str,
        message_text: str,
        metadata: Dict[str, Any],
        artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ts = self._now_iso()
        status_obj = {"state": state, "timestamp": ts}
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing is None:
                task = {
                    "id": task_id,
                    "sessionId": metadata.get("session_id") or f"sess_{task_id}",
                    "status": status_obj,
                    "history": [
                        {
                            "role": "user",
                            "parts": [{"type": "text", "text": message_text}],
                            "timestamp": ts,
                        }
                    ],
                    "artifacts": list(artifacts or []),
                    "metadata": dict(metadata),
                }
                if len(self._tasks) >= self._max_tasks:
                    oldest_key = next(iter(self._tasks))
                    self._tasks.pop(oldest_key, None)
                self._tasks[task_id] = task
            else:
                existing["status"] = status_obj
                existing["metadata"].update(metadata)
                if artifacts is not None:
                    existing["artifacts"] = list(artifacts)
                task = existing
            return json.loads(json.dumps(task))

    def _execute_skill(
        self,
        skill_id: str,
        user_text: str,
        tenant_id: str,
        effective_roles: List[str],
        metadata: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        if skill_id == "live_sql_ops_audit":
            snap = sql_inventory_tool.get_live_snapshot()
            md = (sql_inventory_tool.maybe_answer_from_live_sql(f"live status {user_text}") or {}).get("markdown", "")
            return md or json.dumps(snap), {"skill_id": skill_id, "snapshot": snap}

        if skill_id == "taxonomy_rule_classifier":
            labels = metadata.get("allowed_labels") or DEFAULT_TAXONOMY_LABELS
            res = execute_structured_taxonomy_query(
                StructuredQueryRequest(
                    text=user_text,
                    allowed_labels=labels,
                    tenant_id=tenant_id,
                    user_roles=effective_roles,
                )
            )
            return res.rationale, res.model_dump()

        if skill_id == "enterprise_rag_qa":
            out = execute_rag_pipeline(
                QueryInput(
                    text=user_text,
                    tenant_id=tenant_id,
                    user_roles=effective_roles,
                    enable_web_search=False,
                    enable_semantic_cache=False,
                )
            )
            return out.get("answer", ""), {
                "skill_id": skill_id,
                "tools": out.get("tools", []),
                "trace_id": out.get("trace", {}).get("trace_id"),
                "citations": out.get("trace", {}).get("citations", []),
            }

        # Default: parallel Specialist Subagents Supervisor
        orch = supervisor_orchestrator.orchestrate(
            OrchestrateRequest(
                query=user_text,
                tenant_id=tenant_id,
                user_roles=effective_roles,
                enable_web_search=False,
                allowed_labels=metadata.get("allowed_labels"),
            )
        )
        return orch.final_answer, {
            "skill_id": "multi_agent_research",
            "trace_id": orch.trace_id,
            "subagents": [s.agent_id for s in orch.subagent_results],
            "compliance_audit": orch.compliance_audit,
            "citations": [c.model_dump() for c in orch.citations],
            "audit_event_hash": orch.audit_event_hash,
        }

    def run_task(self, params: Dict[str, Any], peer_agent_id: Optional[str]) -> Dict[str, Any]:
        task_id = str(params.get("id") or f"a2a_{uuid.uuid4().hex[:12]}")
        user_text = _extract_user_text(params)
        if not user_text:
            raise ValueError("A2A message must contain at least one non-empty text part.")

        raw_meta = dict(params.get("metadata") or {})
        tenant_id = str(raw_meta.get("tenant_id") or "default")
        skill_id = str(raw_meta.get("skill_id") or "multi_agent_research")
        effective_roles, stripped_roles = resolve_effective_roles(
            peer_agent_id=peer_agent_id,
            requested_roles=raw_meta.get("user_roles"),
        )
        task_meta = {
            **raw_meta,
            "peer_agent_id": peer_agent_id or "anonymous-peer",
            "tenant_id": tenant_id,
            "skill_id": skill_id,
            "effective_roles": effective_roles,
            "confused_deputy_roles_stripped": stripped_roles,
        }

        self._create_or_update_task(task_id, "submitted", user_text, task_meta)
        self._create_or_update_task(task_id, "working", user_text, task_meta)

        try:
            answer_text, data_payload = self._execute_skill(
                skill_id=skill_id,
                user_text=user_text,
                tenant_id=tenant_id,
                effective_roles=effective_roles,
                metadata=task_meta,
            )
            artifacts = [
                {
                    "name": f"{skill_id}_result",
                    "parts": [
                        {"type": "text", "text": answer_text},
                        {"type": "data", "data": data_payload},
                    ],
                }
            ]
            with self._lock:
                self._tasks[task_id]["history"].append(
                    {
                        "role": "agent",
                        "parts": [{"type": "text", "text": answer_text}],
                        "timestamp": self._now_iso(),
                    }
                )
            final_task = self._create_or_update_task(task_id, "completed", user_text, task_meta, artifacts=artifacts)
            audit_logger.log_event(
                event_type="a2a_task_completed",
                actor_tenant=tenant_id,
                actor_roles=effective_roles,
                summary=(
                    f"A2A peer='{task_meta['peer_agent_id']}' skill='{skill_id}' task='{task_id}' "
                    f"(stripped_roles={stripped_roles})"
                ),
                metadata={
                    "task_id": task_id,
                    "peer_agent_id": task_meta["peer_agent_id"],
                    "confused_deputy_roles_stripped": stripped_roles,
                },
            )
            return final_task
        except Exception as exc:
            return self._create_or_update_task(
                task_id,
                "failed",
                user_text,
                {**task_meta, "error": str(exc)},
                artifacts=[],
            )

    def handle_jsonrpc(self, payload: Dict[str, Any], peer_agent_id: Optional[str] = None) -> Dict[str, Any]:
        req_id = payload.get("id")
        method = payload.get("method", "")
        params = payload.get("params") or {}

        if method == "tasks/send":
            try:
                task = self.run_task(params, peer_agent_id=peer_agent_id)
                return {"jsonrpc": "2.0", "id": req_id, "result": task}
            except ValueError as exc:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": str(exc)}}

        if method == "tasks/get":
            task_id = str(params.get("id") or "")
            with self._lock:
                task = self._tasks.get(task_id)
            if not task:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32001, "message": f"Task '{task_id}' not found"}}
            return {"jsonrpc": "2.0", "id": req_id, "result": json.loads(json.dumps(task))}

        if method == "tasks/cancel":
            task_id = str(params.get("id") or "")
            with self._lock:
                task = self._tasks.get(task_id)
                if not task:
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32001, "message": f"Task '{task_id}' not found"}}
                task["status"] = {"state": "canceled", "timestamp": self._now_iso()}
                snapshot = json.loads(json.dumps(task))
            audit_logger.log_event(
                event_type="a2a_task_canceled",
                summary=f"A2A task '{task_id}' canceled",
                metadata={"task_id": task_id},
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": snapshot}

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unsupported A2A method '{method}'"}}

    def stream_task_sse(self, payload: Dict[str, Any], peer_agent_id: Optional[str] = None) -> Iterator[str]:
        """Streams A2A tasks/sendSubscribe status and artifact updates over Server-Sent Events."""
        req_id = payload.get("id")
        params = payload.get("params") or {}
        task_id = str(params.get("id") or f"a2a_{uuid.uuid4().hex[:12]}")
        params = {**params, "id": task_id}

        working_evt = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "id": task_id,
                "status": {"state": "working", "timestamp": self._now_iso()},
                "final": False,
            },
        }
        yield f"data: {json.dumps(working_evt)}\n\n"

        final_task = self.run_task(params, peer_agent_id=peer_agent_id)
        for art in final_task.get("artifacts", []):
            art_evt = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"id": task_id, "artifact": art},
            }
            yield f"data: {json.dumps(art_evt)}\n\n"

        done_evt = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "id": task_id,
                "status": final_task["status"],
                "final": True,
            },
        }
        yield f"data: {json.dumps(done_evt)}\n\n"


a2a_manager = A2ATaskManager()
