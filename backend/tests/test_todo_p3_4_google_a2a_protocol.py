"""TDD tests for P3.4: Google A2A Protocol (Agent Card, JSON-RPC 2.0 lifecycle, SSE streaming & Anti-Confused-Deputy RBAC)."""

import json
from fastapi.testclient import TestClient

from main import app
from app.ingestion.pipeline import ingestion_pipeline
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_a2a_agent_card_discovery_endpoint_advertises_skills_and_capabilities():
    """GET /.well-known/agent.json returns a spec-compliant A2A Agent Card with 4 skills & streaming capability."""
    resp = client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"]
    assert card["capabilities"]["streaming"] is True
    assert card["capabilities"]["stateTransitionHistory"] is True
    skill_ids = {s["id"] for s in card["skills"]}
    assert skill_ids == {
        "enterprise_rag_qa",
        "live_sql_ops_audit",
        "taxonomy_rule_classifier",
        "multi_agent_research",
    }


def test_a2a_jsonrpc_tasks_send_get_cancel_and_sse_streaming():
    """POST /a2a handles JSON-RPC 2.0 tasks/send, tasks/get, tasks/cancel, and tasks/sendSubscribe SSE."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    # 1. tasks/send -> executes multi_agent_research skill and returns typed A2A Artifacts
    send_resp = client.post(
        "/a2a",
        headers={"X-Peer-Agent-Id": "peer-ops-agent"},
        json={
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tasks/send",
            "params": {
                "id": "task-a2a-001",
                "metadata": {
                    "skill_id": "multi_agent_research",
                    "tenant_id": "default",
                    "user_roles": ["public", "engineering"],
                },
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Check live service inventory status for AegisPay ERR_LEDGER_9042 and give the remediation command.",
                        }
                    ],
                },
            },
        },
    )
    assert send_resp.status_code == 200
    payload = send_resp.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "req-1"
    task = payload["result"]
    assert task["id"] == "task-a2a-001"
    assert task["status"]["state"] == "completed"
    assert len(task["artifacts"]) >= 1
    combined_text = json.dumps(task["artifacts"])
    assert "ERR_LEDGER_9042" in combined_text
    assert "aegispay-ledger" in combined_text.lower() or "inc-9042" in combined_text.lower()

    # 2. tasks/get -> retrieves persisted task state & history
    get_resp = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "req-2", "method": "tasks/get", "params": {"id": "task-a2a-001"}},
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["result"]["status"]["state"] == "completed"
    assert len(get_resp.json()["result"]["history"]) >= 2

    # 3. tasks/cancel -> marks task canceled cleanly
    cancel_resp = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "req-3", "method": "tasks/cancel", "params": {"id": "task-a2a-001"}},
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["result"]["status"]["state"] == "canceled"

    # 4. tasks/sendSubscribe -> streams SSE status & artifact events
    with client.stream(
        "POST",
        "/a2a",
        headers={"X-Peer-Agent-Id": "peer-ops-agent"},
        json={
            "jsonrpc": "2.0",
            "id": "req-4",
            "method": "tasks/sendSubscribe",
            "params": {
                "id": "task-a2a-sse-002",
                "metadata": {"skill_id": "enterprise_rag_qa", "user_roles": ["public", "engineering"]},
                "message": {"role": "user", "parts": [{"type": "text", "text": "What triggers ERR_LEDGER_9042?"}]},
            },
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers.get("content-type", "")
        raw_stream = "".join(stream_resp.iter_text())
    assert '"state": "working"' in raw_stream
    assert '"state": "completed"' in raw_stream
    assert "ERR_LEDGER_9042" in raw_stream


def test_a2a_anti_confused_deputy_prevents_peer_agent_privilege_escalation():
    """OWASP Agentic Security: peer-support-bot (ceiling={public}) requesting compliance role cannot leak Governance data."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    resp = client.post(
        "/a2a",
        headers={"X-Peer-Agent-Id": "peer-support-bot"},
        json={
            "jsonrpc": "2.0",
            "id": "req-sec-1",
            "method": "tasks/send",
            "params": {
                "id": "task-a2a-sec-001",
                "metadata": {
                    "skill_id": "multi_agent_research",
                    "tenant_id": "default",
                    "user_roles": ["compliance"],  # Escalation attempt above peer-support-bot ceiling!
                },
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "What is the EMEA sovereign compliance PagerDuty rotation ID?"}],
                },
            },
        },
    )
    assert resp.status_code == 200
    task = resp.json()["result"]
    meta = task["metadata"]
    assert "compliance" in meta["confused_deputy_roles_stripped"]
    assert "compliance" not in meta["effective_roles"]
    assert "PD-EMEA-8821" not in json.dumps(task["artifacts"])


def test_all_p0_through_p3_roadmap_todos_completed():
    """Verifies every P0, P1, P2, and P3 checkbox in docs/INDUSTRY_RAG_TODO.md is checked off."""
    from pathlib import Path

    todo_path = Path(__file__).resolve().parents[2] / "docs" / "INDUSTRY_RAG_TODO.md"
    content = todo_path.read_text(encoding="utf-8")
    p0_to_p3 = content.split("### P4", 1)[0]
    assert "🔲 TODO" not in p0_to_p3
    assert "- [ ]" not in p0_to_p3
