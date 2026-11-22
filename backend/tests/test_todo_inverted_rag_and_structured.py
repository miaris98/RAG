"""TDD tests for Branch 4: DoorDash Inverted RAG constrained taxonomy & Ramp rule-grounded structured JSON."""

from fastapi.testclient import TestClient

from main import app
from app.ingestion.pipeline import ingestion_pipeline
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_inverted_rag_constrains_classification_to_allowed_taxonomy_enum():
    """DoorDash Inverted RAG pattern: output label must strictly belong to allowed_labels with ranked scores."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    labels = [
        "payment_ledger_quorum_incident",
        "hsm_key_rotation_security",
        "data_retention_encryption_policy",
        "sovereign_compliance_escalation",
    ]
    resp = client.post(
        "/query-structured",
        json={
            "text": "What causes ERR_LEDGER_9042 quorum drift timeout and how do we resync Raft?",
            "allowed_labels": labels,
            "top_k": 3,
            "user_roles": ["public", "engineering"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification_label"] in labels
    assert data["classification_label"] == "payment_ledger_quorum_incident"
    assert 0.0 < data["confidence"] <= 1.0
    assert set(data["candidate_label_scores"].keys()) == set(labels)
    assert len(data["citations"]) >= 1
    assert data["audit_event_hash"]


def test_ramp_rule_grounded_json_enforces_rbac_and_verifiable_rules():
    """Ramp pattern: returns verifiable rules matched from authorized corpus and blocks unauthorized compliance rules."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    labels = [
        "data_retention_encryption_policy",
        "sovereign_compliance_escalation",
    ]

    # Authorized compliance caller can see sovereign governance escalation rules
    resp_auth = client.post(
        "/query-structured",
        json={
            "text": "Who handles EMEA sovereign regulatory inquiries and what is the PagerDuty rotation?",
            "allowed_labels": labels,
            "top_k": 3,
            "user_roles": ["public", "compliance"],
        },
    )
    assert resp_auth.status_code == 200
    d_auth = resp_auth.json()
    assert d_auth["classification_label"] == "sovereign_compliance_escalation"
    assert len(d_auth["verifiable_rules_matched"]) >= 1
    assert any("PD-EMEA-8821" in rule or "cco-emea" in rule.lower() for rule in d_auth["verifiable_rules_matched"])

    # Unauthorized caller (engineering only) must NOT leak compliance doc citations
    resp_unauth = client.post(
        "/query-structured",
        json={
            "text": "Who handles EMEA sovereign regulatory inquiries and what is the PagerDuty rotation?",
            "allowed_labels": labels,
            "top_k": 3,
            "user_roles": ["engineering"],
        },
    )
    assert resp_unauth.status_code == 200
    d_unauth = resp_unauth.json()
    assert not any("PD-EMEA-8821" in rule for rule in d_unauth["verifiable_rules_matched"])
    assert not any("governance" in c["title"].lower() for c in d_unauth["citations"])


def test_structured_query_rejects_empty_allowed_labels():
    """Pydantic hard-type validation rejects empty allowed_labels enum list with 422."""
    resp = client.post(
        "/query-structured",
        json={
            "text": "Classify this incident.",
            "allowed_labels": [],
        },
    )
    assert resp.status_code == 422
