"""TDD tests for P2.1: LLM-as-a-Judge, filename/citation hallucination check, and human calibration."""

from fastapi.testclient import TestClient

from main import app
from app.evaluation.judge import rag_judge


client = TestClient(app)


def test_judge_detects_hallucinated_filenames_and_invalid_citations():
    """Dropbox Dash pattern: judge must flag fabricated filenames/identifiers and out-of-range [Doc i] citations."""
    contexts = [
        "[Document: AegisPay RFC-402] ERR_LEDGER_9042 occurs when quorum lease expires. Run `aegis-ctl raft resync`."
    ]
    grounded_answer = (
        "ERR_LEDGER_9042 is triggered when the quorum lease expires; operators must run `aegis-ctl raft resync` [Doc 1]."
    )
    hallucinated_answer = (
        "Run `fake-remediation-script.sh` as documented in `nonexistent_spec_999.md` [Doc 4]."
    )

    good = rag_judge.evaluate_response(
        query="What is the remediation command for ERR_LEDGER_9042?",
        answer=grounded_answer,
        contexts=contexts,
        use_llm_judge=False,
    )
    bad = rag_judge.evaluate_response(
        query="What is the remediation command for ERR_LEDGER_9042?",
        answer=hallucinated_answer,
        contexts=contexts,
        use_llm_judge=False,
    )

    assert good["verdict"] == "pass"
    assert good["faithfulness_score"] >= 0.75
    assert good["citation_grounding_score"] == 1.0
    assert good["hallucinated_references"] == []

    assert bad["verdict"] == "fail"
    assert bad["citation_grounding_score"] < 0.7
    assert any("fake-remediation-script.sh" in ref or "nonexistent_spec_999.md" in ref for ref in bad["hallucinated_references"])
    assert any("Doc 4" in ref for ref in bad["hallucinated_references"])


def test_human_calibration_alignment_exceeds_threshold():
    """Twilio / Booking.com pattern: automated judge must align with human-annotated calibration set."""
    cal = rag_judge.run_human_calibration_check()
    assert cal["num_cases"] >= 4
    assert cal["human_agreement_rate"] >= 0.9
    assert cal["mean_absolute_error"] <= 0.25


def test_evaluate_endpoint_includes_judge_and_calibration_report():
    """POST /evaluate with enable_judge_eval=True returns judge_summary, calibration, and Markdown rubric table."""
    resp = client.post(
        "/evaluate",
        json={
            "top_k": 3,
            "modes": ["hybrid_graph_rerank"],
            "seed_golden_corpus": True,
            "enable_judge_eval": True,
            "use_llm_judge": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "judge_summary" in data
    assert "calibration" in data
    assert data["judge_summary"]["num_evaluated"] >= 1
    assert data["judge_summary"]["mean_faithfulness"] >= 0.5
    assert data["calibration"]["human_agreement_rate"] >= 0.9
    assert "LLM-as-a-Judge" in data["markdown_report"]
    assert "Human Calibration Agreement" in data["markdown_report"]
