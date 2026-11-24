"""P2.1: LLM-as-a-Judge & Human-Calibrated RAG Evaluator (Booking.com / Dropbox Dash / Twilio / NVIDIA pattern)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import requests

from app.config import settings
from app.observability.metrics import CircuitBreaker
from app.retrieval.reranker import tokenize_content_words


judge_llm_breaker = CircuitBreaker(name="ollama_judge", failure_threshold=2, recovery_timeout_sec=60.0)


DOC_REF_RE = re.compile(r"\[Doc\s+(\d+)(?::[^\]]*)?\]", re.IGNORECASE)
BACKTICK_ARTIFACT_RE = re.compile(r"`([^`\n]{3,80})`")
FILENAME_OR_CODE_RE = re.compile(
    r"(?:\.[a-z0-9]{1,5}$|_|/|\\|--|^ERR_|^PD-)", re.IGNORECASE
)


HUMAN_CALIBRATION_SET: List[Dict[str, Any]] = [
    {
        "id": "CAL_1_grounded_exact",
        "query": "What triggers ERR_LEDGER_9042 and what is the remediation command?",
        "contexts": [
            "ERR_LEDGER_9042 (Quorum Drift Timeout): Triggered when the Milvus/Etcd coordination lease expires or clock skew exceeds 45ms. Remediation: Execute `aegis-ctl raft resync --force-quorum`."
        ],
        "answer": (
            "ERR_LEDGER_9042 is triggered when the Milvus/Etcd coordination lease expires or clock skew exceeds 45ms. "
            "Operators should execute `aegis-ctl raft resync --force-quorum` [Doc 1]."
        ),
        "expected_human_verdict": "pass",
        "expected_human_score": 0.95,
    },
    {
        "id": "CAL_2_grounded_policy",
        "query": "Which encryption cipher and MinIO bucket store transaction audit embeddings?",
        "contexts": [
            "All transaction embeddings and audit payloads are encrypted with AES-256-GCM inside MinIO object storage bucket `aegis-audit-prod` for 90 days."
        ],
        "answer": (
            "Transaction audit embeddings are encrypted using AES-256-GCM and stored in MinIO bucket `aegis-audit-prod` for 90 days [Doc 1]."
        ),
        "expected_human_verdict": "pass",
        "expected_human_score": 0.95,
    },
    {
        "id": "CAL_3_hallucinated_filename_and_doc_ref",
        "query": "What triggers ERR_LEDGER_9042 and what is the remediation command?",
        "contexts": [
            "ERR_LEDGER_9042 occurs when quorum lease expires. Execute `aegis-ctl raft resync`."
        ],
        "answer": (
            "Run `deploy_hotfix_v9.sh` according to `secret_runbook_404.md` to clear the cache [Doc 5]."
        ),
        "expected_human_verdict": "fail",
        "expected_human_score": 0.15,
    },
    {
        "id": "CAL_4_ungrounded_fabrication",
        "query": "What is the p95 retrieval latency SLO?",
        "contexts": [
            "Retrieval p95 Latency must remain under 85ms for hybrid search and reranking on CPU."
        ],
        "answer": (
            "Quantum satellites orbit Mars every 42 hours using helium propulsion engines without latency constraints."
        ),
        "expected_human_verdict": "fail",
        "expected_human_score": 0.10,
    },
]


class RAGJudgeEvaluator:
    """Evaluates answer faithfulness, query relevance, and citation/filename hallucination with human calibration."""

    def evaluate_response(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        use_llm_judge: bool = False,
    ) -> Dict[str, Any]:
        """Scores a RAG answer via deterministic rubric + optional Ollama LLM-as-a-Judge refinement."""
        clean_answer = self._strip_citation_appendix(answer or "")
        joined_context = "\n".join(contexts or [])
        context_lower = joined_context.lower()
        context_tokens = set(tokenize_content_words(joined_context))

        faithfulness = self._score_faithfulness(clean_answer, context_tokens)
        relevance = self._score_relevance(query, clean_answer)
        hallucinated_refs, citation_score = self._audit_citations_and_filenames(
            clean_answer, contexts or [], context_lower
        )

        judge_backend = "deterministic_rubric"
        if use_llm_judge and contexts and clean_answer:
            llm_scores = self._try_ollama_judge(query, clean_answer, joined_context)
            if llm_scores is not None:
                faithfulness = round(0.5 * faithfulness + 0.5 * llm_scores["faithfulness"], 4)
                relevance = round(0.5 * relevance + 0.5 * llm_scores["relevance"], 4)
                judge_backend = f"ollama_hybrid_judge ({settings.llm_model})"

        overall = round(
            0.45 * faithfulness + 0.30 * relevance + 0.25 * citation_score,
            4,
        )
        verdict = "pass" if (overall >= 0.70 and not hallucinated_refs) else "fail"

        return {
            "faithfulness_score": round(faithfulness, 4),
            "answer_relevance_score": round(relevance, 4),
            "citation_grounding_score": round(citation_score, 4),
            "overall_score": overall,
            "verdict": verdict,
            "hallucinated_references": hallucinated_refs,
            "judge_backend": judge_backend,
        }

    def run_human_calibration_check(self) -> Dict[str, Any]:
        """Measures automated judge agreement and MAE against the human-annotated calibration benchmark."""
        agreements = 0
        abs_errors: List[float] = []
        details: List[Dict[str, Any]] = []

        for case in HUMAN_CALIBRATION_SET:
            res = self.evaluate_response(
                query=case["query"],
                answer=case["answer"],
                contexts=case["contexts"],
                use_llm_judge=False,
            )
            agreed = res["verdict"] == case["expected_human_verdict"]
            if agreed:
                agreements += 1
            err = abs(res["overall_score"] - float(case["expected_human_score"]))
            abs_errors.append(err)
            details.append(
                {
                    "id": case["id"],
                    "expected_verdict": case["expected_human_verdict"],
                    "judge_verdict": res["verdict"],
                    "agreed": agreed,
                    "expected_score": case["expected_human_score"],
                    "judge_score": res["overall_score"],
                    "hallucinated_references": res["hallucinated_references"],
                }
            )

        n = max(1, len(HUMAN_CALIBRATION_SET))
        return {
            "num_cases": len(HUMAN_CALIBRATION_SET),
            "human_agreement_rate": round(agreements / n, 4),
            "mean_absolute_error": round(sum(abs_errors) / n, 4),
            "cases": details,
        }

    def _strip_citation_appendix(self, answer: str) -> str:
        """Removes the auto-appended evidence appendix and extractive mode banners before scoring claims."""
        marker = "### Verified Inline Citations"
        if marker in answer:
            answer = answer.split(marker, 1)[0]
        # Strip extractive-mode framing banners emitted when Ollama is offline
        cleaned_lines: List[str] = []
        for line in answer.splitlines():
            low = line.strip().lower()
            if not low:
                continue
            if any(
                banner in low
                for banner in (
                    "ollama offline",
                    "extractive synthesis",
                    "grounded extractive",
                    "verified evidence from",
                    "direct answer from retrieved",
                )
            ):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _score_faithfulness(self, answer: str, context_tokens: set[str]) -> float:
        if not answer or not context_tokens:
            return 0.0
        # Remove inline [Doc i: ...] tags and bold headings before evaluating claim tokens
        prose = DOC_REF_RE.sub("", answer)
        prose = re.sub(r"\*\*[^*\n]{1,60}\*\*:?", "", prose)
        sentences = [s.strip(" -*•") for s in re.split(r"(?<=[.!?])\s+|\n+", prose) if len(s.strip()) >= 12]
        if not sentences:
            sentences = [prose]

        glue_words = {
            "triggered", "operators", "must", "should", "based", "retrieved", "according",
            "states", "specifies", "documented", "indicates", "shows", "using", "used",
            "provides", "includes", "handles", "requires", "between", "within", "during",
            "through", "across", "under", "about", "after", "before", "where", "which",
            "while", "their", "these", "those", "other", "would", "could", "also", "into",
            "from", "with", "that", "this", "have", "been", "were", "when", "local",
            "answer", "query", "source", "document", "section", "evidence", "internal",
        }
        ctx_stems = {t[:5] for t in context_tokens if len(t) >= 5}

        ratios: List[float] = []
        for sent in sentences:
            tokens = [t for t in tokenize_content_words(sent) if len(t) >= 3 and t not in glue_words]
            if not tokens:
                continue
            supported = sum(
                1
                for t in tokens
                if t in context_tokens or (len(t) >= 5 and t[:5] in ctx_stems)
            )
            ratios.append(min(1.0, supported / max(1, len(tokens))))
        return (sum(ratios) / len(ratios)) if ratios else 0.0

    def _score_relevance(self, query: str, answer: str) -> float:
        q_tokens = [t for t in tokenize_content_words(query) if len(t) >= 3]
        if not q_tokens or not answer:
            return 0.0
        a_tokens = set(tokenize_content_words(answer))
        matched = sum(1 for t in q_tokens if t in a_tokens)
        # Saturate at 65% query token coverage since valid answers often use concise phrasing
        return min(1.0, (matched / len(q_tokens)) / 0.65)

    def _audit_citations_and_filenames(
        self, answer: str, contexts: List[str], context_lower: str
    ) -> tuple[List[str], float]:
        hallucinated: List[str] = []
        total_checked = 0

        for match in DOC_REF_RE.finditer(answer):
            total_checked += 1
            idx = int(match.group(1))
            if idx < 1 or idx > len(contexts):
                hallucinated.append(f"Invalid citation [Doc {idx}] (only {len(contexts)} contexts)")

        for match in BACKTICK_ARTIFACT_RE.finditer(answer):
            artifact = match.group(1).strip()
            if not FILENAME_OR_CODE_RE.search(artifact):
                continue
            total_checked += 1
            if artifact.lower() not in context_lower:
                hallucinated.append(f"Ungrounded artifact `{artifact}`")

        if not hallucinated:
            return [], 1.0
        penalty = len(hallucinated) / max(1, total_checked)
        return hallucinated, max(0.0, round(1.0 - penalty, 4))

    def _try_ollama_judge(self, query: str, answer: str, context: str) -> Dict[str, float] | None:
        if not judge_llm_breaker.allow_request():
            return None
        prompt = (
            "Return ONLY valid JSON with float keys 'faithfulness' and 'relevance' in [0.0, 1.0].\n"
            f"Query: {query}\nContext: {context[:1200]}\nAnswer: {answer[:800]}"
        )
        try:
            resp = requests.post(
                f"{settings.llm_api_base.rstrip('/')}/chat/completions",
                json={
                    "model": settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=3.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                return None
            parsed = json.loads(match.group(0))
            f_val = max(0.0, min(1.0, float(parsed["faithfulness"])))
            r_val = max(0.0, min(1.0, float(parsed["relevance"])))
            judge_llm_breaker.record_success()
            return {"faithfulness": f_val, "relevance": r_val}
        except Exception:
            judge_llm_breaker.record_failure()
            return None


rag_judge = RAGJudgeEvaluator()
