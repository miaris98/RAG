"""Resilient RabbitMQ worker and job orchestration with automatic local thread fallback."""

import json
import threading
import time
import uuid
from typing import Any, Dict, Optional

try:
    import pika
except ImportError:
    pika = None  # type: ignore[assignment]

from app.agent.graph import execute_rag_pipeline
from app.config import settings
from app.ingestion.pipeline import ingestion_pipeline
from app.models import DocumentInput, QueryInput


JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.RLock()


def process_rag_job(query_payload: Dict[str, Any], job_id: str) -> None:
    """Executes the LangGraph CRAG + Self-RAG pipeline for a queued job and updates JOBS[job_id]."""
    print(f"[Worker] Processing Job {job_id}...")
    with _JOBS_LOCK:
        if job_id not in JOBS:
            JOBS[job_id] = {}
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["started_at"] = time.time()

    try:
        # Support both legacy {"prompt": "..."} payloads and full QueryInput payloads
        if "text" not in query_payload and "prompt" in query_payload:
            query_payload = {**query_payload, "text": query_payload["prompt"]}

        query_input = QueryInput(**query_payload)
        output = execute_rag_pipeline(query_input)

        with _JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "completed",
                    "result": output["answer"],
                    "tools": output["tools"],
                    "trace": output["trace"],
                    "completed_at": time.time(),
                }
            )
        print(f"[Worker] Job {job_id} Completed. Tools: {output['tools']}")
    except Exception as exc:
        with _JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "failed",
                    "result": str(exc),
                    "tools": [],
                    "trace": None,
                    "completed_at": time.time(),
                }
            )
        print(f"[Worker] Job {job_id} Failed: {exc}")


def enqueue_rag_job(query_input: QueryInput) -> Dict[str, Any]:
    """Publishes a RAG query job to RabbitMQ, or falls back to a background thread if broker is offline."""
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "prompt": query_input.text,
        "text": query_input.text,
        "retrieval_mode": query_input.retrieval_mode,
        "top_k": query_input.top_k,
        "tenant_id": query_input.tenant_id,
        "user_roles": query_input.user_roles,
        "enable_query_expansion": query_input.enable_query_expansion,
        "enable_web_search": query_input.enable_web_search,
        "enable_self_rag": query_input.enable_self_rag,
        "enable_semantic_cache": query_input.enable_semantic_cache,
        "enable_guardrails": query_input.enable_guardrails,
    }

    with _JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "result": None,
            "tools": [],
            "trace": None,
            "created_at": time.time(),
        }

    try:
        if pika is None:
            raise RuntimeError("pika not installed in local host Python")
        params = pika.ConnectionParameters(
            host=settings.rabbitmq_host,
            port=settings.rabbitmq_port,
            connection_attempts=1,
            socket_timeout=2.0,
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue=settings.rabbitmq_queue)
        channel.basic_publish(
            exchange="",
            routing_key=settings.rabbitmq_queue,
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        connection.close()
        return {"job_id": job_id, "status": "queued", "broker": "rabbitmq"}
    except Exception as exc:
        if settings.async_fallback_on_broker_down:
            print(f"[Worker] RabbitMQ unreachable ({exc}); executing job {job_id} via local async thread.")
            t = threading.Thread(target=process_rag_job, args=(payload, job_id), daemon=True)
            t.start()
            return {"job_id": job_id, "status": "queued", "broker": "in_process_thread_fallback"}
        raise RuntimeError(f"RabbitMQ unavailable: {exc}") from exc


def process_ingestion_job(doc_payload: Dict[str, Any], job_id: str) -> None:
    """Executes asynchronous document chunking, guardrail sanitization, and indexing."""
    with _JOBS_LOCK:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["started_at"] = time.time()
    try:
        doc_input = DocumentInput(**doc_payload)
        res = ingestion_pipeline.ingest(doc_input)
        with _JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "completed",
                    "result": res,
                    "completed_at": time.time(),
                }
            )
    except Exception as exc:
        with _JOBS_LOCK:
            JOBS[job_id].update({"status": "failed", "result": str(exc), "completed_at": time.time()})


def enqueue_ingestion_job(doc_input: DocumentInput) -> Dict[str, Any]:
    """Queues an async document ingestion job (RabbitMQ or local thread fallback)."""
    job_id = str(uuid.uuid4())
    payload = doc_input.model_dump()
    with _JOBS_LOCK:
        JOBS[job_id] = {"job_id": job_id, "status": "queued", "result": None, "created_at": time.time()}

    t = threading.Thread(target=process_ingestion_job, args=(payload, job_id), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued", "broker": "async_ingestion_worker"}


def get_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the current state and telemetry of a job by ID."""
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None


def consume_messages_forever() -> None:
    """Long-running daemon loop consuming RAG jobs from RabbitMQ."""
    if pika is None:
        print(" [*] RabbitMQ client (pika) not installed on host; using local async thread worker.")
        return
    print(" [*] RabbitMQ Worker thread started...")
    while True:
        try:
            params = pika.ConnectionParameters(
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
                heartbeat=600,
            )
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=settings.rabbitmq_queue)
            channel.basic_qos(prefetch_count=1)

            def callback(ch, method, properties, body):
                data = json.loads(body)
                job_id = data.get("job_id")
                print(f" [x] Received Job from RabbitMQ: {job_id}")
                if job_id:
                    process_rag_job(data, job_id)
                ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_consume(queue=settings.rabbitmq_queue, on_message_callback=callback)
            print(" [*] Worker: Connected to RabbitMQ and waiting for messages...")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            time.sleep(5)
        except Exception as exc:
            print(f" [!] Worker error: {exc}. Retrying in 5s...")
            time.sleep(5)


def start_background_worker() -> threading.Thread:
    """Spawns the RabbitMQ consumer daemon thread."""
    thread = threading.Thread(target=consume_messages_forever, daemon=True)
    thread.start()
    return thread
