# This single file simulates all backend services to demonstrate the workflow.
# In a real-world scenario, these services would be separate microservices.

import os
import pika
import json
import requests
import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# --- Service Configuration ---
# These hostnames correspond to the service names in your docker-compose.yml
# This is how services communicate within the Docker network.
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
GEMINI_API_KEY = ""  # Your Gemini API key, or leave blank to use the canvas API.

# In a real app, Milvus would be connected here. For this demo,
# we'll simulate the vector store with an in-memory list.
vector_store = []

app = FastAPI()


# --- RabbitMQ Connections (Simulated for this demo) ---
def get_rabbitmq_connection():
    """Establishes a connection to RabbitMQ with retries."""
    max_retries = 10
    for i in range(max_retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
            return connection
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection attempt {i + 1} failed: {e}")
            if i < max_retries - 1:
                asyncio.sleep(5)  # Wait before retrying
    raise Exception("Could not connect to RabbitMQ after multiple retries.")


# --- Models ---
class Document(BaseModel):
    content: str


class Query(BaseModel):
    text: str


class Chunk(BaseModel):
    content: str
    metadata: Dict


# --- Offline Ingestion Pipeline Simulation ---

def publish_ingestion_message(content: str):
    """
    Simulates the Watcher Service.
    It takes a new document and publishes a message to the ingestion queue.
    """
    try:
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        channel.queue_declare(queue='ingestion_queue')

        message = {"event": "new_file", "content": content}
        channel.basic_publish(
            exchange='',
            routing_key='ingestion_queue',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ))
        print("Message published to ingestion_queue.")
        connection.close()
    except Exception as e:
        print(f"Failed to publish ingestion message: {e}")


@app.post("/ingest")
async def ingest_document(doc: Document, background_tasks: BackgroundTasks):
    """
    API endpoint to trigger the ingestion process.
    It uses a background task to simulate the event-driven nature of the pipeline.
    """
    background_tasks.add_task(process_ingestion_queue, doc.content)
    return {"status": "Ingestion initiated", "message": "Document is being processed in the background."}


def process_ingestion_queue(content: str):
    """
    Simulates the Chunking Director and Chunking Service.
    Consumes from 'ingestion_queue', chunks the document, and prepares it for embedding.
    """
    print("Received message from ingestion_queue. Starting chunking process...")

    # Simple chunking strategy: split by paragraph.
    chunks_data = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

    # Simulates publishing to the embedding queue
    for chunk_text in chunks_data:
        chunk = Chunk(content=chunk_text, metadata={"source": "user_upload", "length": len(chunk_text)})
        # This would be where you publish to 'embedding_queue' in a real setup.
        # For this demo, we'll directly call the "updater" function.
        vector_store_updater(chunk)


def vector_store_updater(chunk: Chunk):
    """
    Simulates the Vector Store Updater service.
    It takes a chunk, generates an embedding, and inserts it into the vector store.

    NOTE: We are skipping the actual embedding generation for this demo.
    In a real system, you would call a local embedding model (e.g., from Ollama) here.
    """
    print(f"Updating vector store with new chunk: '{chunk.content[:30]}...'")
    # In a real app, you'd use a Milvus client to insert the embedding.
    # We are just adding the text to our in-memory store.
    vector_store.append(chunk)


# --- Online RAG Application Simulation ---

@app.post("/query")
async def process_query(query: Query):
    """
    API endpoint that simulates the Orchestrator Service.
    It coordinates the Retriever, Assembler, and LLM services.
    """
    print(f"Received query: '{query.text}'")

    if not vector_store:
        raise HTTPException(status_code=404, detail="No documents have been ingested yet.")

    # Step 1: Simulate the Retriever Service
    # A simple keyword-based similarity search
    retrieved_chunks = []
    for chunk in vector_store:
        if query.text.lower() in chunk.content.lower():
            retrieved_chunks.append(chunk.content)
            if len(retrieved_chunks) >= 3:  # Get top-3 results
                break

    print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

    # Step 2: Simulate the Assembler Service
    context = " ".join(retrieved_chunks)
    augmented_prompt = f"Based on the following context, answer the question. If the context does not contain the answer, state that you do not have enough information.\n\nContext: {context}\n\nQuestion: {query.text}"
    print("Assembled augmented prompt. Sending to LLM...")

    # Step 3: Simulate the LLM Service (making a real API call to Gemini)
    try:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [{"parts": [{"text": augmented_prompt}]}],
            "generationConfig": {"temperature": 0.5, "maxOutputTokens": 500}
        }

        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()

        result = response.json()
        candidate = result.get('candidates', [{}])[0]
        llm_response = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')

        # Step 4: Simulate the Verification Service (simple check)
        # In a real app, this would be a more sophisticated check.
        is_verified = "Based on the context" not in llm_response  # Simple negative check
        if is_verified:
            print("Response verified.")
        else:
            print("Response verification failed (hallucination check).")

        return {"response": llm_response, "source_chunks": retrieved_chunks, "verified": is_verified}

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with LLM service: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from LLM.")
