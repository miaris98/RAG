import uuid
import json
import threading
import time
import os
import pika
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# LangChain & LangGraph Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document as LangChainDocument
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pymilvus import connections

# --- App & Global State ---
app = FastAPI()

# --- CONFIGURATION ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "127.0.0.1")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# --- GLOBAL STORAGE (In-Memory for Demo) ---
JOBS: Dict[str, Dict] = {}

# --- CONNECTIONS ---
# Connect to Milvus
print(f"Connecting to Milvus at {MILVUS_HOST}...")
try:
    connections.connect("default", host=MILVUS_HOST, port="19530")
except Exception as e:
    print(f"Warning: Milvus connection failed (expected during build): {e}")

# Global State for Logic
vectorstore = None
graph = None

# LLM Setup
llm = ChatOpenAI(
    temperature=0,
    openai_api_base="http://host.docker.internal:11434/v1",
    openai_api_key="ollama",
    model_name="llama3.1"
)


# --- Pydantic Models ---
class DocumentInput(BaseModel):
    content: str


class QueryInput(BaseModel):
    text: str


# --- Tools ---
@tool
def document_retriever(query: str) -> str:
    """Find information inside uploaded documents."""
    global vectorstore
    if vectorstore is None:
        return "No documents have been uploaded yet."
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def internet_search(query: str) -> str:
    """Search public internet for current events."""
    search = DuckDuckGoSearchRun()
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def summarise_content(text: str) -> str:
    """Condense long information."""
    summary_prompt = f"Summarize this:\n\n{text}"
    response = llm.invoke(summary_prompt)
    return response.content


# --- UPDATED: Graph Initialization with Strict Prompts ---
def initialize_graph():
    global graph
    tools = [document_retriever, internet_search, summarise_content]
    memory = MemorySaver()

    # We force the LLM to prioritize the retriever
    system_prompt = """You are a RAG assistant. Your PRIMARY source of truth is the 'document_retriever' tool.

    RULES:
    1. ALWAYS check 'document_retriever' first.
    2. If the document answers the question, output that answer EXACTLY as written in the text, even if it contradicts real-world facts.
    3. If the user says "the president is Dennis", then the president is Dennis. Do not correct it with real-world knowledge.
    4. Only use 'internet_search' if the document is empty or irrelevant.
    """

    graph = create_react_agent(model=llm, tools=tools, checkpointer=memory, prompt=system_prompt)


initialize_graph()


# --- BACKGROUND WORKER (The "Async" Part) ---

# --- UPDATED: Process Job with Tool Detection ---
def process_rag_job(prompt: str, job_id: str):
    """Runs the actual RAG logic and detects tool usage."""
    print(f"Processing Job {job_id}...")
    try:
        global graph
        config = {"configurable": {"thread_id": job_id}}
        inputs = {"messages": [HumanMessage(content=prompt)]}

        # Run the Agent
        final_state = graph.invoke(inputs, config=config)
        last_message = final_state["messages"][-1]

        # --- DETECT TOOLS 🔧 ---
        # We look through the message history for "ToolMessage"
        tools_used = []
        for msg in final_state["messages"]:
            if msg.type == "tool":
                # msg.name often contains the tool name
                tools_used.append(msg.name)

        # Remove duplicates
        tools_used = list(set(tools_used))

        # Update Job Status
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = last_message.content
        JOBS[job_id]["tools"] = tools_used  # <--- Saved for Frontend

        print(f"Job {job_id} Completed. Tools: {tools_used}")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["result"] = str(e)
        JOBS[job_id]["tools"] = []
        print(f"Job {job_id} Failed: {e}")


def consume_messages():
    """Robust background thread that auto-reconnects to RabbitMQ."""
    print(" [*] Worker thread started...")

    while True:
        try:
            # 1. Connect directly inside the thread (Separate connection from API)
            params = pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue='rag_jobs')
            channel.basic_qos(prefetch_count=1)

            # 2. Define the callback logic
            def callback(ch, method, properties, body):
                data = json.loads(body)
                job_id = data.get("job_id")
                prompt = data.get("prompt")

                print(f" [x] Received Job: {job_id}")

                if job_id and prompt:
                    JOBS[job_id] = {"status": "processing", "result": None}
                    process_rag_job(prompt, job_id)

                # Acknowledge completion
                ch.basic_ack(delivery_tag=method.delivery_tag)

            # 3. Start consuming
            channel.basic_consume(queue='rag_jobs', on_message_callback=callback)
            print(" [*] Worker: Connected and waiting for messages...")
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            print(f" [!] Worker: Connection failed (RabbitMQ might be starting). Retrying in 5s... Error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f" [!] Worker: crashed with error: {e}. Restarting in 5s...")
            time.sleep(5)


# Start Consumer in Background Thread on Startup
@app.on_event("startup")
def startup_event():
    # daemon=True ensures thread dies when main app dies
    thread = threading.Thread(target=consume_messages, daemon=True)
    thread.start()


# --- API Endpoints ---

@app.post("/ingest")
async def ingest_document(doc: DocumentInput):
    global vectorstore
    if not doc.content:
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    try:
        docs = [LangChainDocument(page_content=doc.content)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return {"status": "success", "message": "Document ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/submit-job")
async def submit_job(query: QueryInput):
    job_id = str(uuid.uuid4())
    payload = {"job_id": job_id, "prompt": query.text}

    # We open a temporary connection just for publishing to ensure thread safety
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue='rag_jobs')
        channel.basic_publish(
            exchange='',
            routing_key='rag_jobs',
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RabbitMQ unavailable: {e}")

    # Initialize job in memory
    JOBS[job_id] = {"status": "queued", "result": None}
    return {"job_id": job_id, "status": "queued"}


@app.get("/get-job/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job