import uuid
import json
import threading
import time
import os
import pika
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# --- CHANGED: Import Milvus instead of FAISS ---
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document as LangChainDocument
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# --- App & Global State ---
app = FastAPI()

# --- CONFIGURATION ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "127.0.0.1")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# --- GLOBAL STORAGE (In-Memory for Demo) ---
JOBS: Dict[str, Dict] = {}

# Global State for Logic
# We don't need to manually connect to Milvus with pymilvus anymore;
# LangChain handles it.
vectorstore = None
graph = None

# Embeddings (Global)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

    # Check if vectorstore is initialized. If not, try to load existing collection.
    if vectorstore is None:
        try:
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name="rag_documents",
                connection_args={"host": MILVUS_HOST, "port": "19530"}
            )
        except Exception:
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


# --- Graph Initialization ---
def initialize_graph():
    global graph
    tools = [document_retriever, internet_search, summarise_content]
    memory = MemorySaver()

    system_prompt = """You are a comprehensive RAG assistant equipped with two distinct information sources: 'document_retriever' (Internal Knowledge) and 'internet_search' (External Knowledge).

    RULES:
    1. ALWAYS query BOTH the 'document_retriever' and 'internet_search' tools for every user request.
    2. You must not merge the information silently. You must explicitly separate the findings.
    3. Your final answer must follow this structure:
       - "According to internal documents:" [Insert answer based strictly on document_retriever]
       - "According to the internet:" [Insert answer based on internet_search]
       - "Comparison:" [Briefly note if they agree, disagree, or if one source offers details the other missed]
    
    4. If the document retriever returns no results, state clearly: "My internal documents do not contain information on this topic."
    """

    graph = create_react_agent(model=llm, tools=tools, checkpointer=memory, prompt=system_prompt)


initialize_graph()


# --- BACKGROUND WORKER ---
def process_rag_job(prompt: str, job_id: str):
    print(f"Processing Job {job_id}...")
    try:
        global graph
        config = {"configurable": {"thread_id": job_id}}
        inputs = {"messages": [HumanMessage(content=prompt)]}

        final_state = graph.invoke(inputs, config=config)
        last_message = final_state["messages"][-1]

        # Detect Tools
        tools_used = []
        for msg in final_state["messages"]:
            if msg.type == "tool":
                tools_used.append(msg.name)
        tools_used = list(set(tools_used))

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = last_message.content
        JOBS[job_id]["tools"] = tools_used
        print(f"Job {job_id} Completed. Tools: {tools_used}")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["result"] = str(e)
        JOBS[job_id]["tools"] = []
        print(f"Job {job_id} Failed: {e}")


def consume_messages():
    print(" [*] Worker thread started...")
    while True:
        try:
            params = pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue='rag_jobs')
            channel.basic_qos(prefetch_count=1)

            def callback(ch, method, properties, body):
                data = json.loads(body)
                job_id = data.get("job_id")
                prompt = data.get("prompt")
                print(f" [x] Received Job: {job_id}")
                if job_id and prompt:
                    JOBS[job_id] = {"status": "processing", "result": None}
                    process_rag_job(prompt, job_id)
                ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_consume(queue='rag_jobs', on_message_callback=callback)
            print(" [*] Worker: Connected and waiting for messages...")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            print(" [!] Worker: Connection failed. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f" [!] Worker crashed: {e}. Restarting in 5s...")
            time.sleep(5)


@app.on_event("startup")
def startup_event():
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

        # --- CHANGED: Save to Milvus ---
        vectorstore = Milvus.from_documents(
            split_docs,
            embeddings,
            collection_name="rag_documents",
            connection_args={"host": MILVUS_HOST, "port": "19530"}
        )
        return {"status": "success", "message": "Document ingested to Milvus."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/submit-job")
async def submit_job(query: QueryInput):
    job_id = str(uuid.uuid4())
    payload = {"job_id": job_id, "prompt": query.text}
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
    JOBS[job_id] = {"status": "queued", "result": None}
    return {"job_id": job_id, "status": "queued"}


@app.get("/get-job/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job