from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# LangChain & LangGraph Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document as LangChainDocument
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun

# LangGraph Prebuilt ReAct Agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os
import pika  # <--- Added missing import
from pymilvus import connections, Collection, utility #


# --- App & Global State ---
app = FastAPI()
# --- CONFIGURATION ---

# 1. RabbitMQ Config
# Default to 127.0.0.1 for local testing, but allow Docker to override it
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "127.0.0.1")

# 2. Milvus Config
# Default to 127.0.0.1 for local testing
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")

# 3. Ollama Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# --- CONNECTIONS ---

# Connect to Milvus
print(f"Connecting to Milvus at {MILVUS_HOST}...")
connections.connect("default", host=MILVUS_HOST, port="19530")

# Connect to RabbitMQ
print(f"Connecting to RabbitMQ at {RABBITMQ_HOST}...")
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_HOST)
)
channel = connection.channel()
channel.queue_declare(queue='rag_jobs')
# Global State
vectorstore = None
graph = None  # This will hold our compiled LangGraph executable

# LLM Setup (Global)
# llm = ChatOpenAI(
#     temperature=0,  # Lower temp is better for tool use
#     openai_api_base="http://localhost:1234/v1",
#     openai_api_key="lm-studio",
#     model_name="meta-llama-3-8b-instruct"
# )

llm = ChatOpenAI(
    temperature=0,
    # Change port 1234 to 11434 if using Ollama
    openai_api_base="http://host.docker.internal:11434/v1",
    openai_api_key="ollama", # Key can be anything for Ollama
    model_name="llama3.1" # Make sure this matches the model you pulled in Ollama
)


# --- Pydantic Models ---
class DocumentInput(BaseModel):
    content: str


class QueryInput(BaseModel):
    text: str
    thread_id: Optional[str] = "default_thread"


# --- Tools ---

@tool
def document_retriever(query: str) -> str:
    """
    Use this tool to find information inside the user's uploaded documents.
    Always try this tool first if the user asks about specific uploaded content.
    """
    global vectorstore
    if vectorstore is None:
        return "No documents have been uploaded yet."

    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the documents."

    return "\n\n".join([doc.page_content for doc in docs])


@tool
def internet_search(query: str) -> str:
    """
    Use this tool to search the public internet for current events, facts,
    or information not found in the uploaded documents.
    """
    search = DuckDuckGoSearchRun()
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def summarise_content(text: str) -> str:
    """
    Use this tool when the user explicitly asks for a summary of a specific text block
    or when the gathered information is too long and needs condensation.
    """
    # We can use a separate LLM call here to ensure high-quality summarization
    # distinct from the conversational flow.
    summary_prompt = f"Please provide a concise, bullet-point summary of the following text:\n\n{text}"
    response = llm.invoke(summary_prompt)
    return response.content


# --- Graph Initialization ---

def initialize_graph():
    """Initializes the LangGraph ReAct agent."""
    global graph

    tools = [document_retriever, internet_search, summarise_content]
    memory = MemorySaver()

    system_prompt = """You are a helpful assistant with access to 3 tools:
    1. document_retriever: For user uploaded files.
    2. internet_search: For general knowledge.
    3. summarise_content: To condense long information.

    Decide which tool to use based on the user's request."""

    # --- FIX: Use 'prompt' for LangGraph v1.0.2+ ---
    graph = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=system_prompt
    )


# Initialize graph on startup
initialize_graph()


# --- API Endpoints ---

@app.post("/ingest")
async def ingest_document(doc: DocumentInput):
    global vectorstore
    if not doc.content:
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    try:
        # Create Documents
        docs = [LangChainDocument(page_content=doc.content)]

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)

        # Embed & Store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        return {"status": "success", "message": "Document ingested. The agent can now access it."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query")
async def process_query(query: QueryInput):
    global graph

    try:
        # LangGraph expects a dictionary with "messages"
        # We use a configurable thread_id for conversation memory
        config = {"configurable": {"thread_id": query.thread_id}}

        inputs = {"messages": [HumanMessage(content=query.text)]}

        # Invoke the graph
        # stream_mode="values" returns the full state at every step
        # We just want the final result.
        final_state = graph.invoke(inputs, config=config)

        # Extract the last message (AIMessage)
        last_message = final_state["messages"][-1]

        return {
            "response": last_message.content,
            "verified": True,
            "tool_used": len(final_state["messages"]) > 2  # Heuristic: if > 2 msgs, tools were likely used
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")