from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document as LangChainDocument
from langchain.tools import tool
from langchain.agents import create_agent

# --- App & Global State ---
app = FastAPI()

qa_agent = None  # Global ReAct agent
vectorstore = None  # Store for documents


# --- Pydantic Models ---
class Document(BaseModel):
    content: str


class Query(BaseModel):
    text: str


# --- Core Functions ---
def create_rag_agent(vectorstore, llm):
    """Creates a ReAct agent with a retriever tool."""
    retriever = vectorstore.as_retriever()

    # Explicitly name the tool
    @tool(description="Retrieve relevant documents from ingested content")
    def document_retriever(query: str):
        """Return relevant document text in a dict."""
        docs = retriever.invoke(query)
        return {"output": "\n\n".join([doc.page_content for doc in docs])}

    agent = create_agent(
        model=llm,
        tools=[document_retriever],
        system_prompt="You are a helpful assistant that can use the document_retriever tool to answer queries directly.",
        debug=True
    )
    return agent


def create_rag_pipeline(document_content: str):
    """Ingest a new document and build the RAG pipeline."""
    global qa_agent, vectorstore

    docs = [LangChainDocument(page_content=document_content)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    llm = ChatOpenAI(
        temperature=0.7,
        max_tokens=2048,
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="lm-studio",
        model_name="meta-llama-3-8b-instruct"
    )

    qa_agent = create_rag_agent(vectorstore, llm)


# --- API Endpoints ---
@app.post("/ingest")
async def ingest_document(doc: Document):
    if not doc.content:
        raise HTTPException(status_code=400, detail="Document content cannot be empty.")
    try:
        create_rag_pipeline(doc.content)
        return {"status": "success", "message": "Document ingested and RAG agent ready."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query")
async def process_query(query: Query):
    global qa_agent
    if qa_agent is None:
        raise HTTPException(status_code=404, detail="No document ingested yet.")

    try:
        # Invoke the agent with plain text
        response = qa_agent.invoke({"input": query.text})

        if hasattr(response, "content"):
            output_text = response.content
        elif isinstance(response, dict) and "output" in response:
            output_text = response["output"]
        else:
            output_text = str(response)

        return {"response": output_text, "verified": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
