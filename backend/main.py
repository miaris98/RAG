from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document as LangChainDocument

# New agent system imports
from langchain.agents import create_agent
from langchain.tools import tool

# --- FastAPI app ---
app = FastAPI()
qa_agent = None  # global variable to hold the ReAct agent

# --- Pydantic models ---
class Document(BaseModel):
    content: str

class Query(BaseModel):
    text: str

# --- Helper functions ---
def create_rag_agent(vectorstore, llm):
    # Create a retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5}
    )

    # Wrap the retriever as a callable tool
    document_tool = tool(
        lambda q: retriever.invoke(q),
        description="Retrieve relevant documents from ingested content"
    )

    # Create the agent with the LLM and tool
    agent = create_agent(
        model=llm,
        tools=[document_tool],
        system_prompt="You are a helpful assistant. Use the tool to answer user questions.",
        debug=True
    )

    return agent




def create_rag_pipeline(document_content: str):
    """Ingest a document, create embeddings, vectorstore, and ReAct agent."""
    global qa_agent
    print("Starting RAG pipeline creation...")

    # 1️⃣ Create LangChain document
    docs = [LangChainDocument(page_content=document_content)]

    # 2️⃣ Chunk the document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    # 3️⃣ Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4️⃣ Build FAISS vectorstore
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 5️⃣ Connect to LLM (ChatOpenAI / local LM)
    llm = ChatOpenAI(
        temperature=0.7,
        max_tokens=2048,
        openai_api_base="http://localhost:1234/v1",  # adjust for your LM
        openai_api_key="lm-studio",                  # adjust key if needed
        model_name="meta-llama-3-8b-instruct"
    )

    # 6️⃣ Create ReAct agent
    qa_agent = create_rag_agent(vectorstore, llm)
    print("RAG pipeline with ReAct agent is ready!")


# --- API endpoints ---
@app.post("/ingest")
async def ingest_document(doc: Document):
    """Ingest document and create the RAG pipeline."""
    if not doc.content:
        raise HTTPException(status_code=400, detail="Document content cannot be empty.")
    try:
        create_rag_pipeline(doc.content)
        return {"status": "success", "message": "Document ingested successfully. You can now query."}
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create RAG pipeline: {e}")


@app.post("/query")
async def process_query(query: Query):
    global qa_agent
    if not qa_agent:
        raise HTTPException(status_code=404, detail="No document has been ingested yet.")

    try:
        # Instead of qa_agent.run(query.text), do:
        response = qa_agent.invoke({"input": query.text})

        # If the response is a dict, you can extract output:
        if isinstance(response, dict):
            output_text = response.get("output", str(response))
        else:
            output_text = str(response)

        return {"response": output_text, "verified": True}

    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")

