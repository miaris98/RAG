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

    @tool(description="Retrieve relevant documents from ingested content")
    def document_retriever(query: str):
        """Return relevant document text."""
        docs = retriever.invoke(query)
        # --- THIS IS THE CHANGE ---
        # Return a simple string. The agent will handle it.
        return "\n\n".join([doc.page_content for doc in docs])

    # This system prompt is still the best one to use
    system_prompt = """You are an assistant for question-answering tasks.
    You must use the 'document_retriever' tool to find relevant information to answer the user's question.
    Your final answer should be based *only* on the content returned by the tool.
    Do not use your own knowledge.
    If the tool returns information, synthesize it into a clear answer.
    If the tool returns no relevant information, just say 'I could not find an answer in the documents.'
    """

    # Your original create_agent call was correct!
    agent = create_agent(
        model=llm,
        tools=[document_retriever],
        system_prompt=system_prompt,
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
        # --- FIX 1: Correct Input Format ---
        # Pass the user's query in the 'messages' list format
        inputs = {"messages": [{"role": "user", "content": query.text}]}

        # Invoke the agent graph
        response = qa_agent.invoke(inputs)

        # --- FIX 2: Correct Output Parsing ---
        # The response is the final state dict. We need to
        # extract the last message, which is the AI's answer.
        if isinstance(response, dict) and "messages" in response:
            final_message = response["messages"][-1]

            # The final_message is an AIMessage object
            if hasattr(final_message, "content"):
                output_text = final_message.content
            else:
                output_text = str(final_message)  # Fallback
        else:
            # Fallback for an unexpected response
            output_text = str(response)

        return {"response": output_text, "verified": True}

    except Exception as e:
        print(f"Error during query: {e}")  # Good for debugging
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")