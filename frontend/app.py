import gradio as gr
import requests
import json

# Define the base URL for the FastAPI backend service
# This hostname must match the service name in your docker-compose.yml
# BACKEND_URL = "http://fastapi-backend:8000"
BACKEND_URL = "http://localhost:8000"

# --- Backend API Calls ---

def ingest_document_to_backend(content):
    """Sends a document to the FastAPI backend for ingestion."""
    payload = {"content": content}
    try:
        response = requests.post(f"{BACKEND_URL}/ingest", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Failed to connect to backend: {e}"}


def query_rag_backend(query):
    """Sends a user query to the FastAPI backend and gets the RAG response."""
    payload = {"text": query}
    try:
        response = requests.post(f"{BACKEND_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Failed to connect to backend: {e}"}


# --- Gradio UI Functions ---

def ingest_document(content):
    """Handles the document ingestion button click."""
    if not content:
        return "Please enter a document to ingest."

    response = ingest_document_to_backend(content)
    if response.get("status") == "error":
        return f"Ingestion failed: {response['message']}"

    return "Document ingestion initiated! It's being processed in the background. You can now submit a query."


def rag_query(query):
    """Handles the RAG query button click and displays the result."""
    if not query:
        return "Please enter a query."

    response = query_rag_backend(query)

    if response.get("status") == "error":
        return f"Query failed: {response['message']}"

    llm_response = response.get("response", "No response from LLM.")
    source_chunks = response.get("source_chunks", [])
    verified = response.get("verified", False)

    # Format the output with source chunks
    output = f"**LLM Response:**\n{llm_response}\n\n"
    output += f"**Verification:** {'✅ Verified' if verified else '❌ Not verified'}\n\n"
    output += "**Source Chunks (Retrieved Context):**\n"
    if source_chunks:
        for i, chunk in enumerate(source_chunks):
            output += f"- Chunk {i + 1}: {chunk}\n"
    else:
        output += "No relevant chunks were found in the document."

    return output


# --- Gradio Interface ---
with gr.Blocks(title="RAG Workflow Demo") as demo:
    gr.Markdown("# RAG Workflow Demo")
    gr.Markdown(
        "This interface simulates a Retrieval-Augmented Generation (RAG) pipeline by connecting to a FastAPI backend.")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            document_input = gr.Textbox(lines=10, label="1. Ingest a Document",
                                        placeholder="Paste a document here to be chunked and embedded.")
            ingest_btn = gr.Button("Ingest Document")
            ingest_output = gr.Textbox(label="Ingestion Status", interactive=False)

        with gr.Column():
            query_input = gr.Textbox(lines=5, label="2. Query the RAG System",
                                     placeholder="Ask a question about the document.")
            query_btn = gr.Button("Submit Query")
            query_output = gr.Markdown(label="RAG Response")

    ingest_btn.click(ingest_document, inputs=document_input, outputs=ingest_output)
    query_btn.click(rag_query, inputs=query_input, outputs=query_output)

if __name__ == "__main__":
    # The Gradio app needs to listen on 0.0.0.0 for Docker to expose it correctly.
    demo.launch(server_name="0.0.0.0", server_port=7860)
