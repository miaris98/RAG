import gradio as gr
import requests
import os
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# --- Backend API Calls ---

def ingest_document_to_backend(content):
    payload = {"content": content}
    try:
        response = requests.post(f"{BACKEND_URL}/ingest", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Connection error: {e}"}


def submit_job_to_backend(query):
    """Submits a job to the RabbitMQ queue."""
    payload = {"text": query}
    try:
        response = requests.post(f"{BACKEND_URL}/submit-job", json=payload)
        response.raise_for_status()
        return response.json()  # Returns {'job_id': '...', 'status': 'queued'}
    except Exception as e:
        return {"error": str(e)}


def get_job_result(job_id):
    """Checks the status of a specific job."""
    try:
        response = requests.get(f"{BACKEND_URL}/get-job/{job_id}")
        if response.status_code == 404:
            return {"status": "not_found"}
        return response.json()
    except Exception as e:
        return {"status": "error", "result": str(e)}


# --- Gradio UI Functions ---

def ingest_document(content):
    if not content:
        return "Please enter a document."
    response = ingest_document_to_backend(content)
    if response.get("status") == "error":
        return f"Failed: {response['message']}"
    return "✅ Document ingested! You can now submit queries."


def submit_query(query):
    if not query:
        return "Please enter a query.", None

    response = submit_job_to_backend(query)
    if "error" in response:
        return f"Error: {response['error']}", None

    job_id = response.get("job_id")
    return f"Job Submitted! ID: {job_id}\nWait for processing...", job_id


def check_status(job_id):
    if not job_id:
        return "No active job. Submit a query first."

    # Poll for result
    status_response = get_job_result(job_id)
    status = status_response.get("status")
    result = status_response.get("result")
    tools = status_response.get("tools", [])  # Get the list of tools

    if status == "queued":
        return f"⏳ Job {job_id} is QUEUED."
    elif status == "processing":
        return f"⚙️ Job {job_id} is PROCESSING..."
    elif status == "completed":
        # Format the Tool Usage
        tool_icons = ""
        if tools:
            tool_icons = f"\n\n🛠️ **Tools Used:** `{', '.join(tools)}`"
        else:
            tool_icons = "\n\n🧠 **Source:** LLM Internal Knowledge (No tools used)"

        return f"✅ **Result:**\n\n{result}{tool_icons}"

    elif status == "failed":
        return f"❌ Job Failed: {result}"
    else:
        return "Unknown status."


# --- Gradio Interface ---
with gr.Blocks(title="Async RAG Workflow") as demo:
    gr.Markdown("# Async RAG Workflow (RabbitMQ)")

    with gr.Row():
        with gr.Column():
            doc_input = gr.Textbox(lines=8, label="1. Ingest Document")
            ingest_btn = gr.Button("Ingest")
            ingest_out = gr.Textbox(label="Ingestion Status")

        with gr.Column():
            q_input = gr.Textbox(lines=4, label="2. Submit Query")
            submit_btn = gr.Button("Submit Job to Queue")

            # Hidden textbox to store Job ID for the UI logic
            job_id_state = gr.State()

            submission_out = gr.Textbox(label="Submission Status")

            check_btn = gr.Button("Check Job Result")
            final_output = gr.Markdown(label="Final Response")

    # Wiring
    ingest_btn.click(ingest_document, inputs=doc_input, outputs=ingest_out)

    # Submit -> Updates text and stores Job ID in state
    submit_btn.click(submit_query, inputs=q_input, outputs=[submission_out, job_id_state])

    # Check Status -> Uses Job ID state to poll backend
    check_btn.click(check_status, inputs=job_id_state, outputs=final_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)