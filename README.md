# 📚 RAG Workflow: Local Docker Deployment

This project implements a Retrieval-Augmented Generation (RAG) system using a microservices architecture orchestrated by Docker Compose. It allows users to ingest documents, chunk/embed them into a vector database, and query them using a local LLM.

## 🏗️ Architecture Overview

| Service | Technology | Internal Port | External Port | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend** | Gradio | `7860` | **7860** | User interface for ingestion and querying. |
| **Backend** | FastAPI | `8000` | **8000** | API handling logic, LangGraph agent, and ingestion. |
| **LLM** | Ollama | `11434` | **11434** | Hosts the local LLM (`llama3.1`). |
| **Vector DB** | Milvus | `19530` | **19530** | Stores document embeddings. |
| **Queue** | RabbitMQ | `5672` | **15672** (UI) | Manages async jobs (infrastructure ready). |
| **Storage** | MinIO | `9000` | N/A | Object storage dependency for Milvus. |
| **Coordination**| Etcd | `2379` | N/A | Metadata storage dependency for Milvus. |

---

## ✅ Prerequisites

1.  **Docker Desktop** installed and running.
2.  **Git** (optional, for cloning).
3.  **Hardware:** Minimum **8GB-16GB RAM** recommended (running Milvus + Ollama Llama 3 concurrently is resource-intensive).

---

## 🚀 Quick Start Guide

### 1. Setup Project Structure
Ensure your files are organized exactly as shown below:

```text
RAG/
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
└── docker-compose.yml
```

### 2. Build and Start Services
Open your terminal in the `RAG/` root directory and run:

```bash
docker-compose up -d --build
```
* `-d`: Runs containers in detached mode (background).
* `--build`: Forces a rebuild of the backend/frontend images.

### 3. ⚠️ Critical Step: Pull the LLM Model
The **Ollama** container starts empty. You must pull the specific model defined in `backend/main.py` (`llama3.1`) for the backend to work.

Run this command while the containers are running:
```bash
docker exec -it ollama ollama pull llama3.1
```
*Wait for the download to complete (approx. 4.7GB).*

---

## 🖥️ How to Use

### 1. Access the Interfaces
* **User Interface (Gradio):** [http://localhost:7860](http://localhost:7860)
* **Backend Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **RabbitMQ Management:** [http://localhost:15672](http://localhost:15672) (User/Pass: `guest`/`guest`)

### 2. Run a RAG Workflow
1.  Open the **Gradio UI**.
2.  **Ingest:** Paste text into the "Ingest a Document" box and click **Ingest Document**.
    * *Check:* You should see a success message indicating chunking and embedding are complete.
3.  **Query:** Type a question related to that text in the "Query" box and click **Submit Query**.
    * *Result:* The LLM will answer based on the context retrieved from Milvus.

---

## 🔍 Monitoring & Logs (The Detailed View)

When things go wrong, you need to check the logs of specific containers. Here is how to view and interpret them.

### 1. View All Logs Live
To see a stream of logs from all services at once:
```bash
docker-compose logs -f
```
*(Press `Ctrl+C` to exit)*

### 2. Backend Logs (FastAPI)
Check here for Python errors, connection issues with Milvus/Ollama, or LangChain errors.
```bash
docker-compose logs -f fastapi-backend
```
**Look for:**
* `Application startup complete`: The API is ready.
* `Connecting to Milvus...`: Connection attempts.
* `HTTPException`: Errors processing requests.

### 3. Frontend Logs (Gradio)
Check here if the UI isn't loading or if buttons aren't responding.
```bash
docker-compose logs -f frontend
```
**Look for:**
* `Running on local URL:  http://0.0.0.0:7860`: The UI is successfully hosted.

### 4. Ollama Logs (LLM)
Check here if the model isn't generating text or if the model pull failed.
```bash
docker-compose logs -f ollama
```
**Look for:**
* `msg="inference compute"`: Indicates the model is actively processing a prompt.
* `Error: model "llama3.1" not found`: You forgot the "Critical Step" (Step 3 above).

### 5. Vector DB Logs (Milvus)
Milvus is complex. If it fails, check the standalone container.
```bash
docker-compose logs -f milvus
```
**Look for:**
* `Milvus Proxy started successfully`: The database is ready to accept connections.

---

## 🛠️ Troubleshooting Common Issues

### ❌ Error: "Model not found" or "Connection refused" to Ollama
* **Cause:** You didn't pull the model inside the container.
* **Fix:** Run `docker exec -it ollama ollama pull llama3.1`.
* **Check:** Run `docker exec -it ollama ollama list` to verify the model exists.

### ❌ Error: "Failed to connect to backend" in Gradio
* **Cause:** The frontend container cannot reach the backend container.
* **Fix:** Ensure `docker-compose.yml` has `backend` service named `fastapi-backend` and the frontend `BACKEND_URL` environment variable matches that name (e.g., `http://fastapi-backend:8000`).
* **Check:** Your provided `docker-compose.yml` correctly sets this up.

### ❌ Error: Milvus Connection Timeout
* **Cause:** Milvus takes a while to start (longer than the backend).
* **Fix:** Restart just the backend service so it retries the connection:
    ```bash
    docker-compose restart fastapi-backend
    ```

### ❌ Port Conflicts (e.g., "Port 8000 is already in use")
* **Cause:** You have another process (or another Docker instance) running on that port.
* **Fix:** Stop the other process or change the left-side port in `docker-compose.yml` (e.g., `"8001:8000"`).

---

## 🧹 Cleanup
To stop all containers and remove the networks (data in volumes will persist):
```bash
docker-compose down
```

To stop everything and **delete all data** (RabbitMQ queues, Milvus vectors, Ollama models):
```bash
docker-compose down -v
```