# RAG (Retrieval-Augmented Generation) Pipeline

This project is a multi-service application that demonstrates a complete offline and online RAG workflow using Docker Compose. It features a scalable, event-driven ingestion pipeline and a real-time query application.

## 🚀 Architecture

The system is built as a set of interconnected microservices managed by Docker Compose. The workflow is split into two main parts:

-   **Offline Ingestion Pipeline:** An event-driven workflow that watches for new documents, chunks them, and stores their vector embeddings in a database.
-   **Online RAG Application:** A user-facing application that takes queries, retrieves relevant information from the vector database, and uses an LLM to generate a response.

### Core Components

-   **FastAPI Backend (`fastapi-backend`):** The central hub of the application. It handles document ingestion, query orchestration, and communication with all other services.
-   **Gradio Frontend (`frontend`):** A simple, interactive user interface for ingesting documents and submitting queries.
-   **RabbitMQ (`rabbitmq`):** The message broker that acts as the "central nervous system" for the ingestion pipeline.
-   **Milvus (`milvus`):** The vector database that stores and indexes the document embeddings for efficient retrieval.
-   **Ollama (`ollama`):** The service that provides a local, self-hosted LLM and embedding model.

## ✅ Prerequisites

Ensure you have the following installed on your machine:

-   **Docker:** https://www.docker.com/get-started
-   **Docker Compose:** Comes with modern Docker installations.

## 💡 Getting Started

Follow these steps to build and run the entire application.

### Step 1: Create the Project Structure

Create the project directories and files as follows. You should already have the `docker-compose.yml` file.

```sh
mkdir my-rag-project
cd my-rag-project
mkdir backend
mkdir frontend

# Create the requirements.txt and Dockerfile for the backend
touch backend/requirements.txt
touch backend/Dockerfile

# Create the main Python application file for the backend
touch backend/main.py

# Create the requirements.txt and Dockerfile for the frontend
touch frontend/requirements.txt
touch frontend/Dockerfile

# Create the main Python application file for the frontend
touch frontend/app.py

```
### Step 2: Populate the Files

Use the content provided to you for each of the files you just created.

-   **`backend/requirements.txt`**: List of Python dependencies for the FastAPI app.
-   **`backend/Dockerfile`**: Instructions to build the backend Docker image.
-   **`backend/main.py`**: The core FastAPI application logic.
-   **`frontend/requirements.txt`**: List of Python dependencies for the Gradio app.
-   **`frontend/Dockerfile`**: Instructions to build the frontend Docker image.
-   **`frontend/app.py`**: The Gradio UI and logic for interacting with the backend.

### Step 3: Launch the Application

From the root of your project directory (`my-rag-project/`), run the following command. This will build the images and start all the services. The `--build` flag ensures that your `Dockerfile`s are executed.

```sh
docker-compose up --build
```
Wait a few moments for all services to start. You will see logs from each container in your terminal.

## 💻 Usage

Once the application is running, you can access the Gradio frontend.

1.  Open your web browser and navigate to: http://localhost:7860
2.  Paste a document into the "Ingest a Document" text box and click "Ingest Document."
3.  Once the ingestion status shows success, you can enter a query in the "Query the RAG System" text box and see the LLM's response based on the retrieved context.