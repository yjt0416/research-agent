# Research Agent Copilot

A full-stack AI research assistant for document-grounded Q&A, report generation, safe tool usage, and downloadable output artifacts.

## Overview

Research Agent Copilot is a LangChain-centered agent application built for technical and research workflows. It combines:

- Conversational chat over a DeepSeek-powered LLM
- Retrieval-augmented generation over uploaded TXT, PDF, and DOCX files
- Safe workspace tools for file reading and restricted Python execution
- Human-in-the-loop confirmation for tool actions
- Downloadable outputs such as Markdown reports, Python files, and ZIP bundles

The project is designed to feel like a practical product, not just a set of isolated demos. It includes a streamlined chat-style frontend, backend APIs, evaluation samples, and automated tests.

## Key Features

- Chat-style frontend with a single entry point for asking questions, uploading files, and approving actions
- LangChain-based orchestration using DeepSeek chat models, tool routing, and prompt templates
- Semantic retrieval with multilingual sentence-transformer embeddings and Chroma vector storage
- Safe Python tool with AST-based restrictions for simple calculations and code demos
- Downloadable artifacts for generated reports and code outputs
- Human confirmation workflow for potentially sensitive tool actions
- Structured logging and retry logic for LLM requests
- Built-in evaluation dataset summary endpoint for regression tracking

## Architecture

```text
Frontend (single-page chat UI)
        |
        v
FastAPI backend
        |
        +-- LangChain chat / tool / report workflow
        +-- RAG pipeline (text extraction -> chunking -> embeddings -> Chroma)
        +-- Memory store (session history + user preferences)
        +-- Artifact store (markdown / python / zip downloads)
        +-- Confirmation store (approve / cancel tool execution)
```

## Tech Stack

- Backend: FastAPI
- LLM / Agent framework: LangChain, langchain-deepseek
- Vector store: Chroma
- Embeddings: sentence-transformers, langchain-huggingface
- Frontend: static HTML/CSS/JavaScript
- Testing: pytest

## Repository Layout

```text
research-agent/
|- backend/
|  |- app/
|  |  |- main.py
|  |  |- agent.py
|  |  |- rag.py
|  |  |- llm.py
|  |  |- tools.py
|  |  |- artifacts.py
|  |  |- confirmations.py
|  |  |- evaluation.py
|  |  |- logging_utils.py
|  |  |- memory.py
|  |  |- prompts.py
|  |  |- schemas.py
|  |  \- config.py
|  \- requirements.txt
|- frontend/
|  \- index.html
|- tests/
|  \- test_api.py
|- data/
|  |- evals/
|  |  \- day5_eval_dataset.jsonl
|  |- processed/
|  |  \- .gitkeep
|  \- raw/
|     |- .gitkeep
|     \- sample_research_note.txt
|- .env.example
|- .gitignore
\- README.md
```

## Getting Started

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r backend\requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your DeepSeek API key:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
SYSTEM_PROMPT=You are Research Agent Copilot, a helpful assistant for research and technical documents.
CHROMA_COLLECTION_NAME=research_docs_semantic_v1
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=4
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_NORMALIZE=true
```

### 4. Run the app

```powershell
uvicorn app.main:app --app-dir backend --reload
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Core API Endpoints

- `POST /chat`
  Standard chat completion

- `POST /documents/upload`
  Upload TXT / PDF / DOCX documents into the RAG knowledge base

- `POST /chat/rag`
  Ask questions grounded in uploaded materials

- `POST /agent/chat`
  Unified entry point for chat, retrieval, report generation, and tool usage

- `POST /agent/confirm/{token}`
  Approve or cancel a pending tool action

- `POST /tools/read-file`
  Read a workspace file

- `POST /tools/python`
  Run restricted Python code

- `GET /artifacts/{artifact_id}/download`
  Download generated files

- `GET /evaluation/dataset`
  Inspect the built-in evaluation dataset summary

## Example Workflows

### Document-grounded summary

1. Upload `data/raw/sample_research_note.txt`
2. Ask:

```text
Please summarize the core idea of RAG and include citations.
```

3. Receive:
- grounded answer
- source chunks
- downloadable Markdown artifact

### Human-approved tool execution

Send:

```json
{
  "message": "Please run this Python code ```python\nprint(sum([10, 20, 30]))\n```",
  "session_id": "session-demo",
  "user_id": "user-demo",
  "mode": "tool",
  "require_confirmation": true
}
```

The agent will pause and return a confirmation token. Then approve or cancel via:

```json
POST /agent/confirm/{token}
{
  "action": "approve"
}
```

## Testing

Run the automated tests with:

```powershell
pytest -q
```

## Notes on What Is Intentionally Excluded

This repository is intentionally kept clean for source control:

- no local virtual environment
- no runtime logs
- no private `.env`
- no personal learning notes
- no generated vector store or memory artifacts
- no duplicated uploaded files produced during local experimentation

## Roadmap

- Streaming responses in the chat UI
- Conversation history sidebar
- More evaluation automation
- Stronger artifact rendering and preview
- Additional guarded tools for research workflows
