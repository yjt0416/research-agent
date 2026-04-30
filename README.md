# Research Agent Copilot

![Research Agent Copilot interface](assets/ui-screenshot.png)

[![FastAPI](https://img.shields.io/badge/backend-FastAPI-0f766e?style=flat-square)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/orchestration-LangChain-1c7c54?style=flat-square)](https://www.langchain.com/)
[![DeepSeek](https://img.shields.io/badge/model-DeepSeek-0f172a?style=flat-square)](https://www.deepseek.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-1d4ed8?style=flat-square)](LICENSE)

A full-stack AI research assistant for document-grounded Q&A, report generation, safe tool execution, and downloadable output artifacts.

## Why This Project

Research Agent Copilot is a practical agent app built around a simple idea: make research workflows feel as easy as chatting, while still keeping retrieval, tool usage, and output generation structured and safe.

It combines:

- Chat-style interaction powered by DeepSeek through LangChain
- Retrieval-augmented generation over uploaded TXT, PDF, and DOCX documents
- Guarded tools for file reading and restricted Python execution
- Human-in-the-loop confirmation before sensitive actions
- Downloadable outputs such as Markdown reports, Python files, and ZIP bundles

## Highlights

- Single-page frontend designed around a clean chat experience
- LangGraph StateGraph workflow for intent routing, planning, retrieval, tool routing, code execution, reflection, and reporting
- Semantic retrieval with multilingual sentence-transformer embeddings and Chroma
- AST-restricted Python execution for simple, inspectable code tasks
- Downloadable artifacts that make generated output easy to save and share
- Structured retries, logging, evaluation utilities, and automated tests
- Paper reproduction flow that can generate runnable Python, figures, metrics JSON, Markdown reports, and ZIP bundles

## Architecture

```text
Frontend (chat UI)
    |
    v
FastAPI backend
    |
    +-- LangGraph workflow orchestration
    +-- RAG pipeline (extract -> chunk -> embed -> retrieve)
    +-- Session memory and user preferences
    +-- Tool execution with confirmation gates
    +-- Artifact generation (md / py / zip)
    +-- Research reproduction pipeline (paper -> code -> figures -> report)
    +-- Evaluation and logging utilities
```

### System Diagram

```mermaid
flowchart LR
    User["User"] --> UI["Chat UI<br/>frontend/index.html"]
    UI --> API["FastAPI API"]
    API --> Graph["LangGraph StateGraph"]

    Graph --> Router["Intent Router"]
    Router --> Planner["Planner"]
    Planner --> Retriever["Retriever"]
    Planner --> ToolRouter["Tool Router"]
    ToolRouter --> CodeExec["Code Executor"]
    Planner --> Reflector["Reflector"]
    Reflector --> Reporter["Reporter"]

    Retriever --> RAG["Chroma + Semantic Embeddings"]
    CodeExec --> Artifacts["Artifacts<br/>md / py / json / image / zip"]
    Reporter --> Artifacts

    API --> Memory["Session Memory + Preferences"]
    API --> Confirm["Human Approval Gate"]
    Graph --> Repro["Paper Reproduction Pipeline"]
    Repro --> Artifacts
```

## Tech Stack

- Backend: FastAPI
- Agent framework: LangChain, langchain-deepseek
- Vector database: Chroma
- Embeddings: sentence-transformers, langchain-huggingface
- Frontend: HTML, CSS, JavaScript
- Testing: pytest

## Repository Layout

```text
research-agent/
|-- backend/
|   |-- app/
|   |   |-- agent.py
|   |   |-- artifacts.py
|   |   |-- confirmations.py
|   |   |-- config.py
|   |   |-- evaluation.py
|   |   |-- llm.py
|   |   |-- logging_utils.py
|   |   |-- main.py
|   |   |-- memory.py
|   |   |-- prompts.py
|   |   |-- rag.py
|   |   |-- reproduction.py
|   |   |-- schemas.py
|   |   `-- tools.py
|   `-- requirements.txt
|-- data/
|   |-- evals/
|   |   `-- day5_eval_dataset.jsonl
|   |-- processed/
|   |   `-- .gitkeep
|   `-- raw/
|       |-- .gitkeep
|       `-- sample_research_note.txt
|-- frontend/
|   `-- index.html
|-- tests/
|   `-- test_api.py
|-- assets/
|   `-- ui-screenshot.png
|-- showcase/
|   |-- demo-script.md
|   |-- interview-talk-tracks.md
|   `-- resume-bullets.md
|-- CONTRIBUTING.md
|-- LICENSE
`-- README.md
```

## Quick Start

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

Copy `.env.example` to `.env` and add your DeepSeek API key.

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-pro
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

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Core API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /chat` | Standard chat completion |
| `POST /documents/upload` | Upload TXT, PDF, or DOCX files into the RAG store |
| `POST /documents/ingest-path` | Ingest an existing local TXT, PDF, or DOCX file by path |
| `POST /chat/rag` | Ask questions grounded in uploaded material |
| `POST /agent/chat` | Unified entry point for chat, retrieval, reports, tools, and research workflows |
| `POST /agent/confirm/{token}` | Approve or cancel a pending tool action |
| `POST /tools/read-file` | Read a workspace file |
| `POST /tools/python` | Run restricted Python code |
| `GET /artifacts/{artifact_id}/download` | Download generated output |
| `GET /evaluation/dataset` | Inspect the bundled evaluation dataset |

## Example Workflows

### 1. Grounded document summary

1. Upload `data/raw/sample_research_note.txt`.
2. Ask: `Please summarize the core idea of RAG and include citations.`
3. Receive a grounded answer, retrieved source chunks, and a downloadable Markdown artifact.

### 2. Human-approved Python execution

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

The agent pauses and returns a confirmation token. Approve or cancel it with:

```json
POST /agent/confirm/{token}
{
  "action": "approve"
}
```

### 3. Paper reproduction workflow

Send a research-mode request with an explicit local paper path:

```json
{
  "message": "Please reproduce this paper with runnable Python code and generate a technical report.",
  "session_id": "paper-repro-demo",
  "user_id": "research-user",
  "mode": "research",
  "document_paths": [
    "D:/papers/alpha-stable-vlf-paper.pdf"
  ]
}
```

The workflow will:

- ingest the paper into the vector store
- plan the task with LangGraph nodes
- retrieve formulas and simulation parameters
- generate and execute a standalone Python reproduction script
- return downloadable figures, metrics JSON, Markdown report, and a ZIP bundle

## Demo Plan

If you want to record a short project demo, use this order:

1. Show the chat homepage and upload a TXT or PDF document.
2. Ask a grounded question and point out the cited sources.
3. Trigger a guarded Python task and explain the approval step.
4. Run a research reproduction request and download the generated report or ZIP bundle.

Ready-to-use speaking notes and screen-recording steps are available in:

- [showcase/demo-script.md](showcase/demo-script.md)
- [showcase/interview-talk-tracks.md](showcase/interview-talk-tracks.md)
- [showcase/resume-bullets.md](showcase/resume-bullets.md)

## Development

Run the test suite:

```powershell
pytest -q
```

Contribution guidelines live in [CONTRIBUTING.md](CONTRIBUTING.md). For larger changes, opening an issue before implementation is recommended.

## Repository Hygiene

This repository is intentionally kept lightweight:

- No local virtual environments or package caches
- No private `.env` values
- No runtime logs
- No personal study notes or learning journals
- No generated vector stores, memory dumps, or local experiment residue

## Roadmap

- Streaming responses in the chat UI
- Conversation history and session management
- Richer artifact previews before download
- Expanded evaluation coverage and automation
- More guarded tools for research-heavy workflows

## License

This project is released under the [MIT License](LICENSE).
