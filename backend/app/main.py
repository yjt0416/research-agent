from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.agent import confirm_agent_action, run_agent_chat
from app.artifacts import build_markdown_artifact, build_python_tool_artifacts, resolve_artifact
from app.evaluation import summarize_eval_dataset
from app.llm import chat_with_deepseek
from app.logging_utils import configure_logging, get_logger
from app.memory import get_session_history, get_user_preferences, save_user_preferences
from app.rag import answer_with_rag, ingest_document, save_uploaded_file
from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ChatRequest,
    ChatResponse,
    ConfirmationDecisionRequest,
    DocumentUploadResponse,
    EvalDatasetResponse,
    FileReadRequest,
    FileReadResponse,
    PreferenceResponse,
    PreferenceUpdateRequest,
    PythonToolRequest,
    PythonToolResponse,
    RagChatResponse,
    SessionMemoryResponse,
)
from app.tools import python_tool, read_file_tool


configure_logging()
logger = get_logger("research_agent.api")
app = FastAPI(title="Research Agent Copilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    request_id = uuid4().hex[:10]
    started_at = time.perf_counter()
    logger.info("request.start id=%s method=%s path=%s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.exception(
            "request.error id=%s method=%s path=%s duration_ms=%s error=%s",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
            exc,
        )
        raise

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "request.end id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def demo_page() -> FileResponse:
    """Serve the simple frontend demo page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint for Day 1/Day 2 verification."""
    return {"status": "ok"}


@app.get("/evaluation/dataset", response_model=EvalDatasetResponse)
def get_eval_dataset_summary() -> EvalDatasetResponse:
    """Read the built-in evaluation dataset summary."""
    return EvalDatasetResponse(**summarize_eval_dataset())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Minimal chat endpoint for Day 2."""
    try:
        answer, model = chat_with_deepseek(
            user_message=request.message,
            history=request.history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    return ChatResponse(answer=answer, model=model, artifacts=[])


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a document and ingest it into the local vector store."""
    try:
        content = await file.read()
        saved_path = save_uploaded_file(file.filename or "upload.txt", content)
        document_id, chunk_count = ingest_document(saved_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc

    return DocumentUploadResponse(
        document_id=document_id,
        filename=saved_path.name,
        chunk_count=chunk_count,
    )


@app.post("/chat/rag", response_model=RagChatResponse)
def rag_chat(request: ChatRequest) -> RagChatResponse:
    """Answer a question using retrieved document chunks."""
    try:
        answer, model, sources = answer_with_rag(
            user_message=request.message,
            history=request.history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG request failed: {exc}") from exc

    artifact = build_markdown_artifact(
        answer,
        base_name="rag_answer",
        session_id="direct-rag",
    )
    return RagChatResponse(answer=answer, model=model, sources=sources, artifacts=[artifact])


@app.post("/tools/read-file", response_model=FileReadResponse)
def read_file(request: FileReadRequest) -> FileReadResponse:
    """Read a text file from the current workspace."""
    try:
        content = read_file_tool(request.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileReadResponse(path=request.path, content=content)


@app.post("/tools/python", response_model=PythonToolResponse)
def run_python_tool(request: PythonToolRequest) -> PythonToolResponse:
    """Run a restricted Python snippet for demo calculations."""
    try:
        output = python_tool(request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    artifacts = build_python_tool_artifacts(
        code=request.code,
        output=output,
        session_id="python-tool",
    )
    return PythonToolResponse(code=request.code, output=output, artifacts=artifacts)


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """Unified agent endpoint with routing, memory, and optional human confirmation."""
    try:
        return run_agent_chat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            mode=request.mode,
            require_confirmation=request.require_confirmation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent request failed: {exc}") from exc


@app.post("/agent/confirm/{token}", response_model=AgentChatResponse)
def confirm_agent(token: str, request: ConfirmationDecisionRequest) -> AgentChatResponse:
    """Approve or cancel a pending agent tool action."""
    try:
        return confirm_agent_action(token, request.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Confirmation request failed: {exc}") from exc


@app.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str) -> FileResponse:
    """Download a generated artifact file."""
    try:
        path, artifact = resolve_artifact(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(path=path, media_type=artifact.media_type, filename=artifact.filename)


@app.get("/memory/session/{session_id}", response_model=SessionMemoryResponse)
def get_session_memory(session_id: str) -> SessionMemoryResponse:
    """Read short-term memory for a session."""
    return SessionMemoryResponse(session_id=session_id, history=get_session_history(session_id))


@app.get("/memory/preferences/{user_id}", response_model=PreferenceResponse)
def get_preferences(user_id: str) -> PreferenceResponse:
    """Read long-term user preferences."""
    return PreferenceResponse(user_id=user_id, preferences=get_user_preferences(user_id))


@app.put("/memory/preferences/{user_id}", response_model=PreferenceResponse)
def update_preferences(user_id: str, request: PreferenceUpdateRequest) -> PreferenceResponse:
    """Update long-term user preferences."""
    preferences = save_user_preferences(user_id, request.preferences)
    return PreferenceResponse(user_id=user_id, preferences=preferences)
