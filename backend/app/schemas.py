from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list)


class ArtifactItem(BaseModel):
    artifact_id: str
    filename: str
    media_type: str
    kind: str
    size_bytes: int
    download_url: str


class ChatResponse(BaseModel):
    answer: str
    model: str
    artifacts: list[ArtifactItem] = Field(default_factory=list)


class SessionMemoryResponse(BaseModel):
    session_id: str
    history: list[ChatMessage]


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class SourceItem(BaseModel):
    source_id: str
    filename: str
    chunk_index: int
    preview: str


class RagChatResponse(BaseModel):
    answer: str
    model: str
    sources: list[SourceItem]
    artifacts: list[ArtifactItem] = Field(default_factory=list)


class ConfirmationItem(BaseModel):
    token: str
    route: Literal["tool"]
    action_type: Literal["read_file", "python"]
    summary: str
    session_id: str
    user_id: str | None = None
    expires_at: str


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(..., min_length=1, max_length=100)
    user_id: str | None = Field(default=None, max_length=100)
    mode: Literal["auto", "chat", "rag", "report", "tool"] = "auto"
    require_confirmation: bool = False


class AgentChatResponse(BaseModel):
    answer: str
    model: str
    route: Literal["chat", "rag", "report", "tool"]
    session_id: str
    sources: list[SourceItem] = Field(default_factory=list)
    short_term_memory_size: int
    applied_preferences: dict[str, str] = Field(default_factory=dict)
    artifacts: list[ArtifactItem] = Field(default_factory=list)
    status: Literal["completed", "awaiting_confirmation", "cancelled"] = "completed"
    confirmation: ConfirmationItem | None = None


class PreferenceUpdateRequest(BaseModel):
    preferences: dict[str, str] = Field(default_factory=dict)


class PreferenceResponse(BaseModel):
    user_id: str
    preferences: dict[str, str]


class FileReadRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=300)


class FileReadResponse(BaseModel):
    path: str
    content: str


class PythonToolRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=4000)


class PythonToolResponse(BaseModel):
    code: str
    output: str
    artifacts: list[ArtifactItem] = Field(default_factory=list)


class ConfirmationDecisionRequest(BaseModel):
    action: Literal["approve", "cancel"]


class EvalDatasetResponse(BaseModel):
    dataset_path: str
    count: int
    route_counts: dict[str, int] = Field(default_factory=dict)
    samples: list[dict] = Field(default_factory=list)
