import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "deepseek-v4-pro"))
    system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "SYSTEM_PROMPT",
            "You are Research Agent Copilot, a helpful assistant for research and technical documents.",
        )
    )
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    chroma_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "chroma")
    vector_store_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "langchain_chroma"
    )
    memory_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "memory")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "logs")
    artifacts_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "artifacts"
    )
    evals_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "evals")
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "research_docs_semantic_v1")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4")))
    session_history_limit: int = field(
        default_factory=lambda: int(os.getenv("SESSION_HISTORY_LIMIT", "8"))
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    embedding_normalize: bool = field(
        default_factory=lambda: os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
    )
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")))
    llm_retry_backoff_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "1.5"))
    )
    confirmation_ttl_minutes: int = field(
        default_factory=lambda: int(os.getenv("CONFIRMATION_TTL_MINUTES", "30"))
    )
    graph_reflection_limit: int = field(
        default_factory=lambda: int(os.getenv("GRAPH_REFLECTION_LIMIT", "1"))
    )
    research_plot_dpi: int = field(default_factory=lambda: int(os.getenv("RESEARCH_PLOT_DPI", "180")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
