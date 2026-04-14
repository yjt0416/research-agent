import sys
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from app.agent import route_message  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import AgentChatResponse, ArtifactItem, ChatMessage, ConfirmationItem  # noqa: E402


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_eval_dataset_summary_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.summarize_eval_dataset",
        lambda: {
            "dataset_path": "data/evals/day5_eval_dataset.jsonl",
            "count": 3,
            "route_counts": {"chat": 1, "tool": 2},
            "samples": [{"id": "chat-001"}],
        },
    )

    response = client.get("/evaluation/dataset")

    assert response.status_code == 200
    assert response.json()["count"] == 3
    assert response.json()["route_counts"] == {"chat": 1, "tool": 2}


def test_chat_endpoint(monkeypatch) -> None:
    def fake_chat_with_deepseek(user_message: str, history: list) -> tuple[str, str]:
        assert user_message == "你好"
        assert history == []
        return "你好，我是测试回复。", "deepseek-chat"

    monkeypatch.setattr("app.main.chat_with_deepseek", fake_chat_with_deepseek)

    response = client.post("/chat", json={"message": "你好", "history": []})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "你好，我是测试回复。",
        "model": "deepseek-chat",
        "artifacts": [],
    }


def test_upload_document_endpoint(monkeypatch, tmp_path) -> None:
    saved_file = tmp_path / "sample.txt"

    def fake_save_uploaded_file(filename: str, content: bytes) -> Path:
        saved_file.write_bytes(content)
        return saved_file

    def fake_ingest_document(file_path: Path) -> tuple[str, int]:
        assert file_path == saved_file
        return "sample", 3

    monkeypatch.setattr("app.main.save_uploaded_file", fake_save_uploaded_file)
    monkeypatch.setattr("app.main.ingest_document", fake_ingest_document)

    response = client.post(
        "/documents/upload",
        files={"file": ("sample.txt", b"RAG is useful for grounded answers.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "sample",
        "filename": "sample.txt",
        "chunk_count": 3,
    }


def test_rag_chat_endpoint(monkeypatch) -> None:
    def fake_answer_with_rag(user_message: str, history: list) -> tuple[str, str, list[dict]]:
        assert user_message == "RAG 有什么作用？"
        assert history == []
        return (
            "RAG 可以先检索资料，再结合资料生成回答。[sample.txt#chunk-0]",
            "deepseek-chat",
            [
                {
                    "source_id": "sample:0",
                    "filename": "sample.txt",
                    "chunk_index": 0,
                    "preview": "RAG 是 Retrieval-Augmented Generation 的缩写。",
                }
            ],
        )

    monkeypatch.setattr("app.main.answer_with_rag", fake_answer_with_rag)
    monkeypatch.setattr(
        "app.main.build_markdown_artifact",
        lambda answer, base_name, session_id: ArtifactItem(
            artifact_id="artifact-rag",
            filename="rag_answer.md",
            media_type="text/markdown",
            kind="markdown",
            size_bytes=128,
            download_url="/artifacts/artifact-rag/download",
        ),
    )

    response = client.post("/chat/rag", json={"message": "RAG 有什么作用？", "history": []})

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["filename"] == "rag_answer.md"


def test_agent_route_message() -> None:
    assert route_message("请根据文档回答这个问题") == "rag"
    assert route_message("请帮我生成实验报告") == "report"
    assert route_message("请读取文件 `README.md`") == "tool"
    assert route_message("你好，介绍一下自己") == "chat"


def test_agent_chat_completed_endpoint(monkeypatch) -> None:
    def fake_run_agent_chat(
        message: str,
        session_id: str,
        user_id: str | None,
        mode: str,
        require_confirmation: bool,
    ) -> AgentChatResponse:
        assert require_confirmation is False
        return AgentChatResponse(
            answer="这是总结结果。",
            model="deepseek-chat",
            route="report",
            session_id=session_id,
            sources=[],
            short_term_memory_size=2,
            applied_preferences={"style": "简洁"},
            artifacts=[],
            status="completed",
            confirmation=None,
        )

    monkeypatch.setattr("app.main.run_agent_chat", fake_run_agent_chat)

    response = client.post(
        "/agent/chat",
        json={
            "message": "请根据文档总结重点",
            "session_id": "session-1",
            "user_id": "user-1",
            "mode": "auto",
            "require_confirmation": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_agent_chat_awaiting_confirmation_endpoint(monkeypatch) -> None:
    def fake_run_agent_chat(
        message: str,
        session_id: str,
        user_id: str | None,
        mode: str,
        require_confirmation: bool,
    ) -> AgentChatResponse:
        assert require_confirmation is True
        return AgentChatResponse(
            answer="这一步已经暂停，等待你人工确认后再执行。",
            model="deepseek-chat",
            route="tool",
            session_id=session_id,
            sources=[],
            short_term_memory_size=0,
            applied_preferences={},
            artifacts=[],
            status="awaiting_confirmation",
            confirmation=ConfirmationItem(
                token="confirm-123",
                route="tool",
                action_type="python",
                summary="Run restricted Python code.",
                session_id=session_id,
                user_id=user_id,
                expires_at="2099-01-01T00:00:00+00:00",
            ),
        )

    monkeypatch.setattr("app.main.run_agent_chat", fake_run_agent_chat)

    response = client.post(
        "/agent/chat",
        json={
            "message": "请运行 Python 代码",
            "session_id": "session-2",
            "user_id": "user-2",
            "mode": "tool",
            "require_confirmation": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "awaiting_confirmation"
    assert response.json()["confirmation"]["token"] == "confirm-123"


def test_agent_confirmation_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.confirm_agent_action",
        lambda token, action: AgentChatResponse(
            answer="已批准并执行完成。",
            model="deepseek-chat",
            route="tool",
            session_id="session-2",
            sources=[],
            short_term_memory_size=2,
            applied_preferences={},
            artifacts=[],
            status="completed",
            confirmation=None,
        ),
    )

    response = client.post("/agent/confirm/confirm-123", json={"action": "approve"})

    assert response.status_code == 200
    assert response.json()["answer"] == "已批准并执行完成。"


def test_get_session_memory_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.get_session_history",
        lambda session_id: [
            ChatMessage(role="user", content="你好"),
            ChatMessage(role="assistant", content="你好，我在。"),
        ],
    )

    response = client.get("/memory/session/session-1")

    assert response.status_code == 200
    assert response.json()["history"][0]["content"] == "你好"


def test_preference_endpoints(monkeypatch) -> None:
    monkeypatch.setattr("app.main.save_user_preferences", lambda user_id, preferences: preferences)
    monkeypatch.setattr(
        "app.main.get_user_preferences",
        lambda user_id: {"style": "简洁", "language": "中文"},
    )

    put_response = client.put(
        "/memory/preferences/user-1",
        json={"preferences": {"style": "简洁", "language": "中文"}},
    )
    get_response = client.get("/memory/preferences/user-1")

    assert put_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json()["preferences"]["language"] == "中文"


def test_read_file_tool_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.read_file_tool", lambda path: "# Demo file content")

    response = client.post("/tools/read-file", json={"path": "README.md"})

    assert response.status_code == 200
    assert response.json() == {
        "path": "README.md",
        "content": "# Demo file content",
    }


def test_python_tool_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.python_tool", lambda code: "16")
    monkeypatch.setattr(
        "app.main.build_python_tool_artifacts",
        lambda code, output, session_id: [
            ArtifactItem(
                artifact_id="artifact-py",
                filename="python_tool.py",
                media_type="text/x-python",
                kind="python",
                size_bytes=24,
                download_url="/artifacts/artifact-py/download",
            )
        ],
    )

    response = client.post("/tools/python", json={"code": "print(4 * 4)"})

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["filename"] == "python_tool.py"


def test_artifact_download(monkeypatch, tmp_path) -> None:
    artifact_file = tmp_path / "answer.md"
    artifact_file.write_text("# Demo", encoding="utf-8")

    monkeypatch.setattr(
        "app.main.resolve_artifact",
        lambda artifact_id: (
            artifact_file,
            ArtifactItem(
                artifact_id=artifact_id,
                filename="answer.md",
                media_type="text/markdown",
                kind="markdown",
                size_bytes=artifact_file.stat().st_size,
                download_url=f"/artifacts/{artifact_id}/download",
            ),
        ),
    )

    response = client.get("/artifacts/demo/download")

    assert response.status_code == 200
    assert response.content == b"# Demo"


def test_demo_page_served() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Research Agent Copilot" in response.text
