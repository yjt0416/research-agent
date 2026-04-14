from __future__ import annotations

from typing import Literal

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.artifacts import build_agent_artifacts
from app.config import get_settings
from app.confirmations import create_confirmation, delete_confirmation, get_confirmation
from app.llm import get_chat_model, history_to_messages, invoke_with_retry
from app.logging_utils import get_logger
from app.memory import append_session_turn, get_session_history, get_user_preferences
from app.prompts import build_mode_prompt, format_preference_block
from app.rag import retrieve_chunks
from app.schemas import AgentChatResponse, ConfirmationItem, SourceItem
from app.tools import parse_tool_intent, python_tool, read_file_tool


logger = get_logger("research_agent.agent")
AgentRoute = Literal["chat", "rag", "report", "tool"]


REPORT_KEYWORDS = (
    "报告",
    "总结",
    "汇报",
    "周报",
    "月报",
    "实验结论",
    "structured report",
    "report",
)

RAG_KEYWORDS = (
    "根据文档",
    "根据资料",
    "根据上传",
    "这份文档",
    "论文",
    "手册",
    "实验方案",
    "引用来源",
    "source",
)

TOOL_KEYWORDS = (
    "读取文件",
    "查看文件",
    "打开文件",
    "read file",
    "file content",
    "计算",
    "运行代码",
    "执行代码",
    "python",
)


class ReadFileToolInput(BaseModel):
    path: str = Field(..., description="A file path inside the current project workspace.")


class PythonToolInput(BaseModel):
    code: str = Field(..., description="A short Python snippet to execute safely.")


@tool("read_workspace_file", args_schema=ReadFileToolInput)
def read_workspace_file(path: str) -> str:
    """Read a text file from the current workspace."""
    return read_file_tool(path)


@tool("run_python_snippet", args_schema=PythonToolInput)
def run_python_snippet(code: str) -> str:
    """Run a short, restricted Python snippet for simple calculations or printed output."""
    return python_tool(code)


ROUTER = RunnableLambda(
    lambda payload: {
        **payload,
        "route": route_message(payload["message"], payload.get("mode", "auto")),
    }
)


def route_message(message: str, mode: str = "auto") -> AgentRoute:
    if mode in {"chat", "rag", "report", "tool"}:
        return mode

    normalized = message.lower()
    if any(keyword in normalized for keyword in TOOL_KEYWORDS):
        return "tool"
    if any(keyword in message for keyword in REPORT_KEYWORDS) or ("总结" in message and "文档" in message):
        return "report"
    if any(keyword in message for keyword in RAG_KEYWORDS):
        return "rag"
    if "上传" in message or "参考资料" in message:
        return "rag"
    return "chat"


def _stringify_content(content: str | list | None) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _build_chat_chain(preferences: dict[str, str]):
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        f"{build_mode_prompt('chat')}\n\n"
                        "Use the short-term memory and the user's stored preferences when relevant.\n"
                        f"User preferences:\n{format_preference_block(preferences)}"
                    ),
                ),
                MessagesPlaceholder("history"),
                ("human", "{message}"),
            ]
        )
        | get_chat_model()
        | StrOutputParser()
    )


def _build_rag_chain(route: AgentRoute, preferences: dict[str, str]):
    route_instruction = (
        "Produce a structured report with sections for Summary, Key Findings, and Next Steps. "
        "If you generate downloadable code or config files, wrap each file in a fenced block like "
        "```python filename=analysis.py```."
        if route == "report"
        else "Answer the user's question directly and cite sources in square brackets when using context."
    )

    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        f"{build_mode_prompt(route, extra_instructions=route_instruction)}\n\n"
                        "Use retrieved context when it is relevant.\n"
                        f"User preferences:\n{format_preference_block(preferences)}"
                    ),
                ),
                MessagesPlaceholder("history"),
                (
                    "human",
                    (
                        "Question:\n{message}\n\n"
                        "Retrieved Context:\n{context}\n\n"
                        "Please answer in Chinese when the user uses Chinese."
                    ),
                ),
            ]
        )
        | get_chat_model()
        | StrOutputParser()
    )


def _build_tool_result_chain(preferences: dict[str, str]):
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        f"{build_mode_prompt('tool')}\n\n"
                        "Explain tool results in a beginner-friendly way. "
                        "If the result came from Python code, briefly explain what the code did.\n"
                        f"User preferences:\n{format_preference_block(preferences)}"
                    ),
                ),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "User request:\n{message}\n\nTool result:\n{tool_result}",
                ),
            ]
        )
        | get_chat_model()
        | StrOutputParser()
    )


def _retrieve_context(message: str) -> tuple[list[SourceItem], str]:
    chunks = retrieve_chunks(message)
    source_items = [
        SourceItem(
            source_id=chunk.source_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            preview=chunk.content[:160].replace("\n", " "),
        )
        for chunk in chunks
    ]
    context_text = "\n\n".join(
        f"[{chunk.filename}#chunk-{chunk.chunk_index}]\n{chunk.content}" for chunk in chunks
    )
    return source_items, context_text


def _run_tool_agent(message: str, history, preferences: dict[str, str]) -> str:
    agent = create_agent(
        model=get_chat_model(),
        tools=[read_workspace_file, run_python_snippet],
        system_prompt=(
            f"{build_mode_prompt('tool')}\n\n"
            "When the user asks to read a file or execute Python, call the correct tool before answering. "
            "Do not invent tool outputs. If you include code intended for download, use fenced blocks with "
            "filename metadata like ```python filename=script.py```.\n"
            f"User preferences:\n{format_preference_block(preferences)}"
        ),
    )

    messages = [{"role": item.role, "content": item.content} for item in history]
    messages.append({"role": "user", "content": message})
    result = invoke_with_retry("agent_tool_route", lambda: agent.invoke({"messages": messages}))
    return _stringify_content(result["messages"][-1].content).strip()


def _create_confirmation_response(
    *,
    session_id: str,
    preferences: dict[str, str],
    route: AgentRoute,
    confirmation: ConfirmationItem,
) -> AgentChatResponse:
    return AgentChatResponse(
        answer="这一步已经暂停，等待你人工确认后再执行。",
        model=get_settings().model_name,
        route=route,
        session_id=session_id,
        sources=[],
        short_term_memory_size=len(get_session_history(session_id)),
        applied_preferences=preferences,
        artifacts=[],
        status="awaiting_confirmation",
        confirmation=confirmation,
    )


def _execute_confirmed_tool_action(record: dict) -> AgentChatResponse:
    session_id = str(record["session_id"])
    user_id = record.get("user_id")
    original_message = str(record["original_message"])
    action_type = str(record["action_type"])
    payload = str(record["payload"])
    history = get_session_history(session_id)
    preferences = get_user_preferences(user_id)

    if action_type == "read_file":
        tool_result = f"Read file: {payload}\n\n{read_file_tool(payload)}"
    elif action_type == "python":
        tool_output = python_tool(payload)
        tool_result = f"Python input:\n{payload}\n\nTool output:\n{tool_output}"
    else:
        raise ValueError(f"Unsupported confirmation action: {action_type}")

    answer = invoke_with_retry(
        "agent_confirmed_tool_explanation",
        lambda: _build_tool_result_chain(preferences).invoke(
            {
                "history": history_to_messages(history),
                "message": original_message,
                "tool_result": tool_result,
            }
        ),
    )
    updated_history = append_session_turn(session_id=session_id, role="user", content=original_message)
    updated_history = append_session_turn(session_id=session_id, role="assistant", content=answer)
    artifacts = build_agent_artifacts(
        answer=answer,
        route="tool",
        session_id=session_id,
        user_message=original_message,
    )
    delete_confirmation(str(record["token"]))

    return AgentChatResponse(
        answer=answer,
        model=get_settings().model_name,
        route="tool",
        session_id=session_id,
        sources=[],
        short_term_memory_size=len(updated_history),
        applied_preferences=preferences,
        artifacts=artifacts,
        status="completed",
        confirmation=None,
    )


def confirm_agent_action(token: str, action: str) -> AgentChatResponse:
    record = get_confirmation(token)
    if not record:
        raise ValueError("Confirmation token not found or expired.")

    if action == "cancel":
        delete_confirmation(token)
        session_id = str(record["session_id"])
        preferences = get_user_preferences(record.get("user_id"))
        return AgentChatResponse(
            answer="这次工具执行已取消，没有对项目状态做任何修改。",
            model=get_settings().model_name,
            route="tool",
            session_id=session_id,
            sources=[],
            short_term_memory_size=len(get_session_history(session_id)),
            applied_preferences=preferences,
            artifacts=[],
            status="cancelled",
            confirmation=None,
        )

    return _execute_confirmed_tool_action(record)


def run_agent_chat(
    message: str,
    session_id: str,
    user_id: str | None = None,
    mode: str = "auto",
    require_confirmation: bool = False,
) -> AgentChatResponse:
    payload = ROUTER.invoke({"message": message, "mode": mode})
    route: AgentRoute = payload["route"]
    history = get_session_history(session_id)
    preferences = get_user_preferences(user_id)
    sources: list[SourceItem] = []

    logger.info(
        "Agent request route=%s session_id=%s user_id=%s require_confirmation=%s",
        route,
        session_id,
        user_id,
        require_confirmation,
    )

    if route == "tool" and require_confirmation:
        intent = parse_tool_intent(message)
        if intent is None:
            raise ValueError("Current message could not be converted into a confirmable tool action.")
        confirmation = create_confirmation(
            session_id=session_id,
            user_id=user_id,
            route="tool",
            action_type=intent.action_type,
            summary=intent.summary,
            payload=intent.payload,
            original_message=message,
        )
        return _create_confirmation_response(
            session_id=session_id,
            preferences=preferences,
            route=route,
            confirmation=confirmation,
        )

    if route in {"rag", "report"}:
        sources, context_text = _retrieve_context(message)
        if sources:
            answer = invoke_with_retry(
                f"agent_{route}_chain",
                lambda: _build_rag_chain(route, preferences).invoke(
                    {
                        "history": history_to_messages(history),
                        "message": message,
                        "context": context_text,
                    }
                ),
            )
        else:
            route = "chat"
            answer = invoke_with_retry(
                "agent_fallback_chat_chain",
                lambda: _build_chat_chain(preferences).invoke(
                    {
                        "history": history_to_messages(history),
                        "message": message,
                    }
                ),
            )
    elif route == "tool":
        answer = _run_tool_agent(message, history, preferences)
    else:
        answer = invoke_with_retry(
            "agent_chat_chain",
            lambda: _build_chat_chain(preferences).invoke(
                {
                    "history": history_to_messages(history),
                    "message": message,
                }
            ),
        )

    updated_history = append_session_turn(session_id=session_id, role="user", content=message)
    updated_history = append_session_turn(session_id=session_id, role="assistant", content=answer)
    artifacts = build_agent_artifacts(
        answer=answer,
        route=route,
        session_id=session_id,
        user_message=message,
    )

    return AgentChatResponse(
        answer=answer,
        model=get_settings().model_name,
        route=route,
        session_id=session_id,
        sources=sources if route in {"rag", "report"} else [],
        short_term_memory_size=len(updated_history),
        applied_preferences=preferences,
        artifacts=artifacts,
        status="completed",
        confirmation=None,
    )
