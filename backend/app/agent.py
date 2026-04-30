from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.artifacts import build_agent_artifacts, build_python_tool_artifacts
from app.config import get_settings
from app.confirmations import create_confirmation, delete_confirmation, get_confirmation
from app.llm import get_chat_model, history_to_messages, invoke_with_retry
from app.logging_utils import get_logger
from app.memory import append_session_turn, get_session_history, get_user_preferences
from app.prompts import RESEARCH_PLANNER_PROMPT, build_mode_prompt, format_preference_block
from app.rag import ingest_existing_document, retrieve_chunks, retrieve_chunks_for_queries
from app.reproduction import looks_like_alpha_vlf_paper, run_alpha_stable_vlf_reproduction
from app.schemas import (
    AgentChatResponse,
    ConfirmationItem,
    IngestedDocumentItem,
    PlanStep,
    SourceItem,
    WorkflowTraceItem,
)
from app.tools import parse_tool_intent, python_tool, read_file_tool


logger = get_logger("research_agent.agent")
AgentRoute = Literal["chat", "rag", "report", "tool", "research"]


REPORT_KEYWORDS = (
    "报告",
    "总结",
    "汇报",
    "实验结论",
    "技术文档",
    "structured report",
    "report",
    "markdown",
    "鎶ュ憡",
    "鎬荤粨",
    "姹囨姤",
    "瀹為獙缁撹",
)

RAG_KEYWORDS = (
    "根据文档",
    "根据资料",
    "根据上传",
    "这份文档",
    "论文",
    "文献",
    "引用来源",
    "source",
    "鏍规嵁鏂囨。",
    "鏍规嵁璧勬枡",
    "鏍规嵁涓婁紶",
    "璁烘枃",
    "寮曠敤鏉ユ簮",
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
    "璇诲彇鏂囦欢",
    "鏌ョ湅鏂囦欢",
    "鎵撳紑鏂囦欢",
    "璁＄畻",
    "杩愯浠ｇ爜",
    "鎵ц浠ｇ爜",
)

RESEARCH_KEYWORDS = (
    "复现",
    "论文复现",
    "实验复现",
    "科研",
    "仿真",
    "simulation",
    "reproduce",
    "benchmark",
    "代码复现",
    "复现实验",
    "浠跨湡",
    "澶嶇幇",
)


class ReadFileToolInput(BaseModel):
    path: str = Field(..., description="A file path inside the current project workspace.")


class PythonToolInput(BaseModel):
    code: str = Field(..., description="A short Python snippet to execute safely.")


class AgentState(TypedDict, total=False):
    message: str
    session_id: str
    user_id: str | None
    mode: str
    document_paths: list[str]
    route: AgentRoute
    history: list[Any]
    preferences: dict[str, str]
    retry_count: int
    workflow_trace: list[WorkflowTraceItem]
    plan_steps: list[PlanStep]
    planner_queries: list[str]
    needs_code: bool
    needs_report: bool
    ingested_documents: list[IngestedDocumentItem]
    sources: list[SourceItem]
    retrieved_context: str
    tool_action: str
    tool_payload: str
    artifacts: list[Any]
    tool_result: str
    code_execution_success: bool
    code_execution_summary: str
    code_execution_report_text: str
    code_execution_metrics: dict[str, Any]
    reflection_note: str
    should_retry: bool
    answer: str


def route_message(message: str, mode: str = "auto") -> AgentRoute:
    if mode in {"chat", "rag", "report", "tool", "research"}:
        return mode  # type: ignore[return-value]

    lowered = message.lower()
    if any(keyword in message for keyword in RESEARCH_KEYWORDS) or (
        ("论文" in message or "文献" in message or "paper" in lowered or "璁烘枃" in message)
        and ("复现" in message or "代码" in message or "实验" in message or "reproduce" in lowered or "澶嶇幇" in message)
    ):
        return "research"
    if any(keyword in message for keyword in TOOL_KEYWORDS) or any(keyword in lowered for keyword in TOOL_KEYWORDS):
        return "tool"
    if any(keyword in message for keyword in REPORT_KEYWORDS):
        return "report"
    if any(keyword in message for keyword in RAG_KEYWORDS) or "上传" in message or "参考资料" in message or "涓婁紶" in message:
        return "rag"
    return "chat"


def _append_trace(state: AgentState, node: str, summary: str) -> list[WorkflowTraceItem]:
    trace = list(state.get("workflow_trace", []))
    trace.append(WorkflowTraceItem(node=node, summary=summary))
    return trace


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


def _build_research_planner_chain(preferences: dict[str, str]):
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", f"{RESEARCH_PLANNER_PROMPT}\n\nUser preferences:\n{format_preference_block(preferences)}"),
                (
                    "human",
                    (
                        "Task:\n{message}\n\n"
                        "Document paths:\n{document_paths}\n\n"
                        "Conversation history:\n{history_summary}\n\n"
                        "Return JSON only."
                    ),
                ),
            ]
        )
        | get_chat_model()
        | StrOutputParser()
    )


def _history_summary(history: list[Any]) -> str:
    if not history:
        return "No prior conversation."
    lines = [f"- {item.role}: {item.content[:120]}" for item in history[-6:]]
    return "\n".join(lines)


def _fallback_plan(route: AgentRoute, message: str, document_paths: list[str]) -> dict[str, object]:
    if route == "research":
        return {
            "objective": "完成论文阅读、实验复现与技术文档输出。",
            "queries": [
                message,
                "Alpha 稳定分布 参数 特征函数 pdf 公式",
                "MSK 采样率 载波频率 码率 误码率 仿真",
            ],
            "steps": [
                {"title": "导入论文", "detail": "接入本地 PDF 并写入向量库。"},
                {"title": "抽取关键参数", "detail": "定位 Alpha 稳定分布和 MSK 仿真的系统参数与目标曲线。"},
                {"title": "生成复现代码", "detail": "生成可下载的 Python 仿真脚本。"},
                {"title": "运行仿真", "detail": "输出 PDF 曲线、噪声波形、BER 曲线和指标文件。"},
                {"title": "撰写技术文档", "detail": "汇总假设、结果和产物下载说明。"},
            ],
            "needs_code": True,
            "needs_report": True,
        }
    if route == "report":
        return {
            "objective": "生成结构化技术报告。",
            "queries": [message],
            "steps": [
                {"title": "检索资料", "detail": "召回与问题最相关的本地文档片段。"},
                {"title": "整理要点", "detail": "压缩上下文并形成分节报告。"},
            ],
            "needs_code": False,
            "needs_report": True,
        }
    if route == "rag":
        return {
            "objective": "基于文档进行可溯源问答。",
            "queries": [message],
            "steps": [{"title": "检索资料", "detail": "召回最相关的 Top-k 文档片段。"}],
            "needs_code": False,
            "needs_report": False,
        }
    if route == "tool":
        return {
            "objective": "执行工具调用。",
            "queries": [],
            "steps": [{"title": "识别工具意图", "detail": "决定读取文件还是运行 Python。"}],
            "needs_code": True,
            "needs_report": False,
        }
    return {
        "objective": "继续对话。",
        "queries": [],
        "steps": [{"title": "回答问题", "detail": "结合上下文自然回应。"}],
        "needs_code": False,
        "needs_report": False,
    }


def _parse_plan_output(route: AgentRoute, text: str, message: str, document_paths: list[str]) -> dict[str, object]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        logger.warning("Planner JSON parsing failed, using fallback.")
    return _fallback_plan(route, message, document_paths)


def _intent_router_node(state: AgentState) -> AgentState:
    route = route_message(state["message"], state.get("mode", "auto"))
    history = get_session_history(state["session_id"])
    preferences = get_user_preferences(state.get("user_id"))
    summary = (
        f"Routed request to {route} with {len(history)} history turns and "
        f"{len(state.get('document_paths', []))} explicit documents."
    )
    return {
        "route": route,
        "history": history,
        "preferences": preferences,
        "retry_count": int(state.get("retry_count", 0)),
        "workflow_trace": _append_trace(state, "intent_router", summary),
    }


def _planner_node(state: AgentState) -> AgentState:
    route: AgentRoute = state["route"]
    preferences = state["preferences"]
    history = state["history"]
    document_paths = state.get("document_paths", [])

    if route in {"chat", "tool"}:
        raw_plan = _fallback_plan(route, state["message"], document_paths)
    else:
        raw_output = invoke_with_retry(
            "research_planner",
            lambda: _build_research_planner_chain(preferences).invoke(
                {
                    "message": state["message"],
                    "document_paths": "\n".join(document_paths) or "No explicit document path provided.",
                    "history_summary": _history_summary(history),
                }
            ),
        )
        raw_plan = _parse_plan_output(route, raw_output, state["message"], document_paths)

    plan_steps = [
        PlanStep(title=item.get("title", "Unnamed step"), detail=item.get("detail", ""), status="pending")
        for item in raw_plan.get("steps", [])
        if isinstance(item, dict)
    ]
    summary = f"Built {len(plan_steps)} plan steps for route={route}."
    return {
        "plan_steps": plan_steps,
        "planner_queries": [query for query in raw_plan.get("queries", []) if isinstance(query, str)],
        "needs_code": bool(raw_plan.get("needs_code", False)),
        "needs_report": bool(raw_plan.get("needs_report", False)),
        "workflow_trace": _append_trace(state, "planner", summary),
    }


def _retriever_node(state: AgentState) -> AgentState:
    ingested_documents = list(state.get("ingested_documents", []))
    for document_path in state.get("document_paths", []):
        if any(item.source_path == document_path for item in ingested_documents):
            continue
        document_id, filename, chunk_count = ingest_existing_document(document_path)
        ingested_documents.append(
            IngestedDocumentItem(
                document_id=document_id,
                filename=filename,
                source_path=document_path,
                chunk_count=chunk_count,
            )
        )

    queries = state.get("planner_queries") or [state["message"]]
    retrieved_chunks = retrieve_chunks_for_queries(queries) if queries else retrieve_chunks(state["message"])
    source_items = [
        SourceItem(
            source_id=chunk.source_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            preview=chunk.content[:160].replace("\n", " "),
        )
        for chunk in retrieved_chunks
    ]
    context_text = "\n\n".join(
        f"[{chunk.filename}#chunk-{chunk.chunk_index}]\n{chunk.content}" for chunk in retrieved_chunks
    )
    summary = f"Ingested {len(ingested_documents)} docs and retrieved {len(source_items)} chunks."
    return {
        "ingested_documents": ingested_documents,
        "sources": source_items,
        "retrieved_context": context_text,
        "workflow_trace": _append_trace(state, "retriever", summary),
    }


def _tool_router_node(state: AgentState) -> AgentState:
    route: AgentRoute = state["route"]
    action = "none"
    payload = ""

    if route == "tool":
        intent = parse_tool_intent(state["message"])
        if intent:
            action = intent.action_type
            payload = intent.payload
    elif route == "research" and state.get("needs_code"):
        retrieved_context = state.get("retrieved_context", "")
        if looks_like_alpha_vlf_paper(retrieved_context) or looks_like_alpha_vlf_paper(state["message"]):
            action = "paper_reproduction"
            if state.get("document_paths"):
                payload = state["document_paths"][0]
            elif state.get("ingested_documents"):
                payload = state["ingested_documents"][0].source_path

    summary = f"Selected tool action: {action}."
    return {
        "tool_action": action,
        "tool_payload": payload,
        "workflow_trace": _append_trace(state, "tool_router", summary),
    }


def _code_executor_node(state: AgentState) -> AgentState:
    action = state.get("tool_action", "none")
    session_id = state["session_id"]
    artifacts = list(state.get("artifacts", []))

    if action == "read_file":
        tool_result = read_file_tool(state["tool_payload"])
        summary = "Read workspace file successfully."
        success = True
        report_text = ""
        metrics: dict[str, Any] = {}
        artifacts = []
    elif action == "python":
        tool_result = python_tool(state["tool_payload"])
        artifacts = build_python_tool_artifacts(
            code=state["tool_payload"],
            output=tool_result,
            session_id=session_id,
            prefix=f"agent_python_{session_id}",
        )
        summary = "Executed restricted Python successfully."
        success = True
        report_text = ""
        metrics = {}
    elif action == "paper_reproduction":
        paper_title = (
            state["ingested_documents"][0].filename
            if state.get("ingested_documents")
            else "Alpha-stable VLF paper"
        )
        result = run_alpha_stable_vlf_reproduction(
            session_id=session_id,
            source_path=state["tool_payload"],
            paper_title=paper_title,
        )
        artifacts = list(result.get("artifacts", []))
        summary = str(result.get("summary", "Reproduction finished."))
        tool_result = str(result.get("stdout", "")).strip()
        if result.get("stderr"):
            tool_result = f"{tool_result}\n\nSTDERR:\n{result['stderr']}".strip()
        success = bool(result.get("success"))
        report_text = str(result.get("report_text", ""))
        metrics = result.get("metrics", {}) if isinstance(result.get("metrics", {}), dict) else {}
    else:
        summary = "No tool execution required."
        tool_result = ""
        success = True
        report_text = ""
        metrics = {}

    return {
        "artifacts": artifacts,
        "tool_result": tool_result,
        "code_execution_success": success,
        "code_execution_summary": summary,
        "code_execution_report_text": report_text,
        "code_execution_metrics": metrics,
        "workflow_trace": _append_trace(state, "code_executor", summary),
    }


def _reflector_node(state: AgentState) -> AgentState:
    success = bool(state.get("code_execution_success", True))
    retry_count = int(state.get("retry_count", 0))
    limit = get_settings().graph_reflection_limit

    if success:
        note = "Execution completed successfully; no retry required."
        should_retry = False
    elif retry_count < limit:
        note = "Execution failed once; scheduling one reflection-driven retry."
        should_retry = True
        retry_count += 1
    else:
        note = "Execution still failed after reflection limit; proceeding to report the failure transparently."
        should_retry = False

    return {
        "reflection_note": note,
        "should_retry": should_retry,
        "retry_count": retry_count,
        "workflow_trace": _append_trace(state, "reflector", note),
    }


def _build_research_answer(state: AgentState, artifacts: list[Any]) -> str:
    metrics = state.get("code_execution_metrics", {})
    detected = metrics.get("detected_parameters", {}) if isinstance(metrics, dict) else {}
    observations = metrics.get("observations", []) if isinstance(metrics, dict) else []
    runtime = metrics.get("runtime_profile", {}) if isinstance(metrics, dict) else {}
    citations = " ".join(f"[{item.filename}#chunk-{item.chunk_index}]" for item in state.get("sources", [])[:4])
    generated_names = ", ".join(artifact.filename for artifact in artifacts) or "No artifacts"
    observation_lines = "\n".join(f"- {item}" for item in observations[:3]) or "- 暂无自动观察结论。"
    execution_summary = state.get("code_execution_summary", "")
    return (
        "已完成论文复现与技术文档输出。\n\n"
        f"1. 已导入论文并完成参数检索，核心上下文来自 {citations or '已导入论文片段'}。\n"
        f"2. 已生成并运行独立 Python 复现脚本，当前产物包括：{generated_names}。\n"
        f"3. 本次工程复现使用的关键参数为：采样率 {detected.get('sampling_rate_hz', 'unknown')} Hz，"
        f"载波 {detected.get('carrier_hz', 'unknown')} Hz，码率 {detected.get('bit_rate_bps', 'unknown')} b/s，"
        f"PDF 曲线 alpha={detected.get('pdf_alphas', [])}，BER 曲线 alpha={detected.get('ber_alphas', [])}。\n"
        f"4. 交互式运行配置为：每次 {runtime.get('bits_per_trial', 'unknown')} bit，"
        f"{runtime.get('monte_carlo_trials', 'unknown')} 次 Monte Carlo，SNR 点 {runtime.get('snr_db_values', [])}。\n"
        f"5. 执行结果摘要：{execution_summary}\n"
        f"6. 自动观察结论：\n{observation_lines}\n"
        "7. 说明：这是面向交互式科研 Agent 的工程复现版本，保留了论文核心趋势，并在技术文档中显式标注了近似假设。"
    )


def _reporter_node(state: AgentState) -> AgentState:
    route: AgentRoute = state["route"]
    history = state["history"]
    preferences = state["preferences"]
    context = state.get("retrieved_context", "")
    artifacts = list(state.get("artifacts", []))

    if route == "chat":
        answer = invoke_with_retry(
            "agent_chat_chain",
            lambda: _build_chat_chain(preferences).invoke(
                {"history": history_to_messages(history), "message": state["message"]}
            ),
        )
    elif route in {"rag", "report"}:
        if context:
            answer = invoke_with_retry(
                f"agent_{route}_chain",
                lambda: _build_rag_chain(route, preferences).invoke(
                    {
                        "history": history_to_messages(history),
                        "message": state["message"],
                        "context": context,
                    }
                ),
            )
        else:
            answer = invoke_with_retry(
                "agent_fallback_chat_chain",
                lambda: _build_chat_chain(preferences).invoke(
                    {"history": history_to_messages(history), "message": state["message"]}
                ),
            )
    elif route == "tool":
        answer = invoke_with_retry(
            "agent_tool_report_chain",
            lambda: _build_tool_result_chain(preferences).invoke(
                {
                    "history": history_to_messages(history),
                    "message": state["message"],
                    "tool_result": state.get("tool_result", "No tool output."),
                }
            ),
        )
    else:
        answer = _build_research_answer(state, artifacts)

    completed_steps = [
        PlanStep(title=step.title, detail=step.detail, status="completed")
        for step in state.get("plan_steps", [])
    ]

    if not artifacts:
        artifacts = build_agent_artifacts(
            answer=answer,
            route=route,
            session_id=state["session_id"],
            user_message=state["message"],
        )

    return {
        "answer": answer,
        "artifacts": artifacts,
        "plan_steps": completed_steps,
        "workflow_trace": _append_trace(state, "reporter", f"Prepared final answer for route={route}."),
    }


def _tool_route_decision(state: AgentState) -> str:
    if state.get("tool_action") and state.get("tool_action") != "none":
        return "execute"
    return "report"


def _reflect_decision(state: AgentState) -> str:
    if state.get("should_retry"):
        return "retry"
    return "report"


@lru_cache
def _compiled_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent_router", _intent_router_node)
    graph.add_node("planner", _planner_node)
    graph.add_node("retriever", _retriever_node)
    graph.add_node("tool_router", _tool_router_node)
    graph.add_node("code_executor", _code_executor_node)
    graph.add_node("reflector", _reflector_node)
    graph.add_node("reporter", _reporter_node)

    graph.add_edge(START, "intent_router")
    graph.add_edge("intent_router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "tool_router")
    graph.add_conditional_edges(
        "tool_router",
        _tool_route_decision,
        {"execute": "code_executor", "report": "reporter"},
    )
    graph.add_edge("code_executor", "reflector")
    graph.add_conditional_edges(
        "reflector",
        _reflect_decision,
        {"retry": "code_executor", "report": "reporter"},
    )
    graph.add_edge("reporter", END)
    return graph.compile()


def _create_confirmation_response(
    *,
    session_id: str,
    preferences: dict[str, str],
    route: AgentRoute,
    confirmation: ConfirmationItem,
) -> AgentChatResponse:
    return AgentChatResponse(
        answer="这一步已经暂停，等待你人工确认后再继续执行。",
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
        artifacts = []
    elif action_type == "python":
        tool_output = python_tool(payload)
        tool_result = f"Python input:\n{payload}\n\nTool output:\n{tool_output}"
        artifacts = build_python_tool_artifacts(
            code=payload,
            output=tool_output,
            session_id=session_id,
            prefix=f"agent_python_{session_id}",
        )
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
        workflow_trace=[WorkflowTraceItem(node="confirmation", summary="Executed user-approved tool action.")],
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
            workflow_trace=[WorkflowTraceItem(node="confirmation", summary="Cancelled pending tool action.")],
        )

    return _execute_confirmed_tool_action(record)


def run_agent_chat(
    message: str,
    session_id: str,
    user_id: str | None = None,
    mode: str = "auto",
    require_confirmation: bool = False,
    document_paths: list[str] | None = None,
) -> AgentChatResponse:
    route = route_message(message, mode)
    preferences = get_user_preferences(user_id)
    logger.info(
        "Agent request route=%s session_id=%s user_id=%s require_confirmation=%s document_paths=%s",
        route,
        session_id,
        user_id,
        require_confirmation,
        document_paths or [],
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

    final_state = _compiled_graph().invoke(
        AgentState(
            message=message,
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            document_paths=document_paths or [],
            workflow_trace=[],
            artifacts=[],
            sources=[],
            ingested_documents=[],
        )
    )

    updated_history = append_session_turn(session_id=session_id, role="user", content=message)
    updated_history = append_session_turn(session_id=session_id, role="assistant", content=final_state["answer"])

    return AgentChatResponse(
        answer=final_state["answer"],
        model=get_settings().model_name,
        route=final_state["route"],
        session_id=session_id,
        sources=final_state.get("sources", []) if final_state["route"] in {"rag", "report", "research"} else [],
        short_term_memory_size=len(updated_history),
        applied_preferences=final_state.get("preferences", preferences),
        artifacts=final_state.get("artifacts", []),
        status="completed",
        confirmation=None,
        plan_steps=final_state.get("plan_steps", []),
        workflow_trace=final_state.get("workflow_trace", []),
        ingested_documents=final_state.get("ingested_documents", []),
    )
