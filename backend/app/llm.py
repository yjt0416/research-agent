from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek

from app.config import get_settings
from app.logging_utils import get_logger
from app.prompts import CHAT_PROMPT_TEMPLATE
from app.schemas import ChatMessage


logger = get_logger("research_agent.llm")


@lru_cache
def get_chat_model() -> ChatDeepSeek:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in environment.")

    return ChatDeepSeek(
        model=settings.model_name,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )


def _stringify_message_content(content: str | list | None) -> str:
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


def history_to_messages(history: list[ChatMessage]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        if item.role == "system":
            messages.append(SystemMessage(content=item.content))
        elif item.role == "assistant":
            messages.append(AIMessage(content=item.content))
        else:
            messages.append(HumanMessage(content=item.content))
    return messages


def dict_messages_to_langchain(messages: list[dict[str, str]]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for item in messages:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def build_messages(user_message: str, history: list[ChatMessage]) -> list[dict[str, str]]:
    settings = get_settings()
    system_prompt = settings.system_prompt or CHAT_PROMPT_TEMPLATE
    if system_prompt == settings.system_prompt:
        system_prompt = f"{CHAT_PROMPT_TEMPLATE}\n\nProject System Prompt:\n{settings.system_prompt}"
    messages = [{"role": "system", "content": system_prompt}]

    for item in history:
        messages.append({"role": item.role, "content": item.content})

    messages.append({"role": "user", "content": user_message})
    return messages


def invoke_with_retry(
    operation_name: str,
    callback: Callable[[], Any],
) -> Any:
    settings = get_settings()
    attempts = max(1, settings.llm_max_retries)
    delay_seconds = max(0.1, settings.llm_retry_backoff_seconds)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            logger.info("Starting %s attempt=%s/%s", operation_name, attempt, attempts)
            return callback()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Operation failed: %s attempt=%s/%s error=%s",
                operation_name,
                attempt,
                attempts,
                exc,
            )
            if attempt == attempts:
                break
            time.sleep(delay_seconds * attempt)

    assert last_error is not None
    logger.error("Operation exhausted retries: %s error=%s", operation_name, last_error)
    raise last_error


def create_chat_completion(messages: list[dict[str, str]]) -> tuple[str, str]:
    response = invoke_with_retry(
        "chat_completion",
        lambda: get_chat_model().invoke(dict_messages_to_langchain(messages)),
    )
    answer = _stringify_message_content(response.content).strip()
    if not answer:
        raise RuntimeError("Model returned an empty response.")
    return answer, get_settings().model_name


def build_chat_chain(system_prompt: str):
    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{message}"),
            ]
        )
        | get_chat_model()
        | StrOutputParser()
    )


def chat_with_deepseek(user_message: str, history: list[ChatMessage]) -> tuple[str, str]:
    settings = get_settings()
    system_prompt = settings.system_prompt or CHAT_PROMPT_TEMPLATE
    if system_prompt == settings.system_prompt:
        system_prompt = f"{CHAT_PROMPT_TEMPLATE}\n\nProject System Prompt:\n{settings.system_prompt}"

    chain = build_chat_chain(system_prompt)
    answer = invoke_with_retry(
        "chat_chain",
        lambda: chain.invoke(
            {
                "history": history_to_messages(history),
                "message": user_message,
            }
        ),
    )
    return answer, settings.model_name
