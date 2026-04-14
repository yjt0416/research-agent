from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.schemas import ChatMessage


def _ensure_memory_dir() -> Path:
    settings = get_settings()
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    return settings.memory_dir


def _session_file() -> Path:
    return _ensure_memory_dir() / "sessions.json"


def _preference_file() -> Path:
    return _ensure_memory_dir() / "preferences.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_session_history(session_id: str) -> list[ChatMessage]:
    sessions = _read_json(_session_file())
    raw_history = sessions.get(session_id, [])
    return [ChatMessage(**item) for item in raw_history]


def save_session_history(session_id: str, history: list[ChatMessage]) -> None:
    settings = get_settings()
    trimmed_history = history[-settings.session_history_limit :]
    sessions = _read_json(_session_file())
    sessions[session_id] = [item.model_dump() for item in trimmed_history]
    _write_json(_session_file(), sessions)


def append_session_turn(session_id: str, role: str, content: str) -> list[ChatMessage]:
    history = get_session_history(session_id)
    history.append(ChatMessage(role=role, content=content))
    save_session_history(session_id, history)
    return get_session_history(session_id)


def get_user_preferences(user_id: str | None) -> dict[str, str]:
    if not user_id:
        return {}
    preferences = _read_json(_preference_file())
    return preferences.get(user_id, {})


def save_user_preferences(user_id: str, preferences: dict[str, str]) -> dict[str, str]:
    all_preferences = _read_json(_preference_file())
    all_preferences[user_id] = preferences
    _write_json(_preference_file(), all_preferences)
    return preferences
