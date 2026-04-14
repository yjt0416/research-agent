from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from app.config import get_settings
from app.schemas import ConfirmationItem


def _confirmation_file() -> Path:
    settings = get_settings()
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    return settings.memory_dir / "confirmations.json"


def _read_payload() -> dict[str, dict]:
    path = _confirmation_file()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(payload: dict[str, dict]) -> None:
    _confirmation_file().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _expires_at_iso() -> str:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.confirmation_ttl_minutes)
    return expires_at.isoformat()


def create_confirmation(
    *,
    session_id: str,
    user_id: str | None,
    route: str,
    action_type: str,
    summary: str,
    payload: str,
    original_message: str,
) -> ConfirmationItem:
    token = uuid4().hex[:16]
    item = ConfirmationItem(
        token=token,
        route=route,
        action_type=action_type,
        summary=summary,
        session_id=session_id,
        user_id=user_id,
        expires_at=_expires_at_iso(),
    )
    data = _read_payload()
    data[token] = {
        "token": token,
        "route": route,
        "action_type": action_type,
        "summary": summary,
        "session_id": session_id,
        "user_id": user_id,
        "payload": payload,
        "original_message": original_message,
        "expires_at": item.expires_at,
    }
    _write_payload(data)
    return item


def get_confirmation(token: str) -> dict | None:
    payload = _read_payload()
    item = payload.get(token)
    if not item:
        return None

    expires_at = datetime.fromisoformat(item["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        delete_confirmation(token)
        return None
    return item


def delete_confirmation(token: str) -> None:
    payload = _read_payload()
    if token in payload:
        del payload[token]
        _write_payload(payload)
