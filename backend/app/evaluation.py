from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings


def dataset_path() -> Path:
    settings = get_settings()
    settings.evals_dir.mkdir(parents=True, exist_ok=True)
    return settings.evals_dir / "day5_eval_dataset.jsonl"


def load_eval_dataset() -> list[dict]:
    path = dataset_path()
    if not path.exists():
        return []

    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        items.append(json.loads(stripped))
    return items


def summarize_eval_dataset() -> dict:
    items = load_eval_dataset()
    route_counts: dict[str, int] = {}
    for item in items:
        route = str(item.get("expected_route", "unknown"))
        route_counts[route] = route_counts.get(route, 0) + 1

    return {
        "dataset_path": str(dataset_path()),
        "count": len(items),
        "route_counts": route_counts,
        "samples": items[:5],
    }
