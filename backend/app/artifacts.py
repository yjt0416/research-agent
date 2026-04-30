from __future__ import annotations

import json
import mimetypes
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from app.config import get_settings
from app.schemas import ArtifactItem


ARTIFACT_REGISTRY = "registry.json"
CODE_BLOCK_PATTERN = re.compile(
    r"```(?P<language>[a-zA-Z0-9_+-]*)[ \t]*(?:filename=(?P<filename>[^\n`]+))?\n(?P<content>[\s\S]*?)```"
)
FILENAME_HINT_PATTERN = re.compile(
    r"^(?:#|//|<!--)\s*filename\s*:\s*(?P<filename>[A-Za-z0-9_.\-/\\]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class GeneratedFile:
    filename: str
    content: str
    media_type: str
    kind: str


def _artifacts_dir() -> Path:
    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings.artifacts_dir


def _registry_path() -> Path:
    return _artifacts_dir() / ARTIFACT_REGISTRY


def _read_registry() -> dict[str, dict[str, str | int]]:
    path = _registry_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(payload: dict[str, dict[str, str | int]]) -> None:
    _registry_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", filename.strip(), flags=re.UNICODE)
    return cleaned or "artifact.txt"


def _guess_media_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    suffix = Path(filename).suffix.lower()
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".py":
        return "text/x-python"
    if suffix == ".zip":
        return "application/zip"
    return "text/plain"


def _guess_kind(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix == ".py":
        return "python"
    if suffix == ".zip":
        return "zip"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix == ".json":
        return "json"
    return "text"


def save_bytes_artifact(
    *,
    filename: str,
    data: bytes,
    session_id: str | None = None,
    media_type: str | None = None,
    kind: str | None = None,
) -> ArtifactItem:
    artifact_id = uuid4().hex[:12]
    safe_filename = _sanitize_filename(filename)
    target_dir = _artifacts_dir() / (session_id or "shared")
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / f"{artifact_id}_{safe_filename}"
    path.write_bytes(data)

    payload = _read_registry()
    payload[artifact_id] = {
        "filename": safe_filename,
        "path": str(path),
        "media_type": media_type or _guess_media_type(safe_filename),
        "kind": kind or _guess_kind(safe_filename),
        "size_bytes": path.stat().st_size,
    }
    _write_registry(payload)

    record = payload[artifact_id]
    return ArtifactItem(
        artifact_id=artifact_id,
        filename=str(record["filename"]),
        media_type=str(record["media_type"]),
        kind=str(record["kind"]),
        size_bytes=int(record["size_bytes"]),
        download_url=f"/artifacts/{artifact_id}/download",
    )


def save_text_artifact(
    *,
    filename: str,
    content: str,
    session_id: str | None = None,
    media_type: str | None = None,
    kind: str | None = None,
) -> ArtifactItem:
    return save_bytes_artifact(
        filename=filename,
        data=content.encode("utf-8"),
        session_id=session_id,
        media_type=media_type,
        kind=kind,
    )


def save_existing_file_artifact(
    *,
    path: Path,
    session_id: str | None = None,
    media_type: str | None = None,
    kind: str | None = None,
) -> ArtifactItem:
    return save_bytes_artifact(
        filename=path.name,
        data=path.read_bytes(),
        session_id=session_id,
        media_type=media_type,
        kind=kind,
    )


def resolve_artifact(artifact_id: str) -> tuple[Path, ArtifactItem]:
    payload = _read_registry()
    record = payload.get(artifact_id)
    if not record:
        raise FileNotFoundError(f"Artifact not found: {artifact_id}")

    path = Path(str(record["path"]))
    if not path.exists():
        raise FileNotFoundError(f"Artifact file missing: {path}")

    return path, ArtifactItem(
        artifact_id=artifact_id,
        filename=str(record["filename"]),
        media_type=str(record["media_type"]),
        kind=str(record["kind"]),
        size_bytes=int(record["size_bytes"]),
        download_url=f"/artifacts/{artifact_id}/download",
    )


def create_zip_artifact(
    *,
    files: list[GeneratedFile],
    filename: str,
    session_id: str | None = None,
) -> ArtifactItem:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.writestr(file.filename, file.content.encode("utf-8"))

    return save_bytes_artifact(
        filename=filename,
        data=buffer.getvalue(),
        session_id=session_id,
        media_type="application/zip",
        kind="zip",
    )


def _infer_generated_filename(language: str, index: int) -> str:
    normalized = language.lower()
    if normalized in {"python", "py"}:
        return f"generated_{index}.py"
    if normalized in {"markdown", "md"}:
        return f"generated_{index}.md"
    if normalized in {"json"}:
        return f"generated_{index}.json"
    return f"generated_{index}.txt"


def extract_generated_files(answer: str) -> list[GeneratedFile]:
    files: list[GeneratedFile] = []
    for index, match in enumerate(CODE_BLOCK_PATTERN.finditer(answer), start=1):
        content = match.group("content").strip()
        if not content:
            continue

        filename = (match.group("filename") or "").strip()
        if not filename:
            first_line = content.splitlines()[0] if content.splitlines() else ""
            filename_hint = FILENAME_HINT_PATTERN.search(first_line)
            if filename_hint:
                filename = filename_hint.group("filename").strip()

        filename = _sanitize_filename(filename or _infer_generated_filename(match.group("language"), index))
        media_type = _guess_media_type(filename)
        files.append(
            GeneratedFile(
                filename=filename,
                content=content,
                media_type=media_type,
                kind=_guess_kind(filename),
            )
        )
    return files


def build_markdown_artifact(markdown: str, *, base_name: str, session_id: str | None = None) -> ArtifactItem:
    filename = _sanitize_filename(f"{base_name}.md")
    return save_text_artifact(
        filename=filename,
        content=markdown,
        session_id=session_id,
        media_type="text/markdown",
        kind="markdown",
    )


def build_python_tool_artifacts(
    *,
    code: str,
    output: str,
    session_id: str | None = None,
    prefix: str = "python_tool",
) -> list[ArtifactItem]:
    generated_files = [
        GeneratedFile(
            filename=f"{_sanitize_filename(prefix)}.py",
            content=code,
            media_type="text/x-python",
            kind="python",
        ),
        GeneratedFile(
            filename=f"{_sanitize_filename(prefix)}_output.txt",
            content=output,
            media_type="text/plain",
            kind="text",
        ),
    ]

    bundle = create_zip_artifact(
        files=generated_files,
        filename=f"{_sanitize_filename(prefix)}.zip",
        session_id=session_id,
    )
    artifacts = [bundle]
    for file in generated_files:
        artifacts.append(
            save_text_artifact(
                filename=file.filename,
                content=file.content,
                session_id=session_id,
                media_type=file.media_type,
                kind=file.kind,
            )
        )
    return artifacts


def build_agent_artifacts(
    *,
    answer: str,
    route: str,
    session_id: str,
    user_message: str,
) -> list[ArtifactItem]:
    artifacts: list[ArtifactItem] = []
    generated_files = extract_generated_files(answer)

    if route in {"rag", "report"}:
        artifacts.append(
            build_markdown_artifact(
                answer,
                base_name=f"{route}_{session_id}",
                session_id=session_id,
            )
        )

    if generated_files:
        if len(generated_files) > 1:
            artifacts.append(
                create_zip_artifact(
                    files=generated_files,
                    filename=f"{route}_{session_id}_bundle.zip",
                    session_id=session_id,
                )
            )
        for file in generated_files:
            artifacts.append(
                save_text_artifact(
                    filename=file.filename,
                    content=file.content,
                    session_id=session_id,
                    media_type=file.media_type,
                    kind=file.kind,
                )
            )
        return artifacts

    if route == "tool":
        from app.tools import extract_python_snippet

        code = extract_python_snippet(user_message)
        if code:
            return build_python_tool_artifacts(
                code=code,
                output=answer,
                session_id=session_id,
                prefix=f"agent_python_{session_id}",
            )

    return artifacts
