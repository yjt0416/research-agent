from __future__ import annotations

import ast
import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT


TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".html", ".css", ".js", ".yml", ".yaml"}

ALLOWED_PYTHON_BUILTINS = {
    "print": print,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "range": range,
    "abs": abs,
    "round": round,
}

DISALLOWED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Attribute,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.While,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
)


@dataclass(frozen=True)
class ToolIntent:
    action_type: str
    summary: str
    payload: str


def resolve_workspace_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    project_root = PROJECT_ROOT.resolve()
    if project_root not in candidate.parents and candidate != project_root:
        raise ValueError("Only files inside the current project workspace can be read.")

    return candidate


def read_file_tool(path: str) -> str:
    target = resolve_workspace_path(path)
    if not target.exists():
        raise ValueError(f"File not found: {target}")
    if target.is_dir():
        raise ValueError("Expected a file path, but got a directory.")
    if target.suffix.lower() not in TEXT_EXTENSIONS:
        raise ValueError(f"Unsupported text file type: {target.suffix or 'unknown'}")

    content = target.read_text(encoding="utf-8")
    if len(content) > 12000:
        content = content[:12000] + "\n\n[Content truncated for demo.]"
    return content


def _validate_python_code(code: str) -> ast.AST:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_AST_NODES):
            raise ValueError(f"Disallowed Python feature: {node.__class__.__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_PYTHON_BUILTINS:
                raise ValueError(f"Only safe built-ins are allowed. Unsupported call: {node.func.id}")
        elif isinstance(node, ast.Call):
            raise ValueError("Only direct calls to safe built-ins are allowed.")
    return tree


def python_tool(code: str) -> str:
    if not code.strip():
        raise ValueError("Python tool input cannot be empty.")

    tree = _validate_python_code(code)
    output_buffer = io.StringIO()
    local_vars: dict[str, object] = {}
    exec_globals = {"__builtins__": ALLOWED_PYTHON_BUILTINS}

    with contextlib.redirect_stdout(output_buffer):
        exec(compile(tree, filename="<tool>", mode="exec"), exec_globals, local_vars)

    output = output_buffer.getvalue().strip()
    if output:
        return output

    visible_locals = {key: value for key, value in local_vars.items() if not key.startswith("_")}
    if visible_locals:
        return str(visible_locals)

    return "Python code executed successfully with no printed output."


def extract_file_path(message: str) -> str | None:
    backtick_match = re.search(r"`([^`]+)`", message)
    if backtick_match:
        candidate = backtick_match.group(1).strip()
        if Path(candidate).suffix.lower() in TEXT_EXTENSIONS:
            return candidate

    for token in re.findall(r"[A-Za-z0-9_./\\:-]+", message):
        if Path(token).suffix.lower() in TEXT_EXTENSIONS:
            return token

    return None


def extract_python_snippet(message: str) -> str | None:
    fenced = re.search(r"```(?:python)?\n([\s\S]*?)```", message, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    inline = re.search(r"`([^`]+)`", message)
    if inline:
        return inline.group(1).strip()

    keywords = ("计算", "运行代码", "执行代码", "python", "代码")
    if any(keyword in message.lower() for keyword in keywords):
        parts = re.split(r"[:：]", message, maxsplit=1)
        if len(parts) == 2:
            return parts[1].strip()

    return None


def parse_tool_intent(message: str) -> ToolIntent | None:
    file_path = extract_file_path(message)
    if file_path:
        return ToolIntent(
            action_type="read_file",
            summary=f"Read file `{file_path}` from the current workspace.",
            payload=file_path,
        )

    code = extract_python_snippet(message)
    if code:
        preview = code.splitlines()[0][:80] if code.splitlines() else code[:80]
        return ToolIntent(
            action_type="python",
            summary=f"Run restricted Python code. Preview: {preview}",
            payload=code,
        )

    return None
