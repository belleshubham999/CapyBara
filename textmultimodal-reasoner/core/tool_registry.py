"""Modular tool registry and callable tool adapters."""

from __future__ import annotations

import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import requests
from bs4 import BeautifulSoup


class ToolRegistry:
    """Registry mapping tool names to simple Python callables."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve() if workspace_root else None
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register_function(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str,
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = fn
        self._schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters_schema or {"type": "object", "properties": {}},
        }

    def execute(self, name: str, parameters: dict[str, Any] | None = None) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**(parameters or {}))

    def tool_schema_list(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())


def get_system_stats() -> dict[str, float]:
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_total_gb": round(memory.total / (1024**3), 2),
    }


def internet_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"online": True}
    except OSError as exc:
        return {"online": False, "error": str(exc)}


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Text-only web search (no browser automation)."""
    resp = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results: list[dict[str, str]] = []
    for node in soup.select("div.result"):
        link = node.select_one("a.result__a")
        snippet = node.select_one("a.result__snippet") or node.select_one("div.result__snippet")
        if not link:
            continue
        results.append(
            {
                "title": link.get_text(" ", strip=True),
                "url": link.get("href", ""),
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


class FileManager:
    """File manager with optional workspace scoping."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root.resolve() if root else None

    def _resolve(self, path: str) -> Path:
        base = self.root if self.root else Path("/")
        candidate = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if self.root and self.root not in candidate.parents and candidate != self.root:
            raise ValueError("Path escapes workspace root.")
        return candidate

    def create_file(self, path: str, content: str = "") -> dict[str, str]:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"status": "ok", "path": str(p)}

    def edit_file(self, path: str, content: str) -> dict[str, str]:
        p = self._resolve(path)
        p.write_text(content, encoding="utf-8")
        return {"status": "ok", "path": str(p)}

    def delete_path(self, path: str) -> dict[str, str]:
        p = self._resolve(path)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        return {"status": "ok", "path": str(p)}

    def create_folder(self, path: str) -> dict[str, str]:
        p = self._resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        return {"status": "ok", "path": str(p)}
