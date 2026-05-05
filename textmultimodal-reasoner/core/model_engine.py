"""Self-contained multimodal model engine with agentic tool-calling loop."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from core.processor import process_interleaved_inputs
from core.tool_registry import FileManager, ToolRegistry, get_system_stats, internet_available, web_search

MAX_TOOL_RESPONSE_CHARS = 1200
MAX_TOOL_LOOPS = 3


@dataclass(slots=True)
class EngineSettings:
    visual_token_budget: int
    stream_reasoning: bool
    enable_tool_calling: bool
    type_k: int
    type_v: int
    flash_attn: bool
    allow_system_wide_file_access: bool


class ModelEngine:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()

        model_cfg = self.config.get("model", {})
        runtime_cfg = self.config.get("runtime", {})

        self.model_path = Path(model_cfg.get("model_path", "models/Gemma-4-E4B-it-UD-IQ2_M.gguf"))
        self.n_ctx = int(model_cfg.get("n_ctx", 4096))
        self.n_gpu_layers = int(model_cfg.get("n_gpu_layers", 0))
        self.settings = EngineSettings(
            visual_token_budget=int(runtime_cfg.get("visual_token_budget", 70)),
            stream_reasoning=bool(runtime_cfg.get("stream_reasoning", True)),
            enable_tool_calling=bool(runtime_cfg.get("enable_tool_calling", True)),
            type_k=int(runtime_cfg.get("type_k", 8)),
            type_v=int(runtime_cfg.get("type_v", 8)),
            flash_attn=bool(runtime_cfg.get("flash_attn", True)),
            allow_system_wide_file_access=bool(runtime_cfg.get("allow_system_wide_file_access", True)),
        )

        self._init_tools()
        self._validate_cpu_mode()

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config.yaml must be a mapping.")
        return loaded

    def _init_tools(self) -> None:
        workspace_root = self.config_path.resolve().parent
        scope = None if self.settings.allow_system_wide_file_access else workspace_root
        self.tools = ToolRegistry(workspace_root=scope)
        fm = FileManager(scope)

        self.tools.register_function("get_system_stats", get_system_stats, "Return current CPU and RAM usage stats.")
        self.tools.register_function("internet_available", internet_available, "Check if internet connectivity is available.")
        self.tools.register_function(
            "web_search",
            web_search,
            "Text-only web search results.",
            {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}},
        )
        self.tools.register_function("create_file", fm.create_file, "Create a file.", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}})
        self.tools.register_function("edit_file", fm.edit_file, "Edit/overwrite a file.", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}})
        self.tools.register_function("delete_path", fm.delete_path, "Delete a file or folder.", {"type": "object", "properties": {"path": {"type": "string"}}})
        self.tools.register_function("create_folder", fm.create_folder, "Create a folder.", {"type": "object", "properties": {"path": {"type": "string"}}})

    def _validate_cpu_mode(self) -> None:
        if self.n_gpu_layers != 0:
            raise ValueError("This engine is CPU-only. Set n_gpu_layers to 0.")

    def _agentic_system_prompt(self) -> str:
        schema = json.dumps(self.tools.tool_schema_list())
        return (
            "[SYSTEM PROMPT: Tools & Instructions]\n"
            f"KV cache quantization: type_k={self.settings.type_k}, type_v={self.settings.type_v}.\n"
            f"Flash attention: {self.settings.flash_attn}.\n"
            "You have access to the following tools: "
            f"{schema}. To call a tool, respond with "
            '<tool_call>{"name":"...","parameters":{...}}</tool_call>.\n'
            "Keep reasoning inside <|think|>...</|think|>."
        )

    def _fake_token_stream(self, prompt: str, max_tokens: int) -> Iterator[str]:
        lowered = prompt.lower()
        if "<tool_response>" in lowered:
            text = "<|think|>Tool output integrated.</|think|>Final answer generated from tool data."
        elif self.settings.enable_tool_calling and ("cpu" in lowered or "ram" in lowered or "system stats" in lowered):
            text = '<tool_call>{"name":"get_system_stats","parameters":{}}</tool_call>'
        else:
            text = "<|think|>Analyzing multimodal context.</|think|>Final answer without tools."
        for token in text.split(" "):
            yield token + " "

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or "name" not in payload:
            return None
        params = payload.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        return {"name": payload["name"], "parameters": params}

    def _run_stream_state_machine(self, stream: Iterator[str], on_reasoning_token: Callable[[str], None] | None) -> str:
        answer_chunks: list[str] = []
        in_think = False
        for token in stream:
            scan = token
            while scan:
                if not in_think and "<|think|>" in scan:
                    pre, scan = scan.split("<|think|>", 1)
                    if pre:
                        answer_chunks.append(pre)
                    in_think = True
                    continue
                if in_think and "</|think|>" in scan:
                    think_text, scan = scan.split("</|think|>", 1)
                    if think_text and on_reasoning_token:
                        on_reasoning_token(think_text)
                    in_think = False
                    continue
                if in_think:
                    if on_reasoning_token:
                        on_reasoning_token(scan)
                    scan = ""
                else:
                    answer_chunks.append(scan)
                    scan = ""
        return "".join(answer_chunks).strip()

    def _safe_tool_execute(self, name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.tools.execute(name, parameters)
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def generate_response(self, interleaved_inputs: list[dict[str, Any]], *, max_tokens: int = 512, temperature: float = 0.7, on_reasoning_token: Callable[[str], None] | None = None) -> str:
        processed = process_interleaved_inputs(interleaved_inputs)
        try:
            user_text = processed.prompt
            prompt = (
                f"{self._agentic_system_prompt()}\n"
                f"<image> [Embedded Image Tokens]</image>\n"
                f"<audio> [Embedded Audio Tokens]</audio>\n"
                f"<user>{user_text}</user>"
            )
            _ = temperature

            for _loop in range(MAX_TOOL_LOOPS):
                raw_output = self._run_stream_state_machine(self._fake_token_stream(prompt, max_tokens), on_reasoning_token)
                call = self._extract_tool_call(raw_output)
                if not call:
                    return raw_output
                tool_result = self._safe_tool_execute(call["name"], call["parameters"])
                compact = json.dumps(tool_result)
                if len(compact) > MAX_TOOL_RESPONSE_CHARS:
                    compact = compact[:MAX_TOOL_RESPONSE_CHARS] + "...<truncated>"
                prompt = f"{prompt}\n<tool_response>{compact}</tool_response>"

            return "Unable to complete tool loop within safety limit."
        finally:
            processed.pixel_values = None
            processed.audio_features = None
            gc.collect()

    def unload(self) -> None:
        gc.collect()
