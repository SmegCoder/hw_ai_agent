"""The tool-calling loop."""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from .client import ModelClient
from .tools import ToolError, ToolRuntime


DEFAULT_SYSTEM_PROMPT = """You are a careful coding agent.
Use the provided tools to inspect and modify only the workspace.
Run the relevant tests after changes.
Tests are read-only. Never modify, rename or delete them.
For testing, call Bash with python -m unittest discover -v.
Bash accepts ONLY python -m unittest, pwd, ls, cat and sleep.
Do not use python -c, run scripts, or create scratch check scripts.
Use path='.' for list_files; never use an empty path.
After relevant tests pass, immediately provide your final summary without more tool calls.
When a tool reports an error, correct the request or explain the limitation.
"""


class AgentError(RuntimeError):
    """Raised when the agent loop cannot finish safely."""


class CodingAgent:
    def __init__(
        self,
        client: ModelClient,
        runtime: ToolRuntime,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_rounds: int = 12,
        max_context_chars: int = 120_000,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.client = client
        self.runtime = runtime
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.max_context_chars = max_context_chars
        self.on_event = on_event
        self.events: list[dict[str, Any]] = []

    def run(self, request: str) -> str:
        if not request.strip():
            raise ValueError("request must not be empty")
        self.events = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": request},
        ]

        for round_number in range(1, self.max_rounds + 1):
            if len(json.dumps(messages, ensure_ascii=False)) > self.max_context_chars:
                raise AgentError("Conversation exceeds the context size limit")
            self._emit({"event": "model_request", "round": round_number})
            response = self.client.complete(messages, self.runtime.tool_schemas)
            assistant = self._assistant_message(response)
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                content = assistant.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise AgentError("Model returned neither tool calls nor a final answer")
                self._emit({"event": "final", "content": content})
                return content

            for call in tool_calls:
                self._emit({"event": "tool_call", "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]})
                tool_result = self._execute_tool_call(call)
                messages.append(tool_result)
                self._emit({"event": "tool_result", **tool_result})

        raise AgentError(f"agent exceeded the limit of {self.max_rounds} model rounds")

    @staticmethod
    def _assistant_message(response: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise AgentError("model returned an invalid assistant message")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response.get("content"),
        }
        tool_calls = response.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise AgentError("model returned invalid tool_calls")
            if len(tool_calls) > 16:
                raise AgentError("Too many tool calls in one model response")
            normalized = []
            seen_ids = set()
            for call in tool_calls:
                if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                    raise AgentError("Malformed tool call from model")
                function = call["function"]
                if not isinstance(function.get("name"), str) or not function["name"]:
                    raise AgentError("Tool call is missing its function name")
                call_id = call.get("id") or "call_" + uuid.uuid4().hex
                if not isinstance(call_id, str) or call_id in seen_ids:
                    raise AgentError("Invalid or duplicate tool call ID")
                seen_ids.add(call_id)
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                normalized.append({"id": call_id, "type": "function", "function": {
                    "name": function["name"], "arguments": arguments,
                }})
            message["tool_calls"] = normalized
        # Some reasoning providers require these fields on replay.
        for field in ("reasoning_content", "reasoning_details"):
            if field in response:
                message[field] = copy.deepcopy(response[field])
        return message

    def _emit(self, event: dict[str, Any]) -> None:
        self.events.append(copy.deepcopy(event))
        if self.on_event:
            self.on_event(event)

    def _execute_tool_call(self, call: Any) -> dict[str, Any]:
        call_id = "unknown"
        name = "unknown"
        try:
            if not isinstance(call, Mapping):
                raise ToolError("tool call must be an object")
            call_id = str(call.get("id") or "unknown")
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise ToolError("tool call has no function object")
            name = str(function.get("name") or "unknown")
            raw_arguments = function.get("arguments", "{}")
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            if not isinstance(arguments, Mapping):
                raise ToolError("tool arguments must be a JSON object")
            content = self.runtime.execute(name, arguments)
        except (ToolError, json.JSONDecodeError, OSError) as exc:
            content = f"ERROR: {exc}"

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }

