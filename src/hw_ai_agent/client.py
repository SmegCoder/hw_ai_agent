from __future__ import annotations
import json
import socket
import ssl
import certifi
from http.client import HTTPException
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from .config import Settings

class ModelClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, Any]],
                 tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...

class ModelClientError(RuntimeError):
    pass

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class OpenAICompatibleClient:
    billing_enabled = True

    def __init__(self, settings: Settings | None = None, *, timeout: float = 90,
                 max_tokens: int = 2048, max_response_bytes: int = 2_000_000) -> None:
        self.settings = settings or Settings.from_env()
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_response_bytes = max_response_bytes
        self.usage: list[dict[str, Any]] = []
        self.price = None
        self.routing = None
        context = ssl.create_default_context(cafile=certifi.where())
        self.opener = build_opener(NoRedirect(), HTTPSHandler(context=context))

    def complete(self, messages: Sequence[Mapping[str, Any]],
                 tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model, "messages": list(messages),
            "tools": list(tools), "stream": False,
        }
        if self.settings.provider == "openai":
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        if self.settings.provider == "openrouter":
            payload["provider"] = {"require_parameters": True}
        if self.settings.provider == "ollama":
            payload["reasoning_effort"] = "none"
        request = Request(f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.settings.api_key}",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ModelClientError("Model response exceeds the size limit")
            body = json.loads(raw)
        except HTTPError as exc:
            exc.close()
            raise ModelClientError(f"{self.settings.provider} returned HTTP {exc.code}") from None
        except (URLError, socket.timeout, OSError, HTTPException) as exc:
            raise ModelClientError(f"Cannot reach {self.settings.provider}: {type(exc).__name__}") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelClientError("Model endpoint returned invalid JSON") from None
        # Billing can be known even if the answer is truncated or malformed.
        if isinstance(body, dict) and isinstance(body.get("usage"), dict):
            usage = dict(body["usage"])
            if body.get("service_tier") not in (None, "default", "standard"):
                usage["_nonstandard_tier"] = True
            self.usage.append(usage)
        try:
            choice = body["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") == "length":
                raise ModelClientError("Model output was cut off; increase --max-tokens")
        except (KeyError, IndexError, TypeError):
            raise ModelClientError("Model response has no assistant message") from None
        if not isinstance(message, Mapping):
            raise ModelClientError("Model returned an invalid assistant message")
        return message
