from __future__ import annotations
import errno
import json
import math
import socket
import ssl
import certifi
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException, IncompleteRead
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from .config import Settings
from .text import model_text

class ModelClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, Any]],
                 tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...

class ModelClientError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


OPENROUTER_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504, 524, 529}
MAX_RETRY_DELAY = 30


def _retry_after_seconds(header: str | None) -> float:
    if not header:
        return 0
    if header.isascii() and header.isdecimal():
        if len(header) > 9:
            return float("inf")
        return int(header)
    try:
        date = parsedate_to_datetime(header)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, (date - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return 0

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
        self.context_window: int | None = None
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
            data=json.dumps(model_text(payload)).encode(),
            headers={"Authorization": f"Bearer {self.settings.api_key}",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ModelClientError("Model response exceeds the size limit")
            body = json.loads(raw)
        except HTTPError as exc:
            try:
                header = exc.headers.get("Retry-After") if exc.headers else None
                raise self._http_error(exc.code, header) from None
            finally:
                exc.close()
        except (OSError, HTTPException) as exc:
            raise self._transport_error(exc) from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelClientError("Model endpoint returned invalid JSON") from None
        # Billing can be known even if the answer is truncated or malformed.
        if isinstance(body, dict) and isinstance(body.get("usage"), dict):
            usage = dict(body["usage"])
            if body.get("service_tier") not in (None, "default", "standard"):
                usage["_nonstandard_tier"] = True
            self.usage.append(usage)
        try:
            if self.settings.provider == "openrouter" and isinstance(body, dict) and "error" in body:
                raise self._openrouter_response_error(body["error"])
            choice = body["choices"][0]
            if self.settings.provider == "openrouter" and choice.get("finish_reason") == "error":
                raise self._openrouter_response_error(choice.get("error"))
            message = choice["message"]
            if choice.get("finish_reason") == "length":
                raise ModelClientError("Model output was cut off; increase --max-tokens")
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ModelClientError("Model response has no assistant message") from None
        if not isinstance(message, Mapping):
            raise ModelClientError("Model returned an invalid assistant message")
        return message

    def _http_error(self, code: int, retry_header: str | None = None) -> ModelClientError:
        if self.settings.provider != "openrouter":
            return ModelClientError(f"{self.settings.provider} returned HTTP {code}")
        descriptions = {
            401: "проверьте API-ключ", 402: "проверьте баланс и лимит ключа",
            408: "истекло время запроса", 429: "слишком много запросов",
            500: "ошибка сервера", 502: "ошибка вышестоящего сервера",
            503: "сервис временно недоступен", 504: "истекло время ожидания ответа",
            524: "истекло время ожидания ответа", 529: "провайдер перегружен",
        }
        detail = descriptions.get(code)
        message = f"OpenRouter · HTTP {code}" + (f" · {detail}" if detail else "")
        wait = _retry_after_seconds(retry_header) if code in OPENROUTER_TRANSIENT_STATUS else None
        if wait is not None and wait > MAX_RETRY_DELAY:
            message += f" · повторите через {math.ceil(wait)} с" if math.isfinite(wait) else " · повторите позже"
        return ModelClientError(message, retry_after=wait)

    @staticmethod
    def _openrouter_response_error(error) -> ModelClientError:
        # The provider can report a generation error after sending HTTP 200.
        code = error.get("code") if isinstance(error, dict) else None
        suffix = f" · код {code}" if type(code) is int and 100 <= code <= 599 else ""
        return ModelClientError(f"OpenRouter · ошибка при генерации ответа{suffix}")

    def _transport_error(self, exc: OSError | HTTPException) -> ModelClientError:
        reason = exc.reason if isinstance(exc, URLError) else exc
        retry = True
        if isinstance(reason, socket.gaierror):
            detail = "ошибка DNS"
            retry = reason.errno == socket.EAI_AGAIN
        elif isinstance(reason, TimeoutError):
            detail = f"время ожидания ответа истекло ({self.timeout:g} с)"
        elif isinstance(reason, (ConnectionError, IncompleteRead, ssl.SSLEOFError)):
            detail = "соединение прервалось" if not isinstance(reason, ConnectionRefusedError) else "соединение отклонено"
        elif isinstance(reason, ssl.SSLError):
            detail, retry = "ошибка TLS", False
        elif isinstance(reason, OSError) and reason.errno in {errno.ENETUNREACH, errno.EHOSTUNREACH}:
            detail = "сеть недоступна"
        else:
            return ModelClientError(f"Cannot reach {self.settings.provider}: {type(exc).__name__}")
        provider = "OpenRouter" if self.settings.provider == "openrouter" else self.settings.provider
        return ModelClientError(f"{provider} · {detail}", retry_after=0 if retry and provider == "OpenRouter" else None)
