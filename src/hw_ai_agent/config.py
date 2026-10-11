"""Provider configuration and a non-executing .env reader."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

class ConfigurationError(ValueError):
    pass

PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "deepseek/deepseek-v3.2"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", ""),
    "ollama": ("http://127.0.0.1:11434/v1", "", "qwen3:4b"),
}

def load_env(path: Path) -> None:
    """Read KEY=value with optional quotes. Never evaluate shell expressions."""
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ConfigurationError(f"Invalid .env setting on line {number}")
        value = value.strip()
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ConfigurationError(f"Unclosed quote on .env line {number}")
            value = value[1:-1]
        os.environ.setdefault(key, value)

@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> Settings:
        provider = provider or os.getenv("AGENT_PROVIDER", "openrouter")
        if provider not in PROVIDERS:
            raise ConfigurationError(f"Unknown provider: {provider}")
        default_url, key_name, default_model = PROVIDERS[provider]
        model = model or os.getenv(f"{provider.upper()}_MODEL") or default_model
        if not model:
            raise ConfigurationError(f"Set {provider.upper()}_MODEL or pass --model")
        key = os.getenv(key_name, "") if key_name else "ollama"
        if key_name and not key:
            raise ConfigurationError(f"Set {key_name}: run hw-ai-agent setup --provider {provider}")
        url = os.getenv(f"{provider.upper()}_BASE_URL", default_url).rstrip("/")
        parsed = urlparse(url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigurationError("API URL must not contain credentials, query or fragment")
        if not parsed.hostname or (parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        )):
            raise ConfigurationError("Use HTTPS, or HTTP only for a local server")
        return cls(provider, model, url, key)
