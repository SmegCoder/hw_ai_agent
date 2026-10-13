from __future__ import annotations
import os
import re
import stat
import tempfile
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

def parse_env(text: str) -> dict[str, str]:
    """Parse KEY=value with optional quotes. Never evaluate shell expressions."""
    values = {}
    for number, line in enumerate(text.splitlines(), 1):
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
        values.setdefault(key, value)
    return values


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for key, value in parse_env(path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)


def load_launch_env(path: Path) -> None:
    """Offline demos need launch defaults, but not provider settings or keys."""
    if not path.exists():
        return
    allowed = {option.env for option in LAUNCH_OPTIONS.values()}
    for key, value in parse_env(path.read_text(encoding="utf-8")).items():
        if key in allowed:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class LaunchOption:
    env: str
    default: object
    kind: type
    minimum: int | None = None
    choices: tuple[str, ...] = ()


LAUNCH_OPTIONS = {
    "max_rounds": LaunchOption("AGENT_MAX_ROUNDS", 12, int, 1),
    "max_tokens": LaunchOption("AGENT_MAX_TOKENS", 2048, int, 1),
    "max_tool_calls": LaunchOption("AGENT_MAX_TOOL_CALLS", 128, int, 0),
    "max_side_effects": LaunchOption("AGENT_MAX_SIDE_EFFECTS", 32, int, 0),
    "max_result_bytes": LaunchOption("AGENT_MAX_RESULT_BYTES", 1_000_000, int, 0),
    "mode": LaunchOption("AGENT_MODE", "safe", str, choices=("safe", "auto")),
    "approvals": LaunchOption("AGENT_APPROVALS", "ask", str, choices=("ask", "deny")),
    "workspace": LaunchOption("AGENT_WORKSPACE", None, Path),
    "policy": LaunchOption("AGENT_POLICY", None, Path),
    "state_db": LaunchOption("AGENT_STATE_DB", Path(".agent/sessions.sqlite3"), Path),
    "plan": LaunchOption("AGENT_PLAN", False, bool),
    "tasks": LaunchOption("AGENT_TASKS", False, bool),
    "journal": LaunchOption("AGENT_JOURNAL", False, bool),
}

START_COMMANDS = {"run", "plan", "demo", "plan-demo", "workflow-demo", "tasks-demo", "rollback-demo"}


def option_value(name: str, value):
    option = LAUNCH_OPTIONS[name]
    label = f"--{name.replace('_', '-')} / {option.env}"
    if value is None:
        return None
    if option.kind is bool:
        if isinstance(value, bool):
            return value
        choices = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        result = choices.get(str(value).lower())
        if result is None:
            raise ConfigurationError(f"{label}: используйте true/false или 1/0")
        return result
    if option.kind is int:
        try:
            if isinstance(value, bool):
                raise ValueError
            result = int(value)
        except (TypeError, ValueError):
            raise ConfigurationError(f"{label}: требуется целое число") from None
        if result < option.minimum:
            raise ConfigurationError(f"{label}: минимум {option.minimum}")
        return result
    value = str(value)
    if not value or any(c in value for c in "\n\r\x00"):
        raise ConfigurationError(f"{label}: требуется непустое значение в одной строке")
    if option.choices and value not in option.choices:
        raise ConfigurationError(f"{label}: допустимы {', '.join(option.choices)}")
    return option.kind(value)


def validate_explicit_options(args) -> None:
    for name in LAUNCH_OPTIONS:
        if getattr(args, name, None) is not None:
            option_value(name, getattr(args, name))


def apply_launch_defaults(args) -> None:
    """CLI > process environment > .env (loaded earlier) > legacy defaults."""
    for name, option in LAUNCH_OPTIONS.items():
        if not hasattr(args, name) or getattr(args, name) is not None:
            continue
        # Demo workspaces are fresh fixtures, never the configured user's project.
        env_value = None if name == "workspace" and args.action not in {"run", "plan"} else os.getenv(option.env)
        default = option.default
        if name == "max_rounds":
            default = 30 if args.action == "tasks-demo" else 20 if args.action in {"workflow-demo", "plan-demo", "rollback-demo"} else 12
        if name == "max_tokens" and args.action == "tasks-demo":
            default = 3072
        setattr(args, name, option_value(name, env_value if env_value is not None else default))
    if args.action == "plan" or getattr(args, "tasks", False):
        args.plan = True
    if args.action == "rollback-demo":
        args.journal = True


def initial_launch_values() -> dict[str, str]:
    return {option.env: str(option.default).lower() if option.kind is bool else str(option.default)
            for option in LAUNCH_OPTIONS.values() if option.default is not None}


def _read_env_file(path: Path):
    previous, original = None, b""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        pass
    else:
        with os.fdopen(descriptor, "rb") as source:
            previous = os.fstat(source.fileno())
            if not stat.S_ISREG(previous.st_mode) or previous.st_nlink != 1:
                raise ConfigurationError("Файл настроек должен быть обычным файлом без жёстких ссылок")
            original = source.read(1_000_001)
            if len(original) > 1_000_000:
                raise ConfigurationError("Файл настроек слишком большой")
    return previous, original


def _file_version(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_nlink)


def update_env(path: Path, updates: dict[str, str]) -> None:
    """Replace only specified keys atomically; preserve secrets and comments."""
    for key, value in updates.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value or any(c in value for c in "\r\n\x00"):
            raise ConfigurationError("Настройка должна быть непустым значением в одной строке")
        value.encode("utf-8")
    previous, original = _read_env_file(path)
    text = original.decode("utf-8")
    parse_env(text)
    newline = "\r\n" if b"\r\n" in original else "\n"
    remaining = dict(updates)
    lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        key = stripped.partition("=")[0].strip() if stripped and not stripped.startswith("#") else None
        if key in updates:
            if key in remaining:
                lines.append(f'{key}="{remaining.pop(key)}"{newline}')
            # Remove duplicate definitions only for keys being changed.
        else:
            lines.append(line)
    if remaining and lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    lines.extend(f'{key}="{value}"{newline}' for key, value in remaining.items())
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            os.fchmod(target.fileno(), 0o600)
            target.write("".join(lines).encode("utf-8"))
            target.flush()
            os.fsync(target.fileno())
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if previous is None:
            if current is not None:
                raise ConfigurationError("Файл настроек появился во время записи; повторите set")
            # link creates a new destination without overwriting an existing file.
            os.link(temporary, path)
        else:
            if current is None or _file_version(current) != _file_version(previous):
                raise ConfigurationError("Файл настроек изменился во время записи; повторите set")
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def set_options(args) -> dict[str, str]:
    updates = {}
    for name, option in LAUNCH_OPTIONS.items():
        value = getattr(args, name, None)
        if value is not None:
            value = option_value(name, value)
            updates[option.env] = str(value).lower() if option.kind is bool else str(value)
    if args.provider is not None:
        updates["AGENT_PROVIDER"] = args.provider
    if args.model is not None:
        # Read provider without loading secrets into the process environment.
        _, content = _read_env_file(args.env_file)
        file_values = parse_env(content.decode("utf-8"))
        provider = args.provider or os.getenv("AGENT_PROVIDER") or file_values.get("AGENT_PROVIDER", "openrouter")
        if provider not in PROVIDERS:
            raise ConfigurationError("Неизвестный провайдер; укажите --provider")
        updates[f"{provider.upper()}_MODEL"] = args.model
    if not updates:
        raise ConfigurationError("Укажите настройку, например: hw-ai-agent set --provider ollama")
    update_env(args.env_file, updates)
    return updates

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
