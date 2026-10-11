from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping


class PolicyError(ValueError):
    pass


class Action(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


TOOLS = frozenset({"list_files", "read_file", "write_file", "Bash"})
FILE_TOOLS = frozenset({"list_files", "read_file", "write_file"})
DEFAULTS = {
    "list_files": Action.ALLOW,
    "read_file": Action.ALLOW,
    "write_file": Action.ASK,
    "Bash": Action.ALLOW,
}


@dataclass(frozen=True)
class Rule:
    tool: str
    action: Action
    path: str | None = None
    command: str | None = None
    reason: str = "Matched permission rule"

    def matches(self, tool: str, path: str | None, command: str | None) -> bool:
        return (
            self.tool in {"*", tool}
            and (self.path is None or (path is not None and fnmatchcase(path, self.path)))
            and (self.command is None or (command is not None and fnmatchcase(command, self.command)))
        )


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


@dataclass(frozen=True)
class ApprovalRequest:
    tool: str
    target: str
    reason: str
    preview: str
    fingerprint: str


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...] = ()
    # Tuple instead of a mutable dict: model calls cannot modify active rules.
    defaults: tuple[tuple[str, Action], ...] = tuple(DEFAULTS.items())

    def decide(self, tool: str, *, path: str | None = None,
               command: str | None = None) -> Decision:
        if tool not in TOOLS:
            return Decision(Action.DENY, "Unknown tool")
        matches = [r for r in self.rules if r.matches(tool, path, command)]
        if matches:
            # Explicit denials always outrank ask/allow, irrespective of rule order.
            priority = {Action.ALLOW: 0, Action.ASK: 1, Action.DENY: 2}
            chosen = max(matches, key=lambda r: priority[r.action])
            return Decision(chosen.action, chosen.reason)
        return Decision(dict(self.defaults).get(tool, Action.DENY), f"Default permission for {tool}")

    @classmethod
    def from_dict(cls, data: Any) -> Policy:
        if not isinstance(data, dict) or set(data) - {"defaults", "rules"}:
            raise PolicyError("Policy must contain only defaults and rules")
        defaults = data.get("defaults", {})
        if not isinstance(defaults, dict) or set(defaults) - TOOLS:
            raise PolicyError("Invalid defaults or unknown tool")
        configured = dict(DEFAULTS)
        try:
            for tool, value in defaults.items():
                configured[tool] = Action(value)
        except (ValueError, TypeError):
            raise PolicyError("Permissions must be allow, ask or deny") from None
        definitions = data.get("rules", [])
        if not isinstance(definitions, list) or len(definitions) > 100:
            raise PolicyError("Rules must be a list of at most 100 entries")
        rules = []
        for item in definitions:
            if not isinstance(item, dict) or set(item) - {"tool", "action", "path", "command", "reason"}:
                raise PolicyError("Invalid rule fields")
            tool = item.get("tool")
            if not isinstance(tool, str) or tool not in TOOLS | {"*"}:
                raise PolicyError("Invalid rule tool")
            try:
                action = Action(item["action"])
            except (KeyError, ValueError, TypeError):
                raise PolicyError("Every rule needs allow, ask or deny") from None
            for field in ("path", "command", "reason"):
                if field in item and (not isinstance(item[field], str) or not item[field] or len(item[field]) > 1000):
                    raise PolicyError(f"{field} must be a non-empty string of at most 1000 characters")
            if "path" in item and tool not in FILE_TOOLS | {"*"}:
                raise PolicyError("Path rules apply only to file tools")
            if "command" in item and tool not in {"Bash", "*"}:
                raise PolicyError("Command rules apply only to Bash")
            if "path" in item and "command" in item:
                raise PolicyError("A rule cannot require both path and command")
            if "path" in item and (Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts):
                raise PolicyError("Rule paths must be relative to workspace without '..'")
            rules.append(Rule(tool, action, item.get("path"), item.get("command"),
                              item.get("reason", "Matched permission rule")))
        return cls(tuple(rules), tuple(configured.items()))

    @classmethod
    def load(cls, path: Path) -> Policy:
        try:
            with path.open("rb") as stream:
                raw = stream.read(64_001)
            if len(raw) > 64_000:
                raise PolicyError("Policy file exceeds 64000 bytes")
            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise PolicyError("Duplicate key in policy JSON")
                    result[key] = value
                return result
            return cls.from_dict(json.loads(raw, object_pairs_hook=unique_object))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise PolicyError("Cannot read valid policy JSON") from None
