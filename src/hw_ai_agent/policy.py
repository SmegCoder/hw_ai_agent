from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping


class PolicyError(ValueError):
    pass


class ApprovalDeferred(RuntimeError):
    """No interactive decision is available; keep the saved call waiting."""


class Action(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


TOOLS = frozenset({"list_files", "read_file", "write_file", "delete_file", "Bash"})
FILE_TOOLS = frozenset({"list_files", "read_file", "write_file", "delete_file"})
DEFAULTS = {
    "list_files": Action.ALLOW,
    "read_file": Action.ALLOW,
    "write_file": Action.ASK,
    "delete_file": Action.ASK,
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
    scope: str = "call"


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...] = ()
    # Tuple instead of a mutable dict: model calls cannot modify active rules.
    defaults: tuple[tuple[str, Action], ...] = tuple(DEFAULTS.items())
    directory_allow: tuple[str, ...] = (".",)
    directory_deny: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"directories": {"allow": list(self.directory_allow), "deny": list(self.directory_deny)},
                "defaults": {tool: action.value for tool, action in self.defaults}, "rules": [
            {key: value for key, value in {"tool": r.tool, "action": r.action.value,
             "path": r.path, "command": r.command, "reason": r.reason}.items() if value is not None}
            for r in self.rules]}

    def directory_allowed(self, path: str) -> bool:
        candidate = Path(path)
        contains = lambda directory: candidate.is_relative_to(Path(directory))
        return any(contains(p) for p in self.directory_allow) and not any(contains(p) for p in self.directory_deny)

    @classmethod
    def restore(cls, data: dict) -> Policy:
        if not isinstance(data, dict) or not isinstance(data.get("defaults", {}), dict):
            return cls.from_dict(data)
        if "delete_file" not in data.get("defaults", {}):
            data = {**data, "defaults": {**data.get("defaults", {}), "delete_file": "deny"}}
        return cls.from_dict(data)

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

    def sandbox_read_condition(self, root: Path) -> str | None:
        rules = [r for r in self.rules if r.tool in {"*", "read_file"} and r.command is None]
        restrictions = [r for r in rules if r.action != Action.ALLOW]
        allowed = [r for r in rules if r.action == Action.ALLOW]
        default = dict(self.defaults).get("read_file", Action.DENY)

        def pattern(rule: Rule) -> str:
            if rule.path == ".":
                return f"(literal {json.dumps(str(root), ensure_ascii=False)})"
            if rule.path is None or rule.path == "*":
                return f"(subpath {json.dumps(str(root), ensure_ascii=False)})"
            if any(c in rule.path for c in "[]\x00"):
                raise PolicyError("Bash read rules support literals, '*' and '?' only; unsupported patterns fail closed")
            relative = "".join("(.|\n)*" if c == "*" else "(.|\n)" if c == "?" else re.escape(c)
                               for c in rule.path)
            regex = "^" + re.escape(str(root)) + "/" + relative + "$"
            return f"(regex {json.dumps(regex, ensure_ascii=False)})"

        def any_of(items: list[str]) -> str:
            return items[0] if len(items) == 1 else "(require-any " + " ".join(items) + ")"

        if not self.directory_allow:
            return None
        conditions = [any_of([f"(subpath {json.dumps(str(root / p), ensure_ascii=False)})"
                              for p in self.directory_allow])]
        if self.directory_deny:
            conditions.append("(require-not " + any_of([
                f"(subpath {json.dumps(str(root / p), ensure_ascii=False)})" for p in self.directory_deny]) + ")")
        if default != Action.ALLOW:
            if not allowed:
                return None
            conditions.append(any_of([pattern(r) for r in allowed]))
        if restrictions:
            conditions.append("(require-not " + any_of([pattern(r) for r in restrictions]) + ")")
        return conditions[0] if len(conditions) == 1 else "(require-all " + " ".join(conditions) + ")"

    @classmethod
    def from_dict(cls, data: Any, *, fail_closed: bool = False) -> Policy:
        if not isinstance(data, dict) or set(data) - {"defaults", "rules", "directories"}:
            raise PolicyError("Policy must contain only defaults, rules and directories")
        directories = data.get("directories", {"allow": [] if fail_closed else ["."], "deny": []})
        if not isinstance(directories, dict) or set(directories) - {"allow", "deny"}:
            raise PolicyError("Directories must contain allow and deny lists")
        paths = {}
        for kind in ("allow", "deny"):
            values = directories.get(kind, [])
            if not isinstance(values, list) or len(values) > 100:
                raise PolicyError("Directory rules must be lists of at most 100 paths")
            for value in values:
                if (not isinstance(value, str) or not value or len(value) > 1000 or
                        any(c in value for c in "\x00*?[]") or Path(value).is_absolute() or
                        ".." in Path(value).parts):
                    raise PolicyError("Directories must be relative workspace paths without patterns or '..'")
            paths[kind] = tuple(Path(value).as_posix() for value in values)
        defaults = data.get("defaults", {})
        if not isinstance(defaults, dict) or set(defaults) - TOOLS:
            raise PolicyError("Invalid defaults or unknown tool")
        configured = dict.fromkeys(TOOLS, Action.DENY) if fail_closed else dict(DEFAULTS)
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
        return cls(tuple(rules), tuple(configured.items()), paths["allow"], paths["deny"])

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
            # A user-supplied file is a complete policy, not an overlay on the
            # built-in interactive defaults. Omitted permissions grant nothing.
            return cls.from_dict(json.loads(raw, object_pairs_hook=unique_object), fail_closed=True)
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise PolicyError("Cannot read valid policy JSON") from None
