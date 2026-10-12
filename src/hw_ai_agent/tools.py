from __future__ import annotations
import fnmatch
import difflib
import hashlib
import json
import os
import selectors
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from .policy import Action, ApprovalRequest, Policy, PolicyError
from .filesystem import TargetChangedError, capture_target, write_target

class ToolError(RuntimeError):
    pass

def protected(relative: Path) -> bool:
    return any(part in {".git", ".venv", ".idea", ".agent", "__pycache__"} or part.startswith(".env")
               for part in relative.parts)

def is_test(relative: Path) -> bool:
    return "tests" in relative.parts or fnmatch.fnmatch(relative.name, "test_*.py") or relative.name.endswith("_test.py")

class ToolRuntime:
    def __init__(self, root: str | Path, *, command_timeout: float = 10,
                 max_output_chars: int = 20_000, max_file_chars: int = 100_000,
                 policy: Policy | None = None,
                 mode: str = "safe",
                 approve: Callable[[ApprovalRequest], bool] | None = None,
                 on_policy_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace does not exist: {self.root}")
        if min(command_timeout, max_output_chars, max_file_chars) <= 0:
            raise ValueError("Runtime limits must be positive")
        self.command_timeout = command_timeout
        self.max_output_chars = max_output_chars
        self.max_file_chars = max_file_chars
        self.policy = policy or Policy.from_dict({"defaults": {"Bash": "ask"}})
        if mode not in {"safe", "auto"}:
            raise ValueError("Execution mode must be safe or auto")
        self.mode = mode
        self.session_permissions: set[str] | None = None
        self.approve = approve
        self.on_policy_event = on_policy_event
        self.policy_events: list[dict[str, Any]] = []

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        string = {"type": "string"}
        definitions = [
            ("list_files", "List workspace entries (up to 500).", {"path": string, "recursive": {"type": "boolean"}}, []),
            ("read_file", "Read a workspace UTF-8 file. Tests are read-only.", {"path": string}, ["path"]),
            ("write_file", "Replace a workspace UTF-8 file; tests and secrets are protected.",
             {"path": string, "content": string}, ["path", "content"]),
            ("Bash", "Run python -m unittest, pwd, ls, cat or sleep inside the sandbox. No pipes/redirects.",
             {"command": string}, ["command"]),
        ]
        return [{"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": False},
        }} for name, description, props, required in definitions]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        """Single entry point for model calls, policy and session permissions."""
        name = "Bash" if name == "bash" else name
        # Own an immutable-value snapshot: an approval callback cannot alter the call.
        if isinstance(arguments, Mapping):
            arguments = dict(arguments)
        try:
            path, command = self._validate_call(name, arguments)
        except ToolError as exc:
            self._policy_event(name, "deny", "blocked", str(exc), "runtime_boundary")
            raise
        target = None
        if name == "write_file":
            try:
                target = capture_target(self.root, self.root / path, arguments["path"])
            except OSError:
                raise ToolError("TARGET_CHANGED: cannot capture a stable write target") from None
        decision = self.policy.decide(name, path=path, command=command)
        if decision.action == Action.DENY:
            self._policy_event(name, "deny", "blocked", decision.reason, "policy")
            raise ToolError(f"POLICY_DENIED: {decision.reason}")
        if decision.action == Action.ASK:
            reusable = name in {"write_file", "Bash"}
            fingerprint = hashlib.sha256(json.dumps(
                {"tool": name, "arguments": arguments, "target": path,
                 "file_signature": target.file_signature if target else None},
                sort_keys=True, ensure_ascii=False
            ).encode()).hexdigest()
            preview = ""
            if name == "write_file":
                file = self._safe_path(arguments["path"])
                try:
                    original = self.read_file(arguments["path"]) if file.is_file() else ""
                    preview = "".join(difflib.unified_diff(original.splitlines(True),
                                arguments["content"].splitlines(True), fromfile="current", tofile="proposed"))
                except (ToolError, OSError, UnicodeError):
                    preview = "Existing content could not be previewed. Proposed content:\n" + arguments["content"]
                preview = preview[:4000] + ("\n... preview truncated ..." if len(preview) > 4000 else "")
            request = ApprovalRequest(name, path or command or ".", decision.reason, preview, fingerprint,
                                      "session" if reusable and self.session_permissions is not None else "call")
            approved = False
            source = "policy"
            if reusable and self.mode == "auto":
                approved, source = True, "auto_mode"
            elif reusable and self.session_permissions is not None and name in self.session_permissions:
                approved, source = True, "session_grant"
            else:
                try:
                    approved = self.approve is not None and self.approve(request) is True
                except Exception:
                    # Broken approval callbacks never authorize execution.
                    approved = False
            if approved and reusable and self.session_permissions is not None:
                self.session_permissions.add(name)
            self._policy_event(name, "ask", "approved" if approved else "refused",
                               decision.reason, source, fingerprint, request.scope)
            if not approved:
                raise ToolError("APPROVAL_REQUIRED: operator approval was not granted")
        else:
            self._policy_event(name, "allow", "permitted", decision.reason, "policy")
        handler = self.bash if name == "Bash" else getattr(self, name)
        try:
            if target is not None:
                write_target(target, arguments["content"])
                return f"wrote {len(arguments['content'])} characters to {path}"
            if path is not None:
                if self._safe_path(arguments.get("path", ".")).relative_to(self.root).as_posix() != path:
                    raise ToolError("TARGET_CHANGED: path changed after permission checking")
                # Execute against the checked canonical path, not its mutable alias.
                arguments = {**arguments, "path": path}
            return handler(**dict(arguments))
        except TargetChangedError:
            self._policy_event(name, "deny", "blocked", "Write target changed", "runtime_boundary")
            raise ToolError("TARGET_CHANGED: file or parent changed; a new request and approval are required") from None
        except (OSError, UnicodeError) as exc:
            raise ToolError(f"{name} failed: {type(exc).__name__}") from None

    def _validate_call(self, name: str, arguments: Mapping[str, Any]) -> tuple[str | None, str | None]:
        schema = next((s["function"] for s in self.tool_schemas if s["function"]["name"] == name), None)
        if schema is None:
            raise ToolError(f"Unknown tool: {name}")
        if not isinstance(arguments, Mapping):
            raise ToolError("Arguments must be a JSON object")
        properties = schema["parameters"]["properties"]
        if set(arguments) - properties.keys():
            raise ToolError(f"Unexpected arguments for {name}")
        if any(key not in arguments for key in schema["parameters"]["required"]):
            raise ToolError(f"Missing required argument for {name}")
        for key, value in arguments.items():
            expected = bool if properties[key]["type"] == "boolean" else str
            if type(value) is not expected:
                raise ToolError(f"{key} must be {properties[key]['type']}")
        if name == "Bash":
            self._argv(arguments["command"])
            return None, shlex.join(shlex.split(arguments["command"]))
        raw_path = arguments.get("path", ".")
        file = self._safe_path(raw_path)
        if name == "write_file":
            if is_test(Path(raw_path)) or is_test(file.relative_to(self.root)):
                raise ToolError("Test files are read-only")
            if len(arguments["content"]) > self.max_file_chars:
                raise ToolError("Content exceeds file size limit")
            if file.is_dir():
                raise ToolError("Write target is a directory")
            if file.is_symlink() or (file.is_file() and file.stat().st_nlink > 1):
                raise ToolError("Writing hard-linked files is not allowed")
        return file.relative_to(self.root).as_posix(), None

    def _policy_event(self, tool: str, action: str, outcome: str, reason: str,
                      source: str, fingerprint: str | None = None, scope: str | None = None) -> None:
        event = {"event": "policy_decision", "tool": tool, "action": action,
                 "outcome": outcome, "reason": reason, "source": source}
        if fingerprint:
            event["fingerprint"] = fingerprint
        if scope:
            event["scope"] = scope
        self.policy_events.append(event.copy())
        if self.on_policy_event:
            self.on_policy_event(event.copy())

    def _safe_path(self, value: str) -> Path:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ToolError("Path must be a non-empty string without NUL")
        candidate = Path(value)
        if candidate.is_absolute():
            raise ToolError("Absolute paths are not allowed")
        try:
            resolved = (self.root / candidate).resolve()
        except (OSError, RuntimeError):
            raise ToolError("Cannot resolve path safely") from None
        if not resolved.is_relative_to(self.root):
            raise ToolError("Path escapes the workspace")
        if protected(candidate) or protected(resolved.relative_to(self.root)):
            raise ToolError("Secrets and internal directories are protected")
        return resolved

    def list_files(self, path: str = ".", recursive: bool = False) -> str:
        directory = self._safe_path(path)
        if not directory.is_dir():
            raise ToolError(f"Not a directory: {path}")
        entries = []
        for current, directories, files in os.walk(directory, followlinks=False):
            directories[:] = sorted(d for d in directories if not protected(Path(d)))
            for name in directories + sorted(files):
                item = Path(current) / name
                if protected(item.relative_to(self.root)):
                    continue
                entries.append(item.relative_to(self.root).as_posix() + ("/" if item.is_dir() else ""))
                if len(entries) == 500:
                    return self._truncate("\n".join(entries) + "\n... list truncated ...")
            if not recursive:
                break
        return self._truncate("\n".join(entries) or "(empty)")

    def read_file(self, path: str) -> str:
        file = self._safe_path(path)
        if not file.is_file():
            raise ToolError(f"Not a file: {path}")
        if file.stat().st_size > self.max_file_chars * 4:
            raise ToolError("File exceeds size limit")
        with file.open(encoding="utf-8") as stream:
            result = stream.read(self.max_file_chars + 1)
        if len(result) > self.max_file_chars:
            raise ToolError("File exceeds size limit; no partial content returned")
        return result

    def write_file(self, path: str, content: str) -> str:
        file = self._safe_path(path)
        if is_test(Path(path)) or is_test(file.relative_to(self.root)):
            raise ToolError("Test files are read-only")
        if not isinstance(content, str) or len(content) > self.max_file_chars:
            raise ToolError("Content must be a string within the file size limit")
        try:
            write_target(capture_target(self.root, file, path), content)
        except TargetChangedError:
            raise ToolError("TARGET_CHANGED: cannot write the same target safely") from None
        return f"wrote {len(content)} characters to {file.relative_to(self.root)}"

    def _argv(self, command: str) -> list[str]:
        if not isinstance(command, str) or not command.strip():
            raise ToolError("Command must be a non-empty string")
        if any(c in command for c in "\n\r\x00$`;|&<>"):
            raise ToolError("Shell expansion, redirects and command chaining are not allowed")
        try:
            args = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"Invalid command: {exc}") from None
        if not args or args[0] not in {"python", "python3", "pwd", "ls", "cat", "sleep"}:
            raise ToolError("Program is not allowed (use python -m unittest, pwd, ls, cat or sleep)")
        program = args[0]
        if program in {"python", "python3"}:
            if args[1:3] != ["-m", "unittest"]:
                raise ToolError("Only python -m unittest is allowed")
            executable = sys.executable
        else:
            executable = {"pwd": "/bin/pwd", "ls": "/bin/ls", "cat": "/bin/cat", "sleep": "/bin/sleep"}[program]
        for arg in args[1:]:
            if not arg.startswith("-"):
                self._safe_path(arg)
        return [executable, *args[1:]]

    def _sandbox(self, argv: list[str]) -> list[str]:
        if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
            raise ToolError("Bash requires macOS sandbox-exec; there is no unsafe fallback")
        quoted = lambda value: json.dumps(str(value), ensure_ascii=False)
        read_roots = ["/System", "/usr", "/bin", "/sbin", "/Library", "/opt", "/dev",
                      "/private/preboot",
                      "/private/etc", "/private/var/db", str(Path(sys.executable).parent.parent)]
        profile = ["(version 1)", "(allow default)", "(deny network*)",
                   "(deny file-read* file-write*)", "(allow file-read-metadata)",
                   '(allow file-read* (literal "/"))']
        profile += [f"(allow file-read* (require-all (subpath {quoted(p)}) (require-not (subpath {quoted(self.root)}))))"
                    for p in read_roots]
        try:
            read_condition = self.policy.sandbox_read_condition(self.root)
        except PolicyError as exc:
            raise ToolError(f"POLICY_DENIED: {exc}") from None
        if read_condition:
            profile.append(f"(allow file-read* {read_condition})")
        # Bash never writes workspace files, even when write_file has an allow rule.
        # Every write must pass through the explicit, approved file tool.
        profile += ['(allow file-write* (literal "/dev/null"))']
        root_pattern = re_escape(str(self.root))
        test_pattern = root_pattern + r"/.*(test_[^/]*\.py|[^/]*_test\.py)$"
        test_dir_pattern = root_pattern + r"/(.*/)?tests(/.*)?$"
        secret_pattern = root_pattern + r"/(.*/)?(\.env[^/]*|\.git|\.venv|\.idea|\.agent)(/.*)?$"
        profile += [f'(deny file-write* (regex {quoted(test_pattern)}))',
                    f'(deny file-write* (subpath {quoted(self.root / "tests")}))',
                    f'(deny file-write* (regex {quoted(test_dir_pattern)}))',
                    f'(deny file-read* file-write* (regex {quoted(secret_pattern)}))']
        return ["/usr/bin/sandbox-exec", "-p", "\n".join(profile), *argv]

    def bash(self, command: str) -> str:
        argv = self._sandbox(self._argv(command))
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.root), "TMPDIR": str(self.root),
               "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
        try:
            process = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, env=env, start_new_session=True)
        except OSError as exc:
            raise ToolError(f"Could not start sandbox: {type(exc).__name__}") from None
        chunks: list[bytes] = []
        count = 0
        deadline = time.monotonic() + self.command_timeout
        selector = selectors.DefaultSelector()
        assert process.stdout is not None
        selector.register(process.stdout, selectors.EVENT_READ)
        reason = ""
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "Command timed out"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fileobj.fileno(), 4096)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    chunks.append(data[:max(0, self.max_output_chars - count)])
                    count += len(data)
                    if count >= self.max_output_chars:
                        reason = "output truncated: command stopped at output limit"
                        break
                if reason:
                    break
            if not reason:
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    reason = "Command timed out"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            selector.close()
            process.stdout.close()
        output = b"".join(chunks).decode("utf-8", errors="replace")
        if "sandbox-exec:" in output and process.returncode != 0:
            raise ToolError("System sandbox is unavailable in this environment; Bash was not run")
        if reason == "Command timed out":
            raise ToolError(f"Command timed out after {self.command_timeout:g} seconds")
        return f"exit_code={process.returncode}\n{output}\n{reason}".rstrip()

    def _truncate(self, value: str) -> str:
        return value if len(value) <= self.max_output_chars else value[:self.max_output_chars] + "\n... output truncated ..."

def re_escape(value: str) -> str:
    import re
    return re.escape(value)
