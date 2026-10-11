"""Validated workspace tools and bounded, OS-sandboxed command execution."""
from __future__ import annotations
import fnmatch
import json
import os
import selectors
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

class ToolError(RuntimeError):
    pass

def protected(relative: Path) -> bool:
    return any(part in {".git", ".venv", ".idea", "__pycache__"} or part.startswith(".env")
               for part in relative.parts)

def is_test(relative: Path) -> bool:
    return "tests" in relative.parts or fnmatch.fnmatch(relative.name, "test_*.py") or relative.name.endswith("_test.py")

class ToolRuntime:
    def __init__(self, root: str | Path, *, command_timeout: float = 10,
                 max_output_chars: int = 20_000, max_file_chars: int = 100_000) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace does not exist: {self.root}")
        if min(command_timeout, max_output_chars, max_file_chars) <= 0:
            raise ValueError("Runtime limits must be positive")
        self.command_timeout = command_timeout
        self.max_output_chars = max_output_chars
        self.max_file_chars = max_file_chars

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
        name = "Bash" if name == "bash" else name
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
        handler = self.bash if name == "Bash" else getattr(self, name)
        try:
            return handler(**dict(arguments))
        except (OSError, UnicodeError) as exc:
            raise ToolError(f"{name} failed: {type(exc).__name__}") from None

    def _safe_path(self, value: str) -> Path:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ToolError("Path must be a non-empty string without NUL")
        candidate = Path(value)
        if candidate.is_absolute():
            raise ToolError("Absolute paths are not allowed")
        resolved = (self.root / candidate).resolve()
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
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
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
        quoted = lambda value: json.dumps(str(value))
        read_roots = ["/System", "/usr", "/bin", "/sbin", "/Library", "/opt", "/dev",
                      "/private/preboot",
                      "/private/etc", "/private/var/db", str(Path(sys.executable).parent.parent), str(self.root)]
        profile = ["(version 1)", "(allow default)", "(deny network*)",
                   "(deny file-read* file-write*)", "(allow file-read-metadata)",
                   '(allow file-read* (literal "/"))']
        profile += [f"(allow file-read* (subpath {quoted(p)}))" for p in read_roots]
        profile += [f"(allow file-write* (subpath {quoted(self.root)}))",
                    '(allow file-write* (literal "/dev/null"))']
        root_pattern = re_escape(str(self.root))
        test_pattern = root_pattern + r"/.*(test_[^/]*\.py|[^/]*_test\.py)$"
        test_dir_pattern = root_pattern + r"/(.*/)?tests(/.*)?$"
        secret_pattern = root_pattern + r"/(.*/)?(\.env[^/]*|\.git|\.venv|\.idea)(/.*)?$"
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
