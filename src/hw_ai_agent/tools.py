from __future__ import annotations
import fnmatch
import difflib
import hashlib
import json
import os
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from .policy import Action, ApprovalDeferred, ApprovalRequest, Decision, Policy, PolicyError
from .budget import truncate_result
from .filesystem import TargetChangedError, capture_target, write_target, remove_target, open_read_file, open_directory

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
                 trash_dir: str | Path | None = None,
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
        for path in (*self.policy.directory_allow, *self.policy.directory_deny):
            canonical = (self.root / path).resolve()
            if (not canonical.is_relative_to(self.root) or
                    canonical.relative_to(self.root).as_posix() != path):
                raise PolicyError("Policy directories must be canonical paths inside workspace, without symlink aliases")
        if mode not in {"safe", "auto"}:
            raise ValueError("Execution mode must be safe or auto")
        self.mode = mode
        self.phase = "execution"
        self.trash_dir = Path(trash_dir or self.root.parent / ".agent" / "trash").expanduser().resolve()
        if self.trash_dir.is_relative_to(self.root):
            raise ValueError("Recovery directory must be outside the workspace")
        self.session_permissions: set[str] | None = None
        self.approve = approve
        self.on_policy_event = on_policy_event
        self.policy_events: list[dict[str, Any]] = []
        self.result_limit_bytes: int | None = None

    @contextmanager
    def output_budget(self, quota: int | None):
        if quota is not None and (type(quota) is not int or quota < 1):
            raise ValueError("Output quota must be positive")
        previous = self.result_limit_bytes
        self.result_limit_bytes = quota
        try:
            yield
        finally:
            self.result_limit_bytes = previous

    def _bounded_output(self, content: str) -> str:
        return content if self.result_limit_bytes is None else truncate_result(content, self.result_limit_bytes)

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        string = {"type": "string"}
        definitions = [
            ("list_files", "List workspace entries (up to 500).", {"path": string, "recursive": {"type": "boolean"}}, []),
            ("read_file", "Read a workspace UTF-8 file. Tests are read-only.", {"path": string}, ["path"]),
            ("write_file", "Replace a workspace UTF-8 file; tests and secrets are protected.",
             {"path": string, "content": string}, ["path", "content"]),
            ("delete_file", "Remove an unnecessary regular file from the workspace, preserving a recovery copy. Tests and secrets are protected.",
             {"path": string}, ["path"]),
            ("Bash", "Run python -m unittest, pwd, ls, cat or sleep inside the sandbox. No pipes/redirects.",
             {"command": string, "cwd": string}, ["command"]),
        ]
        return [{"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": False},
        }} for name, description, props, required in definitions
            if self.phase != "planning" or name in {"list_files", "read_file"}]

    def _check_phase(self, name: str):
        if self.phase == "planning" and name not in {"list_files", "read_file"}:
            self._policy_event(name, "deny", "blocked", "Plan is not approved", "planning_boundary")
            raise ToolError("PLAN_MODE: only reading is allowed before explicit plan approval")

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        """Single entry point for model calls, policy and mode approvals."""
        name = "Bash" if name == "bash" else name
        self._check_phase(name)
        # Own an immutable-value snapshot: an approval callback cannot alter the call.
        if isinstance(arguments, Mapping):
            arguments = dict(arguments)
        try:
            path, command = self._validate_call(name, arguments)
        except ToolError as exc:
            self._policy_event(name, "deny", "blocked", str(exc), "runtime_boundary")
            raise
        target = None
        if name in {"write_file", "delete_file"}:
            try:
                target = capture_target(self.root, self.root / path, arguments["path"])
            except OSError:
                raise ToolError("TARGET_CHANGED: cannot capture a stable write target") from None
        decision = self.policy.decide(name, path=path, command=command)
        if decision.action == Action.DENY:
            self._policy_event(name, "deny", "blocked", decision.reason, "policy")
            raise ToolError(f"POLICY_DENIED: {decision.reason}")
        if name == "delete_file" or (self.mode == "safe" and name in {"write_file", "Bash"}):
            decision = Decision(Action.ASK, f"{self.mode} mode requires approval for {name}")
        elif self.mode == "auto" and name in {"write_file", "Bash"}:
            decision = Decision(Action.ALLOW, f"auto mode permits {name} after runtime and policy checks")
        if decision.action == Action.ASK:
            fingerprint = hashlib.sha256(json.dumps(
                {"tool": name, "arguments": arguments, "target": path,
                 "file_signature": target.file_signature if target else None,
                 "file_digest": target.file_digest if target else None},
                sort_keys=True, ensure_ascii=False
            ).encode()).hexdigest()
            preview = ""
            if name == "write_file":
                file = self._safe_path(arguments["path"])
                try:
                    with self.output_budget(None):
                        original = self.read_file(arguments["path"]) if file.is_file() else ""
                    preview = "".join(difflib.unified_diff(original.splitlines(True),
                                arguments["content"].splitlines(True), fromfile="current", tofile="proposed"))
                except (ToolError, OSError, UnicodeError):
                    preview = "Existing content could not be previewed. Proposed content:\n" + arguments["content"]
                preview = preview[:4000] + ("\n... preview truncated ..." if len(preview) > 4000 else "")
            elif name == "delete_file":
                preview = f"Remove {path} from workspace; preserve a recovery copy."
                if target.file_signature is not None:
                    preview += f"\nSize: {target.file_signature[2]} bytes."
            elif name == "Bash":
                preview = f"Working directory: {arguments.get('cwd', '.')}\nCommand: {command}"
            request = ApprovalRequest(name, path or command or ".", decision.reason, preview, fingerprint)
            approved = False
            source = "policy"
            self._policy_event(name, "ask", "waiting", decision.reason, "policy", fingerprint, request.scope)
            try:
                approved = self.approve is not None and self.approve(request) is True
            except ApprovalDeferred:
                raise
            except Exception:
                # Broken approval callbacks never authorize execution.
                approved = False
            self._policy_event(name, "ask", "approved" if approved else "refused",
                               decision.reason, source, fingerprint, request.scope)
            if not approved:
                raise ToolError("APPROVAL_REQUIRED: operator approval was not granted")
        else:
            self._policy_event(name, "allow", "permitted", decision.reason, "policy")
        handler = self.bash if name == "Bash" else getattr(self, name)
        try:
            if target is not None and name == "delete_file":
                recovery_id = remove_target(target, self.trash_dir)
                return (f"removed {path}; recovery_id={recovery_id}" if recovery_id else f"already absent: {path}")
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
            directory = self._safe_path(arguments.get("cwd", "."))
            if not directory.is_dir():
                raise ToolError("Bash cwd must be an existing workspace directory")
            self._argv(arguments["command"], directory)
            return None, shlex.join(shlex.split(arguments["command"]))
        raw_path = arguments.get("path", ".")
        file = self._safe_path(raw_path)
        if name in {"write_file", "delete_file"}:
            if is_test(Path(raw_path)) or is_test(file.relative_to(self.root)):
                raise ToolError("Test files are read-only")
            if name == "write_file" and len(arguments["content"]) > self.max_file_chars:
                raise ToolError("Content exceeds file size limit")
            if file.is_dir():
                raise ToolError("Write target is a directory")
            if file.is_symlink() or (file.is_file() and file.stat().st_nlink > 1):
                raise ToolError("Writing hard-linked files is not allowed")
            if name == "delete_file" and (self.root / raw_path).is_symlink():
                raise ToolError("Deleting symlinks is not allowed")
        return file.relative_to(self.root).as_posix(), None

    def _policy_event(self, tool: str, action: str, outcome: str, reason: str,
                      source: str, fingerprint: str | None = None, scope: str | None = None) -> None:
        event = {"event": "policy_decision", "tool": tool, "action": action,
                 "outcome": outcome, "reason": reason, "source": source, "approval_contract": 2}
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
        if ".." in candidate.parts:
            raise ToolError("Path escapes the workspace: '..' is not allowed")
        try:
            resolved = (self.root / candidate).resolve()
        except (OSError, RuntimeError):
            raise ToolError("Cannot resolve path safely") from None
        if not resolved.is_relative_to(self.root):
            raise ToolError("Path escapes the workspace")
        if protected(candidate) or protected(resolved.relative_to(self.root)):
            raise ToolError("Secrets and internal directories are protected")
        if not self.policy.directory_allowed(resolved.relative_to(self.root).as_posix()):
            raise ToolError("POLICY_DENIED: directory is not allowed")
        return resolved

    def list_files(self, path: str = ".", recursive: bool = False) -> str:
        directory = self._safe_path(path)
        if not directory.is_dir():
            raise ToolError(f"Not a directory: {path}")
        entries = []
        try:
            with open_directory(self.root, directory) as directory_fd:
                for current, directories, files, current_fd in os.fwalk(".", follow_symlinks=False, dir_fd=directory_fd):
                    relative = directory.relative_to(self.root) / current
                    directories[:] = sorted(d for d in directories if not protected(Path(d))
                        and self.policy.directory_allowed((relative / d).as_posix()))
                    for name in directories + sorted(files):
                        item = relative / name
                        if protected(item) or not self.policy.directory_allowed(item.as_posix()):
                            continue
                        try:
                            info = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        entries.append(item.as_posix() + ("/" if stat.S_ISDIR(info.st_mode) else ""))
                        if self.result_limit_bytes is not None and len("\n".join(entries).encode("utf-8")) > self.result_limit_bytes:
                            return self._bounded_output("\n".join(entries))
                        if len(entries) == 500:
                            return self._truncate("\n".join(entries) + "\n... list truncated ...")
                    if not recursive:
                        break
        except TargetChangedError:
            raise ToolError("TARGET_CHANGED: cannot list the checked workspace directory safely") from None
        return self._truncate("\n".join(entries) or "(empty)")

    def read_file(self, path: str) -> str:
        file = self._safe_path(path)
        if not file.is_file():
            raise ToolError(f"Not a file: {path}")
        try:
            with open_read_file(self.root, file) as descriptor:
                if os.fstat(descriptor).st_size > self.max_file_chars * 4:
                    raise ToolError("File exceeds size limit")
                with os.fdopen(descriptor, encoding="utf-8", closefd=False) as stream:
                    limit = self.max_file_chars + 1
                    if self.result_limit_bytes is not None:
                        limit = min(limit, self.result_limit_bytes + 1)
                    result = stream.read(limit)
        except TargetChangedError:
            raise ToolError("TARGET_CHANGED: cannot read the checked workspace file safely") from None
        if len(result) > self.max_file_chars:
            raise ToolError("File exceeds size limit; no partial content returned")
        return self._bounded_output(result)

    def write_file(self, path: str, content: str) -> str:
        self._check_phase("write_file")
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

    def delete_file(self, path: str) -> str:
        return self.execute("delete_file", {"path": path})

    def _argv(self, command: str, cwd: Path | None = None) -> list[str]:
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
                relative = (cwd or self.root).relative_to(self.root) / arg
                self._safe_path(relative.as_posix())
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

    def bash(self, command: str, cwd: str = ".") -> str:
        self._check_phase("Bash")
        directory = self._safe_path(cwd)
        if not directory.is_dir():
            raise ToolError("Bash cwd must be an existing workspace directory")
        command_argv = self._argv(command, directory)
        argv = self._sandbox(["/bin/bash", "--noprofile", "--norc", "-c", "exec " + shlex.join(command_argv)])
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.root), "TMPDIR": str(self.root),
               "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
        try:
            process = subprocess.Popen(argv, cwd=directory, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, env=env, start_new_session=True)
        except OSError as exc:
            raise ToolError(f"Could not start sandbox: {type(exc).__name__}") from None
        chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
        count = 0
        output_limit = min(self.max_output_chars, self.result_limit_bytes) if self.result_limit_bytes is not None else self.max_output_chars
        deadline = time.monotonic() + self.command_timeout
        selector = selectors.DefaultSelector()
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
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
                    chunks[key.data].append(data[:max(0, output_limit - count)])
                    count += len(data)
                    if count >= output_limit:
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
                self._terminate_process_group(process)
            finally:
                selector.close()
                process.stdout.close()
                process.stderr.close()
        stdout = b"".join(chunks["stdout"]).decode("utf-8", errors="replace")
        stderr = b"".join(chunks["stderr"]).decode("utf-8", errors="replace")
        output = stdout + stderr
        if "sandbox-exec:" in output and process.returncode != 0:
            raise ToolError("System sandbox is unavailable in this environment; Bash was not run")
        if reason == "Command timed out":
            raise ToolError(f"Command timed out after {self.command_timeout:g} seconds")
        return self._bounded_output(f"exit_code={process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n{reason}".rstrip())

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS may report EPERM for a group whose leader already exited.
            # Only a confirmed exit makes this harmless; a live command must stop.
            if process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                raise ToolError("Cannot terminate command process group safely") from None
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            raise ToolError("Command process did not stop after termination") from None

    def _truncate(self, value: str) -> str:
        result = value if len(value) <= self.max_output_chars else value[:self.max_output_chars] + "\n... output truncated ..."
        return self._bounded_output(result)

def re_escape(value: str) -> str:
    import re
    return re.escape(value)
