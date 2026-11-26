from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from .filesystem import open_read_file


PROTOCOL = "2025-11-25"
VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", PROTOCOL}
MAX_FRAME = 128 * 1024
NAME = re.compile(r"[A-Za-z0-9_-]+")
SERVER_NAME = re.compile(r"[a-z][a-z0-9-]{0,23}")
TRACKER = Path(__file__).resolve().with_name("tracker_mcp.py")


class MCPError(ValueError):
    pass


def mcp_fields(name: str) -> dict[str, str]:
    if not isinstance(name, str):
        return {}
    prefix, _, tail = name.partition("__")
    server, _, tool = tail.partition("__")
    if prefix == "mcp" and SERVER_NAME.fullmatch(server) and NAME.fullmatch(tool) and len(name) <= 64:
        return {"server": server, "mcp_tool": tool}
    return {}


def mcp_event(event: dict) -> dict:
    if event["event"] in {"tool_call", "tool_result", "policy_decision"}:
        return {**event, **mcp_fields(event.get("name", event.get("tool", "")))}
    return event


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True).encode()).hexdigest()


def read_json(path: Path, limit: int):
    with open_read_file(Path("/"), path) as descriptor:
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            data = stream.read(limit + 1)
    if len(data) > limit:
        raise MCPError("MCP_LIMIT: input file exceeds the byte limit")
    return json.loads(data.decode("utf-8"))


class Stdio:
    def __init__(self, process: subprocess.Popen, timeout: float):
        self.process = process
        self.deadline = time.monotonic() + timeout
        self.buffer = bytearray()
        self.request_id = 0
        self.reader = process.stdout.fileno()
        self.writer = process.stdin.fileno()
        os.set_blocking(self.reader, False)
        os.set_blocking(self.writer, False)

    def wait(self, descriptor: int, events: int) -> None:
        remaining = self.deadline - time.monotonic()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, events)
            if remaining <= 0 or not selector.select(remaining):
                raise MCPError("MCP_TIMEOUT: server did not respond within the configured time")

    def send(self, message: dict) -> None:
        data = memoryview(json_bytes({"jsonrpc": "2.0", **message}) + b"\n")
        if len(data) > MAX_FRAME:
            raise MCPError("MCP_LIMIT: request exceeds the byte limit")
        while data:
            self.wait(self.writer, selectors.EVENT_WRITE)
            data = data[os.write(self.writer, data):]

    def receive(self) -> dict:
        while (end := self.buffer.find(b"\n")) == -1:
            if len(self.buffer) >= MAX_FRAME:
                raise MCPError("MCP_LIMIT: response exceeds the byte limit")
            self.wait(self.reader, selectors.EVENT_READ)
            chunk = os.read(self.reader, min(4096, MAX_FRAME - len(self.buffer)))
            if not chunk:
                raise MCPError("MCP_DISCONNECTED: server closed the connection")
            self.buffer.extend(chunk)
        line = bytes(self.buffer[:end])
        del self.buffer[:end + 1]
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError):
            raise MCPError("MCP_PROTOCOL: invalid JSON-RPC message") from None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise MCPError("MCP_PROTOCOL: expected a JSON-RPC 2.0 object")
        return message

    def request(self, method: str, params: dict) -> dict:
        self.request_id += 1
        self.send({"id": self.request_id, "method": method, "params": params})
        while time.monotonic() < self.deadline:
            message = self.receive()
            if isinstance(message.get("method"), str):
                if "id" in message:
                    reply = ({"result": {}} if message["method"] == "ping" else
                             {"error": {"code": -32601, "message": "Unsupported client method"}})
                    self.send({"id": message["id"], **reply})
                continue
            if type(message.get("id")) is not int or message["id"] != self.request_id:
                raise MCPError("MCP_PROTOCOL: response ID does not match")
            if "error" in message:
                error = message["error"]
                code = error.get("code") if isinstance(error, dict) else None
                raise MCPError(f"MCP_RPC_ERROR: {method} returned code {code if type(code) is int else 'unknown'}")
            result = message.get("result")
            if isinstance(result, dict):
                return result
            raise MCPError("MCP_PROTOCOL: response has no result object")
        raise MCPError("MCP_TIMEOUT: server did not respond within the configured time")

    def initialize(self) -> None:
        response = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                               "clientInfo": {"name": "agi", "version": "0.1.0"}})
        if (not isinstance(response.get("protocolVersion"), str) or response["protocolVersion"] not in VERSIONS
                or not isinstance(response.get("capabilities"), dict)
                or not isinstance(response["capabilities"].get("tools"), dict)):
            raise MCPError("MCP_PROTOCOL: unsupported version or missing tools capability")
        self.send({"method": "notifications/initialized"})

    def list_tools(self):
        params = {}
        cursors = set()
        while True:
            page = self.request("tools/list", params)
            if not isinstance(page.get("tools"), list):
                raise MCPError("MCP_PROTOCOL: tools/list has no tools array")
            yield from page["tools"]
            cursor = page.get("nextCursor")
            if not cursor:
                return
            if not isinstance(cursor, str) or cursor in cursors or len(cursors) >= 31:
                raise MCPError("MCP_PROTOCOL: invalid tools/list pagination")
            cursors.add(cursor)
            params = {"cursor": cursor}


@dataclass(frozen=True)
class Server:
    name: str
    allow_tools: tuple[str, ...]
    timeout: float
    snapshot: Path | None
    command: tuple[str, ...]
    read_paths: tuple[Path, ...]

    @classmethod
    def parse(cls, name, value, base, workspace):
        if not SERVER_NAME.fullmatch(name) or not isinstance(value, dict):
            raise MCPError("Invalid MCP server name or settings")
        tools = value.get("allow_tools")
        if (not isinstance(tools, list) or not 1 <= len(tools) <= 32
                or any(not isinstance(tool, str) or not mcp_fields(f"mcp__{name}__{tool}") for tool in tools)
                or len(set(tools)) != len(tools)):
            raise MCPError(f"{name}: allow_tools must contain unique MCP tool names")
        timeout = value.get("timeout", 5)
        if type(timeout) not in (int, float) or not 0.05 <= timeout <= 30:
            raise MCPError(f"{name}: timeout must be between 0.05 and 30 seconds")
        fields = {"allow_tools", "timeout"}
        if "snapshot" in value and not set(value) - (fields | {"snapshot"}):
            return cls(name, tuple(tools), timeout, configured_path(value["snapshot"], base, workspace), (), ())
        if "command" not in value or set(value) - (fields | {"command", "read_paths"}):
            raise MCPError(f"{name}: configure either snapshot or command")
        command = value["command"]
        paths = value.get("read_paths", [])
        if (not isinstance(command, list) or not command or len(command) > 32
                or any(not isinstance(part, str) or not part or "\x00" in part for part in command)
                or not isinstance(paths, list) or len(paths) > 32):
            raise MCPError(f"{name}: invalid command or read_paths")
        executable = configured_path(command[0], base, workspace)
        return cls(name, tuple(tools), timeout, None, (str(executable), *command[1:]),
                   tuple(configured_path(path, base, workspace) for path in paths))

    def argv(self, workspace: Path, snapshot_fd: int = -1) -> tuple[str, ...]:
        if self.snapshot is not None:
            return (sys.executable, "-I", "-B", str(TRACKER), str(snapshot_fd))
        sandbox = "/usr/bin/sandbox-exec"
        if sys.platform != "darwin" or not Path(sandbox).is_file():
            raise MCPError("MCP_SANDBOX: external commands require macOS sandbox-exec")

        def quoted(path):
            return json.dumps(str(path), ensure_ascii=False)

        profile = ["(version 1)", "(allow default)", "(deny network*)", "(deny file-read* file-write*)",
                   "(allow file-read-metadata)", '(allow file-write* (literal "/dev/null"))',
                   '(allow file-read* (literal "/"))', f"(allow file-read* (literal {quoted(self.command[0])}))"]
        for directory in ("/System", "/usr/lib", "/usr/share", "/usr/bin", "/bin", "/dev",
                          "/Library/Frameworks/Python.framework", "/private/var/db", *self.read_paths):
            profile.append(f"(allow file-read* (subpath {quoted(directory)}))")
        profile.append(f"(deny file-read-data file-write* (subpath {quoted(workspace)}))")
        return (sandbox, "-p", "\n".join(profile), *self.command)


def configured_path(value, base: Path, workspace: Path) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MCPError("MCP paths must be non-empty strings")
    path = (base / Path(value).expanduser()).resolve()
    if path.is_relative_to(workspace) or workspace.is_relative_to(path):
        raise MCPError("MCP paths must be outside the agent workspace")
    return path


@contextmanager
def connect(server: Server, workspace: Path):
    with tempfile.TemporaryDirectory(prefix="agi-mcp-") as directory, ExitStack() as files:
        descriptor = -1
        if server.snapshot is not None:
            try:
                descriptor = files.enter_context(open_read_file(Path("/"), server.snapshot))
            except OSError:
                pass
        try:
            process = subprocess.Popen(server.argv(workspace, descriptor), cwd=directory, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       env={"PATH": os.defpath, "HOME": directory, "TMPDIR": directory,
                                            "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
                                       start_new_session=True, bufsize=0, pass_fds=(descriptor,) if descriptor >= 0 else ())
        except OSError:
            raise MCPError("MCP_START: cannot launch the configured server") from None
        try:
            connection = Stdio(process, server.timeout)
            connection.initialize()
            yield connection
        except (OSError, UnicodeError, RecursionError):
            raise MCPError("MCP_PROTOCOL: could not complete the stdio exchange") from None
        finally:
            process.stdin.close()
            process.stdout.close()
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if process.poll() is None:
                        process.kill()
                process.wait(timeout=1)


@dataclass(frozen=True)
class MCPTool:
    server: Server
    name: str
    description: str
    schema: dict

    @property
    def alias(self) -> str:
        return f"mcp__{self.server.name}__{self.name}"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": self.alias, "description": self.description,
                                                "parameters": self.schema}}

    def validate(self, arguments: dict) -> None:
        required = self.schema.get("required", [])
        if set(required) - arguments.keys():
            raise MCPError("MCP_ARGUMENTS: missing required arguments")
        if self.schema.get("additionalProperties") is False and arguments.keys() - self.schema.get("properties", {}).keys():
            raise MCPError("MCP_ARGUMENTS: unexpected arguments")
        try:
            size = len(json_bytes(arguments))
        except (ValueError, RecursionError):
            raise MCPError("MCP_ARGUMENTS: expected finite JSON values") from None
        if size > MAX_FRAME // 2:
            raise MCPError("MCP_LIMIT: arguments exceed the byte limit")


class MCPRegistry:
    def __init__(self, path: Path, workspace: Path, *, expected_digest: str | None = None):
        self.path = path.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()
        configured_path(str(self.path), self.path.parent, self.workspace)
        try:
            value = read_json(self.path, 64_000)
        except (ValueError, OSError, RecursionError):
            raise MCPError("Cannot read valid MCP configuration") from None
        entries = value.get("servers") if isinstance(value, dict) and set(value) == {"servers"} else None
        if not isinstance(entries, dict) or not 1 <= len(entries) <= 8:
            raise MCPError("MCP configuration must contain between one and eight servers")
        self.config_digest = digest(value)
        if expected_digest is not None and self.config_digest != expected_digest:
            raise MCPError("MCP configuration changed since the saved session")
        self.tools: dict[str, MCPTool] = {}
        self.events: list[dict] = []
        for name, options in entries.items():
            server = Server.parse(name, options, self.path.parent, self.workspace)
            try:
                found = self.discover(server)
            except (MCPError, ValueError) as exc:
                self.events.append({"event": "mcp_error", "server": name, "phase": "discovery",
                                    "error": str(exc) if isinstance(exc, MCPError) else "MCP_PROTOCOL: invalid schema"})
                continue
            self.tools.update((tool.alias, tool) for tool in found.values())
            self.events.append({"event": "mcp_discovered", "server": name, "names": list(found)})

    def discover(self, server: Server) -> dict[str, MCPTool]:
        found = {}
        with connect(server, self.workspace) as connection:
            for item in connection.list_tools():
                if not isinstance(item, dict) or item.get("name") not in server.allow_tools:
                    continue
                name, schema = item["name"], item.get("inputSchema")
                if name in found or not isinstance(schema, dict) or schema.get("type") != "object":
                    raise MCPError("MCP_PROTOCOL: invalid or duplicate tool schema")
                properties, required = schema.get("properties", {}), schema.get("required", [])
                if (not isinstance(properties, dict) or not isinstance(required, list)
                        or any(not isinstance(key, str) for key in required)
                        or len(json_bytes(schema)) > 24_000):
                    raise MCPError("MCP_PROTOCOL: unsupported tool schema")
                description = item.get("description") or name
                if not isinstance(description, str):
                    raise MCPError("MCP_PROTOCOL: invalid tool description")
                found[name] = MCPTool(server, name, description[:1024], schema)
        return found

    def call(self, tool: MCPTool, arguments: dict) -> str:
        with connect(tool.server, self.workspace) as connection:
            result = connection.request("tools/call", {"name": tool.name, "arguments": arguments})
        content = result.get("content", [])
        if not isinstance(content, list):
            raise MCPError("MCP_PROTOCOL: invalid tool content")
        if any(not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str)
               for item in content):
            raise MCPError("MCP_PROTOCOL: only text and structured tool results are supported")
        text = "\n".join(item["text"] for item in content)
        if result.get("isError") is True:
            raise MCPError("MCP_TOOL_ERROR: " + (text[:1000] or "server tool failed"))
        if "structuredContent" in result:
            try:
                return json.dumps(result["structuredContent"], ensure_ascii=True, allow_nan=False)
            except (ValueError, RecursionError):
                raise MCPError("MCP_PROTOCOL: invalid structured result") from None
        return text
