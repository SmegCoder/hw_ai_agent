"""Read-only stdio server for one study-tracker snapshot."""

import json
import os
import sys


VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
LIMIT = 48_000
TOOL = {
    "name": "list_tasks",
    "description": "Read all study tracker tasks from the configured snapshot. Returns IDs, subjects, statuses and due dates when present.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    "annotations": {"readOnlyHint": True},
}


def reply(identifier, *, result=None, error=None):
    field = {"result": result} if error is None else {"error": {"code": error[0], "message": error[1]}}
    print(json.dumps({"jsonrpc": "2.0", "id": identifier, **field}, ensure_ascii=False, allow_nan=False), flush=True)


def read_tasks(snapshot_fd: int) -> dict:
    try:
        with os.fdopen(os.dup(snapshot_fd), "rb") as stream:
            stream.seek(0)
            data = stream.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise ValueError
        value = json.loads(data.decode("utf-8"))
        tasks = value.get("tasks") if isinstance(value, dict) else value
        if not isinstance(tasks, list):
            raise ValueError
        result = {"tasks": tasks}
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        return {"structuredContent": result, "content": [{"type": "text", "text": encoded}]}
    except (OSError, UnicodeError, ValueError, RecursionError):
        return {"isError": True, "content": [{"type": "text", "text": "Tracker snapshot is unavailable or invalid"}]}


def serve(snapshot_fd: int) -> None:
    initialized = False
    for line in iter(lambda: sys.stdin.buffer.readline(128 * 1024 + 1), b""):
        if len(line) > 128 * 1024:
            return
        try:
            message = json.loads(line)
        except (ValueError, UnicodeError, RecursionError):
            reply(None, error=(-32700, "Invalid JSON"))
            continue
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            reply(None, error=(-32600, "Invalid request"))
            continue
        method = message.get("method")
        if method == "notifications/initialized":
            initialized = True
        if "id" not in message:
            continue
        identifier, params = message["id"], message.get("params", {})
        if not isinstance(params, dict):
            reply(identifier, error=(-32602, "Invalid parameters"))
        elif method == "initialize":
            version = params.get("protocolVersion")
            if not isinstance(version, str) or version not in VERSIONS:
                version = "2025-11-25"
            reply(identifier, result={"protocolVersion": version, "capabilities": {"tools": {}},
                                      "serverInfo": {"name": "study-tracker", "version": "1.0.0"}})
        elif method == "ping":
            reply(identifier, result={})
        elif not initialized:
            reply(identifier, error=(-32002, "Server is not initialized"))
        elif method == "tools/list":
            reply(identifier, result={"tools": [TOOL]})
        elif method == "tools/call":
            if params.get("name") == TOOL["name"] and params.get("arguments", {}) == {}:
                reply(identifier, result=read_tasks(snapshot_fd))
            else:
                reply(identifier, error=(-32602, "Only list_tasks with an empty object is allowed"))
        else:
            reply(identifier, error=(-32601, "Unknown method"))


if __name__ == "__main__":
    serve(int(sys.argv[1]))
