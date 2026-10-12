from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .agent import CodingAgent
from .demo import initialize_demo
from .policy import Policy
from .session import SessionStore
from .tools import ToolRuntime


class CountingRuntime(ToolRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writes = 0

    def execute(self, name, arguments):
        if name == "write_file":
            self.writes += 1
        return super().execute(name, arguments)


def run_session_demo(on_event=None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = root / "workspace"
        initialize_demo(workspace)
        test_bytes = (workspace / "test_order.py").read_bytes()
        policy = Policy.from_dict({"defaults": {"write_file": "deny", "Bash": "deny"}, "rules": [
            {"tool": "write_file", "path": "order.py", "action": "allow"},
            {"tool": "Bash", "command": "python -m unittest discover -v", "action": "allow"}]})
        def call(call_id, name, arguments):
            return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        code = ("def calculate_total(prices, discount_percent):\n"
                "    if not 0 <= discount_percent <= 100:\n"
                "        raise ValueError('invalid discount')\n"
                "    return sum(prices) * (1 - discount_percent / 100)\n")
        class FirstClient:
            def complete(self, messages, tools):
                return {"tool_calls": [call("write", "write_file", {"path": "order.py", "content": code}),
                                       call("tests", "Bash", {"command": "python -m unittest discover -v"})]}
        class ResumedClient:
            def complete(self, messages, tools):
                if not any(m.get("role") == "tool" and "Ran 6 tests" in m.get("content", "")
                           and m["content"].startswith("exit_code=0") for m in messages):
                    raise AssertionError("Pending tests were not recovered")
                return {"content": "Recovered; all tests pass."}
        def interrupt(event):
            if on_event:
                on_event(event)
            if event["event"] == "tool_result" and event["name"] == "write_file":
                raise KeyboardInterrupt
        db = root / ".agent/sessions.sqlite3"
        with SessionStore(db) as store:
            first_runtime = CountingRuntime(workspace, policy=policy)
            first = CodingAgent(FirstClient(), first_runtime, store=store, on_event=interrupt)
            try:
                first.run("Fix discount and run tests")
            except KeyboardInterrupt:
                pass
            session_id = first.session_id
            interrupted = store.get(session_id).status == "interrupted"
        # New store, client and runtime: no in-memory history is used for continuation.
        with SessionStore(db) as store:
            resumed_runtime = CountingRuntime(workspace, policy=policy)
            resumed = CodingAgent(ResumedClient(), resumed_runtime, store=store, on_event=on_event)
            result = resumed.resume(session_id)
            checks = {"Session interrupted after saved write": interrupted,
                      "Original write happened once": first_runtime.writes == 1,
                      "Resume did not repeat the write": resumed_runtime.writes == 0,
                      "Pending tests completed": "all tests pass" in result,
                      "Session completed": store.get(session_id).status == "completed",
                      "Test file unchanged": test_bytes == (workspace / "test_order.py").read_bytes()}
    return {"success": all(checks.values()), "checks": checks}
