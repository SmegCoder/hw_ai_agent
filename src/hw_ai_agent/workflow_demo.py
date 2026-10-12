from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from .agent import CodingAgent
from .reporting import public_report
from .session import SessionStore
from .tools import ToolRuntime


FIXED_TAX = "def add_tax(amount):\n    return amount * 1.2\n"
FIXED_RECEIPT = "def receipt_total(prices):\n    return sum(prices)\n"
WORKFLOW_TESTS = '''import unittest
from tax import add_tax
from receipt import receipt_total

class CheckoutTests(unittest.TestCase):
    def test_tax(self):
        self.assertAlmostEqual(add_tax(100), 120)

    def test_receipt(self):
        self.assertEqual(receipt_total([10, 20, 30]), 60)
'''


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


class WorkflowClient:
    usage: list = []

    def complete(self, messages, tools):
        if not any(m.get("role") == "assistant" for m in messages):
            return {"tool_calls": [
                call("tax", "write_file", {"path": "tax.py", "content": FIXED_TAX}),
                call("receipt", "write_file", {"path": "receipt.py", "content": FIXED_RECEIPT}),
                call("list", "Bash", {"command": "ls"}),
                call("tests", "Bash", {"command": "python -m unittest discover -v"}),
            ]}
        return {"content": "Сценарий завершён. Результат проверяется независимым запуском тестов."}


class CrashAfterWriteRuntime(ToolRuntime):
    def execute(self, name, arguments):
        result = super().execute(name, arguments)
        if name == "write_file":
            print("DEMO CRASH: файл записан, но результат ещё не сохранён. Код завершения: 86.", flush=True)
            os._exit(86)
        return result


def run_workflow_demo(workspace: Path, db_path: Path, *, mode="safe", approve=None,
                      approvals="ask", on_event=None, crash_after_write=False):
    if db_path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("Session database must be outside workspace")
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "tax.py").write_text("def add_tax(amount):\n    return amount + 20\n", encoding="utf-8")
    (workspace / "receipt.py").write_text("def receipt_total(prices):\n    return len(prices)\n", encoding="utf-8")
    # Use values that distinguish a percentage from a fixed addition.
    tests = WORKFLOW_TESTS.replace("add_tax(100), 120", "add_tax(50), 60")
    (workspace / "test_checkout.py").write_text(tests, encoding="utf-8")
    runtime_class = CrashAfterWriteRuntime if crash_after_write else ToolRuntime
    from .policy import Policy
    policy = Policy.from_dict({"defaults": {"write_file": "ask", "Bash": "ask"}})
    runtime = runtime_class(workspace, policy=policy, mode=mode, approve=approve)
    baseline = runtime.bash("python -m unittest discover -v")
    if not baseline.startswith("exit_code=1\n"):
        raise ValueError("Expected failing workflow baseline")
    metadata = {"kind": "workflow", "approvals": approvals,
                "demo": {"tests": tests, "baseline": baseline}}
    with SessionStore(db_path) as store:
        agent = CodingAgent(WorkflowClient(), runtime, store=store, on_event=on_event,
                            session_metadata=metadata)
        agent.run("Fix tax.py and receipt.py, list files and run the tests.")
        return verify_workflow(agent, runtime, metadata)


def verify_workflow(agent, runtime, metadata):
    output = runtime.bash("python -m unittest discover -v")
    unchanged = (runtime.root / "test_checkout.py").read_bytes() == metadata["demo"]["tests"].encode("utf-8")
    policy_events = [e for e in agent.events if e.get("event") == "policy_decision"]
    prompts = Counter(e["tool"] for e in policy_events if e.get("action") == "ask" and e.get("source") == "policy")
    attempts = Counter(json.loads(e["arguments"])["path"] for e in agent.events
                       if e.get("event") == "tool_call" and e.get("name") == "write_file")
    report = {"workspace": str(runtime.root), "session_id": agent.session_id,
              "mode": runtime.mode, "baseline": metadata["demo"]["baseline"],
              "verification": output, "tests_unchanged": unchanged,
              "success": output.startswith("exit_code=0\n") and unchanged,
              "permission_requests": dict(prompts), "write_attempts": dict(attempts),
              "events": agent.events}
    (runtime.root / "workflow-report.json").write_text(
        json.dumps(public_report(report, runtime.root), ensure_ascii=False, indent=2), encoding="utf-8")
    return report
